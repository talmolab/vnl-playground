"""Unified reference clips loader for motion capture data.

This module provides a single `ReferenceClips` class that loads motion capture
reference data from HDF5 files. It automatically detects the H5 format and
provides a consistent API regardless of the underlying data structure.

Supported H5 Formats
--------------------
1. **Named-array format** (preferred, used by fruitfly):
   - Root state: `position`, `velocity`, `quaternion`, `angular_velocity`
   - Joint state: `joints`, `joints_velocity`
   - Body state: `body_positions`, `body_quaternions`
   - Data may be under `/all_clips/` group or at root level

2. **Legacy flat-array format** (used by rodent):
   - State vectors: `qpos`, `qvel` (flat arrays containing root + joints)
   - Body state: `xpos`, `xquat`
   - Metadata: `names_qpos`, `names_xpos`, `config`
   - Data requires reshaping from (n_total_frames, ...) to (n_clips, n_frames, ...)

Usage
-----
>>> from vnl_playground.tasks.reference_clips import ReferenceClips
>>>
>>> # Load reference clips (format auto-detected)
>>> clips = ReferenceClips(
...     data_path="path/to/reference_clips.h5",
...     n_frames_per_clip=250,
... )
>>>
>>> # Access data using consistent API
>>> clips.qpos.shape        # (n_clips, n_frames, n_dof)
>>> clips.joints.shape      # (n_clips, n_frames, n_joints)
>>> clips.root_position     # (n_clips, n_frames, 3)
>>>
>>> # Slice operations
>>> frame = clips.at(clip=0, frame=10)      # Single frame
>>> seq = clips.slice(clip=0, start_frame=0, length=50)  # Frame sequence
>>>
>>> # Train/test split
>>> train_clips, test_clips = clips.split(train_ratio=0.8, seed=42)

See Also
--------
- `vnl_playground.tasks.wrappers` : Environment wrappers
- `vnl_playground.env_loader` : High-level environment loading API
"""

import copy
import re
from ctypes import Array
from typing import Any, Mapping, Optional

import h5py
import jax
import jax.numpy as jp
import numpy as np
import yaml
import logging
import warnings


class ReferenceClips:
    """Reference clips loader for motion capture data.

    This class loads motion capture reference data from HDF5 files and provides
    a unified API for accessing the data regardless of the underlying format.
    The format is auto-detected during loading.

    The class supports two H5 formats:

    1. **Named-array format** (preferred):
       Data stored as separate semantic arrays, optionally under `/all_clips/` group::

           position          (n_clips, n_frames, 3)     - Root XYZ position
           velocity          (n_clips, n_frames, 3)     - Root velocity
           quaternion        (n_clips, n_frames, 4)     - Root orientation
           angular_velocity  (n_clips, n_frames, 3)     - Root angular velocity
           joints            (n_clips, n_frames, n_j)   - Joint angles
           joints_velocity   (n_clips, n_frames, n_j)   - Joint velocities
           body_positions    (n_clips, n_frames, n_b, 3) - Body positions
           body_quaternions  (n_clips, n_frames, n_b, 4) - Body orientations

    2. **Legacy flat-array format**:
       Data stored as flat state vectors that need reshaping::

           qpos        (n_total_frames, n_dof)   - Joint positions (root + joints)
           qvel        (n_total_frames, n_dof-1) - Joint velocities
           xpos        (n_total_frames, n_b, 3)  - Body world positions
           xquat       (n_total_frames, n_b, 4)  - Body world orientations
           names_qpos  (n_dof,)                  - DOF names
           names_xpos  (n_b,)                    - Body names
           config      YAML string               - Metadata including clip names

    Attributes
    ----------
    clip_names : np.ndarray or None
        Behavior names for each clip (e.g., "Walk", "Run"). Only available
        for legacy format files that include config metadata.

    Examples
    --------
    >>> clips = ReferenceClips("data.h5", n_frames_per_clip=250)
    >>> print(f"Loaded {clips.qpos.shape[0]} clips")
    >>> train, test = clips.split(train_ratio=0.8)
    """

    # Named-array format keys
    _NAMED_ARRAYS = [
        "position",
        "velocity",
        "quaternion",
        "angular_velocity",
        "joints",
        "joints_velocity",
        "body_positions",
        "body_quaternions",
    ]

    # Legacy flat-array format keys
    _LEGACY_ARRAYS = ["qpos", "qvel", "xpos", "xquat"]

    def __init__(
        self,
        data_path: str,
        n_frames_per_clip: int,
        keep_clips_idx: Optional[Array[int]] = None,
        joint_names: Optional[list[str]] = None,
        body_names: Optional[list[str]] = None,
    ):
        """Load reference clips from an HDF5 file.

        Parameters
        ----------
        data_path : str
            Path to the HDF5 data file containing motion capture data.
        n_frames_per_clip : int
            Number of frames in each clip. For legacy format, this is used
            to reshape flat arrays into (n_clips, n_frames, ...) shape.
        keep_clips_idx : array-like of int, optional
            Indices of clips to keep. If None, all clips are loaded.
            Useful for loading a subset of data for debugging.
        joint_names : list of str, optional
            Override joint names. If None, names are read from H5 metadata
            (legacy: `names_qpos[7:]`, named: `/metadata/joint_names`)
            or generated as `joint_0`, `joint_1`, etc.
        body_names : list of str, optional
            Override body names. If None, names are read from H5 metadata
            (legacy: `names_xpos`, named: `/metadata/body_names`)
            or generated as `body_0`, `body_1`, etc.

        Raises
        ------
        FileNotFoundError
            If data_path does not exist.
        KeyError
            If required arrays are missing from the H5 file.
        """
        self._data_arrays: dict[str, jp.ndarray] = {}
        self._joint_names_list: list[str] = []
        self._body_names_map: dict[str, int] = {}
        self._is_legacy_format = False
        self._config: Optional[dict] = None
        self.clip_names: Optional[np.ndarray] = None

        self._load_from_disk(
            data_path, n_frames_per_clip, keep_clips_idx, joint_names, body_names
        )

    def _load_from_disk(
        self,
        data_path: str,
        n_frames_per_clip: int,
        keep_clips_idx: Optional[Array[int]],
        joint_names: Optional[list[str]],
        body_names: Optional[list[str]],
    ) -> None:
        """Load data from H5 file, auto-detecting format."""
        with h5py.File(data_path, "r") as fid:
            # Detect format by checking for named arrays
            group = fid["all_clips"] if "all_clips" in fid else fid

            if "joints" in group or "position" in group:
                self._load_named_format(
                    fid, group, keep_clips_idx, joint_names, body_names
                )
            else:
                self._load_legacy_format(
                    fid, n_frames_per_clip, keep_clips_idx, joint_names, body_names
                )

    def _load_named_format(
        self,
        fid: h5py.File,
        group: h5py.Group,
        keep_clips_idx: Optional[Array[int]],
        joint_names: Optional[list[str]],
        body_names: Optional[list[str]],
    ) -> None:
        """Load named-array format (fruitfly-style)."""
        self._is_legacy_format = False

        for k in self._NAMED_ARRAYS:
            if k in group:
                arr = group[k][()]
                self._data_arrays[k] = jp.array(arr)
                if keep_clips_idx is not None:
                    logging.info(f"{k}: Keeping {len(keep_clips_idx)} clips")
                    self._data_arrays[k] = self._data_arrays[k][keep_clips_idx]

        # Load joint names
        if joint_names is not None:
            self._joint_names_list = list(joint_names)
        elif "metadata" in fid and "joint_names" in fid["metadata"]:
            self._joint_names_list = list(
                fid["metadata"]["joint_names"][()].astype(str)
            )
        else:
            n_joints = self._data_arrays["joints"].shape[-1]
            self._joint_names_list = [f"joint_{i}" for i in range(n_joints)]

        # Load body names
        if body_names is not None:
            self._body_names_map = {name: i for i, name in enumerate(body_names)}
        elif "metadata" in fid and "body_names" in fid["metadata"]:
            names = list(fid["metadata"]["body_names"][()].astype(str))
            self._body_names_map = {name: i for i, name in enumerate(names)}
        else:
            n_bodies = self._data_arrays["body_positions"].shape[-2]
            self._body_names_map = {f"body_{i}": i for i in range(n_bodies)}

    def _load_legacy_format(
        self,
        fid: h5py.File,
        n_frames_per_clip: int,
        keep_clips_idx: Optional[Array[int]],
        joint_names: Optional[list[str]],
        body_names: Optional[list[str]],
    ) -> None:
        """Load legacy flat-array format (rodent-style)."""
        self._is_legacy_format = True

        # Load config if available
        if "config" in fid:
            self._config = yaml.safe_load(fid["config"][()])
            self.clip_names = self._extract_clip_names(self._config)

        # Load and reshape data arrays
        for k in self._LEGACY_ARRAYS:
            if k in fid:
                arr = fid[k][()]
                n_clips = arr.shape[0] // n_frames_per_clip
                arr = arr.reshape(n_clips, n_frames_per_clip, *arr.shape[1:])
                self._data_arrays[k] = jp.array(arr)
                if keep_clips_idx is not None:
                    logging.info(f"{k}: Keeping {len(keep_clips_idx)} clips")
                    self._data_arrays[k] = self._data_arrays[k][keep_clips_idx]
                    if self.clip_names is not None:
                        self.clip_names = self.clip_names[keep_clips_idx]

        # Load name mappings
        if "names_qpos" in fid:
            names_qpos = fid["names_qpos"][()].astype(str)
            self._qpos_names = {n: i for (i, n) in enumerate(names_qpos)}
            if joint_names is not None:
                self._joint_names_list = list(joint_names)
            else:
                self._joint_names_list = list(names_qpos[7:])

        if "names_xpos" in fid:
            names_xpos = fid["names_xpos"][()].astype(str)
            self._xpos_names = {n: i for (i, n) in enumerate(names_xpos)}
            if body_names is not None:
                self._body_names_map = {name: i for i, name in enumerate(body_names)}
            else:
                self._body_names_map = {n: i for (i, n) in enumerate(names_xpos)}

    def _extract_clip_names(self, config: Mapping[str, Any]) -> np.ndarray:
        """Extract behavior names from legacy config metadata."""
        original_filenames = config["model"]["snips_order"]
        pattern = r"([A-Za-z]+)_\d+(\.p)?"
        clip_names = []
        for fn in original_filenames:
            m = re.search(pattern, fn)
            if m is None:
                raise ValueError(
                    f"Clip name {fn} does not match pattern {pattern}."
                )
            clip_names.append(m.group(1))
        return np.array(clip_names)

    @property
    def _shape_check_key(self) -> str:
        """Key to use for shape checks."""
        return "qpos" if self._is_legacy_format else "joints"

    @property
    def _data_array_keys(self) -> list[str]:
        """List of data array keys based on format."""
        return self._LEGACY_ARRAYS if self._is_legacy_format else self._NAMED_ARRAYS

    # -------------------------------------------------------------------------
    # Slicing and splitting operations
    # -------------------------------------------------------------------------

    def at(self, clip: int, frame: int) -> "ReferenceClips":
        """Extract a single frame from a specific clip.

        Parameters
        ----------
        clip : int
            Index of the clip to select.
        frame : int
            Index of the frame within the selected clip.

        Returns
        -------
        ReferenceClips
            A new instance with data sliced to the specified clip and frame.
            All array shapes will have their first two dimensions collapsed.

        Raises
        ------
        IndexError
            If called on an already-sliced ReferenceClips instance.
        """
        if len(self._data_arrays[self._shape_check_key].shape) < 3:
            raise IndexError("Trying to slice already sliced ReferenceClip.")
        subslice = copy.copy(self)
        subslice._data_arrays = {
            k: self._data_arrays[k][clip, frame]
            for k in self._data_array_keys
            if k in self._data_arrays
        }
        return subslice

    def slice(self, clip: int, start_frame: int, length: int) -> "ReferenceClips":
        """Extract a contiguous slice of frames from a specific clip.

        Parameters
        ----------
        clip : int
            Index of the clip to slice.
        start_frame : int
            Starting frame index within the clip.
        length : int
            Number of frames to include in the slice.

        Returns
        -------
        ReferenceClips
            A new instance with data sliced to the specified range.
            Arrays will have shape (length, ...) instead of (n_clips, n_frames, ...).

        Raises
        ------
        IndexError
            If called on an already-sliced ReferenceClips instance.
        """
        if len(self._data_arrays[self._shape_check_key].shape) < 3:
            raise IndexError("Trying to slice already sliced ReferenceClip.")
        subslice = copy.copy(self)
        subslice._data_arrays = {}
        for key in self._data_array_keys:
            if key not in self._data_arrays:
                continue
            clip_array = self._data_arrays[key][clip]
            slice_data = jax.lax.dynamic_slice(
                clip_array,
                (start_frame, *jp.zeros(clip_array.ndim - 1, dtype=int)),
                (length, *clip_array.shape[1:]),
            )
            subslice._data_arrays[key] = slice_data
        return subslice

    def split(
        self, train_ratio: float = 0.8, seed: int = 0
    ) -> tuple["ReferenceClips", "ReferenceClips"]:
        """Split the reference clips into train and test sets.

        Randomly shuffles clip indices and splits into training and test sets.
        The split is reproducible given the same seed.

        Parameters
        ----------
        train_ratio : float, default=0.8
            Proportion of clips to use for training (0.0 to 1.0).
            If this results in an empty test set, both train and test
            will contain all clips with a warning.
        seed : int, default=0
            Random seed for reproducible splits.

        Returns
        -------
        train_clips : ReferenceClips
            Training set containing `int(n_clips * train_ratio)` clips.
        test_clips : ReferenceClips
            Test set containing the remaining clips.

        Examples
        --------
        >>> clips = ReferenceClips("data.h5", n_frames_per_clip=250)
        >>> train, test = clips.split(train_ratio=0.8, seed=42)
        >>> print(f"Train: {train.qpos.shape[0]}, Test: {test.qpos.shape[0]}")
        """
        n_clips = self._data_arrays[self._shape_check_key].shape[0]
        n_train = int(n_clips * train_ratio)

        if n_clips == n_train:
            warnings.warn(
                "train_ratio results in empty test set; using full dataset for both."
            )
            logging.info(f"Training clips: {n_train}; Test clips: {n_train}")
            return copy.copy(self), copy.copy(self)

        logging.info(f"Training clips: {n_train}; Test clips: {n_clips - n_train}")

        rng = np.random.RandomState(seed)
        indices = rng.permutation(n_clips)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

        train_clips = copy.copy(self)
        train_clips._data_arrays = {
            k: self._data_arrays[k][train_indices]
            for k in self._data_array_keys
            if k in self._data_arrays
        }
        if self.clip_names is not None:
            train_clips.clip_names = self.clip_names[train_indices]

        test_clips = copy.copy(self)
        test_clips._data_arrays = {
            k: self._data_arrays[k][test_indices]
            for k in self._data_array_keys
            if k in self._data_arrays
        }
        if self.clip_names is not None:
            test_clips.clip_names = self.clip_names[test_indices]

        return train_clips, test_clips

    # -------------------------------------------------------------------------
    # Core data properties
    # -------------------------------------------------------------------------

    @property
    def qpos(self) -> jp.ndarray:
        """Joint positions array."""
        if self._is_legacy_format:
            return self._data_arrays["qpos"]
        return jp.concatenate(
            [self.position, self.quaternion, self.joints], axis=-1
        )

    @property
    def qvel(self) -> jp.ndarray:
        """Joint velocities array."""
        if self._is_legacy_format:
            return self._data_arrays["qvel"]
        return jp.concatenate(
            [self.velocity, self.angular_velocity, self.joints_velocity], axis=-1
        )

    @property
    def root_position(self) -> jp.ndarray:
        """Root XYZ position."""
        if self._is_legacy_format:
            return self.qpos[..., :3]
        return self._data_arrays["position"]

    @property
    def root_quaternion(self) -> jp.ndarray:
        """Root orientation quaternion."""
        if self._is_legacy_format:
            return self.qpos[..., 3:7]
        return self._data_arrays["quaternion"]

    @property
    def joints(self) -> jp.ndarray:
        """Joint angles (excluding root)."""
        if self._is_legacy_format:
            return self.qpos[..., 7:]
        return self._data_arrays["joints"]

    @property
    def joints_velocity(self) -> jp.ndarray:
        """Joint velocities (excluding root)."""
        if self._is_legacy_format:
            return self.qvel[..., 6:]
        return self._data_arrays["joints_velocity"]

    # Named-array format specific properties
    @property
    def position(self) -> jp.ndarray:
        """Root position (named format) or computed from qpos (legacy)."""
        if self._is_legacy_format:
            return self.qpos[..., :3]
        return self._data_arrays["position"]

    @property
    def velocity(self) -> jp.ndarray:
        """Root velocity (named format) or computed from qvel (legacy)."""
        if self._is_legacy_format:
            return self.qvel[..., :3]
        return self._data_arrays["velocity"]

    @property
    def quaternion(self) -> jp.ndarray:
        """Root quaternion (named format) or computed from qpos (legacy)."""
        if self._is_legacy_format:
            return self.qpos[..., 3:7]
        return self._data_arrays["quaternion"]

    @property
    def angular_velocity(self) -> jp.ndarray:
        """Root angular velocity (named format) or computed from qvel (legacy)."""
        if self._is_legacy_format:
            return self.qvel[..., 3:6]
        return self._data_arrays["angular_velocity"]

    @property
    def body_positions(self) -> jp.ndarray:
        """Body positions array."""
        if self._is_legacy_format:
            return self._data_arrays["xpos"]
        return self._data_arrays["body_positions"]

    @property
    def body_quaternions(self) -> jp.ndarray:
        """Body quaternions array."""
        if self._is_legacy_format:
            return self._data_arrays["xquat"]
        return self._data_arrays["body_quaternions"]

    # Legacy format specific properties
    @property
    def xpos(self) -> jp.ndarray:
        """Body positions (legacy name)."""
        return self.body_positions

    @property
    def xquat(self) -> jp.ndarray:
        """Body quaternions (legacy name)."""
        return self.body_quaternions

    # -------------------------------------------------------------------------
    # Metadata properties
    # -------------------------------------------------------------------------

    @property
    def joint_names(self) -> list[str]:
        """List of joint names."""
        return self._joint_names_list

    @property
    def body_names(self) -> list[str]:
        """List of body names."""
        return list(self._body_names_map.keys())

    # -------------------------------------------------------------------------
    # Body access methods
    # -------------------------------------------------------------------------

    def body_xpos(self, name: str) -> jp.ndarray:
        """Get the global position of a body by name."""
        if name not in self._body_names_map:
            raise KeyError(
                f"Body '{name}' not found. Available: {list(self._body_names_map.keys())}"
            )
        idx = self._body_names_map[name]
        return self.body_positions[..., idx, :]

    def body_xquat(self, name: str) -> jp.ndarray:
        """Get the global orientation of a body by name."""
        if name not in self._body_names_map:
            raise KeyError(
                f"Body '{name}' not found. Available: {list(self._body_names_map.keys())}"
            )
        idx = self._body_names_map[name]
        return self.body_quaternions[..., idx, :]
