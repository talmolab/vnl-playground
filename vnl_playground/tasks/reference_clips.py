"""Unified reference clips loader for motion capture data.

This module provides a single ReferenceClips class that loads motion capture
data from H5 files using the named-array pattern:
- position, velocity, quaternion, angular_velocity (root state)
- joints, joints_velocity (joint state)
- body_positions, body_quaternions (body state)

This pattern is more readable and maintainable than flat qpos/qvel arrays
with magic slice indices.
"""

import copy
from ctypes import Array
from typing import Optional

import h5py
import jax
import jax.numpy as jp
import numpy as np
import logging
import warnings


class ReferenceClips:
    """Reference clips loader for motion capture data.

    Loads data from H5 files with named arrays for semantic clarity.
    Works with any organism (rodent, fruitfly, etc.) as long as the H5
    file follows the expected structure.

    Expected H5 structure:
        /all_clips/  (or root level)
            position          (n_clips, n_frames, 3)
            velocity          (n_clips, n_frames, 3)
            quaternion        (n_clips, n_frames, 4)
            angular_velocity  (n_clips, n_frames, 3)
            joints            (n_clips, n_frames, n_joints)
            joints_velocity   (n_clips, n_frames, n_joints)
            body_positions    (n_clips, n_frames, n_bodies, 3)
            body_quaternions  (n_clips, n_frames, n_bodies, 4)
        /metadata/  (optional)
            joint_names       (n_joints,)
            body_names        (n_bodies,)
    """

    _DATA_ARRAYS = [
        "position",
        "velocity",
        "quaternion",
        "angular_velocity",
        "joints",
        "joints_velocity",
        "body_positions",
        "body_quaternions",
    ]

    def __init__(
        self,
        data_path: str,
        n_frames_per_clip: int,
        keep_clips_idx: Optional[Array[int]] = None,
        joint_names: Optional[list[str]] = None,
        body_names: Optional[list[str]] = None,
    ):
        """Load reference clips from an H5 file.

        Args:
            data_path: Path to the H5 data file.
            n_frames_per_clip: Number of frames in each clip (used for validation).
            keep_clips_idx: Optional indices of clips to keep. If None, all
                clips are kept.
            joint_names: Optional list of joint names. If None, will try to
                read from H5 metadata or use indices.
            body_names: Optional list of body names. If None, will try to
                read from H5 metadata or use indices.
        """
        self._data_arrays = {}
        self._joint_names_list: list[str] = []
        self._body_names_map: dict[str, int] = {}
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
        """Load data from H5 file."""
        with h5py.File(data_path, "r") as fid:
            # Data may be at root level or under 'all_clips' group
            group = fid["all_clips"] if "all_clips" in fid else fid

            for k in self._DATA_ARRAYS:
                if k in group:
                    arr = group[k][()]
                    self._data_arrays[k] = jp.array(arr)
                    if keep_clips_idx is not None:
                        logging.info(f"{k}: Keeping {len(keep_clips_idx)} clips")
                        self._data_arrays[k] = self._data_arrays[k][keep_clips_idx]

            # Load joint names from parameter, H5 metadata, or generate indices
            if joint_names is not None:
                self._joint_names_list = list(joint_names)
            elif "metadata" in fid and "joint_names" in fid["metadata"]:
                self._joint_names_list = list(fid["metadata"]["joint_names"][()].astype(str))
            else:
                n_joints = self._data_arrays["joints"].shape[-1]
                self._joint_names_list = [f"joint_{i}" for i in range(n_joints)]

            # Load body names from parameter, H5 metadata, or generate indices
            if body_names is not None:
                self._body_names_map = {name: i for i, name in enumerate(body_names)}
            elif "metadata" in fid and "body_names" in fid["metadata"]:
                names = list(fid["metadata"]["body_names"][()].astype(str))
                self._body_names_map = {name: i for i, name in enumerate(names)}
            else:
                n_bodies = self._data_arrays["body_positions"].shape[-2]
                self._body_names_map = {f"body_{i}": i for i in range(n_bodies)}

    # -------------------------------------------------------------------------
    # Slicing and splitting operations
    # -------------------------------------------------------------------------

    def at(self, clip: int, frame: int) -> "ReferenceClips":
        """Extract a single frame from a specific clip.

        Args:
            clip: Index of the clip to select.
            frame: Index of the frame within the selected clip.

        Returns:
            A new ReferenceClips instance with each field sliced to the
            specified clip and frame.

        Raises:
            IndexError: If trying to slice an already-sliced ReferenceClips.
        """
        if len(self._data_arrays["joints"].shape) < 3:
            raise IndexError("Trying to slice already sliced ReferenceClip.")
        subslice = copy.copy(self)
        subslice._data_arrays = {
            k: self._data_arrays[k][clip, frame] for k in self._DATA_ARRAYS
        }
        return subslice

    def slice(self, clip: int, start_frame: int, length: int) -> "ReferenceClips":
        """Extract a contiguous slice of frames from a specific clip.

        Args:
            clip: Index of the clip to slice.
            start_frame: The starting frame index for the slice.
            length: The number of frames to include in the slice.

        Returns:
            A new ReferenceClips instance containing the sliced data.

        Raises:
            IndexError: If trying to slice an already-sliced ReferenceClips.
        """
        if len(self._data_arrays["joints"].shape) < 3:
            raise IndexError("Trying to slice already sliced ReferenceClip.")
        subslice = copy.copy(self)
        subslice._data_arrays = {}
        for key in self._DATA_ARRAYS:
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

        Args:
            train_ratio: Proportion of clips to use for training (0.0 to 1.0).
                If set to 1.0, the full dataset is used for both train and test.
            seed: Random seed for reproducible splits.

        Returns:
            Tuple of (train_clips, test_clips) ReferenceClips instances.
        """
        n_clips = self._data_arrays["joints"].shape[0]
        n_train = int(n_clips * train_ratio)

        if n_clips == n_train:
            warnings.warn(
                "train_ratio results in an empty test set; "
                "using full dataset for both train and test."
            )
            logging.info(
                f"Number of training clips: {n_train}; Number of test clips: {n_train}"
            )
            return copy.copy(self), copy.copy(self)

        logging.info(
            f"Number of training clips: {n_train}; "
            f"Number of test clips: {n_clips - n_train}"
        )

        # Shuffle indices with seed for reproducibility
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n_clips)

        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

        # Create new ReferenceClips instances with filtered data
        train_clips = copy.copy(self)
        train_clips._data_arrays = {
            k: self._data_arrays[k][train_indices] for k in self._DATA_ARRAYS
        }

        test_clips = copy.copy(self)
        test_clips._data_arrays = {
            k: self._data_arrays[k][test_indices] for k in self._DATA_ARRAYS
        }

        return train_clips, test_clips

    # -------------------------------------------------------------------------
    # Named data array properties (direct access - preferred pattern)
    # -------------------------------------------------------------------------

    @property
    def position(self) -> jp.ndarray:
        """Root position array (3D)."""
        return self._data_arrays["position"]

    @property
    def velocity(self) -> jp.ndarray:
        """Root velocity array (3D)."""
        return self._data_arrays["velocity"]

    @property
    def quaternion(self) -> jp.ndarray:
        """Root quaternion array (4D)."""
        return self._data_arrays["quaternion"]

    @property
    def angular_velocity(self) -> jp.ndarray:
        """Root angular velocity array (3D)."""
        return self._data_arrays["angular_velocity"]

    @property
    def joints(self) -> jp.ndarray:
        """Joint angles array."""
        return self._data_arrays["joints"]

    @property
    def joints_velocity(self) -> jp.ndarray:
        """Joint velocities array."""
        return self._data_arrays["joints_velocity"]

    @property
    def body_positions(self) -> jp.ndarray:
        """Body positions array (n_bodies, 3D)."""
        return self._data_arrays["body_positions"]

    @property
    def body_quaternions(self) -> jp.ndarray:
        """Body quaternions array (n_bodies, 4D)."""
        return self._data_arrays["body_quaternions"]

    # -------------------------------------------------------------------------
    # Compatibility properties (for code expecting qpos/qvel interface)
    # -------------------------------------------------------------------------

    @property
    def root_position(self) -> jp.ndarray:
        """Root position (alias for position)."""
        return self.position

    @property
    def root_quaternion(self) -> jp.ndarray:
        """Root quaternion (alias for quaternion)."""
        return self.quaternion

    @property
    def qpos(self) -> jp.ndarray:
        """Construct qpos from position, quaternion, and joints.

        Returns array of shape (..., 7 + n_joints) containing:
        - position (3)
        - quaternion (4)
        - joints (n_joints)
        """
        return jp.concatenate(
            [self.position, self.quaternion, self.joints], axis=-1
        )

    @property
    def qvel(self) -> jp.ndarray:
        """Construct qvel from velocity, angular_velocity, and joints_velocity.

        Returns array of shape (..., 6 + n_joints) containing:
        - velocity (3)
        - angular_velocity (3)
        - joints_velocity (n_joints)
        """
        return jp.concatenate(
            [self.velocity, self.angular_velocity, self.joints_velocity], axis=-1
        )

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
        """Get the global position of a body by name.

        Args:
            name: Name of the body part.

        Returns:
            Global XYZ position of the body.

        Raises:
            KeyError: If body name is not found.
        """
        if name not in self._body_names_map:
            raise KeyError(
                f"Body '{name}' not found. Available: {list(self._body_names_map.keys())}"
            )
        idx = self._body_names_map[name]
        return self.body_positions[..., idx, :]

    def body_xquat(self, name: str) -> jp.ndarray:
        """Get the global orientation of a body by name.

        Args:
            name: Name of the body part.

        Returns:
            Global orientation quaternion of the body.

        Raises:
            KeyError: If body name is not found.
        """
        if name not in self._body_names_map:
            raise KeyError(
                f"Body '{name}' not found. Available: {list(self._body_names_map.keys())}"
            )
        idx = self._body_names_map[name]
        return self.body_quaternions[..., idx, :]
