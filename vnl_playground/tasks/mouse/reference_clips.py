"""Reference clip loading for mouse arm imitation."""

import copy
import glob
import os
from typing import List, Optional, TYPE_CHECKING

import h5py
import jax
import jax.numpy as jp
import mujoco
import numpy as np
import yaml

if TYPE_CHECKING:
    from mujoco import MjModel


class MouseReferenceClips:
    """Load and manage reference clips for mouse arm imitation.

    Unlike the rodent ReferenceClips which loads from a single h5 file,
    this class loads from multiple h5 files (one per clip) and handles
    the fixed-base arm structure (no root freejoint).
    """

    _DATA_ARRAYS = ["qpos", "qvel", "xpos", "xquat"]

    def __init__(
        self,
        data_path: str,
        n_frames_per_clip: Optional[int] = None,
        keep_clips_idx: Optional[List[int]] = None,
    ):
        """Load reference clips from a directory of h5 files.

        Args:
            data_path: Path to directory containing h5 files.
            n_frames_per_clip: Number of frames per clip (for consistency with rodent API).
                              If None, uses the native frame count from files.
            keep_clips_idx: Indices of clips to keep. If None, all clips are kept.
        """
        self._data_path = data_path
        self._n_frames_per_clip = n_frames_per_clip
        self._load_from_disk(data_path, keep_clips_idx)

    def _load_from_disk(
        self, data_path: str, keep_clips_idx: Optional[List[int]] = None
    ):
        """Load all h5 files from the directory into stacked JAX arrays.

        Args:
            data_path: Path to directory containing h5 files (one per clip).
            keep_clips_idx: Optional indices of clips to keep after loading.
                If None, all clips are retained.
        """
        # Find all h5 files
        h5_files = sorted(glob.glob(os.path.join(str(data_path), "*.h5")))
        if not h5_files:
            raise ValueError(f"No h5 files found in {data_path}")

        # Load first file to get structure
        with h5py.File(h5_files[0], "r") as f:
            self._config = yaml.safe_load(f["config"][()])
            self._names_qpos = f["names_qpos"][()].astype(str)
            self._names_xpos = f["names_xpos"][()].astype(str)
            n_frames = f["qpos"].shape[0]

        self._qpos_names = {n: i for (i, n) in enumerate(self._names_qpos)}
        self._xpos_names = {n: i for (i, n) in enumerate(self._names_xpos)}

        # If n_frames_per_clip not specified, use native frame count
        if self._n_frames_per_clip is None:
            self._n_frames_per_clip = n_frames

        # Load all clips
        all_qpos = []
        all_qvel = []
        all_xpos = []
        all_xquat = []
        clip_names = []

        for h5_path in h5_files:
            with h5py.File(h5_path, "r") as f:
                qpos = f["qpos"][()]
                qvel = f["qvel"][()]
                xpos = f["xpos"][()]
                xquat = f["xquat"][()]

                # Truncate or pad to n_frames_per_clip
                n = min(qpos.shape[0], self._n_frames_per_clip)
                all_qpos.append(qpos[:n])
                all_qvel.append(qvel[:n])
                all_xpos.append(xpos[:n])
                all_xquat.append(xquat[:n])

            # Extract clip name from filename
            clip_name = os.path.basename(h5_path).replace("_ik.h5", "")
            clip_names.append(clip_name)

        # Stack into arrays: (n_clips, n_frames, ...)
        self._data_arrays = {
            "qpos": jp.array(np.stack(all_qpos, axis=0)),
            "qvel": jp.array(np.stack(all_qvel, axis=0)),
            "xpos": jp.array(np.stack(all_xpos, axis=0)),
            "xquat": jp.array(np.stack(all_xquat, axis=0)),
        }
        self.clip_names = np.array(clip_names)

        # Apply clip filtering if requested
        if keep_clips_idx is not None:
            keep_clips_idx = np.array(keep_clips_idx)
            for k in self._DATA_ARRAYS:
                self._data_arrays[k] = self._data_arrays[k][keep_clips_idx]
            self.clip_names = self.clip_names[keep_clips_idx]

        print(
            f"Loaded {len(self.clip_names)} clips with {self._n_frames_per_clip} frames each"
        )
        print(f"  qpos shape: {self._data_arrays['qpos'].shape}")
        print(f"  xpos shape: {self._data_arrays['xpos'].shape}")

    def recompute_kinematics(
        self, mj_model: "MjModel", strip_suffix: str = "-mouse"
    ) -> None:
        """Recompute xpos and xquat using the given MuJoCo model.

        This is necessary when the reference data was generated with a different
        model than the simulation model. The qpos values are used to compute
        new xpos/xquat values that are consistent with the simulation model.

        Args:
            mj_model: MuJoCo model to use for forward kinematics.
            strip_suffix: Suffix to strip from body names (e.g., "-mouse") to maintain
                compatibility with code that uses base body names.
        """
        print("Recomputing reference kinematics using simulation model...")

        qpos = np.array(self._data_arrays["qpos"])
        n_clips, n_frames, n_joints = qpos.shape

        # Get body names from the simulation model, stripping suffix if present
        model_body_names = []
        for i in range(mj_model.nbody):
            name = mj_model.body(i).name
            if strip_suffix and name.endswith(strip_suffix):
                name = name[: -len(strip_suffix)]
            model_body_names.append(name)

        # Create new xpos/xquat arrays with simulation model's body count
        n_bodies = mj_model.nbody
        new_xpos = np.zeros((n_clips, n_frames, n_bodies, 3), dtype=np.float32)
        new_xquat = np.zeros((n_clips, n_frames, n_bodies, 4), dtype=np.float32)

        mj_data = mujoco.MjData(mj_model)

        for clip_idx in range(n_clips):
            for frame_idx in range(n_frames):
                mj_data.qpos[:] = qpos[clip_idx, frame_idx]
                mujoco.mj_forward(mj_model, mj_data)
                new_xpos[clip_idx, frame_idx] = mj_data.xpos
                new_xquat[clip_idx, frame_idx] = mj_data.xquat

        # Update data arrays
        self._data_arrays["xpos"] = jp.array(new_xpos)
        self._data_arrays["xquat"] = jp.array(new_xquat)

        # Update body name mappings to use simulation model's names (with suffix stripped)
        self._names_xpos = np.array(model_body_names)
        self._xpos_names = {n: i for (i, n) in enumerate(model_body_names)}

        print(f"  Recomputed xpos shape: {self._data_arrays['xpos'].shape}")
        print(f"  Body names: {model_body_names}")

    def at(self, clip: int, frame: int) -> "MouseReferenceClips":
        """Get reference data at a specific clip and frame.

        Args:
            clip: Clip index.
            frame: Frame index within the clip.

        Returns:
            A new MouseReferenceClips with data at the specified location.
        """
        if len(self.qpos.shape) < 3:
            raise IndexError("Trying to slice already sliced ReferenceClip.")
        subslice = copy.copy(self)
        subslice._data_arrays = {
            k: self._data_arrays[k][clip, frame] for k in self._DATA_ARRAYS
        }
        return subslice

    def slice(self, clip: int, start_frame: int, length: int) -> "MouseReferenceClips":
        """Extract a contiguous slice of frames from a specific clip.

        Args:
            clip: Index of the clip to slice.
            start_frame: Starting frame index.
            length: Number of frames to include.

        Returns:
            A new MouseReferenceClips with the sliced data.
        """
        if len(self.qpos.shape) < 3:
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
    ) -> tuple["MouseReferenceClips", "MouseReferenceClips"]:
        """Split clips into train and test sets.

        Args:
            train_ratio: Proportion of clips for training (0.0 to 1.0).
            seed: Random seed for reproducible splits.

        Returns:
            Tuple of (train_clips, test_clips).
        """
        n_clips = self.qpos.shape[0]
        n_train = int(n_clips * train_ratio)

        if n_clips == n_train:
            print("train_ratio results in empty test set; using full dataset for both")
            return copy.copy(self), copy.copy(self)

        rng = np.random.RandomState(seed)
        indices = rng.permutation(n_clips)

        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

        train_clips = copy.copy(self)
        train_clips._data_arrays = {
            k: self._data_arrays[k][train_indices] for k in self._DATA_ARRAYS
        }
        train_clips.clip_names = self.clip_names[train_indices]

        test_clips = copy.copy(self)
        test_clips._data_arrays = {
            k: self._data_arrays[k][test_indices] for k in self._DATA_ARRAYS
        }
        test_clips.clip_names = self.clip_names[test_indices]

        print(
            f"Split: {len(train_indices)} train clips, {len(test_indices)} test clips"
        )
        return train_clips, test_clips

    @property
    def qpos(self) -> jp.ndarray:
        """Joint positions: (n_clips, n_frames, n_joints) or (n_frames, n_joints) if sliced."""
        return self._data_arrays["qpos"]

    @property
    def qvel(self) -> jp.ndarray:
        """Joint velocities: (n_clips, n_frames, n_joints) or similar."""
        return self._data_arrays["qvel"]

    @property
    def xpos(self) -> jp.ndarray:
        """Body positions: (n_clips, n_frames, n_bodies, 3) or similar."""
        return self._data_arrays["xpos"]

    @property
    def xquat(self) -> jp.ndarray:
        """Body quaternions: (n_clips, n_frames, n_bodies, 4) or similar."""
        return self._data_arrays["xquat"]

    @property
    def joints(self) -> jp.ndarray:
        """Joint angles (same as qpos for fixed-base arm)."""
        return self.qpos

    @property
    def joints_velocity(self) -> jp.ndarray:
        """Joint velocities (same as qvel for fixed-base arm)."""
        return self.qvel

    @property
    def joint_names(self):
        """Names of the joints."""
        return self._names_qpos

    @property
    def body_names(self):
        """Names of the bodies."""
        return self._names_xpos

    def body_xpos(self, name: str) -> jp.ndarray:
        """Get position of a specific body.

        Args:
            name: Body name.

        Returns:
            Body position array.
        """
        return self.xpos[..., self._xpos_names[name], :]

    def body_xquat(self, name: str) -> jp.ndarray:
        """Get quaternion of a specific body.

        Args:
            name: Body name.

        Returns:
            Body quaternion array.
        """
        return self.xquat[..., self._xpos_names[name], :]
