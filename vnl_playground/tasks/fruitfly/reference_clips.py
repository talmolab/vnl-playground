"""Reference clips loader for fruitfly motion capture data."""

import copy
from typing import Any, Mapping, Optional
from ctypes import Array

import h5py
import jax
import jax.numpy as jp
import numpy as np
import logging
import warnings

from vnl_playground.tasks.fruitfly import consts


class ReferenceClips:
    """Reference clips loader for fruitfly motion capture data.

    The fruitfly H5 data has a different structure from rodent:
    - Data is stored under 'all_clips' group
    - Arrays: position, velocity, quaternion, angular_velocity,
              joints, joints_velocity, body_positions, body_quaternions
    - Data is already organized by clips (n_clips, n_frames, ...)
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
    ):
        """
        Load reference clips from a h5 file.

        Args:
            data_path (str): Path to the h5 data file.
            n_frames_per_clip (int): Number of frames in each clip.
            keep_clips_idx (Array[int]): Indices of the clips to keep. If None, all
                                         clips are kept.
        """
        self._load_from_disk(data_path, n_frames_per_clip, keep_clips_idx)

    def at(self, clip: int, frame: int) -> "ReferenceClips":
        """
        Create a ReferenceClips subarray at the specified clip and frame indices.

        Args:
            clip (int): The index of the clip to select.
            frame (int): The index of the frame within the selected clip.

        Returns:
            A new ReferenceClips instance with each field sliced to the specified
            clip and frame.
        """
        if len(self._data_arrays["joints"].shape) < 3:
            raise IndexError("Trying to slice already sliced ReferenceClip.")
        subslice = copy.copy(self)
        subslice._data_arrays = {
            k: self._data_arrays[k][clip, frame] for k in self._DATA_ARRAYS
        }
        return subslice

    def slice(self, clip: int, start_frame: int, length: int) -> "ReferenceClips":
        """
        Extracts a contiguous slice of frames from a specific clip.

        Args:
            clip (int): Index of the clip to slice.
            start_frame (int): The starting frame index for the slice.
            length (int): The number of frames to include in the slice.

        Returns:
            ReferenceClips: A new ReferenceClips object containing the sliced data.
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
        """
        Split the reference clips into train and test sets.

        Args:
            train_ratio (float): Proportion of clips to use for training (0.0 to 1.0).
            seed (int): Random seed for reproducible splits.

        Returns:
            tuple[ReferenceClips, ReferenceClips]: (train_clips, test_clips)
        """
        n_clips = self._data_arrays["joints"].shape[0]
        n_train = int(n_clips * train_ratio)

        if n_clips == n_train:
            warnings.warn(
                "train_ratio results in an empty test set; using full dataset for both."
            )
            logging.info(
                f"Number of training clips: {n_train}; Number of test clips: {n_train}"
            )
            return copy.copy(self), copy.copy(self)

        logging.info(
            f"Number of training clips: {n_train}; Number of test clips: {n_clips - n_train}"
        )

        rng = np.random.RandomState(seed)
        indices = rng.permutation(n_clips)

        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

        train_clips = copy.copy(self)
        train_clips._data_arrays = {
            k: self._data_arrays[k][train_indices] for k in self._DATA_ARRAYS
        }

        test_clips = copy.copy(self)
        test_clips._data_arrays = {
            k: self._data_arrays[k][test_indices] for k in self._DATA_ARRAYS
        }

        return train_clips, test_clips

    def _load_from_disk(
        self,
        data_path: str,
        n_frames_per_clip: int,
        keep_clips_idx: Optional[Array[int]],
    ):
        """Load data from H5 file."""
        self._data_arrays = {}
        with h5py.File(data_path, "r") as fid:
            # Fruitfly data is stored under 'all_clips' group
            group = fid["all_clips"] if "all_clips" in fid else fid

            for k in self._DATA_ARRAYS:
                if k in group:
                    arr = group[k][()]
                    self._data_arrays[k] = jp.array(arr)
                    if keep_clips_idx is not None:
                        logging.info(f"{k}: Keeping {len(keep_clips_idx)} clips")
                        self._data_arrays[k] = self._data_arrays[k][keep_clips_idx]

        # Build body name to index mapping using config constants
        self._body_names = {name: i for i, name in enumerate(consts.BODIES)}

        # Store joint names
        self._joint_names = consts.JOINTS

    @property
    def position(self) -> jp.ndarray:
        """Root position array."""
        return self._data_arrays["position"]

    @property
    def velocity(self) -> jp.ndarray:
        """Root velocity array."""
        return self._data_arrays["velocity"]

    @property
    def quaternion(self) -> jp.ndarray:
        """Root quaternion array."""
        return self._data_arrays["quaternion"]

    @property
    def angular_velocity(self) -> jp.ndarray:
        """Root angular velocity array."""
        return self._data_arrays["angular_velocity"]

    @property
    def joints(self) -> jp.ndarray:
        """Joint angles array (36 joints)."""
        return self._data_arrays["joints"]

    @property
    def joints_velocity(self) -> jp.ndarray:
        """Joint velocities array (36 joints)."""
        return self._data_arrays["joints_velocity"]

    @property
    def body_positions(self) -> jp.ndarray:
        """Body positions array (68 bodies, 3D)."""
        return self._data_arrays["body_positions"]

    @property
    def body_quaternions(self) -> jp.ndarray:
        """Body quaternions array (68 bodies, 4D)."""
        return self._data_arrays["body_quaternions"]

    # Aliases for compatibility with rodent-style interface
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
        - joints (36)
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
        - joints_velocity (36)
        """
        return jp.concatenate(
            [self.velocity, self.angular_velocity, self.joints_velocity], axis=-1
        )

    @property
    def joint_names(self):
        """Return joint names."""
        return self._joint_names

    @property
    def body_names(self):
        """Return body names."""
        return list(self._body_names.keys())

    def body_xpos(self, name: str) -> jp.ndarray:
        """Get the reference for a global euclidean position of a body part.

        Args:
            name (str): Name of the body part.

        Returns:
            jp.ndarray: The global position of the body part.
        """
        if name not in self._body_names:
            raise KeyError(f"Body '{name}' not found. Available: {list(self._body_names.keys())}")
        idx = self._body_names[name]
        return self.body_positions[..., idx, :]

    def body_xquat(self, name: str) -> jp.ndarray:
        """Get the reference for a global orientation of a body part.

        Args:
            name (str): Name of the body part.

        Returns:
            jp.ndarray: The global orientation of the body part as a quaternion.
        """
        if name not in self._body_names:
            raise KeyError(f"Body '{name}' not found. Available: {list(self._body_names.keys())}")
        idx = self._body_names[name]
        return self.body_quaternions[..., idx, :]
