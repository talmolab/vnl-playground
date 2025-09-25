"""Reference clips for C. elegans motion data.

This module provides utilities for loading and processing motion capture data
for C. elegans, including train/test splitting and data access methods.
"""

from typing import List, Tuple
import copy

import h5py
import jax
import jax.numpy as jp

from vnl_mjx.tasks.celegans.consts import JOINTS


class ReferenceClips:
    """Class for loading and managing C. elegans reference motion clips.

    This class handles loading motion data from HDF5 files and provides
    utilities for accessing specific clips, frames, and slicing data.

    Attributes:
        _DATA_ARRAYS: List of data array names stored in the HDF5 file.
    """

    _DATA_ARRAYS: List[str] = ["qpos", "qvel", "xpos", "xquat"]

    def __init__(self, data_path: str, n_frames_per_clip: int) -> None:
        """Initialize ReferenceClips with data from HDF5 file.

        Args:
            data_path: Path to the HDF5 file containing motion data.
            n_frames_per_clip: Number of frames per motion clip.
        """
        self._n_frames_per_clip = n_frames_per_clip
        self._data_path = data_path
        self._load_from_disk(self._data_path, self._n_frames_per_clip)

    def at(self, clip: int, frame: int) -> "ReferenceClips":
        """Get data at a specific clip and frame.

        Args:
            clip: Index of the clip to access.
            frame: Index of the frame within the clip.

        Returns:
            A new ReferenceClips instance containing data for the specified
            clip and frame.

        Raises:
            IndexError: If trying to slice an already sliced ReferenceClip.
        """
        if self.qpos.ndim < 3:
            raise IndexError("Trying to slice already sliced ReferenceClip.")
        subslice = copy.copy(self)
        subslice._data_arrays = {
            k: subslice._data_arrays[k][clip, frame] for k in self._DATA_ARRAYS
        }
        return subslice

    def slice(self, clip: int, start_frame: int, length: int) -> "ReferenceClips":
        """Extract a slice of frames from a specific clip.

        Args:
            clip: Index of the clip to slice.
            start_frame: Starting frame index for the slice.
            length: Number of frames to include in the slice.

        Returns:
            A new ReferenceClips instance containing the sliced data.

        Raises:
            IndexError: If trying to slice an already sliced ReferenceClip.
        """
        if self.qpos.ndim < 3:
            raise IndexError("Trying to slice already sliced ReferenceClip.")
        subslice = copy.copy(self)
        subslice._data_arrays = copy.copy(self._data_arrays)
        for key in subslice._DATA_ARRAYS:
            clip_array = subslice._data_arrays[key][clip]
            slice = jax.lax.dynamic_slice(
                clip_array,
                (start_frame, *jp.zeros(clip_array.ndim - 1, dtype=int)),
                (length, *clip_array.shape[1:]),
            )
            subslice._data_arrays[key] = slice
        return subslice

    def __len__(self) -> int:
        """Return the number of clips.

        Returns:
            Number of clips in the dataset.
        """
        return len(self.qpos)

    def _load_from_disk(self, data_path: str, n_frames_per_clip: int) -> None:
        """Load motion data from HDF5 file.

        Args:
            data_path: Path to the HDF5 file.
            n_frames_per_clip: Number of frames per clip.
        """
        self._data_arrays = {}
        with h5py.File(data_path, "r") as fid:
            for k in self._DATA_ARRAYS:
                arr = fid[k][()]
                n_clips = arr.shape[0] // n_frames_per_clip
                arr = arr.reshape(n_clips, n_frames_per_clip, *arr.shape[1:])
                self._data_arrays[k] = jp.array(arr)
            self._names_qpos = fid["names_qpos"][()].astype(str)
            self._names_xpos = fid["names_xpos"][()].astype(str)
            self._qpos_names = {n: i for (i, n) in enumerate(self._names_qpos)}
            self._xpos_names = {n: i for (i, n) in enumerate(self._names_xpos)}
        self._clip_idx = jp.arange(n_clips)

    @classmethod
    def generate_train_test_split(
        cls,
        data_path: str,
        n_frames_per_clip: int,
        test_ratio: float = 0.1,
    ) -> Tuple["ReferenceClips", "ReferenceClips"]:
        """Generate train and test splits from the clips.

        The split is done by randomly sampling clips from the dataset.

        Args:
            data_path: Path to the HDF5 file containing motion data.
            n_frames_per_clip: Number of frames per clip.
            test_ratio: Ratio of clips to use for testing. Defaults to 0.1.

        Returns:
            A tuple containing (train_set, test_set) as ReferenceClips objects.
        """
        train_set = cls(data_path, n_frames_per_clip)
        test_set = cls(data_path, n_frames_per_clip)

        num_clips = len(train_set)
        test_size = int(num_clips * test_ratio)

        if test_size == 0:
            import warnings

            warnings.warn(
                "No test set found, please increase the test ratio! Train set and test set will be the same."
            )
        else:
            split_rng = jax.random.PRNGKey(0)
            indices = jp.arange(num_clips)
            test_idx = jax.random.choice(
                split_rng, indices, shape=(test_size,), replace=False
            )
            train_idx = indices[~jp.isin(indices, test_idx)]

            train_idx.sort()
            test_idx.sort()

            train_set._data_arrays = {
                k: train_set._data_arrays[k][train_idx] for k in train_set._DATA_ARRAYS
            }
            test_set._data_arrays = {
                k: test_set._data_arrays[k][test_idx] for k in test_set._DATA_ARRAYS
            }

            train_set._clip_idx = train_idx
            test_set._clip_idx = test_idx

        return train_set, test_set

    @property
    def qpos(self) -> jp.ndarray:
        """Joint positions from the reference data.

        Returns:
            Array of joint positions for all clips and frames.
        """
        return self._data_arrays["qpos"]

    @property
    def qvel(self) -> jp.ndarray:
        """Joint velocities from the reference data.

        Returns:
            Array of joint velocities for all clips and frames.
        """
        return self._data_arrays["qvel"]

    @property
    def xpos(self) -> jp.ndarray:
        """Body positions from the reference data.

        Returns:
            Array of body positions for all clips and frames.
        """
        return self._data_arrays["xpos"]

    @property
    def xquat(self) -> jp.ndarray:
        """Body quaternions from the reference data.

        Returns:
            Array of body quaternions for all clips and frames.
        """
        return self._data_arrays["xquat"]

    @property
    def root_position(self) -> jp.ndarray:
        """Root body position from the reference data.

        Returns:
            Position of the root body (typically index 1 in xpos).
        """
        return self.xpos[..., 1, :]

    @property
    def root_quaternion(self) -> jp.ndarray:
        """Root body quaternion from the reference data.

        Returns:
            Quaternion of the root body (typically index 1 in xquat).
        """
        return self.xquat[..., 1, :]

    @property
    def joints(self) -> jp.ndarray:
        """Joint angles for joints defined in JOINTS constant.

        Returns:
            Array of joint angles filtered to include only joints in JOINTS.
        """
        joint_idx = [idx for name, idx in self._qpos_names.items() if name in JOINTS]
        return self.qpos[..., joint_idx]

    @property
    def joints_velocity(self) -> jp.ndarray:
        """Joint velocities for joints defined in JOINTS constant.

        Returns:
            Array of joint velocities filtered to include only joints in JOINTS.
        """
        start_idx = self.qvel.shape[-1] - len(JOINTS)
        return self.qvel[..., start_idx:]

    @property
    def joint_names(self) -> List[str]:
        """Get list of joint names.

        Returns:
            List of joint names that are included in JOINTS constant.
        """
        return [name for name in self._names_qpos if name in JOINTS]

    @property
    def body_names(self) -> jp.ndarray:
        """Get list of body names.

        Returns:
            Array of body names from the reference data.
        """
        return self._names_xpos

    def body_xpos(self, name: str) -> jp.ndarray:
        """Get position data for a specific body.

        Args:
            name: Name of the body.

        Returns:
            Position array for the specified body.
        """
        return self.xpos[..., self._xpos_names[name], :]

    def body_xquat(self, name: str) -> jp.ndarray:
        """Get quaternion data for a specific body.

        Args:
            name: Name of the body.

        Returns:
            Quaternion array for the specified body.
        """
        return self.xquat[..., self._xpos_names[name], :]

    @property
    def n_frames(self) -> int:
        """Get number of frames in the clips.

        Returns:
            Number of frames per clip.
        """
        return self.qpos.shape[1]

    @property
    def n_clips(self) -> int:
        """Get number of clips in the dataset.

        Returns:
            Number of clips available.
        """
        return self.qpos.shape[0] if self.qpos.ndim >= 3 else 1

    @property
    def data_path(self) -> str:
        """Get the path to the data file.

        Returns:
            Path to the HDF5 file containing the motion data.
        """
        return self._data_path

    @property
    def frames_per_clip(self) -> int:
        """Get the number of frames per clip.

        Returns:
            Number of frames per motion clip.
        """
        return self._n_frames_per_clip

    @property
    def clip_indices(self) -> jp.ndarray:
        """Get the indices of clips in this dataset.

        Returns:
            Array of clip indices.
        """
        return self._clip_idx

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the motion data.

        Returns:
            Shape tuple of (n_clips, n_frames, ...) or (n_frames, ...) for sliced data.
        """
        return self.qpos.shape

    @property
    def is_sliced(self) -> bool:
        """Check if this is a sliced reference clips object.

        Returns:
            True if this object represents a single clip/frame slice.
        """
        return self.qpos.ndim < 3

    @property
    def joint_indices(self) -> List[int]:
        """Get indices of joints that are included in JOINTS constant.

        Returns:
            List of indices for joints included in JOINTS.
        """
        return [idx for name, idx in self._qpos_names.items() if name in JOINTS]
