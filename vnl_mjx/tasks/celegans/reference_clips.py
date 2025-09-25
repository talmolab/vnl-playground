from typing import Tuple, Self
from calendar import c
import copy
import h5py

import jax
import jax.numpy as jp

from vnl_mjx.tasks.celegans.consts import JOINTS


class ReferenceClips:
    _DATA_ARRAYS = ["qpos", "qvel", "xpos", "xquat"]

    def __init__(self, data_path: str, n_frames_per_clip: int):
        self._n_frames_per_clip = n_frames_per_clip
        self._data_path = data_path
        self._load_from_disk(self._data_path, self._n_frames_per_clip)

    def at(self, clip: int, frame: int) -> "ReferenceClips":
        if self.qpos.ndim < 3:
            raise IndexError("Trying to slice already sliced ReferenceClip.")
        subslice = copy.copy(self)
        subslice._data_arrays = {
            k: subslice._data_arrays[k][clip, frame] for k in self._DATA_ARRAYS
        }
        return subslice

    def slice(self, clip: int, start_frame: int, length: int) -> "ReferenceClips":
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

    def __len__(self):
        return len(self.qpos)

    def _load_from_disk(self, data_path: str, n_frames_per_clip: int):
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
    ) -> Tuple[Self, Self]:
        """
        Generates a train-test split of the clips based on the provided ratio.
        The split is done by randomly sampling clips from the metadata list.
        The function returns two ReferenceClip objects: one for training and one for testing.

        Args:
            data (ReferenceClip): The ReferenceClip object containing the clips to be split.
            test_ratio (float, optional): ratio of the test set. Defaults to 0.1.

        Returns:
            Tuple[ReferenceClip, ReferenceClip]: training set and testing set as ReferenceClip objects.
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
        return self._data_arrays["qpos"]

    @property
    def qvel(self) -> jp.ndarray:
        return self._data_arrays["qvel"]

    @property
    def xpos(self) -> jp.ndarray:
        return self._data_arrays["xpos"]

    @property
    def xquat(self) -> jp.ndarray:
        return self._data_arrays["xquat"]

    @property
    def root_position(self) -> jp.ndarray:
        return self.xpos[..., 1, :]

    @property
    def root_quaternion(self) -> jp.ndarray:
        return self.xquat[..., 1, :]

    @property
    def joints(self) -> jp.ndarray:
        joint_idx = [idx for name, idx in self._qpos_names.items() if name in JOINTS]
        return self.qpos[..., joint_idx]

    @property
    def joints_velocity(self) -> jp.ndarray:
        start_idx = self.qvel.shape[-1] - len(JOINTS)
        return self.qvel[..., start_idx:]

    @property
    def joint_names(self):
        return [name for name in self._names_qpos if name in JOINTS]

    @property
    def body_names(self):
        return self._names_xpos

    def body_xpos(self, name: str) -> jp.ndarray:
        return self.xpos[..., self._xpos_names[name], :]

    def body_xquat(self, name: str) -> jp.ndarray:
        return self.xquat[..., self._xpos_names[name], :]

    @property
    def n_frames(self):
        return self.qpos.shape[1]
