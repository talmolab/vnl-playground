from calendar import c
import copy
import h5py

import jax
import jax.numpy as jp


class ReferenceClips:

    _DATA_ARRAYS = ["qpos", "qvel", "xpos", "xquat"]

    def __init__(self, data_path: str, n_frames_per_clip: int):
        self._load_from_disk(data_path, n_frames_per_clip)

    def at(self, clip: int, frame: int) -> "ReferenceClips":
        if self.qpos.ndim < 3:
            raise IndexError("Trying to slice already sliced ReferenceClip.")
        subslice = copy.copy(self)
        subslice._data_arrays = {k: subslice._data_arrays[k][clip, frame] for k in self._DATA_ARRAYS}
        return subslice
    
    def slice(self, clip: int, start_frame: int, length: int) -> "ReferenceClips":
        if self.qpos.ndim < 3:
            raise IndexError("Trying to slice already sliced ReferenceClip.")
        subslice = copy.copy(self)
        subslice._data_arrays = copy.copy(self._data_arrays)
        for key in subslice._DATA_ARRAYS:
            clip_array = subslice._data_arrays[key][clip]
            slice = jax.lax.dynamic_slice(clip_array,
                                          (start_frame, *jp.zeros(clip_array.ndim - 1, dtype=int)),
                                          (length, *clip_array.shape[1:]))
            subslice._data_arrays[key] = slice
        return subslice

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
        return self.qpos[..., :3]
    
    @property
    def root_quaternion(self) -> jp.ndarray:
        return self.xquat[..., 1, :]
    
    @property
    def joints(self) -> jp.ndarray:
        return self.qpos[..., [idx for name, idx in self._qpos_names.items() if name in consts.JOINTS]]
    
    @property
    def joints_velocity(self) -> jp.ndarray:
        return self.qvel[..., 6:]
    
    @property
    def joint_names(self):
        return self._names_qpos[7:]
    
    @property
    def body_names(self):
        return self._names_xpos
    
    def body_xpos(self, name: str) -> jp.ndarray:
        return self.xpos[..., self._xpos_names[name], :]
    
    def body_xquat(self, name: str) -> jp.ndarray:
        return self.xquat[..., self._xpos_names[name], :]
    