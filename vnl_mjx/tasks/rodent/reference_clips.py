import copy
import re
from typing import Any, Mapping

import h5py
import jax
import jax.numpy as jp
import numpy as np
import yaml


class ReferenceClips:
    _DATA_ARRAYS = ["qpos", "qvel", "xpos", "xquat"]

    def __init__(self, data_path: str, n_frames_per_clip: int):
        """
        Load reference clips from a h5 file.
        Args:
            data_path (str): Path to the h5 data file.
            n_frames_per_clip (int): Number of frames in each clip. This is
                                     needed because the clips are stored in
                                     a contiguous array in the h5 file.
        """

        self._load_from_disk(data_path, n_frames_per_clip)

    def at(self, clip: int, frame: int) -> "ReferenceClips":
        """
        Create a ReferenceClips subarray at the specified clip and frame indices.
        Args:
            clip (int):  The index of the clip to select.
            frame (int): The index of the frame within the selected clip.
        Returns:
            A new ReferenceClips instance with each field sliced to the specified
            clip and frame.
        """
        if len(self.qpos.shape) < 3:
            raise IndexError("Trying to slice already sliced ReferenceClip.")
        subslice = copy.copy(self)
        subslice._data_arrays = {
            k: self._data_arrays[k][clip, frame] for k in self._DATA_ARRAYS
        }
        return subslice

    def slice(self, clip: int, start_frame: int, length: int) -> "ReferenceClips":
        """
        Extracts a contiguous slice of frames from a specific clip in the
        ReferenceClips object.
        
        Args:
            clip (int): Index of the clip to slice.
            start_frame (int): The starting frame index for the slice.
            length (int): The number of frames to include in the slice.
        Returns:
            ReferenceClips: A new ReferenceClips object containing the sliced data.
        """
        if len(self.qpos.shape) < 3:
            raise IndexError("Trying to slice already sliced ReferenceClip.")
        subslice = copy.copy(self)
        subslice._data_arrays = {}
        for key in subslice._DATA_ARRAYS:
            clip_array = self._data_arrays[key][clip]
            slice = jax.lax.dynamic_slice(
                clip_array,
                (start_frame, *jp.zeros(clip_array.ndim - 1, dtype=int)),
                (length, *clip_array.shape[1:]),
            )
            subslice._data_arrays[key] = slice
        return subslice

    def _load_from_disk(self, data_path: str, n_frames_per_clip: int):
        self._data_arrays = {}
        with h5py.File(data_path, "r") as fid:
            self._config = yaml.safe_load(fid["config"][()])
            self.clip_names = self._extract_clip_names(self._config)
            for k in self._DATA_ARRAYS:
                arr = fid[k][()]
                n_clips = arr.shape[0] // n_frames_per_clip
                arr = arr.reshape(n_clips, n_frames_per_clip, *arr.shape[1:])
                # Storing the references as regular jax.arrays encourages
                # const folding durig jitting, which can blow up RAM use during
                # compilation.
                # TODO: Check if jax.device_put(arr) or jax.array_ref(jp.array(arr))
                # blocks const folding.
                self._data_arrays[k] = jp.array(arr)
            self._names_qpos = fid["names_qpos"][()].astype(str)
            self._names_xpos = fid["names_xpos"][()].astype(str)
            self._qpos_names = {n: i for (i, n) in enumerate(self._names_qpos)}
            self._xpos_names = {n: i for (i, n) in enumerate(self._names_xpos)}

    def _extract_clip_names(self, config: Mapping[str, Any]) -> np.ndarray:
        """Each clip has a behavior name, e.g. "Walk", "FastWalk", "LGroom", etc. This
        function extracts these names from the config metadata of the h5 file."""
        original_filenames = config["model"]["snips_order"]
        pattern = r"([A-Za-z]+)_\d+\.p"
        clip_names = []
        for fn in original_filenames:
            m = re.search(pattern, fn)
            if m is None:
                raise ValueError(
                    f"Original name of clip {fn} does not match expected pattern {pattern}."
                )
            clip_names.append(m.group(1))
        return np.array(clip_names)

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
        """First 3 elements of qpos, corresponding to the root position for
           the rodent."""
        return self.qpos[..., :3]

    @property
    def root_quaternion(self) -> jp.ndarray:
        """Elements 3-6 of qpos, corresponding to the root quaterinion for
           the rodent."""
        return self.qpos[..., 3:7]

    @property
    def joints(self) -> jp.ndarray:
        return self.qpos[..., 7:]

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
        """Get the reference for a global euclidean position of a body part.
        Args:
            name (str): Name of the body part.
        """
        return self.xpos[..., self._xpos_names[name], :]

    def body_xquat(self, name: str) -> jp.ndarray:
        """Get the reference for a global orientation of a body part.
        Args:
            name (str): Name of the body part.
        Returns:
            jp.ndarray: The global orientation of the body part as a quaternion.
        """
        return self.xquat[..., self._xpos_names[name], :]
