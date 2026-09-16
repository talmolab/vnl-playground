"""Typed reference-motion data and HDF5 loaders.

Reference files use the native ``stac-mjx`` output contract: ``qpos``, ``qvel``,
``xpos``, ``xquat``, and their associated state names.
"""

import os
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Self

import h5py
import jax
import jax.numpy as jp
import numpy as np
import yaml
from jaxtyping import Array, Float, Integer


@dataclass(frozen=True, eq=False)
class ReferenceClipData:
    """Canonical MuJoCo state arrays with shared leading dimensions."""

    qpos: Float[Array, "*batch nq"]
    qvel: Float[Array, "*batch nv"]
    xpos: Float[Array, "*batch nbody 3"]
    xquat: Float[Array, "*batch nbody 4"]

    def __post_init__(self) -> None:
        """Validate the shared leading dimensions used by all data views."""
        leading = self.qpos.shape[:-1]
        if self.qvel.shape[:-1] != leading:
            raise ValueError("qpos and qvel must have identical leading dimensions.")
        if self.xpos.shape[:-2] != leading or self.xquat.shape[:-2] != leading:
            raise ValueError("All state arrays must have identical leading dimensions.")
        if self.xpos.shape[-2] != self.xquat.shape[-2]:
            raise ValueError("xpos and xquat must contain the same number of bodies.")
        if self.xpos.shape[-1] != 3 or self.xquat.shape[-1] != 4:
            raise ValueError("xpos and xquat must end in XYZ and WXYZ dimensions.")


@dataclass(frozen=True)
class ReferenceClipMetadata:
    """Names and producer metadata associated with reference-motion arrays."""

    qpos_names: tuple[str, ...]
    xpos_names: tuple[str, ...]
    joint_names: tuple[str, ...]
    tracked_body_names: tuple[str, ...]
    joint_qpos_indices: tuple[int, ...]
    joint_qvel_indices: tuple[int, ...]
    root_body_name: str | None = None
    stac_config: Mapping[str, Any] | None = None


@dataclass(frozen=True, eq=False)
class ReferenceClips:
    """Immutable reference clips backed by canonical MuJoCo state arrays."""

    data: ReferenceClipData
    metadata: ReferenceClipMetadata
    data_path: str
    source_clip_indices: Integer[Array, "..."]
    behaviour_labels: tuple[str, ...] | None = None

    def __repr__(self) -> str:
        return (
            "ReferenceClips("
            f"data_path={self.data_path!r}, "
            f"n_clips={self.n_clips}, "
            f"n_frames_per_clip={self.n_frames})"
        )

    def __len__(self) -> int:
        return self.n_clips

    def _require_collection(self) -> None:
        if self.qpos.ndim < 3:
            raise IndexError("Cannot slice an already-sliced ReferenceClips object.")

    def _label_for_clip(self, clip: int | Array) -> tuple[str, ...] | None:
        if self.behaviour_labels is None or isinstance(clip, jax.core.Tracer):
            return None
        return (self.behaviour_labels[int(clip)],)

    def at(self, clip: int | Array, frame: int | Array) -> Self:
        """Return one frame from one clip."""
        self._require_collection()
        data = ReferenceClipData(
            qpos=self.qpos[clip, frame],
            qvel=self.qvel[clip, frame],
            xpos=self.xpos[clip, frame],
            xquat=self.xquat[clip, frame],
        )
        return replace(
            self,
            data=data,
            source_clip_indices=self.source_clip_indices[clip],
            behaviour_labels=self._label_for_clip(clip),
        )

    def slice(
        self,
        clip: int | Array,
        start_frame: int | Array,
        length: int,
        stride: int = 1,
    ) -> Self:
        """Return a possibly strided sequence from one clip."""
        self._require_collection()
        if length <= 0:
            raise ValueError("length must be positive.")
        if stride <= 0:
            raise ValueError("stride must be positive.")

        total_length = (length - 1) * stride + 1

        def slice_array(array: Array) -> Array:
            clip_array = array[clip]
            contiguous = jax.lax.dynamic_slice(
                clip_array,
                (start_frame, *jp.zeros(clip_array.ndim - 1, dtype=int)),
                (total_length, *clip_array.shape[1:]),
            )
            return contiguous[::stride]

        data = ReferenceClipData(
            qpos=slice_array(self.qpos),
            qvel=slice_array(self.qvel),
            xpos=slice_array(self.xpos),
            xquat=slice_array(self.xquat),
        )
        return replace(
            self,
            data=data,
            source_clip_indices=self.source_clip_indices[clip],
            behaviour_labels=self._label_for_clip(clip),
        )

    def split(self, train_ratio: float = 0.8, seed: int = 0) -> tuple[Self, Self]:
        """Split clips reproducibly into train and test collections."""
        self._require_collection()
        if not 0.0 <= train_ratio <= 1.0:
            raise ValueError("train_ratio must be between 0 and 1.")

        n_train = int(self.n_clips * train_ratio)
        if n_train == self.n_clips:
            warnings.warn(
                "train_ratio results in an empty test set; using all clips for both.",
                stacklevel=2,
            )
            return self, self

        indices = np.random.RandomState(seed).permutation(self.n_clips)
        return self._select(indices[:n_train]), self._select(indices[n_train:])

    def recompute_body_poses(
        self,
        mj_model: Any,
        strip_body_suffix: str = "",
        tracked_body_names: Sequence[str] | None = None,
    ) -> Self:
        """Return a copy whose body states are recomputed from ``qpos``."""
        self._require_collection()
        import mujoco

        qpos = np.asarray(self.qpos)
        n_clips, n_frames = qpos.shape[:2]
        xpos = np.empty((n_clips, n_frames, mj_model.nbody, 3), dtype=qpos.dtype)
        xquat = np.empty((n_clips, n_frames, mj_model.nbody, 4), dtype=qpos.dtype)
        mj_data = mujoco.MjData(mj_model)

        for clip_index in range(n_clips):
            for frame_index in range(n_frames):
                mj_data.qpos[:] = qpos[clip_index, frame_index]
                mujoco.mj_kinematics(mj_model, mj_data)
                xpos[clip_index, frame_index] = mj_data.xpos
                xquat[clip_index, frame_index] = mj_data.xquat

        xpos_names = tuple(
            mj_model.body(i).name.removesuffix(strip_body_suffix)
            for i in range(mj_model.nbody)
        )
        tracked_bodies = (
            tuple(tracked_body_names)
            if tracked_body_names is not None
            else self.metadata.tracked_body_names
        )
        missing_bodies = [name for name in tracked_bodies if name not in xpos_names]
        if missing_bodies:
            raise ValueError(
                f"Recomputed model does not contain tracked bodies: {missing_bodies}"
            )

        data = replace(self.data, xpos=jp.asarray(xpos), xquat=jp.asarray(xquat))
        metadata = replace(
            self.metadata, xpos_names=xpos_names, tracked_body_names=tracked_bodies
        )
        return replace(self, data=data, metadata=metadata)

    def bind_model_layout(
        self,
        *,
        joint_names: Sequence[str] | None = None,
        body_names: Sequence[str] | None = None,
        root_body_name: str | None = None,
    ) -> Self:
        """Validate and bind model-facing names to already-loaded arrays."""
        metadata = _build_metadata(
            self.data,
            qpos_names=self.metadata.qpos_names,
            xpos_names=self.metadata.xpos_names,
            stac_config=self.metadata.stac_config,
            joint_names=_normalize_names(joint_names),
            body_names=_normalize_names(body_names),
            root_body_name=root_body_name,
        )
        return replace(self, metadata=metadata)

    def _select(self, indices: Sequence[int] | np.ndarray) -> Self:
        index_array = np.asarray(indices, dtype=int)
        data = ReferenceClipData(
            qpos=self.qpos[index_array],
            qvel=self.qvel[index_array],
            xpos=self.xpos[index_array],
            xquat=self.xquat[index_array],
        )
        labels = (
            tuple(self.behaviour_labels[i] for i in index_array)
            if self.behaviour_labels is not None
            else None
        )
        return replace(
            self,
            data=data,
            source_clip_indices=self.source_clip_indices[index_array],
            behaviour_labels=labels,
        )

    @property
    def qpos(self) -> Float[Array, "*batch nq"]:
        """Generalized positions."""
        return self.data.qpos

    @property
    def qvel(self) -> Float[Array, "*batch nv"]:
        """Generalized velocities."""
        return self.data.qvel

    @property
    def xpos(self) -> Float[Array, "*batch nbody 3"]:
        """World-space body positions."""
        return self.data.xpos

    @property
    def xquat(self) -> Float[Array, "*batch nbody 4"]:
        """World-space body orientations in MuJoCo ``wxyz`` order."""
        return self.data.xquat

    @property
    def root_position(self) -> Float[Array, "*batch 3"]:
        """World-space position of the configured root body."""
        if (
            root_body_name := self.metadata.root_body_name
        ) is not None and root_body_name in self.metadata.xpos_names:
            return self.body_xpos(root_body_name)
        if self.qpos.shape[-1] < 3:
            raise ValueError("Reference data does not contain a root position.")
        return self.qpos[..., :3]

    @property
    def root_quaternion(self) -> Float[Array, "*batch 4"]:
        """World-space orientation of the configured root body."""
        if (
            root_body_name := self.metadata.root_body_name
        ) is not None and root_body_name in self.metadata.xpos_names:
            return self.body_xquat(root_body_name)
        if self.qpos.shape[-1] < 7:
            raise ValueError("Reference data does not contain a root quaternion.")
        return self.qpos[..., 3:7]

    @property
    def joints(self) -> Float[Array, "*batch njoint"]:
        """Generalized positions selected as model joints."""
        return self.qpos[..., jp.asarray(self.metadata.joint_qpos_indices)]

    @property
    def joints_velocity(self) -> Float[Array, "*batch njoint"]:
        """Generalized velocities selected as model joints."""
        return self.qvel[..., jp.asarray(self.metadata.joint_qvel_indices)]

    @property
    def velocity(self) -> Float[Array, "*batch 3"]:
        """Translational free-root velocity."""
        self._require_free_root_velocity()
        return self.qvel[..., :3]

    @property
    def angular_velocity(self) -> Float[Array, "*batch 3"]:
        """Angular free-root velocity."""
        self._require_free_root_velocity()
        return self.qvel[..., 3:6]

    @property
    def joint_names(self) -> list[str]:
        """Selected model joint names."""
        return list(self.metadata.joint_names)

    @property
    def body_names(self) -> list[str]:
        """Tracked model body names."""
        return list(self.metadata.tracked_body_names)

    @property
    def clip_indices(self) -> Integer[Array, "..."]:
        """Indices of the selected clips in the source dataset."""
        return self.source_clip_indices

    @property
    def n_frames(self) -> int:
        """Number of frames represented by this view."""
        if self.qpos.ndim >= 3:
            return self.qpos.shape[1]
        if self.qpos.ndim == 2:
            return self.qpos.shape[0]
        return 1

    @property
    def n_clips(self) -> int:
        """Number of clips represented by this view."""
        return self.qpos.shape[0] if self.qpos.ndim >= 3 else 1

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the generalized-position array."""
        return self.qpos.shape

    @property
    def is_sliced(self) -> bool:
        """Whether this view represents fewer than two leading clip axes."""
        return self.qpos.ndim < 3

    @property
    def joint_indices(self) -> list[int]:
        """Generalized-position indices for selected joints."""
        return list(self.metadata.joint_qpos_indices)

    @property
    def stac_config(self) -> Mapping[str, Any] | None:
        """Configuration embedded by ``stac-mjx``, when present."""
        return self.metadata.stac_config

    @property
    def scale_factor(self) -> float | None:
        """Model scale factor recorded by ``stac-mjx``, when present."""
        if self.stac_config is None:
            return None
        model_config = self.stac_config.get("model")
        if not isinstance(model_config, Mapping):
            return None
        value = model_config.get("SCALE_FACTOR")
        return float(value) if value is not None else None

    def body_xpos(self, name: str) -> Float[Array, "*batch 3"]:
        """Return the world-space position of a named body."""
        return self.xpos[..., self._body_index(name), :]

    def body_xquat(self, name: str) -> Float[Array, "*batch 4"]:
        """Return the world-space orientation of a named body."""
        return self.xquat[..., self._body_index(name), :]

    def _body_index(self, name: str | None) -> int:
        if name is not None:
            try:
                return self.metadata.xpos_names.index(name)
            except ValueError:
                pass
        raise KeyError(
            f"Body {name!r} not found. Available bodies: "
            f"{list(self.metadata.xpos_names)}"
        )

    def _require_free_root_velocity(self) -> None:
        if self.qpos.shape[-1] != self.qvel.shape[-1] + 1 or self.qvel.shape[-1] < 6:
            raise ValueError("Reference data does not use a MuJoCo free-root layout.")


def load_reference_clips(
    data_path: str | os.PathLike[str],
    *,
    n_frames_per_clip: int | None = None,
    clip_indices: Sequence[int] | np.ndarray | Array | None = None,
    joint_names: Sequence[str] | None = None,
    body_names: Sequence[str] | None = None,
    root_body_name: str | None = None,
) -> ReferenceClips:
    """Load reference motion into the canonical STAC/MuJoCo representation."""
    if n_frames_per_clip is not None and n_frames_per_clip <= 0:
        raise ValueError("n_frames_per_clip must be positive.")
    if (normalized_indices := _normalize_indices(clip_indices)) is not None and any(
        i < 0 for i in normalized_indices
    ):
        raise ValueError("clip_indices cannot contain negative indices.")
    normalized_joint_names = _normalize_names(joint_names)
    normalized_body_names = _normalize_names(body_names)
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(path)
    clips = (
        _load_reference_directory(path, n_frames_per_clip)
        if path.is_dir()
        else _load_reference_file(path, n_frames_per_clip)
    )

    if normalized_indices is not None:
        selected = np.asarray(normalized_indices, dtype=int)
        if selected.size and int(selected.max()) >= clips.n_clips:
            raise IndexError(
                f"clip_indices contains an index outside [0, {clips.n_clips})."
            )
        clips = clips._select(selected)
    return clips.bind_model_layout(
        joint_names=normalized_joint_names,
        body_names=normalized_body_names,
        root_body_name=root_body_name,
    )


def prepare_reference_clips(
    config: Mapping[str, Any],
    clips: ReferenceClips | None,
    *,
    joint_names: Sequence[str] | None = None,
    body_names: Sequence[str] | None = None,
    root_body_name: str | None = None,
) -> ReferenceClips:
    """Load missing clips and bind them to an environment's model layout."""
    if clips is None:
        return load_reference_clips(
            config["reference_data_path"],
            n_frames_per_clip=config["clip_length"],
            clip_indices=config.get("clip_indices"),
            joint_names=joint_names,
            body_names=body_names,
            root_body_name=root_body_name,
        )

    if joint_names is None and body_names is None and root_body_name is None:
        return clips
    return clips.bind_model_layout(
        joint_names=joint_names,
        body_names=body_names,
        root_body_name=root_body_name,
    )


def _load_reference_file(path: Path, n_frames_per_clip: int | None) -> ReferenceClips:
    with h5py.File(path, "r") as h5:
        required = ("qpos", "xpos", "xquat", "names_qpos", "names_xpos")
        _require_h5_keys(h5, required)
        if "qvel" not in h5 or h5["qvel"].size == 0:
            raise ValueError(
                f"{path}: reference data has no inferred qvel values; "
                "rerun stac-mjx with stac.infer_qvels=true."
            )
        arrays = {name: h5[name][()] for name in ("qpos", "qvel", "xpos", "xquat")}
        qpos_names = _decode_names(h5["names_qpos"][()])
        xpos_names = _decode_names(h5["names_xpos"][()])
        stac_config = _load_config(h5)

    data = _canonical_data(**arrays, n_frames_per_clip=n_frames_per_clip)
    if len(qpos_names) != data.qpos.shape[-1]:
        raise ValueError(
            f"{path}: names_qpos has {len(qpos_names)} entries; "
            f"qpos has {data.qpos.shape[-1]} columns."
        )
    if len(xpos_names) != data.xpos.shape[-2]:
        raise ValueError(
            f"{path}: names_xpos has {len(xpos_names)} entries; "
            f"xpos has {data.xpos.shape[-2]} bodies."
        )

    labels = _extract_behaviour_labels(stac_config, data.qpos.shape[0])
    return _make_reference_clips(
        path,
        data,
        qpos_names=qpos_names,
        xpos_names=xpos_names,
        stac_config=stac_config,
        behaviour_labels=labels,
    )


def _load_reference_directory(
    path: Path, n_frames_per_clip: int | None
) -> ReferenceClips:
    h5_paths = sorted(path.glob("*.h5"))
    if not h5_paths:
        raise ValueError(f"No HDF5 files found in {path}.")

    clips: list[ReferenceClips] = []
    target_frames = n_frames_per_clip
    for h5_path in h5_paths:
        clip = _load_reference_file(h5_path, None)
        if clip.data.qpos.shape[0] != 1:
            raise ValueError(f"Expected one clip per file in {h5_path}.")
        native_frames = clip.data.qpos.shape[1]
        if target_frames is None:
            target_frames = native_frames
        if native_frames < target_frames:
            raise ValueError(
                f"{h5_path} has {native_frames} frames; expected at least {target_frames}."
            )
        clips.append(clip)

    first = clips[0]
    for clip, h5_path in zip(clips[1:], h5_paths[1:]):
        if (
            clip.metadata.qpos_names != first.metadata.qpos_names
            or clip.metadata.xpos_names != first.metadata.xpos_names
        ):
            raise ValueError(f"State names in {h5_path} do not match {h5_paths[0]}.")

    assert target_frames is not None
    data = ReferenceClipData(
        qpos=jp.concatenate([clip.data.qpos[:, :target_frames] for clip in clips]),
        qvel=jp.concatenate([clip.data.qvel[:, :target_frames] for clip in clips]),
        xpos=jp.concatenate([clip.data.xpos[:, :target_frames] for clip in clips]),
        xquat=jp.concatenate([clip.data.xquat[:, :target_frames] for clip in clips]),
    )
    labels = tuple(re.sub(r"_ik$", "", h5_path.stem) for h5_path in h5_paths)
    return _make_reference_clips(
        path,
        data,
        qpos_names=first.metadata.qpos_names,
        xpos_names=first.metadata.xpos_names,
        stac_config=first.metadata.stac_config,
        behaviour_labels=labels,
    )


def _make_reference_clips(
    path: Path,
    data: ReferenceClipData,
    *,
    qpos_names: tuple[str, ...],
    xpos_names: tuple[str, ...],
    stac_config: Mapping[str, Any] | None = None,
    behaviour_labels: tuple[str, ...] | None = None,
) -> ReferenceClips:
    metadata = _build_metadata(
        data,
        qpos_names=qpos_names,
        xpos_names=xpos_names,
        stac_config=stac_config,
    )
    return ReferenceClips(
        data=data,
        metadata=metadata,
        data_path=str(path),
        source_clip_indices=jp.arange(data.qpos.shape[0]),
        behaviour_labels=behaviour_labels,
    )


def _build_metadata(
    data: ReferenceClipData,
    *,
    qpos_names: tuple[str, ...],
    xpos_names: tuple[str, ...],
    stac_config: Mapping[str, Any] | None,
    joint_names: tuple[str, ...] | None = None,
    body_names: tuple[str, ...] | None = None,
    root_body_name: str | None = None,
) -> ReferenceClipMetadata:
    requested_joints = joint_names
    if requested_joints is None:
        has_free_root = data.qpos.shape[-1] == data.qvel.shape[-1] + 1
        root_qpos_dims = 7 if has_free_root else 0
        root_qvel_dims = 6 if has_free_root else 0
        joint_qpos_indices = tuple(range(root_qpos_dims, data.qpos.shape[-1]))
        joint_qvel_indices = tuple(range(root_qvel_dims, data.qvel.shape[-1]))
        requested_joints = tuple(qpos_names[i] for i in joint_qpos_indices)
    else:
        joint_qpos_indices = _resolve_unique_names(
            requested_joints, qpos_names, "joint"
        )
        if data.qpos.shape[-1] == data.qvel.shape[-1]:
            joint_qvel_indices = joint_qpos_indices
        elif data.qpos.shape[-1] == data.qvel.shape[-1] + 1 and all(
            index >= 7 for index in joint_qpos_indices
        ):
            joint_qvel_indices = tuple(index - 1 for index in joint_qpos_indices)
        else:
            raise ValueError(
                "Cannot infer qvel indices from this qpos/qvel layout. Current "
                "stac-mjx metadata supports fixed roots or one leading free joint."
            )

    tracked_bodies = body_names or xpos_names
    missing_bodies = [name for name in tracked_bodies if name not in xpos_names]
    if missing_bodies:
        raise ValueError(f"Bodies not found in reference data: {missing_bodies}")

    return ReferenceClipMetadata(
        qpos_names=qpos_names,
        xpos_names=xpos_names,
        joint_names=requested_joints,
        tracked_body_names=tracked_bodies,
        joint_qpos_indices=joint_qpos_indices,
        joint_qvel_indices=joint_qvel_indices,
        root_body_name=root_body_name,
        stac_config=stac_config,
    )


def _canonical_data(
    qpos: np.ndarray,
    qvel: np.ndarray,
    xpos: np.ndarray,
    xquat: np.ndarray,
    *,
    n_frames_per_clip: int | None,
) -> ReferenceClipData:
    arrays = {
        "qpos": np.asarray(qpos),
        "qvel": np.asarray(qvel),
        "xpos": np.asarray(xpos),
        "xquat": np.asarray(xquat),
    }
    if arrays["qpos"].ndim == 2:
        qpos_frames = arrays["qpos"].shape[0]
        frames_per_clip = n_frames_per_clip or qpos_frames
        if qpos_frames % frames_per_clip:
            raise ValueError(
                f"{qpos_frames} frames is not divisible by "
                f"n_frames_per_clip={frames_per_clip}."
            )
        n_clips = qpos_frames // frames_per_clip
        for name, array in arrays.items():
            if array.shape[0] != qpos_frames:
                raise ValueError(
                    f"{name} has {array.shape[0]} frames; expected {qpos_frames}."
                )
            arrays[name] = array.reshape(n_clips, frames_per_clip, *array.shape[1:])

    leading = arrays["qpos"].shape[:-1]
    expected_ranks = {"qpos": 3, "qvel": 3, "xpos": 4, "xquat": 4}
    for name, expected_rank in expected_ranks.items():
        if arrays[name].ndim != expected_rank:
            raise ValueError(
                f"{name} must have rank {expected_rank}, got {arrays[name].shape}."
            )
    if arrays["qvel"].shape[:-1] != leading:
        raise ValueError("qpos and qvel must have identical clip/frame axes.")
    if arrays["xpos"].shape[:-2] != leading or arrays["xquat"].shape[:-2] != leading:
        raise ValueError("All state arrays must have identical clip/frame axes.")
    if arrays["xpos"].shape[-2] != arrays["xquat"].shape[-2]:
        raise ValueError("xpos and xquat must contain the same number of bodies.")
    if arrays["xpos"].shape[-1] != 3 or arrays["xquat"].shape[-1] != 4:
        raise ValueError("xpos and xquat must end in XYZ and WXYZ dimensions.")
    if n_frames_per_clip is not None and arrays["qpos"].shape[1] != n_frames_per_clip:
        raise ValueError(
            f"Reference data contains {arrays['qpos'].shape[1]} frames per clip; "
            f"expected {n_frames_per_clip}."
        )
    return ReferenceClipData(
        **{name: jp.asarray(array) for name, array in arrays.items()}
    )


def _require_h5_keys(group: h5py.Group, required: Sequence[str]) -> None:
    missing = [name for name in required if name not in group]
    if missing:
        present = list(group.keys())
        raise ValueError(
            f"Invalid reference data: missing {missing}; present keys: {present}."
        )


def _decode_names(values: np.ndarray) -> tuple[str, ...]:
    return tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    )


def _load_config(h5: h5py.File) -> Mapping[str, Any] | None:
    if "config" not in h5:
        return None
    raw = h5["config"][()]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if (config := yaml.safe_load(raw)) is None:
        return None
    if not isinstance(config, Mapping):
        raise TypeError("Reference config must decode to a mapping.")
    return config


def _extract_behaviour_labels(
    config: Mapping[str, Any] | None, n_clips: int
) -> tuple[str, ...] | None:
    if config is None:
        return None
    if not isinstance(model_config := config.get("model"), Mapping):
        return None
    if (filenames := model_config.get("snips_order")) is None:
        return None
    if len(filenames) != n_clips:
        raise ValueError(
            f"config.model.snips_order has {len(filenames)} entries; expected {n_clips}."
        )
    labels = []
    for filename in filenames:
        stem = Path(str(filename)).stem
        match = re.fullmatch(r"(.+?)_\d+", stem)
        labels.append(match.group(1) if match else stem)
    return tuple(labels)


def _resolve_unique_names(
    requested: tuple[str, ...], available: tuple[str, ...], kind: str
) -> tuple[int, ...]:
    indices = []
    for name in requested:
        matches = [i for i, candidate in enumerate(available) if candidate == name]
        if not matches:
            raise ValueError(f"{kind.title()} {name!r} not found in reference data.")
        if len(matches) > 1:
            raise ValueError(
                f"{kind.title()} {name!r} maps to multiple state dimensions: {matches}."
            )
        indices.append(matches[0])
    return tuple(indices)


def _normalize_indices(
    values: Sequence[int] | np.ndarray | Array | None,
) -> tuple[int, ...] | None:
    if values is None:
        return None
    return tuple(int(value) for value in np.asarray(values).tolist())


def _normalize_names(values: Sequence[str] | None) -> tuple[str, ...] | None:
    if values is None:
        return None
    return tuple(str(value) for value in values)
