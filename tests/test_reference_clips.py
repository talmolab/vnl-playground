"""Tests for the canonical reference-motion data contract."""

import importlib.util
import inspect
from pathlib import Path

import h5py
import jax
import jax.numpy as jp
import numpy as np
import pytest

MODULE_PATH = Path(__file__).parents[1] / "vnl_playground/tasks/reference_clips.py"
SPEC = importlib.util.spec_from_file_location("reference_clips_under_test", MODULE_PATH)
reference_clips = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(reference_clips)


def test_loader_has_one_canonical_format() -> None:
    parameters = inspect.signature(reference_clips.load_reference_clips).parameters

    assert "data_format" not in parameters


def test_data_rejects_inconsistent_leading_axes() -> None:
    with pytest.raises(ValueError, match="identical leading dimensions"):
        reference_clips.ReferenceClipData(
            qpos=jp.zeros((2, 3, 4)),
            qvel=jp.zeros((2, 4, 4)),
            xpos=jp.zeros((2, 3, 5, 3)),
            xquat=jp.zeros((2, 3, 5, 4)),
        )


def _write_reference(
    path: Path,
    *,
    n_clips: int = 2,
    n_frames: int = 3,
    fixed_root: bool = False,
) -> None:
    n_qpos = 2 if fixed_root else 9
    n_qvel = 2 if fixed_root else 8
    total_frames = n_clips * n_frames
    qpos = np.arange(total_frames * n_qpos, dtype=np.float32).reshape(
        total_frames, n_qpos
    )
    qvel = np.arange(total_frames * n_qvel, dtype=np.float32).reshape(
        total_frames, n_qvel
    )
    xpos = np.zeros((total_frames, 3, 3), dtype=np.float32)
    xpos[:, 1] = np.array([5.0, 6.0, 7.0])
    xquat = np.zeros((total_frames, 3, 4), dtype=np.float32)
    xquat[..., 0] = 1.0
    qpos_names = (
        ["joint_a", "joint_b"] if fixed_root else ["root"] * 7 + ["joint_a", "joint_b"]
    )

    with h5py.File(path, "w") as h5:
        h5.create_dataset("qpos", data=qpos)
        h5.create_dataset("qvel", data=qvel)
        h5.create_dataset("xpos", data=xpos)
        h5.create_dataset("xquat", data=xquat)
        h5.create_dataset("names_qpos", data=np.asarray(qpos_names, dtype="S"))
        h5.create_dataset(
            "names_xpos", data=np.asarray(["world", "root", "limb"], dtype="S")
        )
        labels = "\n".join(f"    - Walk_{i}.p" for i in range(n_clips))
        h5.create_dataset(
            "config",
            data=np.bytes_(f"model:\n  SCALE_FACTOR: 1.0\n  snips_order:\n{labels}\n"),
        )


def _write_fly_reference(path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    n_clips, n_frames, n_joints, n_bodies = 2, 3, 36, 68
    leading = (n_clips * n_frames,)
    joint_names = tuple(f"joint_{i}" for i in range(n_joints))
    body_names = ("world", "thorax", *(f"body_{i}" for i in range(n_bodies - 2)))

    qpos = np.zeros((*leading, 7 + n_joints), dtype=np.float32)
    qpos[:, :3] = 1.0
    qpos[:, 3] = 1.0
    qpos[:, 7:] = 4.0
    qvel = np.zeros((*leading, 6 + n_joints), dtype=np.float32)
    qvel[:, 3:6] = 3.0
    qvel[:, 6:] = 5.0
    xpos = np.full((*leading, n_bodies, 3), 6.0, dtype=np.float32)
    xquat = np.zeros((*leading, n_bodies, 4), dtype=np.float32)
    xquat[..., 0] = 1.0

    with h5py.File(path, "w") as h5:
        h5.create_dataset("qpos", data=qpos)
        h5.create_dataset("qvel", data=qvel)
        h5.create_dataset("xpos", data=xpos)
        h5.create_dataset("xquat", data=xquat)
        h5.create_dataset(
            "names_qpos", data=np.asarray(["root"] * 7 + list(joint_names), dtype="S")
        )
        h5.create_dataset("names_xpos", data=np.asarray(body_names, dtype="S"))

    return joint_names, body_names


def test_reference_file_preserves_semantics(tmp_path: Path) -> None:
    path = tmp_path / "reference.h5"
    _write_reference(path)

    clips = reference_clips.load_reference_clips(
        path,
        n_frames_per_clip=3,
        joint_names=("joint_a", "joint_b"),
        body_names=("limb",),
        root_body_name="root",
    )

    assert clips.qpos.shape == (2, 3, 9)
    assert clips.qvel.shape == (2, 3, 8)
    assert clips.joints.shape == (2, 3, 2)
    assert clips.joints_velocity.shape == (2, 3, 2)
    np.testing.assert_allclose(
        clips.root_position,
        np.broadcast_to([5.0, 6.0, 7.0], clips.root_position.shape),
    )
    np.testing.assert_allclose(clips.angular_velocity, clips.qvel[..., 3:6])
    assert clips.joint_indices == [7, 8]
    assert clips.body_names == ["limb"]
    assert clips.behaviour_labels == ("Walk", "Walk")


def test_selection_slicing_and_split_preserve_source_indices(tmp_path: Path) -> None:
    path = tmp_path / "reference.h5"
    _write_reference(path, n_clips=3, n_frames=4)
    clips = reference_clips.load_reference_clips(
        path, n_frames_per_clip=4, clip_indices=(2, 0)
    )

    assert clips.clip_indices.tolist() == [2, 0]
    assert clips.at(1, 2).clip_indices.item() == 0
    assert clips.slice(0, 0, 2, stride=2).qpos.shape == (2, 9)
    train, test = clips.split(train_ratio=0.5, seed=0)
    assert sorted(train.clip_indices.tolist() + test.clip_indices.tolist()) == [0, 2]


def test_numpy_clip_indices_are_supported(tmp_path: Path) -> None:
    path = tmp_path / "reference.h5"
    _write_reference(path, n_clips=3)

    clips = reference_clips.load_reference_clips(
        path,
        n_frames_per_clip=3,
        clip_indices=np.asarray([2, 0]),
    )

    assert clips.clip_indices.tolist() == [2, 0]


def test_typed_views_accept_traced_indices(tmp_path: Path) -> None:
    path = tmp_path / "reference.h5"
    _write_reference(path)
    clips = reference_clips.load_reference_clips(path, n_frames_per_clip=3)

    get_joints = jax.jit(lambda clip, frame: clips.at(clip, frame).joints)
    get_sequence = jax.jit(lambda clip, frame: clips.slice(clip, frame, 2).joints)

    assert get_joints(jp.asarray(0), jp.asarray(0)).shape == (2,)
    assert get_sequence(jp.asarray(0), jp.asarray(0)).shape == (2, 2)


def test_fly_reference_uses_the_canonical_loader(tmp_path: Path) -> None:
    path = tmp_path / "fly_reference.h5"
    joint_names, body_names = _write_fly_reference(path)
    config = {
        "reference_data_path": path,
        "clip_length": 3,
        "clip_indices": None,
    }

    clips = reference_clips.prepare_reference_clips(
        config,
        None,
        joint_names=joint_names,
        body_names=body_names[1:],
        root_body_name="thorax",
    )

    assert clips.qpos.shape == (2, 3, 43)
    assert clips.qvel.shape == (2, 3, 42)
    assert clips.xpos.shape == (2, 3, 68, 3)
    assert clips.joints.shape == (2, 3, 36)
    assert clips.joint_names == list(joint_names)
    assert clips.body_names == list(body_names[1:])
    np.testing.assert_allclose(clips.root_position, 6.0)
    np.testing.assert_allclose(clips.angular_velocity, 3.0)
    np.testing.assert_allclose(clips.joints, 4.0)
    np.testing.assert_allclose(clips.joints_velocity, 5.0)


def test_prepare_reference_clips_loads_canonical_file(tmp_path: Path) -> None:
    path = tmp_path / "reference.h5"
    _write_reference(path)
    config = {
        "reference_data_path": path,
        "clip_length": 3,
        "clip_indices": None,
    }

    clips = reference_clips.prepare_reference_clips(
        config,
        None,
        joint_names=("joint_a", "joint_b"),
        body_names=("limb",),
        root_body_name="root",
    )

    assert clips.qpos.shape == (2, 3, 9)


def test_fixed_root_uses_matching_qpos_and_qvel_indices(tmp_path: Path) -> None:
    path = tmp_path / "fixed.h5"
    _write_reference(path, fixed_root=True)
    clips = reference_clips.load_reference_clips(
        path,
        n_frames_per_clip=3,
        joint_names=("joint_b",),
        body_names=("limb",),
    )

    assert clips.joint_indices == [1]
    np.testing.assert_allclose(clips.joints, clips.qpos[..., 1:2])
    np.testing.assert_allclose(clips.joints_velocity, clips.qvel[..., 1:2])
    with pytest.raises(ValueError, match="free-root"):
        _ = clips.angular_velocity


def test_preloaded_clips_can_be_bound_to_an_environment_layout(tmp_path: Path) -> None:
    path = tmp_path / "reference.h5"
    _write_reference(path)
    clips = reference_clips.load_reference_clips(path, n_frames_per_clip=3)

    bound = clips.bind_model_layout(
        joint_names=("joint_b",),
        body_names=("limb",),
        root_body_name="root",
    )

    assert bound.joint_names == ["joint_b"]
    assert bound.body_names == ["limb"]
    assert bound.joint_indices == [8]
    np.testing.assert_allclose(bound.joints_velocity, bound.qvel[..., 7:8])


def test_invalid_frame_partition_fails_early(tmp_path: Path) -> None:
    path = tmp_path / "reference.h5"
    _write_reference(path, n_clips=1, n_frames=5)

    with pytest.raises(ValueError, match="not divisible"):
        reference_clips.load_reference_clips(path, n_frames_per_clip=3)


def test_missing_inferred_velocities_has_actionable_error(tmp_path: Path) -> None:
    path = tmp_path / "reference.h5"
    _write_reference(path)
    with h5py.File(path, "a") as h5:
        del h5["qvel"]
        h5.create_dataset("qvel", data=np.asarray([], dtype=np.float32))

    with pytest.raises(ValueError, match=r"stac\.infer_qvels=true"):
        reference_clips.load_reference_clips(path, n_frames_per_clip=3)


def test_reference_directory_stacks_one_file_per_clip(tmp_path: Path) -> None:
    directory = tmp_path / "clips"
    directory.mkdir()
    _write_reference(directory / "reach_ik.h5", n_clips=1, n_frames=4, fixed_root=True)
    _write_reference(directory / "walk_ik.h5", n_clips=1, n_frames=4, fixed_root=True)

    clips = reference_clips.load_reference_clips(directory, n_frames_per_clip=3)

    assert clips.qpos.shape == (2, 3, 2)
    assert clips.behaviour_labels == ("reach", "walk")
    assert clips.at(1, 2).behaviour_labels == ("walk",)
    assert clips.slice(0, 0, 2).behaviour_labels == ("reach",)
