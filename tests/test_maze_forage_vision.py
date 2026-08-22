"""Tests for the vision-guided sparse maze-foraging task.

Covers DESIGN.md section 4 items 2-8:

2. the env builds / compiles and the model counts match the configuration,
3. ``reset()`` works under ``jax.jit`` **and** ``jax.vmap`` and genuinely
   randomises the spawn and every treat across a batch of rngs,
4. proprioception slicing survives the qpos shift the treat slide joints
   introduce (the highest-risk silent-corruption path in the task),
5. ``step()`` runs and the obs tree matches the declared sizes,
6. the pure-vision contract: ``task_obs`` carries no treat information,
7. the ``collected`` bitmask does not ratchet across an auto-reset boundary,
8. a vision-render smoke test (GPU only),

plus section 9, the regressions from the 2026-08-21 adversarial review: the
frozen-layout / ratcheting-``info`` pair that ``full_reset=False`` causes and
the entry-point guard against it, the poisoned mutable default config, the
corridor width vs ``cell_size`` mismatch, the one-frame ghost of a collected
treat, and the two observation-contract claims (``origin``,
``privileged_state``),

plus section 10, the 2026-08-21 resize: the arena is a fixed 6.46 m x 6.46 m
square for every ``maze_cells``, ``cell_size`` is derived from it, and
``task_obs`` no longer carries the allocentric ``origin`` fix by default.

These tests touch the GPU (``mujoco_impl="warp"`` is mandatory for this env),
so run them serially::

    python -m pytest tests/test_maze_forage_vision.py -x -q
"""

import collections
import pathlib

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import pytest
from mujoco import mjx

from vnl_playground.tasks.rodent import maze_utils
from vnl_playground.tasks.rodent.maze_forage_vision import (
    INFO_RESET_KEYS,
    MazeForageVision,
    default_config,
)

# --- Declared task_obs components ------------------------------------------
# task_obs = [prev_action, kinematic_sensors, touch_sensors] (+ origin iff
# config.include_origin, which is OFF by default since the 2026-08-21 resize).
N_KINEMATIC_SENSORS = 9  # accelerometer(3) + velocimeter(3) + gyro(3)
N_TOUCH_SENSORS = 4  # consts.TOUCH_SENSORS
N_ORIGIN = 3  # origin in the torso frame

# The arena is this many metres across, for every maze_cells (config
# `maze_extent`; `cell_size` is derived as maze_extent / (2*maze_cells + 1)).
MAZE_EXTENT_M = 6.46

# Maze sizes the renders in the next phase will choose between.  All three must
# build, be fully connected, and admit the rat.
DEFAULT_MAZE_CELLS = 10
PARAMETERISED_MAZE_CELLS = (8, 10, 12)
_DEFAULT_GRID = 2 * DEFAULT_MAZE_CELLS + 1

_HAS_GPU = any(d.platform == "gpu" for d in jax.devices())


def _build_env(**overrides) -> MazeForageVision:
    """Builds a MazeForageVision with ``default_config()`` plus overrides."""
    cfg = default_config()
    for key, value in overrides.items():
        cfg[key] = value
    return MazeForageVision(config=cfg)


@pytest.fixture(scope="module")
def env() -> MazeForageVision:
    """Default-config env, shared by the whole module (construction is slow)."""
    return _build_env()


@pytest.fixture(scope="module")
def small_env() -> MazeForageVision:
    """Control env with a different treat count, for delta assertions."""
    return _build_env(n_treats=2)


@pytest.fixture(scope="module")
def origin_env() -> MazeForageVision:
    """Ablation env with the allocentric ``origin`` fix switched back on."""
    return _build_env(include_origin=True)


def _joint_names(model: mujoco.MjModel) -> list:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]


def _body_names(model: mujoco.MjModel) -> list:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(model.nbody)
    ]


def _geom_names(model: mujoco.MjModel) -> list:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        for i in range(model.ngeom)
    ]


def _treat_xy(env: MazeForageVision, data) -> np.ndarray:
    """World xy of every treat body, from ``data.xpos``."""
    return np.asarray(data.xpos[..., np.asarray(env._treat_body_ids), :2])


def _spawn_xy(env: MazeForageVision, data) -> np.ndarray:
    """World xy written into the rodent root free joint."""
    root = env._rodent_root_qpos
    return np.asarray(data.qpos[..., root : root + 2])


# ===========================================================================
# 2. Env builds, mj_model compiles, counts match the configuration
# ===========================================================================


def test_env_builds_and_model_compiles(env):
    assert isinstance(env.mj_model, mujoco.MjModel)
    assert env.mjx_model is not None
    assert env.action_size == env.mj_model.nu > 0


def test_requires_warp_backend():
    with pytest.raises(ValueError, match="warp"):
        _build_env(mujoco_impl="jax")


def test_maze_grid_matches_configured_size(env):
    n_cells = int(env._config.maze_cells)
    assert n_cells == DEFAULT_MAZE_CELLS, (
        "default maze_cells changed; update the sizing tests"
    )
    expected = 2 * n_cells + 1
    assert env.maze_grid.shape == (expected, expected)
    # Fixed maze: rebuilding with the same seed reproduces it exactly.
    again = maze_utils.generate_maze(
        maze_cells=n_cells,
        seed=int(env._config.maze_seed),
        loop_fraction=float(env._config.maze_loop_fraction),
    )
    np.testing.assert_array_equal(env.maze_grid, again)


def test_treat_bodies_joints_and_geoms_exist(env):
    model = env.mj_model
    n = env.n_treats
    bodies = set(_body_names(model))
    geoms = set(_geom_names(model))
    joints = _joint_names(model)

    for i in range(n):
        assert f"treat_{i}" in bodies
        assert f"treat_{i}_geom" in geoms
        for axis in ("x", "y", "z"):
            name = f"treat_{i}_slide_{axis}"
            assert name in joints
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE
            assert model.jnt_stiffness[jid] == 0.0
            dof = model.jnt_dofadr[jid]
            # damping=1e8 is what pins the treat in place (run_gap pattern).
            assert model.dof_damping[dof] == pytest.approx(1e8)

        # Treats are non-colliding trigger volumes.
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"treat_{i}_geom")
        assert model.geom_contype[gid] == 0
        assert model.geom_conaffinity[gid] == 0


def test_wall_geom_count_matches_covering(env):
    wall_geoms = [g for g in _geom_names(env.mj_model) if g.startswith("maze_wall_")]
    assert len(wall_geoms) == len(env.maze_walls)
    # The greedy covering is what keeps the geom count small: far fewer boxes
    # than there are wall cells.
    n_wall_cells = int(np.sum(env.maze_grid == maze_utils.WALL_CHAR))
    assert 0 < len(env.maze_walls) < n_wall_cells


def test_joint_and_qpos_counts_match_n_treats(env):
    model = env.mj_model
    n = env.n_treats
    jnt_type = model.jnt_type
    n_slide = int(np.sum(jnt_type == mujoco.mjtJoint.mjJNT_SLIDE))
    n_free = int(np.sum(jnt_type == mujoco.mjtJoint.mjJNT_FREE))
    n_hinge = int(np.sum(jnt_type == mujoco.mjtJoint.mjJNT_HINGE))

    assert n_slide == 3 * n
    assert n_free == 1  # the rodent root
    assert model.njnt == 3 * n + 1 + n_hinge
    assert model.nq == 3 * n + 7 + n_hinge
    assert model.nv == 3 * n + 6 + n_hinge


def test_model_counts_scale_with_n_treats(env, small_env):
    """Every treat costs exactly 1 body, 1 geom, 3 joints, 3 qpos, 3 qvel."""
    d_treats = env.n_treats - small_env.n_treats
    assert d_treats > 0
    assert env.mj_model.nbody - small_env.mj_model.nbody == d_treats
    assert env.mj_model.ngeom - small_env.mj_model.ngeom == d_treats
    assert env.mj_model.njnt - small_env.mj_model.njnt == 3 * d_treats
    assert env.mj_model.nq - small_env.mj_model.nq == 3 * d_treats
    assert env.mj_model.nv - small_env.mj_model.nv == 3 * d_treats
    # Same fixed maze in both -> same wall geoms.
    assert len(env.maze_walls) == len(small_env.maze_walls)


def test_treat_slide_joints_precede_the_rodent(env):
    """Treats must occupy the LOW qpos addresses, ahead of the rodent root."""
    idxs = np.asarray(env._treat_slide_qpos_idxs_np).ravel()
    np.testing.assert_array_equal(np.sort(idxs), np.arange(3 * env.n_treats))
    assert env._rodent_root_qpos == 3 * env.n_treats


def test_construction_rejects_too_many_treats():
    """More treats than free cells must raise, not silently overlap."""
    with pytest.raises(ValueError, match="free cells"):
        _build_env(maze_cells=1, n_treats=64)


# ===========================================================================
# 3. reset() under jit and vmap; spawn + treats genuinely differ
# ===========================================================================


def test_reset_under_jit(env):
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert state.reward.shape == ()
    assert state.done.shape == ()
    assert not np.isnan(np.asarray(state.reward))
    assert set(state.obs.keys()) == {"state", "privileged_state"}


def test_reset_is_deterministic_for_a_given_rng(env):
    reset = jax.jit(env.reset)
    a = reset(jax.random.PRNGKey(11))
    b = reset(jax.random.PRNGKey(11))
    np.testing.assert_array_equal(np.asarray(a.data.qpos), np.asarray(b.data.qpos))


def test_vmap_reset_spawns_and_treats_differ(env):
    """The whole point of the design: per-episode layout novelty in Data.

    The legacy implementation silently failed exactly here (Model-side
    per-world edits are ignored on the warp backend), so assert it loudly.
    """
    batch = 8
    rngs = jax.random.split(jax.random.PRNGKey(0), batch)
    states = jax.jit(jax.vmap(env.reset))(rngs)

    spawn = _spawn_xy(env, states.data)
    assert spawn.shape == (batch, 2)
    n_unique_spawn = np.unique(spawn, axis=0).shape[0]
    assert n_unique_spawn >= 4, f"only {n_unique_spawn}/{batch} distinct spawns"

    root = env._rodent_root_qpos
    yaw_quat = np.asarray(states.data.qpos[:, root + 3 : root + 7])
    assert np.unique(yaw_quat, axis=0).shape[0] == batch, "spawn yaw is not random"

    treats = _treat_xy(env, states.data)
    assert treats.shape == (batch, env.n_treats, 2)
    layouts = np.unique(np.sort(treats.reshape(batch, -1), axis=1), axis=0)
    assert layouts.shape[0] == batch, "treat layouts repeat across the batch"

    # Treat z is at the live height in every world (nothing starts parked).
    treat_z = np.asarray(states.data.xpos[:, np.asarray(env._treat_body_ids), 2])
    np.testing.assert_allclose(
        treat_z, float(env._config.treat_height), atol=1e-6
    )


def test_vmap_reset_samples_without_replacement(env):
    """Treats land on distinct free cells and never on the spawn cell."""
    batch = 8
    rngs = jax.random.split(jax.random.PRNGKey(1), batch)
    states = jax.jit(jax.vmap(env.reset))(rngs)
    treats = _treat_xy(env, states.data)
    spawn = _spawn_xy(env, states.data)
    free = env.free_cell_positions

    for w in range(batch):
        assert np.unique(treats[w], axis=0).shape[0] == env.n_treats
        assert not np.any(np.all(np.isclose(treats[w], spawn[w]), axis=-1))
        for xy in treats[w]:
            assert np.any(np.all(np.isclose(free, xy, atol=1e-5), axis=-1))
        assert np.any(np.all(np.isclose(free, spawn[w], atol=1e-5), axis=-1))


def test_reset_propagates_kinematics(env):
    """``mjx.forward`` must run in reset, else xpos holds the compile pose."""
    state = jax.jit(env.reset)(jax.random.PRNGKey(5))
    treat_qpos = np.asarray(
        state.data.qpos[np.asarray(env._treat_slide_qpos_idxs_np)]
    )
    xy_from_qpos = treat_qpos[:, :2]
    np.testing.assert_allclose(_treat_xy(env, state.data), xy_from_qpos, atol=1e-6)
    # Compile-time pose is (0, 0, treat_height); at least one treat moved.
    assert np.any(np.abs(xy_from_qpos) > 1e-3)


# ===========================================================================
# 4. Proprioception slicing despite the qpos shift (highest-risk path)
# ===========================================================================


def test_joint_angle_slice_length_matches_motor_joints(env):
    """``_get_joint_angles`` must return the rodent's hinge count, not nq."""
    model = env.mj_model
    n_hinge = int(np.sum(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE))
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))

    angles = env._get_joint_angles(state.data)
    assert angles.shape == (n_hinge,)
    assert angles.shape[0] != model.nq  # would be the silent-corruption case

    ang_vels = env._get_joint_ang_vels(state.data)
    assert ang_vels.shape == (n_hinge,)

    # The base class returns the FULL qfrc_actuator of a bare rodent: the
    # rodent's 6 root dofs + its hinge dofs.
    ctrl = env._get_actuator_ctrl(state.data)
    assert ctrl.shape == (6 + n_hinge,)


def test_joint_angle_slice_covers_exactly_the_rodent_hinges(env):
    """The slice's address set must equal the rodent hinge joints' qposadr."""
    model = env.mj_model
    hinge_adr = sorted(
        int(model.jnt_qposadr[j])
        for j in range(model.njnt)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    )
    sliced = list(range(env._rodent_qpos_start, model.nq))
    assert sliced == hinge_adr

    hinge_dof = sorted(
        int(model.jnt_dofadr[j])
        for j in range(model.njnt)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    )
    assert list(range(env._rodent_qvel_start, model.nv)) == hinge_dof


def _posed_state(env, pitch_deg, root_z=None):
    """Reset state with the root free joint pitched ``pitch_deg`` about +y.

    ``root_z`` matters as much as the angle. Rotating the root about its own
    origin drops the torso to the floor, which is a SPRAWL, not a rear -- real
    rearing lifts the torso (reference: past 90 deg it is never below 0.1146 m).
    Pass a height from the reference distribution to pose an actual rear.
    """
    st = jax.jit(env.reset)(jax.random.PRNGKey(0))
    root = env._rodent_root_qpos
    half = np.deg2rad(pitch_deg) / 2.0
    quat = jp.array([np.cos(half), 0.0, np.sin(half), 0.0], dtype=jp.float32)
    qpos = st.data.qpos.at[root + 3 : root + 7].set(quat)
    if root_z is not None:
        qpos = qpos.at[root + 2].set(root_z)
    return st, mjx.forward(env.mjx_model, st.data.replace(qpos=qpos))


def _hold_pose(env, data, info, n_steps):
    """Run the posture counters ``n_steps`` times against a frozen pose."""
    info = dict(info)
    for _ in range(n_steps):
        env._update_posture_counters(data, info)
    return info


# (pitch, root height) drawn from the reference conditional medians: a real rat
# at higher tilt is rearing, and rearing is HIGH. See _belly_up_termination.
_REARING_POSES = [(0, 0.070), (45, 0.115), (90, 0.147), (120, 0.158)]


@pytest.mark.parametrize("pitch_deg,root_z", _REARING_POSES)
def test_belly_up_never_fires_on_rearing(env, pitch_deg, root_z):
    """Rearing must not trip the gate, at any pitch-and-height a real rat reaches.

    The heights are not decoration. Posing by rotation alone puts the torso on
    the floor, which the height arm of the gate correctly reads as fallen -- so
    a test that omitted them would be asserting the wrong thing about the wrong
    posture. These pairs come from the reference distribution.
    """
    st, data = _posed_state(env, pitch_deg, root_z=root_z)
    info = _hold_pose(env, data, st.info, 400)  # 8x the default patience
    assert int(info["belly_up_steps"]) == 0
    assert not bool(env._belly_up_termination(data, info))


def test_belly_up_catches_a_sprawled_rat_that_never_reaches_140_deg(env):
    """The case a pure tilt gate misses: down, but only ~94 deg from upright.

    Dropping the rodent at a 150 deg roll and letting it settle gives torso tilt
    94.3 deg at z = -0.004 m -- nowhere near 140 deg, so a tilt-only gate stays
    silent and the episode runs until the separate height gate ends it. Real
    rearing reaches 120.8 deg, so no tilt cut separates the two; height does,
    because a reference rat past 90 deg is rearing and never below 0.1146 m.
    """
    crit = env._config.termination_criteria["belly_up"]
    st = jax.jit(env.reset)(jax.random.PRNGKey(0))
    root = env._rodent_root_qpos

    # 100 deg roll, torso pushed to the floor: down, but far short of 140 deg.
    half = np.deg2rad(100.0) / 2.0
    qpos = st.data.qpos.at[root + 3 : root + 7].set(
        jp.array([np.cos(half), np.sin(half), 0.0, 0.0], dtype=jp.float32)
    )
    qpos = qpos.at[root + 2].set(0.02)
    data = mjx.forward(env.mjx_model, st.data.replace(qpos=qpos))

    torso_z, cos_tilt = env._torso_posture(data)
    tilt = np.degrees(np.arccos(np.clip(float(cos_tilt), -1.0, 1.0)))
    assert tilt < float(crit["max_tilt_angle"]), "pose must NOT trip the tilt-only arm"
    assert tilt > float(crit["fallen_tilt_angle"])
    assert float(torso_z) < float(crit["fallen_max_torso_z"])

    info = _hold_pose(env, data, st.info, int(crit["patience"]))
    assert bool(env._belly_up_termination(data, info))


def test_belly_up_stays_silent_on_a_reared_rat_at_the_same_tilt(env):
    """Same tilt as the sprawled case, but HIGH -- that is rearing, not falling.

    This is the pair that makes the height term load-bearing: identical torso
    orientation, opposite verdict, decided entirely by z.
    """
    crit = env._config.termination_criteria["belly_up"]
    st = jax.jit(env.reset)(jax.random.PRNGKey(0))
    root = env._rodent_root_qpos

    half = np.deg2rad(100.0) / 2.0
    qpos = st.data.qpos.at[root + 3 : root + 7].set(
        jp.array([np.cos(half), np.sin(half), 0.0, 0.0], dtype=jp.float32)
    )
    # 0.15 m is inside the reference range for a reared rat (min 0.1274 at
    # 100-110 deg tilt); the sprawled twin above sits at 0.02.
    qpos = qpos.at[root + 2].set(0.15)
    data = mjx.forward(env.mjx_model, st.data.replace(qpos=qpos))

    assert float(env._torso_posture(data)[0]) > float(crit["fallen_max_torso_z"])
    info = _hold_pose(env, data, st.info, 400)
    assert int(info["belly_up_steps"]) == 0
    assert not bool(env._belly_up_termination(data, info))


def test_belly_up_fires_when_inverted_and_only_after_patience(env):
    """Inverted terminates, but only once it has held -- that is 'cannot recover'."""
    patience = int(env._config.termination_criteria["belly_up"]["patience"])
    st, data = _posed_state(env, 178)

    _, cos_tilt = env._torso_posture(data)
    assert float(cos_tilt) < np.cos(np.deg2rad(140.0)), "pose is not actually inverted"

    just_short = _hold_pose(env, data, st.info, patience - 1)
    assert not bool(env._is_done(data, just_short, dict(st.metrics)))
    long_enough = _hold_pose(env, data, st.info, patience)
    assert bool(env._is_done(data, long_enough, dict(st.metrics)))


def test_posture_counters_reset_on_a_single_good_step(env):
    """One recovered step wipes the counter, so a tumble-and-right never ends it."""
    st, inverted = _posed_state(env, 178)
    _, upright = _posed_state(env, 0)
    patience = int(env._config.termination_criteria["belly_up"]["patience"])

    info = _hold_pose(env, inverted, st.info, patience - 1)
    assert int(info["belly_up_steps"]) == patience - 1
    env._update_posture_counters(upright, info)
    assert int(info["belly_up_steps"]) == 0


def test_torso_too_low_needs_patience_and_clears_normal_heights(env):
    """The height gate is below anything the reference rat does (min 0.0354 m)."""
    st = jax.jit(env.reset)(jax.random.PRNGKey(0))
    root = env._rodent_root_qpos
    patience = int(env._config.termination_criteria["torso_too_low"]["patience"])
    min_z = float(env._config.termination_criteria["torso_too_low"]["min_torso_z"])

    standing_z, _ = env._torso_posture(st.data)
    assert float(standing_z) > min_z
    assert int(_hold_pose(env, st.data, st.info, 400)["low_torso_steps"]) == 0

    sunk = mjx.forward(
        env.mjx_model, st.data.replace(qpos=st.data.qpos.at[root + 2].add(-0.05))
    )
    assert float(env._torso_posture(sunk)[0]) < min_z
    assert not bool(env._is_done(sunk, _hold_pose(env, sunk, st.info, patience - 1),
                                 dict(st.metrics)))
    assert bool(env._is_done(sunk, _hold_pose(env, sunk, st.info, patience),
                             dict(st.metrics)))


_REFERENCE_CLIPS = pathlib.Path(
    "/home/talmolab/Desktop/SalkResearch/SCAMPER/data/rodent/rodent_reference_clips.h5"
)


@pytest.mark.skipif(
    not _REFERENCE_CLIPS.exists(), reason="reference clip file not available"
)
def test_no_posture_gate_fires_on_real_rat_motion(env):
    """THE guarantee: replay real rat motion and terminate on none of it.

    ``rodent_reference_clips.h5`` is the data this task's frozen prior was
    trained to imitate, so it is the operational definition of "normal walking
    and rearing". Both gates are calibrated to sit outside it -- measured on
    every 20th frame, replayed through THIS model (so heights are in our units,
    not the reference's):

        torso z   p0 0.0355   p1 0.0494   p50 0.0728   (gate 0.0325)
        tilt      p50 11.4    p99 98.3    max 110.9    (gate 140.0)

    A tilt gate of 85 deg -- what ``fallen`` and the DMPO gap arms use -- would
    fire on 8.13% of these frames. That is the regression this test exists to
    prevent.
    """
    import h5py

    with h5py.File(_REFERENCE_CLIPS, "r") as handle:
        reference = handle["qpos"][::200].astype(np.float64)

    # CPU MuJoCo on purpose: this is a calibration check on the thresholds, and
    # forward kinematics is identical. Looping mjx.forward here allocates a full
    # device Data per frame and OOMs whenever a training run holds the GPU.
    crit = env._config.termination_criteria
    min_z = float(crit["torso_too_low"]["min_torso_z"])
    max_tilt = float(crit["belly_up"]["max_tilt_angle"])

    model = env.mj_model
    data = mujoco.MjData(model)
    root = env._rodent_root_qpos
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso-rodent")

    heights = np.empty(len(reference))
    cos_tilt = np.empty(len(reference))
    for i, frame in enumerate(reference):
        data.qpos[root:] = frame
        mujoco.mj_forward(model, data)
        heights[i] = data.xpos[torso_id, 2]
        cos_tilt[i] = data.xmat[torso_id].reshape(3, 3)[2, 2]
    tilt = np.degrees(np.arccos(np.clip(cos_tilt, -1.0, 1.0)))

    assert heights.min() > min_z, (
        f"height gate {min_z} fires on real motion (min {heights.min():.4f} m)"
    )
    assert tilt.max() < max_tilt, (
        f"tilt gate {max_tilt} fires on real motion (max {tilt.max():.1f} deg)"
    )
    # And the regression this is really guarding: the gap arms' 85 deg gate
    # would fire on a large fraction of ordinary rearing.
    assert (tilt > 85.0).mean() > 0.05


def test_nan_termination_watches_qpos_and_qvel_only(env):
    """The NaN tripwire reads the integrator state, deliberately not all of ``data``.

    The obvious implementation -- ``ravel_pytree(data)`` -- is what this env
    shipped with, and on the warp backend it is fatal rather than merely slow:
    ``data`` carries contact buffers sized O(naconmax), and flattening it under
    ``vmap`` copies them per world. At the shipped 2048 envs / naconmax=65536
    that is a single 21.7 GiB allocation and the DMPO run dies before its first
    step. ``run_gap.py`` narrowed the same check for the same reason (PR #67).

    The narrowing is safe because qpos and qvel ARE the integrator state: a NaN
    anywhere in the pipeline reaches qvel on the step it is produced. The third
    assertion pins the accepted cost -- a NaN confined to a non-integrator field
    is not caught on that step -- so nobody "fixes" it back to a full ravel.
    """
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    data = state.data
    info, metrics = state.info, dict(state.metrics)

    assert not bool(env._is_done(data, info, metrics))
    assert bool(
        env._is_done(data.replace(qpos=data.qpos.at[0].set(jp.nan)), info, metrics)
    )
    assert bool(
        env._is_done(data.replace(qvel=data.qvel.at[0].set(jp.nan)), info, metrics)
    )
    # Accepted blind spot, stated rather than hidden.
    assert not bool(
        env._is_done(data.replace(xpos=data.xpos.at[0, 0].set(jp.nan)), info, metrics)
    )


def test_proprioception_ignores_the_treat_slide_joints(env):
    """Writing garbage into the treat dofs must not leak into proprioception.

    This is the concrete form of the qpos-shift bug: an off-by-3N slice would
    read treat slide offsets as if they were joint angles.
    """
    state = jax.jit(env.reset)(jax.random.PRNGKey(2))
    data = state.data
    baseline = np.asarray(env._get_joint_angles(data))
    baseline_v = np.asarray(env._get_joint_ang_vels(data))

    treat_idxs = jp.asarray(np.asarray(env._treat_slide_qpos_idxs_np).ravel())
    marker = jp.full((treat_idxs.shape[0],), -7.5)
    poisoned = data.replace(qpos=data.qpos.at[treat_idxs].set(marker))
    poisoned = poisoned.replace(
        qvel=poisoned.qvel.at[: 3 * env.n_treats].set(-7.5)
    )

    np.testing.assert_array_equal(
        np.asarray(env._get_joint_angles(poisoned)), baseline
    )
    np.testing.assert_array_equal(
        np.asarray(env._get_joint_ang_vels(poisoned)), baseline_v
    )
    assert -7.5 not in set(baseline.tolist())


def test_proprioception_tracks_the_rodent_joints(env):
    """A marker written into a rodent hinge must appear at the right index."""
    state = jax.jit(env.reset)(jax.random.PRNGKey(2))
    data = state.data
    k = 5
    addr = env._rodent_qpos_start + k
    marked = data.replace(qpos=data.qpos.at[addr].set(0.4242))
    angles = np.asarray(env._get_joint_angles(marked))
    assert angles[k] == pytest.approx(0.4242)


def test_proprioception_obs_size_is_rodent_sized(env, small_env):
    """The proprioception subtree must not grow with the treat count."""
    assert int(env.proprioceptive_obs_size) == int(
        small_env.proprioceptive_obs_size
    )


# ===========================================================================
# 5. step() runs; obs keys/shapes match the declared sizes
# ===========================================================================


def test_step_runs_and_keeps_obs_structure(env):
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    step = jax.jit(env.step)
    action = env.null_action()
    for _ in range(5):
        state = step(state, action)
        assert not np.any(np.isnan(np.asarray(state.reward)))
        assert state.reward.shape == ()
        assert state.done.shape == ()
    assert set(state.obs.keys()) == {"state", "privileged_state"}
    assert list(state.obs["state"].keys()) == [
        "task_obs",
        "proprioception",
        "vision",
    ]
    assert int(state.info["step_count"]) == 5


def test_obs_shapes_match_non_flattened_observation_size(env):
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    declared = env.non_flattened_observation_size
    actual = jax.tree_util.tree_map(
        lambda x: int(np.prod(x.shape)), state.obs
    )
    declared_i = jax.tree_util.tree_map(lambda x: int(x), declared)
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(
        declared_i
    )
    assert jax.tree_util.tree_leaves(actual) == jax.tree_util.tree_leaves(
        declared_i
    )


def test_observation_size_excludes_vision_and_privileged(env):
    """``observation_size`` = task_obs + proprioception only."""
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    obs = state.obs["state"]
    task = int(obs["task_obs"].shape[0])
    prop = int(
        jax.flatten_util.ravel_pytree(obs["proprioception"])[0].shape[0]
    )
    assert int(env.observation_size) == task + prop
    assert int(env.proprioceptive_obs_size) == prop
    assert env.vision_obs_size == int(np.prod(env.vision_shape))
    assert obs["vision"].shape == env.vision_shape
    # The vision placeholder is zeros until VisionRenderWrapper fills it.
    assert not np.any(np.asarray(obs["vision"]))


def test_vmap_step_runs(env):
    batch = 4
    rngs = jax.random.split(jax.random.PRNGKey(0), batch)
    state = jax.jit(jax.vmap(env.reset))(rngs)
    state = jax.jit(jax.vmap(env.step))(state, jp.zeros((batch, env.action_size)))
    assert state.reward.shape == (batch,)
    assert state.obs["state"]["task_obs"].shape[0] == batch


# ===========================================================================
# 6. PURE-VISION CONTRACT: task_obs carries no treat information
# ===========================================================================


def test_task_obs_size_equals_declared_components(env):
    """[prev_action, kinematic_sensors, touch] and nothing else.

    ``origin`` is NOT in the default contract any more: it is an exact
    allocentric position + heading fix in a maze that never changes, i.e. free
    global self-localisation, which both defeats the vision-only premise and
    confounds any place-coding claim.  ``include_origin=True`` puts it back.
    """
    assert bool(env._config.include_origin) is False
    expected = env.action_size + N_KINEMATIC_SENSORS + N_TOUCH_SENSORS
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert state.obs["state"]["task_obs"].shape == (expected,)
    declared = env.non_flattened_observation_size["state"]["task_obs"]
    assert int(declared) == expected
    # No room for an egocentric treat vector (3 * n_treats).
    assert expected < expected + 3 * env.n_treats


def test_task_obs_is_unchanged_when_the_treats_move(env):
    """Strong form: hold the rodent fixed, move every treat, diff task_obs.

    If any treat information leaked into ``task_obs`` this would change.
    """
    state = jax.jit(env.reset)(jax.random.PRNGKey(4))
    data = state.data
    info = state.info

    treat_idxs = jp.asarray(np.asarray(env._treat_slide_qpos_idxs_np))
    free = env.free_cell_positions
    # Deterministically relocate every treat to a different free cell.
    old_xy = _treat_xy(env, data)
    new_xy = []
    for i in range(env.n_treats):
        for cand in free:
            if not np.allclose(cand, old_xy[i], atol=1e-5):
                new_xy.append(cand)
                break
    new_offsets = jp.asarray(
        np.concatenate([np.asarray(new_xy), np.zeros((env.n_treats, 1))], axis=-1),
        dtype=jp.float32,
    )

    moved = data.replace(qpos=data.qpos.at[treat_idxs].set(new_offsets))
    moved = mjx.forward(env.mjx_model, moved)

    # Sanity: the treats really did move.
    assert not np.allclose(_treat_xy(env, moved), old_xy, atol=1e-4)

    before = env._get_obs(data, info)
    after = env._get_obs(moved, info)

    np.testing.assert_array_equal(
        np.asarray(after["state"]["task_obs"]),
        np.asarray(before["state"]["task_obs"]),
    )
    # Proprioception is likewise treat-blind (treats are non-colliding).
    flat_before = jax.flatten_util.ravel_pytree(
        before["state"]["proprioception"]
    )[0]
    flat_after = jax.flatten_util.ravel_pytree(after["state"]["proprioception"])[0]
    np.testing.assert_array_equal(np.asarray(flat_after), np.asarray(flat_before))

    # ...but the critic's privileged view DID change -- proving the mover
    # actually moved something observable.
    assert not np.allclose(
        np.asarray(after["privileged_state"]["treat_vectors"]),
        np.asarray(before["privileged_state"]["treat_vectors"]),
        atol=1e-4,
    )


def test_privileged_state_carries_treat_information(env):
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    priv = state.obs["privileged_state"]
    assert "treat_vectors" in priv
    assert "collected" in priv
    assert priv["treat_vectors"].shape == (3 * env.n_treats,)
    assert priv["collected"].shape == (env.n_treats,)
    # task_obs must stay present (HighLevelWrapper indexes it).
    assert "task_obs" in priv


def test_no_treat_key_leaks_into_the_policy_obs(env):
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    policy_obs = state.obs["state"]
    assert "treat_vectors" not in policy_obs
    assert "collected" not in policy_obs
    assert "ego_target" not in policy_obs
    prop = policy_obs["proprioception"]
    assert isinstance(prop, collections.OrderedDict)
    assert not any("treat" in k or "target" in k for k in prop)


# ===========================================================================
# 7. INFO-RESET REGRESSION: `collected` must not ratchet across auto-reset
# ===========================================================================


def _wrapped_stack(env, episode_length=8, info_reset=True):
    """brax training stack (full_reset=False), optionally + info reset."""
    from mujoco_playground._src import wrapper as mp_wrapper

    from vnl_playground.tasks.wrappers_info_reset import InfoResetOnDoneWrapper

    wrapped = mp_wrapper.wrap_for_brax_training(
        env, episode_length=episode_length, action_repeat=1, full_reset=False
    )
    if info_reset:
        wrapped = InfoResetOnDoneWrapper(wrapped, keys=INFO_RESET_KEYS)
    return wrapped


def test_fresh_reset_has_no_collected_treats(env):
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert state.info["collected"].shape == (env.n_treats,)
    assert state.info["collected"].dtype == jp.bool_
    assert not np.any(np.asarray(state.info["collected"]))
    assert int(state.info["n_collected"]) == 0
    assert int(state.info["step_count"]) == 0
    assert float(state.done) == 0.0


def test_info_reset_keys_are_all_written_by_reset(env):
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    missing = [k for k in INFO_RESET_KEYS if k not in state.info]
    assert not missing, f"INFO_RESET_KEYS not produced by reset(): {missing}"


def test_collected_does_not_ratchet_across_autoreset(env):
    """The eight-lost-DMPO-runs bug shape, exercised through the real stack.

    Force the mask full, let ``all_treats_collected`` fire, and check the mask
    is back to all-False on the other side of the auto-reset boundary -- and
    stays False on the following step.
    """
    batch = 2
    wrapped = _wrapped_stack(env, episode_length=8, info_reset=True)
    state = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(0), batch))
    assert not np.any(np.asarray(state.info["collected"]))

    step = jax.jit(wrapped.step)
    action = jp.zeros((batch, env.action_size))

    # Simulate "every treat collected" without having to walk the maze.
    state.info["collected"] = jp.ones((batch, env.n_treats), dtype=bool)
    state.info["n_collected"] = jp.full((batch,), env.n_treats, dtype=jp.int32)

    state = step(state, action)
    assert np.all(np.asarray(state.done) > 0.5), "all_treats_collected did not fire"
    assert not np.any(np.asarray(state.info["collected"])), (
        "collected ratcheted across the auto-reset boundary"
    )
    assert np.all(np.asarray(state.info["n_collected"]) == 0)
    assert np.all(np.asarray(state.info["step_count"]) == 0)

    # And the new episode behaves like a fresh one.
    state = step(state, action)
    assert np.all(np.asarray(state.done) < 0.5)
    assert not np.any(np.asarray(state.info["collected"]))


def test_collected_ratchets_without_the_info_reset_wrapper(env):
    """Documents *why* InfoResetOnDoneWrapper is mandatory for this env.

    ``BraxAutoResetWrapper(full_reset=False)`` swaps data/obs but not info, so
    the bare stack keeps the mask set forever.  If this test ever starts
    failing because the bare stack no longer ratchets, the wrapper requirement
    can be relaxed -- until then it must stay.
    """
    batch = 2
    bare = _wrapped_stack(env, episode_length=8, info_reset=False)
    state = jax.jit(bare.reset)(jax.random.split(jax.random.PRNGKey(0), batch))
    step = jax.jit(bare.step)
    action = jp.zeros((batch, env.action_size))

    state.info["collected"] = jp.ones((batch, env.n_treats), dtype=bool)
    state = step(state, action)
    assert np.all(np.asarray(state.done) > 0.5)
    assert np.all(np.asarray(state.info["collected"])), (
        "bare stack no longer ratchets; revisit the wrapper requirement"
    )


def test_reward_pays_once_per_treat(env):
    """A treat newly in reach pays; the same treat next step does not."""
    state = jax.jit(env.reset)(jax.random.PRNGKey(6))
    data = state.data
    info = dict(state.info)

    # Teleport treat 0 onto the torso so it is unambiguously "reached".
    torso_xy = np.asarray(env._torso(data).xpos[:2])
    idx = jp.asarray(np.asarray(env._treat_slide_qpos_idxs_np)[0, :2])
    data = data.replace(qpos=data.qpos.at[idx].set(jp.asarray(torso_xy)))
    data = mjx.forward(env.mjx_model, data)

    metrics = {}
    first = float(env._get_reward(data, info, metrics))
    assert first == pytest.approx(
        float(env._config.reward_terms["treat_collected"]["weight"])
    )

    info["collected"] = info["collected"].at[0].set(True)
    second = float(env._get_reward(data, info, {}))
    assert second == 0.0


def test_collected_treats_are_parked_underground(env):
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    collected = jp.array([True] + [False] * (env.n_treats - 1))
    parked = env._park_collected_treats(state.data, collected)
    z = np.asarray(parked.qpos[np.asarray(env._treat_z_qpos_idxs)])
    expected = -(float(env._config.park_depth) + float(env._config.treat_height))
    assert z[0] == pytest.approx(expected)
    np.testing.assert_allclose(z[1:], 0.0, atol=1e-7)
    # The park depth has to be inside the slide joint's range, otherwise
    # MuJoCo fights it with a limit constraint instead of clamping.
    assert abs(expected) <= env._slide_range


# ===========================================================================
# 8. Vision render smoke test (GPU only)
# ===========================================================================


@pytest.mark.skipif(not _HAS_GPU, reason="vision rendering needs a GPU")
def test_vision_render_smoke():
    """VisionRenderWrapper must overwrite the zeros placeholder with pixels."""
    from mujoco_playground._src import wrapper as mp_wrapper

    from vnl_playground.tasks.rodent.vision_jax import VisionRenderWrapper

    vis_env = _build_env(vision_width=32, vision_height=32)
    batch = 2
    wrapped = mp_wrapper.wrap_for_brax_training(
        vis_env, episode_length=8, action_repeat=1, full_reset=False
    )
    wrapped = VisionRenderWrapper(
        wrapped,
        mj_model=vis_env.mj_model,
        mjx_model=vis_env.mjx_model,
        width=32,
        height=32,
        grayscale=bool(vis_env._config.grayscale),
        render_depth=False,
        use_textures=bool(vis_env._config.use_textures),
        use_shadows=bool(vis_env._config.use_shadows),
        camera_name=str(vis_env._config.vision_camera_name),
    )

    state = wrapped.reset(jax.random.split(jax.random.PRNGKey(0), batch))
    vision = np.asarray(state.obs["state"]["vision"])
    assert vision.shape == (batch,) + vis_env.vision_shape
    assert np.all(np.isfinite(vision))
    assert np.any(vision != 0.0), "renderer returned an all-zero image"

    state = wrapped.step(state, jp.zeros((batch, vis_env.action_size)))
    vision2 = np.asarray(state.obs["state"]["vision"])
    assert vision2.shape == (batch,) + vis_env.vision_shape
    assert np.any(vision2 != 0.0)


# ===========================================================================
# 9. ADVERSARIAL-REVIEW REGRESSIONS (2026-08-21)
# ===========================================================================
# Everything below pins a defect that three independent reviewers found in the
# first cut of this task.  The two critical ones both come from
# ``BraxAutoResetWrapper(full_reset=False)``, which the entry point used to
# hardcode: it restores ``data``/``obs`` from the FIRST reset and never clears
# ``state.info``, so ``env.reset`` ran exactly once per env index for a whole
# 1e9-step run.


def _full_reset_stack(env, cfg, episode_length=4):
    """Wraps ``env`` exactly the way ``train_highlvl``'s vision path does."""
    from vnl_playground import train_highlvl

    return train_highlvl._wrap_for_brax_training(
        env, cfg, episode_length=episode_length, action_repeat=1
    )


def test_entry_point_wrapper_block_gives_per_episode_layout(env):
    """The design's central premise, exercised through the real wrap helper.

    ``wrappers.full_reset: true`` must make the auto-reset call ``env.reset``,
    so the spawn pose, the heading and every treat cell are re-drawn on the
    other side of a ``done`` -- and ``collected`` comes back all-False with it.
    With ``full_reset=False`` all of that is bit-identical forever (see the
    companion test below), and the agent can memorise one route per worker.
    """
    from omegaconf import OmegaConf

    batch = 3
    cfg = OmegaConf.create({"wrappers": {"full_reset": True}})
    wrapped = _full_reset_stack(env, cfg, episode_length=4)
    state = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(0), batch))
    step = jax.jit(wrapped.step)
    action = jp.zeros((batch, env.action_size))

    first_treats = _treat_xy(env, state.data)
    root = env._rodent_root_qpos
    first_root = np.asarray(state.data.qpos[:, root : root + 7])

    # Force the episode to end (all treats collected).
    state.info["collected"] = jp.ones((batch, env.n_treats), dtype=bool)
    state = step(state, action)
    assert np.all(np.asarray(state.done) > 0.5)

    assert not np.allclose(_treat_xy(env, state.data), first_treats, atol=1e-6), (
        "treat layout did not change across the auto-reset boundary"
    )
    new_root = np.asarray(state.data.qpos[:, root : root + 7])
    assert not np.allclose(new_root[:, :2], first_root[:, :2], atol=1e-6), (
        "spawn xy did not change across the auto-reset boundary"
    )
    assert not np.allclose(new_root[:, 3:7], first_root[:, 3:7], atol=1e-6), (
        "spawn heading did not change across the auto-reset boundary"
    )
    # ...and the info half of the same bug is fixed too.
    assert not np.any(np.asarray(state.info["collected"]))
    assert np.all(np.asarray(state.info["n_collected"]) == 0)

    # The next episode runs normally instead of terminating on step 1.
    state = step(state, action)
    assert np.all(np.asarray(state.done) < 0.5)


@pytest.mark.parametrize("info_reset", [False, True])
def test_layout_freezes_without_full_reset(env, info_reset):
    """Documents *why* ``full_reset`` is mandatory, and why info-reset is not enough.

    ``BraxAutoResetWrapper(full_reset=False)`` restores ``data`` from the first
    reset, and the spawn/treat layout lives entirely in ``data.qpos`` -- so the
    layout is frozen per env index for the whole run.  ``InfoResetOnDoneWrapper``
    fixes the ``info`` ratchet only; it cannot unfreeze the layout, which is
    exactly why the entry-point guard demands ``full_reset``.
    """
    batch = 2
    wrapped = _wrapped_stack(env, episode_length=4, info_reset=info_reset)
    state = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(0), batch))
    step = jax.jit(wrapped.step)
    action = jp.zeros((batch, env.action_size))

    first_treats = _treat_xy(env, state.data)
    root = env._rodent_root_qpos
    first_root = np.asarray(state.data.qpos[:, root : root + 7])

    state.info["collected"] = jp.ones((batch, env.n_treats), dtype=bool)
    state = step(state, action)
    assert np.all(np.asarray(state.done) > 0.5)

    np.testing.assert_allclose(_treat_xy(env, state.data), first_treats, atol=1e-7)
    np.testing.assert_allclose(
        np.asarray(state.data.qpos[:, root : root + 7]), first_root, atol=1e-7
    )


@pytest.mark.parametrize(
    "entry_point_module",
    ["train_highlvl", "train_highlvl_dmpo_kl_anchor"],
)
def test_entry_point_refuses_a_frozen_layout_config(env, entry_point_module):
    """Both entry points must raise rather than launch a silently dead run.

    The DMPO one is the launcher this task actually ships with
    (``config/maze_forage_vision/dmpo_maze_forage.yaml``), and it had the
    frozen-layout auto-reset hardcoded until the ``wrappers:`` block was ported
    into it -- so it needs the same coverage as the PPO one, not less.
    """
    import importlib

    from omegaconf import OmegaConf

    train_highlvl = importlib.import_module(f"vnl_playground.{entry_point_module}")

    assert env.requires_per_episode_reset is True
    assert tuple(env.info_reset_keys) == tuple(INFO_RESET_KEYS)

    # No `wrappers:` block at all -- the shape that shipped.
    with pytest.raises(ValueError, match="requires_per_episode_reset"):
        train_highlvl._validate_reset_requirements(
            OmegaConf.create({}), env, "RodentMazeForageVision"
        )

    # Info-reset alone does NOT unfreeze the layout, so it is still refused.
    with pytest.raises(ValueError, match="requires_per_episode_reset"):
        train_highlvl._validate_reset_requirements(
            OmegaConf.create(
                {
                    "wrappers": {
                        "info_reset_on_done": True,
                        "info_reset_keys": list(INFO_RESET_KEYS),
                    }
                }
            ),
            env,
            "RodentMazeForageVision",
        )

    # The shipped config passes.
    train_highlvl._validate_reset_requirements(
        OmegaConf.create({"wrappers": {"full_reset": True}}),
        env,
        "RodentMazeForageVision",
    )

    # Explicit frozen-layout ablation is allowed, but only with a full
    # info-reset behind it.
    train_highlvl._validate_reset_requirements(
        OmegaConf.create(
            {
                "wrappers": {
                    "allow_frozen_layout": True,
                    "info_reset_on_done": True,
                    "info_reset_keys": list(INFO_RESET_KEYS),
                }
            }
        ),
        env,
        "RodentMazeForageVision",
    )
    # ...and an incomplete key list is not a full info reset.
    with pytest.raises(ValueError, match="requires_per_episode_reset"):
        train_highlvl._validate_reset_requirements(
            OmegaConf.create(
                {
                    "wrappers": {
                        "allow_frozen_layout": True,
                        "info_reset_on_done": True,
                        "info_reset_keys": ["prev_action"],
                    }
                }
            ),
            env,
            "RodentMazeForageVision",
        )


def test_shipped_config_enables_per_episode_resets():
    """The yaml the Usage header tells you to run must satisfy the guard."""
    from omegaconf import OmegaConf

    from vnl_playground import train_highlvl

    cfg_path = (
        pathlib.Path(train_highlvl.__file__).parent
        / "config"
        / "maze_forage_vision"
        / "maze_forage.yaml"
    )
    cfg = OmegaConf.load(cfg_path)
    full_reset, _, _ = train_highlvl._wrapper_settings(cfg)
    assert full_reset is True, "maze_forage.yaml must set wrappers.full_reset"
    # Every env_args key must exist in default_config (MjxEnv locks the dict).
    defaults = default_config()
    for key in cfg.env_config.env_args:
        assert key in defaults, f"env_args.{key} is not in default_config()"


def test_default_config_is_not_poisoned_by_config_overrides():
    """A mutable ``config=default_config()`` default argument leaked overrides.

    ``MjxEnv.__init__`` locks the dict and then mutates it in place with
    ``config_overrides``, so one instance built with ``n_treats=6`` used to make
    every later default-constructed instance a 6-treat env (different nq, qpos
    layout and obs tree, no error).
    """
    poisoner = MazeForageVision(config_overrides={"n_treats": 6})
    assert poisoner.n_treats == 6
    assert default_config().n_treats == 20
    assert MazeForageVision().n_treats == 20


# --- Geometry: cell_size is the grid pitch, not the corridor width ---------


def _wall_boxes_from_model(env):
    """``(pos_xy, half_size_xy)`` of every compiled maze wall geom."""
    model = env.mj_model
    ids = [
        i
        for i in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith(
            "maze_wall_"
        )
    ]
    assert ids
    return model.geom_pos[ids][:, :2], model.geom_size[ids][:, :2]


def _free_run_along_y(env, x, y0, y1, step=2.5e-4):
    """Longest gap (in metres) with no wall box, along the segment x=const."""
    pos, half = _wall_boxes_from_model(env)
    ys = np.arange(min(y0, y1), max(y0, y1), step)
    pts = np.stack([np.full_like(ys, x), ys], axis=-1)
    inside = np.any(
        np.all(np.abs(pts[:, None, :] - pos[None]) <= half[None] + 1e-9, axis=-1),
        axis=-1,
    )
    best = run = 0
    for occupied in inside:
        run = 0 if occupied else run + 1
        best = max(best, run)
    return best * step


def _covering_rect(env, row, col):
    """The covering rectangle that owns wall cell ``(row, col)`` (end-exclusive)."""
    for rect in env.maze_walls:
        if (
            int(rect.start.y) <= row < int(rect.end.y)
            and int(rect.start.x) <= col < int(rect.end.x)
        ):
            return rect
    return None


def _one_cell_corridor_cells(env):
    """``(row, col)`` of every wall / floor / wall triple down a grid column."""
    grid = env.maze_grid
    wall = grid == maze_utils.WALL_CHAR
    return [
        (row, col)
        for col in range(grid.shape[1])
        for row in range(grid.shape[0] - 2)
        if wall[row, col] and not wall[row + 1, col] and wall[row + 2, col]
    ]


def _corridor_scans(env):
    """Per one-cell corridor: clear width, and whether anything else is in it.

    ``n_boxes_on_line`` counts the compiled wall boxes that the scan line
    actually passes through inside the corridor.  Two is the clean case (just
    the pair of flanking walls); three means a *perpendicular* wall rectangle
    has been seal-extended into the corridor at a T-junction, which narrows the
    measured width to ``1.5 * cell_size + wall_thickness / 2`` -- see
    ``test_corridor_width_is_two_cells_minus_thickness``.
    """
    grid = env.maze_grid
    cell = env.cell_size
    pos, half = _wall_boxes_from_model(env)
    scans = []
    for row, col in _one_cell_corridor_cells(env):
        near = maze_utils.grid_to_world((row, col), cell, grid.shape)
        far = maze_utils.grid_to_world((row + 2, col), cell, grid.shape)
        x, y_hi, y_lo = float(near[0]), float(near[1]), float(far[1])
        on_line = int(
            np.sum(
                (np.abs(pos[:, 0] - x) <= half[:, 0] + 1e-9)
                & (pos[:, 1] + half[:, 1] > y_lo)
                & (pos[:, 1] - half[:, 1] < y_hi)
            )
        )
        near_rect = _covering_rect(env, row, col)
        far_rect = _covering_rect(env, row + 2, col)
        scans.append(
            dict(
                row=row,
                col=col,
                gap=_free_run_along_y(env, x, y_lo, y_hi),
                n_boxes_on_line=on_line,
                both_flanks_thinned=(
                    int(near_rect.end.y) - int(near_rect.start.y) == 1
                    and int(far_rect.end.y) - int(far_rect.start.y) == 1
                ),
            )
        )
    return scans


def _measured_one_cell_corridors(env) -> np.ndarray:
    """Clear width of every one-cell corridor, rasterised off the box geoms."""
    return np.asarray([scan["gap"] for scan in _corridor_scans(env)], dtype=float)


def _occupancy_raster(env, step=1e-3, pad=0.3):
    """Wall-occupancy raster over the maze footprint plus a ``pad`` margin.

    Returns ``(xs, ys, occupied)``; ``occupied[i, j]`` is True where the point
    ``(xs[i], ys[j])`` is inside a compiled maze wall box.
    """
    pos, half = _wall_boxes_from_model(env)
    half_x, half_y = env.maze_half_extent
    xs = np.arange(-half_x - pad, half_x + pad, step)
    ys = np.arange(-half_y - pad, half_y + pad, step)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    occupied = np.zeros(grid_x.shape, dtype=bool)
    for p, h in zip(pos, half):
        occupied |= (np.abs(grid_x - p[0]) <= h[0]) & (
            np.abs(grid_y - p[1]) <= h[1]
        )
    return xs, ys, occupied


def _sealed_enclosure_bbox(env, step=1e-3, pad=0.3):
    """Bounding box of the free region reachable from inside the maze.

    Flood-fills the free space of :func:`_occupancy_raster` from a known open
    cell.  If the maze leaked, the component would reach into the ``pad``
    margin and the bbox would blow past the footprint -- so this doubles as the
    "the arena is sealed" check.
    """
    from scipy import ndimage  # already a transitive dep (bowl_escape imports it)

    xs, ys, occupied = _occupancy_raster(env, step=step, pad=pad)
    labels, _ = ndimage.label(~occupied)
    seed_x, seed_y = env.free_cell_positions[0]
    label = labels[
        int(np.argmin(np.abs(xs - seed_x))), int(np.argmin(np.abs(ys - seed_y)))
    ]
    assert label > 0, "seed point is inside a wall"
    mask = labels == label
    xi = np.flatnonzero(mask.any(axis=1))
    yi = np.flatnonzero(mask.any(axis=0))
    return (
        float(xs[xi[0]]),
        float(xs[xi[-1]]),
        float(ys[yi[0]]),
        float(ys[yi[-1]]),
    )


def test_corridor_width_is_two_cells_minus_thickness(env):
    """``cell_size`` is the grid PITCH; thinning roughly doubles the clear width.

    ``_wall_box_geometry`` thins a one-cell-wide wall rectangle to
    ``wall_thickness``, handing ``cell_size - wall_thickness`` of floor back to
    the cell on each side, so a nominally one-cell corridor flanked by two
    thinned walls is really ``2 * cell_size - wall_thickness`` across -- 0.334 m
    at the defaults (``cell_size = 2/11``, ``wall_thickness = 0.03``), not the
    0.182 m grid pitch.  Measured here by rasterising the compiled box geoms, so
    the docs (and ``corridor_width``) cannot drift away from the geometry.

    Two shapes deliver LESS than that, and both are checked, not hidden:

    * a flanking wall that spans several cells along the scan axis is not
      thinned on that axis, costing ``(cell_size - wall_thickness) / 2`` per
      such side;
    * a perpendicular wall rectangle seal-extended into a T-junction pokes a
      nub ``cell_size / 2 - 1.5 * wall_thickness`` into the corridor (the
      extension reaches across the neighbouring wall *cell*, and that cell is
      only ``wall_thickness`` full once the neighbour is thinned).  Measured
      here: 0.2878 m instead of 0.3336 m at ``maze_cells=5``, i.e. a corner
      nub, not a narrower corridor along its length.

    Corridors with nothing but their two thinned flanking walls on the scan
    line hit the formula exactly; the floor over *all* corridors is one cell.
    """
    cell = env.cell_size  # DERIVED: config.cell_size is None by default
    thickness = float(env._config.wall_thickness)
    expected = 2.0 * cell - thickness
    assert env.corridor_width == pytest.approx(expected)
    assert env.corridor_width > cell  # the docs used to claim == cell

    scans = _corridor_scans(env)
    measured = [scan["gap"] for scan in scans]
    clean = [
        scan["gap"]
        for scan in scans
        if scan["both_flanks_thinned"] and scan["n_boxes_on_line"] == 2
    ]
    nubbed = [
        scan["gap"]
        for scan in scans
        if scan["both_flanks_thinned"] and scan["n_boxes_on_line"] > 2
    ]

    assert measured, "no one-cell corridor found in the maze"
    assert clean, "no corridor flanked by two thinned walls and nothing else"
    for gap in clean:
        assert gap == pytest.approx(expected, abs=2e-3)
    # The T-junction nub is a known, bounded loss -- pin its size so it cannot
    # grow silently into a real narrowing.  A nub can only ever *remove* floor,
    # so its reach is `min(clean corridor, nub face)`: once the walls are thick
    # enough that `0.5 * cell < 1.5 * thickness` the seal extension no longer
    # reaches past the flanking wall's inner face and costs nothing at all,
    # which is the regime the shipped 0.15 m walls are in.
    for gap in nubbed:
        assert gap == pytest.approx(
            min(expected, 1.5 * cell + thickness / 2.0), abs=2e-3
        )
    assert max(measured) == pytest.approx(expected, abs=2e-3)
    assert min(measured) >= cell - 2e-3


def _ideal_lattice_boxes(env):
    """The centre-to-centre lattice ``_wall_box_geometry`` claims to build.

    Independent of the covering-rectangle decomposition: every wall cell owns a
    ``wall_thickness`` square about its centre, and every 4-adjacent pair of
    wall cells is joined by a ``wall_thickness``-wide band between their
    centres. Derived straight from the grid, so it cannot drift with the
    implementation it checks.
    """
    grid = env.maze_grid
    cell = env.cell_size
    half_t = float(env._config.wall_thickness) / 2.0
    wall = grid == maze_utils.WALL_CHAR
    height, width = grid.shape
    boxes = []
    for row in range(height):
        for col in range(width):
            if not wall[row, col]:
                continue
            cx, cy = maze_utils.grid_to_world((row, col), cell, grid.shape)
            boxes.append((cx - half_t, cx + half_t, cy - half_t, cy + half_t))
            for drow, dcol in ((0, 1), (1, 0)):
                r2, c2 = row + drow, col + dcol
                if r2 < height and c2 < width and wall[r2, c2]:
                    nx, ny = maze_utils.grid_to_world((r2, c2), cell, grid.shape)
                    boxes.append((min(cx, nx) - half_t, max(cx, nx) + half_t,
                                  min(cy, ny) - half_t, max(cy, ny) + half_t))
    return boxes


def _footprint_diff(env, step=2e-3):
    """``(gap, tab)`` area in m^2 between the compiled boxes and the ideal lattice."""
    pos, half = _wall_boxes_from_model(env)
    extent = float(env.maze_half_extent[0])
    axis = np.arange(-extent, extent + step, step)
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")

    def cover(boxes):
        out = np.zeros(grid_x.shape, dtype=bool)
        for lo_x, hi_x, lo_y, hi_y in boxes:
            out |= (grid_x >= lo_x) & (grid_x <= hi_x) & (grid_y >= lo_y) & (grid_y <= hi_y)
        return out

    actual = cover([(p[0] - h[0], p[0] + h[0], p[1] - h[1], p[1] + h[1])
                    for p, h in zip(pos, half)])
    ideal = cover(_ideal_lattice_boxes(env))
    area = step * step
    return float((ideal & ~actual).sum() * area), float((actual & ~ideal).sum() * area)


@pytest.mark.parametrize("cells,thickness", [(10, 0.15), (5, 0.03), (6, 0.25)])
def test_wall_boxes_are_exactly_the_centre_to_centre_lattice(cells, thickness):
    """The compiled walls cover the ideal lattice and nothing else.

    This is the whole placement contract in one assertion, and it is what the
    thin-then-seal rule it replaced could not satisfy: that rule ran a
    rectangle's long axis to the grid-cell BOUNDARY, half a cell past where the
    perpendicular wall's surface ends, so at the shipped geometry it put
    0.60 m^2 of wall outside the lattice -- 0.0788 m of overhang at every
    border corner, plus T-junction nubs -- while still leaving 0.0048 m^2 of
    the lattice uncovered.

    Gap means a hole; tab means wall sticking out into space the lattice says
    is free.  Both must be zero to the raster's own resolution.
    """
    env = _build_env(maze_cells=cells, wall_thickness=thickness, n_treats=4)
    gap, tab = _footprint_diff(env)
    assert gap == pytest.approx(0.0, abs=1e-4), f"lattice not covered: {gap:.5f} m^2"
    assert tab == pytest.approx(0.0, abs=1e-3), f"wall outside the lattice: {tab:.5f} m^2"


def test_border_corner_faces_are_flush():
    """The two border walls meeting at a corner end on the same two planes.

    The user-visible symptom of the old rule was here: each border wall ran to
    the grid-cell boundary on its long axis, so it overhung the perpendicular
    border wall's outer face by ``(cell_size - wall_thickness) / 2`` -- 78.8 mm,
    half a wall thickness, at every one of the four corners.
    """
    env = _build_env(maze_cells=6, wall_thickness=0.15, n_treats=4)
    pos, half = _wall_boxes_from_model(env)
    low, high = pos - half, pos + half
    extent_x, extent_y = env.maze_half_extent
    cell = env.cell_size
    thickness = float(env._config.wall_thickness)
    # Outer surface of the border: the corner cell's centre line, offset by half
    # a wall. The footprint edge is (cell - thickness) / 2 further out.
    outer_x = extent_x - cell / 2.0 + thickness / 2.0
    outer_y = extent_y - cell / 2.0 + thickness / 2.0

    assert low[:, 0].min() == pytest.approx(-outer_x, abs=1e-9)
    assert high[:, 0].max() == pytest.approx(outer_x, abs=1e-9)
    assert low[:, 1].min() == pytest.approx(-outer_y, abs=1e-9)
    assert high[:, 1].max() == pytest.approx(outer_y, abs=1e-9)

    # Every box that reaches the outer plane on one axis must stop at or inside
    # the outer plane on the other -- i.e. nothing overhangs at a corner.
    for i in range(len(pos)):
        if low[i, 0] == pytest.approx(-outer_x, abs=1e-9):
            assert low[i, 1] >= -outer_y - 1e-9
            assert high[i, 1] <= outer_y + 1e-9


def test_no_wall_box_reaches_a_free_cell():
    """No wall material anywhere inside a free cell -- the strongest no-nub claim.

    The old thin-then-seal rule extended a sealed side past the perpendicular
    wall's surface, poking a corner nub into the corridor and narrowing it from
    ``2 * cell_size - wall_thickness`` to ``1.5 * cell_size + wall_thickness / 2``
    at every T-junction. The centre-to-centre lattice cannot do that at all: a
    wall cell owns only a ``wall_thickness`` square about its centre, which stops
    ``(cell_size - wall_thickness) / 2`` short of the cell boundary, and an
    extension stops at the next wall cell's centre. So free cells are wholly
    free, checked here as a strict box/cell intersection rather than by
    measuring a corridor and hoping the worst case was sampled.
    """
    for cells, thickness in ((5, 0.03), (10, 0.15)):
        env = _build_env(maze_cells=cells, wall_thickness=thickness, n_treats=4)
        pos, half = _wall_boxes_from_model(env)
        grid = env.maze_grid
        cell = env.cell_size
        free = np.argwhere(grid != maze_utils.WALL_CHAR)
        centres = maze_utils.grid_to_world(free, cell, grid.shape)
        slack = (cell - thickness) / 2.0
        for (cx, cy) in centres:
            overlap = (
                (np.abs(pos[:, 0] - cx) < half[:, 0] + cell / 2.0 - 1e-9)
                & (np.abs(pos[:, 1] - cy) < half[:, 1] + cell / 2.0 - 1e-9)
            )
            assert not overlap.any(), (
                f"cells={cells}: a wall box reaches free cell at "
                f"({cx:.4f}, {cy:.4f}); every wall should stop {slack:.4f} m short"
            )


def test_corridors_measure_full_width_at_thin_walls():
    """Every one-cell corridor is ``2 * cell_size - wall_thickness`` across.

    Thin walls are the strict case: the wider the wall, the less room a mistake
    in where the wall ENDS has to show up as a narrowing.
    """
    thin = _build_env(maze_cells=5, wall_thickness=0.03, n_treats=4)
    expected = 2.0 * thin.cell_size - 0.03
    scans = [scan for scan in _corridor_scans(thin) if scan["both_flanks_thinned"]]
    assert scans, "no corridor flanked by two thinned walls"
    for scan in scans:
        assert scan["gap"] == pytest.approx(expected, abs=2e-3)


def test_wall_facades_are_never_exactly_coplanar():
    """No two texture facades share a plane, so none can z-fight.

    They would otherwise: a wall's end cap lands on the same plane as the SIDE
    of the wall it ends against and overlaps it only partly, so neither is
    redundant enough to drop. ``_add_wall_texturing_planes`` ranks such a pair
    and steps the standoff by rank. Side faces alone (the default) never
    collide -- two rectangles flanking a shared wall cell meet exactly at its
    centre, edge to edge -- so this is really a guard on the end-cap option.
    """
    import mujoco as _mj

    for end_caps in (False, True):
        env = _build_env(
            maze_cells=6,
            n_treats=4,
            aesthetic="outdoor_natural",
            wall_texture="dm_control",
            wall_facade_end_caps=end_caps,
        )
        model = env.mj_model
        ids = [
            i
            for i in range(model.ngeom)
            if "_face_" in (_mj.mj_id2name(model, _mj.mjtObj.mjOBJ_GEOM, i) or "")
        ]
        assert ids, "aesthetic build emitted no facade planes"
        pos, quat, size = model.geom_pos[ids], model.geom_quat[ids], model.geom_size[ids]

        def frame(i):
            mat = np.zeros(9)
            _mj.mju_quat2Mat(mat, quat[i])
            return mat.reshape(3, 3)

        for a in range(len(ids)):
            fa = frame(a)
            for b in range(a + 1, len(ids)):
                fb = frame(b)
                if abs(abs(float(fa[:, 2] @ fb[:, 2])) - 1.0) > 1e-6:
                    continue  # not parallel
                axis = int(np.argmax(np.abs(fa[:, 2])))
                if abs(pos[a, axis] - pos[b, axis]) > 1e-9:
                    continue  # different depth: cannot z-fight
                span = [
                    sum(np.abs(f[:, k]) * s[k] for k in range(2))
                    for f, s in ((fa, size[a]), (fb, size[b]))
                ]
                overlaps = all(
                    min(pos[a, k] + span[0][k], pos[b, k] + span[1][k])
                    - max(pos[a, k] - span[0][k], pos[b, k] - span[1][k])
                    > 1e-6
                    for k in range(3)
                    if k != axis
                )
                assert not overlaps, (
                    f"end_caps={end_caps}: facades "
                    f"{_mj.mj_id2name(model, _mj.mjtObj.mjOBJ_GEOM, ids[a])} and "
                    f"{_mj.mj_id2name(model, _mj.mjtObj.mjOBJ_GEOM, ids[b])} "
                    "are exactly coplanar and overlap"
                )


def test_collected_treat_is_hidden_on_the_collection_step(env):
    """``_park_collected_treats`` must move the derived pose, not just qpos.

    ``mjx.render`` reads ``geom_xpos``; writing only ``qpos`` left the treat
    visible to the camera (and to ``privileged_state``) for one control step
    after it had been collected and could no longer pay.
    """
    state = jax.jit(env.reset)(jax.random.PRNGKey(6))
    data = state.data
    torso_xy = np.asarray(env._torso(data).xpos[:2])
    idx = jp.asarray(np.asarray(env._treat_slide_qpos_idxs_np)[0, :2])
    data = data.replace(qpos=data.qpos.at[idx].set(jp.asarray(torso_xy)))
    data = mjx.forward(env.mjx_model, data)
    state = state.replace(data=data)

    state = jax.jit(env.step)(state, env.null_action())

    assert float(state.reward) == pytest.approx(1.0)
    assert bool(state.info["collected"][0])
    parked_z = -float(env._config.park_depth)
    body = int(np.asarray(env._treat_body_ids)[0])
    geom = int(np.asarray(env._treat_geom_ids)[0])
    assert float(state.data.xpos[body, 2]) == pytest.approx(parked_z, abs=1e-5)
    assert float(state.data.geom_xpos[geom, 2]) == pytest.approx(parked_z, abs=1e-5)
    # The obs is built after the park, so the critic channel agrees with it.
    treat_vectors = np.asarray(
        state.obs["privileged_state"]["treat_vectors"]
    ).reshape(env.n_treats, 3)
    assert treat_vectors[0, 2] < -0.9
    assert float(state.obs["privileged_state"]["collected"][0]) == 1.0
    # Uncollected treats stay exactly where they were.
    for i in range(1, env.n_treats):
        other = int(np.asarray(env._treat_body_ids)[i])
        assert float(state.data.xpos[other, 2]) == pytest.approx(
            float(env._config.treat_height), abs=1e-5
        )


# --- Observation-contract truthfulness ------------------------------------


def _task_obs_shift_when_translated(env, dx=0.5):
    """L2 norm of the ``task_obs`` delta when the rodent is translated in world x.

    ``origin`` is ``-torso_pos`` rotated into the torso frame, so a rigid world
    translation of ``dx`` changes it by a vector of norm exactly ``dx`` (spread
    over the three components by the current heading).

    The rodent is lifted clear of the walls FIRST, and both the before and the
    after pose are measured up there.  A translation at floor level walks the
    body into a wall box -- the touch sensors then fire on the penetration and
    ``task_obs`` moves for a reason that has nothing to do with allocentric
    leakage (it swamps it: ~245 at the shipped 0.15 m walls).  Comparing two
    contact-free poses isolates the question this test is actually asking.
    """
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    root = env._rodent_root_qpos
    lift = float(env._config.wall_height) + 1.0
    qpos = state.data.qpos.at[root + 2].add(lift)
    data = mjx.forward(env.mjx_model, state.data.replace(qpos=qpos))
    moved = mjx.forward(env.mjx_model, data.replace(qpos=qpos.at[root].add(dx)))
    before = np.asarray(env._get_obs(data, state.info)["state"]["task_obs"])
    after = np.asarray(env._get_obs(moved, state.info)["state"]["task_obs"])
    return float(np.linalg.norm(after - before))


def test_include_origin_adds_exactly_three_elements(env, origin_env):
    """``origin`` is an exact position+heading fix in a fixed maze; A/B-able.

    Default is OFF (the vision-only premise); ``include_origin=True`` recovers
    ``go_to_target_vision``'s contract (DESIGN.md 3e) as an explicit ablation.
    The delta must be exactly the 3 elements of ``origin`` and nothing else.
    """
    assert bool(env._config.include_origin) is False
    assert bool(origin_env._config.include_origin) is True

    base = env.action_size + N_KINEMATIC_SENSORS + N_TOUCH_SENSORS
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert state.obs["state"]["task_obs"].shape == (base,)
    assert int(env.non_flattened_observation_size["state"]["task_obs"]) == base

    with_origin = jax.jit(origin_env.reset)(jax.random.PRNGKey(0))
    assert with_origin.obs["state"]["task_obs"].shape == (base + N_ORIGIN,)
    assert int(
        origin_env.non_flattened_observation_size["state"]["task_obs"]
    ) == base + N_ORIGIN

    # ...and the leading elements are the same quantities in both.
    np.testing.assert_allclose(
        np.asarray(with_origin.obs["state"]["task_obs"])[:base],
        np.asarray(state.obs["state"]["task_obs"]),
        atol=1e-6,
    )

    # observation_size (task_obs + proprioception) tracks the same delta.
    assert int(origin_env.observation_size) - int(env.observation_size) == N_ORIGIN


def test_task_obs_is_invariant_to_a_global_translation(env, origin_env):
    """No global self-localisation channel in the default ``task_obs``.

    Rigidly translating the rodent through the world must leave ``task_obs``
    unchanged: everything in it is egocentric (previous action, IMU, touch).
    With ``include_origin=True`` the same translation moves ``task_obs`` by
    exactly the translation distance -- that IS the confound, measured.
    """
    for dx in (0.25, 0.5, 0.9):
        assert _task_obs_shift_when_translated(env, dx) < 1e-4
        assert _task_obs_shift_when_translated(origin_env, dx) == pytest.approx(
            dx, abs=1e-3
        )


def test_privileged_state_has_no_duplicate_vision_buffer(env):
    """The vision placeholder belongs to ``state`` only.

    A second zeros image in ``privileged_state`` is allocated every reset and
    step, then filled in by ``VisionRenderWrapper._inject_vision`` and dropped.
    """
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert "vision" not in state.obs["privileged_state"]
    assert "vision" in state.obs["state"]


def test_highlvl_wrapper_drops_privileged_state_in_vision_mode(env):
    """Pins the docstring claim: the shipped arch gives the critic no treat info.

    ``arch_name: shared_vision_task_obs`` builds
    ``HighLevelWrapper(pass_vision=True, pass_task_obs=True)``, whose
    ``_process_state`` rebuilds the observation from ``obs['state']`` alone.  If
    this ever starts failing because ``privileged_state`` survives, the
    asymmetric-critic caveat in the module docstring can be deleted.
    """
    from vnl_playground.tasks.wrappers import HighLevelWrapper

    latent_size = 8
    action_size = env.action_size

    def decoder_inference_fn(x):
        del x
        return jp.zeros(action_size), {}

    wrapped = HighLevelWrapper(
        env,
        decoder_inference_fn=decoder_inference_fn,
        latent_size=latent_size,
        pass_vision=True,
        pass_task_obs=True,
    )
    state = wrapped.reset(jax.random.PRNGKey(0))
    assert set(state.obs) == {"imitation_target", "proprioception", "vision"}
    assert "privileged_state" not in state.obs
    assert "treat_vectors" not in state.obs


# ===========================================================================
# 10. SIZING: fixed 6.46 m x 6.46 m arena, derived cell_size (2026-08-21)
# ===========================================================================
# `maze_extent` is now what stays fixed and `cell_size` is derived from it, so
# turning `maze_cells` up narrows the corridors instead of growing the arena.
# The renders in the next phase pick between maze_cells 4 / 5 / 6, so all three
# have to build, be fully connected and admit the rat.


def _grid_size(maze_cells: int) -> int:
    """Character-grid side length for ``maze_cells`` logical cells."""
    return 2 * int(maze_cells) + 1


def _wall_outer_faces(env):
    """``(min_x, max_x, min_y, max_y)`` of the compiled wall boxes, in metres."""
    pos, half = _wall_boxes_from_model(env)
    lo = (pos - half).min(axis=0)
    hi = (pos + half).max(axis=0)
    return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])


def _free_cells_are_connected(grid) -> bool:
    """4-connectivity flood fill over the non-wall cells of a maze grid."""
    free = grid != maze_utils.WALL_CHAR
    idx = np.argwhere(free)
    assert idx.size, "maze has no free cells"
    height, width = grid.shape
    start = (int(idx[0][0]), int(idx[0][1]))
    seen = {start}
    stack = [start]
    while stack:
        row, col = stack.pop()
        for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n_row, n_col = row + d_row, col + d_col
            if (
                0 <= n_row < height
                and 0 <= n_col < width
                and free[n_row, n_col]
                and (n_row, n_col) not in seen
            ):
                seen.add((n_row, n_col))
                stack.append((n_row, n_col))
    return len(seen) == int(free.sum())


def _rodent_aabb(env) -> np.ndarray:
    """World-axis-aligned bounding box ``(dx, dy, dz)`` of the rodent, in metres.

    Built from every non-maze, non-treat, non-worldbody geom's own ``geom_aabb``
    rotated into the world at the compile pose, which is a much tighter bound
    than ``geom_rbound`` (0.308 x 0.080 x 0.092 m vs 0.308 x 0.099 x 0.111 m).
    Planes are skipped -- MuJoCo gives them an effectively infinite aabb.
    """
    model = env.mj_model
    data = mujoco.MjData(model)
    root = env._rodent_root_qpos
    data.qpos[:] = model.qpos0
    data.qpos[root : root + 3] = 0.0
    data.qpos[root + 3 : root + 7] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)

    corners = np.array(
        [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
        dtype=float,
    )
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if name.startswith("maze_wall_") or name.startswith("treat_"):
            continue
        if model.geom_bodyid[gid] == 0:  # worldbody (floor plane)
            continue
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        centre, half = model.geom_aabb[gid, :3], model.geom_aabb[gid, 3:]
        rot = data.geom_xmat[gid].reshape(3, 3)
        pts = data.geom_xpos[gid] + (rot @ (centre + corners * half).T).T
        lo = np.minimum(lo, pts.min(axis=0))
        hi = np.maximum(hi, pts.max(axis=0))
    return hi - lo


def test_maze_extent_is_exactly_two_metres(env):
    """The headline number: a 6.46 m x 6.46 m square arena, measured three ways."""
    grid_size = _grid_size(int(env._config.maze_cells))
    assert env.maze_grid.shape == (grid_size, grid_size)

    # 1. the configured contract.
    assert float(env._config.maze_extent) == MAZE_EXTENT_M
    # 2. grid_size * cell_size, i.e. the derivation itself.
    assert env.cell_size == pytest.approx(MAZE_EXTENT_M / grid_size, abs=1e-12)
    assert grid_size * env.cell_size == pytest.approx(MAZE_EXTENT_M, abs=1e-9)
    # 3. the realised footprint the env reports (re-derived off the grid).
    assert env.maze_extent == pytest.approx(
        (MAZE_EXTENT_M, MAZE_EXTENT_M), abs=1e-9
    )
    assert env.maze_half_extent == pytest.approx(
        (MAZE_EXTENT_M / 2, MAZE_EXTENT_M / 2), abs=1e-9
    )

    # ...and 4: the compiled wall boxes lie inside that footprint, and their
    # bounding box is SYMMETRIC and sits on the border cells' outer surface --
    # half a wall thickness outside the border cell centres, i.e. inset by
    # `(cell_size - wall_thickness) / 2` from the grid footprint. The centre-to-
    # centre lattice puts every rectangle's long axis there whatever the greedy
    # covering did with it. It did NOT used to: a rectangle's long axis ran to
    # the grid-cell boundary, so a one-row border rectangle reached the
    # footprint edge while the border column it met stopped 0.0788 m inside it,
    # and the bbox came out asymmetric -- the visible corner overhang.
    half = MAZE_EXTENT_M / 2
    outer = half - env.cell_size / 2.0 + float(env._config.wall_thickness) / 2.0
    min_x, max_x, min_y, max_y = _wall_outer_faces(env)
    assert min_x >= -half - 1e-9 and max_x <= half + 1e-9
    assert min_y >= -half - 1e-9 and max_y <= half + 1e-9
    for face in (min_x, min_y):
        assert face == pytest.approx(-outer, abs=1e-9)
    for face in (max_x, max_y):
        assert face == pytest.approx(outer, abs=1e-9)

    # 5: the arena is SEALED, and the enclosed span is the footprint minus one
    # border cell and one wall thickness (the border walls are thinned and sit
    # on their cell centres).  A leak would let the flood fill escape into the
    # padding and blow this bbox past the footprint.
    step = 1e-3
    encl_min_x, encl_max_x, encl_min_y, encl_max_y = _sealed_enclosure_bbox(
        env, step=step
    )
    expected_span = MAZE_EXTENT_M - env.cell_size - float(
        env._config.wall_thickness
    )
    assert encl_max_x - encl_min_x == pytest.approx(expected_span, abs=2 * step)
    assert encl_max_y - encl_min_y == pytest.approx(expected_span, abs=2 * step)
    # Symmetric about the origin, and strictly inside the footprint.
    assert encl_min_x == pytest.approx(-encl_max_x, abs=2 * step)
    assert encl_min_y == pytest.approx(-encl_max_y, abs=2 * step)
    assert encl_max_x < half and encl_max_y < half


def test_corridor_width_matches_the_formula(env):
    """``corridor_width == 2 * cell_size - wall_thickness``, from the extent."""
    grid_size = _grid_size(int(env._config.maze_cells))
    cell = MAZE_EXTENT_M / grid_size
    thickness = float(env._config.wall_thickness)
    assert 0.0 < thickness < cell  # else the thinning falls back to full cells
    assert env.cell_size == pytest.approx(cell, abs=1e-12)
    assert env.corridor_width == pytest.approx(2.0 * cell - thickness, abs=1e-12)
    # 6.46/21 m pitch, 0.15 m walls -> 0.4652 m of clear corridor.
    assert env.corridor_width == pytest.approx(0.465238, abs=1e-6)


def test_cell_size_is_derived_and_scales_with_maze_cells():
    """Changing ``maze_cells`` re-scales the corridors, never the arena."""
    extents, cells = [], []
    for maze_cells in PARAMETERISED_MAZE_CELLS:
        sized = _build_env(maze_cells=maze_cells)
        extents.append(sized.maze_extent)
        cells.append(sized.cell_size)
        assert sized.cell_size == pytest.approx(
            MAZE_EXTENT_M / _grid_size(maze_cells), abs=1e-12
        )
    for extent in extents:
        assert extent == pytest.approx((MAZE_EXTENT_M, MAZE_EXTENT_M), abs=1e-9)
    # Strictly decreasing pitch: 0.3800 -> 0.3076 -> 0.2584.
    assert cells == sorted(cells, reverse=True)
    assert len(set(cells)) == len(cells)


def test_explicit_cell_size_must_agree_with_maze_extent():
    """A stale ``cell_size:`` in a yaml must raise, not silently resize the arena."""
    with pytest.raises(ValueError, match="maze_extent"):
        _build_env(cell_size=0.35)  # inconsistent with the 21x21 grid
    with pytest.raises(ValueError, match="maze_extent"):
        _build_env(  # right size, wrong grid
            maze_cells=DEFAULT_MAZE_CELLS - 1, cell_size=MAZE_EXTENT_M / _DEFAULT_GRID
        )

    # A consistent explicit value is accepted and honoured.
    explicit = _build_env(cell_size=MAZE_EXTENT_M / _DEFAULT_GRID)
    assert explicit.cell_size == pytest.approx(MAZE_EXTENT_M / _DEFAULT_GRID, abs=1e-12)
    assert explicit.maze_extent == pytest.approx(
        (MAZE_EXTENT_M, MAZE_EXTENT_M), abs=1e-9
    )


def test_non_positive_maze_extent_raises():
    with pytest.raises(ValueError, match="maze_extent must be > 0"):
        _build_env(maze_extent=0.0)


@pytest.mark.parametrize("maze_cells", PARAMETERISED_MAZE_CELLS)
def test_parameterised_maze_sizes_build_and_are_navigable(maze_cells):
    """maze_cells 8 / 10 / 12 all build, keep the 6.46 m arena, and are navigable.

    "Navigable" here is three concrete checks, not a vibe:

    1. every open grid cell is reachable from every other one (4-connectivity
       flood fill) -- a perfect maze is a spanning tree, so an island would
       mean the carve or the wall covering is broken;
    2. the *narrowest* corridor measured off the compiled box geoms still
       clears the rat's body width, so no corridor is a wall;
    3. the sampled spawn/treat cells all sit inside the arena footprint.

    Turning in place additionally wants a corridor wider than the rat is LONG
    (0.308 m measured).  Only maze_cells 4 and 5 clear that on their widest
    corridors; the narrowest segments (flanked by a multi-cell wall rectangle,
    which is not thinned on that axis) do not, at any of the three sizes.  That
    is reported, not asserted -- it is a design trade-off for the human picking
    a size from the renders, not a correctness property.
    """
    sized = _build_env(maze_cells=maze_cells)
    grid_size = _grid_size(maze_cells)

    # -- 6.46 m arena, derived pitch -------------------------------------
    assert sized.maze_grid.shape == (grid_size, grid_size)
    assert sized.maze_extent == pytest.approx(
        (MAZE_EXTENT_M, MAZE_EXTENT_M), abs=1e-9
    )
    assert sized.cell_size == pytest.approx(MAZE_EXTENT_M / grid_size, abs=1e-12)
    assert sized.corridor_width == pytest.approx(
        2.0 * sized.cell_size - float(sized._config.wall_thickness), abs=1e-12
    )

    # -- 1. connectivity --------------------------------------------------
    assert _free_cells_are_connected(sized.maze_grid)
    assert sized.n_free_cells == int(
        np.sum(sized.maze_grid != maze_utils.WALL_CHAR)
    )

    # -- 2. the rat fits down the narrowest corridor ----------------------
    body = _rodent_aabb(sized)
    body_width = float(body[1])
    gaps = _measured_one_cell_corridors(sized)
    assert gaps.size, "no one-cell corridor found"
    assert gaps.min() > body_width, (
        f"narrowest corridor {gaps.min():.4f} m does not clear the "
        f"{body_width:.4f} m body width"
    )
    assert gaps.max() == pytest.approx(sized.corridor_width, abs=2e-3)

    # -- 3. sampled cells are inside the footprint ------------------------
    half_x, half_y = sized.maze_half_extent
    free = sized.free_cell_positions
    assert np.all(np.abs(free[:, 0]) <= half_x + 1e-9)
    assert np.all(np.abs(free[:, 1]) <= half_y + 1e-9)

    state = jax.jit(sized.reset)(jax.random.PRNGKey(0))
    treats = _treat_xy(sized, state.data)
    spawn = _spawn_xy(sized, state.data)
    assert np.all(np.abs(treats[:, 0]) <= half_x) and np.all(
        np.abs(treats[:, 1]) <= half_y
    )
    assert abs(spawn[0]) <= half_x and abs(spawn[1]) <= half_y

    # -- reported, not asserted: exploration sparsity ---------------------
    print(
        f"maze_cells={maze_cells}: grid {grid_size}x{grid_size}, "
        f"cell {sized.cell_size:.6f} m, corridor {sized.corridor_width:.4f} m "
        f"(measured max {gaps.max():.4f}, min {gaps.min():.4f}), "
        f"{sized.n_free_cells} free cells, "
        f"open-cell fraction {sized.open_cell_fraction:.4f}, "
        f"treat/free-cell fraction {sized.treat_cell_fraction:.4f}"
    )


def test_treat_density_is_reported_by_the_env(env):
    """``treat_cell_fraction`` is the sparsity of the exploration problem.

    Not a threshold -- ``n_treats`` is a human decision that this pins the
    consequences of: at the shipped ``maze_cells=10`` / ``n_treats=20`` that is
    20/199 of the reachable cells, and it gets strictly sparser as
    ``maze_cells`` grows because the arena size is fixed.
    """
    assert env.n_free_cells == int(np.sum(env.maze_grid != maze_utils.WALL_CHAR))
    assert env.treat_cell_fraction == pytest.approx(
        env.n_treats / env.n_free_cells
    )
    assert env.open_cell_fraction == pytest.approx(
        env.n_free_cells / env.maze_grid.size
    )
    denser = _build_env(maze_cells=DEFAULT_MAZE_CELLS - 2)
    sparser = _build_env(maze_cells=DEFAULT_MAZE_CELLS + 2)
    assert (
        sparser.treat_cell_fraction
        < env.treat_cell_fraction
        < denser.treat_cell_fraction
    ), "treat density must fall as the maze gains cells at fixed arena size"
