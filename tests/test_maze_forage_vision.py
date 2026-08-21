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
``privileged_state``).

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

# --- Declared task_obs components (DESIGN.md 3e) ---------------------------
# task_obs = [prev_action, kinematic_sensors, touch_sensors, origin]
N_KINEMATIC_SENSORS = 9  # accelerometer(3) + velocimeter(3) + gyro(3)
N_TOUCH_SENSORS = 4  # consts.TOUCH_SENSORS
N_ORIGIN = 3  # origin in the torso frame

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
    """[prev_action, kinematic_sensors, touch, origin] and nothing else."""
    expected = (
        env.action_size + N_KINEMATIC_SENSORS + N_TOUCH_SENSORS + N_ORIGIN
    )
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


def test_entry_point_refuses_a_frozen_layout_config(env):
    """``train_highlvl`` must raise rather than launch a silently dead run."""
    from omegaconf import OmegaConf

    from vnl_playground import train_highlvl

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
    assert default_config().n_treats == 4
    assert MazeForageVision().n_treats == 4


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


def test_corridor_width_is_two_cells_minus_thickness(env):
    """``cell_size`` is the grid PITCH; thinning roughly doubles the clear width.

    ``_wall_box_geometry`` thins a one-cell-wide wall rectangle to
    ``wall_thickness``, handing ``cell_size - wall_thickness`` of floor back to
    the cell on each side, so a nominally one-cell corridor flanked by two
    thinned walls is really ``2 * cell_size - wall_thickness`` across -- 0.65 m
    at the defaults, not the 0.35 m the config comment used to claim.  Measured
    here by rasterising the compiled box geoms, so the docs (and
    ``corridor_width``) cannot drift away from the geometry.

    Corridors flanked by a wall that spans several cells along the scan axis
    (those are not thinned on that axis) are narrower; the invariant checked
    for them is only that thinning never makes a corridor *narrower* than one
    cell.
    """
    cell = float(env._config.cell_size)
    thickness = float(env._config.wall_thickness)
    expected = 2.0 * cell - thickness
    assert env.corridor_width == pytest.approx(expected)
    assert env.corridor_width > cell  # the docs used to claim == cell

    grid = env.maze_grid
    wall = grid == maze_utils.WALL_CHAR
    measured, both_thinned = [], []
    for col in range(grid.shape[1]):
        for row in range(grid.shape[0] - 2):
            # wall / free / wall in a column -> a one-cell corridor cell.
            if not (wall[row, col] and not wall[row + 1, col] and wall[row + 2, col]):
                continue
            x = maze_utils.grid_to_world((row, col), cell, grid.shape)[0]
            y_hi = maze_utils.grid_to_world((row, col), cell, grid.shape)[1]
            y_lo = maze_utils.grid_to_world((row + 2, col), cell, grid.shape)[1]
            gap = _free_run_along_y(env, x, y_lo, y_hi)
            measured.append(gap)
            near = _covering_rect(env, row, col)
            far = _covering_rect(env, row + 2, col)
            if (
                int(near.end.y) - int(near.start.y) == 1
                and int(far.end.y) - int(far.start.y) == 1
            ):
                both_thinned.append(gap)

    assert measured, "no one-cell corridor found in the maze"
    assert both_thinned, "no corridor flanked by two thinned walls"
    for gap in both_thinned:
        assert gap == pytest.approx(expected, abs=2e-3)
    assert max(measured) == pytest.approx(expected, abs=2e-3)
    assert min(measured) >= cell - 2e-3


# --- Collected treats must vanish on the step they stop paying -------------


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
    """
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    data = state.data
    root = env._rodent_root_qpos
    moved = mjx.forward(env.mjx_model, data.replace(qpos=data.qpos.at[root].add(dx)))
    before = np.asarray(env._get_obs(data, state.info)["state"]["task_obs"])
    after = np.asarray(env._get_obs(moved, state.info)["state"]["task_obs"])
    return float(np.linalg.norm(after - before))


def test_include_origin_false_drops_the_allocentric_fix(env):
    """``origin`` is an exact position+heading fix in a fixed maze; make it A/B-able.

    Per DESIGN.md 3e ``task_obs`` copies ``go_to_target_vision``'s contract, so
    ``origin`` stays on by default -- but in a maze that never changes it is a
    ground-truth place code, which confounds any "the CNN learned to localise"
    claim.  The switch exists so that can be ablated without editing the task.
    """
    no_origin = _build_env(include_origin=False)
    state = jax.jit(no_origin.reset)(jax.random.PRNGKey(0))
    expected = no_origin.action_size + N_KINEMATIC_SENSORS + N_TOUCH_SENSORS
    assert state.obs["state"]["task_obs"].shape == (expected,)
    assert int(no_origin.non_flattened_observation_size["state"]["task_obs"]) == (
        expected
    )
    assert int(env.non_flattened_observation_size["state"]["task_obs"]) == (
        expected + N_ORIGIN
    )

    # Translating the rodent moves task_obs by exactly the translation with
    # `origin` on, and by nothing (bar float noise) with it off.
    assert _task_obs_shift_when_translated(env, 0.5) == pytest.approx(0.5, abs=1e-3)
    assert _task_obs_shift_when_translated(no_origin, 0.5) < 1e-4


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
