"""Tests for the vision-guided Y/T-maze spout env (`RodentSpoutMazeVision`).

The env is warp-only (``mujoco_impl="warp"``), so everything except the
registry/config checks needs a CUDA GPU.  Every vision assertion renders through
``BinocularVisionRenderWrapper`` -- the wrapper the DMPO training stack puts around
the brax-wrapped env -- with the 32x32 grayscale binocular settings used for
training, never through a separate renderer.

Run serially::

    MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        python -m pytest tests/test_spout_maze.py -x -q
"""

import math
import types

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import pytest
from mujoco import mjx

from vnl_playground.tasks.rodent import maze_forage_vision, spout_maze
from vnl_playground.tasks.rodent.spout_geometry import locate
from vnl_playground.tasks.rodent.spout_logic import TrialState, critic_privileged_vector

_HAS_GPU = any(d.platform == "gpu" for d in jax.devices())
_skip_without_gpu = pytest.mark.skipif(
    not _HAS_GPU, reason="the warp env and its renderer need a CUDA GPU"
)


def needs_gpu(fn):
    """``gpu``-marked (selected by ``-m gpu``) and skipped when no GPU is visible."""
    return pytest.mark.gpu(_skip_without_gpu(fn))


N_ENVS = 4
EPISODE_LENGTH = 2000
# Maze-overview pixel test: a pixel "changes" when live minus treat-parked differs by more than 40/255,
# and a background pixel is "bright" (wall/floor, not the near-black warp sky)
# above 80/255.
CHANGE_THRESH = 40.0 / 255.0
BRIGHT_THRESH = 80.0 / 255.0
MIN_TREAT_PX = 6

# Vision settings used for training.
_TRAINING_VISION = dict(
    vision_width=32,
    vision_height=32,
    grayscale=True,
    binocular=True,
    use_textures=True,
    use_shadows=True,
    key_light_diffuse=0.30,
    wall_light_diffuse=0.25,
)

# Parent registry contents, pinned so a subclass decorating against the
# parent's module-level registry is caught.
_PARENT_REWARDS = {"treat_collected", "upright", "speed", "termination_penalty"}
_PARENT_TERMINATIONS = {
    "all_treats_collected",
    "belly_up",
    "torso_too_low",
    "fallen",
    "timeout",
    "nan_termination",
}


def _build_env(**overrides) -> spout_maze.RodentSpoutMazeVision:
    cfg = spout_maze.default_config()
    for key, value in {**_TRAINING_VISION, **overrides}.items():
        cfg[key] = value
    return spout_maze.RodentSpoutMazeVision(config=cfg)


def _keys(seed: int, n: int = N_ENVS) -> jax.Array:
    return jax.random.split(jax.random.PRNGKey(seed), n)


@pytest.fixture(scope="module")
def env():
    """Jitter-free env (the start view is exactly reproducible)."""
    if not _HAS_GPU:
        pytest.skip("needs a CUDA GPU")
    return _build_env(start_yaw_jitter_deg=0.0)


def _make_stack(env):
    """``env`` wrapped exactly like the DMPO training stack: brax (full_reset) + binocular vision."""
    from mujoco_playground._src import wrapper as mp_wrapper

    from vnl_playground.tasks.rodent.vision_jax import BinocularVisionRenderWrapper

    brax_env = mp_wrapper.wrap_for_brax_training(
        env, episode_length=EPISODE_LENGTH, action_repeat=1, full_reset=True
    )
    vis = BinocularVisionRenderWrapper(
        brax_env,
        mj_model=env.mj_model,
        mjx_model=env.mjx_model,
        width=32,
        height=32,
        grayscale=True,
        render_depth=False,
        use_textures=True,
        use_shadows=True,
        left_camera_name="eye_left-rodent",
        right_camera_name="eye_right-rodent",
    )
    forward = jax.jit(jax.vmap(lambda d: mjx.forward(env.mjx_model, d)))
    return types.SimpleNamespace(
        vis=vis,
        reset=jax.jit(vis.reset),
        step=jax.jit(vis.step),
        forward=forward,
        render=jax.jit(vis._render_binocular),
    )


@pytest.fixture(scope="module")
def stack(env):
    """``env`` wrapped exactly like the DMPO training stack: brax (full_reset) + binocular vision."""
    return _make_stack(env)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vision(state) -> np.ndarray:
    return np.asarray(state.obs["state"]["vision"])


def _task_obs(state) -> np.ndarray:
    return np.asarray(state.obs["state"]["task_obs"])


def _proprio(state) -> np.ndarray:
    from jax import flatten_util

    flat = jax.vmap(lambda p: flatten_util.ravel_pytree(p)[0])(
        state.obs["state"]["proprioception"]
    )
    return np.asarray(flat)


def _imitation_target(state) -> np.ndarray:
    """What the DMPO policy reads as ``imitation_target`` (KLAnchor wrapper path)."""
    from vnl_playground.tasks.wrappers_kl_anchor import flatten_obs_dict

    return np.asarray(flatten_obs_dict(state.obs)["imitation_target"])


def _torso_xy(env, state) -> np.ndarray:
    tid = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso-rodent")
    return np.asarray(state.data.xpos[:, tid, :2])


def _trial(state) -> dict:
    return {k: np.asarray(v) for k, v in state.info["trial"].items()}


def _yaw_quat(yaw: np.ndarray) -> np.ndarray:
    yaw = np.asarray(yaw, dtype=np.float32)
    return np.stack(
        [np.cos(yaw / 2), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2)], -1
    )


def _torso_offset_local(env, state) -> np.ndarray:
    """Torso xy minus root xy, in the rodent's heading frame (from a reset pose)."""
    root = env._rodent_root_qpos
    q = np.asarray(state.data.qpos[0])
    yaw = 2.0 * math.atan2(q[root + 6], q[root + 3])
    off = _torso_xy(env, state)[0] - q[root : root + 2]
    c, s = math.cos(-yaw), math.sin(-yaw)
    return np.array([c * off[0] - s * off[1], s * off[0] + c * off[1]])


def _pose(env, stack, state, torso_xy, yaw, bait=None):
    """Puts each env's torso at ``torso_xy`` facing ``yaw`` (standing joints, zero qvel)."""
    local = _torso_offset_local(env, state)
    yaw = np.asarray(yaw, dtype=np.float32)
    c, s = np.cos(yaw), np.sin(yaw)
    root_xy = np.asarray(torso_xy) - np.stack(
        [c * local[0] - s * local[1], s * local[0] + c * local[1]], -1
    )
    root = env._rodent_root_qpos
    qpos = state.data.qpos
    qpos = qpos.at[:, root : root + 2].set(jp.asarray(root_xy, jp.float32))
    qpos = qpos.at[:, root + 2].set(float(env._config.spawn_height))
    qpos = qpos.at[:, root + 3 : root + 7].set(jp.asarray(_yaw_quat(yaw)))
    data = state.data.replace(qpos=qpos, qvel=jp.zeros_like(state.data.qvel))
    data = stack.forward(data)
    info = dict(state.info)
    if bait is not None:
        trial = dict(info["trial"])
        trial["bait"] = jp.asarray(bait, jp.int32)
        info["trial"] = trial
    return state.replace(data=data, info=info)


def _arm_pose(layout, arm: int, dist_to_end: float):
    """Torso xy on arm ``arm``'s centreline ``dist_to_end`` before its end, facing out."""
    d = layout.arm_dir[arm]
    xy = layout.arm_end_xy[arm] - dist_to_end * d
    return xy, math.atan2(d[1], d[0])


def _treat_geom_z(env, state) -> np.ndarray:
    return np.asarray(state.data.geom_xpos[:, np.asarray(env._treat_geom_ids), 2])


# ---------------------------------------------------------------------------
# Registry and config (CPU)
# ---------------------------------------------------------------------------


def test_parent_registry_untouched():
    """The subclass registers against its OWN registry, never the parent's."""
    parent = maze_forage_vision._registry
    assert set(parent.rewards) == _PARENT_REWARDS
    assert set(parent.terminations) == _PARENT_TERMINATIONS
    assert spout_maze._registry is not parent
    assert spout_maze.RodentSpoutMazeVision._registry is spout_maze._registry
    assert maze_forage_vision.MazeForageVision._registry is parent
    assert set(spout_maze._registry.rewards) == {"spout_treat"}
    assert set(spout_maze._registry.terminations) == {
        "belly_up",
        "torso_too_low",
        "nan_termination",
    }


def test_registered_with_default_config():
    from vnl_playground import tasks

    assert tasks._envs["RodentSpoutMazeVision"] is spout_maze.RodentSpoutMazeVision
    cfg = tasks.get_default_config("RodentSpoutMazeVision")
    parent = maze_forage_vision.default_config()
    assert set(parent.keys()) <= set(cfg.keys())
    expected = dict(
        stem_length=0.9,
        arm_length=0.9,
        arm_angle_deg=120.0,
        corridor_width=0.465,
        appear_distance=0.5,
        spout_radius=0.15,
        treat_inset=0.15,
        n_arms=2,
        pay_prob=1.0,
        dwell_steps=30,
        trial_timeout_steps=800,
        start_yaw_jitter_deg=10.0,
        critic_priv_max_trials=20,
        n_treats=2,
        ctrl_dt=0.01,
        mujoco_impl="warp",
    )
    for key, value in expected.items():
        if isinstance(value, float):
            assert cfg[key] == pytest.approx(value), key
        else:
            assert cfg[key] == value, key
    assert cfg.corridor_width >= 0.465
    assert cfg.reward_terms.to_dict() == {"spout_treat": {"weight": 1.0}}
    term = cfg.termination_criteria.to_dict()
    parent_term = parent.termination_criteria.to_dict()
    assert term == {
        "belly_up": parent_term["belly_up"],
        "torso_too_low": parent_term["torso_too_low"],
        "nan_termination": {},
    }
    # The parent's defaults are not mutated by building the subclass config.
    assert parent.n_treats == 20
    assert "spout_treat" not in parent.reward_terms


# ---------------------------------------------------------------------------
# Build / reset / rollout
# ---------------------------------------------------------------------------


@needs_gpu
def test_build_and_reset(env, stack):
    layout = env.layout
    model = env.mj_model
    # Rotated wall boxes from the Y layout, nothing from the grid maze.
    wall_ids = [
        i
        for i in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith(
            "maze_wall_"
        )
    ]
    assert len(wall_ids) == len(layout.wall_pos)

    state = stack.reset(_keys(0))
    # Wall boxes sit where the layout says, rotated by wall_yaw (radians).
    gx = np.asarray(state.data.geom_xpos[0, wall_ids])
    gmat = np.asarray(state.data.geom_xmat[0, wall_ids]).reshape(-1, 3, 3)
    np.testing.assert_allclose(gx[:, :2], layout.wall_pos[:, :2], atol=1e-5)
    np.testing.assert_allclose(gmat[:, 0, 0], np.cos(layout.wall_yaw), atol=1e-5)
    np.testing.assert_allclose(gmat[:, 1, 0], np.sin(layout.wall_yaw), atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(model.geom_size[wall_ids]), layout.wall_halfsize, atol=1e-6
    )

    # The torso starts in the stem, near start_xy, facing the junction (+y).
    torso = _torso_xy(env, state)
    assert np.all(np.linalg.norm(torso - layout.start_xy, axis=-1) < 0.06)
    skull = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "skull-rodent")
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso-rodent")
    heading = np.array(state.data.xpos[:, skull, :2] - state.data.xpos[:, tid, :2])
    heading /= np.linalg.norm(heading, axis=-1, keepdims=True)
    assert np.all(heading[:, 1] > 0.99), heading
    arm_id, _ = locate(layout, jp.asarray(torso))
    assert np.all(np.asarray(arm_id) == -1)

    # Both treats parked underground, on their spouts' xy.
    assert np.all(_treat_geom_z(env, state) < -0.5)
    treat_xy = np.asarray(state.data.xpos[:, np.asarray(env._treat_body_ids), :2])
    np.testing.assert_allclose(
        treat_xy, np.broadcast_to(layout.treat_xy, treat_xy.shape), atol=1e-5
    )
    assert np.all(np.asarray(state.info["collected"]))

    # Fresh trial state and the critic vector.
    trial = _trial(state)
    assert set(trial["bait"]) <= {0, 1}
    assert np.all(trial["choice"] == -1) and np.all(trial["trial_idx"] == 0)
    priv = np.asarray(state.info["critic_priv"])
    assert priv.shape == (N_ENVS, 8)
    np.testing.assert_allclose(
        priv,
        np.asarray(critic_privileged_vector(TrialState(**state.info["trial"]), 20)),
    )
    assert np.all(np.asarray(state.done) == 0)
    assert _vision(state).shape == (N_ENVS, 32, 32, 2)
    assert np.all(np.isfinite(_vision(state))) and np.any(_vision(state) > 0)
    for key in ("spout/" + k for k in spout_maze.SPOUT_METRICS):
        assert key in state.metrics, key
    assert "rewards/treat_collected" in state.metrics


@needs_gpu
def test_start_yaw_jitter():
    jit_env = _build_env(start_yaw_jitter_deg=10.0)
    state = jax.jit(jax.vmap(jit_env.reset))(_keys(3, 64))
    root = jit_env._rodent_root_qpos
    q = np.asarray(state.data.qpos)
    yaw = 2.0 * np.arctan2(q[:, root + 6], q[:, root + 3])
    dev = np.rad2deg(yaw - math.pi / 2)
    assert np.all(np.abs(dev) <= 10.0 + 1e-3)
    assert dev.std() > 2.0
    np.testing.assert_allclose(
        q[:, root : root + 2],
        np.broadcast_to(jit_env.layout.start_xy, (64, 2)),
        atol=1e-6,
    )


@needs_gpu
def test_random_action_rollout_no_nan(env, stack):
    state = stack.reset(_keys(1))
    key = jax.random.PRNGKey(11)
    for _ in range(300):
        key, sub = jax.random.split(key)
        action = jax.random.uniform(
            sub, (N_ENVS, env.action_size), minval=-1.0, maxval=1.0
        )
        state = stack.step(state, action)
        assert np.all(np.isfinite(np.asarray(state.data.qpos)))
        assert np.all(np.isfinite(np.asarray(state.reward)))
    for leaf in jax.tree.leaves(state.obs):
        assert np.all(np.isfinite(np.asarray(leaf)))
    assert np.all(np.isfinite(np.asarray(state.info["critic_priv"])))
    trial = _trial(state)
    assert np.all((trial["choice"] >= -1) & (trial["choice"] <= 1))


# ---------------------------------------------------------------------------
# Teleport (forced trial end on step 5)
# ---------------------------------------------------------------------------


def _forced_teleport_run(env, stack):
    """Four random-action steps, a forced trial end on step 5, then one more step."""
    keys = _keys(7)
    state = stack.reset(keys)
    key = jax.random.PRNGKey(5)
    actions = []
    for _ in range(4):
        key, sub = jax.random.split(key)
        actions.append(
            jax.random.uniform(sub, (N_ENVS, env.action_size), minval=-1.0, maxval=1.0)
        )
        state = stack.step(state, actions[-1])
    s4 = state
    key, sub = jax.random.split(key)
    a5 = jax.random.uniform(sub, (N_ENVS, env.action_size), minval=-1.0, maxval=1.0)
    env._force_trial_end = True
    try:
        # A fresh function object, so jit re-traces with the hook on.
        forced = jax.jit(lambda s, a: stack.vis.step(s, a))
        s5 = forced(s4, a5)
    finally:
        env._force_trial_end = False
    key, sub = jax.random.split(key)
    a6 = jax.random.uniform(sub, (N_ENVS, env.action_size), minval=-1.0, maxval=1.0)
    s6 = stack.step(s5, a6)
    fresh = stack.reset(keys)  # same keys -> same bait
    return types.SimpleNamespace(s4=s4, s5=s5, s6=s6, fresh=fresh, a5=a5, a6=a6)


@pytest.fixture(scope="module")
def teleport_run(env, stack):
    """Four random-action steps, a forced trial end on step 5, then one more step."""
    return _forced_teleport_run(env, stack)


@needs_gpu
def test_teleport_obs_equals_reset_obs(env, teleport_run):
    r = teleport_run
    np.testing.assert_array_equal(_trial(r.s5)["bait"], _trial(r.fresh)["bait"])
    # Not vacuous: just before the teleport the obs is not the start obs.
    assert not np.allclose(_task_obs(r.s4), _task_obs(r.fresh))

    v5, vf = _vision(r.s5), _vision(r.fresh)
    equal = np.isclose(v5, vf, atol=1e-6)
    assert equal.mean() >= 0.99, (
        f"only {equal.mean():.4f} of pixels match a fresh reset"
    )
    for e in range(N_ENVS):
        for eye in range(2):
            assert equal[e, ..., eye].mean() >= 0.99, (
                e,
                eye,
                equal[e, ..., eye].mean(),
            )

    np.testing.assert_allclose(
        _imitation_target(r.s5), _imitation_target(r.fresh), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        _task_obs(r.s5), _task_obs(r.fresh), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(_proprio(r.s5), _proprio(r.fresh), rtol=1e-4, atol=1e-4)
    # The simulation inputs forward() saw are exactly a fresh reset's.  (On this
    # warp build forward() leaves actuator_force at 0 for any ctrl, so a stale
    # ctrl would not show in the obs; pin the inputs themselves.)
    for name in ("qpos", "qvel", "ctrl", "act", "qacc_warmstart"):
        np.testing.assert_array_equal(
            np.asarray(getattr(r.s5.data, name)),
            np.asarray(getattr(r.fresh.data, name)),
            name,
        )


@needs_gpu
def test_teleport_zeroes_action_history(env, teleport_run):
    r = teleport_run
    nu = env.action_size
    assert np.any(_task_obs(r.s4)[:, :nu] != 0.0)
    assert np.all(np.asarray(r.s5.info["action"]) == 0.0)
    assert np.all(np.asarray(r.s5.info["prev_action"]) == 0.0)
    assert np.all(_task_obs(r.s5)[:, :nu] == 0.0)
    prev = np.asarray(r.s5.obs["state"]["proprioception"]["prev_action"])
    assert np.all(prev == 0.0)
    # The step after the teleport sees a zero previous action.
    assert np.all(np.asarray(r.s6.info["prev_action"]) == 0.0)
    np.testing.assert_array_equal(np.asarray(r.s6.info["action"]), np.asarray(r.a6))


@needs_gpu
def test_teleport_keeps_episode(env, teleport_run):
    r = teleport_run
    t4, t5 = _trial(r.s4), _trial(r.s5)
    assert np.all(np.asarray(r.s4.done) == 0)
    assert np.all(np.asarray(r.s5.done) == 0)
    np.testing.assert_array_equal(t5["trial_idx"], t4["trial_idx"] + 1)
    np.testing.assert_array_equal(t5["bait"], t4["bait"])
    assert np.all(t5["choice"] == -1) and np.all(t5["trial_steps"] == 0)
    assert np.all(np.asarray(r.s5.metrics["spout/trial_end"]) == 1.0)
    # The episode runs on: the env and wrapper step counters keep counting.
    np.testing.assert_array_equal(
        np.asarray(r.s5.info["step_count"]), np.asarray(r.s4.info["step_count"]) + 1
    )
    np.testing.assert_array_equal(np.asarray(r.s5.info["steps"]), 5)
    np.testing.assert_allclose(
        np.asarray(r.s5.info["critic_priv"]),
        np.asarray(critic_privileged_vector(TrialState(**r.s5.info["trial"]), 20)),
    )
    np.testing.assert_allclose(np.asarray(r.s5.info["critic_priv"])[:, 6], 1.0 / 20)
    # Back at the start, both treats parked.
    np.testing.assert_allclose(_torso_xy(env, r.s5), _torso_xy(env, r.fresh), atol=1e-5)
    assert np.all(_treat_geom_z(env, r.s5) < -0.5)


# ---------------------------------------------------------------------------
# Trial flow through the real (unforced) path
# ---------------------------------------------------------------------------


@needs_gpu
def test_eat_and_visit_end_trials_after_the_dwell(env, stack):
    """Bait arm: lock + show + eat -> +1; other arm: visit -> 0.  Both end D calls later."""
    # The default (flag off): an eaten treat is parked on the step it pays, below.  With
    # treat_visible_until_teleport it stays through the dwell (test_eaten_treat_stays_visible_...).
    assert not env._config.get("treat_visible_until_teleport", False)
    layout = env.layout
    dwell = int(env._config.dwell_steps)
    state = stack.reset(_keys(2))
    poses = [
        _arm_pose(layout, 0, float(env._config.treat_inset)),  # on treat 0
        _arm_pose(layout, 1, float(env._config.treat_inset)),  # on treat 1
        _arm_pose(layout, 0, 0.05),  # arm 0 end, bait elsewhere
        _arm_pose(layout, 1, 0.05),  # arm 1 end, bait elsewhere
    ]
    state = _pose(
        env,
        stack,
        state,
        np.stack([p[0] for p in poses]),
        np.array([p[1] for p in poses]),
        bait=np.array([0, 1, 1, 0]),
    )
    zeros = jp.zeros((N_ENVS, env.action_size))

    state = stack.step(state, zeros)
    np.testing.assert_array_equal(np.asarray(state.reward), [1.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(
        np.asarray(state.metrics["rewards/treat_collected"]), [1, 1, 0, 0]
    )
    np.testing.assert_array_equal(
        np.asarray(state.metrics["spout/show_treat"]), [1, 1, 0, 0]
    )
    np.testing.assert_array_equal(np.asarray(state.info["n_collected"]), [1, 1, 0, 0])
    trial = _trial(state)
    np.testing.assert_array_equal(trial["choice"], [0, 1, 0, 1])
    np.testing.assert_array_equal(trial["dwell"], dwell - 1)
    # An eaten treat is parked on the step it pays.
    assert np.all(_treat_geom_z(env, state) < -0.5)

    for call in range(2, dwell + 1):
        state = stack.step(state, zeros)
        assert np.all(np.asarray(state.reward) == 0.0), call
        ended = np.asarray(state.metrics["spout/trial_end"])
        if call < dwell:
            assert np.all(ended == 0.0), call
    assert np.all(ended == 1.0)
    np.testing.assert_array_equal(np.asarray(state.metrics["spout/hit"]), [1, 1, 0, 0])
    assert np.all(np.asarray(state.done) == 0)
    trial = _trial(state)
    np.testing.assert_array_equal(trial["trial_idx"], 1)
    np.testing.assert_array_equal(trial["hits"], [1, 1, 0, 0])
    # Teleported back to the start with the treats parked.
    assert np.all(
        np.linalg.norm(_torso_xy(env, state) - layout.start_xy, axis=-1) < 0.06
    )
    assert np.all(_treat_geom_z(env, state) < -0.5)
    assert np.all(np.asarray(state.info["action"]) == 0.0)


# ---------------------------------------------------------------------------
# Spike S5: the treat is visible, against the end wall, from the appear zone
# ---------------------------------------------------------------------------


def _treat_pixel_counts(live: np.ndarray, parked: np.ndarray):
    """Per env and eye: changed pixels, and changed pixels whose background is bright."""
    changed = np.abs(live - parked) > CHANGE_THRESH
    on_wall = changed & (parked > BRIGHT_THRESH)
    return changed.sum(axis=(1, 2)), on_wall.sum(axis=(1, 2))


@needs_gpu
def test_treat_visible_from_appear_zone(env, stack):
    layout = env.layout
    boundary = float(env._config.appear_distance) - 0.01
    state = stack.reset(_keys(4))
    poses = [_arm_pose(layout, arm, boundary) for arm in (0, 1, 0, 1)]
    state = _pose(
        env,
        stack,
        state,
        np.stack([p[0] for p in poses]),
        np.array([p[1] for p in poses]),
        bait=np.array([0, 1, 1, 0]),  # envs 2/3 face the unbaited arm
    )
    state = stack.step(state, jp.zeros((N_ENVS, env.action_size)))
    np.testing.assert_array_equal(
        np.asarray(state.metrics["spout/show_treat"]), [1, 1, 0, 0]
    )
    np.testing.assert_array_equal(_trial(state)["choice"], [0, 1, 0, 1])
    # Shown = unparked at treat height; the unbaited arm shows nothing.
    z = _treat_geom_z(env, state)
    th = float(env._config.treat_height)
    np.testing.assert_allclose([z[0, 0], z[1, 1]], th, atol=1e-4)
    assert z[0, 1] < -0.5 and z[1, 0] < -0.5 and np.all(z[2:] < -0.5)

    live = _vision(state)
    all_parked = jax.vmap(env._park_collected_treats)(
        state.data, jp.ones((N_ENVS, env.n_treats), dtype=bool)
    )
    parked = np.asarray(stack.render(all_parked))
    changed, on_wall = _treat_pixel_counts(live, parked)
    for e in (0, 1):
        for eye in (0, 1):
            assert changed[e, eye] >= MIN_TREAT_PX, (e, eye, changed[e])
            assert on_wall[e, eye] >= MIN_TREAT_PX, (e, eye, on_wall[e])
    # Nothing was shown in envs 2/3, so parking changes nothing there.
    assert np.all(changed[2:] == 0), changed


# ---------------------------------------------------------------------------
# T maze (maze_shape="T", a mouse T maze at x2.5)
# ---------------------------------------------------------------------------

_T_OVERRIDES = dict(maze_shape="T", appear_distance=0.70)
_T_DEFAULTS = dict(
    t_corridor_width=0.25,
    t_segment_length=0.50,
    t_n_stem_segments=5,
    t_arm_length=0.6875,
    t_spout_offset=0.10,
    t_wall_thickness=0.05,
    t_entry_margin=0.03,
)


def test_maze_shape_config_defaults():
    cfg = spout_maze.default_config()
    assert cfg.maze_shape == "Y"
    for key, value in _T_DEFAULTS.items():
        assert cfg[key] == pytest.approx(value), key
    assert isinstance(cfg.t_n_stem_segments, int)


def test_invalid_maze_shape_raises():
    cfg = spout_maze.default_config()
    cfg.maze_shape = "L"
    with pytest.raises(ValueError, match="maze_shape"):
        spout_maze.RodentSpoutMazeVision(config=cfg)


@pytest.fixture(scope="module")
def t_env():
    """Jitter-free T-maze env."""
    if not _HAS_GPU:
        pytest.skip("needs a CUDA GPU")
    return _build_env(start_yaw_jitter_deg=0.0, **_T_OVERRIDES)


@pytest.fixture(scope="module")
def t_stack(t_env):
    return _make_stack(t_env)


@pytest.fixture(scope="module")
def t_teleport_run(t_env, t_stack):
    return _forced_teleport_run(t_env, t_stack)


def _segment(state) -> np.ndarray:
    seg = state.info["segment"]
    assert seg.dtype == jp.int32
    return np.asarray(seg)


@needs_gpu
def test_t_build_and_reset(t_env, t_stack):
    from vnl_playground.tasks.rodent.spout_geometry import (
        TMazeSpec,
        build_t_maze,
        segment_of,
    )

    layout = t_env.layout
    model = t_env.mj_model
    expected = build_t_maze(TMazeSpec(wall_height=float(t_env._config.wall_height)))
    for name in (
        "wall_pos",
        "wall_halfsize",
        "wall_yaw",
        "start_xy",
        "arm_dir",
        "arm_end_xy",
        "treat_xy",
    ):
        np.testing.assert_array_equal(
            getattr(layout, name), getattr(expected, name), name
        )
    assert t_env.corridor_width == pytest.approx(0.25)

    wall_ids = [
        i
        for i in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith(
            "maze_wall_"
        )
    ]
    assert len(wall_ids) == len(layout.wall_pos) == 8

    state = t_stack.reset(_keys(0))
    gx = np.asarray(state.data.geom_xpos[0, wall_ids])
    gmat = np.asarray(state.data.geom_xmat[0, wall_ids]).reshape(-1, 3, 3)
    np.testing.assert_allclose(gx[:, :2], layout.wall_pos[:, :2], atol=1e-5)
    np.testing.assert_allclose(gmat[:, 0, 0], np.cos(layout.wall_yaw), atol=1e-5)
    np.testing.assert_allclose(gmat[:, 1, 0], np.sin(layout.wall_yaw), atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(model.geom_size[wall_ids]), layout.wall_halfsize, atol=1e-6
    )

    # Torso near the start (centre of the `start` segment), facing the junction (+y).
    torso = _torso_xy(t_env, state)
    np.testing.assert_allclose(layout.start_xy, [0.0, -2.375], atol=1e-12)
    assert np.all(np.linalg.norm(torso - layout.start_xy, axis=-1) < 0.06)
    skull = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "skull-rodent")
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso-rodent")
    heading = np.array(state.data.xpos[:, skull, :2] - state.data.xpos[:, tid, :2])
    heading /= np.linalg.norm(heading, axis=-1, keepdims=True)
    assert np.all(heading[:, 1] > 0.99), heading
    arm_id, _ = locate(layout, jp.asarray(torso))
    assert np.all(np.asarray(arm_id) == -1)
    assert np.all(np.asarray(segment_of(layout, jp.asarray(torso))) == 0)
    np.testing.assert_array_equal(_segment(state), 0)

    # Both treats parked underground, on their spout points (+-0.7125, 0).
    assert np.all(_treat_geom_z(t_env, state) < -0.5)
    treat_xy = np.asarray(state.data.xpos[:, np.asarray(t_env._treat_body_ids), :2])
    np.testing.assert_allclose(
        treat_xy,
        np.broadcast_to([[-0.7125, 0.0], [0.7125, 0.0]], treat_xy.shape),
        atol=1e-5,
    )
    assert np.all(np.asarray(state.info["collected"]))

    # Lights follow layout.extent (ring at radius 4 * extent, height 2 * extent).
    for i in range(int(t_env._config.wall_lights)):
        lid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, f"wall_light_{i}")
        assert model.light_pos[lid][2] == pytest.approx(2.0 * layout.extent, rel=1e-6)

    assert _vision(state).shape == (N_ENVS, 32, 32, 2)
    assert np.all(np.isfinite(_vision(state))) and np.any(_vision(state) > 0)


@needs_gpu
def test_t_random_action_rollout_no_nan(t_env, t_stack):
    state = t_stack.reset(_keys(1))
    key = jax.random.PRNGKey(11)
    for _ in range(300):
        key, sub = jax.random.split(key)
        action = jax.random.uniform(
            sub, (N_ENVS, t_env.action_size), minval=-1.0, maxval=1.0
        )
        state = t_stack.step(state, action)
        assert np.all(np.isfinite(np.asarray(state.data.qpos)))
        assert np.all(np.isfinite(np.asarray(state.reward)))
        seg = _segment(state)
        assert np.all((seg >= -1) & (seg <= 7))
    for leaf in jax.tree.leaves(state.obs):
        assert np.all(np.isfinite(np.asarray(leaf)))
    assert np.all(np.isfinite(np.asarray(state.info["critic_priv"])))
    trial = _trial(state)
    assert np.all((trial["choice"] >= -1) & (trial["choice"] <= 1))


@needs_gpu
def test_t_teleport_obs_equals_reset_obs(t_env, t_teleport_run):
    r = t_teleport_run
    np.testing.assert_array_equal(_trial(r.s5)["bait"], _trial(r.fresh)["bait"])
    assert not np.allclose(_task_obs(r.s4), _task_obs(r.fresh))
    assert np.all(np.asarray(r.s5.metrics["spout/trial_end"]) == 1.0)

    v5, vf = _vision(r.s5), _vision(r.fresh)
    equal = np.isclose(v5, vf, atol=1e-6)
    for e in range(N_ENVS):
        for eye in range(2):
            assert equal[e, ..., eye].mean() >= 0.99, (
                e,
                eye,
                equal[e, ..., eye].mean(),
            )
    np.testing.assert_allclose(
        _imitation_target(r.s5), _imitation_target(r.fresh), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        _task_obs(r.s5), _task_obs(r.fresh), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(_proprio(r.s5), _proprio(r.fresh), rtol=1e-4, atol=1e-4)
    for name in ("qpos", "qvel", "ctrl", "act", "qacc_warmstart"):
        np.testing.assert_array_equal(
            np.asarray(getattr(r.s5.data, name)),
            np.asarray(getattr(r.fresh.data, name)),
            name,
        )
    np.testing.assert_array_equal(_segment(r.s5), 0)
    np.testing.assert_array_equal(_segment(r.fresh), 0)


@needs_gpu
def test_t_segment_is_post_teleport(t_env, t_stack):
    """``info["segment"]`` reads the torso AFTER the teleport: arms (6, 7) before, start (0) after."""
    layout = t_env.layout
    state = t_stack.reset(_keys(9))
    poses = [_arm_pose(layout, arm, 0.30) for arm in (0, 1, 0, 1)]
    state = _pose(
        t_env,
        t_stack,
        state,
        np.stack([p[0] for p in poses]),
        np.array([p[1] for p in poses]),
    )
    zeros = jp.zeros((N_ENVS, t_env.action_size))
    stepped = t_stack.step(state, zeros)
    np.testing.assert_array_equal(_segment(stepped), [6, 7, 6, 7])
    t_env._force_trial_end = True
    try:
        forced = jax.jit(lambda s, a: t_stack.vis.step(s, a))
        teleported = forced(state, zeros)
    finally:
        t_env._force_trial_end = False
    assert np.all(np.asarray(teleported.metrics["spout/trial_end"]) == 1.0)
    np.testing.assert_array_equal(_segment(teleported), 0)


@needs_gpu
@pytest.mark.parametrize("maze_shape", ["Y", "T"])
def test_info_segment_at_reset(request, maze_shape):
    """``info["segment"]``: int32, 0 (``start``) at reset in the T, -1 everywhere in the Y."""
    stack_ = request.getfixturevalue("stack" if maze_shape == "Y" else "t_stack")
    state = stack_.reset(_keys(12))
    np.testing.assert_array_equal(_segment(state), 0 if maze_shape == "T" else -1)
    state = stack_.step(
        state, jp.zeros((N_ENVS, request.getfixturevalue("env").action_size))
    )
    np.testing.assert_array_equal(_segment(state), 0 if maze_shape == "T" else -1)


@needs_gpu
def test_t_treat_visible_from_arm(t_env, t_stack):
    """Rat 0.30 m from the spout, facing it: the shown treat changes >= 6 pixels in BOTH eyes."""
    layout = t_env.layout
    state = t_stack.reset(_keys(4))
    poses = [_arm_pose(layout, arm, 0.30) for arm in (0, 1, 0, 1)]
    state = _pose(
        t_env,
        t_stack,
        state,
        np.stack([p[0] for p in poses]),
        np.array([p[1] for p in poses]),
        bait=np.array([0, 1, 1, 0]),  # envs 2/3 face the unbaited arm
    )
    state = t_stack.step(state, jp.zeros((N_ENVS, t_env.action_size)))
    assert t_env.maze_shape == "T"
    np.testing.assert_array_equal(_segment(state), [6, 7, 6, 7])  # in the T's arms
    # The choice locks on the first in-arm step (appear_distance 0.70 >= every in-arm distance).
    np.testing.assert_array_equal(
        np.asarray(state.metrics["spout/show_treat"]), [1, 1, 0, 0]
    )
    np.testing.assert_array_equal(_trial(state)["choice"], [0, 1, 0, 1])
    z = _treat_geom_z(t_env, state)
    th = float(t_env._config.treat_height)
    np.testing.assert_allclose([z[0, 0], z[1, 1]], th, atol=1e-4)
    assert z[0, 1] < -0.5 and z[1, 0] < -0.5 and np.all(z[2:] < -0.5)

    live = _vision(state)
    all_parked = jax.vmap(t_env._park_collected_treats)(
        state.data, jp.ones((N_ENVS, t_env.n_treats), dtype=bool)
    )
    parked = np.asarray(t_stack.render(all_parked))
    changed, on_wall = _treat_pixel_counts(live, parked)
    print(
        f"T treat pixels at 0.30 m: changed={changed.tolist()} on_bright={on_wall.tolist()}"
    )
    for e in (0, 1):
        for eye in (0, 1):
            assert changed[e, eye] >= MIN_TREAT_PX, (e, eye, changed[e])
    assert np.all(changed[2:] == 0), changed


# ---------------------------------------------------------------------------
# Inter-trial hold + striped start gate (hold_enabled, T maze only)
# ---------------------------------------------------------------------------

_HOLD_DEFAULTS = dict(
    hold_enabled=False,
    hold_min_s=1.0,
    hold_max_s=3.0,
    gate_boundaries=(1,),
    gate_open_steps=20,
    gate_thickness=0.03,
    gate_bar_width=0.05,
)
_HOLD_OVERRIDES = dict(_T_OVERRIDES, hold_enabled=True)
# Boundary 1 (start | seg1) of the x2.5 T maze: -2.625 + 0.5 * 1.
_GATE1_Y = -2.125
# The closed gate must change at least this many pixels of each 32x32 eye from the start pose.
MIN_GATE_PX = 50


def test_hold_config_defaults():
    cfg = spout_maze.default_config()
    for key, value in _HOLD_DEFAULTS.items():
        if isinstance(value, float):
            assert cfg[key] == pytest.approx(value), key
        else:
            assert cfg[key] == value, key
    assert isinstance(cfg.gate_open_steps, int)


def test_treat_visible_until_teleport_default_off():
    """The eaten treat stays visible through the dwell only when asked for (default off)."""
    cfg = spout_maze.default_config()
    assert cfg.treat_visible_until_teleport is False


@pytest.mark.parametrize(
    "overrides, match",
    [
        (dict(hold_enabled=True), "maze_shape"),  # the default Y
        (dict(_HOLD_OVERRIDES, gate_boundaries=(0,)), "gate_boundaries"),
        (dict(_HOLD_OVERRIDES, gate_boundaries=(6,)), "gate_boundaries"),
        (dict(_HOLD_OVERRIDES, hold_min_s=2.0, hold_max_s=1.0), "hold_max_s"),
        (dict(_HOLD_OVERRIDES, gate_bar_width=0.25), "gate_bar_width"),
        (dict(_HOLD_OVERRIDES, gate_thickness=0.0), "gate_thickness"),
    ],
)
def test_hold_invalid_config_raises(overrides, match):
    """Raised at build time, before the model is compiled (so no GPU needed)."""
    cfg = spout_maze.default_config()
    for key, value in overrides.items():
        cfg[key] = value
    with pytest.raises(ValueError, match=match):
        spout_maze.RodentSpoutMazeVision(config=cfg)


@pytest.fixture(scope="module")
def hold_env():
    """Jitter-free T maze with the hold on (default 1-3 s hold, gate at boundary 1) and
    ``treat_visible_until_teleport`` on (no hold test reads it)."""
    if not _HAS_GPU:
        pytest.skip("needs a CUDA GPU")
    return _build_env(
        start_yaw_jitter_deg=0.0,
        **dict(_HOLD_OVERRIDES, treat_visible_until_teleport=True),
    )


@pytest.fixture(scope="module")
def hold_stack(hold_env):
    return _make_stack(hold_env)


def _hold(state) -> dict:
    return {k: np.asarray(v) for k, v in state.info["hold"].items()}


def _set_hold(state, **fields):
    """``state`` with ``info["hold"][k] = v`` (broadcast over the envs)."""
    info = dict(state.info)
    hold = dict(info["hold"])
    for key, value in fields.items():
        hold[key] = jp.broadcast_to(jp.asarray(value, jp.int32), (N_ENVS,))
    info["hold"] = hold
    return state.replace(info=info)


def _gate_body_xpos(env, state) -> np.ndarray:
    return np.asarray(state.data.xpos[:, env._gate_body_id])


def _gate_geom_top_z(env, state) -> np.ndarray:
    """Highest point of every gate geom (all boxes are axis-aligned), ``(n_env, n_geom)``."""
    half_h = np.asarray(env.mj_model.geom_size[np.asarray(env._gate_geom_ids), 2])
    return (
        np.asarray(state.data.geom_xpos[:, np.asarray(env._gate_geom_ids), 2]) + half_h
    )


def _with_gate_at(env, data, z):
    """``data`` with every env's gate moved to height offset ``z`` at its own boundary."""
    y = data.qpos[:, env._gate_qpos_idxs[0]]
    return jax.vmap(lambda d, yy: env._place_gate(d, yy, jp.float32(z)))(data, y)


def _gate_pixel_stats(live: np.ndarray, gateless: np.ndarray):
    """Per env and eye: changed pixels, stripe contrast and horizontal structure of the gate.

    The gate region is where the live frame differs from the gate-parked frame
    by more than ``CHANGE_THRESH``.  Inside it, per image row with >= 4 gate
    pixels: the horizontal std of the live intensity, and the number of
    bright/dark alternations (bar edges) along the row.

    Returns:
        dict of ``(n_env, 2)`` arrays: ``changed``, ``contrast`` (p90 - p10 of
        the live gate pixels), ``row_std`` (median over rows) and ``edges``
        (max over rows).
    """
    changed = np.abs(live - gateless) > CHANGE_THRESH
    n_env, _, _, n_eye = live.shape
    out = {
        k: np.zeros((n_env, n_eye)) for k in ("changed", "contrast", "row_std", "edges")
    }
    for e in range(n_env):
        for eye in range(n_eye):
            mask = changed[e, ..., eye]
            img = live[e, ..., eye]
            out["changed"][e, eye] = mask.sum()
            if mask.sum() < 4:
                continue
            vals = img[mask]
            out["contrast"][e, eye] = np.percentile(vals, 90) - np.percentile(vals, 10)
            mid = 0.5 * (np.percentile(vals, 90) + np.percentile(vals, 10))
            stds, edges = [], []
            for r in range(img.shape[0]):
                row = img[r][mask[r]]
                if row.size < 4:
                    continue
                stds.append(row.std())
                level = row > mid
                edges.append(int(np.count_nonzero(level[1:] != level[:-1])))
            out["row_std"][e, eye] = np.median(stds) if stds else 0.0
            out["edges"][e, eye] = max(edges) if edges else 0
    return out


def _drive_forward(env, stack, state, n, speed=0.5):
    """``n`` zero-action steps; before each one the root's world +y velocity is raised to >= ``speed``.

    Returns the final state and per-step ``(torso_y, done, states)`` histories.  ``states`` holds
    host (numpy) copies: 100 device-resident states grow the process's GPU pool by ~0.9 GB, and
    the suite is already close to the warp per-process allocation limit.
    """
    dof = env._rodent_root_dof
    zeros = jp.zeros((N_ENVS, env.action_size))
    torso_y, done, states = [], [], []
    for _ in range(n):
        qvel = state.data.qvel
        qvel = qvel.at[:, dof + 1].set(jp.maximum(qvel[:, dof + 1], speed))
        state = state.replace(data=state.data.replace(qvel=qvel))
        state = stack.step(state, zeros)
        torso_y.append(_torso_xy(env, state)[:, 1])
        done.append(np.asarray(state.done))
        states.append(jax.device_get(state))
    return state, np.stack(torso_y), np.stack(done), states


def _pose_behind_gate(env, stack, state, gap):
    """Torso on the stem centreline, ``gap`` m behind the gate's start-side face, facing +y."""
    face = _GATE1_Y - 0.5 * float(env._config.gate_thickness)
    xy = np.tile([0.0, face - gap], (N_ENVS, 1))
    return _pose(env, stack, state, xy, np.full(N_ENVS, math.pi / 2))


@needs_gpu
def test_hold_build_and_reset(hold_env, hold_stack, t_env, t_stack):
    env, model = hold_env, hold_env.mj_model
    assert env.hold_enabled and not t_env.hold_enabled
    assert mujoco.mj_name2id(t_env.mj_model, mujoco.mjtObj.mjOBJ_BODY, "gate") == -1

    # --- Model: one gate body on y/z slide joints, a colliding box, a striped non-colliding face ---
    bid = env._gate_body_id
    assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) == "gate"
    assert model.body_parentid[bid] == 0
    jnt = [j for j in range(model.njnt) if model.jnt_bodyid[j] == bid]
    assert [int(model.jnt_type[j]) for j in jnt] == [
        int(mujoco.mjtJoint.mjJNT_SLIDE)
    ] * 2
    np.testing.assert_array_equal(model.jnt_axis[jnt], [[0, 1, 0], [0, 0, 1]])
    box = env._gate_box_geom_id
    np.testing.assert_allclose(model.geom_size[box], [0.125, 0.015, 0.15], atol=1e-9)
    assert model.geom_type[box] == mujoco.mjtGeom.mjGEOM_BOX
    # Collides with the rat (rodent geoms are contype 1) and nothing in the world body.
    rat_geoms = [
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] > bid and model.geom_contype[g]
    ]
    assert rat_geoms and all(
        (model.geom_contype[g] & model.geom_conaffinity[box])
        or (model.geom_contype[box] & model.geom_conaffinity[g])
        for g in rat_geoms
    )
    assert (0 << 16) + bid in set(model.exclude_signature.tolist())
    bars = np.asarray(env._gate_bar_geom_ids)
    assert len(bars) == 5
    assert np.all(model.geom_contype[bars] == 0) and np.all(
        model.geom_conaffinity[bars] == 0
    )
    np.testing.assert_allclose(
        model.geom_size[bars, 0], 0.025, atol=1e-9
    )  # 0.05 m bars
    np.testing.assert_allclose(
        model.geom_size[bars, 2], 0.15, atol=1e-9
    )  # full gate height
    np.testing.assert_allclose(
        np.sort(model.geom_pos[bars, 0]), [-0.1, -0.05, 0.0, 0.05, 0.1], atol=1e-9
    )
    order = np.argsort(model.geom_pos[bars, 0])
    grey = model.geom_rgba[bars[order], 0]
    np.testing.assert_allclose(grey, [0.95, 0.25, 0.95, 0.25, 0.95], atol=1e-6)
    assert np.all(model.geom_matid[bars] == -1)
    # On the start (-y) side, ~1 mm in front of the collision box.
    bar_back = model.geom_pos[bars, 1] + model.geom_size[bars, 1]
    assert np.all((bar_back < -0.015) & (bar_back > -0.015 - 0.003)), bar_back

    # Proprioception / task obs unchanged by the gate's dofs.
    state = hold_stack.reset(_keys(0))
    t_state = t_stack.reset(_keys(0))
    assert _proprio(state).shape == _proprio(t_state).shape
    np.testing.assert_allclose(_proprio(state), _proprio(t_state), atol=1e-5)
    np.testing.assert_allclose(_task_obs(state), _task_obs(t_state), atol=1e-6)
    assert np.asarray(state.info["critic_priv"]).shape == (N_ENVS, 8)

    # --- Reset: the gate closed at boundary 1, the hold armed ---
    assert env.gate_boundary_y(1) == pytest.approx(_GATE1_Y)
    np.testing.assert_allclose(
        _gate_body_xpos(env, state),
        np.tile([0.0, _GATE1_Y, 0.0], (N_ENVS, 1)),
        atol=1e-6,
    )
    # Same z extent as the walls: centred 0.15 - embed.
    np.testing.assert_allclose(
        np.asarray(state.data.geom_xpos[:, box]),
        np.tile(
            [0.0, _GATE1_Y, 0.15 - maze_forage_vision._WALL_FLOOR_EMBED], (N_ENVS, 1)
        ),
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        np.asarray(state.data.qvel[:, env._gate_dof_idxs]), 0.0
    )
    hold = _hold(state)
    assert set(hold) == {"hold_left", "gate_k", "open_t"}
    assert np.all((hold["hold_left"] >= 100) & (hold["hold_left"] <= 300))
    np.testing.assert_array_equal(hold["gate_k"], 1)
    np.testing.assert_array_equal(hold["open_t"], 0)
    np.testing.assert_array_equal(np.asarray(state.metrics["spout/hold_active"]), 1.0)
    np.testing.assert_array_equal(np.asarray(state.metrics["spout/gate_release"]), 0.0)
    np.testing.assert_array_equal(_trial(state)["trial_steps"], 0)
    # The whole rat starts behind the gate.
    rat_body = (
        np.asarray(model.body_rootid)
        == model.body_rootid[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso-rodent")
        ]
    )
    assert np.all(np.asarray(state.data.xpos[:, rat_body, 1]) < _GATE1_Y - 0.015)


@needs_gpu
def test_hold_gate_boundaries_map_to_stem_edges(hold_env, hold_stack):
    """Boundary k sits at y = -2.625 + 0.5 k (k = 1 ... 5)."""
    for k in range(1, 6):
        assert hold_env.gate_boundary_y(k) == pytest.approx(-2.625 + 0.5 * k)
    state = hold_stack.reset(_keys(14))
    state = _set_hold(state, gate_k=np.array([2, 3, 4, 5]), hold_left=200)
    state = hold_stack.step(state, jp.zeros((N_ENVS, hold_env.action_size)))
    np.testing.assert_allclose(
        _gate_body_xpos(hold_env, state)[:, 1],
        [-1.625, -1.125, -0.625, -0.125],
        atol=1e-6,
    )
    np.testing.assert_allclose(_gate_body_xpos(hold_env, state)[:, 2], 0.0, atol=1e-7)


@needs_gpu
def test_hold_gate_blocks(hold_env, hold_stack):
    """A rat driven into the closed gate for 100 steps never gets its torso past the gate plane."""
    env = hold_env
    state = hold_stack.reset(_keys(15))
    state = _set_hold(state, hold_left=250)  # the hold outlasts the push
    state = _pose_behind_gate(env, hold_stack, state, gap=0.12)
    state, torso_y, done, states = _drive_forward(env, hold_stack, state, 100)
    plane = _GATE1_Y - 0.5 * float(env._config.gate_thickness)
    print(
        f"gate-block: max torso y {torso_y.max():.4f} (gate face {plane:.4f}); "
        f"closest approach {plane - torso_y.max():.4f} m; done={int(done.sum())}"
    )
    assert np.all(done == 0)
    assert np.all(torso_y < plane), torso_y.max(axis=0)
    # Not vacuous: the push reached the gate (it would have carried the torso 0.5 m).
    assert torso_y.max() > plane - 0.12, torso_y.max()
    # The gate never moved.
    for s in states:
        np.testing.assert_array_equal(
            _gate_body_xpos(env, s),
            np.tile(np.float32([0.0, _GATE1_Y, 0.0]), (N_ENVS, 1)),
        )
        np.testing.assert_array_equal(
            np.asarray(s.data.qvel[:, env._gate_dof_idxs]), 0.0
        )
    assert np.all(_hold(state)["hold_left"] > 0)
    np.testing.assert_array_equal(_trial(state)["trial_steps"], 0)


@needs_gpu
def test_hold_gate_opens_then_rat_passes(hold_env, hold_stack):
    """Release: the gate slides below the floor over gate_open_steps, is not rendered, and the rat passes."""
    env = hold_env
    n_open = int(env._config.gate_open_steps)
    park_z = env.hold_params.park_z
    state = hold_stack.reset(_keys(16))
    state = _pose_behind_gate(env, hold_stack, state, gap=0.12)
    state = _set_hold(state, hold_left=1)  # the next step releases the gate
    state, torso_y, done, states = _drive_forward(env, hold_stack, state, n_open + 60)
    release = np.stack([np.asarray(s.metrics["spout/gate_release"]) for s in states])
    active = np.stack([np.asarray(s.metrics["spout/hold_active"]) for s in states])
    np.testing.assert_array_equal(release[0], 1.0)
    np.testing.assert_array_equal(release[1:], 0.0)
    np.testing.assert_array_equal(active[0], 1.0)
    np.testing.assert_array_equal(active[1:], 0.0)
    z = np.stack([_gate_body_xpos(env, s)[:, 2] for s in states])
    open_t = np.stack([_hold(s)["open_t"] for s in states])
    np.testing.assert_array_equal(open_t[:n_open, 0], np.arange(1, n_open + 1))
    np.testing.assert_array_equal(open_t[n_open:], n_open)
    np.testing.assert_allclose(
        z[: n_open - 1],
        np.tile(-0.3 * np.arange(1, n_open)[:, None] / n_open, (1, N_ENVS)),
        atol=1e-6,
    )
    np.testing.assert_allclose(z[n_open - 1 :], park_z, atol=1e-6)
    assert np.all(np.diff(z[:n_open], axis=0) < 0)
    # Fully below the floor after gate_open_steps steps, and invisible.
    opened = states[n_open - 1]
    assert np.all(_gate_geom_top_z(env, opened) < 0.0)
    assert np.all(
        _gate_geom_top_z(env, states[n_open - 2]) > 0.0
    )  # still showing one step before
    far = np.asarray(hold_stack.render(_with_gate_at(env, opened.data, -50.0)))
    assert np.array_equal(_vision(opened), far), "the parked gate shows in the eyes"
    # The trial clock starts after the release.
    np.testing.assert_array_equal(_trial(states[0])["trial_steps"], 0)
    np.testing.assert_array_equal(_trial(states[5])["trial_steps"], 5)
    # The same push now carries the torso through the gate plane.
    plane = _GATE1_Y - 0.5 * float(env._config.gate_thickness)
    print(
        f"gate-open: max torso y {torso_y.max(axis=0)} vs gate plane {plane:.4f}; done={int(done.sum())}"
    )
    assert np.all(torso_y.max(axis=0) > _GATE1_Y + 0.05), torso_y.max(axis=0)


def _geom_z_extent(model, geom_id: int) -> tuple[float, float]:
    """``(bottom, top)`` z of an axis-aligned (about z) box geom in the compiled model's world frame."""
    return (
        float(model.geom_pos[geom_id, 2] - model.geom_size[geom_id, 2]),
        float(model.geom_pos[geom_id, 2] + model.geom_size[geom_id, 2]),
    )


@needs_gpu
def test_hold_gate_spans_the_wall_z_range(hold_env, hold_stack):
    """The closed gate (box and bars) spans the walls' z range, so no slot is left under it.

    The walls sit ``_WALL_FLOOR_EMBED`` into the floor plane (z = -0.005); a gate standing on z = 0
    left a 5 mm slot below it that a finger capsule fits into and gets swept on release.
    """
    env, model = hold_env, hold_env.mj_model
    embed = maze_forage_vision._WALL_FLOOR_EMBED
    height = float(env._config.wall_height)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    floor_z = float(model.geom_pos[floor, 2])
    wall_bottoms, wall_tops = zip(
        *(
            _geom_z_extent(
                model,
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"maze_wall_{i}"),
            )
            for i in range(env.layout.wall_pos.shape[0])
        )
    )
    wall_bottom, wall_top = wall_bottoms[0], wall_tops[0]
    assert wall_bottom == pytest.approx(-embed, abs=1e-9) and wall_top == pytest.approx(
        height - embed, abs=1e-9
    )
    assert np.allclose(wall_bottoms, wall_bottom) and np.allclose(wall_tops, wall_top)

    # --- closed (reset): every gate geom's bottom is the walls' bottom, below the floor plane ---
    state = hold_stack.reset(_keys(19))
    half_h = np.asarray(model.geom_size[np.asarray(env._gate_geom_ids), 2])
    centre_z = np.asarray(state.data.geom_xpos[:, np.asarray(env._gate_geom_ids), 2])
    bottom, top = centre_z - half_h, centre_z + half_h
    print(
        f"gate z extent (closed): box [{bottom[0, 0]:.6f}, {top[0, 0]:.6f}] bars "
        f"[{bottom[0, 1:].min():.6f}, {top[0, 1:].max():.6f}] walls [{wall_bottom:.6f}, {wall_top:.6f}] floor {floor_z}"
    )
    assert np.all(bottom <= floor_z), bottom.max()
    np.testing.assert_allclose(bottom, wall_bottom, atol=1e-6)
    np.testing.assert_allclose(top, wall_top, atol=1e-6)
    np.testing.assert_allclose(top, np.asarray(_gate_geom_top_z(env, state)), atol=1e-7)
    # The compiled model agrees at the closed pose (gate body at z = 0).
    for g in np.asarray(env._gate_geom_ids):
        assert _geom_z_extent(model, int(g)) == pytest.approx(
            (wall_bottom, wall_top), abs=1e-6
        ), int(g)

    # --- parked: the highest point of every gate geom, bars included, is below the floor ---
    parked = _with_gate_at(env, state.data, env.hold_params.park_z)
    parked_top = (
        np.asarray(parked.geom_xpos[:, np.asarray(env._gate_geom_ids), 2]) + half_h
    )
    print(
        f"gate z extent (parked): highest point {parked_top.max():.4f} (floor {floor_z})"
    )
    assert np.all(parked_top < floor_z), parked_top.max()


@needs_gpu
def test_hold_gate_visible_from_start(hold_env, hold_stack):
    """The closed striped gate, seen from the start pose, changes >= 50 px per eye and its bars resolve."""
    env = hold_env
    state = hold_stack.reset(_keys(17))
    live = _vision(state)
    gateless = np.asarray(
        hold_stack.render(_with_gate_at(env, state.data, env.hold_params.park_z))
    )
    stats = _gate_pixel_stats(live, gateless)
    print(
        "gate pixels from the start pose (per env, [left, right]): "
        + " ".join(
            f"{k}={stats[k].round(3).tolist()}"
            for k in ("changed", "contrast", "row_std", "edges")
        )
    )
    assert np.all(stats["changed"] >= MIN_GATE_PX), stats["changed"]
    # The bars resolve: strong bright/dark contrast and several bar edges along a row.
    assert np.all(stats["contrast"] >= 0.15), stats["contrast"]
    assert np.all(stats["row_std"] >= 0.05), stats["row_std"]
    assert np.all(stats["edges"] >= 3), stats["edges"]


@needs_gpu
def test_hold_random_action_rollout_no_nan(hold_env, hold_stack):
    env = hold_env
    state = hold_stack.reset(_keys(1))
    key = jax.random.PRNGKey(11)
    released = np.zeros(N_ENVS, bool)
    for _ in range(300):
        key, sub = jax.random.split(key)
        action = jax.random.uniform(
            sub, (N_ENVS, env.action_size), minval=-1.0, maxval=1.0
        )
        state = hold_stack.step(state, action)
        assert np.all(np.isfinite(np.asarray(state.data.qpos)))
        assert np.all(np.isfinite(np.asarray(state.reward)))
        hold, trial = _hold(state), _trial(state)
        assert np.all(trial["trial_steps"][hold["hold_left"] > 0] == 0)
        for key_ in ("spout/hold_active", "spout/gate_release"):
            assert set(np.unique(np.asarray(state.metrics[key_]))) <= {0.0, 1.0}
        gx = _gate_body_xpos(env, state)
        np.testing.assert_allclose(
            gx[:, :2], np.tile([0.0, _GATE1_Y], (N_ENVS, 1)), atol=1e-6
        )
        released |= np.asarray(state.metrics["spout/gate_release"]) > 0
    assert released.any()  # holds are 1-3 s, so most envs released within 3 s
    for leaf in jax.tree.leaves(state.obs):
        assert np.all(np.isfinite(np.asarray(leaf)))
    assert np.all(np.isfinite(np.asarray(state.info["critic_priv"])))


@needs_gpu
def test_hold_trial_clock_frozen(hold_env, hold_stack):
    """``trial_steps`` stays 0 while ``hold_left > 0`` and counts from the release on."""
    env = hold_env
    holds = np.array([10, 20, 30, 40])
    state = hold_stack.reset(_keys(13))
    state = _set_hold(state, hold_left=holds)
    zeros = jp.zeros((N_ENVS, env.action_size))
    for call in range(1, 51):
        state = hold_stack.step(state, zeros)
        hold, trial = _hold(state), _trial(state)
        assert np.all(np.asarray(state.done) == 0)
        np.testing.assert_array_equal(hold["hold_left"], np.maximum(holds - call, 0))
        np.testing.assert_array_equal(
            trial["trial_steps"], np.maximum(call - holds, 0), err_msg=str(call)
        )
        np.testing.assert_array_equal(
            np.asarray(state.metrics["spout/hold_active"]),
            (call <= holds).astype(float),
        )
        np.testing.assert_array_equal(
            np.asarray(state.metrics["spout/gate_release"]),
            (call == holds).astype(float),
        )
        np.testing.assert_array_equal(np.asarray(state.metrics["spout/trial_end"]), 0.0)


@needs_gpu
def test_hold_no_trial_event_during_hold(hold_env, hold_stack):
    """With every unfrozen step forced to end the trial, nothing ends during the hold; the end re-arms it."""
    env = hold_env
    holds = np.array([3, 5, 7, 9])
    state = hold_stack.reset(_keys(18))
    state = _set_hold(state, hold_left=holds)
    zeros = jp.zeros((N_ENVS, env.action_size))
    env._force_trial_end = True
    try:
        forced = jax.jit(lambda s, a: hold_stack.vis.step(s, a))
        ends = []
        for _ in range(12):
            state = forced(state, zeros)
            ends.append(np.asarray(state.metrics["spout/trial_end"]))
    finally:
        env._force_trial_end = False
    ends = np.stack(ends)
    for e, h in enumerate(holds):
        expected = np.zeros(12)
        expected[h] = 1.0  # call h + 1: the first unfrozen step
        np.testing.assert_array_equal(ends[:, e], expected, err_msg=f"env {e}")
    hold = _hold(state)
    assert np.all(hold["hold_left"] >= 100 - 12)  # re-armed by the trial end
    np.testing.assert_array_equal(_trial(state)["trial_idx"], 1)


def _hold_forced_teleport_run(env, stack):
    """Reset, release the gate on step 1, three more random steps, a forced trial end on step 5."""
    keys = _keys(7)
    state = stack.reset(keys)
    state = _set_hold(state, hold_left=1)
    key = jax.random.PRNGKey(5)
    for _ in range(4):
        key, sub = jax.random.split(key)
        state = stack.step(
            state,
            jax.random.uniform(sub, (N_ENVS, env.action_size), minval=-1.0, maxval=1.0),
        )
    s4 = state
    key, sub = jax.random.split(key)
    a5 = jax.random.uniform(sub, (N_ENVS, env.action_size), minval=-1.0, maxval=1.0)
    env._force_trial_end = True
    try:
        s5 = jax.jit(lambda s, a: stack.vis.step(s, a))(s4, a5)
    finally:
        env._force_trial_end = False
    return types.SimpleNamespace(s4=s4, s5=s5, fresh=stack.reset(keys))


@needs_gpu
def test_hold_teleport_obs_equals_reset_obs(hold_env, hold_stack):
    env = hold_env
    r = _hold_forced_teleport_run(env, hold_stack)
    # Not vacuous: before the teleport the gate is dropping and the trial is running.
    np.testing.assert_array_equal(_hold(r.s4)["hold_left"], 0)
    np.testing.assert_array_equal(_hold(r.s4)["open_t"], 4)
    assert np.all(_gate_body_xpos(env, r.s4)[:, 2] < -0.05)
    assert np.all(_trial(r.s4)["trial_steps"] == 3)

    np.testing.assert_array_equal(np.asarray(r.s5.metrics["spout/trial_end"]), 1.0)
    np.testing.assert_array_equal(np.asarray(r.s5.metrics["spout/hold_active"]), 0.0)
    np.testing.assert_array_equal(np.asarray(r.s5.metrics["spout/gate_release"]), 0.0)
    hold = _hold(r.s5)
    assert np.all((hold["hold_left"] >= 100) & (hold["hold_left"] <= 300))
    np.testing.assert_array_equal(hold["open_t"], 0)
    np.testing.assert_array_equal(hold["gate_k"], 1)
    np.testing.assert_array_equal(_trial(r.s5)["trial_steps"], 0)

    v5, vf = _vision(r.s5), _vision(r.fresh)
    equal = np.isclose(v5, vf, atol=1e-6)
    for e in range(N_ENVS):
        for eye in range(2):
            assert equal[e, ..., eye].mean() >= 0.99, (
                e,
                eye,
                equal[e, ..., eye].mean(),
            )
    np.testing.assert_allclose(
        _imitation_target(r.s5), _imitation_target(r.fresh), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        _task_obs(r.s5), _task_obs(r.fresh), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(_proprio(r.s5), _proprio(r.fresh), rtol=1e-4, atol=1e-4)
    for name in ("qpos", "qvel", "ctrl", "act", "qacc_warmstart"):
        np.testing.assert_array_equal(
            np.asarray(getattr(r.s5.data, name)),
            np.asarray(getattr(r.fresh.data, name)),
            name,
        )
    # The closed gate, exactly as at reset.
    gate_geoms = np.asarray(env._gate_geom_ids)
    np.testing.assert_array_equal(
        np.asarray(r.s5.data.geom_xpos[:, gate_geoms]),
        np.asarray(r.fresh.data.geom_xpos[:, gate_geoms]),
    )
    np.testing.assert_allclose(
        _gate_body_xpos(env, r.s5),
        np.tile([0.0, _GATE1_Y, 0.0], (N_ENVS, 1)),
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# treat_visible_until_teleport: the eaten treat stays through the dwell
# ---------------------------------------------------------------------------

# 0.09 m before the spout point: inside the 0.1 m reach, so a standing rat eats on the first call.
_EAT_DIST = 0.09


def _changed_vs_parked(env, stack, data, live: np.ndarray) -> np.ndarray:
    """``(n_env, 2)`` pixels of ``live`` (the eye frames of ``data``) that differ from ``data`` with both treats parked."""
    parked = jax.vmap(env._park_collected_treats)(
        data, jp.ones((N_ENVS, env.n_treats), dtype=bool)
    )
    return _treat_pixel_counts(live, np.asarray(stack.render(parked)))[0]


def _eat_run(env, stack, pixel_calls=()):
    """Four rats 0.09 m from a spout facing it (no hold left, if any), zero actions for one full dwell.

    Envs 0 / 1 are on the baited arm (0 / 1: the treat appears and is eaten on call 1); envs 2 / 3 are
    on the other arm (no treat: a visit dwell).  Returns the per-call ``SimpleNamespace`` records
    (call 1 ... ``dwell_steps``, the last being the trial-end / teleport call) with the treat geom
    z, the reward metric, ``trial_end``, ``info["collected"]``, the trial dwell, and the changed
    pixels of the live eye frames against the same data with both treats parked, on the calls in
    ``pixel_calls`` only (``None`` elsewhere): ``changed`` for the frame the step itself rendered (a
    zero-action rat topples and turns away within a few calls, so this is only reliable on the first)
    and ``changed_far`` with the rat re-posed 0.30 m before the spout facing it (same joints, same
    treat state), the viewing distance of ``test_t_treat_visible_from_arm``.

    Renders are kept to ``pixel_calls``: the suite is already close to the warp per-process
    allocation limit (the reason the suites run as two processes); with a render on every call of
    the dwell the later default-off guard test was observed to fail with a warp out-of-memory.
    """
    layout = env.layout
    dwell = int(env._config.dwell_steps)
    state = stack.reset(_keys(21))
    poses = [_arm_pose(layout, arm, _EAT_DIST) for arm in (0, 1, 0, 1)]
    state = _pose(
        env,
        stack,
        state,
        np.stack([p[0] for p in poses]),
        np.array([p[1] for p in poses]),
        bait=np.array([0, 1, 1, 0]),
    )
    if env.hold_enabled:
        state = _set_hold(state, hold_left=0, open_t=int(env._config.gate_open_steps))
    zeros = jp.zeros((N_ENVS, env.action_size))
    far = [_arm_pose(layout, arm, 0.30) for arm in (0, 1, 0, 1)]
    far_xy, far_yaw = np.stack([p[0] for p in far]), np.array([p[1] for p in far])

    def changed_far(st):
        far_data = _pose(env, stack, st, far_xy, far_yaw).data
        return _changed_vs_parked(
            env, stack, far_data, np.asarray(stack.render(far_data))
        )

    records = []
    for call in range(1, dwell + 1):
        state = stack.step(state, zeros)
        records.append(
            types.SimpleNamespace(
                call=call,
                z=_treat_geom_z(env, state),
                reward=np.asarray(state.metrics["rewards/treat_collected"]),
                trial_end=np.asarray(state.metrics["spout/trial_end"]),
                collected=np.asarray(state.info["collected"]),
                dwell=_trial(state)["dwell"],
                n_collected=np.asarray(state.info["n_collected"]),
                changed=_changed_vs_parked(env, stack, state.data, _vision(state))
                if call in pixel_calls
                else None,
                changed_far=changed_far(state) if call in pixel_calls else None,
            )
        )
    return records


@needs_gpu
def test_eaten_treat_stays_visible_through_the_dwell(hold_env, hold_stack):
    """Flag ON (the T + hold env): the treat stays rendered on calls 1 ... D-1 (after the eat),
    is parked on the teleport call, and pays once."""
    env = hold_env
    assert env._config.treat_visible_until_teleport is True
    dwell, th = int(env._config.dwell_steps), float(env._config.treat_height)
    shown_calls = (1, dwell // 2, dwell - 1, dwell)
    records = _eat_run(env, hold_stack, pixel_calls=shown_calls)
    first = records[0]
    np.testing.assert_array_equal(first.reward, [1, 1, 0, 0])
    np.testing.assert_array_equal(
        first.dwell, dwell - 1
    )  # the eat call starts the dwell: dwell >= 0 from here

    for r in records[:-1]:  # calls 1 ... D-1: the dwell
        np.testing.assert_allclose(
            [r.z[0, 0], r.z[1, 1]], th, atol=1e-4, err_msg=f"call {r.call}"
        )
        # Only the eaten treat is out: the other arm's treat, and every treat of the no-treat envs, stay parked.
        assert r.z[0, 1] < -0.5 and r.z[1, 0] < -0.5 and np.all(r.z[2:] < -0.5), r.call
        np.testing.assert_array_equal(
            r.collected, [[False, True], [True, False], [True, True], [True, True]]
        )
        np.testing.assert_array_equal(r.trial_end, 0.0)
        assert np.all(r.dwell == dwell - r.call), (r.call, r.dwell)
    seen = [r for r in records[:-1] if r.changed is not None]
    near = np.array(
        [r.changed[:2].max(axis=-1) for r in seen]
    )  # (calls, 2): best eye per eaten env
    far = np.array([r.changed_far[:2].max(axis=-1) for r in seen])
    print(
        f"treat pixels changed vs parked, best eye, calls {[r.call for r in seen]} (envs 0/1): "
        f"in the step's own frame: call 1 {near[0].tolist()}; from 0.30 m: {far.tolist()}"
    )
    # Rendered, not just placed: through the training wrapper, at least one eye resolves it from the
    # usual viewing distance, and in the step's own frame on the eat step itself.
    assert np.all(near[0] >= MIN_TREAT_PX), near[0]
    assert np.all(far >= MIN_TREAT_PX), far
    # Nothing is shown in the visit-dwell envs.
    assert all(
        np.all(r.changed[2:] == 0) and np.all(r.changed_far[2:] == 0) for r in seen
    )

    last = records[-1]  # the teleport call
    np.testing.assert_array_equal(last.trial_end, 1.0)
    assert np.all(last.z < -0.5)
    assert np.all(last.collected)
    assert np.all(last.changed == 0)  # the start view shows no treat
    assert np.all(last.changed_far == 0)
    # Reward logic is unchanged: paid exactly once per eaten treat, on the eat call.
    total = np.sum([r.reward for r in records], axis=0)
    np.testing.assert_array_equal(total, [1, 1, 0, 0])
    np.testing.assert_array_equal(last.n_collected, [1, 1, 0, 0])
    assert all(np.all(r.reward == 0) for r in records[1:])
