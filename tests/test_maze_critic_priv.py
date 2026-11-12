"""Tests for the full-maze privileged critic vector, visited-cell map and metric.

``MazeForageVision`` with ``maze_critic_priv=True`` writes ``info["visited"]``
(``bool[n_free]``) and ``info["critic_priv"]`` (``float32[P]``, ``P = n_free +
3 * n_treats + 5``) at reset and at the end of ``step``, plus the metric
``maze/visited_cells``.  With the flag off (the default) the maze env, and the
spout env built on it, must be identical to the commit before the flag existed
(``maze_reference``).

The env is warp-only: building and tracing it runs on the CPU, executing it
needs a CUDA GPU.  The CPU tests compare models, traced ``reset`` / ``step``
and output structures; the ``gpu`` tests run the env.  Run serially::

    MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        python -m pytest tests/test_maze_critic_priv.py -x -q
"""

import types

import jax
import jax.numpy as jp
import maze_reference as ref_tools
import mujoco
import numpy as np
import pytest
from mujoco import mjx

from vnl_playground.tasks.rodent import maze_forage_vision as mfv
from vnl_playground.tasks.rodent import spout_maze

_HAS_GPU = any(d.platform == "gpu" for d in jax.devices())
_skip_without_gpu = pytest.mark.skipif(
    not _HAS_GPU, reason="the warp env needs a CUDA GPU"
)


def needs_gpu(fn):
    """``gpu``-marked (selected by ``-m gpu``) and skipped when no GPU is visible."""
    return pytest.mark.gpu(_skip_without_gpu(fn))


EPISODE_LENGTH = 50  # small, so step_count / episode_length is far from 0
EXTENT = 6.46  # maze_extent default
HALF = EXTENT / 2.0


def _build_env(**overrides) -> mfv.MazeForageVision:
    cfg = mfv.default_config()
    for key, value in overrides.items():
        cfg[key] = value
    return mfv.MazeForageVision(config=cfg)


@pytest.fixture(scope="module")
def env():
    """``maze_critic_priv=True`` env with a short episode (production maze: 199 cells, 20 treats)."""
    return _build_env(maze_critic_priv=True, episode_length=EPISODE_LENGTH)


@pytest.fixture(scope="module")
def off_env():
    """The default (``maze_critic_priv=False``) env, built once for the guard tests."""
    return _build_env()


@pytest.fixture(scope="module")
def jit_fns(env):
    return types.SimpleNamespace(
        reset=jax.jit(env.reset),
        step=jax.jit(env.step),
        forward=jax.jit(lambda d: mjx.forward(env.mjx_model, d)),
    )


# ---------------------------------------------------------------------------
# Independent re-derivations of the vector (numpy, from `data` only)
# ---------------------------------------------------------------------------


def _torso(env, data):
    tid = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso-rodent")
    return (
        np.asarray(data.xpos[tid], np.float64),
        np.asarray(data.xmat[tid], np.float64).reshape(3, 3),
    )


def _nearest_cell(env, torso_xy) -> int:
    d2 = ((env.free_cell_positions.astype(np.float64) - np.asarray(torso_xy)) ** 2).sum(
        -1
    )
    return int(np.argmin(d2))


def _split(env, priv):
    """The layout of ``critic_priv``: visited, collected, ego (n, 2), xy, heading, t."""
    n, k = env.n_free_cells, env.n_treats
    priv = np.asarray(priv)
    out = types.SimpleNamespace(
        visited=priv[:n],
        collected=priv[n : n + k],
        ego=priv[n + k : n + 3 * k].reshape(k, 2),
        xy=priv[n + 3 * k : n + 3 * k + 2],
        heading=priv[n + 3 * k + 2 : n + 3 * k + 4],
        t=priv[n + 3 * k + 4 :],
    )
    assert out.t.shape == (1,)
    return out


def _check_state(env, state, expected_step_count):
    """Every documented property of the vector, against ``state`` (one unbatched env state)."""
    priv = np.asarray(state.info["critic_priv"])
    vis = np.asarray(state.info["visited"])
    n, k = env.n_free_cells, env.n_treats
    assert env.critic_priv_dim == n + 3 * k + 5 == priv.shape[0]
    assert priv.dtype == np.float32 and np.all(np.isfinite(priv))
    assert vis.dtype == np.bool_ and vis.shape == (n,)
    p = _split(env, priv)
    pos, mat = _torso(env, state.data)

    # visited: the slice IS info["visited"], its sum IS the metric, and the cell the
    # torso is in is marked.
    np.testing.assert_array_equal(p.visited, vis.astype(np.float32))
    assert float(state.metrics["maze/visited_cells"]) == float(vis.sum())
    assert state.metrics["maze/visited_cells"].dtype == jp.float32
    assert vis[_nearest_cell(env, pos[:2])]

    # collected slice == info["collected"].
    collected = np.asarray(state.info["collected"])
    np.testing.assert_array_equal(p.collected, collected.astype(np.float32))

    # Treat xy in the YAW-ONLY torso frame (world planar difference rotated by -yaw)
    # / (extent / 2); 0 where collected.
    treats = np.asarray(state.data.xpos[np.asarray(env._treat_body_ids)], np.float64)
    yaw = np.arctan2(mat[1, 0], mat[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    d = treats[:, :2] - pos[:2]
    ego = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], -1) / HALF
    ego = np.where(collected[:, None], 0.0, ego)
    np.testing.assert_allclose(p.ego, ego, atol=2e-6)
    assert np.all(p.ego[collected] == 0.0)
    # Ego, xy and heading share one yaw frame: world xy of every live treat is
    # recoverable from the vector's own entries (xy + R(heading) @ ego * extent / 2),
    # whatever the torso's pitch / roll.
    hc, hs = p.heading
    world = (
        p.xy * HALF
        + np.stack(
            [hc * p.ego[:, 0] - hs * p.ego[:, 1], hs * p.ego[:, 0] + hc * p.ego[:, 1]],
            -1,
        )
        * HALF
    )
    np.testing.assert_allclose(world[~collected], treats[~collected, :2], atol=2e-5)

    # torso world xy / (extent / 2); heading cos, sin of the torso yaw.
    np.testing.assert_allclose(p.xy, pos[:2] / HALF, atol=2e-6)
    np.testing.assert_allclose(p.heading, [np.cos(yaw), np.sin(yaw)], atol=2e-6)
    assert abs(float(p.heading @ p.heading) - 1.0) < 2e-6

    # step_count / episode_length, with the count AFTER this step's increment.
    assert int(state.info["step_count"]) == expected_step_count
    np.testing.assert_allclose(
        p.t[0], expected_step_count / EPISODE_LENGTH, rtol=1e-6, atol=1e-9
    )
    assert priv[-1] == p.t[0]


# ---------------------------------------------------------------------------
# Layout at reset and after steps
# ---------------------------------------------------------------------------


def test_production_dimensions(env):
    assert env.n_free_cells == 199 and env.n_treats == 20
    assert env.critic_priv_dim == 199 + 20 + 40 + 2 + 2 + 1 == 264


@needs_gpu
def test_reset_layout(env, jit_fns):
    for seed in (0, 1, 2):
        state = jit_fns.reset(jax.random.PRNGKey(seed))
        _check_state(env, state, expected_step_count=0)
        vis = np.asarray(state.info["visited"])
        assert vis.sum() == 1  # the spawn cell, nothing else
        assert not np.any(np.asarray(state.info["collected"]))
        # No treat is collected: every egocentric entry is a live vector.
        assert np.any(_split(env, state.info["critic_priv"]).ego != 0.0)


@needs_gpu
def test_reset_marks_the_spawn_cell(env, jit_fns):
    """The visited bit is the cell the rat was spawned in (the root's xy is a cell centre)."""
    root = env._rodent_root_qpos
    for seed in range(4):
        state = jit_fns.reset(jax.random.PRNGKey(seed))
        spawn_xy = np.asarray(state.data.qpos[root : root + 2])
        spawn_cell = _nearest_cell(env, spawn_xy)
        np.testing.assert_allclose(
            env.free_cell_positions[spawn_cell], spawn_xy, atol=1e-6
        )
        assert np.flatnonzero(np.asarray(state.info["visited"])).tolist() == [
            spawn_cell
        ]


@needs_gpu
def test_free_cell_centres_are_in_the_torso_frame(env, jit_fns):
    """The cell table lives in the frame of ``xpos``: the torso at reset is within a
    fraction of a cell of the spawn centre, never half a pitch away."""
    root = env._rodent_root_qpos
    state = jit_fns.reset(jax.random.PRNGKey(3))
    pos, _ = _torso(env, state.data)
    spawn_xy = np.asarray(state.data.qpos[root : root + 2])
    assert np.linalg.norm(pos[:2] - spawn_xy) < 0.5 * env.cell_size


@needs_gpu
def test_heading_is_the_spawn_yaw(env, jit_fns):
    """Heading (cos, sin) at reset equals the spawn yaw written into the root quaternion."""
    root = env._rodent_root_qpos
    state = jit_fns.reset(jax.random.PRNGKey(5))
    q = np.asarray(state.data.qpos[root + 3 : root + 7])
    yaw = 2.0 * np.arctan2(q[3], q[0])
    np.testing.assert_allclose(
        _split(env, state.info["critic_priv"]).heading,
        [np.cos(yaw), np.sin(yaw)],
        atol=1e-4,
    )


@needs_gpu
def test_step_layout_and_monotone_visited_over_random_actions(env, jit_fns):
    state = jit_fns.reset(jax.random.PRNGKey(7))
    rng = np.random.default_rng(0)
    prev = np.asarray(state.info["visited"])
    counts = [int(prev.sum())]
    for t in range(1, 201):
        action = jp.asarray(rng.uniform(-1.0, 1.0, env.action_size), jp.float32)
        state = jit_fns.step(state, action)
        vis = np.asarray(state.info["visited"])
        assert np.all(vis >= prev), f"a visited bit was cleared at step {t}"
        prev = vis
        counts.append(int(vis.sum()))
        if t in (1, 2, 50, 200):
            # (EPISODE_LENGTH is only the normaliser; the raw env is stepped past it.)
            _check_state(env, state, expected_step_count=t)
    assert counts == sorted(counts)
    print(f"visited cells over 200 random-action steps: {counts[0]} -> {counts[-1]}")


@needs_gpu
def test_teleport_into_a_far_cell_marks_it_on_the_next_step(env, jit_fns):
    state = jit_fns.reset(jax.random.PRNGKey(11))
    pos, _ = _torso(env, state.data)
    root = env._rodent_root_qpos
    root_xy = np.asarray(state.data.qpos[root : root + 2], np.float64)
    offset = pos[:2] - root_xy  # torso minus root, a few cm
    free = env.free_cell_positions.astype(np.float64)
    spawn_cell = _nearest_cell(env, pos[:2])
    far = int(np.argmax(((free - free[spawn_cell]) ** 2).sum(-1)))
    assert far != spawn_cell
    assert not bool(np.asarray(state.info["visited"])[far])

    qpos = state.data.qpos.at[root : root + 2].set(
        jp.asarray(free[far] - offset, jp.float32)
    )
    data = jit_fns.forward(
        state.data.replace(qpos=qpos, qvel=jp.zeros_like(state.data.qvel))
    )
    state = state.replace(data=data)
    state = jit_fns.step(state, jp.zeros(env.action_size))
    vis = np.asarray(state.info["visited"])
    assert vis[far], "the far cell the torso was teleported into is not marked"
    assert vis[spawn_cell], "the spawn cell bit was cleared"
    assert vis.sum() == 2
    assert float(state.metrics["maze/visited_cells"]) == 2.0
    _check_state(env, state, expected_step_count=1)


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


@needs_gpu
@pytest.mark.parametrize("pitch_deg", [60.0, 110.0])
def test_treat_vectors_are_yaw_only_when_the_torso_is_pitched(env, jit_fns, pitch_deg):
    """A pitched torso (rearing; 110 deg is past vertical, where the projected x axis
    flips): the vector's treat entries are still the yaw-frame planar difference, share
    the heading's yaw, and differ from the full-body-frame vector the policy-side
    ``privileged_state`` carries."""
    state = jit_fns.reset(jax.random.PRNGKey(17))
    root = env._rodent_root_qpos
    q = np.asarray(state.data.qpos[root + 3 : root + 7], np.float64)
    a = np.deg2rad(pitch_deg)
    # Pitch about the body y axis.
    q = _quat_mul(q, np.array([np.cos(a / 2), 0.0, np.sin(a / 2), 0.0]))
    qpos = state.data.qpos.at[root + 3 : root + 7].set(jp.asarray(q, jp.float32))
    qpos = qpos.at[root + 2].set(0.35)  # clear of the floor for the one step below
    data = jit_fns.forward(
        state.data.replace(qpos=qpos, qvel=jp.zeros_like(state.data.qvel))
    )
    state = jit_fns.step(state.replace(data=data), jp.zeros(env.action_size))
    _, mat = _torso(env, state.data)
    assert abs(mat[2, 2]) < 0.6, f"torso is not pitched: xmat[2,2] = {mat[2, 2]:.2f}"
    _check_state(env, state, expected_step_count=1)
    full_frame = np.asarray(env._egocentric_treat_vectors(state.data))[:, :2] / HALF
    p = _split(env, state.info["critic_priv"])
    assert np.max(np.abs(full_frame - p.ego)) > 1e-2, (
        "yaw-only vector equals the full-frame one"
    )


@needs_gpu
def test_policy_side_privileged_state_is_unchanged(env, jit_fns):
    """The critic vector does not touch ``privileged_state`` (full-frame 3-D treat vectors)."""
    state = jit_fns.reset(jax.random.PRNGKey(19))
    priv = state.obs["privileged_state"]
    np.testing.assert_allclose(
        np.asarray(priv["treat_vectors"]),
        np.asarray(env._egocentric_treat_vectors(state.data)).ravel(),
        atol=1e-6,
    )
    assert priv["treat_vectors"].shape == (3 * env.n_treats,)


@needs_gpu
def test_collected_treats_are_zeroed_in_the_vector(env, jit_fns):
    """Force a few treats collected: the mask slice follows and their vector entries are 0."""
    state = jit_fns.reset(jax.random.PRNGKey(13))
    # Put the rat on treat 3 and treat 11 one at a time by teleporting onto them.
    root = env._rodent_root_qpos
    pos, _ = _torso(env, state.data)
    offset = pos[:2] - np.asarray(state.data.qpos[root : root + 2], np.float64)
    for treat in (3, 11):
        txy = np.asarray(
            state.data.xpos[int(env._treat_body_ids[treat]), :2], np.float64
        )
        qpos = state.data.qpos.at[root : root + 2].set(
            jp.asarray(txy - offset, jp.float32)
        )
        data = jit_fns.forward(
            state.data.replace(qpos=qpos, qvel=jp.zeros_like(state.data.qvel))
        )
        state = jit_fns.step(state.replace(data=data), jp.zeros(env.action_size))
    collected = np.asarray(state.info["collected"])
    assert collected[3] and collected[11]
    p = _split(env, state.info["critic_priv"])
    assert np.all(p.ego[collected] == 0.0)
    assert np.any(p.ego[~collected] != 0.0)
    _check_state(env, state, expected_step_count=2)


# ---------------------------------------------------------------------------
# State structure and the reset contract
# ---------------------------------------------------------------------------


def test_metrics_keys_equal_between_reset_and_step(env):
    key = jax.random.PRNGKey(0)
    s0 = jax.eval_shape(env.reset, key)
    s1 = jax.eval_shape(
        env.step, s0, jax.ShapeDtypeStruct((env.action_size,), jp.float32)
    )
    assert "maze/visited_cells" in s0.metrics
    assert set(s0.metrics) == set(s1.metrics)
    assert jax.tree.structure(s0) == jax.tree.structure(s1)
    assert (
        s0.info["critic_priv"].shape
        == s1.info["critic_priv"].shape
        == (env.critic_priv_dim,)
    )
    assert s0.info["critic_priv"].dtype == s1.info["critic_priv"].dtype == jp.float32
    assert s0.info["visited"].shape == s1.info["visited"].shape == (env.n_free_cells,)
    assert s0.info["visited"].dtype == jp.bool_


def test_flag_adds_info_and_metric_keys_only(env, off_env):
    """The policy's observation is the same with the flag on; info gains the two keys
    and metrics the one."""
    key = jax.random.PRNGKey(0)
    on, off = jax.eval_shape(env.reset, key), jax.eval_shape(off_env.reset, key)
    assert ref_tools.shape_tree(on.obs) == ref_tools.shape_tree(off.obs)
    assert set(on.info) - set(off.info) == {"visited", "critic_priv"}
    assert set(off.info) <= set(on.info)
    assert set(on.metrics) - set(off.metrics) == {"maze/visited_cells"}
    assert set(off.metrics) <= set(on.metrics)


def test_info_reset_keys_when_enabled(env):
    assert env.info_reset_keys == (*mfv.INFO_RESET_KEYS, "visited", "critic_priv")
    assert mfv.INFO_RESET_KEYS == (  # the module constant is not changed
        "prev_action",
        "action",
        "step_count",
        "collected",
        "n_collected",
        "belly_up_steps",
        "low_torso_steps",
    )
    state = jax.eval_shape(env.reset, jax.random.PRNGKey(0))
    assert not [k for k in env.info_reset_keys if k not in state.info]


@needs_gpu
def test_after_autoreset_visited_holds_only_the_new_spawn_cell(env):
    """Every episode boundary runs ``reset`` under ``full_reset=True``."""
    from mujoco_playground._src import wrapper as mp_wrapper

    batch = 3
    wrapped = mp_wrapper.wrap_for_brax_training(
        env, episode_length=EPISODE_LENGTH, action_repeat=1, full_reset=True
    )
    state = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(0), batch))
    step = jax.jit(wrapped.step)
    action = jp.zeros((batch, env.action_size))
    for _ in range(3):  # grow the visited set a little before the boundary
        state = step(state, action)
    # Force the episode to end: every treat collected.
    state.info["collected"] = jp.ones((batch, env.n_treats), dtype=bool)
    state = step(state, action)
    assert np.all(np.asarray(state.done) > 0.5)

    vis = np.asarray(state.info["visited"])
    assert np.all(vis.sum(-1) == 1), vis.sum(-1)
    root = env._rodent_root_qpos
    for e in range(batch):
        spawn_xy = np.asarray(state.data.qpos[e, root : root + 2])
        assert np.flatnonzero(vis[e]).tolist() == [_nearest_cell(env, spawn_xy)]
    # critic_priv is the fresh episode's: step_count 0, visited slice == the new visited map.
    priv = np.asarray(state.info["critic_priv"])
    assert np.all(priv[:, -1] == 0.0)
    np.testing.assert_array_equal(priv[:, : env.n_free_cells], vis.astype(np.float32))
    assert np.all(priv[:, env.n_free_cells : env.n_free_cells + env.n_treats] == 0.0)
    # The next episode runs: the new episode's visited set only grows from the new spawn.
    state2 = step(state, action)
    assert np.all(np.asarray(state2.done) < 0.5)
    vis2 = np.asarray(state2.info["visited"])
    assert np.all(vis2 >= vis)


# ---------------------------------------------------------------------------
# Default off / spout
# ---------------------------------------------------------------------------


def test_default_config_key_is_off():
    assert mfv.default_config().maze_critic_priv is False
    assert spout_maze.default_config().maze_critic_priv is False


def test_episode_length_must_be_positive_with_the_flag():
    with pytest.raises(ValueError, match="episode_length > 0"):
        _build_env(maze_critic_priv=True, episode_length=0)


def test_spout_env_raises_if_maze_critic_priv_is_true():
    """The spout env has its own always-on ``critic_priv``; it refuses the maze flag
    (raised before the model is built)."""
    cfg = spout_maze.default_config()
    cfg["maze_critic_priv"] = True
    with pytest.raises(ValueError, match="maze_critic_priv"):
        spout_maze.RodentSpoutMazeVision(config=cfg)


def test_disabled_env_has_no_new_keys_metrics_or_properties(off_env):
    off = off_env
    state = jax.eval_shape(off.reset, jax.random.PRNGKey(0))
    assert set(state.info) == set(mfv.INFO_RESET_KEYS)
    assert not [k for k in state.metrics if k.startswith("maze/")]
    assert off.info_reset_keys is mfv.INFO_RESET_KEYS
    with pytest.raises(AttributeError, match="maze_critic_priv"):
        _ = off.critic_priv_dim


# ---------------------------------------------------------------------------
# Default off: identical to the env before the flag existed
# ---------------------------------------------------------------------------

_REF_REV, _REF_SKIP = ref_tools.reference_rev("maze_critic_priv")


@pytest.fixture(scope="module")
def ref_modules():
    """The maze and spout env modules of the commit before ``maze_critic_priv`` existed."""
    if _REF_REV is None:
        pytest.skip(_REF_SKIP)
    return ref_tools.load_reference(_REF_REV)


# (name, env class, config overrides): the full maze and the Y- and T-shaped
# spout mazes, each at its default config.
_DEFAULT_ENVS = [
    ("maze", "MazeForageVision", {}),
    ("spout_y", "RodentSpoutMazeVision", {}),
    ("spout_t", "RodentSpoutMazeVision", {"maze_shape": "T"}),
]
_ENV_IDS = [e[0] for e in _DEFAULT_ENVS]
_ENV_PARAMS = [e[1:] for e in _DEFAULT_ENVS]

# Config keys this file's feature adds; each is off by default.
_NEW_KEYS = {"maze_critic_priv": False}


def _configs(ref_modules, cls_name, overrides):
    """``(new_cfg, ref_cfg, new_cls, ref_cls)`` of the current and the reference env."""
    if cls_name == "MazeForageVision":
        new_mod, ref_mod = mfv, ref_modules.maze
    else:
        new_mod, ref_mod = spout_maze, ref_modules.spout
    out = []
    for mod in (new_mod, ref_mod):
        cfg = mod.default_config()
        for key, value in overrides.items():
            cfg[key] = value
        out.append(cfg)
    return (*out, getattr(new_mod, cls_name), getattr(ref_mod, cls_name))


def _env_pair(ref_modules, cls_name, overrides):
    """``(new_env, ref_env)``: the current and the reference env at one config."""
    new_cfg, ref_cfg, new_cls, ref_cls = _configs(ref_modules, cls_name, overrides)
    return new_cls(config=new_cfg), ref_cls(config=ref_cfg)


@pytest.mark.parametrize("cls_name,overrides", _ENV_PARAMS, ids=_ENV_IDS)
def test_default_config_only_gains_keys_that_are_off(ref_modules, cls_name, overrides):
    """Every reference key keeps its value; the added keys default to off."""
    new_cfg, ref_cfg, _, _ = _configs(ref_modules, cls_name, overrides)
    new, ref = new_cfg.to_dict(), ref_cfg.to_dict()
    assert {k: new[k] for k in ref} == ref
    for key, off in _NEW_KEYS.items():
        assert new[key] == off, key


@pytest.mark.parametrize("cls_name,overrides", _ENV_PARAMS, ids=_ENV_IDS)
def test_default_off_traces_identically_to_the_reference(
    ref_modules, cls_name, overrides
):
    """Default config: the same compiled model, the same observation / info / metrics
    structure, and ``reset`` / ``step`` trace to the same jaxpr with the same
    constants as before the flag existed.  Identical traces give identical outputs."""
    new_env, ref_env = _env_pair(ref_modules, cls_name, overrides)
    assert not ref_tools.model_mismatches(new_env.mj_model, ref_env.mj_model)
    assert tuple(new_env.info_reset_keys) == tuple(ref_env.info_reset_keys)

    key = jax.random.PRNGKey(0)
    ref_reset = jax.eval_shape(ref_env.reset, key)
    action = jax.ShapeDtypeStruct((ref_env.action_size,), jp.float32)
    ref_step = jax.eval_shape(ref_env.step, ref_reset, action)
    new_reset = jax.eval_shape(new_env.reset, key)
    assert ref_tools.shape_tree(new_reset) == ref_tools.shape_tree(ref_reset)
    for part in ("obs", "info", "metrics"):
        assert ref_tools.shape_tree(getattr(new_reset, part)) == ref_tools.shape_tree(
            getattr(ref_reset, part)
        ), part
    new_step = jax.eval_shape(new_env.step, ref_reset, action)
    assert ref_tools.shape_tree(new_step) == ref_tools.shape_tree(ref_step)

    hint = f"the default env no longer traces like {_REF_REV} (see maze_reference)"
    assert not ref_tools.jaxpr_mismatch(new_env.reset, ref_env.reset, key), hint
    assert not ref_tools.jaxpr_mismatch(
        new_env.step, ref_env.step, ref_reset, action
    ), hint


def test_the_trace_comparison_sees_the_flag(ref_modules, env):
    """The guard above has teeth: the flag-on env does not trace like the reference."""
    cfg = ref_modules.maze.default_config()
    cfg["episode_length"] = EPISODE_LENGTH
    ref_env = ref_modules.maze.MazeForageVision(config=cfg)
    assert ref_tools.jaxpr_mismatch(env.reset, ref_env.reset, jax.random.PRNGKey(0))


@needs_gpu
@pytest.mark.parametrize("cls_name,overrides", _ENV_PARAMS, ids=_ENV_IDS)
def test_default_off_runs_identically_to_the_reference(
    ref_modules, cls_name, overrides
):
    """Default config: the same reset data bitwise, and the same step data to the warp
    solver's run-to-run noise, leaf for leaf, over five random actions."""
    new_env, ref_env = _env_pair(ref_modules, cls_name, overrides)
    key = jax.random.PRNGKey(0)
    rng = np.random.default_rng(1)
    actions = [
        jp.asarray(rng.uniform(-1, 1, new_env.action_size), jp.float32)
        for _ in range(5)
    ]
    runs = []
    for e in (new_env, ref_env):
        reset, step = jax.jit(e.reset), jax.jit(e.step)
        states = [reset(key)]
        for a in actions:
            states.append(step(states[-1], a))
        runs.append(states)
    for i, (s_new, s_ref) in enumerate(zip(*runs)):
        compared, bad = ref_tools.state_mismatches(s_new, s_ref, exact=(i == 0))
        print(
            f"state {i} vs {_REF_REV}: {len(compared)} leaves compared "
            f"({'bitwise' if i == 0 else 'to solver noise'}), {len(bad)} differ"
        )
        assert not bad, (i, bad)
        for leaf in (
            ".data.qpos",
            ".data.xpos",
            ".obs",
            ".reward",
            ".metrics",
            ".info",
        ):
            assert any(leaf in c for c in compared), (i, leaf)
