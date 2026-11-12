"""Tests for the opt-in dm_control-style ``contact_termination`` of ``MazeForageVision``.

dm_control's ``RandomGoalMaze`` fails an episode on ANY contact between a walker geom
that is not an allowed ground-contact geom and a ground geom.  Here the criterion is
``@_registry.termination("contact_termination")``: present only when a config lists
``termination_criteria.contact_termination``, evaluated by an exact geometric test
against the floor plane (``floor_contact.py``; the contact list itself cannot be read
per world under warp without the O(naconmax) copy ``nan_termination`` warns about).

Two layers:

* CPU -- the closed-form lowest-point / margin / filter logic checked against
  MuJoCo's own contact list and ``mj_geomDistance`` on small hand-written models;
  the config contract; the candidate geoms of the production walker (resolved at
  build time); and the flag-off guard: the default env traces like the commit
  before the criterion existed (``maze_reference``).
* GPU (executing the warp env) -- the real production walker: >= 600 random,
  fallen and lying poses against CPU ``mj_forward`` contacts, the standing spawn
  pose, the firing / non-firing posture cases, ``vmap`` isolation between worlds,
  the metrics, and the flag-off run against the reference.

Run serially::

    MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        python -m pytest tests/test_maze_contact_termination.py -x -q
"""

import os
import pathlib
import types

import jax
import jax.numpy as jp
import maze_reference as ref_tools
import mujoco
import numpy as np
import pytest
from mujoco import mjx

from vnl_playground.tasks.rodent import floor_contact as fc
from vnl_playground.tasks.rodent import maze_forage_vision as mfv

_HAS_GPU = any(d.platform == "gpu" for d in jax.devices())
_skip_without_gpu = pytest.mark.skipif(
    not _HAS_GPU, reason="the warp env needs a CUDA GPU"
)


def needs_gpu(fn):
    """``gpu``-marked (selected by ``-m gpu``) and skipped when no GPU is visible."""
    return pytest.mark.gpu(_skip_without_gpu(fn))


_PRIOR_ROOT = os.environ.get("DMPO_PRIOR_ROOT")
_REFERENCE_CLIPS = (
    pathlib.Path(_PRIOR_ROOT) / "rodent_reference_clips.h5" if _PRIOR_ROOT else None
)
if _REFERENCE_CLIPS is None:
    _REFERENCE_CLIPS_SKIP = "DMPO_PRIOR_ROOT is not set"
elif not _REFERENCE_CLIPS.exists():
    _REFERENCE_CLIPS_SKIP = f"{_REFERENCE_CLIPS} not found under DMPO_PRIOR_ROOT"
else:
    _REFERENCE_CLIPS_SKIP = ""

ALLOWED = ("foot_L", "foot_R", "hand_L", "hand_R", "vertebra_C1")


# ===========================================================================
# CPU: the geometric test against MuJoCo itself, on small hand-written models
# ===========================================================================

_TINY_XML = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="floor_body">
      <geom name="floor" type="plane" size="5 5 .1" contype="0" conaffinity="1"
            pos="0 0 -0.002" {floor_attrs}/>
    </body>
    <body name="wall_body">
      <geom name="wall" type="box" size=".5 .02 .1" pos="0 1 .1" contype="1" conaffinity="1"/>
    </body>
    <body name="walker-w" pos="0 0 .2">
      <freejoint name="root"/>
      <geom name="g_sphere" type="sphere" size=".03" contype="1" conaffinity="0" {cand_attrs}/>
      <body name="b_capsule-w" pos=".1 0 0"><joint type="ball"/>
        <geom name="g_capsule" type="capsule" size=".02 .04" contype="1" conaffinity="0" {cand_attrs}/></body>
      <body name="b_ellipsoid-w" pos="-.1 0 0"><joint type="ball"/>
        <geom name="g_ellipsoid" type="ellipsoid" size=".05 .02 .03" contype="1" conaffinity="0" {cand_attrs}/></body>
      <body name="b_box-w" pos="0 .1 0"><joint type="ball"/>
        <geom name="g_box" type="box" size=".04 .02 .03" contype="1" conaffinity="0" {cand_attrs}/></body>
      <body name="b_cylinder-w" pos="0 -.1 0"><joint type="ball"/>
        <geom name="g_cylinder" type="cylinder" size=".03 .05" contype="1" conaffinity="0" {cand_attrs}/></body>
      <body name="b_foot-w" pos=".1 .1 0"><joint type="ball"/>
        <geom name="g_foot" type="capsule" size=".01 .03" contype="1" conaffinity="0"/>
        <body name="b_toe-w" pos=".05 0 0"><joint type="ball"/>
          <geom name="g_toe" type="sphere" size=".01" contype="1" conaffinity="0"/></body></body>
      <body name="b_inert-w" pos="-.1 .1 0"><joint type="ball"/>
        <geom name="g_inert" type="sphere" size=".02" contype="0" conaffinity="0"/></body>
      <body name="b_excluded-w" pos="-.1 -.1 0"><joint type="ball"/>
        <geom name="g_excluded" type="sphere" size=".02" contype="1" conaffinity="0"/></body>
      <body name="b_paired-w" pos=".1 -.1 0"><joint type="ball"/>
        <geom name="g_paired" type="sphere" size=".02" contype="0" conaffinity="0"/></body>
    </body>
  </worldbody>
  <contact>
    <exclude body1="floor_body" body2="b_excluded-w"/>
    {pair}
  </contact>
</mujoco>
"""


def _tiny_model(floor_attrs="", cand_attrs="", pair=True, pair_margin=0.0):
    xml = _TINY_XML.format(
        floor_attrs=floor_attrs,
        cand_attrs=cand_attrs,
        pair=(
            f'<pair geom1="g_paired" geom2="floor" margin="{pair_margin}"/>'
            if pair
            else ""
        ),
    )
    return mujoco.MjModel.from_xml_string(xml)


def _tiny_plan(model, margin=0.0, allowed=("b_foot",)):
    return fc.plan_contact_termination(model, "walker-w", allowed, margin, suffix="-w")


def _random_tiny_qpos(model, n, seed):
    rng = np.random.default_rng(seed)
    out = np.tile(model.qpos0, (n, 1))
    for q in out:
        q[0:3] = (*rng.uniform(-0.3, 0.3, 2), rng.uniform(-0.02, 0.2))
        quats = rng.normal(size=(1 + (model.nq - 7) // 4, 4))
        quats /= np.linalg.norm(quats, axis=1, keepdims=True)
        q[3:] = quats.ravel()
    return out


def _cpu_touching(model, data, geom_ids, floor_id, margin):
    """Bool per candidate: MuJoCo's own contact list has a (geom, floor) pair under ``margin``."""
    touching = np.zeros(len(geom_ids), bool)
    index = {int(g): i for i, g in enumerate(geom_ids)}
    for c in data.contact[: data.ncon]:
        pair = {int(c.geom1), int(c.geom2)}
        if floor_id in pair:
            other = (pair - {floor_id}).pop() if len(pair) == 2 else None
            if other in index and c.dist < margin:
                touching[index[other]] = True
    return touching


@pytest.mark.parametrize(
    "floor_attrs,cand_attrs,margin",
    [
        pytest.param("", "", 0.0, id="no-margins"),
        pytest.param('margin="0.01"', "", 0.0, id="floor-geom-margin"),
        pytest.param("", 'margin="0.008"', 0.0, id="candidate-geom-margin"),
        # Both geoms carry a margin: MuJoCo lists contacts out to their SUM, whichever
        # priority is higher (not the max, and priority does not pick one).
        pytest.param('margin="0.01"', 'margin="0.03"', 0.0, id="margins-add"),
        pytest.param(
            'margin="0.01" priority="1"',
            'margin="0.03"',
            0.0,
            id="margins-add-floor-priority",
        ),
        pytest.param(
            'margin="0.03"',
            'margin="0.01" priority="1"',
            0.0,
            id="margins-add-candidate-priority",
        ),
    ],
)
def test_floor_test_matches_mujoco_contacts_per_geom(floor_attrs, cand_attrs, margin):
    """Per candidate geom and pose, ``clearance < 0`` iff MuJoCo lists a (geom, floor)
    contact -- for all five primitive types, with every margin / priority combination --
    and ``clearance`` equals the contact ``dist`` to float precision."""
    model = _tiny_model(floor_attrs, cand_attrs)
    contact_plan = _tiny_plan(model, margin)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    data = mujoco.MjData(model)

    # Candidates: the five shapes + the explicit <pair> geom. NOT the allowed foot/toe,
    # the contype-0 geom, the excluded body's geom, or anything of the static wall.
    assert set(contact_plan.geom_names) == {
        "g_sphere",
        "g_capsule",
        "g_ellipsoid",
        "g_box",
        "g_cylinder",
        "g_paired",
    }

    fired = disagreements = n_cases = 0
    worst = 0.0
    for q in _random_tiny_qpos(model, 400, seed=0):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        clr = fc.clearance(contact_plan, data.geom_xpos, data.geom_xmat, np)
        cpu = _cpu_touching(model, data, contact_plan.geom_ids, floor_id, np.inf)
        mine = np.asarray(clr) < 0
        n_cases += len(mine)
        fired += int(mine.sum())
        for i in np.flatnonzero(mine != cpu):
            disagreements += 1
            worst = max(worst, abs(float(clr[i])))
        # clearance IS the contact dist (+ margin offsets) wherever a contact exists.
        contacts = {}
        for c in data.contact[: data.ncon]:
            pair = {int(c.geom1), int(c.geom2)}
            if floor_id in pair and len(pair) == 2:
                contacts[(pair - {floor_id}).pop()] = min(
                    contacts.get((pair - {floor_id}).pop(), np.inf), float(c.dist)
                )
        for i, g in enumerate(contact_plan.geom_ids):
            if int(g) in contacts:
                margin_applied = contact_plan.trigger_z[i] - contact_plan.floor_z
                assert float(clr[i]) + margin_applied == pytest.approx(
                    contacts[int(g)], abs=1e-6
                )
    assert fired > 0.15 * n_cases, "the pose sampler must exercise both outcomes"
    assert fired < 0.85 * n_cases
    assert disagreements == 0 or worst < 1e-6, (disagreements, worst)


def test_param_margin_is_added_to_the_geom_margin():
    """``margin`` extends the contact distance: the contact_plan matches a CPU model whose floor
    carries that margin as its own geom margin (and whose explicit pair carries it too)."""
    contact_plan = _tiny_plan(_tiny_model(), margin=0.01)
    ref_model = _tiny_model(floor_attrs='margin="0.01"', pair_margin=0.01)
    ref_plan = _tiny_plan(ref_model, margin=0.0)
    np.testing.assert_allclose(contact_plan.trigger_z, ref_plan.trigger_z, atol=1e-7)
    floor_id = mujoco.mj_name2id(ref_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    data = mujoco.MjData(ref_model)
    for q in _random_tiny_qpos(ref_model, 150, seed=3):
        data.qpos[:] = q
        mujoco.mj_forward(ref_model, data)
        mine = (
            np.asarray(fc.clearance(contact_plan, data.geom_xpos, data.geom_xmat, np))
            < 0
        )
        cpu = _cpu_touching(ref_model, data, contact_plan.geom_ids, floor_id, np.inf)
        assert (mine == cpu).all()


def test_plan_groups_and_allowed_subtree():
    contact_plan = _tiny_plan(_tiny_model())
    kinds = [k for k, _, _ in contact_plan.groups]
    assert kinds == sorted(kinds, key=fc._SUPPORTED_TYPES.index)
    assert sum(b - a for _, a, b in contact_plan.groups) == contact_plan.n_candidates
    model = _tiny_model()
    # Allowed = the foot AND its child toe (dm_control's find_all('geom') is recursive).
    allowed = {model.geom(int(g)).name for g in contact_plan.allowed_geom_ids}
    assert allowed == {"g_foot", "g_toe"}
    assert contact_plan.allowed_bodies == ("b_foot-w",)


def test_unmatched_allowed_body_raises():
    with pytest.raises(ValueError, match="nope"):
        _tiny_plan(_tiny_model(), allowed=("b_foot", "nope"))


def test_allowed_body_outside_the_walker_raises():
    with pytest.raises(ValueError, match="wall_body"):
        fc.plan_contact_termination(
            _tiny_model(), "walker-w", ("wall_body",), 0.0, suffix="-w"
        )


def test_unsupported_candidate_type_raises_naming_the_geom():
    xml = (
        _TINY_XML.format(floor_attrs="", cand_attrs="", pair="")
        .replace(
            '<geom name="g_inert" type="sphere" size=".02" contype="0" conaffinity="0"/>',
            '<geom name="g_mesh" type="mesh" mesh="m" contype="1" conaffinity="0"/>',
        )
        .replace(
            "<worldbody>",
            '<asset><mesh name="m" vertex="0 0 0 .01 0 0 0 .01 0 0 0 .01"/></asset><worldbody>',
            1,
        )
    )
    with pytest.raises(ValueError, match="g_mesh"):
        _tiny_plan(mujoco.MjModel.from_xml_string(xml))


def test_unsupported_type_that_cannot_touch_the_floor_is_fine():
    xml = (
        _TINY_XML.format(floor_attrs="", cand_attrs="", pair="")
        .replace(
            '<geom name="g_inert" type="sphere" size=".02" contype="0" conaffinity="0"/>',
            '<geom name="g_mesh" type="mesh" mesh="m" contype="0" conaffinity="0"/>',
        )
        .replace(
            "<worldbody>",
            '<asset><mesh name="m" vertex="0 0 0 .01 0 0 0 .01 0 0 0 .01"/></asset><worldbody>',
            1,
        )
    )
    contact_plan = _tiny_plan(mujoco.MjModel.from_xml_string(xml))
    assert "g_mesh" not in contact_plan.geom_names


def test_tilted_floor_raises():
    xml = _TINY_XML.format(floor_attrs='euler="0.3 0 0"', cand_attrs="", pair="")
    with pytest.raises(ValueError, match="horizontal"):
        _tiny_plan(mujoco.MjModel.from_xml_string(xml))


def test_no_candidate_raises():
    """If every geom is allowed or inert the criterion could never fire -- say so."""
    with pytest.raises(ValueError, match="could never fire"):
        fc.plan_contact_termination(
            _tiny_model(), "walker-w", ("walker",), 0.0, suffix="-w"
        )


def test_non_finite_margin_raises():
    with pytest.raises(ValueError, match="finite"):
        _tiny_plan(_tiny_model(), margin=float("nan"))


@pytest.mark.parametrize("kind", fc._SUPPORTED_TYPES)
def test_z_extent_is_exact_against_mj_geom_distance(kind):
    """The closed-form extent equals MuJoCo's own geom-to-plane distance, per type."""
    name = {
        int(mujoco.mjtGeom.mjGEOM_SPHERE): "g_sphere",
        int(mujoco.mjtGeom.mjGEOM_CAPSULE): "g_capsule",
        int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "g_ellipsoid",
        int(mujoco.mjtGeom.mjGEOM_BOX): "g_box",
        int(mujoco.mjtGeom.mjGEOM_CYLINDER): "g_cylinder",
    }[kind]
    model = _tiny_model(pair=False)
    contact_plan = _tiny_plan(model)
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    col = int(np.flatnonzero(contact_plan.geom_ids == gid)[0])
    data = mujoco.MjData(model)
    fromto = np.zeros(6)
    checked = 0
    for q in _random_tiny_qpos(model, 120, seed=7):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        dist = mujoco.mj_geomDistance(model, data, gid, floor_id, 1.0, fromto)
        if abs(dist) >= 0.9:
            continue
        mine = float(
            fc.clearance(contact_plan, data.geom_xpos, data.geom_xmat, np)[col]
        )
        # clearance is measured against the plane at z = -0.002 with no margins.
        assert mine == pytest.approx(dist, abs=1e-6), (name, mine, dist)
        checked += 1
    assert checked > 60


def test_clearance_is_traceable_and_agrees_in_float32():
    """The jax path (what the env runs) matches the float64 numpy path to float32 noise."""
    model = _tiny_model()
    contact_plan = _tiny_plan(model)
    data = mujoco.MjData(model)
    cpu_dev = jax.devices("cpu")[0]
    fn = jax.jit(lambda gp, gm: fc.clearance(contact_plan, gp, gm, jp), device=cpu_dev)
    for q in _random_tiny_qpos(model, 40, seed=11):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        ref = fc.clearance(contact_plan, data.geom_xpos, data.geom_xmat, np)
        got = np.asarray(
            fn(
                jp.asarray(data.geom_xpos, jp.float32),
                jp.asarray(data.geom_xmat.reshape(-1, 3, 3), jp.float32),
            )
        )
        np.testing.assert_allclose(got, ref, atol=2e-6)


# ===========================================================================
# CPU: config contract
# ===========================================================================


def test_default_config_does_not_carry_the_criterion():
    assert "contact_termination" not in mfv.default_config().termination_criteria
    assert "contact_termination" in mfv._registry.terminations


def test_default_allowed_bodies_are_dm_controls():
    assert fc.DEFAULT_ALLOWED_BODIES == ALLOWED


# ===========================================================================
# The production walker (built and traced on the CPU; executed on the GPU)
# ===========================================================================


def _build_env(contact=True, **contact_cfg) -> mfv.MazeForageVision:
    cfg = mfv.default_config()
    assert cfg.rescale_factor == 0.9  # the production rescale
    cfg.ctrl_dt = 0.01  # the production control step (5 physics substeps)
    if contact:
        cfg.termination_criteria["contact_termination"] = contact_cfg
    return mfv.MazeForageVision(config=cfg)


@pytest.fixture(scope="module")
def env():
    return _build_env()


@pytest.fixture(scope="module")
def off_env():
    return _build_env(contact=False)


@pytest.fixture(scope="module")
def jit_fns(env):
    return types.SimpleNamespace(
        reset=jax.jit(env.reset),
        step=jax.jit(env.step),
        forward=jax.jit(lambda d, q: mjx.forward(env.mjx_model, d.replace(qpos=q))),
        clearance=jax.jit(env.contact_clearance),
        fires=jax.jit(
            lambda d: env._registry.terminations["contact_termination"](env, d, {})
        ),
    )


# --- Matched bodies / geoms --------------------------------------------------


def test_matched_bodies_and_geoms(env):
    contact_plan = env._contact_plan
    model = env.mj_model
    assert contact_plan.allowed_bodies == tuple(f"{b}-rodent" for b in ALLOWED)
    allowed_names = {model.geom(int(g)).name for g in contact_plan.allowed_geom_ids}
    # dm_control's find_all('geom') is recursive: the toes, the fingers and the tail chain.
    for needed in (
        "foot_L_collision-rodent",
        "toe_L0_collision-rodent",
        "hand_R_collision-rodent",
        "finger_R2_collision-rodent",
        "vertebra_C1_collision-rodent",
        "vertebra_C30_collision-rodent",
    ):
        assert needed in allowed_names, needed
    # vertebra_C1 has no colliding geom in rodent_no_tail_collisions: nothing is lost by
    # allowing it, and nothing of the tail is a candidate either.
    tail = [
        g
        for g in contact_plan.allowed_geom_ids
        if "vertebra_C" in model.geom(int(g)).name
    ]
    assert tail and all(
        model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0 for g in tail
    )
    assert not [
        g
        for g in contact_plan.allowed_colliding_geom_ids
        if "vertebra_C" in model.geom(int(g)).name
    ]
    assert not [n for n in contact_plan.geom_names if "vertebra_C" in n]
    # Candidates: torso, pelvis, legs, arms, scapulae, spine spheres, skull, jaw; no
    # allowed geom, no treat, no wall, never the floor.
    assert set(contact_plan.geom_names).isdisjoint(allowed_names)
    assert not [
        n
        for n in contact_plan.geom_names
        if n.startswith(("maze_wall", "treat", "floor"))
    ]
    assert "collision_torso-rodent" in contact_plan.geom_names
    assert "lower_leg_L_collision-rodent" in contact_plan.geom_names
    assert "skull_0_collision-rodent" in contact_plan.geom_names
    assert contact_plan.floor_geom_ids.tolist() == [model.geom("floor").id]
    assert contact_plan.floor_z == pytest.approx(-0.005, abs=1e-6)
    # Every candidate collides with the floor under contype / conaffinity.
    for g in contact_plan.geom_ids:
        assert (
            model.geom_contype[g]
            & model.geom_conaffinity[contact_plan.floor_geom_ids[0]]
        )


def test_unmatched_allowed_body_raises_at_build_time():
    with pytest.raises(ValueError, match="hand_X"):
        _build_env(allowed_bodies=["foot_L", "hand_X"])


def test_an_empty_config_entry_raises_with_the_fix_in_the_message():
    cfg = mfv.default_config()
    # A yaml `contact_termination:` with no value.
    cfg.termination_criteria["contact_termination"] = None
    with pytest.raises(ValueError, match=r"contact_termination: \{\}"):
        mfv.MazeForageVision(config=cfg)


def test_contact_clearance_needs_the_criterion(off_env):
    state = jax.eval_shape(off_env.reset, jax.random.PRNGKey(0))
    with pytest.raises(RuntimeError, match="without"):
        off_env.contact_clearance(state.data)


def test_a_user_supplied_allowed_list_changes_the_candidates(env):
    plan_all = env._contact_plan
    small = _build_env(allowed_bodies=["foot_L", "foot_R"])
    # Hands and fingers are candidates now (hand/finger geoms), the vertebra chain no longer allowed.
    assert "hand_L_collision-rodent" in small._contact_plan.geom_names
    assert "finger_L0_collision-rodent" in small._contact_plan.geom_names
    assert "hand_L_collision-rodent" not in plan_all.geom_names


# --- Proxy correctness against CPU contacts ----------------------------------


def _sample_qpos(env, n_per_mode, seed):
    """Random walker poses: uniform SO(3), lying on a side, on its back, and upright;
    each at three height regimes (right at the floor, wide, anywhere) so the threshold
    is probed from both sides.  Returns ``(qpos, mode_names)``."""
    model = env.mj_model
    root = env._rodent_root_qpos
    rng = np.random.default_rng(seed)
    hinges = [
        (
            int(model.jnt_qposadr[j]),
            model.jnt_range[j] if model.jnt_limited[j] else (-1.0, 1.0),
        )
        for j in range(model.njnt)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        and model.jnt_qposadr[j] >= root + 7
    ]

    def quat_about(axis, angle, noise):
        axis = np.asarray(axis, float)
        axis = axis + noise * rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        a = angle + noise * rng.normal()
        return np.concatenate([[np.cos(a / 2)], np.sin(a / 2) * axis])

    def yaw(q):
        h = rng.uniform(-np.pi, np.pi) / 2
        qz = np.array([np.cos(h), 0, 0, np.sin(h)])
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, qz, q)
        return out

    modes = {
        "so3": lambda: (lambda g: g / np.linalg.norm(g))(rng.normal(size=4)),
        "side": lambda: yaw(
            quat_about([1, 0, 0], rng.choice([-1, 1]) * np.pi / 2, 0.25)
        ),
        "back": lambda: yaw(quat_about([1, 0, 0], np.pi, 0.25)),
        "upright": lambda: yaw(quat_about([0, 1, 0], 0.0, 0.2)),
    }
    qpos, names = [], []
    for mode, make_quat in modes.items():
        for i in range(n_per_mode):
            q = np.array(model.qpos0)
            q[root + 3 : root + 7] = make_quat()
            q[root : root + 2] = rng.uniform(-1, 1, 2)
            for adr, (lo, hi) in hinges:
                q[adr] = rng.uniform(lo, hi)
            q[root + 2] = 0.0
            qpos.append((q, i % 3))
            names.append(mode)
    return qpos, names


def _settle_heights(env, qpos_regime, seed):
    """Fix the root z of each sampled pose by regime: 0 = lowest candidate geom within +-1 cm
    of the floor (the threshold stress test), 1 = within -0.1 .. +0.3 m, 2 = root z anywhere in 0 .. 0.25."""
    model = env.mj_model
    contact_plan = env._contact_plan
    root = env._rodent_root_qpos
    floor_id = int(contact_plan.floor_geom_ids[0])
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    fromto = np.zeros(6)
    out = []
    for q, regime in qpos_regime:
        q = q.copy()
        if regime == 2:
            q[root + 2] = rng.uniform(0.0, 0.25)
        else:
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            lowest = min(
                mujoco.mj_geomDistance(model, data, int(g), floor_id, 1.0, fromto)
                for g in contact_plan.geom_ids
            )
            target = rng.uniform(-0.01, 0.01) if regime == 0 else rng.uniform(-0.1, 0.3)
            q[root + 2] = target - lowest
        out.append(q)
    return np.stack(out)


def _cpu_fires(model, data, contact_plan, margin):
    """The dm_control rule on MuJoCo CPU: any candidate-geom x floor contact under ``margin``."""
    cand = set(int(g) for g in contact_plan.geom_ids)
    floors = set(int(g) for g in contact_plan.floor_geom_ids)
    hits = []
    for c in data.contact[: data.ncon]:
        g1, g2 = int(c.geom1), int(c.geom2)
        if (
            (g1 in cand and g2 in floors) or (g2 in cand and g1 in floors)
        ) and c.dist < margin:
            hits.append((g1 if g1 in cand else g2, float(c.dist)))
    return bool(hits), hits


def _proxy_rows(env, fns, qpos, margin_for_cpu):
    """Per pose: the GPU criterion and its min clearance, MuJoCo CPU's contact-list verdict under
    ``margin_for_cpu``, and the exact CPU ``mj_geomDistance`` minimum over the candidate geoms."""
    model = env.mj_model
    contact_plan = env._contact_plan
    floor = int(contact_plan.floor_geom_ids[0])
    cpu = mujoco.MjData(model)
    state = fns.reset(jax.random.PRNGKey(0))
    fromto = np.zeros(6)
    rows = []
    for q in qpos:
        cpu.qpos[:] = q
        mujoco.mj_forward(model, cpu)
        fired_cpu, _ = _cpu_fires(model, cpu, contact_plan, margin_for_cpu)
        gd = min(
            mujoco.mj_geomDistance(model, cpu, int(g), floor, 1.0, fromto)
            for g in contact_plan.geom_ids
        )
        data = fns.forward(state.data, jp.asarray(q, jp.float32))
        clr = np.asarray(fns.clearance(data))
        fired = bool(np.asarray(fns.fires(data)))
        assert fired == bool(
            (clr < 0).any()
        )  # the criterion IS "some candidate is negative"
        rows.append(
            types.SimpleNamespace(
                cpu=fired_cpu,
                gpu=fired,
                clr=float(clr.min()),
                gd=float(gd),
                xpos_err=float(
                    np.abs(np.asarray(data.geom_xpos) - cpu.geom_xpos).max()
                ),
            )
        )
    return rows


@needs_gpu
def test_proxy_matches_cpu_contacts_at_the_production_rescale(env, jit_fns):
    """>= 99 % agreement with CPU ``mj_forward`` contacts over 800 random, fallen and lying
    poses (a third of them within +-1 cm of the threshold); every disagreement must sit within
    1e-4 m of it.  Also agrees with MuJoCo's exact geom-to-plane distance."""
    sampled, modes = _sample_qpos(env, 200, seed=0)
    qpos = _settle_heights(env, sampled, seed=1)
    assert len(qpos) >= 600
    rows = _proxy_rows(env, jit_fns, qpos, margin_for_cpu=0.0)

    cpu = np.array([r.cpu for r in rows])
    gpu = np.array([r.gpu for r in rows])
    agree = (cpu == gpu).mean()
    # Both classes are well populated, in every posture family.
    assert 0.25 < cpu.mean() < 0.85
    for mode in set(modes):
        sel = np.array([m == mode for m in modes])
        assert 0.1 < cpu[sel].mean() < 0.95, mode
    bad = [(i, modes[i], rows[i].clr) for i in np.flatnonzero(cpu != gpu)]
    print(
        f"proxy vs CPU contact list: {len(rows)} poses, {cpu.mean():.1%} fire, agreement "
        f"{agree:.4%}, {len(bad)} disagreements: {bad}"
    )
    assert agree >= 0.99
    for i, mode, min_clearance in bad:
        assert abs(min_clearance) < 1e-4, (i, mode, min_clearance)

    # And against MuJoCo's exact distance (not just its contact list): same sign, same value.
    gd = np.array([r.gd for r in rows])
    clr = np.array([r.clr for r in rows])
    in_range = np.abs(gd) < 0.9  # mj_geomDistance saturates at distmax = 1
    assert in_range.mean() > 0.9
    assert np.abs(clr - gd)[in_range].max() < 5e-5, np.abs(clr - gd)[in_range].max()
    wrong_sign = np.flatnonzero((clr < 0) != (gd < 0))
    assert all(abs(gd[i]) < 1e-4 for i in wrong_sign)
    print(
        f"clearance vs mj_geomDistance: max |diff| {np.abs(clr - gd)[in_range].max():.2e} m over "
        f"{in_range.sum()} poses; GPU geom_xpos vs CPU max {max(r.xpos_err for r in rows):.2e} m"
    )
    # The GPU kinematics themselves are the CPU's, to float32.
    assert max(r.xpos_err for r in rows) < 5e-5


@needs_gpu
def test_proxy_with_a_margin_is_exact_and_never_misses_a_mujoco_contact():
    """``margin = 4 mm``.  The oracle is MuJoCo's exact distance, not its contact list: with a
    positive margin MuJoCo's broadphase drops some tiny tilted ellipsoids (the skull / jaw tooth
    geoms) before they reach ``margin`` -- measured: a 2 x 1.2 x 6.3 mm ellipsoid is listed out to
    ~3.0 mm but not at 3.5 mm under a 4 mm margin -- so the list is a subset of the true
    ``dist < margin`` set.  The criterion must be that exact set, and so a superset of the list.
    (At margin 0, the production setting, the two coincide: see the test above.)"""
    margin = 0.004
    menv = _build_env(margin=margin)
    assert menv._contact_plan.margin == margin
    sampled, modes = _sample_qpos(menv, 100, seed=5)
    qpos = _settle_heights(menv, sampled, seed=6)
    fns = types.SimpleNamespace(
        reset=jax.jit(menv.reset),
        forward=jax.jit(lambda d, q: mjx.forward(menv.mjx_model, d.replace(qpos=q))),
        clearance=jax.jit(menv.contact_clearance),
        fires=jax.jit(
            lambda d: menv._registry.terminations["contact_termination"](menv, d, {})
        ),
    )
    model = menv.mj_model
    floor_id = int(menv._contact_plan.floor_geom_ids[0])
    saved = float(model.geom_margin[floor_id])
    model.geom_margin[floor_id] = margin  # CPU contacts are listed up to this margin
    try:
        rows = _proxy_rows(menv, fns, qpos, margin_for_cpu=margin)
    finally:
        model.geom_margin[floor_id] = saved
    cpu = np.array([r.cpu for r in rows])
    gpu = np.array([r.gpu for r in rows])
    gd = np.array([r.gd for r in rows])
    # 1. no MuJoCo-listed contact is missed.
    assert not (cpu & ~gpu).any()
    # 2. the criterion equals the exact "some candidate within margin of the floor".
    exact = gd < margin
    off = np.flatnonzero(exact != gpu)
    assert all(abs(gd[i] - margin) < 1e-4 for i in off), [(i, gd[i]) for i in off]
    extra = np.flatnonzero(gpu & ~cpu)
    # 3. whatever MuJoCo's list dropped was in the margin band, not a penetration.
    assert all(gd[i] >= -1e-4 for i in extra), [(i, gd[i]) for i in extra]
    print(
        f"margin {margin}: {len(rows)} poses, {gpu.mean():.1%} fire; exact-set mismatches {len(off)}; "
        f"MuJoCo list omits {len(extra)} poses the criterion flags (all in the 0..{margin * 1e3:.0f} mm band)"
    )


# --- Standing spawn pose -------------------------------------------------------


@needs_gpu
def test_reset_pose_does_not_fire_and_has_clearance(env, jit_fns):
    state = jit_fns.reset(jax.random.PRNGKey(0))
    clr = np.asarray(jit_fns.clearance(state.data))
    names = env._contact_plan.geom_names
    print(
        f"reset pose: nearest non-allowed geom {names[int(clr.argmin())]} at "
        f"{clr.min() * 1e3:.2f} mm above the floor"
    )
    assert float(state.metrics["terminations/contact_termination"]) == 0.0
    assert float(state.done) == 0.0
    assert clr.min() > 0.005  # the standing pose clears the floor by >= 5 mm
    keys = jax.random.split(jax.random.PRNGKey(1), 4)
    batch = jax.jit(jax.vmap(env.reset))(keys)
    assert (
        np.asarray(batch.metrics["terminations/contact_termination"]).tolist()
        == [0.0] * 4
    )


@needs_gpu
def test_zero_torque_rat_sags_onto_a_limb_geom_and_fires_after_the_reset_pose_is_clean(
    env, jit_fns
):
    """With ``torque_actuators`` the null action is ZERO TORQUE: the rat is limp, and the standing
    spawn pose is not held (torso z 0.064 -> 0.050 in 6 control steps).  Measured: clean through
    step 5 (nearest non-allowed geom 1.4 mm above the floor), then the left lower leg touches.
    So 'no fire over the first 10 zero-action steps' does NOT hold -- by physics, not by the
    criterion: a trained policy holds the pose with its own torques."""
    state = jit_fns.reset(jax.random.PRNGKey(0))
    zero = jp.zeros(env.action_size)
    names = env._contact_plan.geom_names
    first, culprit = None, None
    for t in range(1, 11):
        state = jit_fns.step(state, zero)
        clr = np.asarray(jit_fns.clearance(state.data))
        fired = float(state.metrics["terminations/contact_termination"])
        assert fired == float((clr < 0).any())
        if fired and first is None:
            first, culprit = t, names[int(clr.argmin())]
    print(f"zero-torque rollout: first fire at step {first} on {culprit}")
    assert first is not None and 3 <= first <= 10
    assert not any(s in culprit for s in ("foot", "hand", "finger", "toe"))


# --- Firing / non-firing postures ------------------------------------------------


def _root_pose(env, state, z=None, quat=None):
    root = env._rodent_root_qpos
    qpos = state.data.qpos
    if z is not None:
        qpos = qpos.at[root + 2].set(z)
    if quat is not None:
        qpos = qpos.at[root + 3 : root + 7].set(jp.asarray(quat, jp.float32))
    return qpos


def _lowered(env, jit_fns, state, target, geom=None, quat=None):
    """``qpos`` of ``state`` (optionally re-oriented) with the root z solved so that the lowest
    candidate geom -- or the named ``geom`` -- sits ``target`` metres above its floor trigger
    (negative = through the floor).  Exact: the floor is horizontal, so a root z shift moves
    every geom by the same amount."""
    qpos = _root_pose(env, state, quat=quat)
    clr = np.asarray(jit_fns.clearance(jit_fns.forward(state.data, qpos)))
    ref = clr.min() if geom is None else clr[env._contact_plan.geom_names.index(geom)]
    return qpos.at[env._rodent_root_qpos + 2].add(target - float(ref))


def _fires(jit_fns, state, qpos):
    return bool(np.asarray(jit_fns.fires(jit_fns.forward(state.data, qpos))))


@needs_gpu
def test_torso_lowered_onto_the_floor_fires_on_that_step(env, jit_fns):
    state = jit_fns.reset(jax.random.PRNGKey(2))
    torso = "collision_torso-rodent"
    # The torso ellipsoid 6 mm through the floor fires; 3 mm above it (and everything else
    # that is lower in the standing pose even further above) does not.
    qpos = _lowered(env, jit_fns, state, -0.006, geom=torso)
    assert _fires(jit_fns, state, qpos)
    stepped = jit_fns.step(
        state.replace(data=state.data.replace(qpos=qpos)), jp.zeros(env.action_size)
    )
    assert float(stepped.metrics["terminations/contact_termination"]) == 1.0
    assert float(stepped.metrics["terminations/any"]) == 1.0
    assert float(stepped.done) == 1.0
    # Only the contact criterion has fired: the posture ones need `patience` steps.
    assert float(stepped.metrics["terminations/belly_up"]) == 0.0
    assert float(stepped.metrics["terminations/torso_too_low"]) == 0.0
    lifted = _lowered(env, jit_fns, state, +0.003)  # lowest non-allowed geom 3 mm up
    assert not _fires(jit_fns, state, lifted)


@needs_gpu
def test_rolling_onto_the_back_fires_and_the_same_pose_lifted_does_not(env, jit_fns):
    state = jit_fns.reset(jax.random.PRNGKey(3))
    inverted = [0.0, 1.0, 0.0, 0.0]  # 180 deg roll about x: belly up
    assert not _fires(
        jit_fns, state, _lowered(env, jit_fns, state, +0.004, quat=inverted)
    )
    qpos = _lowered(env, jit_fns, state, -0.004, quat=inverted)
    data = jit_fns.forward(state.data, qpos)
    assert bool(np.asarray(jit_fns.fires(data)))
    clr = np.asarray(jit_fns.clearance(data))
    touching = [env._contact_plan.geom_names[j] for j in np.flatnonzero(clr < 0)]
    print(f"belly-up, lowest geom 4 mm through the floor: touching {touching}")
    # the geoms that meet the floor under an upturned rat are the back, never the paws
    assert any(
        "torso" in n or "skull" in n or "vertebra" in n or "pelvis" in n
        for n in touching
    )
    assert not any(s in n for n in touching for s in ("foot", "hand", "finger", "toe"))


@needs_gpu
def test_lying_on_its_side_fires(env, jit_fns):
    state = jit_fns.reset(jax.random.PRNGKey(4))
    side = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]  # 90 deg roll
    assert not _fires(jit_fns, state, _lowered(env, jit_fns, state, +0.004, quat=side))
    assert _fires(jit_fns, state, _lowered(env, jit_fns, state, -0.004, quat=side))


@needs_gpu
def test_only_allowed_geoms_touching_does_not_fire(env, jit_fns):
    """Drive the standing rat down until its feet and hands are well below the floor: CPU
    MuJoCo lists those contacts, and the criterion stays silent because none belongs to a
    candidate geom."""
    model = env.mj_model
    contact_plan = env._contact_plan
    state = jit_fns.reset(jax.random.PRNGKey(5))
    cpu = mujoco.MjData(model)
    floor_id = int(contact_plan.floor_geom_ids[0])
    allowed_colliding = set(int(g) for g in contact_plan.allowed_colliding_geom_ids)
    assert allowed_colliding, "feet / hands / toes / fingers do collide with the floor"
    q0 = np.asarray(state.data.qpos, np.float64)
    root = env._rodent_root_qpos
    # The reset pose's own geometry: where, relative to the root, do paws and candidates bottom out?
    cpu.qpos[:] = q0
    mujoco.mj_forward(model, cpu)
    fromto = np.zeros(6)
    d_allowed = {
        g: mujoco.mj_geomDistance(model, cpu, g, floor_id, 1.0, fromto)
        for g in allowed_colliding
    }
    d_cand = float(np.asarray(jit_fns.clearance(state.data)).min())
    paw_bottom = min(d_allowed.values())
    print(
        f"reset pose: lowest allowed geom {paw_bottom * 1e3:.2f} mm, lowest candidate {d_cand * 1e3:.2f} mm above the floor"
    )
    # In the standing pose the paws are the lowest part of the body, by more than the gap
    # needed to push them through the floor without a candidate following.
    assert paw_bottom < d_cand
    # Shift the whole rat down by just under that gap: paws penetrate, nothing else does.
    shift = 0.5 * (paw_bottom + d_cand)
    q = q0.copy()
    q[root + 2] -= (
        shift + 0.0
    )  # paws now -(shift - paw_bottom) below, candidates still above
    cpu.qpos[:] = q
    mujoco.mj_forward(model, cpu)
    fired_cpu, _ = _cpu_fires(model, cpu, contact_plan, 0.0)
    paw_contacts = [
        c
        for c in cpu.contact[: cpu.ncon]
        if floor_id in (c.geom1, c.geom2)
        and (c.geom2 if c.geom1 == floor_id else c.geom1) in allowed_colliding
    ]
    assert paw_contacts, (
        "the paws must really be in contact for this test to mean anything"
    )
    assert not fired_cpu
    data = jit_fns.forward(state.data, jp.asarray(q, jp.float32))
    assert not bool(np.asarray(jit_fns.fires(data)))


# --- Metrics ---------------------------------------------------------------------


@needs_gpu
def test_metric_is_emitted_and_feeds_terminations_any(env, jit_fns):
    key = jax.random.PRNGKey(6)
    state = jit_fns.reset(key)
    assert "terminations/contact_termination" in state.metrics
    shapes = jax.eval_shape(env.reset, key).metrics
    assert shapes["terminations/contact_termination"].dtype == jp.float32
    # The metric keys at reset and at step are the same set (the wrapper stack needs that).
    step_metrics = jax.eval_shape(
        env.step,
        jax.eval_shape(env.reset, key),
        jax.ShapeDtypeStruct((env.action_size,), jp.float32),
    ).metrics
    assert set(step_metrics) == set(shapes)
    # any = OR of every criterion: contact alone makes `any` 1.
    lowered = jit_fns.step(
        state.replace(
            data=state.data.replace(qpos=_lowered(env, jit_fns, state, -0.006))
        ),
        jp.zeros(env.action_size),
    )
    flags = {
        k: float(v) for k, v in lowered.metrics.items() if k.startswith("terminations/")
    }
    assert flags["terminations/contact_termination"] == 1.0
    assert flags["terminations/any"] == 1.0
    others = [
        v
        for k, v in flags.items()
        if k not in ("terminations/contact_termination", "terminations/any")
    ]
    assert max(others) == 0.0


@needs_gpu
def test_a_firing_world_does_not_leak_into_its_neighbours_under_vmap(env):
    """The reason the criterion is geometric: per-world data, no shared contact buffer."""
    keys = jax.random.split(jax.random.PRNGKey(7), 4)
    state = jax.jit(jax.vmap(env.reset))(keys)
    root = env._rodent_root_qpos
    clr = np.asarray(
        jax.jit(jax.vmap(env.contact_clearance))(state.data)
    )  # [4, n_candidates]
    assert (clr.min(axis=1) > 0.005).all()
    # world 2 sinks 6 mm through the floor; the other three stay where they are
    qpos = state.data.qpos.at[2, root + 2].add(-0.006 - float(clr[2].min()))
    state = state.replace(data=state.data.replace(qpos=qpos))
    out = jax.jit(jax.vmap(env.step))(state, jp.zeros((4, env.action_size)))
    assert np.asarray(out.metrics["terminations/contact_termination"]).tolist() == [
        0.0,
        0.0,
        1.0,
        0.0,
    ]
    assert np.asarray(out.done).tolist() == [0.0, 0.0, 1.0, 0.0]


# --- Real rat motion (calibration, informational) -------------------------------------


@needs_gpu
@pytest.mark.skipif(bool(_REFERENCE_CLIPS_SKIP), reason=_REFERENCE_CLIPS_SKIP)
def test_reference_motion_clearance_is_reported(env):
    """Replays real rat motion through the criterion on CPU kinematics and reports how
    close the nearest non-allowed geom comes to the floor.  The retargeted reference is
    a fit, not physics, so momentary floor penetration is not itself a failure; the number
    printed here is what tells us whether a gait that imitates it would trip the flag."""
    import h5py

    with h5py.File(_REFERENCE_CLIPS, "r") as handle:
        reference = handle["qpos"][::200].astype(np.float64)
    model, contact_plan = env.mj_model, env._contact_plan
    data = mujoco.MjData(model)
    root = env._rodent_root_qpos
    lows = np.empty(len(reference))
    for i, frame in enumerate(reference):
        data.qpos[root:] = frame
        mujoco.mj_forward(model, data)
        lows[i] = fc.clearance(contact_plan, data.geom_xpos, data.geom_xmat, np).min()
    print(
        f"reference motion ({len(lows)} frames): nearest non-allowed geom above the floor "
        f"p0 {lows.min() * 1e3:.2f} p5 {np.percentile(lows, 5) * 1e3:.2f} "
        f"p50 {np.median(lows) * 1e3:.2f} mm; fires on {(lows < 0).mean():.2%} of frames"
    )
    assert np.isfinite(lows).all()


# --- Flag-off guard: the default env is the one before the criterion existed ------

_REF_REV, _REF_SKIP = ref_tools.reference_rev("contact_termination")


@pytest.fixture(scope="module")
def ref_modules():
    """The maze and spout env modules of the commit before ``contact_termination`` existed."""
    if _REF_REV is None:
        pytest.skip(_REF_SKIP)
    return ref_tools.load_reference(_REF_REV)


def _ref_env(ref_modules):
    ref_cfg = ref_modules.maze.default_config()
    ref_cfg.ctrl_dt = 0.01  # same production control step as `off_env`
    return ref_modules.maze.MazeForageVision(config=ref_cfg)


def test_flag_off_traces_identically_to_the_reference(ref_modules, off_env):
    """Without ``contact_termination`` in the config: the same model, the same
    output structure, and ``reset`` / ``step`` trace to the same jaxpr with the same
    constants as before the criterion existed."""
    assert "contact_termination" not in off_env._config.termination_criteria
    assert off_env._contact_plan is None
    ref_env = _ref_env(ref_modules)
    assert not ref_tools.model_mismatches(off_env.mj_model, ref_env.mj_model)
    assert off_env.info_reset_keys == ref_env.info_reset_keys == mfv.INFO_RESET_KEYS
    assert set(ref_modules.maze.default_config().termination_criteria) == set(
        mfv.default_config().termination_criteria
    )

    key = jax.random.PRNGKey(0)
    single = jax.eval_shape(ref_env.reset, key)
    assert not [k for k in single.metrics if "contact" in k]
    action = jax.ShapeDtypeStruct((ref_env.action_size,), jp.float32)
    assert ref_tools.shape_tree(
        jax.eval_shape(off_env.reset, key)
    ) == ref_tools.shape_tree(single)
    assert ref_tools.shape_tree(
        jax.eval_shape(off_env.step, single, action)
    ) == ref_tools.shape_tree(jax.eval_shape(ref_env.step, single, action))
    hint = f"the default env no longer traces like {_REF_REV} (see maze_reference)"
    assert not ref_tools.jaxpr_mismatch(off_env.reset, ref_env.reset, key), hint
    assert not ref_tools.jaxpr_mismatch(off_env.step, ref_env.step, single, action), (
        hint
    )


def test_the_trace_comparison_sees_the_criterion(ref_modules, env):
    """The guard above has teeth: with the criterion on, ``reset`` (which runs the
    termination check) traces differently."""
    ref_env = _ref_env(ref_modules)
    assert ref_tools.jaxpr_mismatch(env.reset, ref_env.reset, jax.random.PRNGKey(0))


@needs_gpu
def test_flag_off_runs_identically_to_the_reference(ref_modules, off_env):
    """The same reset data bitwise, and the same step data to the warp solver's
    run-to-run noise, leaf for leaf, over five random actions."""
    ref_env = _ref_env(ref_modules)
    key = jax.random.PRNGKey(0)
    rng = np.random.default_rng(1)
    actions = [
        jp.asarray(rng.uniform(-1, 1, off_env.action_size), jp.float32)
        for _ in range(5)
    ]
    runs = []
    for e in (off_env, ref_env):
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


def test_flag_on_leaves_every_other_metric_and_the_model_unchanged(env, off_env):
    """Enabling the criterion adds exactly one metric and changes nothing about the model."""
    assert not ref_tools.model_mismatches(env.mj_model, off_env.mj_model)
    key = jax.random.PRNGKey(0)
    on, off = jax.eval_shape(env.reset, key), jax.eval_shape(off_env.reset, key)
    assert set(on.metrics) - set(off.metrics) == {"terminations/contact_termination"}
    assert set(on.info) == set(off.info)


def test_a_config_entry_builds_the_criterion_through_the_task_registry():
    """An ``env_config`` that lists the criterion, passed the way the DMPO entry
    point passes it (``tasks.load(..., config_overrides=...)`` with the keys the
    default config knows), registers the criterion with its defaults; this also
    shows the empty-valued key is accepted by the locked default config."""
    from omegaconf import OmegaConf

    from vnl_playground import tasks

    defaults = mfv.default_config().termination_criteria.to_dict()
    cfg = OmegaConf.create(
        {
            "env_name": "RodentMazeForageVision",
            "env_config": {
                "env_name": "RodentMazeForageVision",
                "ctrl_dt": 0.01,
                "termination_criteria": {
                    "belly_up": defaults["belly_up"],
                    "torso_too_low": defaults["torso_too_low"],
                    "all_treats_collected": {},
                    "nan_termination": {},
                    "contact_termination": {},
                },
            },
        }
    )
    env_args = OmegaConf.to_container(cfg["env_config"], resolve=True)
    valid = set(tasks.get_default_config(str(cfg["env_name"])).keys())
    env_args = {
        k: v
        for k, v in env_args.items()
        if k in valid and k not in ("env_name", "flatten_obs")
    }
    built = tasks.load(
        str(cfg["env_name"]), flatten_obs=False, config_overrides=env_args
    )
    raw = built.env if hasattr(built, "env") else built
    assert set(raw._config.termination_criteria) == {
        "belly_up",
        "torso_too_low",
        "all_treats_collected",
        "nan_termination",
        "contact_termination",
    }
    assert raw._contact_plan is not None
    assert raw._contact_plan.allowed_bodies == tuple(f"{b}-rodent" for b in ALLOWED)
    assert raw._contact_plan.margin == 0.0
