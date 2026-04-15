# s15: IK-Driven Moving-Shoulder Imitation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a training variant (`train_mouse_janelia_sigmoid_moving_shoulder.py`) that uses the STAC-mjx v16 IK solution to kinematically drive the shoulder tx/ty/tz every step, while the muscle policy learns only the 4 hinge-joint activations.

**Architecture:** Build a new muscle-model XML (existing `mouse_forelimb_right.xml` + 3 new slide joints on `clavicle`, no actuators on them), point a subclassed imitation env at it, override `qpos[:3]` and `qvel[:3]` from the reference IK after every `mjx_env.step`, and mask those 3 dims out of the `joints` / `joints_vel` reward and `pose_error` termination. Reference data comes from per-trial IK h5 files produced by `run_stac_janelia_moving_shoulder_v16.py`.

**Tech Stack:** Python 3.11, MuJoCo 3.x (`mujoco`, `mujoco.mjx`), JAX, Brax PPO, `mujoco_playground`, the existing `MouseImitation` / `MouseBaseEnv` stack in `vnl_playground/tasks/mouse/`.

**Spec:** `docs/superpowers/specs/2026-04-15-s15-moving-shoulder-ik-driven-design.md`

---

## File Structure

**Create:**
- `vnl_playground/tasks/mouse/xmls/mouse_forelimb_right_moving_shoulder_ik.xml` — muscle model with 3 IK-driven slide joints on clavicle.
- `vnl_playground/tasks/mouse/reference_data_moving_shoulder/` — directory of symlinks to v16 IK h5 files.
- `vnl_playground/tasks/mouse/imitation_moving_shoulder.py` — `MouseImitationMovingShoulder` subclass.
- `train_mouse_janelia_sigmoid_moving_shoulder.py` — training entry point (top-level).
- `scripts/sanity_check_s15_moving_shoulder.py` — standalone sanity test (env instantiation + override correctness).

**Modify:**
- `vnl_playground/tasks/mouse/consts.py` — add two path constants.

**Do not touch:**
- `base.py`, `imitation.py`, `reference_clips.py`, `train_mouse_janelia_sigmoid_normal.py`.

---

## Task 1: Add path constants

**Files:**
- Modify: `vnl_playground/tasks/mouse/consts.py`

- [ ] **Step 1: Add two new constants**

Append to `vnl_playground/tasks/mouse/consts.py` (currently ends at line 16):

```python
JANELIA_MOUSE_MOVING_SHOULDER_IK_XML_PATH = (
    MOUSE_PATH / "xmls" / "mouse_forelimb_right_moving_shoulder_ik.xml"
)
MOUSE_REFERENCE_DATA_MOVING_SHOULDER_PATH = (
    MOUSE_PATH / "reference_data_moving_shoulder"
)
```

- [ ] **Step 2: Verify the import resolves (paths don't need to exist yet)**

Run: `python -c "from vnl_playground.tasks.mouse.consts import JANELIA_MOUSE_MOVING_SHOULDER_IK_XML_PATH, MOUSE_REFERENCE_DATA_MOVING_SHOULDER_PATH; print(JANELIA_MOUSE_MOVING_SHOULDER_IK_XML_PATH); print(MOUSE_REFERENCE_DATA_MOVING_SHOULDER_PATH)"`

Expected: prints two absolute paths ending in `.../xmls/mouse_forelimb_right_moving_shoulder_ik.xml` and `.../reference_data_moving_shoulder`. Exit code 0.

- [ ] **Step 3: Commit**

```bash
git add vnl_playground/tasks/mouse/consts.py
git commit -m "s15: add moving-shoulder xml and reference data path constants"
```

---

## Task 2: Create reference data directory (symlinks to v16 IK)

**Files:**
- Create: `vnl_playground/tasks/mouse/reference_data_moving_shoulder/` (directory of symlinks)

- [ ] **Step 1: Create dir and symlink all v16 IK files**

```bash
SRC=/root/vast/eric/stac-mjx/refined_STACed_data_janelia_moving_shoulder_v2
DST=/root/vast/eric/vnl-playground/vnl_playground/tasks/mouse/reference_data_moving_shoulder
mkdir -p "$DST"
for f in "$SRC"/A36-1_2023-07-18_16-54-01_lightOff_trial*_ik.h5; do
  ln -sf "$f" "$DST/$(basename "$f")"
done
ls "$DST" | wc -l
```

Expected: prints `46` (or whatever trial count matches `$SRC`; both `ls $SRC | grep _ik.h5 | wc -l` and the dst count should match).

- [ ] **Step 2: Verify schema matches what MouseReferenceClips expects**

```bash
python - <<'EOF'
import h5py, glob
fs = sorted(glob.glob("/root/vast/eric/vnl-playground/vnl_playground/tasks/mouse/reference_data_moving_shoulder/*_ik.h5"))
with h5py.File(fs[0], "r") as f:
    assert set(["qpos","qvel","xpos","xquat","names_qpos","names_xpos","config"]).issubset(f.keys()), list(f.keys())
    assert f["qpos"].shape == (100, 7), f["qpos"].shape
    assert f["qvel"].shape == (100, 7), f["qvel"].shape
    names = [n.decode() for n in f["names_qpos"][:]]
    assert names == ["sh_tx","sh_ty","sh_tz","sh_rotation","sh_extension","sh_elv","elbow"], names
print("OK", len(fs), "clips; qpos names:", names)
EOF
```

Expected: `OK 46 clips; qpos names: ['sh_tx', 'sh_ty', 'sh_tz', 'sh_rotation', 'sh_extension', 'sh_elv', 'elbow']`

- [ ] **Step 3: Commit**

Symlinks are tracked by git. Add them explicitly (`-A` picks them up).

```bash
git add vnl_playground/tasks/mouse/reference_data_moving_shoulder
git commit -m "s15: symlink v16 IK reference clips into moving-shoulder data dir"
```

---

## Task 3: Create the moving-shoulder muscle XML

This is the surgical part. We need to produce `mouse_forelimb_right_moving_shoulder_ik.xml` which is the existing muscle XML with two changes:

1. Add three slide joints (`sh_tx`, `sh_ty`, `sh_tz`) to the `clavicle` body, **before** any sites or child bodies, with no damping/stiffness and no actuators.
2. Reorder the three humerus hinge joints to `sh_rotation`, `sh_extension`, `sh_elv` (to match the STAC IK qpos column order).

MuJoCo traverses bodies and joints in XML order for qpos layout, so after these edits `qpos` will be `[sh_tx, sh_ty, sh_tz, sh_rotation, sh_extension, sh_elv, elbow]` — matching the IK files bit-for-bit.

**Files:**
- Create: `vnl_playground/tasks/mouse/xmls/mouse_forelimb_right_moving_shoulder_ik.xml`

- [ ] **Step 1: Copy the base muscle XML**

```bash
SRC=/root/vast/eric/vnl-playground/vnl_playground/tasks/mouse/xmls/mouse_forelimb_right.xml
DST=/root/vast/eric/vnl-playground/vnl_playground/tasks/mouse/xmls/mouse_forelimb_right_moving_shoulder_ik.xml
cp "$SRC" "$DST"
```

- [ ] **Step 2: Add the three slide joints inside the clavicle body**

In the new file, locate the `<body name="clavicle" ...>` line (currently line 162 of the source file). Immediately after that opening tag and before `<geom type="mesh" mesh="clavicle_right_mesh"/>`, insert:

```xml
      <!-- IK-driven shoulder translation: no damping, no stiffness, no actuators.
           Overridden each step by MouseImitationMovingShoulder from v16 IK. -->
      <joint name="sh_tx" type="slide" axis="1 0 0" range="-0.01 0.01" damping="0" stiffness="0"/>
      <joint name="sh_ty" type="slide" axis="0 1 0" range="-0.01 0.01" damping="0" stiffness="0"/>
      <joint name="sh_tz" type="slide" axis="0 0 1" range="-0.01 0.01" damping="0" stiffness="0"/>

```

Use Edit with `old_string` being the full `<body name="clavicle" ...>` line plus a newline plus the next indent, and `new_string` being that same prefix plus the three joint lines above.

- [ ] **Step 3: Reorder the three humerus hinge joints**

Find the humerus joints block (currently lines 187–189 of the source):

```xml
          <joint name="sh_elv" type="hinge" pos="0 0 0" axis="0 1 0" range="-70 60" limited="true"/>
          <joint name="sh_extension" type="hinge" pos="0 0 0" axis="0 0 1" range="-70 40" limited="true"/>
          <joint name="sh_rotation" type="hinge" pos="0 0 0" axis="1 0 0" range="-25 60" limited="true"/>
```

Replace with:

```xml
          <joint name="sh_rotation" type="hinge" pos="0 0 0" axis="1 0 0" range="-25 60" limited="true"/>
          <joint name="sh_extension" type="hinge" pos="0 0 0" axis="0 0 1" range="-70 40" limited="true"/>
          <joint name="sh_elv" type="hinge" pos="0 0 0" axis="0 1 0" range="-70 60" limited="true"/>
```

(Same three joints, reordered so XML traversal gives `[sh_rotation, sh_extension, sh_elv]` — matching the IK h5.)

- [ ] **Step 4: Verify compilation and qpos layout**

```bash
python - <<'EOF'
import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path(
    "/root/vast/eric/vnl-playground/vnl_playground/tasks/mouse/xmls/mouse_forelimb_right_moving_shoulder_ik.xml"
)
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
print("njnt:", m.njnt, "nq:", m.nq, "nu:", m.nu)
print("joints:", names)
assert m.nq == 7, m.nq
# Confirm the qpos dof mapping matches expected order:
qpos_names = []
for j, jname in enumerate(names):
    qadr = m.jnt_qposadr[j]
    qpos_names.append((qadr, jname))
qpos_names.sort()
ordered = [n for _, n in qpos_names]
assert ordered == ["sh_tx","sh_ty","sh_tz","sh_rotation","sh_extension","sh_elv","elbow"], ordered
# Confirm there are muscle actuators and no sh_tx/ty/tz actuators:
act_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
print("actuators:", act_names)
for forbidden in ("sh_tx_act","sh_ty_act","sh_tz_act"):
    assert forbidden not in act_names, forbidden
# Confirm shoulder translation joints have zero damping/stiffness:
for jname in ("sh_tx","sh_ty","sh_tz"):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
    dofadr = m.jnt_dofadr[jid]
    assert m.dof_damping[dofadr] == 0.0, (jname, m.dof_damping[dofadr])
    assert m.jnt_stiffness[jid] == 0.0, (jname, m.jnt_stiffness[jid])
print("OK: xml compiles, qpos layout matches IK, no shoulder-translation actuators, zero passive forces on sh_tx/ty/tz.")
EOF
```

Expected: prints counts, joint list, actuator list, then `OK: ...`. All asserts pass.

- [ ] **Step 5: Commit**

```bash
git add vnl_playground/tasks/mouse/xmls/mouse_forelimb_right_moving_shoulder_ik.xml
git commit -m "s15: add muscle xml with IK-driven shoulder translation joints"
```

---

## Task 4: Create `MouseImitationMovingShoulder` env subclass

**Files:**
- Create: `vnl_playground/tasks/mouse/imitation_moving_shoulder.py`

- [ ] **Step 1: Write the subclass**

Create `vnl_playground/tasks/mouse/imitation_moving_shoulder.py` with this exact content:

```python
"""Mouse arm imitation with IK-driven shoulder translation.

The shoulder_tx/ty/tz DOFs are overwritten from the STAC v16 IK reference
after every env step. The muscle policy learns only the 4 hinge-joint
actuations (sh_rotation, sh_extension, sh_elv, elbow). The three IK-driven
dims are masked out of the `joints` and `joints_vel` rewards and the
`pose_error` termination.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from vnl_playground.tasks.mouse.consts import (
    JANELIA_MOUSE_MOVING_SHOULDER_IK_XML_PATH,
    MOUSE_REFERENCE_DATA_MOVING_SHOULDER_PATH,
)
from vnl_playground.tasks.mouse.imitation import (
    MouseImitation,
    default_config as imitation_default_config,
)
from vnl_playground.tasks.mouse.reference_clips import MouseReferenceClips
from vnl_playground.tasks.reward_registry import RewardRegistry


def default_config() -> config_dict.ConfigDict:
    """Moving-shoulder defaults: muscle xml + v16 IK clips + 3 IK-driven dims."""
    cfg = imitation_default_config()
    cfg.walker_xml_path = str(JANELIA_MOUSE_MOVING_SHOULDER_IK_XML_PATH)
    cfg.reference_data_path = str(MOUSE_REFERENCE_DATA_MOVING_SHOULDER_PATH)
    cfg.recompute_kinematics = False  # ref was STAC-fit with same kinematic chain
    cfg.ik_driven_dims = 3  # leading qpos/qvel dims to snap + mask
    return cfg


_registry = RewardRegistry()


class MouseImitationMovingShoulder(MouseImitation):
    """MouseImitation variant that snaps leading qpos dims to IK every step."""

    _registry = _registry

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list, dict]]] = None,
        clips: Optional[MouseReferenceClips] = None,
    ) -> None:
        super().__init__(config, config_overrides, clips)

        n_ik = int(self._config.ik_driven_dims)
        assert n_ik >= 0, n_ik
        if n_ik > 0:
            # Sanity: reference qpos leading dims stay inside the slide-joint range.
            # reference_clips.qpos shape: (n_clips, n_frames, nq)
            lead = self.reference_clips.qpos[:, :, :n_ik]
            max_abs = float(jp.max(jp.abs(lead)))
            assert max_abs < 0.01, (
                f"IK-driven qpos leading dims exceed slide-joint range "
                f"(max |q|={max_abs:.4f} >= 0.01). Widen the XML range or "
                f"rescale the IK before training."
            )

    def _override_ik_dims(
        self, data: mjx.Data, info: Dict[str, Any]
    ) -> mjx.Data:
        """Snap the leading qpos/qvel dims to the IK reference for the current frame."""
        n = int(self._config.ik_driven_dims)
        if n <= 0:
            return data
        cur_frame = self._get_cur_frame(data, info)
        last_valid = self._clip_length() - 1
        cur_frame_clamped = jp.minimum(cur_frame, last_valid)
        ref = self.reference_clips.at(
            clip=info["reference_clip"], frame=cur_frame_clamped
        )
        data = data.replace(
            qpos=data.qpos.at[:n].set(ref.qpos[:n]),
            qvel=data.qvel.at[:n].set(ref.qvel[:n]),
        )
        # Refresh xpos/xquat so downstream consumers (rewards, obs) see the
        # snapped base pose in world coordinates.
        data = mjx.forward(self.mjx_model, data)
        return data

    def reset(
        self,
        rng: jax.Array,
        clip_idx: Optional[int] = None,
        start_frame: Optional[int] = None,
    ) -> mjx_env.State:
        state = super().reset(rng, clip_idx, start_frame)
        # Reference qpos already matches at start, but the snap is idempotent and
        # guards against any rounding drift.
        data = self._override_ik_dims(state.data, state.info)
        obs = self._get_obs(data, state.info)
        return state.replace(data=data, obs=obs)

    def step(
        self,
        state: mjx_env.State,
        action: jax.Array,
    ) -> mjx_env.State:
        n_steps = int(self._config.ctrl_dt / self._config.sim_dt)
        data = mjx_env.step(self.mjx_model, state.data, action, n_steps)

        info = state.info
        # Snap IK-driven dims BEFORE computing obs/reward/termination so that
        # every downstream consumer sees the kinematically correct pose.
        data = self._override_ik_dims(data, info)

        last_valid_frame = self._clip_length() - self._config.reference_length - 1
        truncated = self._get_cur_frame(data, info) > last_valid_frame
        info["truncated"] = jp.astype(truncated, float)
        info["prev_action"] = state.info["action"]
        info["action"] = action

        obs = self._get_obs(data, info)
        terminated = self._is_done(data, info, state.metrics)
        done = jp.logical_or(terminated, info["truncated"])
        reward = self._get_reward(data, info, state.metrics)
        reward = jp.nan_to_num(reward)

        state = state.replace(
            data=data,
            obs=obs,
            info=info,
            reward=reward,
            done=done.astype(float),
        )
        current_frame = self._get_cur_frame(data, info)
        state.metrics["current_frame"] = jp.astype(current_frame, float)
        return state

    # ---- Reward / termination overrides: mask leading IK-driven dims ----

    @_registry.reward("joints")
    def _joints_reward(self, data, info, metrics, weight, exp_scale) -> float:
        target = self._get_current_target(data, info)
        n = int(self._config.ik_driven_dims)
        distance = jp.linalg.norm(target.joints[n:] - data.qpos[n:])
        metrics["joint_l2_error"] = distance
        reward = weight * jp.exp(-((distance / exp_scale) ** 2) / 2)
        metrics["rewards/joints"] = reward
        return reward

    @_registry.reward("joints_vel")
    def _joints_vel_reward(self, data, info, metrics, weight, exp_scale) -> float:
        target = self._get_current_target(data, info)
        n = int(self._config.ik_driven_dims)
        distance = jp.linalg.norm(target.joints_velocity[n:] - data.qvel[n:])
        metrics["joint_vel_l2_error"] = distance
        reward = weight * jp.exp(-((distance / exp_scale) ** 2) / 2)
        metrics["rewards/joints_vel"] = reward
        return reward

    @_registry.termination("pose_error")
    def _bad_pose(self, data, info, max_l2_error) -> bool:
        target = self._get_current_target(data, info)
        n = int(self._config.ik_driven_dims)
        pose_error = jp.linalg.norm(target.joints[n:] - data.qpos[n:])
        return pose_error > max_l2_error
```

- [ ] **Step 2: Verify it imports and the class compiles**

Run:

```bash
python -c "from vnl_playground.tasks.mouse.imitation_moving_shoulder import MouseImitationMovingShoulder, default_config; c=default_config(); print(c.walker_xml_path); print(c.reference_data_path); print('ik_driven_dims=', c.ik_driven_dims)"
```

Expected: prints the two paths from Task 1 and `ik_driven_dims= 3`. Exit code 0.

- [ ] **Step 3: Commit**

```bash
git add vnl_playground/tasks/mouse/imitation_moving_shoulder.py
git commit -m "s15: add MouseImitationMovingShoulder env with IK-driven qpos override"
```

---

## Task 5: Write sanity check script (end-to-end env correctness)

This is the smoke test — it validates the full stack (XML compiles, reference loads, env steps, override is bit-exact, rewards don't crash) before we build the training script.

**Files:**
- Create: `scripts/sanity_check_s15_moving_shoulder.py`

- [ ] **Step 1: Write the script**

```python
"""Sanity check for MouseImitationMovingShoulder.

Verifies:
  1. Env instantiates from defaults.
  2. Reset places qpos[:3] exactly at ref.qpos[:3] of frame 0.
  3. After N zero-action steps, qpos[:3] still matches ref.qpos[:3] at current frame.
  4. Reward terms return finite scalars and `joints` L2 error excludes the IK dims.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jp
import numpy as np

from vnl_playground.tasks.mouse.imitation_moving_shoulder import (
    MouseImitationMovingShoulder,
    default_config,
)


def main() -> None:
    cfg = default_config()
    env = MouseImitationMovingShoulder(cfg)
    print(f"nq={env.mjx_model.nq}, nu={env.mjx_model.nu}, "
          f"ik_driven_dims={cfg.ik_driven_dims}")
    assert env.mjx_model.nq == 7, env.mjx_model.nq

    rng = jax.random.PRNGKey(0)
    state = env.reset(rng, clip_idx=0, start_frame=0)

    ref0 = env.reference_clips.at(clip=0, frame=0)
    n = int(cfg.ik_driven_dims)

    # (2) Reset matches reference exactly on IK dims.
    np.testing.assert_allclose(
        np.asarray(state.data.qpos[:n]), np.asarray(ref0.qpos[:n]), atol=0, rtol=0,
        err_msg="reset did not snap IK dims to reference",
    )
    print("reset qpos[:3] matches ref:", np.asarray(state.data.qpos[:n]))

    # (3) After zero-action steps, IK dims still track reference.
    action = env.null_action()
    for i in range(1, 20):
        state = env.step(state, action)
        cur_frame = int(state.metrics["current_frame"])
        ref = env.reference_clips.at(clip=0, frame=cur_frame)
        np.testing.assert_allclose(
            np.asarray(state.data.qpos[:n]), np.asarray(ref.qpos[:n]),
            atol=0, rtol=0,
            err_msg=f"step {i}: IK dims drifted from reference at frame {cur_frame}",
        )

    # (4) Rewards / metrics are finite.
    assert np.isfinite(float(state.reward)), state.reward
    assert "joint_l2_error" in state.metrics
    # joint_l2_error should only use dims [n:], so if we pass a pose that
    # differs only on IK dims, joints L2 should stay at whatever it was.
    print("reward:", float(state.reward))
    print("joint_l2_error:", float(state.metrics["joint_l2_error"]))
    print("OK")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the sanity check**

```bash
cd /root/vast/eric/vnl-playground
PYTHONPATH=. python scripts/sanity_check_s15_moving_shoulder.py
```

Expected output (roughly):
```
nq=7, nu=5, ik_driven_dims=3
reset qpos[:3] matches ref: [-3.213e-04 -4.025e-04  1.964e-03]
reward: <finite number>
joint_l2_error: <small finite number>
OK
```

Exit code 0. If the numpy_testing assertion fires, the override isn't landing — investigate before proceeding.

- [ ] **Step 3: Commit**

```bash
git add scripts/sanity_check_s15_moving_shoulder.py
git commit -m "s15: add end-to-end sanity check for IK-driven qpos override"
```

---

## Task 6: Create the training script

The training script is a near-verbatim copy of `train_mouse_janelia_sigmoid_normal.py` with the env class, XML path, reference data path, and wandb tags swapped.

**Files:**
- Create: `train_mouse_janelia_sigmoid_moving_shoulder.py`

- [ ] **Step 1: Copy the baseline training script**

```bash
cp /root/vast/eric/vnl-playground/train_mouse_janelia_sigmoid_normal.py \
   /root/vast/eric/vnl-playground/train_mouse_janelia_sigmoid_moving_shoulder.py
```

- [ ] **Step 2: Update the module docstring (top of file)**

Replace the opening docstring block (lines 1–29) with:

```python
"""Janelia mouse forelimb imitation — Sigmoid-Normal with IK-driven moving shoulder (s15).

Identical to train_mouse_janelia_sigmoid_normal.py in policy, distribution, and
PPO hyperparams. Differences:
  - Uses mouse_forelimb_right_moving_shoulder_ik.xml (adds sh_tx/ty/tz slide
    joints to the clavicle; muscles unchanged; no shoulder-translation actuators).
  - Uses reference_data_moving_shoulder/ (STAC v16 IK clips with 7-dim qpos).
  - Env is MouseImitationMovingShoulder: snaps qpos[:3] and qvel[:3] to the IK
    reference after every step, masks those dims out of joints/joints_vel
    rewards and pose_error termination.

Rationale: freezing the shoulder at the origin shifts triceps burst timing
relative to biology. Kinematically driving the shoulder from IK removes that
confound so the muscle policy learns the correct onset.
"""
```

- [ ] **Step 3: Update imports (around lines 65–68 in the baseline)**

Find the import block:

```python
from vnl_playground.tasks.mouse.imitation import MouseImitation, default_config
from vnl_playground.tasks.mouse.reference_clips import MouseReferenceClips
from vnl_playground.tasks.mouse.consts import JANELIA_MOUSE_XML_PATH, MOUSE_REFERENCE_DATA_PATH
```

Replace with:

```python
from vnl_playground.tasks.mouse.imitation_moving_shoulder import (
    MouseImitationMovingShoulder,
    default_config,
)
from vnl_playground.tasks.mouse.reference_clips import MouseReferenceClips
from vnl_playground.tasks.mouse.consts import (
    JANELIA_MOUSE_MOVING_SHOULDER_IK_XML_PATH,
    MOUSE_REFERENCE_DATA_MOVING_SHOULDER_PATH,
)
```

- [ ] **Step 4: Swap env class and paths (around lines 735–743 in the baseline)**

Find:

```python
if args.walker_xml is not None:
    env_cfg.walker_xml_path = epath.Path(args.walker_xml)
else:
    env_cfg.walker_xml_path = JANELIA_MOUSE_XML_PATH

env_cfg.tracked_bodies = ["scapula", "humerus", "ulna", "wrist"]
env_cfg.end_effector = "wrist"
env_cfg.recompute_kinematics = False  # IK data already from same model
```

Replace with:

```python
if args.walker_xml is not None:
    env_cfg.walker_xml_path = epath.Path(args.walker_xml)
else:
    env_cfg.walker_xml_path = JANELIA_MOUSE_MOVING_SHOULDER_IK_XML_PATH

env_cfg.reference_data_path = str(MOUSE_REFERENCE_DATA_MOVING_SHOULDER_PATH)
env_cfg.tracked_bodies = ["scapula", "humerus", "ulna", "wrist"]
env_cfg.end_effector = "wrist"
env_cfg.recompute_kinematics = False  # IK from same kinematic chain as sim model
env_cfg.ik_driven_dims = 3
```

- [ ] **Step 5: Swap every env factory call from `MouseImitation` to `MouseImitationMovingShoulder`**

Run:

```bash
grep -n "MouseImitation" /root/vast/eric/vnl-playground/train_mouse_janelia_sigmoid_moving_shoulder.py
```

For each match that is a **class instantiation** `MouseImitation(` or a **type reference** (not the import), replace `MouseImitation` with `MouseImitationMovingShoulder`. Leave any reference that is already `MouseImitationMovingShoulder` alone.

Use `sed` for the bulk rewrite and then re-grep:

```bash
sed -i 's/\bMouseImitation\b/MouseImitationMovingShoulder/g' \
  /root/vast/eric/vnl-playground/train_mouse_janelia_sigmoid_moving_shoulder.py
grep -n "MouseImitation" /root/vast/eric/vnl-playground/train_mouse_janelia_sigmoid_moving_shoulder.py
```

Expected: every remaining occurrence is `MouseImitationMovingShoulder` (the `\b` boundaries guarantee we don't double-rewrite).

- [ ] **Step 6: Update wandb group / run-name default**

Find the default `--run-name` argparse entry (search for `run-name` or `run_name` in the file). Change the default to reflect s15. For the wandb `group=` arg, use `"s15_moving_shoulder"`.

Example (adapt to the exact argparse/wandb code in the baseline):

```python
parser.add_argument("--run-name", type=str,
                    default="s15_moving_shoulder_sigmoid",
                    help="wandb run name / checkpoint dir suffix")
# ... and later:
wandb.init(project=..., group="s15_moving_shoulder", name=args.run_name, ...)
```

If the baseline uses a computed group string (e.g., from constants at top of file), update that constant instead.

- [ ] **Step 7: Smoke test — short run with a tiny budget**

Run from the repo root:

```bash
cd /root/vast/eric/vnl-playground
PYTHONPATH=. python train_mouse_janelia_sigmoid_moving_shoulder.py \
  --num-timesteps 200000 \
  --num-envs 64 \
  --batch-size 32 \
  --run-name s15_smoke \
  --wandb-mode disabled 2>&1 | tail -40
```

(Adjust CLI flag names to whatever the baseline uses — check `python train_mouse_janelia_sigmoid_moving_shoulder.py --help` first if needed.)

Expected: script runs, prints env info (`nq=7, nu=5` or similar), runs at least one PPO iteration without NaNs, exits cleanly. No tracebacks. Total wall time a few minutes on GPU.

If anything fails, go back through Tasks 3–5 to find which step broke the contract.

- [ ] **Step 8: Commit**

```bash
git add train_mouse_janelia_sigmoid_moving_shoulder.py
git commit -m "s15: add moving-shoulder sigmoid training entry point"
```

---

## Task 7: Final end-to-end validation

- [ ] **Step 1: Confirm git state is clean**

```bash
cd /root/vast/eric/vnl-playground
git status
```

Expected: working tree clean on branch `eric/janelia`, with new commits from Tasks 1–6 on top of the spec commit.

- [ ] **Step 2: Re-run the sanity check (regression guard)**

```bash
PYTHONPATH=. python scripts/sanity_check_s15_moving_shoulder.py
```

Expected: prints `OK`, exit 0.

- [ ] **Step 3: Launch the real training run**

Use the standard launch command format (mirroring how other s-sweeps are launched in this repo — see existing `S*_LAUNCH.md` files for examples of the full invocation the user runs). Hand off back to the user to kick off the real run.

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task(s) |
|---|---|
| New XML with IK-driven slide joints, no actuators on them | Task 3 |
| Reference data dir with v16 IK h5 files | Task 2 |
| New path constants | Task 1 |
| `MouseImitationMovingShoulder` subclass with IK override | Task 4 |
| Reward/termination masking on leading dims | Task 4 (Step 1, `_joints_reward` / `_joints_vel_reward` / `_bad_pose`) |
| Range sanity assertion on IK qpos | Task 4 (Step 1, in `__init__`) |
| New training entry point | Task 6 |
| Sanity script | Task 5 |
| Frame-index clamp on override | Task 4 (Step 1, `_override_ik_dims`) |
| `recompute_kinematics=False` noted | Task 4, Task 6 |

**Placeholder scan:** no `TBD`/`TODO`/"add appropriate error handling"/"similar to Task N" strings. Every code step shows the code. Every command step shows the command and the expected output.

**Type consistency:**
- Class name `MouseImitationMovingShoulder` is identical across Tasks 4, 5, 6.
- Constants `JANELIA_MOUSE_MOVING_SHOULDER_IK_XML_PATH` and `MOUSE_REFERENCE_DATA_MOVING_SHOULDER_PATH` use identical spellings in Tasks 1, 4, 6.
- `ik_driven_dims` config key spelled identically in Tasks 4 and 6.
- Filename `mouse_forelimb_right_moving_shoulder_ik.xml` is identical in Tasks 1, 3, 4.
- Training script filename `train_mouse_janelia_sigmoid_moving_shoulder.py` is identical in Task 6 and the user's original request.
