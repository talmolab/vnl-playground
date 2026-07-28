"""
Imitation training script for the Janelia v22 mouse arm+hand+joystick model.
https://wandb.ai/vnl/vnl-mjx-rl/overview
Uses mouse_forelimb_right_janelia_arm_hand_v22_contacts_patched.xml (35
muscles, 27 joints, full arm+hand+joystick) with PPO via Brax, tracking
STAC v22-native reference clips fit directly against this model by
stac-mjx (/root/vast/eric/stac-mjx/refined_STACed_data_v22) --
scripts/convert_stac_v21_to_v22.py and its v21-transplant output are
superseded, see that script's own docstring.
Shoulder translation (sh_tx/ty/tz) is IK-driven every step; the joystick's
x_slide/y_slide are left to hand-joystick contact physics; everything else
is muscle-actuated. Logs rollout videos + metrics to wandb.

Unlike the other train_mouse_janelia*.py scripts, the walker XML and
reference-data paths are real CLI flags here (--xml-path/--data-path), not
hardcoded python constants, so this script can be repointed at a future
model/dataset without editing source.
"""

import argparse
import os

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import functools
import json
from datetime import datetime
from pprint import pprint

import imageio
import jax
import jax.numpy as jp
import mujoco
import numpy as np
import wandb

from vnl_playground import jax_compat
jax_compat.install()

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.acme import running_statistics
from etils import epath
from flax.training import orbax_utils
from ml_collections import config_dict
from orbax import checkpoint as ocp

from mujoco_playground import wrapper
from vnl_playground.tasks.mouse.imitation_arm_hand import (
    MouseImitationArmHand,
    default_config,
    default_config_no_joystick,
    default_config_raw_joystick,
    default_config_v23,
    default_config_v25,
)
from vnl_playground.tasks.wrappers import FlattenObsWrapper

# Enable persistent compilation cache.
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


# ── CLI arguments ────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Janelia v22 mouse arm+hand+joystick imitation training"
    )

    # Model / data overrides (this script's real CLI surface, unlike the
    # other train_mouse_janelia*.py scripts which hardcode these as python
    # constants).
    p.add_argument(
        "--xml-path", type=str, default=None,
        help="Walker MJCF path (default: JANELIA_MOUSE_ARM_HAND_V22_XML_PATH)"
    )
    p.add_argument(
        "--data-path", type=str, default=None,
        help="Reference clip directory (default: MOUSE_REFERENCE_DATA_JANELIA_V22_PATH)"
    )
    p.add_argument(
        "--no-joystick", action="store_true",
        help="Use the v22x config (default_config_no_joystick): arm+hand "
             "reach-tracking only, joystick removed entirely. --xml-path/"
             "--data-path still override on top of this if also passed."
    )
    p.add_argument(
        "--v23", action="store_true",
        help="Use the v23 config (default_config_v23): v22 arm+hand+joystick "
             "with simplified primitive collision geoms (ellipsoid/capsule) "
             "on hand/wrist bones instead of raw meshes, to fix broadphase "
             "contact-buffer overflow. Mutually exclusive with --no-joystick "
             "and --raw-joystick."
    )
    p.add_argument(
        "--raw-joystick", action="store_true",
        help="Use the raw-joystick config (default_config_raw_joystick): "
             "same mesh collision geoms as the default v22 config (already "
             "verified to train cleanly to 500M steps), but with the "
             "joystick reverted to its raw stac-mjx-synced (tilted) "
             "position instead of the manually de-tilted patch, plus a "
             "fixed diagnostic camera for legible renders. Mutually "
             "exclusive with --no-joystick and --v23."
    )
    p.add_argument(
        "--v25", action="store_true",
        help="Use the v25 config (default_config_v25): v24 arm+hand+neck+"
             "head rig (already-simplified ellipsoid/capsule collision) "
             "plus a new capsule-shaft joystick. Joystick present "
             "physically but untargeted for now. Mutually exclusive with "
             "the other model flags."
    )

    p.add_argument(
        "--finetune-path", type=str, default=None,
        help="Path to a checkpoint dir (e.g. checkpoints/<exp>/<step>) to "
             "restore policy/value/normalizer params from via brax's "
             "restore_checkpoint_path. Does NOT restore env-step counter or "
             "optimizer momentum -- the new run's step count and eval "
             "cadence start fresh."
    )

    # Sweep metadata
    p.add_argument("--tag", type=str, default=None, help="Sweep run tag (e.g. 'baseline', 'low-damp')")
    p.add_argument("--run-name", type=str, default=None, help="Full run name (e.g. 'S1-00-baseline')")
    p.add_argument("--wandb-group", type=str, default=None, help="Wandb group (e.g. 'sweep1-physics')")
    p.add_argument("--wandb-tags", type=str, nargs="*", default=None, help="Wandb tags list")
    p.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")

    # Physics overrides
    p.add_argument("--joint-damping", type=float, default=None, help="Override dof_damping[:] (default: XML value)")
    p.add_argument("--joint-armature", type=float, default=None, help="Override dof_armature[:] (default: XML value)")
    p.add_argument("--joint-stiffness", type=float, default=None, help="Override jnt_stiffness[:] (default: XML value)")
    p.add_argument("--force-scale", type=float, default=None, help="Multiply actuator_gainprm[:, 0] (default: 1.0)")
    p.add_argument("--iterations", type=int, default=None, help="Override cfg.iterations (solver iterations; default: whatever the env config sets)")
    p.add_argument("--ls-iterations", type=int, default=None, help="Override cfg.ls_iterations (line-search iterations; default: whatever the env config sets)")

    # Training / env overrides
    p.add_argument("--reference-length", type=int, default=None, help="Frames of future reference in observation")
    p.add_argument("--episode-length", type=int, default=None, help="PPO episode length (env steps)")
    p.add_argument("--entropy-cost", type=float, default=None, help="PPO entropy cost coefficient")
    p.add_argument("--learning-rate", type=float, default=None, help="PPO learning rate")
    p.add_argument("--discounting", type=float, default=None, help="PPO discount factor")
    p.add_argument("--batch-size", type=int, default=None, help="PPO batch size")
    p.add_argument("--num-minibatches", type=int, default=None, help="PPO num minibatches")
    p.add_argument("--num-timesteps", type=int, default=None, help="Total training timesteps")
    p.add_argument("--eval-every", type=int, default=None, help="Env steps between eval/wandb log points (default: 100_000_000 -- too sparse to see progress on a multi-hour run, override for visibility)")

    # Reward overrides
    p.add_argument("--control-cost", type=float, default=None, help="Control cost weight")
    p.add_argument("--control-diff-cost", type=float, default=None, help="Control diff cost weight")

    return p.parse_args()


args = parse_args()


# ── Environment config ──────────────────────────────────────────────────────
assert sum([args.no_joystick, args.v23, args.raw_joystick, args.v25]) <= 1, (
    "--no-joystick, --v23, --raw-joystick, and --v25 are mutually exclusive"
)
if args.no_joystick:
    env_cfg = default_config_no_joystick()
elif args.v23:
    env_cfg = default_config_v23()
elif args.raw_joystick:
    env_cfg = default_config_raw_joystick()
elif args.v25:
    env_cfg = default_config_v25()
else:
    env_cfg = default_config()
if args.xml_path is not None:
    env_cfg.walker_xml_path = epath.Path(args.xml_path)
if args.data_path is not None:
    env_cfg.reference_data_path = args.data_path

# Apply physics overrides from CLI
if args.joint_damping is not None:
    env_cfg.joint_damping = args.joint_damping
if args.joint_armature is not None:
    env_cfg.joint_armature = args.joint_armature
if args.joint_stiffness is not None:
    env_cfg.joint_stiffness = args.joint_stiffness
if args.force_scale is not None:
    env_cfg.force_scale = args.force_scale
if args.iterations is not None:
    env_cfg.iterations = args.iterations
if args.ls_iterations is not None:
    env_cfg.ls_iterations = args.ls_iterations

# Apply env/reward overrides from CLI
if args.reference_length is not None:
    env_cfg.reference_length = args.reference_length
if args.control_cost is not None:
    env_cfg.reward_terms["control_cost"]["weight"] = args.control_cost
if args.control_diff_cost is not None:
    env_cfg.reward_terms["control_diff_cost"]["weight"] = args.control_diff_cost


# ── PPO hyper-parameters ────────────────────────────────────────────────────
ppo_params = config_dict.create(
    num_envs=4096,
    num_timesteps=int(500_000_000),
    batch_size=1024,
    num_minibatches=16,
    num_updates_per_batch=3,
    learning_rate=1e-4,
    clipping_epsilon=0.1,
    discounting=0.95,
    action_repeat=1,
    entropy_cost=1e-4,
    reward_scaling=1.0,
    normalize_observations=True,
    unroll_length=20,
    # 126 reference frames at mocap_hz=25 / ctrl_dt=0.02 (2026-07-17: 2
    # control steps per mocap frame, switched from v1/v3's 0.0025/16-steps-
    # per-frame pair to fix the staircase-target oscillation seen in
    # checkpoints/janelia-v22-arm-hand-20260717-060110-track-a-smoke; see
    # imitation_arm_hand.py's default_config() for the full reasoning and
    # the sim_dt=0.001 pairing) -> 126/(0.02*25) = 252 control steps to
    # span the full STAC clip (126 frames verified via the original
    # registered-mocap trial's own reprojection video metadata: fps=25.0,
    # 126 frames, duration=5.04s -- and independently re-verified
    # 2026-07-17 directly against every trial's own h5 file, not just this
    # one). Must move together with ctrl_dt*mocap_hz or episodes stop
    # partway through each clip instead of spanning it fully.
    # imitation_arm_hand's default_config() also sets clip_length=None so
    # MouseReferenceClips doesn't truncate to the generic 50-frame default.
    episode_length=252,
    max_grad_norm=1.0,
    network_factory=config_dict.create(
        policy_hidden_layer_sizes=(512, 512, 512),
        value_hidden_layer_sizes=(512, 512, 512),
    ),
    eval_every=100_000_000,
)

# Apply PPO overrides from CLI
if args.entropy_cost is not None:
    ppo_params.entropy_cost = args.entropy_cost
if args.learning_rate is not None:
    ppo_params.learning_rate = args.learning_rate
if args.discounting is not None:
    ppo_params.discounting = args.discounting
if args.batch_size is not None:
    ppo_params.batch_size = args.batch_size
if args.num_minibatches is not None:
    ppo_params.num_minibatches = args.num_minibatches
if args.episode_length is not None:
    ppo_params.episode_length = args.episode_length
if args.num_timesteps is not None:
    ppo_params.num_timesteps = args.num_timesteps
if args.eval_every is not None:
    ppo_params.eval_every = args.eval_every

pprint(ppo_params)


# ── Experiment naming / checkpoint restore ──────────────────────────────────
if args.no_joystick:
    env_name = "janelia-v22x-arm-only"
elif args.v23:
    env_name = "janelia-v23-simplified-collision"
elif args.raw_joystick:
    env_name = "janelia-v22-raw-joystick-camera"
elif args.v25:
    env_name = "janelia-v25-arm-hand-joystick"
else:
    env_name = "janelia-v22-arm-hand"
SUFFIX = None
FINETUNE_PATH = args.finetune_path

now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S")

if args.run_name is not None:
    exp_name = args.run_name
else:
    exp_name = f"{env_name}-{timestamp}"
    if args.tag is not None:
        exp_name += f"-{args.tag}"
    elif SUFFIX is not None:
        exp_name += f"-{SUFFIX}"

# Build a param summary string for the wandb name
_param_parts = []
_param_map = [
    ("damp", args.joint_damping), ("arm", args.joint_armature),
    ("stiff", args.joint_stiffness), ("fscale", args.force_scale),
    ("ref", args.reference_length), ("ep", args.episode_length),
    ("ent", args.entropy_cost), ("lr", args.learning_rate),
    ("disc", args.discounting), ("bs", args.batch_size),
    ("mb", args.num_minibatches), ("ctrl", args.control_cost),
    ("cdiff", args.control_diff_cost),
]
for short, val in _param_map:
    if val is not None:
        _param_parts.append(f"{short}={val:g}" if isinstance(val, float) else f"{short}={val}")
param_suffix = "_".join(_param_parts)
wandb_name = f"{exp_name}_{param_suffix}" if param_suffix else exp_name

print(f"Experiment name: {exp_name}")
print(f"Wandb name: {wandb_name}")
print(f"Walker xml: {env_cfg.walker_xml_path}")
print(f"Reference data: {env_cfg.reference_data_path}")

if FINETUNE_PATH is not None:
    FINETUNE_PATH = epath.Path(FINETUNE_PATH)
    latest_ckpts = [c for c in FINETUNE_PATH.glob("*") if c.is_dir()]
    latest_ckpts.sort(key=lambda x: int(x.name))
    restore_checkpoint_path = latest_ckpts[-1]
    print(f"Restoring from: {restore_checkpoint_path}")
else:
    restore_checkpoint_path = None


# ── Checkpoint directory ────────────────────────────────────────────────────
ckpt_path = epath.Path("checkpoints").resolve() / exp_name
ckpt_path.mkdir(parents=True, exist_ok=True)
print(f"Checkpoint dir: {ckpt_path}")

with open(ckpt_path / "config.json", "w") as fp:
    json.dump(env_cfg.to_dict(), fp, indent=4, default=str)


# ── Wandb ───────────────────────────────────────────────────────────────────
USE_WANDB = not args.no_wandb

if USE_WANDB:
    wandb_kwargs = dict(
        project="new-janelia",
        config=env_cfg,
        name=wandb_name,
        id=f"janelia-v22-arm-hand-{exp_name}",
    )
    if args.wandb_group is not None:
        wandb_kwargs["group"] = args.wandb_group
    if args.wandb_tags is not None:
        wandb_kwargs["tags"] = args.wandb_tags

    wandb.init(**wandb_kwargs)
    wandb.config.update({"env_name": env_name})
    # Log all sweep params for easy filtering
    sweep_config = {
        "sweep/tag": args.tag,
        "sweep/joint_damping": args.joint_damping,
        "sweep/joint_armature": args.joint_armature,
        "sweep/joint_stiffness": args.joint_stiffness,
        "sweep/force_scale": args.force_scale,
        "sweep/reference_length": args.reference_length,
        "sweep/episode_length": args.episode_length,
        "sweep/entropy_cost": args.entropy_cost,
        "sweep/learning_rate": args.learning_rate,
        "sweep/discounting": args.discounting,
        "sweep/batch_size": args.batch_size,
        "sweep/control_cost": args.control_cost,
        "sweep/control_diff_cost": args.control_diff_cost,
    }
    wandb.config.update({k: v for k, v in sweep_config.items() if v is not None})


def wandb_progress(num_steps, metrics):
    pprint(f"Step {num_steps}")
    pprint(metrics)
    # 2026-07-21, per Eric: wandb.log()'s own step counter previously only
    # advanced once per eval call, not per real env timestep, so the
    # dashboard x-axis showed "eval number" -- no way to tell how far into
    # num_timesteps a run actually was without cross-referencing the console
    # log (this cost a live run getting stopped one eval short of where it
    # would have been let run, since 94.4M looked less far along than it
    # was against a 500M target). Log real progress explicitly instead.
    metrics["timesteps"] = num_steps
    metrics["total_timesteps"] = ppo_params.num_timesteps
    wandb.log(metrics)


def progress(num_steps, metrics):
    pprint(f"Step {num_steps}")
    pprint(metrics)


progress_fn = wandb_progress if USE_WANDB else progress


# ── Network / inference setup ───────────────────────────────────────────────
training_params = dict(ppo_params)
del training_params["network_factory"]
del training_params["eval_every"]

network_factory = functools.partial(
    ppo_networks.make_ppo_networks, **ppo_params.network_factory
)
normalize = lambda x, y: x
if training_params["normalize_observations"]:
    normalize = running_statistics.normalize

train_fn = functools.partial(
    ppo.train,
    **training_params,
    num_evals=int(ppo_params.num_timesteps / ppo_params.eval_every),
    network_factory=network_factory,
    restore_checkpoint_path=restore_checkpoint_path,
    progress_fn=progress_fn,
    wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training, full_reset=True),
)


def make_logging_inference_fn(ppo_networks):
    """Build a deterministic inference function for rollout rendering."""

    def make_logging_policy(deterministic=False):
        policy_network = ppo_networks.policy_network
        parametric_action_distribution = ppo_networks.parametric_action_distribution

        def logging_policy(params, observations, key_sample):
            param_subset = (params[0], params[1])
            logits = policy_network.apply(*param_subset, observations)
            if deterministic:
                return (
                    jp.array(parametric_action_distribution.mode(logits)),
                    {},
                )
            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )
            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
            postprocessed_actions = parametric_action_distribution.postprocess(
                raw_actions
            )
            return jp.array(postprocessed_actions), {
                "log_prob": log_prob,
                "raw_action": raw_actions,
            }

        return logging_policy

    return make_logging_policy


# ── Build rendering model with proper camera ────────────────────────────────
def _attach_walker(render_spec, walker_xml_path, root_bodies, suffix):
    """Attach every root body of a (possibly multi-root) walker XML.

    Mirrors MouseBaseEnv.add_mouse's single-vs-multi-root branching: a
    single named root uses attach_body (cheap, exact); more than one root
    (e.g. v22's disconnected "shoulder_base" arm + "joystick_base"
    manipulandum) uses MjSpec's whole-model attach() instead, since
    attach_body() re-imports the *entire* source asset table on every call
    and two such calls from the same file collide on mesh names.
    """
    frame = render_spec.worldbody.add_frame(pos=[0, 0, 0], quat=[1, 0, 0, 0])
    if len(root_bodies) == 1:
        walker_spec = mujoco.MjSpec.from_file(walker_xml_path)
        return frame, frame.attach_body(walker_spec.body(root_bodies[0]), "", suffix)
    walker_spec = mujoco.MjSpec.from_file(walker_xml_path)
    render_spec.attach(walker_spec, prefix="", suffix=suffix, frame=frame)
    return frame, None  # multi-root: no single body handle to return


def build_render_model(env):
    """Build a MuJoCo model for rendering with a camera aimed at the arm+hand."""
    root_bodies = env._config.root_bodies
    render_spec = mujoco.MjSpec.from_file(str(env.arena_xml_path))
    _attach_walker(render_spec, str(env.walker_xml_path), root_bodies, "-mouse")

    # Add ghost walker for reference motion visualisation
    _, ghost_body = _attach_walker(
        render_spec, str(env.walker_xml_path), root_bodies, "-ghost"
    )

    def recolor_geoms(body, rgba):
        for g in body.geoms:
            g.rgba = rgba
            g.contype = 0
            g.conaffinity = 0
        for child in body.bodies:
            recolor_geoms(child, rgba)

    if ghost_body is not None:
        recolor_geoms(ghost_body, [0.3, 0.8, 1.0, 0.4])
    else:
        for body in render_spec.worldbody.bodies:
            if body.name.endswith("-ghost"):
                recolor_geoms(body, [0.3, 0.8, 1.0, 0.4])

    # Camera that looks at the arm+hand from a profile angle. Model extent
    # is larger than the arm-only models (full hand chain); revisit this
    # framing once real rollouts are available (Task 6).
    render_spec.worldbody.add_camera(
        name="janelia_cam",
        pos=[0.10, 0.01, 0.07],
        xyaxes=[0, 1, 0, -0.3, 0, 1],
        fovy=45,
    )

    rm = render_spec.compile()
    rm.opt.timestep = env._config.sim_dt
    return rm


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    env = FlattenObsWrapper(MouseImitationArmHand(config=env_cfg))
    eval_env = FlattenObsWrapper(MouseImitationArmHand(config=env_cfg))

    # ── Rendering setup ─────────────────────────────────────────────────────
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    rng = jax.random.PRNGKey(0)
    start_state = jit_reset(rng)

    render_mj_model = build_render_model(eval_env)
    render_mj_data = mujoco.MjData(render_mj_model)
    renderer = mujoco.Renderer(render_mj_model, height=480, width=854)

    # Hide group-1 collision-proxy geoms (v24-derived models, incl. v25, have
    # giant torso/skull ellipsoids -- T13_col, Skull_col, etc, several cm
    # across -- that occlude the arm/hand/joystick entirely if shown; see
    # train_mouse_janelia_v24.py's build_render_model for the same fix).
    render_scene_option = mujoco.MjvOption()
    render_scene_option.geomgroup[1] = 0

    fixed_cam_id = mujoco.mj_name2id(
        render_mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "arm_joystick_fixed_view"
    )
    if fixed_cam_id >= 0:
        # Walker XML carries a joystick-tilt-matched camera (v23/raw-joystick
        # variants) -- use it by name so eval videos actually roll to match
        # the shaft instead of the generic auto-framed free camera below.
        render_cam = "arm_joystick_fixed_view"
    else:
        # No named camera (v25): centroid/span over arm+hand+joystick bodies
        # only, not the whole skeleton -- an all-body centroid (old fallback,
        # pre-2026-07-19) is dominated by the giant fixed torso/skull chain
        # and points the camera at the spine instead of the arm/joystick,
        # same root cause v24 already hit and fixed (see
        # docs/2026-07-17-v24-v25-buildout-and-full-run.md).
        render_mj_data.qpos[:render_mj_model.nq // 2] = np.array(start_state.data.qpos)
        mujoco.mj_forward(render_mj_model, render_mj_data)
        arm_hand_joystick_names = [
            "humerus_right-mouse", "ulna_right-mouse", "radius_right-mouse",
            "N_L_C_right-mouse", "joystick_base-mouse", "joystick-mouse",
        ] + [
            mujoco.mj_id2name(render_mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(render_mj_model.nbody)
            if (mujoco.mj_id2name(render_mj_model, mujoco.mjtObj.mjOBJ_BODY, i) or "")
            .startswith(("Metacarpal_hand", "Phalanx_hand"))
            and (mujoco.mj_id2name(render_mj_model, mujoco.mjtObj.mjOBJ_BODY, i) or "")
            .endswith("-mouse")
        ]
        arm_hand_joystick_ids = [
            mujoco.mj_name2id(render_mj_model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in arm_hand_joystick_names
        ]
        arm_hand_joystick_ids = [i for i in arm_hand_joystick_ids if i >= 0]
        arm_hand_xpos = render_mj_data.xpos[arm_hand_joystick_ids]
        centroid = arm_hand_xpos.mean(axis=0)
        span = np.linalg.norm(arm_hand_xpos.max(axis=0) - arm_hand_xpos.min(axis=0))

        render_cam = mujoco.MjvCamera()
        render_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        render_cam.lookat[:] = centroid
        render_cam.distance = span * 2.2
        # azimuth=90, not v24's 0 -- swept and checked visually, this is the angle that shows real vs ghost joystick side by side
        render_cam.azimuth = 90
        render_cam.elevation = -45

    ppo_network = network_factory(
        start_state.obs.shape[-1],
        eval_env.action_size,
        preprocess_observations_fn=normalize,
    )
    make_logging_policy = make_logging_inference_fn(ppo_network)
    jit_logging_inference_fn = jax.jit(make_logging_policy(deterministic=True))

    def policy_params_fn(current_step, make_policy, params, jit_logging_inference_fn):
        del make_policy  # Unused.

        # ── Generate rollout ────────────────────────────────────────────
        rollout = [start_state]
        state = start_state
        rng = jax.random.PRNGKey(0)
        for _ in range(ppo_params.episode_length):
            _, rng = jax.random.split(rng)
            action, _ = jit_logging_inference_fn(params, state.obs, rng)
            state = jit_step(state, action)
            rollout.append(state)

        # ── Get reference qpos for ghost ────────────────────────────────
        ref_clips = eval_env.env.reference_clips  # unwrap FlattenObsWrapper

        # ── Render video ────────────────────────────────────────────────
        # rollout has one state per control step (ctrl_dt seconds each) --
        # set the video's fps to exactly match the control rate and render
        # every state, so playback duration equals real simulated time with
        # no rounding drift (e.g. a subsample-to-30fps stride would round
        # 1/30/ctrl_dt to the nearest integer and drift off real time).
        video_fps = round(1.0 / env_cfg.ctrl_dt)
        video_path = f"{ckpt_path}/{current_step}.mp4"
        with imageio.get_writer(video_path, fps=video_fps) as video:
            for s in rollout:
                frame_idx = eval_env.env._get_cur_frame(s.data, s.info)
                clip_idx = s.info["reference_clip"]
                ref = ref_clips.at(clip=clip_idx, frame=frame_idx)

                # Main arm+hand qpos + ghost qpos (reference)
                render_mj_data.qpos[:] = np.concatenate(
                    [np.array(s.data.qpos), np.array(ref.qpos)]
                )
                render_mj_data.qvel[:] = np.concatenate(
                    [np.array(s.data.qvel), np.array(ref.qvel)]
                )
                mujoco.mj_forward(render_mj_model, render_mj_data)
                renderer.update_scene(render_mj_data, camera=render_cam, scene_option=render_scene_option)
                video.append_data(renderer.render())

        if USE_WANDB:
            wandb.log(
                {"eval/rollout": wandb.Video(video_path, format="mp4")},
                commit=False,
            )
        print(f"  -> saved rollout video: {video_path}")

        # ── Save checkpoint ─────────────────────────────────────────────
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        path = ckpt_path / f"{current_step}"
        orbax_checkpointer.save(path, params, force=True, save_args=save_args)

    # ── Train ───────────────────────────────────────────────────────────────
    print(
        f"Starting training: {ppo_params.num_timesteps:,} timesteps, "
        f"{ppo_params.num_envs} envs"
    )
    print(f"Action size: {eval_env.action_size}, Obs size: {start_state.obs.shape[-1]}")

    make_inference_fn, params, _ = train_fn(
        environment=env,
        eval_env=eval_env,
        policy_params_fn=functools.partial(
            policy_params_fn, jit_logging_inference_fn=jit_logging_inference_fn
        ),
    )
