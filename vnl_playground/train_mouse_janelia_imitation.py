"""
Imitation training script for the Janelia mouse forelimb model.
https://wandb.ai/vnl/vnl-mjx-rl/overview
Uses the janelia_mouse_forelimb.xml model (12 muscles, 4 DOF arm)
with PPO via Brax, tracking reference motion clips.
Logs rollout videos + metrics to wandb.
"""

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
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.acme import running_statistics
from etils import epath
from flax.training import orbax_utils
from ml_collections import config_dict
from orbax import checkpoint as ocp

from mujoco_playground import wrapper
from vnl_playground.tasks.mouse.imitation import MouseImitation, default_config
from vnl_playground.tasks.mouse.consts import JANELIA_MOUSE_XML_PATH
from vnl_playground.tasks.wrappers import FlattenObsWrapper

# Enable persistent compilation cache.
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


# ── Environment config ──────────────────────────────────────────────────────
env_cfg = default_config()
env_cfg.walker_xml_path = JANELIA_MOUSE_XML_PATH

# Janelia model bodies: clavicle, scapula, humerus, ulna, wrist
# (no separate 'radius' or 'elbow' bodies)
env_cfg.tracked_bodies = ["scapula", "humerus", "ulna", "wrist"]
env_cfg.end_effector = "wrist"
env_cfg.recompute_kinematics = False  # IK data already from same model


# ── PPO hyper-parameters ────────────────────────────────────────────────────
ppo_params = config_dict.create(
    num_envs=4096,
    num_timesteps=int(1_000_000_000),
    batch_size=1024,
    num_minibatches=16,
    num_updates_per_batch=3,
    learning_rate=1e-3,
    clipping_epsilon=0.1,
    discounting=0.95,
    action_repeat=1,
    entropy_cost=1e-2,
    reward_scaling=1.0,
    normalize_observations=True,
    unroll_length=20,
    episode_length=50,
    max_grad_norm=1.0,
    network_factory=config_dict.create(
        policy_hidden_layer_sizes=(512, 512, 512),
        value_hidden_layer_sizes=(512, 512, 512),
    ),
    eval_every=100_000_000,
)

pprint(ppo_params)


# ── Experiment naming / checkpoint restore ──────────────────────────────────
env_name = "janelia-mouse-imitation"
SUFFIX = None
FINETUNE_PATH = None

now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S")
exp_name = f"{env_name}-{timestamp}"
if SUFFIX is not None:
    exp_name += f"-{SUFFIX}"
print(f"Experiment name: {exp_name}")

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
USE_WANDB = True

if USE_WANDB:
    wandb.init(
        project="vnl-mjx-rl",
        config=env_cfg,
        id=f"janelia-imit-{exp_name}",
    )
    wandb.config.update({"env_name": env_name})


def wandb_progress(num_steps, metrics):
    pprint(f"Step {num_steps}")
    pprint(metrics)
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
def build_render_model(env):
    """Build a MuJoCo model for rendering with a camera aimed at the Janelia arm."""
    render_spec = mujoco.MjSpec.from_file(str(env.arena_xml_path))
    walker_spec = mujoco.MjSpec.from_file(str(env.walker_xml_path))
    spawn_frame = render_spec.worldbody.add_frame(pos=[0, 0, 0], quat=[1, 0, 0, 0])
    spawn_frame.attach_body(walker_spec.body("clavicle"), "", "-mouse")

    # Add ghost walker for reference motion visualisation
    ghost_spec = mujoco.MjSpec.from_file(str(env.walker_xml_path))
    ghost_frame = render_spec.worldbody.add_frame(pos=[0, 0, 0], quat=[1, 0, 0, 0])
    ghost_body = ghost_frame.attach_body(ghost_spec.body("clavicle"), "", "-ghost")

    def recolor_geoms(body, rgba):
        for g in body.geoms:
            g.rgba = rgba
            g.contype = 0
            g.conaffinity = 0
        for child in body.bodies:
            recolor_geoms(child, rgba)

    recolor_geoms(ghost_body, [0.3, 0.8, 1.0, 0.4])

    # Camera that looks at the arm from a profile angle
    # Model centroid is approx (0.015, 0.010, 0.067)
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
    env = FlattenObsWrapper(MouseImitation(config=env_cfg))
    eval_env = FlattenObsWrapper(MouseImitation(config=env_cfg))

    # ── Rendering setup ─────────────────────────────────────────────────────
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    rng = jax.random.PRNGKey(0)
    start_state = jit_reset(rng)

    render_mj_model = build_render_model(eval_env)
    render_mj_data = mujoco.MjData(render_mj_model)
    renderer = mujoco.Renderer(render_mj_model, height=512, width=512)

    # Compute centroid for camera lookat from initial pose
    render_mj_data.qpos[:render_mj_model.nq // 2] = np.array(start_state.data.qpos)
    mujoco.mj_forward(render_mj_model, render_mj_data)
    centroid = render_mj_data.xpos[1:render_mj_model.nbody // 2 + 1].mean(axis=0)

    render_cam = mujoco.MjvCamera()
    render_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    render_cam.lookat[:] = centroid
    render_cam.distance = 0.055
    render_cam.azimuth = 165
    render_cam.elevation = -25

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
        video_path = f"{ckpt_path}/{current_step}.mp4"
        with imageio.get_writer(video_path, fps=30) as video:
            for s in rollout:
                frame_idx = eval_env.env._get_cur_frame(s.data, s.info)
                clip_idx = s.info["reference_clip"]
                ref = ref_clips.at(clip=clip_idx, frame=frame_idx)

                # Main arm qpos + ghost arm qpos (reference)
                render_mj_data.qpos[:] = np.concatenate(
                    [np.array(s.data.qpos), np.array(ref.qpos)]
                )
                render_mj_data.qvel[:] = np.concatenate(
                    [np.array(s.data.qvel), np.array(ref.qvel)]
                )
                mujoco.mj_forward(render_mj_model, render_mj_data)
                renderer.update_scene(render_mj_data, camera=render_cam)
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
