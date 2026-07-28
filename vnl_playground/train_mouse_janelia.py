"""
Training script for the Janelia mouse forelimb reaching task.

Uses the janelia_mouse_forelimb.xml model (12 muscles, 4 DOF arm)
with PPO via Brax. Logs rollout videos + metrics to wandb.
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

import imageio
import jax
import mujoco
import numpy as np
import wandb
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from etils import epath
from flax.training import orbax_utils
from ml_collections import config_dict
from orbax import checkpoint as ocp

from mujoco_playground import wrapper
from vnl_playground.tasks.mouse.mouse_reach import MouseReach, default_config
from vnl_playground.tasks.mouse.consts import JANELIA_MOUSE_XML_PATH
from vnl_playground.tasks.wrappers import BraxObsWrapper

# Enable persistent compilation cache.
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


# ── Environment config ──────────────────────────────────────────────────────
def janelia_env_config():
    """MouseReach config with the Janelia forelimb model and workspace."""
    cfg = default_config()
    cfg.unlock()
    cfg.walker_xml_path = JANELIA_MOUSE_XML_PATH
    # Target volume matched to the Janelia model workspace.
    # Wrist reachable space: x [0.006, 0.032], y [-0.007, 0.027], z [0.054, 0.080]
    cfg.target_volume_min = (0.008, -0.004, 0.056)
    cfg.target_volume_max = (0.030, 0.025, 0.078)
    cfg.lock()
    return cfg


env_cfg = janelia_env_config()


# ── PPO hyper-parameters ────────────────────────────────────────────────────
ppo_params = config_dict.create(
    num_timesteps=1_000_000_000,
    num_evals=30,
    reward_scaling=1.0,
    episode_length=100,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=20,
    num_minibatches=32,
    num_updates_per_batch=4,
    discounting=0.97,
    learning_rate=1e-4,
    entropy_cost=1e-4,
    num_envs=4096,
    batch_size=256,
    max_grad_norm=1.0,
    network_factory=config_dict.create(
        policy_hidden_layer_sizes=(512, 512, 512),
        value_hidden_layer_sizes=(512, 512, 512),
        policy_obs_key="state",
        value_obs_key="state",
    ),
)


# ── Experiment naming / checkpoint restore ──────────────────────────────────
env_name = "janelia-mouse-reach"
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
        config=env_cfg.to_dict(),
        id=f"janelia-reach-{exp_name}",
    )
    wandb.config.update({"env_name": env_name})

times = [datetime.now()]


def progress(num_steps, metrics):
    times.append(datetime.now())
    elapsed = times[-1] - times[0]
    print(
        f"[{elapsed}] step {num_steps:>12,}  "
        f"reward={metrics.get('eval/episode_reward', 0):.4f}  "
        f"reward_ctrl={metrics.get('eval/episode_reward_ctrl', 0):.4f}"
    )
    if USE_WANDB:
        wandb.log(metrics, step=num_steps)


# ── Create envs ─────────────────────────────────────────────────────────────
env = BraxObsWrapper(MouseReach(config=janelia_env_config()))
eval_env = BraxObsWrapper(MouseReach(config=janelia_env_config()))

# ── Video rendering setup ───────────────────────────────────────────────────
jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)
rollout_rng = jax.random.PRNGKey(0)

# Build a rendering model with a camera aimed at the Janelia arm.
# Model centroid is approx (0.015, 0.010, 0.067).
render_mj_model = eval_env.mj_model
render_mj_data = mujoco.MjData(render_mj_model)

# Add a fixed camera via spec recompilation for clean rendering
render_spec = mujoco.MjSpec.from_file(str(eval_env.arena_xml_path))
walker_spec = mujoco.MjSpec.from_file(str(eval_env.walker_xml_path))
spawn_frame = render_spec.worldbody.add_frame(pos=[0, 0, 0], quat=[1, 0, 0, 0])
spawn_frame.attach_body(walker_spec.body("clavicle"), "", "-mouse")

# Add a camera that looks at the arm from a profile angle
render_spec.worldbody.add_camera(
    name="janelia_cam",
    pos=[0.10, 0.01, 0.07],
    xyaxes=[0, 1, 0, -0.3, 0, 1],
    fovy=45,
)

# Add the same mocap target body that MouseReach creates, so the green
# sphere renders in rollout videos.
target_body = render_spec.worldbody.add_body(
    name="target", mocap=True, pos=[0.002, 0.010, -0.006],
)
target_body.add_geom(
    name="target_geom",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=[0.001, 0, 0],
    rgba=[0, 1, 0, 0.5],
    contype=0,
    conaffinity=0,
)
render_mj_model = render_spec.compile()
render_mj_model.opt.timestep = eval_env._config.sim_dt
render_mj_data = mujoco.MjData(render_mj_model)
renderer = mujoco.Renderer(render_mj_model, height=512, width=512)
camera_name = "janelia_cam"


def policy_params_fn(current_step, make_policy, params):
    global rollout_rng

    # ── Generate rollout with current policy ──
    inference_fn = make_policy(params, deterministic=True)
    rollout_rng, reset_rng = jax.random.split(rollout_rng)
    state = jit_reset(reset_rng)
    rollout = [state]

    rng = rollout_rng
    for _ in range(ppo_params.episode_length):
        act_rng, rng = jax.random.split(rng)
        action, _ = inference_fn(state.obs, act_rng)
        state = jit_step(state, action)
        rollout.append(state)

    # ── Render frames ──
    video_path = str(ckpt_path / f"{current_step}.mp4")
    fps = max(1, int(1.0 / eval_env.dt))

    with imageio.get_writer(video_path, fps=fps) as video:
        for s in rollout:
            render_mj_data.qpos[:] = np.array(s.data.qpos)
            render_mj_data.qvel[:] = np.array(s.data.qvel)
            # Set mocap target so the green sphere renders
            render_mj_data.mocap_pos[:] = np.array(s.data.mocap_pos)
            mujoco.mj_forward(render_mj_model, render_mj_data)
            renderer.update_scene(render_mj_data, camera=camera_name)
            video.append_data(renderer.render())

    # ── Log video to wandb (commit=False so progress_fn commits) ──
    if USE_WANDB:
        wandb.log(
            {"eval/rollout": wandb.Video(video_path, format="mp4")},
            commit=False,
        )
    print(f"  -> saved rollout video: {video_path}")

    # ── Save checkpoint ──
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    path = ckpt_path / f"{current_step}"
    orbax_checkpointer.save(path, params, force=True, save_args=save_args)


# ── Build train function ────────────────────────────────────────────────────
training_params = dict(ppo_params)
del training_params["network_factory"]

train_fn = functools.partial(
    ppo.train,
    **training_params,
    network_factory=functools.partial(
        ppo_networks.make_ppo_networks, **ppo_params.network_factory
    ),
    restore_checkpoint_path=restore_checkpoint_path,
    progress_fn=progress,
    wrap_env_fn=wrapper.wrap_for_brax_training,
    policy_params_fn=policy_params_fn,
)


# ── Train ───────────────────────────────────────────────────────────────────
print(f"Starting training: {ppo_params.num_timesteps:,} timesteps, "
      f"{ppo_params.num_envs} envs, {ppo_params.num_evals} evals")
print(f"Action size: {env.action_size} (12 muscles), Obs: 14D")

make_inference_fn, params, _ = train_fn(environment=env, eval_env=eval_env)

if len(times) > 1:
    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")
