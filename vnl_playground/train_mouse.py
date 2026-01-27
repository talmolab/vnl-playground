"""
Training script for mouse reaching task.
"""

import os
import sys

# Add the parent directory to Python path to ensure we import local version
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import functools
import json
import numpy as np
import cv2
from datetime import datetime

import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import mediapy as media
import mujoco
import wandb
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from etils import epath
from flax.training import orbax_utils
from orbax import checkpoint as ocp
from ml_collections import config_dict
from pprint import pprint

from mujoco_playground import wrapper

from vnl_playground.tasks.mouse import mouse_reach

# Enable persistent compilation cache.
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


ppo_params = config_dict.create(
    num_timesteps=1_000_000_000,
    num_evals=10,
    reward_scaling=1.0,
    episode_length=150,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=10,  # Reduced for better compilation/batching
    num_minibatches=32,
    num_updates_per_batch=4,
    discounting=0.97,
    learning_rate=1e-4,
    entropy_cost=1e-4,
    num_envs=4096,  # Increased for better GPU utilization
    batch_size=512,  # Increased proportionally with num_envs
    max_grad_norm=1.0,
    network_factory=config_dict.create(
        policy_hidden_layer_sizes=(512, 512, 512),
        value_hidden_layer_sizes=(512, 512, 512),
    ),
)

env_name = "mouse-reach"

SUFFIX = None
FINETUNE_PATH = None

# Generate unique experiment name.
now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S")
exp_name = f"{env_name}-{timestamp}"
if SUFFIX is not None:
    exp_name += f"-{SUFFIX}"
print(f"Experiment name: {exp_name}")

# Possibly restore from the latest checkpoint.
if FINETUNE_PATH is not None:
    FINETUNE_PATH = epath.Path(FINETUNE_PATH)
    latest_ckpts = list(FINETUNE_PATH.glob("*"))
    latest_ckpts = [ckpt for ckpt in latest_ckpts if ckpt.is_dir()]
    latest_ckpts.sort(key=lambda x: int(x.name))
    latest_ckpt = latest_ckpts[-1]
    restore_checkpoint_path = latest_ckpt
    print(f"Restoring from: {restore_checkpoint_path}")
else:
    restore_checkpoint_path = None


ckpt_path = epath.Path("checkpoints").resolve() / exp_name
ckpt_path.mkdir(parents=True, exist_ok=True)
print(f"{ckpt_path}")

# Get env config for logging
env_cfg = mouse_reach.default_config()

# Convert config to dict and handle non-serializable types (e.g., PosixGPath)
env_cfg_dict = env_cfg.to_dict()
for k, v in env_cfg_dict.items():
    if hasattr(v, "__fspath__"):  # Path-like objects
        env_cfg_dict[k] = str(v)

with open(ckpt_path / "config.json", "w") as fp:
    json.dump(env_cfg_dict, fp, indent=4)

# Setup wandb logging.
USE_WANDB = True

if USE_WANDB:
    wandb.init(project="vnl-mjx-rl", config=env_cfg, id=f"mouse-reach-{exp_name}")
    wandb.config.update(
        {
            "env_name": env_name,
        }
    )

x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]


def progress(num_steps, metrics):
    pprint(f"Step {num_steps}")
    pprint(metrics)
    # Log to wandb.
    if USE_WANDB:
        wandb.log(metrics, step=num_steps)


# Create renderer and jit functions for video generation (setup before training)
def create_video_logging_fn(env, ckpt_path, episode_length):
    """Create a function for generating and logging rollout videos."""
    mj_model = env._mj_model
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=512, width=512)

    # JIT-compiled rollout generation using jax.lax.scan
    @functools.partial(jax.jit, static_argnums=(1,))
    def generate_rollout(rng, inference_fn):
        """Generate a full rollout in one JIT-compiled function."""
        state = env.reset(rng)

        def step_fn(carry, _):
            state, rng = carry
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = inference_fn(state.obs, act_rng)
            next_state = env.step(state, ctrl)
            return (next_state, rng), state.data.qpos

        # Scan over episode length to collect all qpos
        (final_state, _), qposes = jax.lax.scan(
            step_fn, (state, rng), None, length=episode_length
        )
        # Prepend initial state qpos
        all_qposes = jp.concatenate([state.data.qpos[None], qposes], axis=0)
        return all_qposes

    def policy_params_fn(current_step, make_policy, params):
        print(f"Generating rollout video and saving checkpoint at step {current_step}")

        # Create inference function
        inference_fn = make_policy(params)

        # Generate rollout using JIT-compiled scan
        rng = jax.random.PRNGKey(current_step)
        qposes_rollout = generate_rollout(rng, inference_fn)
        qposes_rollout = np.asarray(qposes_rollout)  # Convert to numpy for rendering
        
        # Render and save video using mujoco renderer + opencv
        video_path = f"{ckpt_path}/{current_step}.mp4"
        
        try:
            # Setup opencv video writer
            fps = int(1.0 / env.dt)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (512, 512))
            
            # Render frames using mujoco renderer
            for qpos in qposes_rollout:
                mj_data.qpos = qpos
                mujoco.mj_forward(mj_model, mj_data)
                renderer.update_scene(mj_data)
                frame = renderer.render()
                # Convert RGB to BGR for opencv
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)
            
            video_writer.release()
            
            # Log to wandb (don't commit because progress_fn is called after)
            if USE_WANDB:
                wandb.log({"eval/rollout": wandb.Video(video_path, format="mp4")}, commit=False)
            print(f"Video saved to {video_path}")
        except Exception as e:
            print(f"Warning: Failed to create video: {e}")
        
        # Save checkpoint
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        path = ckpt_path / f"{current_step}"
        orbax_checkpointer.save(path, params, force=True, save_args=save_args)
        print(f"Checkpoint saved to {path}")
    
    return policy_params_fn


def old_policy_params_fn(current_step, make_policy, params):
    del make_policy  # Unused.
    print(f"Saving checkpoint at step {current_step}")
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    path = ckpt_path / f"{current_step}"
    orbax_checkpointer.save(path, params, force=True, save_args=save_args)
    print(f"Checkpoint saved to {path}")


training_params = dict(ppo_params)
del training_params["network_factory"]

if __name__ == "__main__":
    print("=" * 80)
    print("Starting Mouse Reaching Training")
    print("=" * 80)
    
    print(f"Creating environments...")
    env = mouse_reach.MouseReach()
    eval_env = mouse_reach.MouseReach()
    print(f"Environment action size: {env.action_size}")
    print(f"Environment observation size: {env.observation_size}")
    
    # Create video logging function with environment context
    policy_params_fn = create_video_logging_fn(env, ckpt_path, ppo_params.episode_length)
    
    # Create train function with video logging
    train_fn = functools.partial(
        ppo.train,
        **training_params,
        network_factory=functools.partial(
            ppo_networks.make_ppo_networks, **ppo_params.network_factory
        ),
        restore_checkpoint_path=restore_checkpoint_path,
        progress_fn=progress,
        wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training, full_reset=True),
        policy_params_fn=policy_params_fn,
    )

    print("Starting PPO training...")
    print("=" * 80)
    make_inference_fn, params, _ = train_fn(environment=env, eval_env=eval_env)
    
    print("=" * 80)
    print("Training complete!")
    print("=" * 80)
