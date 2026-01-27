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
import imageio
from datetime import datetime

import jax
import jax.numpy as jp
import mujoco
import wandb
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.acme import running_statistics
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
    num_evals=20,
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
    num_envs=8192,  # Increased for better GPU utilization
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


def make_logging_inference_fn(ppo_network):
    """Creates a logging inference function that supports deterministic evaluation.

    Args:
        ppo_network: The PPO network containing policy_network and parametric_action_distribution.

    Returns:
        A function that creates a logging policy with deterministic option.
    """
    def make_logging_policy(deterministic=False):
        policy_network = ppo_network.policy_network
        parametric_action_distribution = ppo_network.parametric_action_distribution

        def logging_policy(params, observations, key_sample):
            param_subset = (params[0], params[1])
            logits = policy_network.apply(*param_subset, observations)

            if deterministic:
                # Use mode (mean) of distribution for deterministic evaluation
                return (
                    jp.array(parametric_action_distribution.mode(logits)),
                    {},
                )

            # Stochastic: sample from distribution
            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )
            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
            postprocessed_actions = parametric_action_distribution.postprocess(raw_actions)
            return jp.array(postprocessed_actions), {
                "log_prob": log_prob,
                "raw_action": raw_actions,
            }

        return logging_policy

    return make_logging_policy


# Create renderer and jit functions for video generation (setup before training)
def create_video_logging_fn(env, ckpt_path, episode_length, ppo_network):
    """Create a function for generating and logging rollout videos.

    Args:
        env: The environment to run rollouts in.
        ckpt_path: Path to save checkpoints and videos.
        episode_length: Length of rollout episodes.
        ppo_network: The PPO network for creating deterministic inference.
    """
    mj_model = env._mj_model
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=512, width=512)

    # Create deterministic logging policy
    # Normalization happens inside policy_network.apply via preprocess_observations_fn
    # params structure: (normalizer_params, policy_params, value_params)
    # policy_network.apply takes (normalizer_params, policy_params, observations)
    make_logging_policy = make_logging_inference_fn(ppo_network)
    jit_deterministic_policy = jax.jit(make_logging_policy(deterministic=True))

    # JIT env functions (only for jax impl, warp handles its own compilation)
    if env._config.mujoco_impl == "jax":
        jit_reset = jax.jit(env.reset)
        jit_step = jax.jit(env.step)
    else:
        jit_reset = env.reset
        jit_step = env.step

    def policy_params_fn(current_step, make_policy, params):
        print(f"Generating rollout video and saving checkpoint at step {current_step}")
        del make_policy  # Unused, we use deterministic_policy instead

        # Generate rollout (matching train_imitation.py style)
        rng = jax.random.PRNGKey(current_step)
        state = jit_reset(rng)
        rollout_qpos = [np.array(state.data.qpos)]

        # Capture mocap_pos for target rendering (set once in reset, constant throughout episode)
        target_mocap_pos = np.array(state.data.mocap_pos)

        for _ in range(episode_length):
            _, rng = jax.random.split(rng)
            # Pass raw obs - normalization happens inside policy via preprocess_observations_fn
            action, _ = jit_deterministic_policy(params, state.obs, rng)
            state = jit_step(state, action)
            rollout_qpos.append(np.array(state.data.qpos))

        qposes_rollout = np.array(rollout_qpos)

        # Render and save video using mujoco renderer + imageio
        video_path = f"{ckpt_path}/{current_step}.mp4"

        try:
            fps = int(1.0 / env.dt)

            # Use imageio for browser-compatible H.264 codec
            with imageio.get_writer(video_path, fps=fps) as video:
                for qpos in qposes_rollout:
                    mj_data.qpos = qpos
                    mj_data.mocap_pos[:] = target_mocap_pos  # Set target position for rendering
                    mujoco.mj_forward(mj_model, mj_data)
                    renderer.update_scene(mj_data)
                    frame = renderer.render()
                    video.append_data(frame)
            
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
    env = mouse_reach.MouseReach(config_overrides={"target_mode": "random_volume"})
    eval_env = mouse_reach.MouseReach(config_overrides={"target_mode": "random_volume"})
    print(f"Environment action size: {env.action_size}")
    print(f"Environment observation size: {env.observation_size}")

    # Create normalization function
    normalize = running_statistics.normalize

    # Create PPO network for deterministic logging inference
    # Use env.observation_size directly (computed from model dimensions, works with jax/warp)
    obs_size = env.observation_size

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **ppo_params.network_factory
    )
    ppo_network = network_factory(
        obs_size,
        env.action_size,
        preprocess_observations_fn=normalize,
    )

    # Create video logging function with environment context and deterministic policy
    policy_params_fn = create_video_logging_fn(
        env, ckpt_path, ppo_params.episode_length, ppo_network
    )

    # Create train function with video logging
    train_fn = functools.partial(
        ppo.train,
        **training_params,
        network_factory=network_factory,
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
