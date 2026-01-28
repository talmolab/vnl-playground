"""
Training script for mouse arm imitation task.
Loads all configuration from YAML file.
"""

import argparse
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
import yaml
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

from vnl_playground.tasks.mouse.imitation import MouseImitation, load_config
from vnl_playground.tasks.mouse.reference_clips import MouseReferenceClips
from vnl_playground.tasks.mouse import consts

# Enable persistent compilation cache.
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def load_full_config(yaml_path: str) -> dict:
    """Load the full YAML config file and return all sections."""
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def flatten_obs(obs):
    """Flatten nested observation dict to a single array."""
    flat_parts = []
    for key in sorted(obs.keys()):
        val = obs[key]
        if isinstance(val, dict):
            flat_parts.append(flatten_obs(val))
        else:
            flat_parts.append(val.flatten())
    return jp.concatenate(flat_parts)


def make_logging_inference_fn(ppo_network):
    """Creates a logging inference function that supports deterministic evaluation."""
    def make_logging_policy(deterministic=False):
        policy_network = ppo_network.policy_network
        parametric_action_distribution = ppo_network.parametric_action_distribution

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
            postprocessed_actions = parametric_action_distribution.postprocess(raw_actions)
            return jp.array(postprocessed_actions), {
                "log_prob": log_prob,
                "raw_action": raw_actions,
            }

        return logging_policy

    return make_logging_policy


def create_video_logging_fn(env, ckpt_path, episode_length, ppo_network, use_wandb):
    """Create a function for generating and logging rollout videos."""
    mj_model = env._mj_model
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=512, width=512)

    make_logging_policy = make_logging_inference_fn(ppo_network)
    jit_deterministic_policy = jax.jit(make_logging_policy(deterministic=True))

    if env._config.mujoco_impl == "jax":
        jit_reset = jax.jit(env.reset)
        jit_step = jax.jit(env.step)
    else:
        jit_reset = env.reset
        jit_step = env.step

    def policy_params_fn(current_step, make_policy, params):
        print(f"Generating rollout video and saving checkpoint at step {current_step}")
        del make_policy

        rng = jax.random.PRNGKey(current_step)
        state = jit_reset(rng)

        rollout_states = [state]
        rollout_qpos = [np.array(state.data.qpos)]

        for _ in range(episode_length):
            _, rng = jax.random.split(rng)
            # Flatten observation dict for policy
            flat_obs = flatten_obs(state.obs)
            action, _ = jit_deterministic_policy(params, flat_obs, rng)
            state = jit_step(state, action)
            rollout_states.append(state)
            rollout_qpos.append(np.array(state.data.qpos))

        qposes_rollout = np.array(rollout_qpos)

        video_path = f"{ckpt_path}/{current_step}.mp4"

        try:
            fps = int(1.0 / env.dt)

            # Render with ghost showing reference motion
            frames = env.render(rollout_states, height=512, width=512, render_ghost=True)

            with imageio.get_writer(video_path, fps=fps) as video:
                for frame in frames:
                    video.append_data(frame)

            if use_wandb:
                wandb.log({"eval/rollout": wandb.Video(video_path, format="mp4")}, commit=False)
            print(f"Video saved to {video_path}")
        except Exception as e:
            print(f"Warning: Failed to create video: {e}")
            import traceback
            traceback.print_exc()

        # Save checkpoint
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        path = ckpt_path / f"{current_step}"
        orbax_checkpointer.save(path, params, force=True, save_args=save_args)
        print(f"Checkpoint saved to {path}")

    return policy_params_fn


def main(config_path: str):
    print("=" * 80)
    print(f"Loading config from: {config_path}")
    print("=" * 80)

    # Load full YAML config
    full_cfg = load_full_config(config_path)

    # Extract sections
    train_setup = full_cfg.get("train_setup", {})
    train_config = train_setup.get("train_config", {})
    network_config = full_cfg.get("network_config", {})
    logging_config = full_cfg.get("logging_config", {})

    # Load env config using the existing load_config function
    env_cfg = load_config(config_path)

    # Print loaded config for verification
    print("\n" + "=" * 40)
    print("TRAIN CONFIG (from YAML)")
    print("=" * 40)
    for k, v in train_config.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 40)
    print("NETWORK CONFIG (from YAML)")
    print("=" * 40)
    for k, v in network_config.items():
        print(f"  {k}: {v}")

    # Build PPO params from YAML config
    ppo_params = config_dict.create(
        num_timesteps=train_config.get("num_timesteps", 500_000_000),
        num_evals=train_config.get("num_timesteps", 500_000_000) // train_setup.get("eval_every", 10_000_000),
        reward_scaling=train_config.get("reward_scaling", 1.0),
        episode_length=train_setup.get("episode_length", 80),
        normalize_observations=train_config.get("normalize_observations", True),
        action_repeat=train_config.get("action_repeat", 1),
        unroll_length=train_config.get("unroll_length", 10),
        num_minibatches=train_config.get("num_minibatches", 32),
        num_updates_per_batch=train_config.get("num_updates_per_batch", 4),
        discounting=train_config.get("discounting", 0.97),
        learning_rate=train_config.get("learning_rate", 3e-4),
        entropy_cost=train_config.get("entropy_cost", 1e-3),
        clipping_epsilon=train_config.get("clipping_epsilon", 0.2),
        num_envs=train_config.get("num_envs", 4096),
        batch_size=train_config.get("batch_size", 256),
        max_grad_norm=train_config.get("max_grad_norm", 1.0),
        seed=train_config.get("seed", 0),
        deterministic_eval=train_config.get("deterministic_eval", True),
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=tuple(network_config.get("policy_hidden_layer_sizes", [256, 256, 256])),
            value_hidden_layer_sizes=tuple(network_config.get("value_hidden_layer_sizes", [256, 256, 256])),
        ),
    )

    print("\n" + "=" * 40)
    print("PPO PARAMS (built from YAML)")
    print("=" * 40)
    pprint(dict(ppo_params))

    # Setup experiment naming
    env_name = "mouse-imitation"
    run_name = train_setup.get("run_name", "imitation")
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    exp_name = f"{run_name}-{timestamp}"
    print(f"\nExperiment name: {exp_name}")

    # Checkpoint path
    model_path = logging_config.get("model_path", "checkpoints")
    ckpt_path = epath.Path(model_path).resolve() / exp_name
    ckpt_path.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint path: {ckpt_path}")

    # Restore checkpoint if specified
    restore_path = train_setup.get("checkpoint_to_restore")
    if restore_path:
        restore_path = epath.Path(restore_path)
        if restore_path.exists():
            latest_ckpts = list(restore_path.glob("*"))
            latest_ckpts = [ckpt for ckpt in latest_ckpts if ckpt.is_dir()]
            latest_ckpts.sort(key=lambda x: int(x.name))
            restore_checkpoint_path = latest_ckpts[-1] if latest_ckpts else None
            print(f"Restoring from: {restore_checkpoint_path}")
        else:
            restore_checkpoint_path = None
            print(f"Warning: checkpoint path {restore_path} does not exist")
    else:
        restore_checkpoint_path = None

    # Save config to checkpoint directory
    with open(ckpt_path / "config.yaml", "w") as fp:
        yaml.dump(full_cfg, fp, default_flow_style=False)

    # Setup wandb logging
    USE_WANDB = True
    project_name = logging_config.get("project_name", "vnl-mjx-rl")

    if USE_WANDB:
        wandb.init(
            project=project_name,
            config=full_cfg,
            name=exp_name,
            id=f"{run_name}-{exp_name}",
        )

    def progress(num_steps, metrics):
        pprint(f"Step {num_steps}")
        pprint(metrics)
        if USE_WANDB:
            wandb.log(metrics, step=num_steps)

    # Load reference clips
    reference_data_path = env_cfg.reference_data_path
    print(f"\nLoading reference clips from {reference_data_path}...")
    reference_clips = MouseReferenceClips(
        str(reference_data_path),
        n_frames_per_clip=env_cfg.clip_length,
    )

    # Split into train/test
    train_ratio = train_setup.get("train_subset_ratio", 0.8)
    train_clips, test_clips = reference_clips.split(train_ratio=train_ratio, seed=42)

    print(f"\nCreating environments...")
    env = MouseImitation(config=env_cfg, clips=train_clips)
    eval_env = MouseImitation(config=env_cfg, clips=test_clips)
    print(f"Environment action size: {env.action_size}")
    print(f"Environment observation size: {env.observation_size}")

    # Compute episode length based on clip structure
    steps_per_frame = (1 / env_cfg.mocap_hz) / env_cfg.ctrl_dt
    computed_episode_length = int(
        (env_cfg.clip_length - env_cfg.start_frame_range[-1] - env_cfg.reference_length)
        * steps_per_frame
    )
    print(f"Computed episode length: {computed_episode_length}")

    # Build training params dict
    training_params = dict(ppo_params)
    del training_params["network_factory"]
    training_params["episode_length"] = computed_episode_length

    # Create normalization function
    normalize = running_statistics.normalize

    # Create PPO network
    obs_size = env.observation_size

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **ppo_params.network_factory
    )
    ppo_network = network_factory(
        obs_size,
        env.action_size,
        preprocess_observations_fn=normalize,
    )

    # Create video logging function
    policy_params_fn = create_video_logging_fn(
        env, ckpt_path, computed_episode_length, ppo_network, USE_WANDB
    )

    # Wrapper that flattens dict observations
    def flatten_obs_wrapper(env_fn):
        """Wrap environment to flatten dict observations."""
        def wrapped_reset(rng):
            state = env_fn.reset(rng)
            flat_obs = flatten_obs(state.obs)
            return state.replace(obs=flat_obs)

        def wrapped_step(state, action):
            new_state = env_fn.step(state, action)
            flat_obs = flatten_obs(new_state.obs)
            return new_state.replace(obs=flat_obs)

        class FlattenedEnv:
            def __init__(self, base_env):
                self._base_env = base_env

            def reset(self, rng):
                return wrapped_reset(rng)

            def step(self, state, action):
                return wrapped_step(state, action)

            @property
            def observation_size(self):
                return self._base_env.observation_size

            @property
            def action_size(self):
                return self._base_env.action_size

            @property
            def dt(self):
                return self._base_env.dt

            def __getattr__(self, name):
                return getattr(self._base_env, name)

        return FlattenedEnv(env_fn)

    # Wrap environments
    wrapped_env = flatten_obs_wrapper(env)
    wrapped_eval_env = flatten_obs_wrapper(eval_env)

    # Create train function
    train_fn = functools.partial(
        ppo.train,
        **training_params,
        network_factory=network_factory,
        restore_checkpoint_path=restore_checkpoint_path,
        progress_fn=progress,
        wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training, full_reset=True),
        policy_params_fn=policy_params_fn,
    )

    print("\n" + "=" * 80)
    print("Starting PPO training...")
    print("=" * 80)
    make_inference_fn, params, _ = train_fn(environment=wrapped_env, eval_env=wrapped_eval_env)

    print("=" * 80)
    print("Training complete!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train mouse imitation")
    parser.add_argument(
        "--config",
        type=str,
        default="vnl_playground/config/mouse_imitation.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    main(args.config)
