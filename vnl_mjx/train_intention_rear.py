"""
Temporary train script for quick-start training
"""

import os
from typing import Callable, Mapping

from mujoco_playground._src import mjx_env
from omegaconf import OmegaConf

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # visible GPU masks
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import functools
import json
from datetime import datetime
import numpy as np
import imageio

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

from mujoco_playground import wrapper

from vnl_mjx.tasks.rodent import head_track_rear
from vnl_mjx.tasks.rodent import wrappers

from track_mjx.agent import checkpointing
from track_mjx.agent import wandb_logging
from track_mjx.agent.mlp_ppo import ppo_networks as track_networks

# Enable persistent compilation cache.
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

env_cfg = head_track_rear.default_config()

mimic_checkpoint_path = "/n/holylabs-olveczky/Users/charleszhang/track-mjx/model_checkpoints/250929_210018_111014"
mimic_cfg = OmegaConf.create(
    checkpointing.load_config_from_checkpoint(mimic_checkpoint_path)
)
decoder_policy_fn = track_networks.make_decoder_policy_fn(mimic_checkpoint_path)


ppo_params = config_dict.create(
    num_timesteps=int(1e9),  # 1 billion
    reward_scaling=1.0,
    episode_length=1500,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=20,
    num_minibatches=8,
    num_updates_per_batch=2,
    discounting=0.9,
    learning_rate=1e-3,
    entropy_cost=1e-2,
    num_envs=8192,
    batch_size=1024,
    max_grad_norm=1.0,
    network_factory=config_dict.create(
        policy_hidden_layer_sizes=(1024, 512, 256),
        value_hidden_layer_sizes=(1024, 512, 256),
    ),
    eval_every=10_000_000,  # num_evals = num_timesteps // eval_every
)

env_name = "head_track_rear"
env_cfg.nconmax *= ppo_params.num_envs

from pprint import pprint

pprint(f"ppo_params: {ppo_params}")


SUFFIX = None
FINETUNE_PATH = None

# Generate unique experiment name.
now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S")
exp_name = f"{env_name}-{timestamp}"
if SUFFIX is not None:
    exp_name += f"-{SUFFIX}"
print(f"Experiment name: {exp_name}")

ckpt_path = epath.Path("checkpoints").resolve() / exp_name
ckpt_path.mkdir(parents=True, exist_ok=True)
print(f"{ckpt_path}")

with open(ckpt_path / "config.json", "w") as fp:
    json.dump(env_cfg.to_dict(), fp, indent=4, default=lambda o: str(o))

# Setup wandb logging.
USE_WANDB = True

if USE_WANDB:
    wandb.init(project="vnl-mjx-rl", config=env_cfg, id=f"head_track_rear-{exp_name}")
    wandb.config.update(
        {
            "env_name": env_name,
        }
    )


def wandb_progress(num_steps, metrics):
    pprint(f"Step {num_steps}")
    pprint(metrics)
    wandb.log(metrics)


def progress(num_steps, metrics):
    pprint(f"Step {num_steps}")
    pprint(metrics)


progress_fn = wandb_progress if USE_WANDB else progress


training_params = dict(ppo_params)
del training_params["network_factory"]
del training_params["eval_every"]

# stuff to make logging inference fn in this file
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
    restore_checkpoint_path=None,
    progress_fn=progress_fn,
    wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training),
    # policy_params_fn=policy_params_fn,
)


def make_logging_inference_fn(ppo_networks):
    """Creates params and inference function for the PPO agent.
    The policy takes the params as an input, so different sets of params can be used.
    """

    def make_logging_policy(deterministic=False):
        policy_network = ppo_networks.policy_network
        # can modify this to provide stochastic action + noise
        parametric_action_distribution = ppo_networks.parametric_action_distribution

        def logging_policy(
            params,
            observations,
            key_sample,
        ):
            param_subset = (params[0], params[1])
            logits = policy_network.apply(*param_subset, observations)
            # logits comes from policy directly, raw predictions that decoder generates (action, intention_mean, intention_logvar)
            if deterministic:
                return (
                    jp.array(ppo_networks.parametric_action_distribution.mode(logits)),
                    {},
                )
            # action sampling is happening here, according to distribution parameter logits
            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )
            # probability of selection specific action, actions with higher reward should have higher probability
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


if __name__ == "__main__":
    env = head_track_rear.HeadTrackRear(config=env_cfg)
    env = wrappers.HighLevelWrapper(
        wrappers.FlattenObsWrapper(env),
        decoder_policy_fn,
        mimic_cfg.network_config.intention_size,
        0,  # the head track task has no non-proprioceptive obs
    )
    eval_env = wrappers.HighLevelWrapper(
        wrappers.FlattenObsWrapper(head_track_rear.HeadTrackRear(config=env_cfg)),
        decoder_policy_fn,
        mimic_cfg.network_config.intention_size,
        0,  # the head track task has no non-proprioceptive obs
    )

    # render a rollout in the policy_params_fn to log to wandb at each step
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    rng = jax.random.PRNGKey(0)
    start_state = jit_reset(rng)
    mj_model = env._mj_model
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=512, width=512)
    ppo_network = network_factory(
        start_state.obs.shape[-1],
        env.action_size,
        preprocess_observations_fn=normalize,
    )
    make_logging_policy = make_logging_inference_fn(ppo_network)
    jit_logging_inference_fn = jax.jit(make_logging_policy(deterministic=True))

    def policy_params_fn(current_step, make_policy, params, jit_logging_inference_fn):
        del make_policy  # Unused.

        # generate a rollout
        rollout = [start_state]
        state = start_state
        rng = jax.random.PRNGKey(0)
        for _ in range(ppo_params.episode_length):
            _, rng = jax.random.split(rng)
            action, _ = jit_logging_inference_fn(params, state.obs, rng)
            state = jit_step(state, action)
            rollout.append(state)

        # render and log
        qposes_rollout = np.array([state.data.qpos for state in rollout])
        video_path = f"{ckpt_path}/{current_step}.mp4"

        with imageio.get_writer(video_path, fps=int((1.0 / env.dt))) as video:
            for qpos in qposes_rollout:
                mj_data.qpos = qpos
                mujoco.mj_forward(mj_model, mj_data)
                renderer.update_scene(
                    mj_data,
                    camera="close_profile-rodent",
                )
                video.append_data(renderer.render())

        # don't commit because progress_fn is called after
        wandb.log({"eval/rollout": wandb.Video(video_path, format="mp4")}, commit=False)
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        path = ckpt_path / f"{current_step}"
        orbax_checkpointer.save(path, params, force=True, save_args=save_args)

    # only run the training if this file is run as a script
    make_inference_fn, params, _ = train_fn(
        environment=env,
        eval_env=eval_env,
        policy_params_fn=functools.partial(
            policy_params_fn, jit_logging_inference_fn=jit_logging_inference_fn
        ),
    )
