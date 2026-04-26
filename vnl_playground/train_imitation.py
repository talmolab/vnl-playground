"""
Temporary train script for quick-start training
"""

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # visible GPU masks
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import vnl_playground.naccdmax_patch  # noqa: F401  # monkey-patch naccdmax default

import functools
import json
from datetime import datetime
import numpy as np
import imageio

import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import mediapy as media
import mujoco
import wandb
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.acme import running_statistics

from etils import epath
from flax.training import orbax_utils
from IPython.display import clear_output, display
from orbax import checkpoint as ocp
from ml_collections import config_dict
from tqdm import tqdm

from mujoco_playground import locomotion, wrapper
from mujoco_playground.config import locomotion_params

from vnl_playground.tasks.rodent import imitation
from vnl_playground.tasks import wrappers as rodent_wrappers

# Enable persistent compilation cache.
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

env_cfg = imitation.default_config()
env_cfg.mujoco_impl = "warp"

# Monsees gap-crossing reference clips
env_cfg.reference_data_path = epath.Path("/home/talmolab/Desktop/SalkResearch/monsees-retarget/output/monsees_gap_reference_clips.h5")
env_cfg.clip_length = 100          # 100 frames per clip (2s @ 50Hz)
env_cfg.start_frame_range = [0, 10]
env_cfg.keep_clips_idx = None      # Use all clips

# Warp backend memory — 256 envs to fit in 32GB VRAM
# (imitation env tracks more state per env than gap-crossing)
env_cfg.naconmax = 12 * 256        # 256 envs * 12 contacts/env
env_cfg.njmax = 400

ppo_params = config_dict.create(
    num_envs=256,
    num_timesteps=int(2_000_000_000),
    batch_size=1024,
    num_minibatches=16,
    num_updates_per_batch=4,
    learning_rate=1e-4,
    clipping_epsilon=0.2,
    discounting=0.97,
    action_repeat=1,
    entropy_cost=1e-3,
    reward_scaling=1.0,
    normalize_observations=True,
    unroll_length=5,
    episode_length=400,
    max_grad_norm=1.0,
    network_factory=config_dict.create(
        policy_hidden_layer_sizes=(512, 512, 512, 512),
        value_hidden_layer_sizes=(512, 512, 512, 512),
    ),
    eval_every=50_000_000,  # num_evals = num_timesteps // eval_every = 40
)

env_name = "monsees-gap-imitation"

from pprint import pprint

pprint(ppo_params)


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

with open(ckpt_path / "config.json", "w") as fp:
    json.dump(env_cfg.to_dict(), fp, indent=4, default=lambda o: str(o))

# Setup wandb logging.
USE_WANDB = True

if USE_WANDB:
    wandb.init(project="vnl-playground", group="monsees-gap-imitation", config=env_cfg, id=f"imitation-{exp_name}")
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
    restore_checkpoint_path=restore_checkpoint_path,
    progress_fn=progress_fn,
    wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training, full_reset=True),
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
    env = rodent_wrappers.FlattenObsWrapper(imitation.Imitation(config=env_cfg))
    eval_env = rodent_wrappers.FlattenObsWrapper(imitation.Imitation(config=env_cfg))

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
    )
