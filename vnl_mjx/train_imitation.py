"""
Testing the rodent imitation environment. Based on train.py.
"""

import os
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import functools
import json
from datetime import datetime

import jax
import jax.numpy as jp
import wandb
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from etils import epath
from flax.training import orbax_utils
from orbax import checkpoint as ocp
from ml_collections import config_dict
from mujoco_playground import wrapper

import vnl_mjx.tasks.rodent.imitation

# Enable persistent compilation cache.
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

env_cfg = vnl_mjx.tasks.rodent.imitation.default_config()

ppo_params = config_dict.create(
    num_timesteps=int(1.5e9),
    num_evals=300,
    reward_scaling=1.0,
    episode_length=200,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=20,
    num_minibatches=16,
    num_updates_per_batch=4,
    discounting=0.98,
    learning_rate=1e-4,
    entropy_cost=1e-2,
    num_envs=4096,
    batch_size=1024,
    max_grad_norm=1.0,
    network_factory=config_dict.create(
        policy_hidden_layer_sizes=(1024, 512, 512, 512, 512, 512, 512, 512, 256, 256),
        value_hidden_layer_sizes=(512, 512, 512, 512, 512, 256),
    ),
)

env_name = "rodent-imitation"

from pprint import pprint
pprint(ppo_params)

# Generate unique experiment name.
now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S")
exp_name = f"{env_name}-{timestamp}"
print(f"Experiment name: {exp_name}")


ckpt_path = epath.Path("checkpoints").resolve() / exp_name
ckpt_path.mkdir(parents=True, exist_ok=True)
print(f"{ckpt_path}")

with open(ckpt_path / "config.json", "w") as fp:
    json.dump(env_cfg.to_dict(), fp, indent=4, default=lambda o: str(o))

# Setup wandb logging.
USE_WANDB = True

if USE_WANDB:
    wanb_config ={
        "env_name": env_name,
        "ppo_params": ppo_params.to_dict(),
        "env_config": env_cfg.to_dict(),
    }
    wandb.init(project="test-playground-refactor", config=wanb_config, name=exp_name)
    wandb.config.update({"env_name": env_name})

def progress(num_steps, metrics):
    # Log to wandb.
    if USE_WANDB:
        wandb.log(metrics, step=num_steps)

def policy_params_fn(current_step, make_policy, params):
    del make_policy  # Unused.
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    path = ckpt_path / f"{current_step}"
    orbax_checkpointer.save(path, params, force=True, save_args=save_args)


training_params = dict(ppo_params)
del training_params["network_factory"]

train_fn = functools.partial(
    ppo.train,
    **training_params,
    network_factory=functools.partial(
        ppo_networks.make_ppo_networks, **ppo_params.network_factory
    ),
    #restore_checkpoint_path=restore_checkpoint_path,
    progress_fn=progress,
    wrap_env_fn=wrapper.wrap_for_brax_training,
    policy_params_fn=policy_params_fn,
)


class FlattenObsWrapper(wrapper.Wrapper):

    def __init__(self, env: wrapper.mjx_env.MjxEnv):
        super().__init__(env)

    def reset(self, rng: jax.Array) -> wrapper.mjx_env.State:
        state = self.env.reset(rng)
        return self._flatten(state)
    
    def step(self, state: wrapper.mjx_env.State, action: jax.Array) -> wrapper.mjx_env.State:
        state = self.env.step(state, action)
        return self._flatten(state)
    
    def _flatten(self, state: wrapper.mjx_env.State) -> wrapper.mjx_env.State:
        state = state.replace(
            obs =jax.flatten_util.ravel_pytree(state.obs)[0],
            metrics = self._flatten_metrics(state.metrics),
        )
        return state
    
    def _flatten_metrics(self, metrics: dict) -> dict:
        new_metrics = {}
        def rec(d: dict, prefix=''):
            for k, v in d.items():
                if isinstance(v, dict):
                    rec(v, prefix + k + '/')
                else:
                    new_metrics[prefix + k] = v
        rec(metrics)
        return new_metrics
    
    @property
    def observation_size(self) -> int:
        rng_shape = jax.eval_shape(jax.random.key, 0)
        flat_obs = lambda rng: self._flatten(self.env.reset(rng).obs)
        obs_size = len(jax.eval_shape(flat_obs, rng_shape))
        return obs_size

env = FlattenObsWrapper(vnl_mjx.tasks.rodent.imitation.Imitation())
eval_env = env#FlattenObsWrapper(vnl_mjx.tasks.rodent.imitation.Imitation())
make_inference_fn, params, _ = train_fn(environment=env, eval_env=eval_env)
