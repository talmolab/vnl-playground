import os

import mujoco_playground
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import functools
import time
import jax
import jax.numpy as jp
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from etils import epath
from ml_collections import config_dict
from mujoco import mjx
import mujoco_playground

import vnl_mjx.tasks.rodent.imitation
import vnl_mjx.flatten_wrapper

import brax.training.agents.ppo.networks
import brax.training.acme.running_statistics

SEED = 42
N_ENVS = 512

env_cfg = vnl_mjx.tasks.rodent.imitation.default_config()
#env_cfg.mujoco_impl = "warp"
#env_cfg.ctrl_dt = 0.01
env_cfg.nconmax *= N_ENVS

print("Creating env")
env = vnl_mjx.tasks.rodent.imitation.Imitation(config=env_cfg)

print("Creating first state")
reset_keys = jax.random.split(jax.random.key(SEED), N_ENVS)
state0 = jax.vmap(env.reset)(reset_keys)

# Env
@jax.vmap
def rollout_env(state0):
    def step_fn(state, _):
        return env.step(state, jp.zeros(env.action_size)), None
    return jax.lax.scan(step_fn, state0, None, length=100)[0]
print("Compiling env rollout")
rollout_env(state0)

print("Rolling out env")
start = time.perf_counter()
jax.block_until_ready(rollout_env(state0))
stop = time.perf_counter()
print(f"Env stepping {100*N_ENVS/(stop-start):.2f} steps/sec")


# Physics
@jax.vmap
def physics_rollout(state0):
    def step_fn(mj_data, _):
        new_mj_data = mjx.step(env.mj_model, mj_data)
        return new_mj_data, None
    return jax.lax.scan(step_fn, state0.data, None, length=1000)[0]
print("Compiling physics rollout")
physics_rollout(state0)

print("Rolling out physics only")
start = time.perf_counter()
jax.block_until_ready(physics_rollout(state0))
stop = time.perf_counter()
print(f"Physics stepping {1000*N_ENVS/(stop-start):.2f} steps/sec")

# Wrapped env
wrapped_env = vnl_mjx.flatten_wrapper.FlattenObsWrapper(env)
wrapped_env = mujoco_playground.wrapper.wrap_for_brax_training(wrapped_env, full_reset=True)
state0_wrapped = jax.vmap(wrapped_env.reset)(reset_keys)
@jax.vmap
def rollout_wrapped_env(state0):
    def step_fn(state, _):
        return wrapped_env.step(state, jp.zeros(wrapped_env.action_size)), None
    return jax.lax.scan(step_fn, state0, None, length=100)[0]
print("Compiling wrapped env rollout")
rollout_wrapped_env(state0_wrapped)

print("Rolling out wrapped env")
start = time.perf_counter()
jax.block_until_ready(rollout_wrapped_env(state0_wrapped))
stop = time.perf_counter()
print(f"Wrapped env stepping {100*N_ENVS/(stop-start):.2f} steps/sec")

obs_preprocessing = brax.training.acme.running_statistics.normalize
nets = brax.training.agents.ppo.networks.make_ppo_networks(
    wrapped_env.observation_size, wrapped_env.action_size,
    obs_preprocessing, (16,), (16,)
)
net_params = nets.policy_network.init(jax.random.key(SEED))
normalizer_params = brax.training.acme.running_statistics.init_state(jp.array((wrapped_env.observation_size,)))

def rollout_with_net(state0, nets, normalizer_params, net_params):
    def step_fn(state, _):
        action = nets.policy_network.apply(normalizer_params, net_params, state.obs)
        return wrapped_env.step(state, action), None
    return jax.lax.scan(step_fn, state0, None, length=100)[0]
rollout_with_net = jax.vmap(rollout_with_net, in_axes=(0, None, None, None))

print("Compiling wrapped env rollout with net")
rollout_with_net(state0_wrapped, nets, normalizer_params, net_params)

print("Rolling out wrapped env with net")
start = time.perf_counter()
jax.block_until_ready(
    rollout_with_net(state0_wrapped, nets, normalizer_params, net_params)
)
stop = time.perf_counter()
print(f"Wrapped env stepping with net {100*N_ENVS/(stop-start):.2f} steps/sec")