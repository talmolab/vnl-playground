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
@jax.jit
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
@jax.jit
@jax.vmap
def physics_rollout(state0):
    def step_fn(mj_data, _):
        new_mj_data = mjx.step(env.mjx_model, mj_data)
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
# Wrapper handles vmapping, I think
#wrapped_env = mujoco_playground.wrapper.wrap_for_brax_training(wrapped_env, full_reset=True)
state0_wrapped = jax.vmap(wrapped_env.reset)(reset_keys)

@jax.jit
@jax.vmap
def rollout_wrapped_env(state0):
    def step_fn(state, _):
        return wrapped_env.step(state, jp.zeros((wrapped_env.action_size,))), None
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
make_policy = brax.training.agents.ppo.networks.make_inference_fn(nets)


def rollout_with_net(state0, normalizer_params, net_params):
    policy = make_policy((normalizer_params, net_params))
    def step_fn(state, _):
        action, _ = policy(state.obs, jax.random.key(0))
        return wrapped_env.step(state, action), None
    return jax.lax.scan(step_fn, state0, None, length=100)[0]
rollout_with_net = jax.vmap(rollout_with_net, in_axes=(0, None, None))
rollout_with_net = jax.jit(rollout_with_net)#, static_argnums=(1,))

print("Compiling wrapped env rollout with net")
rollout_with_net(state0_wrapped, normalizer_params, net_params)

print("Rolling out wrapped env with net")
start = time.perf_counter()
jax.block_until_ready(
    rollout_with_net(state0_wrapped, normalizer_params, net_params)
)
stop = time.perf_counter()
print(f"Wrapped env stepping with net {100*N_ENVS/(stop-start):.2f} steps/sec")


'''
import yaml
from brax import envs
from track_mjx.io import load
from track_mjx.environment.task.multi_clip_tracking import MultiClipTracking
from track_mjx.environment.task.single_clip_tracking import SingleClipTracking
from track_mjx.environment.walker.rodent import Rodent
from track_mjx.environment.task.reward import RewardConfig

cfg = yaml.load(open("../track-mjx/track_mjx/config/rodent-full-clips.yaml", "r"), yaml.Loader)
all_clips = load.make_multiclip_data("../track-mjx/track_mjx/"+cfg["data_path"])
envs.register_environment("rodent_multi_clip", MultiClipTracking)
env_args = cfg["env_config"]["env_args"]
env_rewards = cfg["env_config"]["reward_weights"]
train_config = cfg["train_setup"]["train_config"]
walker_config = cfg["walker_config"]
traj_config = cfg["reference_config"]

env_args["reset_noise_scale"] = float(env_args["reset_noise_scale"])
env_rewards["var_coeff"] = float(env_rewards["var_coeff"])
env_rewards["jerk_coeff"] = float(env_rewards["jerk_coeff"])

walker = Rodent(**walker_config)
reward_config = RewardConfig(**env_rewards)
track_mjx_env = envs.get_environment(
    env_name=cfg["env_config"]["env_name"],
    reference_clip=all_clips,
    walker=walker,
    reward_config=reward_config,
    **env_args,
    **traj_config,
)

env._mjx_model = track_mjx_env.sys

print("Creating state with track-mjx mjx_model")
reset_keys = jax.random.split(jax.random.key(SEED), N_ENVS)
state0 = jax.vmap(env.reset)(reset_keys)

# Env
@jax.jit
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
'''
