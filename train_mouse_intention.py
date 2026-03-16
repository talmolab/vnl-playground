"""
Training script for mouse arm imitation with Variational Information Bottleneck.

Architecture:
- Encoder: obs(task_obs) → MLP+LayerNorm → (μ, log σ²)  [multivariate Gaussian]
- Reparameterize: z ~ N(μ, σ²)
- Decoder: concat(z, proprioception) → MLP → action params

Loss = PPO surrogate + VF loss + entropy bonus
     + KL(q(z|x) || N(0,I))        [information bottleneck]
     + AR(1) temporal smoothness    [latent consistency]
"""

import os
import sys

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Any, Mapping, NamedTuple, Sequence

import jax
import jax.numpy as jp
import mujoco
import optax
import wandb
from brax.training import distribution
from brax.training import networks
from brax.training.acme import running_statistics
from etils import epath
from flax.training import orbax_utils
from flax import linen
from orbax import checkpoint as ocp
from ml_collections import config_dict
from pprint import pprint

from mujoco_playground import wrapper

from vnl_playground.tasks.mouse.imitation import MouseImitation, default_config
from vnl_playground.tasks.mouse.reference_clips import MouseReferenceClips
from vnl_playground.tasks.mouse import consts

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


# =============================================================================
# Variational Intention Network (Encoder → Gaussian → Decoder)
# =============================================================================


class Encoder(linen.Module):
    """Maps task observations → (mean, logvar) of latent Gaussian."""
    layer_sizes: Sequence[int]
    latents: int
    activation: networks.ActivationFn = linen.silu
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()

    @linen.compact
    def __call__(self, x):
        for i, hidden_size in enumerate(self.layer_sizes):
            x = linen.Dense(
                hidden_size, name=f"hidden_{i}",
                kernel_init=self.kernel_init,
            )(x)
            x = self.activation(x)
            x = linen.LayerNorm()(x)
        mean = linen.Dense(self.latents, name="fc_mean")(x)
        logvar = linen.Dense(self.latents, name="fc_logvar")(x)
        return mean, logvar


class Decoder(linen.Module):
    """Maps concat(z, proprioception) → action distribution params."""
    layer_sizes: Sequence[int]
    activation: networks.ActivationFn = linen.silu
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()

    @linen.compact
    def __call__(self, x):
        for i, hidden_size in enumerate(self.layer_sizes):
            x = linen.Dense(
                hidden_size, name=f"hidden_{i}",
                kernel_init=self.kernel_init,
            )(x)
            if i != len(self.layer_sizes) - 1:
                x = self.activation(x)
                x = linen.LayerNorm()(x)
        return x


class IntentionPolicy(linen.Module):
    """Full encoder-decoder VAE policy operating on flat observations.

    Splits flat obs into [proprioception | task_obs] based on proprio_size,
    encodes task_obs to latent distribution, samples z, decodes action params.
    """
    encoder_layers: Sequence[int]
    decoder_layers: Sequence[int]
    latents: int
    proprio_size: int

    def setup(self):
        self.encoder = Encoder(
            layer_sizes=self.encoder_layers, latents=self.latents
        )
        self.decoder = Decoder(layer_sizes=self.decoder_layers)

    def __call__(self, obs_flat, key, deterministic=False):
        proprio = obs_flat[..., :self.proprio_size]
        task_obs = obs_flat[..., self.proprio_size:]

        mean, logvar = self.encoder(task_obs)

        # Use where instead of if to keep JIT-safe when deterministic is traced
        std = jp.exp(0.5 * logvar)
        eps = jax.random.normal(key, logvar.shape)
        z_sampled = mean + eps * std
        z = jp.where(deterministic, mean, z_sampled)

        decoder_input = jp.concatenate([z, proprio], axis=-1)
        logits = self.decoder(decoder_input)
        return logits, mean, logvar


# =============================================================================
# VAE Loss Components
# =============================================================================


def compute_kl_to_gaussian_prior(latent_mean, latent_logvar):
    """KL(q(z|x) || N(0,I)). Inputs shape [T, B, D] or [N, D]."""
    return -0.5 * jp.mean(
        1 + latent_logvar - jp.square(latent_mean) - jp.exp(latent_logvar)
    )


def compute_ar1_temporal_loss(latent_mean, discount, truncation):
    """L2 smoothness between consecutive latent means.

    Masks out episode boundaries (done or truncated).
    latent_mean: (T, B, D), discount/truncation: (T, B).
    """
    z_prev = latent_mean[:-1]
    z_curr = latent_mean[1:]
    valid_mask = discount[:-1] * (1.0 - truncation[:-1])
    l2_diff = jp.mean(jp.square(z_curr - z_prev), axis=-1)
    masked_l2 = l2_diff * valid_mask
    return jp.sum(masked_l2) / jp.maximum(jp.sum(valid_mask), 1.0)


def create_ramp_schedule(
    max_value=0.1, min_value=0.0001, ramp_steps=1000,
    warmup_steps=0, schedule="linear", period=45,
):
    """Create a schedule function for KL/AR1 weight annealing."""
    def schedule_fn(step):
        step = jp.asarray(step, dtype=jp.float32)
        if schedule == "linear":
            progress = jp.clip((step - warmup_steps) / ramp_steps, 0.0, 1.0)
            is_warmup = step < warmup_steps
            return jp.where(
                is_warmup, min_value,
                min_value + progress * (max_value - min_value),
            )
        elif schedule == "cosine":
            angle = (2 * jp.pi * step) / period
            amp = (max_value - min_value) / 2
            mid = (max_value + min_value) / 2
            return mid + amp * jp.cos(angle)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
    return schedule_fn


# =============================================================================
# PPO Utilities
# =============================================================================


class Transition(NamedTuple):
    obs: Any          # (B, obs_dim) flat
    action: Any       # (B, act_dim)
    raw_action: Any   # (B, act_dim)
    log_prob: Any     # (B,)
    value: Any        # (B,)
    reward: Any       # (B,)
    done: Any         # (B,)
    truncation: Any   # (B,)


def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    """Vectorised GAE via reverse scan. All inputs (T, B)."""
    T = rewards.shape[0]

    def body(carry, t_rev):
        gae, next_val = carry
        t = T - 1 - t_rev
        delta = rewards[t] + gamma * next_val * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * gae
        return (gae, values[t]), gae

    _, advantages_rev = jax.lax.scan(
        body, (jp.zeros_like(last_value), last_value), jp.arange(T)
    )
    advantages = advantages_rev[::-1]
    returns = advantages + values
    return advantages, returns


def flatten_obs(obs):
    """Flatten nested observation dict to a single array (sorted keys)."""
    flat_parts = []
    for key in sorted(obs.keys()):
        val = obs[key]
        if isinstance(val, dict):
            flat_parts.append(flatten_obs(val))
        else:
            flat_parts.append(val.flatten())
    return jp.concatenate(flat_parts)


# =============================================================================
# Config
# =============================================================================

ppo_params = config_dict.create(
    num_timesteps=500_000_000,
    num_evals=10,
    reward_scaling=1.0,
    episode_length=100,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=10,
    num_minibatches=32,
    num_updates_per_batch=4,
    discounting=0.97,
    learning_rate=3e-4,
    entropy_cost=1e-3,
    num_envs=4096,
    max_grad_norm=1.0,
    gae_lambda=0.95,
    clip_eps=0.3,
    vf_coef=0.5,
    latent_kl_weight=1e-3,
    latent_ar1_weight=1e-3,
    network_factory=config_dict.create(
        encoder_hidden_layer_sizes=(512, 512, 512),
        decoder_hidden_layer_sizes=(512, 512, 512),
        value_hidden_layer_sizes=(512, 512, 512),
        latent_size=4,
    ),
)

env_name = "mouse-imitation-intention"
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

ckpt_path = epath.Path("checkpoints").resolve() / exp_name
ckpt_path.mkdir(parents=True, exist_ok=True)
print(f"{ckpt_path}")

env_cfg = default_config()
env_cfg_dict = env_cfg.to_dict()
for k, v in env_cfg_dict.items():
    if hasattr(v, "__fspath__"):
        env_cfg_dict[k] = str(v)
with open(ckpt_path / "config.json", "w") as fp:
    json.dump(env_cfg_dict, fp, indent=4, default=str)

USE_WANDB = True
if USE_WANDB:
    wandb.init(project="vnl-mjx-rl", config=env_cfg,
               id=f"intention-{exp_name}")
    wandb.config.update({
        "env_name": env_name,
        "policy_type": "variational_intention_bottleneck",
        **dict(ppo_params.network_factory),
    })


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Mouse Arm Imitation -- Variational Information Bottleneck")
    print("=" * 80)
    nf = ppo_params.network_factory
    print(f"Encoder layers: {nf.encoder_hidden_layer_sizes}")
    print(f"Decoder layers: {nf.decoder_hidden_layer_sizes}")
    print(f"Latent size:    {nf.latent_size}")
    print(f"KL weight:      {ppo_params.latent_kl_weight}")
    print(f"AR(1) weight:   {ppo_params.latent_ar1_weight}")

    # ------------------------------------------------------------------
    # Environment setup
    # ------------------------------------------------------------------
    print(f"Loading reference clips from {consts.MOUSE_REFERENCE_DATA_PATH}...")
    reference_clips = MouseReferenceClips(
        str(consts.MOUSE_REFERENCE_DATA_PATH),
        n_frames_per_clip=env_cfg.clip_length,
    )
    train_clips, test_clips = reference_clips.split(train_ratio=0.8, seed=42)

    env = MouseImitation(config=env_cfg, clips=train_clips)
    eval_env = MouseImitation(config=env_cfg, clips=test_clips)
    print(f"Action size: {env.action_size}  Obs size: {env.observation_size}")

    steps_per_frame = (1 / env_cfg.mocap_hz) / env_cfg.ctrl_dt
    episode_length = int(
        (env_cfg.clip_length - env_cfg.start_frame_range[-1]
         - env_cfg.reference_length) * steps_per_frame
    )
    print(f"Episode length: {episode_length}")

    # Determine obs split: flatten_obs sorts keys alphabetically
    # Top-level: proprioception (p) < task_obs (t)
    # So flat_obs = [proprioception | task_obs(joint, wrist)]
    _dummy_rng = jax.random.PRNGKey(99)
    _dummy_state = env.reset(_dummy_rng)
    _dummy_obs = _dummy_state.obs
    _proprio_flat = _dummy_obs["proprioception"].flatten()
    _task_parts = []
    for k in sorted(_dummy_obs["task_obs"].keys()):
        _task_parts.append(_dummy_obs["task_obs"][k].flatten())
    _task_flat = jp.concatenate(_task_parts)
    proprio_size = _proprio_flat.shape[0]
    task_obs_size = _task_flat.shape[0]
    obs_size = proprio_size + task_obs_size
    act_size = env.action_size
    print(f"Proprio size: {proprio_size}  Task obs size: {task_obs_size}  "
          f"Total obs: {obs_size}")

    # ------------------------------------------------------------------
    # Wrap envs for vectorised rollouts
    # ------------------------------------------------------------------
    def flatten_obs_wrapper(env_fn):
        class W:
            def __init__(self, e):
                self._e = e
            def reset(self, rng):
                s = self._e.reset(rng)
                return s.replace(obs=flatten_obs(s.obs))
            def step(self, state, action):
                s = self._e.step(state, action)
                return s.replace(obs=flatten_obs(s.obs))
            @property
            def observation_size(self):
                return self._e.observation_size
            @property
            def action_size(self):
                return self._e.action_size
            @property
            def dt(self):
                return self._e.dt
            def __getattr__(self, name):
                return getattr(self._e, name)
        return W(env_fn)

    wrapped_env = flatten_obs_wrapper(env)
    wrapped_eval_env = flatten_obs_wrapper(eval_env)

    wrap_fn = functools.partial(wrapper.wrap_for_brax_training, full_reset=True)
    train_env = wrap_fn(
        wrapped_env,
        episode_length=episode_length,
        action_repeat=ppo_params.action_repeat,
    )
    test_env = wrap_fn(
        wrapped_eval_env,
        episode_length=episode_length,
        action_repeat=ppo_params.action_repeat,
    )

    # ------------------------------------------------------------------
    # Networks
    # ------------------------------------------------------------------
    action_dist = distribution.NormalTanhDistribution(event_size=act_size)
    param_size = action_dist.param_size

    policy_module = IntentionPolicy(
        encoder_layers=nf.encoder_hidden_layer_sizes,
        decoder_layers=list(nf.decoder_hidden_layer_sizes) + [param_size],
        latents=nf.latent_size,
        proprio_size=proprio_size,
    )
    value_module = networks.MLP(
        layer_sizes=list(nf.value_hidden_layer_sizes) + [1],
        activation=linen.swish,
        kernel_init=jax.nn.initializers.lecun_uniform(),
    )

    num_envs = ppo_params.num_envs
    dummy_obs = jp.zeros((1, obs_size))
    dummy_key = jax.random.PRNGKey(0)

    key = jax.random.PRNGKey(0)
    key, pk, vk, ek = jax.random.split(key, 4)
    policy_params = policy_module.init(pk, dummy_obs, dummy_key)
    value_params = value_module.init(vk, dummy_obs)
    normalizer_params = running_statistics.init_state(jp.zeros(obs_size))

    optimizer = optax.chain(
        optax.clip_by_global_norm(ppo_params.max_grad_norm),
        optax.adam(ppo_params.learning_rate),
    )
    opt_state = optimizer.init((policy_params, value_params))

    # ------------------------------------------------------------------
    # Hyperparameters for JIT closures
    # ------------------------------------------------------------------
    unroll_length = ppo_params.unroll_length
    num_updates = ppo_params.num_updates_per_batch
    num_minibatches = ppo_params.num_minibatches
    gamma = ppo_params.discounting
    gae_lambda = ppo_params.gae_lambda
    reward_scaling = ppo_params.reward_scaling
    clip_eps = ppo_params.clip_eps
    vf_coef = ppo_params.vf_coef
    entropy_cost = ppo_params.entropy_cost
    kl_weight = ppo_params.latent_kl_weight
    ar1_weight = ppo_params.latent_ar1_weight
    mb_env_size = num_envs // num_minibatches  # envs per minibatch

    # ------------------------------------------------------------------
    # JIT-compiled core functions
    # ------------------------------------------------------------------

    @jax.jit
    def collect_rollout(policy_params, value_params, normalizer_params,
                        env_state, rng):
        """Collect unroll_length transitions (no recurrent carry)."""

        def step_fn(carry, _):
            state, k = carry
            k, ak, pk = jax.random.split(k, 3)

            obs_norm = running_statistics.normalize(state.obs, normalizer_params)
            logits, _, _ = policy_module.apply(
                policy_params, obs_norm, pk
            )
            raw_action = action_dist.sample_no_postprocessing(logits, ak)
            log_prob = action_dist.log_prob(logits, raw_action)
            action = action_dist.postprocess(raw_action)
            value = jp.squeeze(
                value_module.apply(value_params, obs_norm), axis=-1
            )

            next_state = train_env.step(state, action)
            truncation = next_state.info.get(
                "truncation", jp.zeros_like(next_state.done)
            )

            transition = Transition(
                obs=state.obs,
                action=action,
                raw_action=raw_action,
                log_prob=log_prob,
                value=value,
                reward=next_state.reward,
                done=next_state.done,
                truncation=truncation,
            )
            return (next_state, k), transition

        (final_state, _), rollout = jax.lax.scan(
            step_fn,
            (env_state, rng),
            None,
            length=unroll_length,
        )
        return final_state, rollout

    def _sgd_step(policy_params, value_params, opt_state, normalizer_params,
                  mb_obs, mb_raw, mb_lp, mb_adv, mb_ret, mb_done, mb_trunc,
                  rng):
        """Single gradient update on a temporal minibatch.

        All mb_* arrays have shape (T, B', ...) where B' = mb_env_size.
        Temporal structure is preserved for AR(1) loss.
        """

        def loss_fn(params):
            pp, vp = params
            T, Bp = mb_obs.shape[:2]

            obs_norm = running_statistics.normalize(mb_obs, normalizer_params)

            # Flatten temporal + env dims for forward pass
            flat_obs = obs_norm.reshape(T * Bp, -1)
            enc_key = jax.random.fold_in(rng, 0)
            flat_logits, flat_mean, flat_logvar = policy_module.apply(
                pp, flat_obs, enc_key
            )

            # Reshape back to (T, B', ...)
            logits = flat_logits.reshape(T, Bp, -1)
            latent_mean = flat_mean.reshape(T, Bp, -1)
            latent_logvar = flat_logvar.reshape(T, Bp, -1)

            # Policy gradient loss
            new_log_prob = action_dist.log_prob(
                logits.reshape(T * Bp, -1),
                mb_raw.reshape(T * Bp, -1),
            ).reshape(T, Bp)

            ratio = jp.exp(new_log_prob - mb_lp)
            adv = (mb_adv - jp.mean(mb_adv)) / (jp.std(mb_adv) + 1e-8)

            pg1 = -adv * ratio
            pg2 = -adv * jp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            policy_loss = jp.mean(jp.maximum(pg1, pg2))

            # Value loss
            new_value = jp.squeeze(
                value_module.apply(vp, flat_obs), axis=-1
            ).reshape(T, Bp)
            value_loss = jp.mean(jp.square(new_value - mb_ret))

            # Entropy
            ent_key = jax.random.fold_in(rng, 1)
            entropy = jp.mean(
                action_dist.entropy(logits.reshape(T * Bp, -1), ent_key)
            )

            # KL divergence to N(0, I)
            kl_loss = compute_kl_to_gaussian_prior(latent_mean, latent_logvar)

            # AR(1) temporal smoothness
            discount = 1.0 - mb_done
            ar1_loss = compute_ar1_temporal_loss(
                latent_mean, discount, mb_trunc
            )

            total = (policy_loss
                     + vf_coef * value_loss
                     - entropy_cost * entropy
                     + kl_weight * kl_loss
                     + ar1_weight * ar1_loss)

            # Per-dim KL for active dim counting
            kl_per_dim = -0.5 * (
                1 + latent_logvar - jp.square(latent_mean)
                - jp.exp(latent_logvar)
            )  # (T, B', D)
            mean_kl_per_dim = jp.mean(kl_per_dim, axis=(0, 1))  # (D,)
            active_dims = jp.sum(mean_kl_per_dim > 0.01)

            return total, {
                "total_loss": total,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "kl_loss": kl_loss,
                "ar1_loss": ar1_loss,
                "approx_kl": jp.mean((ratio - 1.0) - jp.log(ratio)),
                "latent_kl_weight": kl_weight,
                "latent_ar1_weight": ar1_weight,
                "latent_mean_norm": jp.mean(jp.sqrt(
                    jp.sum(jp.square(latent_mean), axis=-1)
                )),
                "latent_std_mean": jp.mean(jp.exp(0.5 * latent_logvar)),
                "active_latent_dims": active_dims,
                "latent_rate_nats": jp.mean(
                    jp.sum(kl_per_dim, axis=-1)),
            }

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            (policy_params, value_params)
        )
        grad_norm = optax.global_norm(grads)
        updates, new_opt_state = optimizer.update(
            grads, opt_state, (policy_params, value_params)
        )
        new_pp, new_vp = optax.apply_updates(
            (policy_params, value_params), updates
        )
        metrics["grad_norm"] = grad_norm
        return new_pp, new_vp, new_opt_state, loss, metrics

    @jax.jit
    def prepare_ppo_data(normalizer_params, rollout, env_state_obs, value_params):
        """JIT'd normalizer update + GAE computation."""
        normalizer_params = running_statistics.update(
            normalizer_params, rollout.obs.reshape(-1, obs_size),
        )
        last_obs_norm = running_statistics.normalize(
            env_state_obs, normalizer_params
        )
        last_value = jp.squeeze(
            value_module.apply(value_params, last_obs_norm), axis=-1
        )
        rewards = rollout.reward * reward_scaling
        advantages, returns = compute_gae(
            rewards, rollout.value, rollout.done, last_value, gamma, gae_lambda,
        )
        return normalizer_params, advantages, returns

    @jax.jit
    def run_ppo_epochs(policy_params, value_params, opt_state,
                       normalizer_params,
                       r_obs, r_raw, r_lp, r_adv, r_ret, r_done, r_trunc,
                       key):
        """PPO update with temporal minibatching along the env (B) dimension.

        All r_* arrays have shape (T, B, ...).
        Minibatches of shape (T, B', ...) preserve temporal ordering for AR(1).
        """
        B = r_obs.shape[1]

        def epoch_step(carry, _):
            pp, vp, os, k = carry
            k, perm_key = jax.random.split(k)
            perm = jax.random.permutation(perm_key, B)

            def mb_step(carry2, mb_idx):
                pp2, vp2, os2, k2 = carry2
                k2, ek = jax.random.split(k2)
                start = mb_idx * mb_env_size
                idx = jax.lax.dynamic_slice(perm, (start,), (mb_env_size,))

                pp2, vp2, os2, loss, metrics = _sgd_step(
                    pp2, vp2, os2, normalizer_params,
                    r_obs[:, idx], r_raw[:, idx], r_lp[:, idx],
                    r_adv[:, idx], r_ret[:, idx],
                    r_done[:, idx], r_trunc[:, idx], ek,
                )
                return (pp2, vp2, os2, k2), (loss, metrics)

            (pp, vp, os, k), (losses, all_metrics) = jax.lax.scan(
                mb_step, (pp, vp, os, k), jp.arange(num_minibatches)
            )
            last_metrics = jax.tree.map(lambda x: x[-1], all_metrics)
            return (pp, vp, os, k), last_metrics

        (pp, vp, os, k), epoch_metrics = jax.lax.scan(
            epoch_step,
            (policy_params, value_params, opt_state, key),
            None, length=num_updates,
        )
        final_metrics = jax.tree.map(lambda x: x[-1], epoch_metrics)
        return pp, vp, os, k, final_metrics

    # ------------------------------------------------------------------
    # Eval helpers
    # ------------------------------------------------------------------

    num_eval_envs = 128

    @jax.jit
    def jit_eval_rollout(policy_params, normalizer_params, eval_state, rng):
        """JIT'd deterministic eval rollout on test_env."""

        def step_fn(carry, _):
            state, k = carry
            k, _ = jax.random.split(k)
            obs_norm = running_statistics.normalize(state.obs, normalizer_params)
            logits, _, _ = policy_module.apply(
                policy_params, obs_norm, k, deterministic=True
            )
            action = action_dist.mode(logits)
            next_state = test_env.step(state, action)
            return (next_state, k), (next_state.reward, next_state.done)

        (final_state, _), (rewards, _) = jax.lax.scan(
            step_fn, (eval_state, rng), None, length=episode_length,
        )
        mean_reward = jp.mean(jp.sum(rewards, axis=0))
        std_reward = jp.std(jp.sum(rewards, axis=0))
        return final_state, {
            "eval/episode_reward": mean_reward,
            "eval/episode_reward_std": std_reward,
            "eval/mean_step_reward": jp.mean(rewards),
        }

    jit_eval_reset = jax.jit(eval_env.reset)
    jit_eval_step = jax.jit(eval_env.step)

    # Separate JIT traces for stochastic (training) vs deterministic (eval)
    @functools.partial(jax.jit, static_argnames=("deterministic",))
    def jit_policy_apply(params, obs, key, deterministic=False):
        return policy_module.apply(params, obs, key, deterministic=deterministic)

    def diagnostic_rollout(policy_params, normalizer_params, seed=0):
        """Single-episode rollout collecting states + per-step diagnostics.

        Returns:
            rollout_states: list of env states (for render_ghost)
            episode_data: dict of numpy arrays keyed by diagnostic name
        """
        rng = jax.random.PRNGKey(seed)
        state = jit_eval_reset(rng)
        rollout_states = [state]
        means_list, logvars_list = [], []
        rewards_list, actions_list = [], []
        proprio_list, task_obs_list = [], []

        for _ in range(episode_length):
            flat = flatten_obs(state.obs)
            obs_norm = running_statistics.normalize(
                flat[None], normalizer_params
            )
            logits, mean, logvar = jit_policy_apply(
                policy_params, obs_norm, rng, deterministic=True
            )
            action = jp.squeeze(action_dist.mode(logits), axis=0)

            # Collect diagnostics
            means_list.append(np.array(mean[0]))
            logvars_list.append(np.array(logvar[0]))
            actions_list.append(np.array(action))
            proprio_list.append(np.array(flat[:proprio_size]))
            task_obs_list.append(np.array(flat[proprio_size:]))

            state = jit_eval_step(state, action)
            rollout_states.append(state)
            rewards_list.append(float(state.reward))

        episode_data = {
            "latent_mean": np.stack(means_list),       # (T, latent_dim)
            "latent_logvar": np.stack(logvars_list),    # (T, latent_dim)
            "reward": np.array(rewards_list),           # (T,)
            "action": np.stack(actions_list),           # (T, act_dim)
            "proprioception": np.stack(proprio_list),   # (T, proprio_size)
            "task_obs": np.stack(task_obs_list),        # (T, task_obs_size)
        }
        return rollout_states, episode_data

    def plot_intention_diagnostics(episode_data, save_path=None):
        """Multi-panel diagnostic figure for the intention bottleneck.

        3 rows x 3 cols = 9 panels:
          Row 0: Latent mean heatmap | Latent std heatmap | KL per step
          Row 1: ||μ|| over time     | AR(1) Δ over time  | Per-dim mean σ
          Row 2: Reward over episode  | Action magnitudes  | Joint tracking
        """
        means = episode_data["latent_mean"]       # (T, D)
        logvars = episode_data["latent_logvar"]   # (T, D)
        rewards = episode_data["reward"]           # (T,)
        actions = episode_data["action"]           # (T, A)
        proprio = episode_data["proprioception"]   # (T, P)
        task_obs = episode_data["task_obs"]        # (T, task)
        stds = np.exp(0.5 * logvars)
        T, D = means.shape

        fig, axes = plt.subplots(3, 3, figsize=(18, 12), squeeze=False)

        # -- Row 0, Col 0: Latent mean heatmap --
        ax = axes[0, 0]
        im = ax.imshow(means.T, aspect="auto", cmap="RdBu_r",
                        interpolation="nearest",
                        vmin=-np.percentile(np.abs(means), 95),
                        vmax=np.percentile(np.abs(means), 95))
        ax.set_xlabel("env step")
        ax.set_ylabel("latent dim")
        ax.set_title("Latent mean μ")
        plt.colorbar(im, ax=ax, fraction=0.046)

        # -- Row 0, Col 1: Latent std heatmap --
        ax = axes[0, 1]
        im = ax.imshow(stds.T, aspect="auto", cmap="viridis",
                        interpolation="nearest")
        ax.set_xlabel("env step")
        ax.set_ylabel("latent dim")
        ax.set_title("Latent std σ")
        plt.colorbar(im, ax=ax, fraction=0.046)

        # -- Row 0, Col 2: KL divergence per step --
        ax = axes[0, 2]
        kl_per_step = -0.5 * np.sum(
            1 + logvars - means**2 - np.exp(logvars), axis=-1
        )
        ax.plot(kl_per_step, linewidth=0.8, color="steelblue")
        ax.fill_between(range(T), kl_per_step, alpha=0.2, color="steelblue")
        ax.set_xlabel("env step")
        ax.set_ylabel("KL(q || N(0,I))")
        ax.set_title(f"KL divergence  (mean={np.mean(kl_per_step):.2f})")

        # -- Row 1, Col 0: Latent mean magnitude over time --
        ax = axes[1, 0]
        mean_norm = np.linalg.norm(means, axis=-1)
        ax.plot(mean_norm, linewidth=0.8, color="darkorange")
        ax.set_xlabel("env step")
        ax.set_ylabel("||μ||₂")
        ax.set_title("Latent mean magnitude")

        # -- Row 1, Col 1: AR(1) delta over time --
        ax = axes[1, 1]
        if T > 1:
            ar1_delta = np.linalg.norm(means[1:] - means[:-1], axis=-1)
            ax.plot(ar1_delta, linewidth=0.8, color="forestgreen")
            ax.fill_between(range(T - 1), ar1_delta, alpha=0.2,
                            color="forestgreen")
            ax.set_title(f"AR(1) ||Δμ||₂  (mean={np.mean(ar1_delta):.3f})")
        else:
            ax.set_title("AR(1) ||Δμ||₂  (N/A)")
        ax.set_xlabel("env step")
        ax.set_ylabel("||μₜ - μₜ₋₁||₂")

        # -- Row 1, Col 2: Per-dimension mean σ (bar chart) --
        ax = axes[1, 2]
        mean_std_per_dim = np.mean(stds, axis=0)
        colors = plt.cm.viridis(np.linspace(0, 1, D))
        ax.bar(range(D), mean_std_per_dim, color=colors, alpha=0.8)
        ax.axhline(1.0, color="red", ls="--", alpha=0.5, label="prior σ=1")
        ax.set_xlabel("latent dim")
        ax.set_ylabel("mean σ")
        ax.set_title("Per-dimension latent std")
        ax.legend(fontsize=7)

        # -- Row 2, Col 0: Reward over episode --
        ax = axes[2, 0]
        ax.plot(rewards, linewidth=0.8, color="crimson")
        ax.fill_between(range(T), rewards, alpha=0.15, color="crimson")
        cum_reward = np.cumsum(rewards)
        ax2 = ax.twinx()
        ax2.plot(cum_reward, linewidth=0.8, color="gray", alpha=0.6, ls="--")
        ax2.set_ylabel("cumulative", color="gray", fontsize=8)
        ax.set_xlabel("env step")
        ax.set_ylabel("reward")
        ax.set_title(f"Episode reward  (total={cum_reward[-1]:.1f})")

        # -- Row 2, Col 1: Action magnitudes --
        ax = axes[2, 1]
        act_norm = np.linalg.norm(actions, axis=-1)
        ax.plot(act_norm, linewidth=0.8, color="mediumpurple")
        # Show a few individual action dims
        n_show = min(actions.shape[1], 4)
        for j in range(n_show):
            ax.plot(actions[:, j], linewidth=0.4, alpha=0.5,
                    label=f"a[{j}]")
        ax.set_xlabel("env step")
        ax.set_ylabel("action")
        ax.set_title("Action magnitudes")
        ax.legend(fontsize=6, ncol=2)

        # -- Row 2, Col 2: Joint tracking (proprioception + task_obs) --
        ax = axes[2, 2]
        # proprioception = [qpos(4), qvel(4)], task_obs = [joint_delta(4), wrist_delta(3)]
        n_joints = proprio_size // 2
        joint_pos = proprio[:, :n_joints]       # qpos
        joint_delta = task_obs[:, :n_joints]    # target - current
        joint_labels = ["sh_elv", "sh_ext", "sh_rot", "elbow"][:n_joints]
        for j in range(n_joints):
            ax.plot(np.abs(joint_delta[:, j]), linewidth=0.8,
                    label=joint_labels[j] if j < len(joint_labels)
                    else f"j{j}")
        ax.set_xlabel("env step")
        ax.set_ylabel("|joint target - current|")
        ax.set_title("Joint tracking error")
        ax.legend(fontsize=7)

        fig.suptitle("Intention Bottleneck Diagnostics", fontsize=13,
                     fontweight="bold", y=1.01)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def eval_and_log(step, policy_params, value_params, normalizer_params):
        """Eval metrics + MuJoCo+ghost video + multi-panel diagnostics + ckpt."""
        import time as _time
        t_eval = _time.time()

        # -- Fast JIT'd eval metrics (128 envs) --
        eval_key_step = jax.random.PRNGKey(step)
        eval_state_reset = test_env.reset(
            jax.random.split(eval_key_step, num_eval_envs)
        )
        _, eval_metrics = jit_eval_rollout(
            policy_params, normalizer_params, eval_state_reset, eval_key_step,
        )
        jax.tree.map(lambda x: x.block_until_ready(), eval_metrics)
        eval_time = _time.time() - t_eval

        eval_log = {k: float(v) for k, v in eval_metrics.items()}
        eval_log["eval/eval_time"] = eval_time
        pprint(eval_log)
        if USE_WANDB:
            wandb.log(eval_log, step=step)

        # -- Single-episode diagnostic rollout (states + latent stats) --
        rollout_states, episode_data = diagnostic_rollout(
            policy_params, normalizer_params, seed=step
        )

        # -- MuJoCo + ghost behavior video --
        try:
            fps = int(1.0 / eval_env.dt)
            frames = eval_env.render(
                rollout_states, height=512, width=512, render_ghost=True
            )
            video_path = f"{ckpt_path}/{step}.mp4"
            with imageio.get_writer(video_path, fps=fps) as vid:
                for f in frames:
                    vid.append_data(f)
            if USE_WANDB:
                wandb.log(
                    {"eval/rollout": wandb.Video(video_path, format="mp4")},
                    step=step,
                )
            print(f"  video -> {video_path}")
        except Exception as e:
            print(f"  video failed: {e}")

        # -- Multi-panel latent diagnostics (3x3 = 9 panels) --
        try:
            diag_path = f"{ckpt_path}/{step}_latent_diag.png"
            fig = plot_intention_diagnostics(episode_data, save_path=diag_path)
            if USE_WANDB:
                wandb.log(
                    {"eval/latent_diagnostics": wandb.Image(fig)}, step=step
                )
            plt.close(fig)
            print(f"  latent diagnostics -> {diag_path}")
        except Exception as e:
            print(f"  latent diagnostics failed: {e}")

        # -- Log episode-level latent summary scalars --
        means = episode_data["latent_mean"]
        logvars = episode_data["latent_logvar"]
        # Per-dim KL: (T, D)
        kl_per_dim = -0.5 * (1 + logvars - means**2 - np.exp(logvars))
        kl_total = float(np.mean(np.sum(kl_per_dim, axis=-1)))
        mean_kl_per_dim = np.mean(kl_per_dim, axis=0)  # (D,)
        active_dims = int(np.sum(mean_kl_per_dim > 0.01))
        ar1_mean = float(np.mean(
            np.linalg.norm(means[1:] - means[:-1], axis=-1)
        )) if len(means) > 1 else 0.0
        latent_scalars = {
            "eval/latent_kl_mean": kl_total,
            "eval/latent_rate_nats": kl_total,
            "eval/latent_ar1_mean": ar1_mean,
            "eval/latent_mean_norm": float(np.mean(
                np.linalg.norm(means, axis=-1)
            )),
            "eval/latent_std_mean": float(np.mean(np.exp(0.5 * logvars))),
            "eval/active_latent_dims": active_dims,
            "eval/episode_reward_single": float(np.sum(
                episode_data["reward"]
            )),
        }
        if USE_WANDB:
            wandb.log(latent_scalars, step=step)

        # -- Checkpoint --
        params_to_save = (normalizer_params, policy_params, value_params)
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params_to_save)
        path = ckpt_path / f"{step}"
        orbax_checkpointer.save(path, params_to_save, force=True,
                                save_args=save_args)
        print(f"  checkpoint -> {path}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    print("Initialising training env state...")
    env_state = train_env.reset(jax.random.split(ek, num_envs))

    total_steps = 0
    steps_per_unroll = unroll_length * num_envs
    num_timesteps = ppo_params.num_timesteps
    num_evals = ppo_params.num_evals
    steps_per_eval = num_timesteps // num_evals
    next_eval_at = steps_per_eval

    print(f"Training for {num_timesteps:,} steps  "
          f"({steps_per_unroll:,} per unroll, eval every {steps_per_eval:,})")
    print(f"Temporal minibatch: {unroll_length} steps x {mb_env_size} envs "
          f"= {unroll_length * mb_env_size} samples")
    print("=" * 80)

    import time as _time
    t0 = _time.time()

    while total_steps < num_timesteps:
        # -- Collect rollout (T, B, ...) --
        key, rollout_key = jax.random.split(key)
        env_state, rollout = collect_rollout(
            policy_params, value_params, normalizer_params,
            env_state, rollout_key,
        )
        total_steps += steps_per_unroll

        # -- Normalizer update + GAE (JIT'd) --
        normalizer_params, advantages, returns = prepare_ppo_data(
            normalizer_params, rollout, env_state.obs, value_params,
        )

        # -- PPO epochs with temporal minibatching --
        key, update_key = jax.random.split(key)
        policy_params, value_params, opt_state, _, metrics = run_ppo_epochs(
            policy_params, value_params, opt_state, normalizer_params,
            rollout.obs, rollout.raw_action, rollout.log_prob,
            advantages, returns,
            rollout.done, rollout.truncation,
            update_key,
        )

        # -- Logging --
        if total_steps >= next_eval_at or total_steps >= num_timesteps:
            elapsed = _time.time() - t0
            sps = total_steps / max(elapsed, 1e-6)
            log_metrics = {k: float(v) for k, v in metrics.items()}
            log_metrics["sps"] = sps
            log_metrics["total_steps"] = total_steps
            log_metrics["mean_reward"] = float(jp.mean(rollout.reward))
            pprint(f"Step {total_steps:>12,}  SPS {sps:,.0f}")
            pprint(log_metrics)
            if USE_WANDB:
                wandb.log(log_metrics, step=total_steps)

            eval_and_log(total_steps, policy_params, value_params,
                         normalizer_params)
            next_eval_at += steps_per_eval

    print("=" * 80)
    print("Training complete!")
    print("=" * 80)
