"""
Training script for mouse arm imitation with Hodgkin-Huxley neurons.

Biophysically detailed HH neurons with learnable conductances (g_Na, g_K, g_L).
Gradients flow directly through continuous HH dynamics — no surrogate gradients
needed. Uses soft spike rate output: sigmoid-smoothed time above threshold.

Exponential Euler for gating variables (allows larger dt), explicit Euler for V.
Small neuron count (20/layer) to manage compute cost of full HH dynamics.
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
from typing import Any, NamedTuple, Tuple

import jax
import jax.numpy as jp
import optax
import wandb
from brax.training import distribution
from brax.training import networks
from brax.training.acme import running_statistics
from etils import epath
from flax.training import orbax_utils
from flax import linen
from flax import struct
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
# HH Gating Rate Functions (standard squid giant axon)
# =============================================================================


def _alpha_m(V):
    """Na activation forward rate."""
    x = V + 40.0
    return jp.where(jp.abs(x) < 1e-6, 1.0,
                    0.1 * x / (1.0 - jp.exp(-x / 10.0)))


def _beta_m(V):
    """Na activation backward rate."""
    return 4.0 * jp.exp(-(V + 65.0) / 18.0)


def _alpha_h(V):
    """Na inactivation forward rate."""
    return 0.07 * jp.exp(-(V + 65.0) / 20.0)


def _beta_h(V):
    """Na inactivation backward rate."""
    return 1.0 / (1.0 + jp.exp(-(V + 35.0) / 10.0))


def _alpha_n(V):
    """K activation forward rate."""
    x = V + 55.0
    return jp.where(jp.abs(x) < 1e-6, 0.1,
                    0.01 * x / (1.0 - jp.exp(-x / 10.0)))


def _beta_n(V):
    """K activation backward rate."""
    return 0.125 * jp.exp(-(V + 65.0) / 80.0)


# =============================================================================
# HH Layer
# =============================================================================


class HHLayer(linen.Module):
    """Hodgkin-Huxley layer with learnable per-neuron conductances.

    - g_Na, g_K, g_L learnable via softplus (positive constraint)
    - Exponential Euler for gating variables (stable at larger dt)
    - Explicit Euler for V with clipping for stability
    - Output: soft spike rate (sigmoid average of V above 0 mV)
    - Zero carry maps to resting state via learnable bias offsets
    """

    n_neurons: int
    n_steps: int = 100
    dt: float = 0.05
    current_scale: float = 20.0
    spike_beta: float = 0.5

    @linen.compact
    def __call__(self, x, V_carry, m_carry, h_carry, n_carry):
        num = self.n_neurons
        dt = self.dt

        # Resting-state biases (zero carry -> V_rest=-65, steady-state gates)
        V_bias = self.param('V_bias',
                            lambda k, s: jp.full(s, -65.0), (num,))

        def _init_m(k, s):
            V0 = jp.full(s, -65.0)
            am, bm = _alpha_m(V0), _beta_m(V0)
            return am / (am + bm)

        def _init_h(k, s):
            V0 = jp.full(s, -65.0)
            ah, bh = _alpha_h(V0), _beta_h(V0)
            return ah / (ah + bh)

        def _init_n(k, s):
            V0 = jp.full(s, -65.0)
            an, bn = _alpha_n(V0), _beta_n(V0)
            return an / (an + bn)

        m_bias = self.param('m_bias', _init_m, (num,))
        h_bias = self.param('h_bias', _init_h, (num,))
        n_bias = self.param('n_bias', _init_n, (num,))

        V_b = jax.lax.stop_gradient(V_bias)
        m_b = jax.lax.stop_gradient(m_bias)
        h_b = jax.lax.stop_gradient(h_bias)
        n_b = jax.lax.stop_gradient(n_bias)

        V = V_carry + V_b
        m = m_carry + m_b
        h = h_carry + h_b
        n = n_carry + n_b

        # Learnable conductances (per neuron)
        g_Na_raw = self.param('g_Na_raw',
                              lambda k, s: jp.full(s, 120.0), (num,))
        g_K_raw = self.param('g_K_raw',
                             lambda k, s: jp.full(s, 36.0), (num,))
        g_L_raw = self.param('g_L_raw',
                             lambda k, s: jp.full(s, -1.05), (num,))

        g_Na = jax.nn.softplus(g_Na_raw)
        g_K = jax.nn.softplus(g_K_raw)
        g_L = jax.nn.softplus(g_L_raw)

        E_Na, E_K, E_L = 50.0, -77.0, -54.4

        I_base = linen.Dense(
            num,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            name="input_proj",
        )(x) * self.current_scale

        def step_fn(carry, _):
            V, m, h, n = carry

            I_Na = g_Na * (m ** 3) * h * (V - E_Na)
            I_K = g_K * (n ** 4) * (V - E_K)
            I_L = g_L * (V - E_L)

            dV = I_base - I_Na - I_K - I_L  # C_m = 1
            V_new = jp.clip(V + dt * dV, -100.0, 100.0)

            # Exponential Euler for gating (exact for linear ODE at const V)
            am, bm = _alpha_m(V), _beta_m(V)
            tau_m = 1.0 / (am + bm + 1e-6)
            m_inf = am * tau_m
            m_new = m_inf + (m - m_inf) * jp.exp(-dt / tau_m)

            ah, bh = _alpha_h(V), _beta_h(V)
            tau_h = 1.0 / (ah + bh + 1e-6)
            h_inf = ah * tau_h
            h_new = h_inf + (h - h_inf) * jp.exp(-dt / tau_h)

            an, bn = _alpha_n(V), _beta_n(V)
            tau_n = 1.0 / (an + bn + 1e-6)
            n_inf = an * tau_n
            n_new = n_inf + (n - n_inf) * jp.exp(-dt / tau_n)

            return (V_new, m_new, h_new, n_new), V_new

        (V_final, m_final, h_final, n_final), V_trace = jax.lax.scan(
            step_fn, (V, m, h, n), None, length=self.n_steps,
        )

        above_thresh = jax.nn.sigmoid(self.spike_beta * (V_trace - 0.0))
        output = jp.mean(above_thresh, axis=0)

        V_out = V_final - V_b
        m_out = m_final - m_b
        h_out = h_final - h_b
        n_out = n_final - n_b

        return output, V_out, m_out, h_out, n_out


class DiagnosticHHLayer(linen.Module):
    """HH layer that also returns full V, m, h, n traces."""

    n_neurons: int
    n_steps: int = 100
    dt: float = 0.05
    current_scale: float = 20.0
    spike_beta: float = 0.5

    @linen.compact
    def __call__(self, x, V_carry, m_carry, h_carry, n_carry):
        num = self.n_neurons
        dt = self.dt

        V_bias = self.param('V_bias',
                            lambda k, s: jp.full(s, -65.0), (num,))

        def _init_m(k, s):
            V0 = jp.full(s, -65.0)
            am, bm = _alpha_m(V0), _beta_m(V0)
            return am / (am + bm)

        def _init_h(k, s):
            V0 = jp.full(s, -65.0)
            ah, bh = _alpha_h(V0), _beta_h(V0)
            return ah / (ah + bh)

        def _init_n(k, s):
            V0 = jp.full(s, -65.0)
            an, bn = _alpha_n(V0), _beta_n(V0)
            return an / (an + bn)

        m_bias = self.param('m_bias', _init_m, (num,))
        h_bias = self.param('h_bias', _init_h, (num,))
        n_bias = self.param('n_bias', _init_n, (num,))

        V_b = jax.lax.stop_gradient(V_bias)
        m_b = jax.lax.stop_gradient(m_bias)
        h_b = jax.lax.stop_gradient(h_bias)
        n_b = jax.lax.stop_gradient(n_bias)

        V = V_carry + V_b
        m = m_carry + m_b
        h = h_carry + h_b
        n = n_carry + n_b

        g_Na_raw = self.param('g_Na_raw',
                              lambda k, s: jp.full(s, 120.0), (num,))
        g_K_raw = self.param('g_K_raw',
                             lambda k, s: jp.full(s, 36.0), (num,))
        g_L_raw = self.param('g_L_raw',
                             lambda k, s: jp.full(s, -1.05), (num,))

        g_Na = jax.nn.softplus(g_Na_raw)
        g_K = jax.nn.softplus(g_K_raw)
        g_L = jax.nn.softplus(g_L_raw)

        E_Na, E_K, E_L = 50.0, -77.0, -54.4

        I_base = linen.Dense(
            num,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            name="input_proj",
        )(x) * self.current_scale

        def step_fn(carry, _):
            V, m, h, n = carry

            I_Na = g_Na * (m ** 3) * h * (V - E_Na)
            I_K = g_K * (n ** 4) * (V - E_K)
            I_L = g_L * (V - E_L)

            dV = I_base - I_Na - I_K - I_L
            V_new = jp.clip(V + dt * dV, -100.0, 100.0)

            am, bm = _alpha_m(V), _beta_m(V)
            tau_m = 1.0 / (am + bm + 1e-6)
            m_inf = am * tau_m
            m_new = m_inf + (m - m_inf) * jp.exp(-dt / tau_m)

            ah, bh = _alpha_h(V), _beta_h(V)
            tau_h = 1.0 / (ah + bh + 1e-6)
            h_inf = ah * tau_h
            h_new = h_inf + (h - h_inf) * jp.exp(-dt / tau_h)

            an, bn = _alpha_n(V), _beta_n(V)
            tau_n = 1.0 / (an + bn + 1e-6)
            n_inf = an * tau_n
            n_new = n_inf + (n - n_inf) * jp.exp(-dt / tau_n)

            return (V_new, m_new, h_new, n_new), (V_new, m_new, h_new, n_new)

        (V_final, m_final, h_final, n_final), (V_all, m_all, h_all, n_all) = \
            jax.lax.scan(
                step_fn, (V, m, h, n), None, length=self.n_steps,
            )

        above_thresh = jax.nn.sigmoid(self.spike_beta * (V_all - 0.0))
        output = jp.mean(above_thresh, axis=0)

        V_out = V_final - V_b
        m_out = m_final - m_b
        h_out = h_final - h_b
        n_out = n_final - n_b

        return output, V_out, m_out, h_out, n_out, {
            "V_trace": V_all,   # (n_steps, B, num)
            "m_trace": m_all,
            "h_trace": h_all,
            "n_trace": n_all,
        }


# =============================================================================
# Policy
# =============================================================================


class RecurrentHHPolicy(linen.Module):
    """Recurrent HH policy. Carry: [V|m|h|n] per layer, 4*sum(sizes) total."""

    layer_sizes: tuple = (20, 20)
    output_size: int = 18
    n_steps: int = 100
    dt: float = 0.05
    current_scale: float = 20.0
    spike_beta: float = 0.5

    @linen.compact
    def __call__(self, obs, carry_flat):
        x = obs
        new_carries = []
        idx = 0

        for i, n_neu in enumerate(self.layer_sizes):
            V_c = carry_flat[:, idx:idx + n_neu]; idx += n_neu
            m_c = carry_flat[:, idx:idx + n_neu]; idx += n_neu
            h_c = carry_flat[:, idx:idx + n_neu]; idx += n_neu
            n_c = carry_flat[:, idx:idx + n_neu]; idx += n_neu

            rate, V_out, m_out, h_out, n_out = HHLayer(
                n_neurons=n_neu,
                n_steps=self.n_steps,
                dt=self.dt,
                current_scale=self.current_scale,
                spike_beta=self.spike_beta,
                name=f"hh_{i}",
            )(x, V_c, m_c, h_c, n_c)

            x = rate
            new_carries.extend([V_out, m_out, h_out, n_out])

        logits = linen.Dense(
            self.output_size,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            name="readout",
        )(x)

        new_carry = jp.concatenate(new_carries, axis=-1)
        return logits, new_carry


class DiagnosticHHPolicy(linen.Module):
    """HH policy with diagnostic traces from all layers."""

    layer_sizes: tuple = (20, 20)
    output_size: int = 18
    n_steps: int = 100
    dt: float = 0.05
    current_scale: float = 20.0
    spike_beta: float = 0.5

    @linen.compact
    def __call__(self, obs, carry_flat):
        x = obs
        new_carries = []
        all_diagnostics = {}
        idx = 0

        for i, n_neu in enumerate(self.layer_sizes):
            V_c = carry_flat[:, idx:idx + n_neu]; idx += n_neu
            m_c = carry_flat[:, idx:idx + n_neu]; idx += n_neu
            h_c = carry_flat[:, idx:idx + n_neu]; idx += n_neu
            n_c = carry_flat[:, idx:idx + n_neu]; idx += n_neu

            rate, V_out, m_out, h_out, n_out, diag = DiagnosticHHLayer(
                n_neurons=n_neu,
                n_steps=self.n_steps,
                dt=self.dt,
                current_scale=self.current_scale,
                spike_beta=self.spike_beta,
                name=f"hh_{i}",
            )(x, V_c, m_c, h_c, n_c)

            x = rate
            new_carries.extend([V_out, m_out, h_out, n_out])
            all_diagnostics[f"hh_{i}"] = diag

        logits = linen.Dense(
            self.output_size,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            name="readout",
        )(x)

        new_carry = jp.concatenate(new_carries, axis=-1)
        return logits, new_carry, all_diagnostics


# =============================================================================
# Training Infrastructure
# =============================================================================


@struct.dataclass
class TrainingState:
    policy_params: Any
    value_params: Any
    opt_state: Any
    normalizer_params: Any
    env_state: Any
    membrane_carry: Any
    rng: Any
    total_steps: int


class Transition(NamedTuple):
    obs: Any
    action: Any
    raw_action: Any
    log_prob: Any
    value: Any
    reward: Any
    done: Any
    carry: Any


def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
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
    flat_parts = []
    for key in sorted(obs.keys()):
        val = obs[key]
        if isinstance(val, dict):
            flat_parts.append(flatten_obs(val))
        else:
            flat_parts.append(val.flatten())
    return jp.concatenate(flat_parts)


# =============================================================================
# Plotting
# =============================================================================


def plot_hh_diagnostics(layer_diag, env_step_idx, layer_names, save_path=None):
    """Plot V traces, gating variables, V heatmap, and rate histogram."""
    n_layers = len(layer_diag)
    fig, axes = plt.subplots(n_layers, 4, figsize=(18, 4 * n_layers),
                             squeeze=False)

    for row, name in enumerate(layer_names):
        diag = layer_diag[name]
        V_data = np.array(diag["V_trace"])
        m_data = np.array(diag["m_trace"])
        h_data = np.array(diag["h_trace"])
        n_data = np.array(diag["n_trace"])
        K, N = V_data.shape

        # Voltage traces
        ax = axes[row, 0]
        for j in np.linspace(0, N - 1, min(8, N), dtype=int):
            ax.plot(V_data[:, j], alpha=0.7, linewidth=1)
        ax.axhline(0.0, color="red", ls="--", alpha=0.5, label="V=0")
        ax.set_xlabel("micro-step")
        ax.set_ylabel("V (mV)")
        ax.set_title(f"{name} voltages (t={env_step_idx})")
        ax.legend(fontsize=7)

        # Gating variables (neuron 0)
        ax = axes[row, 1]
        ax.plot(m_data[:, 0], label="m (Na act.)", color="#1f77b4")
        ax.plot(h_data[:, 0], label="h (Na inact.)", color="#ff7f0e")
        ax.plot(n_data[:, 0], label="n (K act.)", color="#2ca02c")
        ax.set_xlabel("micro-step")
        ax.set_ylabel("gate value")
        ax.set_title(f"{name} gating (neuron 0)")
        ax.legend(fontsize=7)
        ax.set_ylim(-0.05, 1.05)

        # Voltage heatmap
        ax = axes[row, 2]
        ax.imshow(V_data.T, aspect='auto', cmap='RdBu_r',
                  interpolation='nearest', vmin=-80, vmax=40)
        ax.set_xlabel("micro-step")
        ax.set_ylabel("neuron")
        ax.set_title(f"{name} V heatmap")

        # Soft spike rate histogram
        ax = axes[row, 3]
        rates = np.mean(1.0 / (1.0 + np.exp(-0.5 * V_data)), axis=0)
        ax.hist(rates, bins=min(20, N), edgecolor="black", alpha=0.7)
        ax.axvline(np.mean(rates), color="red", ls="--",
                   label=f"mean={np.mean(rates):.3f}")
        ax.set_xlabel("soft spike rate")
        ax.set_ylabel("count")
        ax.set_title(f"{name} rate dist")
        ax.legend(fontsize=7)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_learned_conductances(policy_params, layer_names, save_path=None):
    """Visualize learned conductances across HH neurons."""
    fig, axes = plt.subplots(len(layer_names), 3,
                             figsize=(12, 3 * len(layer_names)),
                             squeeze=False)

    cond_names = ['g_Na', 'g_K', 'g_L']
    raw_names = ['g_Na_raw', 'g_K_raw', 'g_L_raw']
    defaults = [120.0, 36.0, 0.3]
    colors = ['#d62728', '#1f77b4', '#2ca02c']

    for row, name in enumerate(layer_names):
        layer_params = policy_params['params'][name]
        for col, (cname, rname, default_val, color) in enumerate(
            zip(cond_names, raw_names, defaults, colors)
        ):
            ax = axes[row, col]
            raw = np.array(layer_params[rname])
            actual = np.log1p(np.exp(raw))  # softplus
            ax.hist(actual, bins=min(15, len(raw)),
                    color=color, edgecolor='black', alpha=0.7)
            ax.axvline(default_val, color='black', ls='--',
                       label=f"default={default_val}")
            ax.axvline(np.mean(actual), color='red', ls='-',
                       label=f"mean={np.mean(actual):.2f}")
            ax.set_xlabel(f"{cname} (mS/cm²)")
            ax.set_ylabel("count")
            ax.set_title(f"{name} {cname}")
            ax.legend(fontsize=7)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


# =============================================================================
# Training Functions
# =============================================================================


def make_training_fns(train_env, policy_module, value_module, action_dist,
                      optimizer, config):
    """Create JIT-compiled training functions."""

    unroll_length = config.unroll_length
    num_minibatches = config.num_minibatches
    num_updates = config.num_updates_per_batch
    batch_size = config.batch_size
    gamma = config.discounting
    gae_lambda = config.gae_lambda
    reward_scaling = config.reward_scaling
    clip_eps = config.clip_eps
    vf_coef = config.vf_coef
    entropy_cost = config.entropy_cost
    obs_size = config.obs_size
    num_envs = config.num_envs

    def collect_rollout(state: TrainingState):
        def step_fn(carry, _):
            env_state, mem_carry, rng = carry
            rng, action_rng = jax.random.split(rng)

            obs_norm = running_statistics.normalize(
                env_state.obs, state.normalizer_params
            )
            logits, new_carry = policy_module.apply(
                state.policy_params, obs_norm, mem_carry
            )
            raw_action = action_dist.sample_no_postprocessing(logits, action_rng)
            log_prob = action_dist.log_prob(logits, raw_action)
            action = action_dist.postprocess(raw_action)
            value = jp.squeeze(
                value_module.apply(state.value_params, obs_norm), axis=-1
            )

            next_env_state = train_env.step(env_state, action)

            done_mask = next_env_state.done.reshape(-1, 1)
            new_carry = new_carry * (1.0 - done_mask)

            transition = Transition(
                obs=env_state.obs,
                action=action,
                raw_action=raw_action,
                log_prob=log_prob,
                value=value,
                reward=next_env_state.reward,
                done=next_env_state.done,
                carry=mem_carry,
            )
            return (next_env_state, new_carry, rng), transition

        (final_env_state, final_carry, rng), rollout = jax.lax.scan(
            step_fn,
            (state.env_state, state.membrane_carry, state.rng),
            None,
            length=unroll_length,
        )
        return final_env_state, final_carry, rng, rollout

    def sgd_step(carry, minibatch_idx, data, perm):
        policy_params, value_params, opt_state, normalizer_params, rng = carry
        rng, entropy_rng = jax.random.split(rng)

        idx = jax.lax.dynamic_slice(
            perm, (minibatch_idx * batch_size,), (batch_size,)
        )

        batch_obs = data['obs'][idx]
        batch_raw_action = data['raw_action'][idx]
        batch_log_prob = data['log_prob'][idx]
        batch_advantage = data['advantage'][idx]
        batch_target = data['target'][idx]
        batch_carry = data['carry'][idx]

        def loss_fn(params):
            pp, vp = params
            obs_norm = running_statistics.normalize(
                batch_obs, normalizer_params
            )

            logits, _ = policy_module.apply(pp, obs_norm, batch_carry)
            new_log_prob = action_dist.log_prob(logits, batch_raw_action)

            ratio = jp.exp(new_log_prob - batch_log_prob)
            adv = batch_advantage
            adv = (adv - jp.mean(adv)) / (jp.std(adv) + 1e-8)

            pg1 = -adv * ratio
            pg2 = -adv * jp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            policy_loss = jp.mean(jp.maximum(pg1, pg2))

            new_value = jp.squeeze(
                value_module.apply(vp, obs_norm), axis=-1
            )
            value_loss = jp.mean(jp.square(new_value - batch_target))

            entropy = jp.mean(action_dist.entropy(logits, entropy_rng))

            total = policy_loss + vf_coef * value_loss - entropy_cost * entropy
            return total, {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "approx_kl": jp.mean((ratio - 1.0) - jp.log(ratio)),
            }

        (_loss, metrics), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )((policy_params, value_params))
        updates, new_opt_state = optimizer.update(
            grads, opt_state, (policy_params, value_params)
        )
        new_pp, new_vp = optax.apply_updates(
            (policy_params, value_params), updates
        )
        return (new_pp, new_vp, new_opt_state, normalizer_params, rng), metrics

    def ppo_update_epoch(carry, _):
        policy_params, value_params, opt_state, normalizer_params, rng, data = carry
        rng, perm_rng = jax.random.split(rng)
        T_times_B = data['obs'].shape[0]
        perm = jax.random.permutation(perm_rng, T_times_B)

        (pp, vp, opt_state, normalizer_params, rng), metrics = jax.lax.scan(
            lambda c, i: sgd_step(c, i, data, perm),
            (policy_params, value_params, opt_state, normalizer_params, rng),
            jp.arange(num_minibatches),
        )
        return (pp, vp, opt_state, normalizer_params, rng, data), metrics

    def train_step(state: TrainingState) -> Tuple[TrainingState, dict]:
        final_env_state, final_carry, rng, rollout = collect_rollout(state)

        new_normalizer_params = running_statistics.update(
            state.normalizer_params, rollout.obs.reshape(-1, obs_size),
        )

        last_obs_norm = running_statistics.normalize(
            final_env_state.obs, new_normalizer_params
        )
        last_value = jp.squeeze(
            value_module.apply(state.value_params, last_obs_norm), axis=-1
        )
        rewards = rollout.reward * reward_scaling
        advantages, returns = compute_gae(
            rewards, rollout.value, rollout.done, last_value, gamma, gae_lambda,
        )

        T, B = rollout.obs.shape[:2]
        flat = lambda x: x.reshape(T * B, *x.shape[2:])

        data = {
            'obs': flat(rollout.obs),
            'raw_action': flat(rollout.raw_action),
            'log_prob': flat(rollout.log_prob),
            'advantage': flat(advantages),
            'target': flat(returns),
            'carry': flat(rollout.carry),
        }

        (pp, vp, opt_state, _, rng, _), all_metrics = jax.lax.scan(
            ppo_update_epoch,
            (state.policy_params, state.value_params, state.opt_state,
             new_normalizer_params, rng, data),
            None,
            length=num_updates,
        )

        metrics = jax.tree.map(lambda x: x[-1, -1], all_metrics)
        metrics['mean_reward'] = jp.mean(rollout.reward)

        new_state = TrainingState(
            policy_params=pp,
            value_params=vp,
            opt_state=opt_state,
            normalizer_params=new_normalizer_params,
            env_state=final_env_state,
            membrane_carry=final_carry,
            rng=rng,
            total_steps=state.total_steps + unroll_length * num_envs,
        )
        return new_state, metrics

    return jax.jit(train_step)


def make_eval_fn(eval_env, diag_policy_module, action_dist, episode_length,
                 carry_dim):
    """Create JIT-compiled evaluation function with diagnostics."""

    @jax.jit
    def eval_episode(policy_params, normalizer_params, rng):
        env_state = eval_env.reset(jax.random.split(rng, 1))
        carry = jp.zeros((1, carry_dim))

        def step_fn(carry_state, _):
            env_state, mem_carry, rng = carry_state
            rng, _ = jax.random.split(rng)

            obs = env_state.obs
            if obs.ndim == 1:
                obs = obs[None, :]
            obs_norm = running_statistics.normalize(obs, normalizer_params)

            logits, new_carry, diagnostics = diag_policy_module.apply(
                policy_params, obs_norm, mem_carry
            )

            action = action_dist.mode(logits)

            next_env_state = eval_env.step(env_state, action)
            new_carry = new_carry * (1.0 - next_env_state.done)

            step_data = {
                'reward': next_env_state.reward,
                'done': next_env_state.done,
            }

            for layer_name, diag in diagnostics.items():
                step_data[f'{layer_name}_V'] = diag['V_trace'][:, 0, :]
                step_data[f'{layer_name}_m'] = diag['m_trace'][:, 0, :]
                step_data[f'{layer_name}_h'] = diag['h_trace'][:, 0, :]
                step_data[f'{layer_name}_n'] = diag['n_trace'][:, 0, :]

            return (next_env_state, new_carry, rng), step_data

        _, episode_data = jax.lax.scan(
            step_fn, (env_state, carry, rng), None, length=episode_length
        )
        return episode_data

    return eval_episode


# =============================================================================
# Config
# =============================================================================


ppo_params = config_dict.create(
    num_timesteps=500_000_000,
    num_evals=10,
    reward_scaling=1.0,
    episode_length=80,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=10,
    num_minibatches=32,
    num_updates_per_batch=4,
    discounting=0.97,
    learning_rate=3e-4,
    entropy_cost=1e-3,
    num_envs=4096,
    batch_size=256,
    max_grad_norm=1.0,
    gae_lambda=0.95,
    clip_eps=0.3,
    vf_coef=0.5,
    network_factory=config_dict.create(
        policy_hidden_layer_sizes=(20, 20),
        n_steps=100,
        dt=0.05,
        current_scale=20.0,
        spike_beta=0.5,
        value_hidden_layer_sizes=(256, 256, 256),
    ),
)

env_name = "mouse-hh"
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
               id=f"hh-{exp_name}")
    wandb.config.update({
        "env_name": env_name,
        "policy_type": "hodgkin_huxley",
        **dict(ppo_params.network_factory),
    })


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    print("=" * 80)
    print("Mouse Arm Imitation -- Hodgkin-Huxley Policy")
    print("=" * 80)
    nf = ppo_params.network_factory
    layer_sizes = tuple(nf.policy_hidden_layer_sizes)
    print(f"HH layers: {layer_sizes}")
    print(f"n_steps={nf.n_steps}, dt={nf.dt}ms "
          f"({nf.n_steps * nf.dt:.1f}ms biological time per env step)")
    print(f"current_scale={nf.current_scale}, spike_beta={nf.spike_beta}")

    layer_names = [f"hh_{i}" for i in range(len(layer_sizes))]

    # Environment setup
    print(f"Loading reference clips from {consts.MOUSE_REFERENCE_DATA_PATH}...")
    reference_clips = MouseReferenceClips(
        str(consts.MOUSE_REFERENCE_DATA_PATH),
        n_frames_per_clip=env_cfg.clip_length,
    )
    train_clips, test_clips = reference_clips.split(train_ratio=0.8, seed=42)

    env = MouseImitation(config=env_cfg, clips=train_clips)
    eval_env_base = MouseImitation(config=env_cfg, clips=test_clips)
    print(f"Action size: {env.action_size}  Obs size: {env.observation_size}")

    steps_per_frame = (1 / env_cfg.mocap_hz) / env_cfg.ctrl_dt
    episode_length = int(
        (env_cfg.clip_length - env_cfg.start_frame_range[-1]
         - env_cfg.reference_length) * steps_per_frame
    )
    print(f"Episode length: {episode_length}")

    obs_size = env.observation_size
    if isinstance(obs_size, (tuple, list)):
        obs_size = obs_size[-1]
    act_size = env.action_size

    # Wrap environments
    def flatten_obs_wrapper(env_fn):
        class W:
            def __init__(self, e): self._e = e
            def reset(self, rng):
                s = self._e.reset(rng)
                return s.replace(obs=flatten_obs(s.obs))
            def step(self, state, action):
                s = self._e.step(state, action)
                return s.replace(obs=flatten_obs(s.obs))
            @property
            def observation_size(self): return self._e.observation_size
            @property
            def action_size(self): return self._e.action_size
            @property
            def dt(self): return self._e.dt
            def __getattr__(self, name): return getattr(self._e, name)
        return W(env_fn)

    wrapped_env = flatten_obs_wrapper(env)
    wrapped_eval_env = flatten_obs_wrapper(eval_env_base)

    wrap_fn = functools.partial(wrapper.wrap_for_brax_training, full_reset=True)
    train_env = wrap_fn(wrapped_env, episode_length=episode_length,
                        action_repeat=ppo_params.action_repeat)
    eval_env = wrap_fn(wrapped_eval_env, episode_length=episode_length,
                       action_repeat=ppo_params.action_repeat)

    # Networks
    action_dist = distribution.NormalTanhDistribution(event_size=act_size)
    param_size = action_dist.param_size

    policy_module = RecurrentHHPolicy(
        layer_sizes=layer_sizes,
        output_size=param_size,
        n_steps=nf.n_steps,
        dt=nf.dt,
        current_scale=nf.current_scale,
        spike_beta=nf.spike_beta,
    )
    diag_policy_module = DiagnosticHHPolicy(
        layer_sizes=layer_sizes,
        output_size=param_size,
        n_steps=nf.n_steps,
        dt=nf.dt,
        current_scale=nf.current_scale,
        spike_beta=nf.spike_beta,
    )
    value_module = networks.MLP(
        layer_sizes=list(nf.value_hidden_layer_sizes) + [1],
        activation=linen.swish,
        kernel_init=jax.nn.initializers.lecun_uniform(),
    )

    # Carry dimension: 4 state vars per neuron (V, m, h, n)
    carry_dim = 4 * sum(layer_sizes)
    print(f"Carry dim: {carry_dim}")

    num_envs = ppo_params.num_envs
    dummy_obs = jp.zeros((1, obs_size))
    dummy_carry = jp.zeros((1, carry_dim))

    key = jax.random.PRNGKey(0)
    key, pk, vk, ek = jax.random.split(key, 4)
    policy_params = policy_module.init(pk, dummy_obs, dummy_carry)
    value_params = value_module.init(vk, dummy_obs)
    normalizer_params = running_statistics.init_state(jp.zeros(obs_size))

    optimizer = optax.chain(
        optax.clip_by_global_norm(ppo_params.max_grad_norm),
        optax.adam(ppo_params.learning_rate),
    )
    opt_state = optimizer.init((policy_params, value_params))

    # Create training and eval functions
    train_config = config_dict.create(
        unroll_length=ppo_params.unroll_length,
        num_minibatches=ppo_params.num_minibatches,
        num_updates_per_batch=ppo_params.num_updates_per_batch,
        batch_size=ppo_params.batch_size,
        discounting=ppo_params.discounting,
        gae_lambda=ppo_params.gae_lambda,
        reward_scaling=ppo_params.reward_scaling,
        clip_eps=ppo_params.clip_eps,
        vf_coef=ppo_params.vf_coef,
        entropy_cost=ppo_params.entropy_cost,
        obs_size=obs_size,
        num_envs=num_envs,
    )

    train_step_fn = make_training_fns(
        train_env, policy_module, value_module, action_dist, optimizer,
        train_config
    )
    eval_fn = make_eval_fn(
        eval_env, diag_policy_module, action_dist, episode_length, carry_dim
    )

    # JIT'd helpers for video rollout on raw eval_env_base (single-env, unbatched)
    jit_vid_reset = jax.jit(eval_env_base.reset)
    jit_vid_step = jax.jit(eval_env_base.step)
    jit_vid_policy = jax.jit(policy_module.apply)

    def video_rollout(policy_params, normalizer_params, seed=0):
        """Fast video rollout: Python loop but all ops are pre-JIT'd."""
        rng = jax.random.PRNGKey(seed)
        state = jit_vid_reset(rng)
        carry = jp.zeros((1, carry_dim))
        rollout_states = [state]
        for _ in range(episode_length):
            flat_obs = flatten_obs(state.obs)
            obs_norm = running_statistics.normalize(
                flat_obs[None], normalizer_params
            )
            logits, new_carry = jit_vid_policy(
                policy_params, obs_norm, carry
            )
            action = jp.squeeze(action_dist.mode(logits), axis=0)
            state = jit_vid_step(state, action)
            carry = new_carry * (1.0 - state.done.reshape(1, 1))
            rollout_states.append(state)
        return rollout_states

    # Initialize training state
    print("Initializing training env state...")
    env_state = train_env.reset(jax.random.split(ek, num_envs))
    membrane_carry = jp.zeros((num_envs, carry_dim))

    state = TrainingState(
        policy_params=policy_params,
        value_params=value_params,
        opt_state=opt_state,
        normalizer_params=normalizer_params,
        env_state=env_state,
        membrane_carry=membrane_carry,
        rng=key,
        total_steps=0,
    )

    # Training loop
    steps_per_unroll = ppo_params.unroll_length * num_envs
    num_timesteps = ppo_params.num_timesteps
    num_evals = ppo_params.num_evals
    steps_per_eval = num_timesteps // num_evals
    next_eval_at = steps_per_eval

    print(f"Training for {num_timesteps:,} steps")
    print("=" * 80)

    # Warmup
    print("Warming up JIT compilation...")
    state, metrics = train_step_fn(state)
    state = state.replace(total_steps=0)
    print("JIT warmup complete.")

    import time as _time
    t0 = _time.time()

    while state.total_steps < num_timesteps:
        state, metrics = train_step_fn(state)

        if state.total_steps >= next_eval_at or state.total_steps >= num_timesteps:
            elapsed = _time.time() - t0
            sps = state.total_steps / max(elapsed, 1e-6)

            log_metrics = {k: float(v) for k, v in metrics.items()}
            log_metrics["sps"] = sps
            log_metrics["total_steps"] = int(state.total_steps)

            pprint(f"Step {state.total_steps:>12,}  SPS {sps:,.0f}")
            pprint(log_metrics)

            if USE_WANDB:
                wandb.log(log_metrics, step=int(state.total_steps))

            # Evaluation with diagnostics
            print(f"[eval] Running evaluation at step {state.total_steps}...")
            key, eval_key = jax.random.split(state.rng)
            state = state.replace(rng=key)

            episode_data = eval_fn(
                state.policy_params, state.normalizer_params, eval_key
            )

            # Plot diagnostics at 3 time points
            for t_idx in [0, episode_length // 2, episode_length - 1]:
                diag_at_t = {}
                for name in layer_names:
                    diag_at_t[name] = {
                        'V_trace': np.array(
                            episode_data[f'{name}_V'][t_idx]),
                        'm_trace': np.array(
                            episode_data[f'{name}_m'][t_idx]),
                        'h_trace': np.array(
                            episode_data[f'{name}_h'][t_idx]),
                        'n_trace': np.array(
                            episode_data[f'{name}_n'][t_idx]),
                    }

                fig = plot_hh_diagnostics(
                    diag_at_t, t_idx, layer_names,
                    save_path=f"{ckpt_path}/{state.total_steps}_hh_t{t_idx}.png",
                )
                if USE_WANDB:
                    wandb.log({f"eval/hh_t{t_idx}": wandb.Image(fig)},
                              commit=False)
                plt.close(fig)

            # Learned conductances
            fig = plot_learned_conductances(
                state.policy_params, layer_names,
                save_path=f"{ckpt_path}/{state.total_steps}_conductances.png",
            )
            if USE_WANDB:
                wandb.log({"eval/conductances": wandb.Image(fig)},
                          commit=False)
            plt.close(fig)

            # Eval reward summary
            rewards = np.array(episode_data['reward'])
            eval_reward = float(np.sum(rewards))
            print(f"  eval reward: {eval_reward:.1f}")
            if USE_WANDB:
                wandb.log({"eval/episode_reward": eval_reward}, commit=False)

            # Log mean firing rates
            for name in layer_names:
                V_all = np.array(episode_data[f'{name}_V'])
                rates = np.mean(
                    1.0 / (1.0 + np.exp(-0.5 * V_all)), axis=(0, 1)
                )
                mean_rate = float(np.mean(rates))
                print(f"  {name} mean rate: {mean_rate:.4f}")
                if USE_WANDB:
                    wandb.log({f"eval/{name}_rate": mean_rate}, commit=False)

            # Video rollout
            try:
                states = video_rollout(
                    state.policy_params, state.normalizer_params,
                    seed=state.total_steps,
                )
                fps = int(1.0 / eval_env_base.dt)
                frames = eval_env_base.render(
                    states, height=512, width=512, render_ghost=True
                )
                video_path = f"{ckpt_path}/{state.total_steps}.mp4"
                with imageio.get_writer(video_path, fps=fps) as vid:
                    for f in frames:
                        vid.append_data(f)
                if USE_WANDB:
                    wandb.log(
                        {"eval/rollout": wandb.Video(video_path, format="mp4")},
                        commit=False,
                    )
                print(f"  video -> {video_path}")
            except Exception as e:
                print(f"  video failed: {e}")

            # Checkpoint
            params_to_save = (state.normalizer_params, state.policy_params,
                              state.value_params)
            orbax_checkpointer = ocp.PyTreeCheckpointer()
            save_args = orbax_utils.save_args_from_target(params_to_save)
            path = ckpt_path / f"{state.total_steps}"
            orbax_checkpointer.save(path, params_to_save, force=True,
                                    save_args=save_args)
            print(f"  checkpoint -> {path}")

            next_eval_at += steps_per_eval

    print("=" * 80)
    print("Training complete!")
    print("=" * 80)
