"""
Training script for mouse arm imitation with ProprioSpinal + Motor Neuron circuit.

Architecture:
  - ProprioSpinal (C3-C4 PNs): E/I LIF layer with Dale's law, refractory period,
    heterogeneous tau. Recurrent E↔I lateral connections.
  - Motor Neuron Pool: Excitatory-only LIF layer. Receives direct projections from
    both E and I propriospinal populations with Dale's law:
      E PNs → MN: excitatory (|W|, positive drive)
      I PNs → MN: inhibitory (-|W|, negative drive)
  - Muscle readout: Dense layer from motor neuron spike rates → action logits.
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
# Module 1: ProprioSpinal (C3-C4 Propriospinal Neurons)
# =============================================================================


class ProprioSpinalModule(linen.Module):
    """C3-C4 propriospinal neurons with E/I populations.

    Recurrent lateral connections with Dale's law:
      E → I: excitatory (|W_ei|)
      I → E: inhibitory (-|W_ie|)

    Output: full population spike rates (E + I concatenated).
    Both E and I spike rates are sent as direct projections to motor neurons.
    """

    n_exc: int
    n_inh: int
    n_micro_steps: int = 8
    tau_min: float = 1.0
    tau_max: float = 5.0
    v_th: float = 0.3
    v_reset: float = 0.0
    beta_surrogate: float = 5.0
    n_refractory: float = 2.0
    dt: float = 1.0

    @linen.compact
    def __call__(self, x, v_carry, refrac_carry):
        n_exc, n_inh = self.n_exc, self.n_inh
        n_total = n_exc + n_inh
        v_th = self.v_th
        v_reset = self.v_reset
        beta = self.beta_surrogate
        n_refrac = self.n_refractory

        I_input = linen.Dense(
            n_total,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            name="input_proj",
        )(x)

        tau_m = self.param(
            'tau_m',
            lambda key, shape: jp.exp(jax.random.uniform(
                key, shape,
                minval=jp.log(self.tau_min), maxval=jp.log(self.tau_max),
            )),
            (n_total,),
        )
        tau_m = jax.lax.stop_gradient(tau_m)
        alpha = jp.exp(-self.dt / tau_m)

        W_ie = self.param('W_ie', jax.nn.initializers.lecun_uniform(),
                          (n_inh, n_exc))
        W_ei = self.param('W_ei', jax.nn.initializers.lecun_uniform(),
                          (n_exc, n_inh))

        prev_spike = jp.zeros_like(v_carry)

        def lif_step(carry, _):
            v, refrac, prev_sp = carry
            sp_e = prev_sp[:, :n_exc]
            sp_i = prev_sp[:, n_exc:]
            I_lat_e = -sp_i @ jp.abs(W_ie)
            I_lat_i = sp_e @ jp.abs(W_ei)
            I_lateral = jp.concatenate([I_lat_e, I_lat_i], axis=-1)
            I_total = I_input + I_lateral

            is_refractory = (refrac > 0.0).astype(v.dtype)
            v = alpha * v + (1.0 - alpha) * I_total
            v = v * (1.0 - is_refractory)

            spike_hard = (v >= v_th).astype(v.dtype)
            spike_soft = jax.nn.sigmoid(beta * (v - v_th))
            spike = jax.lax.stop_gradient(spike_hard - spike_soft) + spike_soft
            spike = spike * (1.0 - is_refractory)

            v_new = v * (1.0 - spike) + v_reset * spike
            new_refrac = jp.where(
                spike > 0.5, n_refrac, jp.maximum(refrac - 1.0, 0.0)
            )
            return (v_new, new_refrac, spike), spike

        (v_final, refrac_final, _), all_spikes = jax.lax.scan(
            lif_step, (v_carry, refrac_carry, prev_spike),
            None, length=self.n_micro_steps,
        )
        spike_rate = jp.mean(all_spikes, axis=0)
        return spike_rate, v_final, refrac_final


# =============================================================================
# Module 2: Motor Neurons (Excitatory-only LIF)
# =============================================================================


class MotorNeuronModule(linen.Module):
    """Motor neuron pool -- excitatory-only LIF.

    Receives pre-computed input current (from Dale's law E/I projections
    in the policy), runs LIF dynamics, outputs spike rates.
    """

    n_neurons: int
    n_micro_steps: int = 8
    tau_min: float = 1.0
    tau_max: float = 5.0
    v_th: float = 0.3
    v_reset: float = 0.0
    beta_surrogate: float = 5.0
    n_refractory: float = 2.0
    dt: float = 1.0

    @linen.compact
    def __call__(self, I_input, v_carry, refrac_carry):
        """
        Args:
            I_input: (B, n_neurons) pre-computed input current from E/I projections
            v_carry: (B, n_neurons) membrane voltage carry
            refrac_carry: (B, n_neurons) refractory counter carry
        """
        n = self.n_neurons
        v_th = self.v_th
        v_reset = self.v_reset
        beta = self.beta_surrogate
        n_refrac = self.n_refractory

        tau_m = self.param(
            'tau_m',
            lambda key, shape: jp.exp(jax.random.uniform(
                key, shape,
                minval=jp.log(self.tau_min), maxval=jp.log(self.tau_max),
            )),
            (n,),
        )
        tau_m = jax.lax.stop_gradient(tau_m)
        alpha = jp.exp(-self.dt / tau_m)

        def lif_step(carry, _):
            v, refrac = carry
            is_refractory = (refrac > 0.0).astype(v.dtype)
            v = alpha * v + (1.0 - alpha) * I_input
            v = v * (1.0 - is_refractory)

            spike_hard = (v >= v_th).astype(v.dtype)
            spike_soft = jax.nn.sigmoid(beta * (v - v_th))
            spike = jax.lax.stop_gradient(spike_hard - spike_soft) + spike_soft
            spike = spike * (1.0 - is_refractory)

            v_new = v * (1.0 - spike) + v_reset * spike
            new_refrac = jp.where(
                spike > 0.5, n_refrac, jp.maximum(refrac - 1.0, 0.0)
            )
            return (v_new, new_refrac), spike

        (v_final, refrac_final), all_spikes = jax.lax.scan(
            lif_step, (v_carry, refrac_carry),
            None, length=self.n_micro_steps,
        )
        spike_rate = jp.mean(all_spikes, axis=0)
        return spike_rate, v_final, refrac_final


# =============================================================================
# Combined Policy: ProprioSpinal -> Motor Neurons
# =============================================================================


class ProprioSpinalMNPolicy(linen.Module):
    """ProprioSpinal E/I -> direct Dale's law projections -> Motor Neurons.

    Signal flow:
      obs -> ProprioSpinal E/I LIF (with recurrent E↔I) -> spike rates
      E spike rates --(+|W_exc|)--> MN input current (excitatory drive)
      I spike rates --(-|W_inh|)--> MN input current (inhibitory drive)
      MN input current -> Motor Neuron LIF -> spike rates -> Dense -> muscle logits

    Carry layout:
      [ps_v | mn_v | ps_r | mn_r]
       (ps)   (mn)   (ps)   (mn)
    """

    propriospinal_size: int = 512
    exc_ratio: float = 0.8
    motor_neuron_size: int = 128
    n_micro_steps: int = 8
    tau_min: float = 1.0
    tau_max: float = 5.0
    v_th: float = 0.3
    v_reset: float = 0.0
    beta_surrogate: float = 5.0
    n_refractory: float = 2.0
    output_size: int = 18

    @linen.compact
    def __call__(self, obs, carry_flat):
        n_ps = self.propriospinal_size
        n_mn = self.motor_neuron_size
        n_exc = round(n_ps * self.exc_ratio)
        n_inh = n_ps - n_exc

        # Split carry
        idx = 0
        ps_v = carry_flat[:, idx:idx + n_ps]; idx += n_ps
        mn_v = carry_flat[:, idx:idx + n_mn]; idx += n_mn
        ps_r = carry_flat[:, idx:idx + n_ps]; idx += n_ps
        mn_r = carry_flat[:, idx:idx + n_mn]

        # 1. ProprioSpinal: obs -> E/I LIF -> spike rates
        spike_rate, ps_v_new, ps_r_new = ProprioSpinalModule(
            n_exc=n_exc, n_inh=n_inh,
            n_micro_steps=self.n_micro_steps,
            tau_min=self.tau_min, tau_max=self.tau_max,
            v_th=self.v_th, v_reset=self.v_reset,
            beta_surrogate=self.beta_surrogate,
            n_refractory=self.n_refractory,
            name="propriospinal",
        )(obs, ps_v, ps_r)

        spike_rate_exc = spike_rate[:, :n_exc]
        spike_rate_inh = spike_rate[:, n_exc:]

        # 2. Direct E/I projections to motor neurons (Dale's law)
        W_exc_mn = self.param(
            'W_exc_mn', jax.nn.initializers.lecun_uniform(),
            (n_mn, n_exc),
        )
        W_inh_mn = self.param(
            'W_inh_mn', jax.nn.initializers.lecun_uniform(),
            (n_mn, n_inh),
        )
        # E -> MN: excitatory (positive)
        I_exc = spike_rate_exc @ jp.abs(W_exc_mn).T
        # I -> MN: inhibitory (negative)
        I_inh = -spike_rate_inh @ jp.abs(W_inh_mn).T
        mn_input = I_exc + I_inh

        # 3. Motor neurons: input current -> LIF -> spike rates
        motor_spike_rate, mn_v_new, mn_r_new = MotorNeuronModule(
            n_neurons=n_mn,
            n_micro_steps=self.n_micro_steps,
            tau_min=self.tau_min, tau_max=self.tau_max,
            v_th=self.v_th, v_reset=self.v_reset,
            beta_surrogate=self.beta_surrogate,
            n_refractory=self.n_refractory,
            name="motor_neurons",
        )(mn_input, mn_v, mn_r)

        # 4. Muscle readout from motor neuron spike rates
        logits = linen.Dense(
            self.output_size,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            name="muscle_readout",
        )(motor_spike_rate)

        # 5. Reassemble carry
        new_carry = jp.concatenate([
            ps_v_new, mn_v_new, ps_r_new, mn_r_new,
        ], axis=-1)

        return logits, new_carry


# =============================================================================
# Diagnostic Variants
# =============================================================================


class DiagnosticProprioSpinalModule(linen.Module):
    """ProprioSpinal with full E/I spike and voltage traces."""

    n_exc: int
    n_inh: int
    n_micro_steps: int = 8
    tau_min: float = 1.0
    tau_max: float = 5.0
    v_th: float = 0.3
    v_reset: float = 0.0
    beta_surrogate: float = 5.0
    n_refractory: float = 2.0
    dt: float = 1.0

    @linen.compact
    def __call__(self, x, v_carry, refrac_carry):
        n_exc, n_inh = self.n_exc, self.n_inh
        n_total = n_exc + n_inh
        v_th, v_reset = self.v_th, self.v_reset
        n_refrac = self.n_refractory

        I_input = linen.Dense(
            n_total,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            name="input_proj",
        )(x)

        tau_m = self.param(
            'tau_m',
            lambda key, shape: jp.exp(jax.random.uniform(
                key, shape,
                minval=jp.log(self.tau_min), maxval=jp.log(self.tau_max),
            )),
            (n_total,),
        )
        tau_m = jax.lax.stop_gradient(tau_m)
        alpha = jp.exp(-self.dt / tau_m)

        W_ie = self.param('W_ie', jax.nn.initializers.lecun_uniform(),
                          (n_inh, n_exc))
        W_ei = self.param('W_ei', jax.nn.initializers.lecun_uniform(),
                          (n_exc, n_inh))

        prev_spike = jp.zeros_like(v_carry)

        def lif_step(carry, _):
            v, refrac, prev_sp = carry
            sp_e = prev_sp[:, :n_exc]
            sp_i = prev_sp[:, n_exc:]
            I_lat_e = -sp_i @ jp.abs(W_ie)
            I_lat_i = sp_e @ jp.abs(W_ei)
            I_lateral = jp.concatenate([I_lat_e, I_lat_i], axis=-1)
            I_total = I_input + I_lateral

            is_refractory = (refrac > 0.0).astype(v.dtype)
            v_pre = alpha * v + (1.0 - alpha) * I_total
            v_pre = v_pre * (1.0 - is_refractory)

            spike = (v_pre >= v_th).astype(v_pre.dtype)
            spike = spike * (1.0 - is_refractory)

            v_new = v_pre * (1.0 - spike) + v_reset * spike
            new_refrac = jp.where(
                spike > 0.5, n_refrac, jp.maximum(refrac - 1.0, 0.0)
            )
            return (v_new, new_refrac, spike), (spike, v_pre)

        (v_final, refrac_final, _), (all_spikes, all_voltages) = jax.lax.scan(
            lif_step, (v_carry, refrac_carry, prev_spike),
            None, length=self.n_micro_steps,
        )
        spike_rate = jp.mean(all_spikes, axis=0)
        return spike_rate, v_final, refrac_final, {
            "spikes_exc": all_spikes[:, :, :n_exc],
            "spikes_inh": all_spikes[:, :, n_exc:],
            "voltages_exc": all_voltages[:, :, :n_exc],
            "voltages_inh": all_voltages[:, :, n_exc:],
        }


class DiagnosticMotorNeuronModule(linen.Module):
    """Motor neuron pool with full spike and voltage traces."""

    n_neurons: int
    n_micro_steps: int = 8
    tau_min: float = 1.0
    tau_max: float = 5.0
    v_th: float = 0.3
    v_reset: float = 0.0
    beta_surrogate: float = 5.0
    n_refractory: float = 2.0
    dt: float = 1.0

    @linen.compact
    def __call__(self, I_input, v_carry, refrac_carry):
        n = self.n_neurons
        v_th, v_reset = self.v_th, self.v_reset
        n_refrac = self.n_refractory

        tau_m = self.param(
            'tau_m',
            lambda key, shape: jp.exp(jax.random.uniform(
                key, shape,
                minval=jp.log(self.tau_min), maxval=jp.log(self.tau_max),
            )),
            (n,),
        )
        tau_m = jax.lax.stop_gradient(tau_m)
        alpha = jp.exp(-self.dt / tau_m)

        def lif_step(carry, _):
            v, refrac = carry
            is_refractory = (refrac > 0.0).astype(v.dtype)
            v_pre = alpha * v + (1.0 - alpha) * I_input
            v_pre = v_pre * (1.0 - is_refractory)

            spike = (v_pre >= v_th).astype(v_pre.dtype)
            spike = spike * (1.0 - is_refractory)

            v_new = v_pre * (1.0 - spike) + v_reset * spike
            new_refrac = jp.where(
                spike > 0.5, n_refrac, jp.maximum(refrac - 1.0, 0.0)
            )
            return (v_new, new_refrac), (spike, v_pre)

        (v_final, refrac_final), (all_spikes, all_voltages) = jax.lax.scan(
            lif_step, (v_carry, refrac_carry),
            None, length=self.n_micro_steps,
        )
        spike_rate = jp.mean(all_spikes, axis=0)
        return spike_rate, v_final, refrac_final, {
            "spikes": all_spikes,
            "voltages": all_voltages,
        }


class DiagnosticProprioSpinalMNPolicy(linen.Module):
    """Combined policy with full diagnostic output."""

    propriospinal_size: int = 512
    exc_ratio: float = 0.8
    motor_neuron_size: int = 128
    n_micro_steps: int = 8
    tau_min: float = 1.0
    tau_max: float = 5.0
    v_th: float = 0.3
    v_reset: float = 0.0
    beta_surrogate: float = 5.0
    n_refractory: float = 2.0
    output_size: int = 18

    @linen.compact
    def __call__(self, obs, carry_flat):
        n_ps = self.propriospinal_size
        n_mn = self.motor_neuron_size
        n_exc = round(n_ps * self.exc_ratio)
        n_inh = n_ps - n_exc

        idx = 0
        ps_v = carry_flat[:, idx:idx + n_ps]; idx += n_ps
        mn_v = carry_flat[:, idx:idx + n_mn]; idx += n_mn
        ps_r = carry_flat[:, idx:idx + n_ps]; idx += n_ps
        mn_r = carry_flat[:, idx:idx + n_mn]

        spike_rate, ps_v_new, ps_r_new, ps_diag = DiagnosticProprioSpinalModule(
            n_exc=n_exc, n_inh=n_inh,
            n_micro_steps=self.n_micro_steps,
            tau_min=self.tau_min, tau_max=self.tau_max,
            v_th=self.v_th, v_reset=self.v_reset,
            beta_surrogate=self.beta_surrogate,
            n_refractory=self.n_refractory,
            name="propriospinal",
        )(obs, ps_v, ps_r)

        spike_rate_exc = spike_rate[:, :n_exc]
        spike_rate_inh = spike_rate[:, n_exc:]

        # Direct E/I projections to motor neurons (Dale's law)
        W_exc_mn = self.param(
            'W_exc_mn', jax.nn.initializers.lecun_uniform(),
            (n_mn, n_exc),
        )
        W_inh_mn = self.param(
            'W_inh_mn', jax.nn.initializers.lecun_uniform(),
            (n_mn, n_inh),
        )
        I_exc = spike_rate_exc @ jp.abs(W_exc_mn).T
        I_inh = -spike_rate_inh @ jp.abs(W_inh_mn).T
        mn_input = I_exc + I_inh

        motor_spike_rate, mn_v_new, mn_r_new, mn_diag = DiagnosticMotorNeuronModule(
            n_neurons=n_mn,
            n_micro_steps=self.n_micro_steps,
            tau_min=self.tau_min, tau_max=self.tau_max,
            v_th=self.v_th, v_reset=self.v_reset,
            beta_surrogate=self.beta_surrogate,
            n_refractory=self.n_refractory,
            name="motor_neurons",
        )(mn_input, mn_v, mn_r)

        logits = linen.Dense(
            self.output_size,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            name="muscle_readout",
        )(motor_spike_rate)

        new_carry = jp.concatenate([
            ps_v_new, mn_v_new, ps_r_new, mn_r_new,
        ], axis=-1)

        diagnostics = {
            "propriospinal": ps_diag,
            "motor_neurons": mn_diag,
            "motor_spike_rate": motor_spike_rate,
            "mn_input_exc": I_exc,
            "mn_input_inh": I_inh,
            "mn_input_total": mn_input,
            "spike_rate_exc": spike_rate_exc,
            "spike_rate_inh": spike_rate_inh,
        }

        return logits, new_carry, diagnostics


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

# Seaborn tab10 E/I colour scheme
C_EXC = '#1f77b4'   # blue  (excitatory)
C_INH = '#ff7f0e'   # orange (inhibitory)

from matplotlib.colors import LinearSegmentedColormap
CMAP_EXC = LinearSegmentedColormap.from_list('exc', ['#ffffff', C_EXC])
CMAP_INH = LinearSegmentedColormap.from_list('inh', ['#ffffff', C_INH])


def plot_circuit_diagnostics(episode_data, v_th, save_path=None):
    """Plot overview of the ProprioSpinal + Motor Neuron circuit activity."""
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))

    # --- Row 0: ProprioSpinal ---
    # E raster
    ax = axes[0, 0]
    spikes_e = np.array(episode_data['ps_spikes_exc'])  # (T, K, N_exc)
    T_ep = spikes_e.shape[0]
    rates_e = np.mean(spikes_e, axis=1)  # (T, N_exc)
    n_show = min(64, rates_e.shape[1])
    ax.imshow(rates_e[:, :n_show].T, aspect='auto', cmap=CMAP_EXC,
              interpolation='nearest')
    ax.set_xlabel("env step")
    ax.set_ylabel("E neuron")
    ax.set_title("ProprioSpinal E activity", color=C_EXC)

    # I raster
    ax = axes[0, 1]
    spikes_i = np.array(episode_data['ps_spikes_inh'])
    rates_i = np.mean(spikes_i, axis=1)
    n_show_i = min(32, rates_i.shape[1])
    ax.imshow(rates_i[:, :n_show_i].T, aspect='auto', cmap=CMAP_INH,
              interpolation='nearest')
    ax.set_xlabel("env step")
    ax.set_ylabel("I neuron")
    ax.set_title("ProprioSpinal I activity", color=C_INH)

    # E rate histogram
    ax = axes[0, 2]
    mean_rates_e = np.mean(rates_e, axis=0)
    ax.hist(mean_rates_e, bins=30, color=C_EXC, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(mean_rates_e), color='black', ls='--',
               label=f"mean={np.mean(mean_rates_e):.3f}")
    ax.set_xlabel("mean spike rate")
    ax.set_ylabel("count")
    ax.set_title("PN E rate distribution", color=C_EXC)
    ax.legend(fontsize=7)

    # I rate histogram
    ax = axes[0, 3]
    mean_rates_i = np.mean(rates_i, axis=0)
    ax.hist(mean_rates_i, bins=30, color=C_INH, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(mean_rates_i), color='black', ls='--',
               label=f"mean={np.mean(mean_rates_i):.3f}")
    ax.set_xlabel("mean spike rate")
    ax.set_ylabel("count")
    ax.set_title("PN I rate distribution", color=C_INH)
    ax.legend(fontsize=7)

    # --- Row 1: Motor Neurons ---
    ax = axes[1, 0]
    mn_spikes = np.array(episode_data['mn_spikes'])  # (T, K, N_mn)
    mn_rates = np.mean(mn_spikes, axis=1)
    n_show_mn = min(64, mn_rates.shape[1])
    ax.imshow(mn_rates[:, :n_show_mn].T, aspect='auto', cmap=CMAP_EXC,
              interpolation='nearest')
    ax.set_xlabel("env step")
    ax.set_ylabel("MN neuron")
    ax.set_title("Motor Neuron activity", color=C_EXC)

    # MN rate histogram
    ax = axes[1, 1]
    mean_mn = np.mean(mn_rates, axis=0)
    ax.hist(mean_mn, bins=30, color=C_EXC, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(mean_mn), color='black', ls='--',
               label=f"mean={np.mean(mean_mn):.3f}")
    ax.set_xlabel("mean spike rate")
    ax.set_ylabel("count")
    ax.set_title("MN rate distribution", color=C_EXC)
    ax.legend(fontsize=7)

    # MN voltage traces
    ax = axes[1, 2]
    mn_volts = np.array(episode_data['mn_voltages'])  # (T, K, N_mn)
    mid = T_ep // 2
    for j in range(min(8, mn_volts.shape[2])):
        ax.plot(mn_volts[mid, :, j], alpha=0.6, linewidth=1, color=C_EXC)
    ax.axhline(v_th, color='black', ls='--', alpha=0.5, label='threshold')
    ax.set_xlabel("micro-step")
    ax.set_ylabel("voltage")
    ax.set_title(f"MN voltages (env step {mid})")
    ax.legend(fontsize=7)

    # E->MN vs I->MN input currents over time
    ax = axes[1, 3]
    mn_exc = np.array(episode_data['mn_input_exc'])  # (T, B, N_mn) or (T, N_mn)
    mn_inh = np.array(episode_data['mn_input_inh'])
    if mn_exc.ndim == 3:
        mn_exc = mn_exc[:, 0, :]
        mn_inh = mn_inh[:, 0, :]
    ax.plot(np.mean(mn_exc, axis=1), color=C_EXC, linewidth=1, label='E→MN (mean)')
    ax.plot(np.mean(mn_inh, axis=1), color=C_INH, linewidth=1, label='I→MN (mean)')
    ax.axhline(0, color='gray', ls='--', alpha=0.3)
    ax.set_xlabel("env step")
    ax.set_ylabel("current")
    ax.set_title("E/I → MN drive")
    ax.legend(fontsize=7)

    # --- Row 2: E/I balance and reward ---
    # E/I balance: mean E drive - mean |I drive| over time
    ax = axes[2, 0]
    balance = np.mean(mn_exc, axis=1) + np.mean(mn_inh, axis=1)  # I is already negative
    ax.plot(balance, color='purple', linewidth=1)
    ax.axhline(0, color='gray', ls='--', alpha=0.3)
    ax.set_xlabel("env step")
    ax.set_ylabel("E + I (net)")
    ax.set_title("E/I balance at MN")

    # PN E→MN correlation
    ax = axes[2, 1]
    ax.scatter(np.mean(rates_e, axis=1), np.mean(mn_rates, axis=1),
               s=3, alpha=0.5, color=C_EXC, label='E→MN')
    ax.set_xlabel("PN E mean rate")
    ax.set_ylabel("MN mean rate")
    ax.set_title("PN E → MN correlation")

    # PN I vs MN (inverse expected)
    ax = axes[2, 2]
    ax.scatter(np.mean(rates_i, axis=1), np.mean(mn_rates, axis=1),
               s=3, alpha=0.5, color=C_INH, label='I→MN')
    ax.set_xlabel("PN I mean rate")
    ax.set_ylabel("MN mean rate")
    ax.set_title("PN I → MN correlation")

    # Reward
    ax = axes[2, 3]
    rewards = np.array(episode_data['reward'])
    if rewards.ndim > 1:
        rewards = rewards[:, 0] if rewards.shape[1] > 0 else rewards.ravel()
    ax.plot(rewards, color='green', linewidth=1)
    ax.set_xlabel("env step")
    ax.set_ylabel("reward")
    ax.set_title(f"Episode reward (total={np.sum(rewards):.1f})")

    fig.suptitle("ProprioSpinal + Motor Neuron Circuit Diagnostics", fontsize=14, y=1.02)
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

        idx = jax.lax.dynamic_slice(perm, (minibatch_idx * batch_size,), (batch_size,))

        batch_obs = data['obs'][idx]
        batch_raw_action = data['raw_action'][idx]
        batch_log_prob = data['log_prob'][idx]
        batch_advantage = data['advantage'][idx]
        batch_target = data['target'][idx]
        batch_carry = data['carry'][idx]

        def loss_fn(params):
            pp, vp = params
            obs_norm = running_statistics.normalize(batch_obs, normalizer_params)

            logits, _ = policy_module.apply(pp, obs_norm, batch_carry)
            new_log_prob = action_dist.log_prob(logits, batch_raw_action)

            ratio = jp.exp(new_log_prob - batch_log_prob)
            adv = batch_advantage
            adv = (adv - jp.mean(adv)) / (jp.std(adv) + 1e-8)

            pg1 = -adv * ratio
            pg2 = -adv * jp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            policy_loss = jp.mean(jp.maximum(pg1, pg2))

            new_value = jp.squeeze(value_module.apply(vp, obs_norm), axis=-1)
            value_loss = jp.mean(jp.square(new_value - batch_target))

            entropy = jp.mean(action_dist.entropy(logits, entropy_rng))

            total = (policy_loss + vf_coef * value_loss
                     - entropy_cost * entropy)
            return total, {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "approx_kl": jp.mean((ratio - 1.0) - jp.log(ratio)),
            }

        (_loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            (policy_params, value_params)
        )
        updates, new_opt_state = optimizer.update(
            grads, opt_state, (policy_params, value_params)
        )
        new_pp, new_vp = optax.apply_updates((policy_params, value_params), updates)
        return (new_pp, new_vp, new_opt_state, normalizer_params, rng), metrics

    def ppo_update_epoch(carry, _):
        policy_params, value_params, opt_state, normalizer_params, rng, data = carry
        rng, perm_rng = jax.random.split(rng)
        T_times_B = data['obs'].shape[0]
        perm = jax.random.permutation(perm_rng, T_times_B)

        (policy_params, value_params, opt_state, normalizer_params, rng), metrics = jax.lax.scan(
            lambda c, i: sgd_step(c, i, data, perm),
            (policy_params, value_params, opt_state, normalizer_params, rng),
            jp.arange(num_minibatches),
        )
        return (policy_params, value_params, opt_state, normalizer_params, rng, data), metrics

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

        (policy_params, value_params, opt_state, _, rng, _), all_metrics = jax.lax.scan(
            ppo_update_epoch,
            (state.policy_params, state.value_params, state.opt_state,
             new_normalizer_params, rng, data),
            None,
            length=num_updates,
        )

        metrics = jax.tree.map(lambda x: x[-1, -1], all_metrics)
        metrics['mean_reward'] = jp.mean(rollout.reward)

        new_state = TrainingState(
            policy_params=policy_params,
            value_params=value_params,
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

            # ProprioSpinal diagnostics
            ps_diag = diagnostics['propriospinal']
            step_data['ps_spikes_exc'] = ps_diag['spikes_exc'][:, 0, :]
            step_data['ps_spikes_inh'] = ps_diag['spikes_inh'][:, 0, :]
            step_data['ps_voltages_exc'] = ps_diag['voltages_exc'][:, 0, :]
            step_data['ps_voltages_inh'] = ps_diag['voltages_inh'][:, 0, :]

            # Motor neuron diagnostics
            mn_diag = diagnostics['motor_neurons']
            step_data['mn_spikes'] = mn_diag['spikes'][:, 0, :]
            step_data['mn_voltages'] = mn_diag['voltages'][:, 0, :]
            step_data['motor_spike_rate'] = diagnostics['motor_spike_rate']

            # E/I drive to MN
            step_data['mn_input_exc'] = diagnostics['mn_input_exc']
            step_data['mn_input_inh'] = diagnostics['mn_input_inh']

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
    num_timesteps=600_000_000,
    num_evals=12,
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
    clip_eps=0.2,
    vf_coef=0.5,
    network_factory=config_dict.create(
        propriospinal_size=512,
        propriospinal_exc_ratio=0.8,
        motor_neuron_size=128,
        n_micro_steps=8,
        tau_min=1.0,
        tau_max=5.0,
        v_th=0.3,
        v_reset=0.0,
        beta_surrogate=5.0,
        n_refractory=2,
        value_hidden_layer_sizes=(512, 512, 512),
    ),
)

env_name = "mouse-propriospinal-mn"
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
               id=f"ps-mn-{exp_name}")
    wandb.config.update({
        "env_name": env_name,
        "policy_type": "propriospinal_mn",
        **dict(ppo_params.network_factory),
    })


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    print("=" * 80)
    print("Mouse Arm Imitation -- ProprioSpinal + Motor Neuron Policy")
    print("=" * 80)
    nf = ppo_params.network_factory
    n_exc = round(nf.propriospinal_size * nf.propriospinal_exc_ratio)
    n_inh = nf.propriospinal_size - n_exc
    print(f"ProprioSpinal: {nf.propriospinal_size} neurons ({n_exc}E + {n_inh}I)")
    print(f"  Recurrent: E→I (|W_ei|, excitatory), I→E (-|W_ie|, inhibitory)")
    print(f"  Direct projections to MN: E→MN (+), I→MN (-)")
    print(f"Motor neurons: {nf.motor_neuron_size} neurons (all excitatory)")
    print(f"Micro-steps: {nf.n_micro_steps}, v_th={nf.v_th}")

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

    policy_module = ProprioSpinalMNPolicy(
        propriospinal_size=nf.propriospinal_size,
        exc_ratio=nf.propriospinal_exc_ratio,
        motor_neuron_size=nf.motor_neuron_size,
        n_micro_steps=nf.n_micro_steps,
        tau_min=nf.tau_min, tau_max=nf.tau_max,
        v_th=nf.v_th, v_reset=nf.v_reset,
        beta_surrogate=nf.beta_surrogate,
        n_refractory=nf.n_refractory,
        output_size=param_size,
    )
    diag_policy_module = DiagnosticProprioSpinalMNPolicy(
        propriospinal_size=nf.propriospinal_size,
        exc_ratio=nf.propriospinal_exc_ratio,
        motor_neuron_size=nf.motor_neuron_size,
        n_micro_steps=nf.n_micro_steps,
        tau_min=nf.tau_min, tau_max=nf.tau_max,
        v_th=nf.v_th, v_reset=nf.v_reset,
        beta_surrogate=nf.beta_surrogate,
        n_refractory=nf.n_refractory,
        output_size=param_size,
    )
    value_module = networks.MLP(
        layer_sizes=list(nf.value_hidden_layer_sizes) + [1],
        activation=linen.swish,
        kernel_init=jax.nn.initializers.lecun_uniform(),
    )

    # Carry dimension: ps_v + mn_v + ps_r + mn_r
    carry_dim = (
        2 * nf.propriospinal_size   # ps_v + ps_r
        + 2 * nf.motor_neuron_size  # mn_v + mn_r
    )
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
        train_env, policy_module, value_module, action_dist, optimizer, train_config
    )
    eval_fn = make_eval_fn(
        eval_env, diag_policy_module, action_dist, episode_length, carry_dim
    )

    # JIT'd helpers for video rollout
    jit_vid_reset = jax.jit(eval_env_base.reset)
    jit_vid_step = jax.jit(eval_env_base.step)
    jit_vid_policy = jax.jit(policy_module.apply)

    def video_rollout(policy_params, normalizer_params, seed=0):
        rng = jax.random.PRNGKey(seed)
        state = jit_vid_reset(rng)
        carry = jp.zeros((1, carry_dim))
        rollout_states = [state]
        for t in range(episode_length):
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

            # Plot circuit diagnostics
            fig = plot_circuit_diagnostics(
                episode_data, v_th=nf.v_th,
                save_path=f"{ckpt_path}/{state.total_steps}_circuit.png",
            )
            if USE_WANDB:
                wandb.log({"eval/circuit": wandb.Image(fig)}, commit=False)
            plt.close(fig)

            # Log summary metrics
            rewards = np.array(episode_data['reward'])
            mn_rates = np.array(episode_data['motor_spike_rate'])
            eval_metrics = {
                "eval/episode_reward": float(np.sum(rewards)),
                "eval/mn_mean_spike_rate": float(np.mean(mn_rates)),
            }
            if USE_WANDB:
                wandb.log(eval_metrics, commit=False)
            print(f"  eval reward: {eval_metrics['eval/episode_reward']:.1f}")

            # Video rollout
            try:
                rollout_states = video_rollout(
                    state.policy_params, state.normalizer_params,
                    seed=state.total_steps,
                )
                fps = int(1.0 / eval_env_base.dt)
                frames = eval_env_base.render(
                    rollout_states, height=512, width=512, render_ghost=True
                )
                video_path = f"{ckpt_path}/{state.total_steps}.mp4"
                with imageio.get_writer(video_path, fps=fps) as vid:
                    for frame in frames:
                        vid.append_data(np.array(frame))
                if USE_WANDB:
                    wandb.log(
                        {"eval/rollout": wandb.Video(video_path, format="mp4")},
                        commit=False,
                    )
                print(f"  video -> {video_path}")
            except Exception as e:
                print(f"  video failed: {e}")

            # Save checkpoint
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
