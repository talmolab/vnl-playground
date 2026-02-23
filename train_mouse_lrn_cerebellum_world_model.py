"""
Training script for mouse arm imitation with LRN-Cerebellum spiking circuit.

Architecture based on Alstermark & Ekerot (2015):
  - ProprioSpinal (C3-C4 PNs): E/I LIF layer with Dale's law, refractory period,
    heterogeneous tau. E PNs drive motor readout; both E and I PNs send
    efference copy to LRN.
  - LRN Relay: Excitatory-only LIF layer with no lateral connections.
    Receives from both E and I PNs, computes, and relays as mossy fiber
    input to cerebellum.
  - Cerebellum: Sensory forward model (Wolpert & Kawato 1998) implemented as a
    nonlinear neural world model. Learned MLP networks estimate the state
    transition, observation mapping, and state update. Receives sensory feedback
    as observation (z) and efference copy via LRN as motor input (u).
    Innovation = sensory prediction error: small during voluntary movement,
    large during perturbation.

References:
  Alstermark B, Ekerot C-F (2015) "The lateral reticular nucleus; integration
  of descending and ascending systems regulating voluntary forelimb movements."
  Front. Comput. Neurosci. 9:102. doi: 10.3389/fncom.2015.00102

  Wolpert DM, Miall RC, Kawato M (1998) "Internal models in the cerebellum."
  Trends Cogn. Sci. 2(9):338-347. doi: 10.1016/S1364-6613(98)01221-2
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
from scipy import stats as scipy_stats

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

    Bifurcating output: E spike rates go to motor readout, E+I go to LRN.
    Dale's law: abs() on lateral weights enforces sign constraints.
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
# Module 2: LRN Relay (Lateral Reticular Nucleus)
# =============================================================================


class LRNRelayModule(linen.Module):
    """Lateral Reticular Nucleus -- excitatory-only LIF relay.

    Receives efference copy from both E and I propriospinal neurons,
    computes via excitatory-only LIF dynamics (no lateral connections),
    and relays as mossy fiber input to cerebellum.
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
    def __call__(self, x, v_carry, refrac_carry):
        n = self.n_neurons
        v_th = self.v_th
        v_reset = self.v_reset
        beta = self.beta_surrogate
        n_refrac = self.n_refractory

        I_input = linen.Dense(
            n,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            name="input_proj",
        )(x)

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
# Module 3: Motor Neurons (Excitatory-only LIF)
# =============================================================================


class MotorNeuronModule(linen.Module):
    """Motor neuron pool -- excitatory-only LIF.

    Receives pre-motor command (raw motor + cerebellar correction),
    transforms through LIF dynamics, output spike rates drive muscles.
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
    def __call__(self, x, v_carry, refrac_carry):
        n = self.n_neurons
        v_th = self.v_th
        v_reset = self.v_reset
        beta = self.beta_surrogate
        n_refrac = self.n_refractory

        I_input = linen.Dense(
            n,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            name="input_proj",
        )(x)

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
# Module 4: Cerebellum (Nonlinear World Model)
# =============================================================================


class CerebellumModule(linen.Module):
    """Cerebellum as a nonlinear sensory forward model.

    Predicts sensory consequences of motor commands (Wolpert & Kawato 1998).
    Innovation = actual sensory feedback - predicted sensory feedback.
    Small during voluntary movement, large during external perturbation.

    Replaces the linear Kalman filter with learned nonlinear networks:
      transition_net: MLP predicting next state from (x_hat, efference_copy)
      observation_net: MLP predicting sensory observation from state
      update_net: MLP computing state correction from (x_hat_pred, innovation)

    Biological interpretation:
      transition_net: internal model of limb dynamics
      observation_net: maps internal state to predicted sensory observation
      innovation: sensory prediction error (climbing fiber analogue)
      correction: output to modify motor command via deep cerebellar nuclei
    """

    state_dim: int
    obs_dim: int
    motor_dim: int
    output_dim: int
    hidden_dim: int = 128

    @linen.compact
    def __call__(self, sensory_obs, efference_copy, x_hat):
        """
        Args:
            sensory_obs: (B, obs_dim) - sensory feedback (proprioception + targets)
            efference_copy: (B, motor_dim) - LRN mossy fiber spike rates
            x_hat: (B, state_dim) - previous state estimate
        Returns:
            correction: (B, output_dim)
            x_hat_new: (B, state_dim)
            sensory_pred_loss: (B,)
        """
        sd = self.state_dim
        hd = self.hidden_dim

        # --- Transition network: f(x_hat, efference_copy) -> x_hat_pred ---
        trans_input = jp.concatenate([x_hat, efference_copy], axis=-1)
        h = linen.Dense(hd, name="trans_h1")(trans_input)
        h = linen.swish(h)
        h = linen.Dense(hd, name="trans_h2")(h)
        h = linen.swish(h)
        # Residual connection + tanh to bound state
        x_hat_pred = jp.tanh(x_hat + linen.Dense(sd, name="trans_out")(h))

        # --- Observation network: g(x_hat_pred) -> z_pred ---
        h = linen.Dense(hd, name="obs_h1")(x_hat_pred)
        h = linen.swish(h)
        z_pred = linen.Dense(self.obs_dim, name="obs_out")(h)

        # --- Innovation (sensory prediction error) ---
        innovation = sensory_obs - z_pred  # (B, obs_dim)

        # --- Supervised prediction loss ---
        sensory_pred_loss = jp.mean(innovation ** 2, axis=-1)  # (B,)

        # --- Update network: h(x_hat_pred, innovation) -> state correction ---
        update_input = jp.concatenate([x_hat_pred, innovation], axis=-1)
        h = linen.Dense(hd, name="update_h1")(update_input)
        h = linen.swish(h)
        state_correction = linen.Dense(sd, name="update_out")(h)
        x_hat_new = jp.tanh(x_hat_pred + state_correction)  # bounded ∈ [-1, 1]

        # --- Innovation-gated correction (DCN output ∝ climbing fiber error) ---
        innov_magnitude = jp.sqrt(
            jp.sum(innovation ** 2, axis=-1, keepdims=True) + 1e-6)
        gate_threshold_raw = self.param(
            'gate_threshold_raw', lambda k, s: jp.full(s, 1.0), (1,))
        gate_scale = self.param(
            'gate_scale', lambda k, s: jp.full(s, 2.0), (1,))
        gate = jax.nn.sigmoid(
            gate_scale * (innov_magnitude - jax.nn.softplus(gate_threshold_raw)))

        innov_features = linen.Dense(self.state_dim, name="innov_proj")(innovation)
        correction_input = jp.concatenate([x_hat_new, innov_features], axis=-1)
        raw_correction = linen.Dense(
            self.output_dim,
            kernel_init=jax.nn.initializers.uniform(0.01),
            name="correction_readout",
        )(correction_input)
        correction = jp.tanh(raw_correction) * gate  # gated by surprise

        return correction, x_hat_new, sensory_pred_loss


# =============================================================================
# Combined Policy
# =============================================================================


class LRNCerebellumPolicy(linen.Module):
    """Combined ProprioSpinal -> LRN -> Cerebellum -> Motor Neurons policy.

    Carry layout:
      [ps_v | lrn_v | mn_v | ps_r | lrn_r | mn_r | cb_x_hat]
       (ps)   (lrn)  (mn)   (ps)   (lrn)   (mn)    (cb)
    """

    propriospinal_size: int = 512
    exc_ratio: float = 0.8
    lrn_size: int = 256
    motor_neuron_size: int = 128
    cerebellum_state_dim: int = 64
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
        n_lrn = self.lrn_size
        n_mn = self.motor_neuron_size
        n_cb = self.cerebellum_state_dim
        n_exc = round(n_ps * self.exc_ratio)
        n_inh = n_ps - n_exc

        # Split carry
        idx = 0
        ps_v = carry_flat[:, idx:idx + n_ps]; idx += n_ps
        lrn_v = carry_flat[:, idx:idx + n_lrn]; idx += n_lrn
        mn_v = carry_flat[:, idx:idx + n_mn]; idx += n_mn
        ps_r = carry_flat[:, idx:idx + n_ps]; idx += n_ps
        lrn_r = carry_flat[:, idx:idx + n_lrn]; idx += n_lrn
        mn_r = carry_flat[:, idx:idx + n_mn]; idx += n_mn
        cb_x_hat = carry_flat[:, idx:idx + n_cb]

        # 1. ProprioSpinal: obs -> E/I LIF -> spike rates (full population)
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

        # 2. Raw motor command from propriospinal E population only
        raw_motor_cmd = linen.Dense(
            self.output_size,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            name="motor_readout",
        )(spike_rate_exc)

        # 3. LRN relay: both E+I PN rates -> LIF -> mossy fiber
        mossy_fiber, lrn_v_new, lrn_r_new = LRNRelayModule(
            n_neurons=n_lrn,
            n_micro_steps=self.n_micro_steps,
            tau_min=self.tau_min, tau_max=self.tau_max,
            v_th=self.v_th, v_reset=self.v_reset,
            beta_surrogate=self.beta_surrogate,
            n_refractory=self.n_refractory,
            name="lrn_relay",
        )(spike_rate, lrn_v, lrn_r)

        # 4. Cerebellum: nonlinear sensory forward model predicts + corrects
        correction, cb_x_hat_new, sensory_pred_loss = CerebellumModule(
            state_dim=n_cb,
            obs_dim=obs.shape[-1],
            motor_dim=n_lrn,
            output_dim=self.output_size,
            name="cerebellum",
        )(obs, mossy_fiber, cb_x_hat)

        # 5. Combine: motor command + weighted cerebellar correction
        w_raw = self.param('correction_weight_raw',
                           lambda k, s: jp.full(s, -1.0), (1,))
        w = jax.nn.sigmoid(w_raw)
        scaled_correction = w * correction
        pre_motor_cmd = raw_motor_cmd + scaled_correction

        # 6. Motor neurons: pre_motor_cmd -> LIF -> spike rates
        motor_spike_rate, mn_v_new, mn_r_new = MotorNeuronModule(
            n_neurons=n_mn,
            n_micro_steps=self.n_micro_steps,
            tau_min=self.tau_min, tau_max=self.tau_max,
            v_th=self.v_th, v_reset=self.v_reset,
            beta_surrogate=self.beta_surrogate,
            n_refractory=self.n_refractory,
            name="motor_neurons",
        )(pre_motor_cmd, mn_v, mn_r)

        # 7. Muscle readout from motor neuron spike rates
        logits = linen.Dense(
            self.output_size,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            name="muscle_readout",
        )(motor_spike_rate)

        # 8. Reassemble carry
        new_carry = jp.concatenate([
            ps_v_new, lrn_v_new, mn_v_new, ps_r_new, lrn_r_new, mn_r_new,
            cb_x_hat_new,
        ], axis=-1)

        return logits, new_carry, sensory_pred_loss, scaled_correction


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


class DiagnosticLRNModule(linen.Module):
    """LRN relay with full spike and voltage traces."""

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
    def __call__(self, x, v_carry, refrac_carry):
        n = self.n_neurons
        v_th, v_reset = self.v_th, self.v_reset
        n_refrac = self.n_refractory

        I_input = linen.Dense(
            n,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            name="input_proj",
        )(x)

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
    def __call__(self, x, v_carry, refrac_carry):
        n = self.n_neurons
        v_th, v_reset = self.v_th, self.v_reset
        n_refrac = self.n_refractory

        I_input = linen.Dense(
            n,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            name="input_proj",
        )(x)

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


class DiagnosticCerebellumModule(linen.Module):
    """Cerebellum nonlinear sensory forward model with diagnostic output."""

    state_dim: int
    obs_dim: int
    motor_dim: int
    output_dim: int
    hidden_dim: int = 128

    @linen.compact
    def __call__(self, sensory_obs, efference_copy, x_hat):
        sd = self.state_dim
        hd = self.hidden_dim

        # --- Transition network ---
        trans_input = jp.concatenate([x_hat, efference_copy], axis=-1)
        h = linen.Dense(hd, name="trans_h1")(trans_input)
        h = linen.swish(h)
        h = linen.Dense(hd, name="trans_h2")(h)
        h = linen.swish(h)
        x_hat_pred = jp.tanh(x_hat + linen.Dense(sd, name="trans_out")(h))

        # --- Observation network ---
        h = linen.Dense(hd, name="obs_h1")(x_hat_pred)
        h = linen.swish(h)
        z_pred = linen.Dense(self.obs_dim, name="obs_out")(h)

        # --- Innovation ---
        innovation = sensory_obs - z_pred

        # --- Supervised prediction loss ---
        sensory_pred_loss = jp.mean(innovation ** 2, axis=-1)

        # --- Update network ---
        update_input = jp.concatenate([x_hat_pred, innovation], axis=-1)
        h = linen.Dense(hd, name="update_h1")(update_input)
        h = linen.swish(h)
        state_correction = linen.Dense(sd, name="update_out")(h)
        x_hat_new = jp.tanh(x_hat_pred + state_correction)

        # --- Innovation-gated correction ---
        innov_magnitude = jp.sqrt(
            jp.sum(innovation ** 2, axis=-1, keepdims=True) + 1e-6)
        gate_threshold_raw = self.param(
            'gate_threshold_raw', lambda k, s: jp.full(s, 1.0), (1,))
        gate_scale = self.param(
            'gate_scale', lambda k, s: jp.full(s, 2.0), (1,))
        gate = jax.nn.sigmoid(
            gate_scale * (innov_magnitude - jax.nn.softplus(gate_threshold_raw)))

        innov_features = linen.Dense(self.state_dim, name="innov_proj")(innovation)
        correction_input = jp.concatenate([x_hat_new, innov_features], axis=-1)
        raw_correction = linen.Dense(
            self.output_dim,
            kernel_init=jax.nn.initializers.uniform(0.01),
            name="correction_readout",
        )(correction_input)
        correction = jp.tanh(raw_correction) * gate

        diagnostics = {
            "x_hat_pred": x_hat_pred,
            "x_hat_new": x_hat_new,
            "innovation": innovation,
            "innovation_norm": jp.sqrt(jp.sum(innovation ** 2, axis=-1)),
            "correction": correction,
            "gate": jp.squeeze(gate, axis=-1),
            "sensory_pred_loss": sensory_pred_loss,
        }

        return correction, x_hat_new, diagnostics


class DiagnosticLRNCerebellumPolicy(linen.Module):
    """Combined policy with full diagnostic output from all modules."""

    propriospinal_size: int = 512
    exc_ratio: float = 0.8
    lrn_size: int = 256
    motor_neuron_size: int = 128
    cerebellum_state_dim: int = 64
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
        n_lrn = self.lrn_size
        n_mn = self.motor_neuron_size
        n_cb = self.cerebellum_state_dim
        n_exc = round(n_ps * self.exc_ratio)
        n_inh = n_ps - n_exc

        idx = 0
        ps_v = carry_flat[:, idx:idx + n_ps]; idx += n_ps
        lrn_v = carry_flat[:, idx:idx + n_lrn]; idx += n_lrn
        mn_v = carry_flat[:, idx:idx + n_mn]; idx += n_mn
        ps_r = carry_flat[:, idx:idx + n_ps]; idx += n_ps
        lrn_r = carry_flat[:, idx:idx + n_lrn]; idx += n_lrn
        mn_r = carry_flat[:, idx:idx + n_mn]; idx += n_mn
        cb_x_hat = carry_flat[:, idx:idx + n_cb]

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

        raw_motor_cmd = linen.Dense(
            self.output_size,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            name="motor_readout",
        )(spike_rate_exc)

        mossy_fiber, lrn_v_new, lrn_r_new, lrn_diag = DiagnosticLRNModule(
            n_neurons=n_lrn,
            n_micro_steps=self.n_micro_steps,
            tau_min=self.tau_min, tau_max=self.tau_max,
            v_th=self.v_th, v_reset=self.v_reset,
            beta_surrogate=self.beta_surrogate,
            n_refractory=self.n_refractory,
            name="lrn_relay",
        )(spike_rate, lrn_v, lrn_r)

        correction, cb_x_hat_new, cb_diag = DiagnosticCerebellumModule(
            state_dim=n_cb,
            obs_dim=obs.shape[-1],
            motor_dim=n_lrn,
            output_dim=self.output_size,
            name="cerebellum",
        )(obs, mossy_fiber, cb_x_hat)

        w_raw = self.param('correction_weight_raw',
                           lambda k, s: jp.full(s, -1.0), (1,))
        w = jax.nn.sigmoid(w_raw)
        scaled_correction = w * correction
        pre_motor_cmd = raw_motor_cmd + scaled_correction

        # Motor neurons: pre_motor_cmd -> LIF -> spike rates
        motor_spike_rate, mn_v_new, mn_r_new, mn_diag = DiagnosticMotorNeuronModule(
            n_neurons=n_mn,
            n_micro_steps=self.n_micro_steps,
            tau_min=self.tau_min, tau_max=self.tau_max,
            v_th=self.v_th, v_reset=self.v_reset,
            beta_surrogate=self.beta_surrogate,
            n_refractory=self.n_refractory,
            name="motor_neurons",
        )(pre_motor_cmd, mn_v, mn_r)

        # Muscle readout from motor neuron spike rates
        logits = linen.Dense(
            self.output_size,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            name="muscle_readout",
        )(motor_spike_rate)

        new_carry = jp.concatenate([
            ps_v_new, lrn_v_new, mn_v_new, ps_r_new, lrn_r_new, mn_r_new,
            cb_x_hat_new,
        ], axis=-1)

        diagnostics = {
            "propriospinal": ps_diag,
            "lrn": lrn_diag,
            "motor_neurons": mn_diag,
            "cerebellum": cb_diag,
            "raw_motor_cmd": raw_motor_cmd,
            "correction": correction,
            "correction_weight": w,
            "motor_spike_rate": motor_spike_rate,
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
    perturb_target: Any  # zeros: L2 regularization target for correction


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

# Seaborn tab10 E/I colour scheme — consistent across all figures
C_EXC = '#1f77b4'   # blue  (excitatory)
C_INH = '#ff7f0e'   # orange (inhibitory)

from matplotlib.colors import LinearSegmentedColormap
CMAP_EXC = LinearSegmentedColormap.from_list('exc', ['#ffffff', C_EXC])
CMAP_INH = LinearSegmentedColormap.from_list('inh', ['#ffffff', C_INH])


def plot_circuit_diagnostics(episode_data, v_th, save_path=None,
                             perturb_start=None, perturb_end=None):
    """Plot overview of the LRN-cerebellum circuit activity."""
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))

    # --- Row 0: ProprioSpinal ---
    # E raster
    ax = axes[0, 0]
    spikes_e = np.array(episode_data['ps_spikes_exc'])  # (T, K, N_exc)
    T_ep = spikes_e.shape[0]
    # Average over micro-steps for each env step
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

    # --- Row 1: LRN ---
    ax = axes[1, 0]
    lrn_spikes = np.array(episode_data['lrn_spikes'])  # (T, K, N_lrn)
    lrn_rates = np.mean(lrn_spikes, axis=1)
    n_show_lrn = min(64, lrn_rates.shape[1])
    ax.imshow(lrn_rates[:, :n_show_lrn].T, aspect='auto', cmap=CMAP_EXC,
              interpolation='nearest')
    ax.set_xlabel("env step")
    ax.set_ylabel("LRN neuron")
    ax.set_title("LRN relay activity (E only)", color=C_EXC)

    # LRN rate histogram
    ax = axes[1, 1]
    mean_lrn = np.mean(lrn_rates, axis=0)
    ax.hist(mean_lrn, bins=30, color=C_EXC, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(mean_lrn), color='black', ls='--',
               label=f"mean={np.mean(mean_lrn):.3f}")
    ax.set_xlabel("mean spike rate")
    ax.set_ylabel("count")
    ax.set_title("LRN rate distribution (E only)", color=C_EXC)
    ax.legend(fontsize=7)

    # LRN voltage traces
    ax = axes[1, 2]
    lrn_volts = np.array(episode_data['lrn_voltages'])  # (T, K, N_lrn)
    # Show micro-step voltages from a few env steps
    mid = T_ep // 2
    for j in range(min(8, lrn_volts.shape[2])):
        ax.plot(lrn_volts[mid, :, j], alpha=0.6, linewidth=1, color=C_EXC)
    ax.axhline(v_th, color='black', ls='--', alpha=0.5, label='threshold')
    ax.set_xlabel("micro-step")
    ax.set_ylabel("voltage")
    ax.set_title(f"LRN voltages (env step {mid})")
    ax.legend(fontsize=7)

    # Input-output correlation
    ax = axes[1, 3]
    ax.scatter(np.mean(rates_e, axis=1), np.mean(lrn_rates, axis=1),
               s=3, alpha=0.5, color=C_EXC)
    ax.set_xlabel("PN E mean rate")
    ax.set_ylabel("LRN mean rate")
    ax.set_title("PN->LRN relay fidelity")

    # --- Row 2: Cerebellum ---
    ax = axes[2, 0]
    innov_norm = np.array(episode_data['cb_innovation_norm'])  # (T,) or (T,1)
    if innov_norm.ndim > 1:
        innov_norm = innov_norm[:, 0]
    ax.plot(innov_norm, color='purple', linewidth=1)
    ax.set_xlabel("env step")
    ax.set_ylabel("||innovation||")
    ax.set_title("Cerebellum prediction error")
    if perturb_start is not None:
        ax.axvspan(perturb_start, perturb_end, alpha=0.15, color='red',
                   label='perturbation')
        ax.legend(fontsize=7)

    # State estimate dims
    ax = axes[2, 1]
    x_hat = np.array(episode_data['cb_x_hat_new'])  # (T, B, state_dim)
    if x_hat.ndim == 3:
        x_hat = x_hat[:, 0, :]
    for d in range(min(8, x_hat.shape[1])):
        ax.plot(x_hat[:, d], alpha=0.6, linewidth=1)
    ax.set_xlabel("env step")
    ax.set_ylabel("x_hat")
    ax.set_title("Cerebellum state estimate")
    if perturb_start is not None:
        ax.axvspan(perturb_start, perturb_end, alpha=0.15, color='red')

    # Correction magnitude
    ax = axes[2, 2]
    corr = np.array(episode_data['cb_correction'])  # (T, B, output_dim)
    if corr.ndim == 3:
        corr = corr[:, 0, :]
    corr_norm = np.sqrt(np.sum(corr ** 2, axis=-1))
    ax.plot(corr_norm, color='darkorange', linewidth=1)
    ax.set_xlabel("env step")
    ax.set_ylabel("||correction||")
    ax.set_title("Cerebellar correction magnitude")
    if perturb_start is not None:
        ax.axvspan(perturb_start, perturb_end, alpha=0.15, color='red')

    # Reward
    ax = axes[2, 3]
    rewards = np.array(episode_data['reward'])
    if rewards.ndim > 1:
        rewards = rewards[:, 0] if rewards.shape[1] > 0 else rewards.ravel()
    ax.plot(rewards, color='green', linewidth=1)
    ax.set_xlabel("env step")
    ax.set_ylabel("reward")
    ax.set_title(f"Episode reward (total={np.sum(rewards):.1f})")
    if perturb_start is not None:
        ax.axvspan(perturb_start, perturb_end, alpha=0.15, color='red')

    fig.suptitle("LRN-Cerebellum Circuit Diagnostics", fontsize=14, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_perturbation_comparison(clean_data, perturbed_data,
                                 perturb_start, perturb_end, save_path=None):
    """Compare clean vs perturbed eval: innovation and correction.

    Y-axes are matched across clean/perturbed panels for direct comparison.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    def _get_innov(data):
        innov = np.array(data['cb_innovation_norm'])
        return innov[:, 0] if innov.ndim > 1 else innov

    def _get_corr_norm(data):
        corr = np.array(data['cb_correction'])
        if corr.ndim == 3:
            corr = corr[:, 0, :]
        return np.sqrt(np.sum(corr ** 2, axis=-1))

    # Pre-compute shared Y limits
    innov_clean = _get_innov(clean_data)
    innov_pert = _get_innov(perturbed_data)
    innov_ymax = max(np.max(innov_clean), np.max(innov_pert)) * 1.1

    corr_clean = _get_corr_norm(clean_data)
    corr_pert = _get_corr_norm(perturbed_data)
    corr_ymax = max(np.max(corr_clean), np.max(corr_pert)) * 1.1

    # Row 0: innovation (matched Y-axes)
    for col, (innov_trace, label) in enumerate(
        [(innov_clean, "Clean"), (innov_pert, "Perturbed")]
    ):
        ax = axes[0, col]
        ax.plot(innov_trace, color='purple', linewidth=1)
        ax.set_xlabel("env step")
        ax.set_ylabel("||innovation||")
        ax.set_title(f"{label} — prediction error")
        ax.set_ylim(0, innov_ymax)
        if col == 1:
            ax.axvspan(perturb_start, perturb_end, alpha=0.15, color='red',
                       label='perturbation')
            ax.legend(fontsize=7)

    # Row 1: correction magnitude (matched Y-axes)
    for col, (corr_trace, label) in enumerate(
        [(corr_clean, "Clean"), (corr_pert, "Perturbed")]
    ):
        ax = axes[1, col]
        ax.plot(corr_trace, color='darkorange', linewidth=1)
        ax.set_xlabel("env step")
        ax.set_ylabel("||correction||")
        ax.set_title(f"{label} — cerebellar correction")
        ax.set_ylim(0, corr_ymax)
        if col == 1:
            ax.axvspan(perturb_start, perturb_end, alpha=0.15, color='red')

    fig.suptitle("Perturbation Response: Clean vs Perturbed", fontsize=14,
                 y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def run_cerebellum_hypothesis_test(clean_data, perturbed_data,
                                    perturb_start, perturb_end,
                                    episode_length,
                                    save_path=None):
    """Test whether cerebellum prediction error tracks initial reach,
    perturbation correction, or both. Also test directional sensitivity.

    Phases:
      - Initial reach: steps [0, perturb_start)
      - Perturbation window: steps [perturb_start, perturb_end)
      - Post-perturbation: steps [perturb_end, episode_length)

    Returns dict of metrics suitable for wandb logging, plus a summary figure.
    """
    # --- Extract signals ---
    def _get_innov_norm(data):
        innov = np.array(data['cb_innovation_norm'])
        return innov[:, 0] if innov.ndim > 1 else innov

    def _get_innov_vec(data):
        """Raw innovation vector (T, obs_dim) or (T, B, obs_dim)."""
        innov = np.array(data['cb_innovation'])
        if innov.ndim == 3:
            innov = innov[:, 0, :]  # take first env
        return innov

    def _get_corr_norm(data):
        corr = np.array(data['cb_correction'])
        if corr.ndim == 3:
            corr = corr[:, 0, :]
        return np.sqrt(np.sum(corr ** 2, axis=-1))

    innov_clean = _get_innov_norm(clean_data)
    innov_pert = _get_innov_norm(perturbed_data)
    corr_clean = _get_corr_norm(clean_data)
    corr_pert = _get_corr_norm(perturbed_data)

    # Raw innovation vectors for directional analysis
    innov_vec_clean = _get_innov_vec(clean_data)
    innov_vec_pert = _get_innov_vec(perturbed_data)

    # --- Phase segmentation ---
    phases = {
        'initial_reach': (0, perturb_start),
        'perturbation': (perturb_start, perturb_end),
        'post_perturbation': (perturb_end, episode_length),
    }

    metrics = {}

    # --- Phase-wise innovation magnitude ---
    for phase_name, (t0, t1) in phases.items():
        if t1 <= t0:
            continue
        c_mean = float(np.mean(innov_clean[t0:t1]))
        p_mean = float(np.mean(innov_pert[t0:t1]))
        c_corr = float(np.mean(corr_clean[t0:t1]))
        p_corr = float(np.mean(corr_pert[t0:t1]))

        metrics[f'hypothesis/{phase_name}/clean_innovation'] = c_mean
        metrics[f'hypothesis/{phase_name}/perturbed_innovation'] = p_mean
        metrics[f'hypothesis/{phase_name}/innovation_diff'] = p_mean - c_mean
        metrics[f'hypothesis/{phase_name}/clean_correction'] = c_corr
        metrics[f'hypothesis/{phase_name}/perturbed_correction'] = p_corr

        # Welch's t-test: is perturbed innovation different from clean
        # in this phase? Use per-timestep values.
        c_vals = innov_clean[t0:t1]
        p_vals = innov_pert[t0:t1]
        if len(c_vals) > 1 and len(p_vals) > 1:
            t_stat, p_val = scipy_stats.ttest_ind(
                p_vals, c_vals, equal_var=False
            )
            metrics[f'hypothesis/{phase_name}/t_stat'] = float(t_stat)
            metrics[f'hypothesis/{phase_name}/p_value'] = float(p_val)

    # --- Which phase shows the largest perturbation effect? ---
    diffs = {}
    for phase_name, (t0, t1) in phases.items():
        if t1 <= t0:
            continue
        diffs[phase_name] = float(
            np.mean(innov_pert[t0:t1]) - np.mean(innov_clean[t0:t1])
        )
    if diffs:
        dominant_phase = max(diffs, key=lambda k: abs(diffs[k]))
        metrics['hypothesis/dominant_phase'] = dominant_phase
        metrics['hypothesis/dominant_phase_diff'] = diffs[dominant_phase]

    # --- Directional analysis ---
    # Mean innovation vector per phase (perturbed - clean) gives the
    # direction in observation space that the cerebellum responds to most.
    dir_metrics = {}
    for phase_name, (t0, t1) in phases.items():
        if t1 <= t0:
            continue
        diff_vec = np.mean(innov_vec_pert[t0:t1], axis=0) - \
                   np.mean(innov_vec_clean[t0:t1], axis=0)
        dir_norm = float(np.linalg.norm(diff_vec))
        dir_metrics[phase_name] = {
            'diff_vec': diff_vec,
            'norm': dir_norm,
        }
        metrics[f'hypothesis/{phase_name}/direction_norm'] = dir_norm

    # Per-dimension variance of the innovation difference (which obs dims
    # drive the prediction error most during perturbation?)
    if 'perturbation' in phases:
        t0, t1 = phases['perturbation']
        if t1 > t0:
            pert_diff = innov_vec_pert[t0:t1] - innov_vec_clean[t0:t1]
            dim_var = np.var(pert_diff, axis=0)
            # Store top-5 most responsive dimensions
            top_dims = np.argsort(dim_var)[::-1][:5]
            for rank, d in enumerate(top_dims):
                metrics[f'hypothesis/top_dim_{rank}/index'] = int(d)
                metrics[f'hypothesis/top_dim_{rank}/variance'] = float(dim_var[d])

    # --- Summary figure ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 0: Innovation traces per phase
    ax = axes[0, 0]
    t = np.arange(episode_length)
    ax.plot(t, innov_clean, color='steelblue', linewidth=1, label='Clean')
    ax.plot(t, innov_pert, color='crimson', linewidth=1, label='Perturbed')
    for phase_name, (t0, t1) in phases.items():
        if phase_name == 'perturbation':
            ax.axvspan(t0, t1, alpha=0.15, color='red')
    ax.set_xlabel("env step")
    ax.set_ylabel("||innovation||")
    ax.set_title("Innovation: clean vs perturbed")
    ax.legend(fontsize=8)

    # Row 0, col 1: Phase means bar chart (innovation)
    ax = axes[0, 1]
    phase_names = [p for p in phases if phases[p][1] > phases[p][0]]
    clean_means = [float(np.mean(innov_clean[phases[p][0]:phases[p][1]]))
                   for p in phase_names]
    pert_means = [float(np.mean(innov_pert[phases[p][0]:phases[p][1]]))
                  for p in phase_names]
    x_pos = np.arange(len(phase_names))
    w = 0.35
    ax.bar(x_pos - w/2, clean_means, w, label='Clean', color='steelblue',
           alpha=0.8)
    ax.bar(x_pos + w/2, pert_means, w, label='Perturbed', color='crimson',
           alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([p.replace('_', '\n') for p in phase_names], fontsize=8)
    ax.set_ylabel("mean ||innovation||")
    ax.set_title("Phase-wise innovation")
    ax.legend(fontsize=8)
    # Add p-values
    for i, p in enumerate(phase_names):
        pv = metrics.get(f'hypothesis/{p}/p_value', None)
        if pv is not None:
            sig = "***" if pv < 0.001 else "**" if pv < 0.01 else \
                  "*" if pv < 0.05 else "ns"
            y_top = max(clean_means[i], pert_means[i])
            ax.text(x_pos[i], y_top * 1.05, sig, ha='center', fontsize=10,
                    fontweight='bold')

    # Row 0, col 2: Correction traces
    ax = axes[0, 2]
    ax.plot(t, corr_clean, color='steelblue', linewidth=1, label='Clean')
    ax.plot(t, corr_pert, color='crimson', linewidth=1, label='Perturbed')
    for phase_name, (t0, t1) in phases.items():
        if phase_name == 'perturbation':
            ax.axvspan(t0, t1, alpha=0.15, color='red')
    ax.set_xlabel("env step")
    ax.set_ylabel("||correction||")
    ax.set_title("Correction: clean vs perturbed")
    ax.legend(fontsize=8)

    # Row 1, col 0: Innovation difference (pert - clean)
    ax = axes[1, 0]
    diff_trace = innov_pert - innov_clean
    ax.plot(t, diff_trace, color='black', linewidth=1)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.axvspan(perturb_start, perturb_end, alpha=0.15, color='red',
               label='perturbation')
    ax.set_xlabel("env step")
    ax.set_ylabel("delta ||innovation||")
    ax.set_title("Perturbation effect (pert - clean)")
    ax.legend(fontsize=8)

    # Row 1, col 1: Directional norms per phase
    ax = axes[1, 1]
    dir_norms = [dir_metrics.get(p, {}).get('norm', 0) for p in phase_names]
    bars = ax.bar(x_pos, dir_norms, color='mediumpurple', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([p.replace('_', '\n') for p in phase_names], fontsize=8)
    ax.set_ylabel("||mean innov direction||")
    ax.set_title("Directional sensitivity by phase")
    # Highlight dominant
    if diffs:
        dom_idx = phase_names.index(dominant_phase) if dominant_phase in phase_names else None
        if dom_idx is not None:
            bars[dom_idx].set_edgecolor('red')
            bars[dom_idx].set_linewidth(2)

    # Row 1, col 2: Top responsive observation dimensions during perturbation
    ax = axes[1, 2]
    if 'perturbation' in phases:
        t0, t1 = phases['perturbation']
        if t1 > t0:
            pert_diff = innov_vec_pert[t0:t1] - innov_vec_clean[t0:t1]
            dim_var = np.var(pert_diff, axis=0)
            top_k = min(20, len(dim_var))
            top_dims_all = np.argsort(dim_var)[::-1][:top_k]
            ax.barh(range(top_k), dim_var[top_dims_all], color='teal', alpha=0.8)
            ax.set_yticks(range(top_k))
            ax.set_yticklabels([f"dim {d}" for d in top_dims_all], fontsize=7)
            ax.set_xlabel("variance of innov diff")
            ax.set_title("Top obs dims (perturbation window)")
            ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "No perturbation phase", ha='center', va='center',
                transform=ax.transAxes)

    # Text summary
    summary_lines = ["Hypothesis Test Summary:"]
    for p in phase_names:
        pv = metrics.get(f'hypothesis/{p}/p_value', None)
        diff_val = metrics.get(f'hypothesis/{p}/innovation_diff', 0)
        sig = f"p={pv:.4f}" if pv is not None else "N/A"
        summary_lines.append(f"  {p}: diff={diff_val:.4f} ({sig})")
    if diffs:
        summary_lines.append(
            f"  Dominant phase: {dominant_phase} "
            f"(diff={diffs[dominant_phase]:.4f})"
        )
    fig.suptitle("\n".join(summary_lines), fontsize=11, y=1.08,
                 family='monospace', ha='center')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")

    return fig, metrics


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
    pred_coef = config.pred_coef
    correction_coef = config.correction_coef
    act_size = config.act_size
    obs_size = config.obs_size
    num_envs = config.num_envs

    def collect_rollout(state: TrainingState):
        def step_fn(carry, _):
            env_state, mem_carry, rng = carry
            rng, action_rng = jax.random.split(rng)

            obs_norm = running_statistics.normalize(
                env_state.obs, state.normalizer_params
            )
            logits, new_carry, _pred_loss, _corr = policy_module.apply(
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

            # Correction regularization: with xfrc_applied perturbation we
            # don't have a clean action-space correction target.  PPO reward
            # gradient trains the cerebellum to compensate during perturbation.
            # corr_loss acts as L2 regularizer on correction (toward zero).
            perturb_target = jp.zeros((env_state.obs.shape[0], 2 * act_size))

            transition = Transition(
                obs=env_state.obs,
                action=action,
                raw_action=raw_action,
                log_prob=log_prob,
                value=value,
                reward=next_env_state.reward,
                done=next_env_state.done,
                carry=mem_carry,
                perturb_target=perturb_target,
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
        batch_perturb_target = data['perturb_target'][idx]

        def loss_fn(params):
            pp, vp = params
            obs_norm = running_statistics.normalize(batch_obs, normalizer_params)

            logits, _, sensory_pred_loss, scaled_correction = policy_module.apply(
                pp, obs_norm, batch_carry)
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

            # Supervised forward model loss: trains cerebellum to predict
            # sensory consequences of efference copy (Wolpert & Kawato 1998)
            pred_loss = jp.mean(sensory_pred_loss)

            # Correction regularization: L2 toward zero.  With xfrc_applied
            # perturbation, the reward gradient teaches useful corrections.
            corr_loss = jp.mean(
                jp.sum((scaled_correction - batch_perturb_target) ** 2, axis=-1))

            total = (policy_loss + vf_coef * value_loss
                     - entropy_cost * entropy + pred_coef * pred_loss
                     + correction_coef * corr_loss)
            return total, {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "pred_loss": pred_loss,
                "corr_loss": corr_loss,
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
            'perturb_target': flat(rollout.perturb_target),
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

            # LRN diagnostics
            lrn_diag = diagnostics['lrn']
            step_data['lrn_spikes'] = lrn_diag['spikes'][:, 0, :]
            step_data['lrn_voltages'] = lrn_diag['voltages'][:, 0, :]

            # Motor neuron diagnostics
            mn_diag = diagnostics['motor_neurons']
            step_data['mn_spikes'] = mn_diag['spikes'][:, 0, :]
            step_data['mn_voltages'] = mn_diag['voltages'][:, 0, :]
            step_data['motor_spike_rate'] = diagnostics['motor_spike_rate']

            # Cerebellum diagnostics
            cb_diag = diagnostics['cerebellum']
            step_data['cb_innovation_norm'] = cb_diag['innovation_norm']
            step_data['cb_innovation'] = cb_diag['innovation']
            step_data['cb_x_hat_new'] = cb_diag['x_hat_new']
            step_data['cb_correction'] = cb_diag['correction']

            # Circuit-level
            step_data['raw_motor_cmd'] = diagnostics['raw_motor_cmd']
            step_data['correction_weight'] = diagnostics['correction_weight']

            return (next_env_state, new_carry, rng), step_data

        _, episode_data = jax.lax.scan(
            step_fn, (env_state, carry, rng), None, length=episode_length
        )
        return episode_data

    return eval_episode


def make_perturbed_eval_fn(eval_env, diag_policy_module, action_dist,
                           episode_length, carry_dim,
                           start_frac, end_frac, scale,
                           perturb_body_id):
    """Diagnostic eval with external force perturbation during a window."""
    start = int(episode_length * start_frac)
    end = int(episode_length * end_frac)

    @jax.jit
    def eval_episode_perturbed(policy_params, normalizer_params, rng):
        rng, reset_rng, force_rng = jax.random.split(rng, 3)
        env_state = eval_env.reset(jax.random.split(reset_rng, 1))
        carry = jp.zeros((1, carry_dim))
        # Random 3D force direction, normalized then scaled
        force_dir = jax.random.normal(force_rng, (3,))
        force_3d = force_dir / (jp.linalg.norm(force_dir) + 1e-8) * scale

        def step_fn(carry_state, step_idx):
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

            # Apply external force perturbation in window via xfrc_applied
            in_window = ((step_idx >= start) & (step_idx < end)).astype(
                jp.float32
            )
            # env_state.data is batched (1, nbody, 6) from VmapWrapper
            xfrc = jp.zeros_like(env_state.data.xfrc_applied)
            xfrc = xfrc.at[..., perturb_body_id, :3].set(force_3d * in_window)
            new_data = env_state.data.replace(xfrc_applied=xfrc)
            env_state_perturbed = env_state.replace(data=new_data)

            next_env_state = eval_env.step(env_state_perturbed, action)
            new_carry = new_carry * (1.0 - next_env_state.done)

            step_data = {
                'reward': next_env_state.reward,
                'done': next_env_state.done,
                'perturb_active': in_window,
            }

            ps_diag = diagnostics['propriospinal']
            step_data['ps_spikes_exc'] = ps_diag['spikes_exc'][:, 0, :]
            step_data['ps_spikes_inh'] = ps_diag['spikes_inh'][:, 0, :]
            step_data['ps_voltages_exc'] = ps_diag['voltages_exc'][:, 0, :]
            step_data['ps_voltages_inh'] = ps_diag['voltages_inh'][:, 0, :]

            lrn_diag = diagnostics['lrn']
            step_data['lrn_spikes'] = lrn_diag['spikes'][:, 0, :]
            step_data['lrn_voltages'] = lrn_diag['voltages'][:, 0, :]

            mn_diag = diagnostics['motor_neurons']
            step_data['mn_spikes'] = mn_diag['spikes'][:, 0, :]
            step_data['mn_voltages'] = mn_diag['voltages'][:, 0, :]
            step_data['motor_spike_rate'] = diagnostics['motor_spike_rate']

            cb_diag = diagnostics['cerebellum']
            step_data['cb_innovation_norm'] = cb_diag['innovation_norm']
            step_data['cb_innovation'] = cb_diag['innovation']
            step_data['cb_x_hat_new'] = cb_diag['x_hat_new']
            step_data['cb_correction'] = cb_diag['correction']

            step_data['raw_motor_cmd'] = diagnostics['raw_motor_cmd']
            step_data['correction_weight'] = diagnostics['correction_weight']

            return (next_env_state, new_carry, rng), step_data

        _, episode_data = jax.lax.scan(
            step_fn, (env_state, carry, rng),
            jp.arange(episode_length), length=episode_length
        )
        return episode_data

    return eval_episode_perturbed


# =============================================================================
# Config
# =============================================================================


ppo_params = config_dict.create(
    num_timesteps=600_000_000,
    num_evals=30,
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
    pred_coef=0.01,
    correction_coef=0.1,
    num_envs=4096,
    batch_size=256,
    max_grad_norm=1.0,
    gae_lambda=0.95,
    clip_eps=0.2,
    vf_coef=0.5,
    network_factory=config_dict.create(
        propriospinal_size=512,
        propriospinal_exc_ratio=0.8,
        lrn_size=256,
        motor_neuron_size=128,
        cerebellum_state_dim=64,
        n_micro_steps=8,
        tau_min=1.0,
        tau_max=5.0,
        v_th=0.3,
        v_reset=0.0,
        beta_surrogate=5.0,
        n_refractory=2,
        value_hidden_layer_sizes=(512, 512, 512),
        # Perturbation during training (gives cerebellum a reason to exist)
        # Keep mild so propriospinal learns stable control first;
        # increase prob/scale later for fine-tuning.
        perturb_start_frac=0.1,     # window starts at 10% of episode
        perturb_end_frac=0.3,       # window ends at 30% (20% window)
        perturb_scale=0.05,         # external force magnitude (Newtons) on target body
        perturb_prob=0.5,           # fraction of episodes with perturbation
        perturb_body="ulna-mouse",  # body to apply external force to
    ),
)

env_name = "mouse-lrn-cerebellum"
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
               id=f"lrn-cb-{exp_name}")
    wandb.config.update({
        "env_name": env_name,
        "policy_type": "lrn_cerebellum",
        **dict(ppo_params.network_factory),
    })


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    print("=" * 80)
    print("Mouse Arm Imitation -- LRN-Cerebellum Circuit Policy")
    print("=" * 80)
    nf = ppo_params.network_factory
    n_exc = round(nf.propriospinal_size * nf.propriospinal_exc_ratio)
    n_inh = nf.propriospinal_size - n_exc
    print(f"ProprioSpinal: {nf.propriospinal_size} neurons ({n_exc}E + {n_inh}I)")
    print(f"LRN relay: {nf.lrn_size} neurons (all excitatory)")
    print(f"Motor neurons: {nf.motor_neuron_size} neurons (all excitatory)")
    print(f"Cerebellum: state_dim={nf.cerebellum_state_dim} (nonlinear world model)")
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

    # Look up perturbation body index from the compiled MuJoCo model
    perturb_body_name = nf.perturb_body
    perturb_body_id = env.mj_model.body(perturb_body_name).id
    nbody = env.mj_model.nbody
    print(f"Perturbation body: {perturb_body_name} (id={perturb_body_id}), "
          f"nbody={nbody}")

    class PerturbationWrapper:
        """Applies sustained random external force on a body during a mid-episode window.

        On reset: samples a random 3D force direction and coin-flips whether
        this episode is perturbed.  On step: sets xfrc_applied on the target
        body when inside the [start, end) window.
        """

        def __init__(self, env, ep_length, start_frac=0.3, end_frac=0.6,
                     scale=0.3, prob=0.5):
            self._env = env
            self._start = int(ep_length * start_frac)
            self._end = int(ep_length * end_frac)
            self._scale = scale
            self._prob = prob

        def reset(self, rng):
            rng, dir_rng, coin_rng = jax.random.split(rng, 3)
            state = self._env.reset(rng)
            # Random 3D force direction, normalized then scaled
            force_dir = jax.random.normal(dir_rng, (3,))
            force_dir = force_dir / (jp.linalg.norm(force_dir) + 1e-8)
            active = (jax.random.uniform(coin_rng) < self._prob).astype(
                jp.float32
            )
            new_info = {
                **state.info,
                'perturb_force_3d': force_dir * self._scale,  # (3,) Newtons
                'perturb_active': active,
                'perturb_step': jp.zeros(()),
            }
            return state.replace(info=new_info)

        def step(self, state, action):
            t = state.info['perturb_step']
            in_window = ((t >= self._start) & (t < self._end)).astype(
                jp.float32
            )
            mask = in_window * state.info['perturb_active']
            # Build xfrc_applied: (nbody, 6) — force_xyz + torque_xyz
            force_3d = state.info['perturb_force_3d'] * mask  # (3,)
            xfrc = jp.zeros((nbody, 6))
            xfrc = xfrc.at[perturb_body_id, :3].set(force_3d)
            # Inject into data before stepping
            new_data = state.data.replace(xfrc_applied=xfrc)
            new_state = state.replace(data=new_data)
            next_state = self._env.step(new_state, action)
            new_info = {
                **next_state.info,
                'perturb_force_3d': state.info['perturb_force_3d'],
                'perturb_active': state.info['perturb_active'],
                'perturb_step': t + 1,
            }
            return next_state.replace(info=new_info)

        @property
        def observation_size(self):
            return self._env.observation_size

        @property
        def action_size(self):
            return self._env.action_size

        @property
        def dt(self):
            return self._env.dt

        def __getattr__(self, name):
            return getattr(self._env, name)

    wrapped_env = flatten_obs_wrapper(env)
    wrapped_eval_env = flatten_obs_wrapper(eval_env_base)

    # Perturbation wrapper — training env only
    perturbed_train_env = PerturbationWrapper(
        wrapped_env, episode_length,
        start_frac=nf.perturb_start_frac,
        end_frac=nf.perturb_end_frac,
        scale=nf.perturb_scale,
        prob=nf.perturb_prob,
    )
    perturb_start = int(episode_length * nf.perturb_start_frac)
    perturb_end = int(episode_length * nf.perturb_end_frac)
    print(f"Perturbation window: steps [{perturb_start}, {perturb_end}), "
          f"scale={nf.perturb_scale}, prob={nf.perturb_prob}")

    wrap_fn = functools.partial(wrapper.wrap_for_brax_training, full_reset=True)
    train_env = wrap_fn(perturbed_train_env, episode_length=episode_length,
                        action_repeat=ppo_params.action_repeat)
    eval_env = wrap_fn(wrapped_eval_env, episode_length=episode_length,
                       action_repeat=ppo_params.action_repeat)

    # Networks
    action_dist = distribution.NormalTanhDistribution(event_size=act_size)
    param_size = action_dist.param_size

    policy_module = LRNCerebellumPolicy(
        propriospinal_size=nf.propriospinal_size,
        exc_ratio=nf.propriospinal_exc_ratio,
        lrn_size=nf.lrn_size,
        motor_neuron_size=nf.motor_neuron_size,
        cerebellum_state_dim=nf.cerebellum_state_dim,
        n_micro_steps=nf.n_micro_steps,
        tau_min=nf.tau_min, tau_max=nf.tau_max,
        v_th=nf.v_th, v_reset=nf.v_reset,
        beta_surrogate=nf.beta_surrogate,
        n_refractory=nf.n_refractory,
        output_size=param_size,
    )
    diag_policy_module = DiagnosticLRNCerebellumPolicy(
        propriospinal_size=nf.propriospinal_size,
        exc_ratio=nf.propriospinal_exc_ratio,
        lrn_size=nf.lrn_size,
        motor_neuron_size=nf.motor_neuron_size,
        cerebellum_state_dim=nf.cerebellum_state_dim,
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

    # Carry dimension
    carry_dim = (
        2 * nf.propriospinal_size   # ps_v + ps_r
        + 2 * nf.lrn_size           # lrn_v + lrn_r
        + 2 * nf.motor_neuron_size  # mn_v + mn_r
        + nf.cerebellum_state_dim   # x_hat (no P_diag with nonlinear world model)
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
        pred_coef=ppo_params.pred_coef,
        correction_coef=ppo_params.correction_coef,
        act_size=act_size,
        obs_size=obs_size,
        num_envs=num_envs,
    )

    train_step_fn = make_training_fns(
        train_env, policy_module, value_module, action_dist, optimizer, train_config
    )
    eval_fn = make_eval_fn(
        eval_env, diag_policy_module, action_dist, episode_length, carry_dim
    )
    perturbed_eval_fn = make_perturbed_eval_fn(
        eval_env, diag_policy_module, action_dist, episode_length, carry_dim,
        start_frac=nf.perturb_start_frac,
        end_frac=nf.perturb_end_frac,
        scale=nf.perturb_scale,
        perturb_body_id=perturb_body_id,
    )

    # JIT'd helpers for video rollout on raw eval_env_base (single-env, unbatched)
    jit_vid_reset = jax.jit(eval_env_base.reset)
    jit_vid_step = jax.jit(eval_env_base.step)
    jit_vid_policy = jax.jit(policy_module.apply)

    # nbody for video rollout xfrc_applied (unbatched env)
    vid_nbody = eval_env_base.mj_model.nbody
    vid_perturb_body_id = eval_env_base.mj_model.body(nf.perturb_body).id

    def video_rollout(policy_params, normalizer_params, seed=0,
                       perturb_force_3d=None):
        """Video rollout with optional external force perturbation.

        Args:
            perturb_force_3d: if not None, (3,) force vector (Newtons) applied
                              to perturb body during [perturb_start, perturb_end).
        """
        rng = jax.random.PRNGKey(seed)
        state = jit_vid_reset(rng)
        carry = jp.zeros((1, carry_dim))
        rollout_states = [state]
        for t in range(episode_length):
            flat_obs = flatten_obs(state.obs)
            obs_norm = running_statistics.normalize(
                flat_obs[None], normalizer_params
            )
            logits, new_carry, _, _ = jit_vid_policy(
                policy_params, obs_norm, carry
            )
            action = jp.squeeze(action_dist.mode(logits), axis=0)
            if perturb_force_3d is not None and perturb_start <= t < perturb_end:
                xfrc = jp.zeros((vid_nbody, 6))
                xfrc = xfrc.at[vid_perturb_body_id, :3].set(perturb_force_3d)
                state = state.replace(data=state.data.replace(xfrc_applied=xfrc))
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

            # Perturbed diagnostic eval
            key, perturbed_eval_key = jax.random.split(key)
            perturbed_data = perturbed_eval_fn(
                state.policy_params, state.normalizer_params,
                perturbed_eval_key,
            )

            # Plot circuit diagnostics (clean, with perturbation window markers)
            fig = plot_circuit_diagnostics(
                episode_data, v_th=nf.v_th,
                save_path=f"{ckpt_path}/{state.total_steps}_circuit.png",
                perturb_start=perturb_start, perturb_end=perturb_end,
            )
            if USE_WANDB:
                wandb.log({"eval/circuit": wandb.Image(fig)}, commit=False)
            plt.close(fig)

            # Plot perturbation comparison
            fig2 = plot_perturbation_comparison(
                episode_data, perturbed_data,
                perturb_start, perturb_end,
                save_path=f"{ckpt_path}/{state.total_steps}_perturb.png",
            )
            if USE_WANDB:
                wandb.log({"eval/perturbation": wandb.Image(fig2)},
                          commit=False)
            plt.close(fig2)

            # Cerebellum hypothesis test
            fig_hyp, hyp_metrics = run_cerebellum_hypothesis_test(
                episode_data, perturbed_data,
                perturb_start, perturb_end,
                episode_length,
                save_path=f"{ckpt_path}/{state.total_steps}_hypothesis.png",
            )
            if USE_WANDB:
                # Log hypothesis figure
                wandb.log({"eval/hypothesis_test": wandb.Image(fig_hyp)},
                          commit=False)
                # Log all scalar hypothesis metrics
                wandb.log(
                    {k: v for k, v in hyp_metrics.items()
                     if isinstance(v, (int, float))},
                    commit=False,
                )
            dom = hyp_metrics.get('hypothesis/dominant_phase', 'N/A')
            dom_diff = hyp_metrics.get('hypothesis/dominant_phase_diff', 0)
            print(f"  hypothesis: dominant phase = {dom} "
                  f"(diff={dom_diff:.4f})")
            for phase in ['initial_reach', 'perturbation', 'post_perturbation']:
                pv = hyp_metrics.get(f'hypothesis/{phase}/p_value', None)
                if pv is not None:
                    sig = "***" if pv < 0.001 else "**" if pv < 0.01 else \
                          "*" if pv < 0.05 else "ns"
                    print(f"    {phase}: p={pv:.4f} {sig}")
            plt.close(fig_hyp)

            # Log summary metrics
            rewards = np.array(episode_data['reward'])
            innov = np.array(episode_data['cb_innovation_norm'])
            p_innov = np.array(perturbed_data['cb_innovation_norm'])
            mn_rates = np.array(episode_data['motor_spike_rate'])
            eval_metrics = {
                "eval/episode_reward": float(np.sum(rewards)),
                "eval/mean_innovation": float(np.mean(innov)),
                "eval/perturbed_mean_innovation": float(np.mean(p_innov)),
                "eval/mn_mean_spike_rate": float(np.mean(mn_rates)),
            }
            if USE_WANDB:
                wandb.log(eval_metrics, commit=False)
            print(f"  eval reward: {eval_metrics['eval/episode_reward']:.1f}")
            print(f"  mean innovation: {eval_metrics['eval/mean_innovation']:.4f}")
            print(f"  perturbed innovation: "
                  f"{eval_metrics['eval/perturbed_mean_innovation']:.4f}")

            # Video rollout: side-by-side clean vs perturbed
            try:
                # Clean rollout
                clean_states = video_rollout(
                    state.policy_params, state.normalizer_params,
                    seed=state.total_steps,
                )
                # Perturbed rollout (same seed for same initial state)
                p_rng = jax.random.PRNGKey(state.total_steps + 1)
                p_dir = jax.random.normal(p_rng, (3,))
                p_force_3d = np.array(
                    p_dir / (jp.linalg.norm(p_dir) + 1e-8) * nf.perturb_scale
                )
                pert_states = video_rollout(
                    state.policy_params, state.normalizer_params,
                    seed=state.total_steps,
                    perturb_force_3d=p_force_3d,
                )

                fps = int(1.0 / eval_env_base.dt)
                clean_frames = eval_env_base.render(
                    clean_states, height=512, width=512, render_ghost=True
                )
                pert_frames = eval_env_base.render(
                    pert_states, height=512, width=512, render_ghost=True
                )

                # Stitch side-by-side with labels
                import cv2
                video_path = f"{ckpt_path}/{state.total_steps}.mp4"
                n_frames = min(len(clean_frames), len(pert_frames))
                with imageio.get_writer(video_path, fps=fps) as vid:
                    for i in range(n_frames):
                        left = np.array(clean_frames[i])
                        right = np.array(pert_frames[i])
                        # Add labels
                        cv2.putText(left, "Clean", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (255, 255, 255), 2)
                        label = "Perturbed" if perturb_start <= i < perturb_end else "Perturbed (off)"
                        color = (0, 0, 255) if perturb_start <= i < perturb_end else (255, 255, 255)
                        cv2.putText(right, label, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    color, 2)
                        grid = np.concatenate([left, right], axis=1)
                        vid.append_data(grid)

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
