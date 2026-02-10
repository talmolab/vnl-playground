"""
Training script for mouse arm imitation with recurrent Izhikevich spiking policy.

Izhikevich neurons offer rich dynamics (bursting, adaptation, resonance) with
only 2 state variables per neuron, making them ~10-20x faster than Hodgkin-Huxley
while capturing similar phenomenology.

Reference: Izhikevich (2003) "Simple Model of Spiking Neurons"
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
import math
import numpy as np
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Any, NamedTuple, Sequence, Tuple

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
# Izhikevich Neuron Model
# =============================================================================

# Standard parameter sets from Izhikevich (2003)
# Each gives different firing patterns
IZHIKEVICH_PRESETS = {
    # (a, b, c, d) parameters
    "regular_spiking": (0.02, 0.2, -65.0, 8.0),      # Most common cortical
    "intrinsic_bursting": (0.02, 0.2, -55.0, 4.0),   # Bursts then regular
    "chattering": (0.02, 0.2, -50.0, 2.0),           # Fast rhythmic bursting
    "fast_spiking": (0.1, 0.2, -65.0, 2.0),          # Interneuron-like
    "low_threshold": (0.02, 0.25, -65.0, 2.0),       # Thalamic
    "resonator": (0.1, 0.26, -65.0, 2.0),            # Subthreshold oscillations
    "mixed": None,  # Randomly sample from above
}


class IzhikevichLayer(linen.Module):
    """
    Izhikevich neuron layer with learnable parameters.
    
    Dynamics:
        dv/dt = 0.04*v^2 + 5*v + 140 - u + I
        du/dt = a*(b*v - u)
        
        if v >= 30: v = c, u = u + d
    
    Parameters a, b, c, d control firing pattern:
        a: recovery time scale (smaller = slower recovery)
        b: sensitivity of u to v (controls subthreshold dynamics)
        c: post-spike reset voltage
        d: post-spike recovery increment
    
    Args:
        n_neurons: Number of neurons.
        n_steps: Integration steps per forward pass.
        dt: Timestep (ms). Can use larger dt than HH (0.5-1.0 ok).
        learn_dynamics: If True, learn a, b, c, d per neuron.
        preset: One of IZHIKEVICH_PRESETS keys, or "mixed" for variety.
        current_scale: Scaling factor for input currents.
    """
    
    n_neurons: int
    n_steps: int = 20
    dt: float = 0.5
    learn_dynamics: bool = True
    preset: str = "regular_spiking"
    current_scale: float = 15.0
    spike_threshold: float = 30.0
    
    def setup(self):
        if self.learn_dynamics:
            # Initialize from preset, but make learnable
            if self.preset == "mixed":
                # Random mix of neuron types
                presets = ["regular_spiking", "fast_spiking", "intrinsic_bursting", 
                          "chattering", "low_threshold"]
                # Will be overridden by learned params anyway
                a0, b0, c0, d0 = IZHIKEVICH_PRESETS["regular_spiking"]
            else:
                a0, b0, c0, d0 = IZHIKEVICH_PRESETS[self.preset]
            
            # Parameterize with constraints:
            # a, b > 0 (use softplus)
            # c < 0 (use -softplus)
            # d > 0 (use softplus)
            self.a_raw = self.param('a_raw', 
                lambda k, s: jp.full(s, self._inverse_softplus(a0)), (self.n_neurons,))
            self.b_raw = self.param('b_raw',
                lambda k, s: jp.full(s, self._inverse_softplus(b0)), (self.n_neurons,))
            self.c_raw = self.param('c_raw',
                lambda k, s: jp.full(s, self._inverse_softplus(-c0)), (self.n_neurons,))
            self.d_raw = self.param('d_raw',
                lambda k, s: jp.full(s, self._inverse_softplus(d0)), (self.n_neurons,))
    
    def _inverse_softplus(self, y):
        """Inverse of softplus for initialization."""
        if y <= 0:
            return -10.0  # Will give ~0 after softplus
        return math.log(math.exp(y) - 1)
    
    def get_params(self):
        """Get constrained parameters."""
        if self.learn_dynamics:
            a = jax.nn.softplus(self.a_raw)
            b = jax.nn.softplus(self.b_raw)
            c = -jax.nn.softplus(self.c_raw)  # Negative reset voltage
            d = jax.nn.softplus(self.d_raw)
        else:
            a0, b0, c0, d0 = IZHIKEVICH_PRESETS[self.preset]
            a = jp.full((self.n_neurons,), a0)
            b = jp.full((self.n_neurons,), b0)
            c = jp.full((self.n_neurons,), c0)
            d = jp.full((self.n_neurons,), d0)
        return a, b, c, d
    
    @linen.compact
    def __call__(self, x, carry=None):
        """
        Forward pass.
        
        Args:
            x: Input, shape (batch, input_dim).
            carry: Optional (v, u) tuple, each (batch, n_neurons).
        
        Returns:
            spike_rate: Output rate, shape (batch, n_neurons).
            new_carry: Updated (v, u) state.
        """
        batch_size = x.shape[0]
        
        # Get (possibly learned) parameters
        a, b, c, d = self.get_params()
        
        # Project input to per-neuron current
        I = linen.Dense(
            self.n_neurons,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            name="current_proj",
        )(x) * self.current_scale
        
        # Initialize state if not provided
        if carry is None:
            v = jp.full((batch_size, self.n_neurons), -65.0)
            u = b * v  # Steady-state approximation
        else:
            v, u = carry
        
        v_th = self.spike_threshold
        dt = self.dt
        
        def step_fn(state, _):
            v, u, spike_count = state
            
            # Izhikevich dynamics (can use 0.5ms steps)
            # dv = 0.04*v^2 + 5*v + 140 - u + I
            # du = a*(b*v - u)
            dv = 0.04 * v * v + 5.0 * v + 140.0 - u + I
            du = a * (b * v - u)
            
            v_new = v + dt * dv
            u_new = u + dt * du
            
            # Spike detection and reset
            spiked = (v_new >= v_th).astype(jp.float32)
            
            # Reset: v -> c, u -> u + d
            v_new = jp.where(spiked > 0.5, c, v_new)
            u_new = jp.where(spiked > 0.5, u_new + d, u_new)
            
            # Clamp v to avoid numerical issues
            v_new = jp.clip(v_new, -100.0, v_th)
            
            return (v_new, u_new, spike_count + spiked), spiked
        
        (v_final, u_final, spike_count), spikes = jax.lax.scan(
            step_fn,
            (v, u, jp.zeros((batch_size, self.n_neurons))),
            None,
            length=self.n_steps,
        )
        
        # Output: normalized spike rate
        spike_rate = spike_count / self.n_steps
        
        return spike_rate, (v_final, u_final)


class IzhikevichLayerWithSurrogate(linen.Module):
    """
    Izhikevich layer with surrogate gradient for spike detection.
    
    Uses STE to allow gradient flow through the threshold crossing.
    This can help with learning compared to hard threshold.
    """
    
    n_neurons: int
    n_steps: int = 20
    dt: float = 0.5
    learn_dynamics: bool = True
    preset: str = "regular_spiking"
    current_scale: float = 15.0
    spike_threshold: float = 30.0
    surrogate_beta: float = 5.0
    
    def setup(self):
        if self.learn_dynamics:
            a0, b0, c0, d0 = IZHIKEVICH_PRESETS.get(
                self.preset, IZHIKEVICH_PRESETS["regular_spiking"]
            )
            self.a_raw = self.param('a_raw',
                lambda k, s: jp.full(s, self._inv_sp(a0)), (self.n_neurons,))
            self.b_raw = self.param('b_raw',
                lambda k, s: jp.full(s, self._inv_sp(b0)), (self.n_neurons,))
            self.c_raw = self.param('c_raw',
                lambda k, s: jp.full(s, self._inv_sp(-c0)), (self.n_neurons,))
            self.d_raw = self.param('d_raw',
                lambda k, s: jp.full(s, self._inv_sp(d0)), (self.n_neurons,))
    
    def _inv_sp(self, y):
        if y <= 0:
            return -10.0
        return math.log(math.exp(y) - 1)
    
    def get_params(self):
        if self.learn_dynamics:
            a = jax.nn.softplus(self.a_raw)
            b = jax.nn.softplus(self.b_raw)
            c = -jax.nn.softplus(self.c_raw)
            d = jax.nn.softplus(self.d_raw)
        else:
            a0, b0, c0, d0 = IZHIKEVICH_PRESETS[self.preset]
            a = jp.full((self.n_neurons,), a0)
            b = jp.full((self.n_neurons,), b0)
            c = jp.full((self.n_neurons,), c0)
            d = jp.full((self.n_neurons,), d0)
        return a, b, c, d
    
    @linen.compact
    def __call__(self, x, carry=None):
        batch_size = x.shape[0]
        a, b, c, d = self.get_params()
        
        I = linen.Dense(
            self.n_neurons,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            name="current_proj",
        )(x) * self.current_scale
        
        if carry is None:
            v = jp.full((batch_size, self.n_neurons), -65.0)
            u = b * v
        else:
            v, u = carry
        
        v_th = self.spike_threshold
        dt = self.dt
        beta = self.surrogate_beta
        
        def step_fn(state, _):
            v, u, spike_count = state
            
            dv = 0.04 * v * v + 5.0 * v + 140.0 - u + I
            du = a * (b * v - u)
            
            v_new = v + dt * dv
            u_new = u + dt * du
            
            # Surrogate gradient spike detection
            spike_hard = (v_new >= v_th).astype(jp.float32)
            spike_soft = jax.nn.sigmoid(beta * (v_new - v_th))
            spike = jax.lax.stop_gradient(spike_hard - spike_soft) + spike_soft
            
            # Reset with soft spike (gradient flows through)
            v_new = v_new * (1.0 - spike) + c * spike
            u_new = u_new + d * spike
            
            v_new = jp.clip(v_new, -100.0, v_th)
            
            return (v_new, u_new, spike_count + spike), spike
        
        (v_final, u_final, spike_count), spikes = jax.lax.scan(
            step_fn,
            (v, u, jp.zeros((batch_size, self.n_neurons))),
            None,
            length=self.n_steps,
        )
        
        spike_rate = spike_count / self.n_steps
        return spike_rate, (v_final, u_final)


# =============================================================================
# Diagnostic Layer (returns full traces)
# =============================================================================

class DiagnosticIzhikevichLayer(linen.Module):
    """Izhikevich layer that returns voltage/recovery traces for visualization."""
    
    n_neurons: int
    n_steps: int = 20
    dt: float = 0.5
    learn_dynamics: bool = True
    preset: str = "regular_spiking"
    current_scale: float = 15.0
    spike_threshold: float = 30.0
    
    def setup(self):
        if self.learn_dynamics:
            a0, b0, c0, d0 = IZHIKEVICH_PRESETS.get(
                self.preset, IZHIKEVICH_PRESETS["regular_spiking"]
            )
            self.a_raw = self.param('a_raw',
                lambda k, s: jp.full(s, self._inv_sp(a0)), (self.n_neurons,))
            self.b_raw = self.param('b_raw',
                lambda k, s: jp.full(s, self._inv_sp(b0)), (self.n_neurons,))
            self.c_raw = self.param('c_raw',
                lambda k, s: jp.full(s, self._inv_sp(-c0)), (self.n_neurons,))
            self.d_raw = self.param('d_raw',
                lambda k, s: jp.full(s, self._inv_sp(d0)), (self.n_neurons,))
    
    def _inv_sp(self, y):
        if y <= 0:
            return -10.0
        return math.log(math.exp(y) - 1)
    
    def get_params(self):
        if self.learn_dynamics:
            return (jax.nn.softplus(self.a_raw),
                    jax.nn.softplus(self.b_raw),
                    -jax.nn.softplus(self.c_raw),
                    jax.nn.softplus(self.d_raw))
        else:
            a0, b0, c0, d0 = IZHIKEVICH_PRESETS[self.preset]
            return (jp.full((self.n_neurons,), a0),
                    jp.full((self.n_neurons,), b0),
                    jp.full((self.n_neurons,), c0),
                    jp.full((self.n_neurons,), d0))
    
    @linen.compact
    def __call__(self, x, carry=None):
        batch_size = x.shape[0]
        a, b, c, d = self.get_params()
        
        I = linen.Dense(
            self.n_neurons,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            name="current_proj",
        )(x) * self.current_scale
        
        if carry is None:
            v = jp.full((batch_size, self.n_neurons), -65.0)
            u = b * v
        else:
            v, u = carry
        
        v_th = self.spike_threshold
        dt = self.dt
        
        def step_fn(state, _):
            v, u = state
            
            dv = 0.04 * v * v + 5.0 * v + 140.0 - u + I
            du = a * (b * v - u)
            
            v_new = v + dt * dv
            u_new = u + dt * du
            
            spiked = (v_new >= v_th).astype(jp.float32)
            v_new = jp.where(spiked > 0.5, c, v_new)
            u_new = jp.where(spiked > 0.5, u_new + d, u_new)
            v_new = jp.clip(v_new, -100.0, v_th)
            
            return (v_new, u_new), (spiked, v_new, u_new)
        
        (v_final, u_final), (spikes, v_trace, u_trace) = jax.lax.scan(
            step_fn, (v, u), None, length=self.n_steps
        )
        
        spike_rate = jp.mean(spikes, axis=0)
        
        diagnostics = {
            "spikes": spikes,      # (n_steps, batch, n_neurons)
            "voltages": v_trace,   # (n_steps, batch, n_neurons)
            "recovery": u_trace,   # (n_steps, batch, n_neurons)
        }
        
        return spike_rate, (v_final, u_final), diagnostics


# =============================================================================
# Recurrent Izhikevich Policy
# =============================================================================

class RecurrentIzhikevichPolicy(linen.Module):
    """
    Recurrent policy using Izhikevich neurons with persistent state.
    
    Args:
        layer_sizes: Hidden layer sizes.
        output_size: Action distribution param size.
        n_steps: Integration steps per env step.
        dt: Timestep (ms).
        learn_dynamics: If True, learn a, b, c, d parameters.
        preset: Neuron type preset.
        use_surrogate: If True, use surrogate gradient for spikes.
    """
    
    layer_sizes: Sequence[int]
    output_size: int
    n_steps: int = 20
    dt: float = 0.5
    learn_dynamics: bool = True
    preset: str = "regular_spiking"
    use_surrogate: bool = True
    
    @linen.compact
    def __call__(self, obs, carry_flat=None):
        """
        Forward pass.
        
        Args:
            obs: Observation, shape (batch, obs_dim).
            carry_flat: Flattened (v, u) state for all layers.
                       Shape (batch, 2 * sum(layer_sizes)).
        
        Returns:
            logits: Action distribution parameters.
            new_carry_flat: Updated state.
        """
        batch_size = obs.shape[0]
        total_neurons = sum(self.layer_sizes)
        
        # Split carry into per-layer (v, u) pairs
        if carry_flat is None:
            carries = [None] * len(self.layer_sizes)
        else:
            # carry_flat is (batch, 2 * total_neurons)
            # First half is v, second half is u
            v_all = carry_flat[:, :total_neurons]
            u_all = carry_flat[:, total_neurons:]
            
            splits = list(np.cumsum(self.layer_sizes[:-1]))
            v_splits = jp.split(v_all, splits, axis=-1)
            u_splits = jp.split(u_all, splits, axis=-1)
            carries = [(v, u) for v, u in zip(v_splits, u_splits)]
        
        LayerClass = IzhikevichLayerWithSurrogate if self.use_surrogate else IzhikevichLayer
        
        x = obs
        new_v_list = []
        new_u_list = []
        
        for i, n_neurons in enumerate(self.layer_sizes):
            layer = LayerClass(
                n_neurons=n_neurons,
                n_steps=self.n_steps,
                dt=self.dt,
                learn_dynamics=self.learn_dynamics,
                preset=self.preset,
                name=f"izh_{i}",
            )
            x, (v_new, u_new) = layer(x, carries[i])
            new_v_list.append(v_new)
            new_u_list.append(u_new)
        
        logits = linen.Dense(
            self.output_size,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            name="readout",
        )(x)
        
        # Flatten carry back
        new_carry_flat = jp.concatenate(
            new_v_list + new_u_list, axis=-1
        )
        
        return logits, new_carry_flat


class DiagnosticIzhikevichPolicy(linen.Module):
    """Same as RecurrentIzhikevichPolicy but returns diagnostics."""
    
    layer_sizes: Sequence[int]
    output_size: int
    n_steps: int = 20
    dt: float = 0.5
    learn_dynamics: bool = True
    preset: str = "regular_spiking"
    
    @linen.compact
    def __call__(self, obs, carry_flat=None):
        batch_size = obs.shape[0]
        total_neurons = sum(self.layer_sizes)
        
        if carry_flat is None:
            carries = [None] * len(self.layer_sizes)
        else:
            v_all = carry_flat[:, :total_neurons]
            u_all = carry_flat[:, total_neurons:]
            splits = list(np.cumsum(self.layer_sizes[:-1]))
            v_splits = jp.split(v_all, splits, axis=-1)
            u_splits = jp.split(u_all, splits, axis=-1)
            carries = [(v, u) for v, u in zip(v_splits, u_splits)]
        
        x = obs
        new_v_list = []
        new_u_list = []
        all_diagnostics = {}
        
        for i, n_neurons in enumerate(self.layer_sizes):
            layer = DiagnosticIzhikevichLayer(
                n_neurons=n_neurons,
                n_steps=self.n_steps,
                dt=self.dt,
                learn_dynamics=self.learn_dynamics,
                preset=self.preset,
                name=f"izh_{i}",
            )
            x, (v_new, u_new), diag = layer(x, carries[i])
            new_v_list.append(v_new)
            new_u_list.append(u_new)
            all_diagnostics[f"izh_{i}"] = diag
        
        logits = linen.Dense(
            self.output_size,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            name="readout",
        )(x)
        
        new_carry_flat = jp.concatenate(new_v_list + new_u_list, axis=-1)
        
        return logits, new_carry_flat, all_diagnostics


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

def plot_izhikevich_diagnostics(layer_diag, env_step_idx, layer_names, save_path=None):
    """
    Plot spike rasters, voltage traces, and recovery variable for Izhikevich layers.
    """
    n_layers = len(layer_diag)
    fig, axes = plt.subplots(n_layers, 4, figsize=(18, 4 * n_layers), squeeze=False)
    
    for row, name in enumerate(layer_names):
        diag = layer_diag[name]
        spikes = np.array(diag["spikes"][:, 0, :])    # (K, N) first batch
        voltages = np.array(diag["voltages"][:, 0, :])
        recovery = np.array(diag["recovery"][:, 0, :])
        K, N = spikes.shape
        
        # Raster
        ax = axes[row, 0]
        t_idx, n_idx = np.where(spikes > 0.5)
        n_show = min(N, 100)
        mask = n_idx < n_show
        ax.scatter(t_idx[mask], n_idx[mask], s=2, c="black", marker="|")
        ax.set_xlim(-0.5, K - 0.5)
        ax.set_ylim(-0.5, n_show - 0.5)
        ax.set_xlabel("step")
        ax.set_ylabel("neuron")
        ax.set_title(f"{name} raster (t={env_step_idx})")
        
        # Voltage traces
        ax = axes[row, 1]
        for j in np.linspace(0, N - 1, min(8, N), dtype=int):
            ax.plot(voltages[:, j], alpha=0.7, linewidth=1)
        ax.axhline(30.0, color="red", ls="--", alpha=0.5, label="threshold")
        ax.set_xlabel("step")
        ax.set_ylabel("v (mV)")
        ax.set_title(f"{name} voltages")
        ax.legend(fontsize=7)
        
        # Recovery variable traces
        ax = axes[row, 2]
        for j in np.linspace(0, N - 1, min(8, N), dtype=int):
            ax.plot(recovery[:, j], alpha=0.7, linewidth=1)
        ax.set_xlabel("step")
        ax.set_ylabel("u")
        ax.set_title(f"{name} recovery")
        
        # Rate histogram
        ax = axes[row, 3]
        rates = np.mean(spikes, axis=0)
        ax.hist(rates, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(np.mean(rates), color="red", ls="--",
                   label=f"mean={np.mean(rates):.3f}")
        ax.set_xlabel("spike rate")
        ax.set_ylabel("count")
        ax.set_title(f"{name} rate dist")
        ax.legend(fontsize=7)
    
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_learned_parameters(policy_params, layer_names, save_path=None):
    """Visualize the learned Izhikevich parameters across neurons."""
    fig, axes = plt.subplots(len(layer_names), 4, figsize=(16, 3 * len(layer_names)))
    if len(layer_names) == 1:
        axes = axes[None, :]
    
    param_names = ['a', 'b', 'c', 'd']
    
    for row, name in enumerate(layer_names):
        layer_params = policy_params['params'][name]
        
        a = jax.nn.softplus(np.array(layer_params['a_raw']))
        b = jax.nn.softplus(np.array(layer_params['b_raw']))
        c = -jax.nn.softplus(np.array(layer_params['c_raw']))
        d = jax.nn.softplus(np.array(layer_params['d_raw']))
        
        for col, (param, pname) in enumerate(zip([a, b, c, d], param_names)):
            ax = axes[row, col]
            ax.hist(param, bins=30, edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(param), color='red', ls='--',
                      label=f'mean={np.mean(param):.3f}')
            ax.set_xlabel(pname)
            ax.set_ylabel('count')
            ax.set_title(f'{name} {pname}')
            ax.legend(fontsize=7)
    
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
    return fig


# =============================================================================
# Config
# =============================================================================

ppo_params = config_dict.create(
    num_timesteps=500_000_000,
    num_evals=50,
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
        policy_hidden_layer_sizes=(25, 25),
        value_hidden_layer_sizes=(25, 25, 25),
        n_steps=20,           # Integration steps per env step
        dt=0.5,               # 0.5ms timestep (10ms total per env step)
        learn_dynamics=True,  # Learn a, b, c, d parameters
        preset="regular_spiking",
        use_surrogate=True,
    ),
)

env_name = "mouse-imitation-izhikevich"
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
               id=f"izhikevich-{exp_name}")
    wandb.config.update({
        "env_name": env_name,
        "policy_type": "recurrent_izhikevich",
        **dict(ppo_params.network_factory),
    })


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

            total = policy_loss + vf_coef * value_loss - entropy_cost * entropy
            return total, {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "approx_kl": jp.mean((ratio - 1.0) - jp.log(ratio)),
            }

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
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


def make_eval_fn(eval_env, diag_policy_module, action_dist, episode_length, carry_dim):
    """Create JIT-compiled evaluation function."""
    
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
            # Add per-layer diagnostics
            for layer_name, diag in diagnostics.items():
                step_data[f'{layer_name}_spikes'] = diag['spikes'][:, 0, :]
                step_data[f'{layer_name}_voltages'] = diag['voltages'][:, 0, :]
                step_data[f'{layer_name}_recovery'] = diag['recovery'][:, 0, :]
            
            return (next_env_state, new_carry, rng), step_data
        
        _, episode_data = jax.lax.scan(
            step_fn, (env_state, carry, rng), None, length=episode_length
        )
        return episode_data
    
    return eval_episode


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Mouse Arm Imitation -- Recurrent Izhikevich Policy")
    print("=" * 80)
    nf = ppo_params.network_factory
    print(f"Izhikevich: n_steps={nf.n_steps}, dt={nf.dt}ms, "
          f"preset={nf.preset}, learn_dynamics={nf.learn_dynamics}")
    print(f"Policy layers: {nf.policy_hidden_layer_sizes}")
    
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
    
    policy_module = RecurrentIzhikevichPolicy(
        layer_sizes=nf.policy_hidden_layer_sizes,
        output_size=param_size,
        n_steps=nf.n_steps,
        dt=nf.dt,
        learn_dynamics=nf.learn_dynamics,
        preset=nf.preset,
        use_surrogate=nf.use_surrogate,
    )
    diag_policy_module = DiagnosticIzhikevichPolicy(
        layer_sizes=nf.policy_hidden_layer_sizes,
        output_size=param_size,
        n_steps=nf.n_steps,
        dt=nf.dt,
        learn_dynamics=nf.learn_dynamics,
        preset=nf.preset,
    )
    value_module = networks.MLP(
        layer_sizes=list(nf.value_hidden_layer_sizes) + [1],
        activation=linen.swish,
        kernel_init=jax.nn.initializers.lecun_uniform(),
    )
    
    # Carry dimension: 2 * sum(layer_sizes) for (v, u) per neuron
    carry_dim = 2 * sum(nf.policy_hidden_layer_sizes)
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
    
    layer_names = [f"izh_{i}" for i in range(len(nf.policy_hidden_layer_sizes))]
    
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
            
            # Evaluation
            print(f"[eval] Running evaluation at step {state.total_steps}...")
            key, eval_key = jax.random.split(state.rng)
            state = state.replace(rng=key)
            
            episode_data = eval_fn(state.policy_params, state.normalizer_params, eval_key)
            
            # Plot diagnostics
            for t_idx in [0, episode_length // 2, episode_length - 1]:
                diag_at_t = {}
                for name in layer_names:
                    diag_at_t[name] = {
                        'spikes': np.array(episode_data[f'{name}_spikes'][t_idx])[:, None, :],
                        'voltages': np.array(episode_data[f'{name}_voltages'][t_idx])[:, None, :],
                        'recovery': np.array(episode_data[f'{name}_recovery'][t_idx])[:, None, :],
                    }
                
                fig = plot_izhikevich_diagnostics(
                    diag_at_t, t_idx, layer_names,
                    save_path=f"{ckpt_path}/{state.total_steps}_izh_t{t_idx}.png",
                )
                if USE_WANDB:
                    wandb.log({f"eval/izh_t{t_idx}": wandb.Image(fig)}, commit=False)
                plt.close(fig)
            
            # Plot learned parameters
            fig = plot_learned_parameters(
                state.policy_params, layer_names,
                save_path=f"{ckpt_path}/{state.total_steps}_params.png",
            )
            if USE_WANDB:
                wandb.log({"eval/learned_params": wandb.Image(fig)}, commit=False)
            plt.close(fig)
            
            # Log spike rate summary
            for name in layer_names:
                rates = np.mean(np.array(episode_data[f'{name}_spikes']), axis=(0, 1))
                if USE_WANDB:
                    wandb.log({
                        f"eval/{name}_mean_rate": float(np.mean(rates)),
                        f"eval/{name}_std_rate": float(np.std(rates)),
                    }, commit=False)
            
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

            # Save checkpoint
            params_to_save = (state.normalizer_params, state.policy_params, state.value_params)
            orbax_checkpointer = ocp.PyTreeCheckpointer()
            save_args = orbax_utils.save_args_from_target(params_to_save)
            path = ckpt_path / f"{state.total_steps}"
            orbax_checkpointer.save(path, params_to_save, force=True, save_args=save_args)
            print(f"  checkpoint -> {path}")
            
            next_eval_at += steps_per_eval
    
    print("=" * 80)
    print("Training complete!")
    print("=" * 80)
