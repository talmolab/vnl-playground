"""
Training script for mouse arm imitation with spike-propagating LIF policy.

Unlike the rate-based model (train_mouse_LIF.py) where each layer independently
runs micro-steps and passes mean spike rates to the next layer, this script uses
a single outer scan: at each micro-step, actual binary spikes propagate from
layer to layer. Surrogate gradients enable backpropagation through the spikes.

Features:
- E/I populations (Dale's law, 80/20 split) with lateral connections
- Spike propagation between layers (not rate-coded)
- Refractory period (configurable micro-steps)
- Heterogeneous membrane time constants (log-uniform, fixed)
- Persistent membrane + refractory state across environment steps

The value network is a standard MLP (no spiking, no carry).
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
from typing import Any, Literal, Mapping, NamedTuple, Sequence, Tuple

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
# Spike-Propagating LIF Network with E/I Populations
# =============================================================================


class SpikePropagatingPolicy(linen.Module):
    """Spike-propagating LIF policy with E/I populations.

    Unlike the rate model, all layers run in lockstep within a single scan.
    At each micro-step, binary spikes (with surrogate gradients) propagate
    from layer i excitatory population to layer i+1.

    carry_flat = concat([v_all_layers, refrac_all_layers]).
    Total dim = 2 * sum(layer_sizes).
    """

    layer_sizes: Sequence[int]
    output_size: int
    exc_ratio: float = 0.8
    n_micro_steps: int = 16
    tau_min: float = 3.0
    tau_max: float = 15.0
    v_th: float = 0.3
    v_reset: float = 0.0
    beta_surrogate: float = 5.0
    n_refractory: float = 2.0

    @linen.compact
    def __call__(self, obs, carry_flat):
        n_layers = len(self.layer_sizes)
        total_neurons = sum(self.layer_sizes)
        v_th = self.v_th
        v_reset = self.v_reset
        beta = self.beta_surrogate
        n_refrac = self.n_refractory

        # Parse external carry
        v_flat = carry_flat[:, :total_neurons]
        refrac_flat = carry_flat[:, total_neurons:]
        splits = list(np.cumsum(self.layer_sizes[:-1]))

        # Pre-compute per-layer info
        layer_info = []  # (n_exc, n_inh, n_total, input_dim)
        for i, n_total in enumerate(self.layer_sizes):
            n_exc = round(n_total * self.exc_ratio)
            n_inh = n_total - n_exc
            if i == 0:
                in_dim = obs.shape[-1]
            else:
                in_dim = round(self.layer_sizes[i - 1] * self.exc_ratio)
            layer_info.append((n_exc, n_inh, n_total, in_dim))

        # Create ALL parameters before the scan
        W_ins, b_ins = [], []
        W_ies, W_eis = [], []
        alphas = []
        for i, (n_exc, n_inh, n_total, in_dim) in enumerate(layer_info):
            W_ins.append(self.param(
                f'lif_{i}_W_in',
                jax.nn.initializers.lecun_uniform(),
                (in_dim, n_total),
            ))
            b_ins.append(self.param(
                f'lif_{i}_b_in',
                jax.nn.initializers.zeros,
                (n_total,),
            ))
            tau_m = self.param(
                f'lif_{i}_tau_m',
                lambda key, shape: jp.exp(jax.random.uniform(
                    key, shape,
                    minval=jp.log(self.tau_min), maxval=jp.log(self.tau_max),
                )),
                (n_total,),
            )
            tau_m = jax.lax.stop_gradient(tau_m)
            alphas.append(jp.exp(-1.0 / tau_m))
            W_ies.append(self.param(
                f'lif_{i}_W_ie',
                jax.nn.initializers.lecun_uniform(),
                (n_inh, n_exc),
            ))
            W_eis.append(self.param(
                f'lif_{i}_W_ei',
                jax.nn.initializers.lecun_uniform(),
                (n_exc, n_inh),
            ))

        n_exc_last = layer_info[-1][0]
        W_readout = self.param(
            'readout_kernel',
            jax.nn.initializers.lecun_uniform(),
            (n_exc_last, self.output_size),
        )
        b_readout = self.param(
            'readout_bias',
            jax.nn.initializers.zeros,
            (self.output_size,),
        )

        # Layer 0 input current is constant across micro-steps
        I_input_0 = obs @ W_ins[0] + b_ins[0]

        # Internal scan carry: v, refrac, prev_spikes
        init_prev_spike = jp.zeros_like(v_flat)

        def micro_step(carry, _):
            v_f, refrac_f, prev_spike_f = carry

            v_layers = jp.split(v_f, splits, axis=-1) if splits else [v_f]
            refrac_layers = jp.split(refrac_f, splits, axis=-1) if splits else [refrac_f]
            prev_spike_layers = jp.split(prev_spike_f, splits, axis=-1) if splits else [prev_spike_f]

            new_v_parts = []
            new_refrac_parts = []
            new_spike_parts = []

            for i in range(n_layers):
                n_exc, n_inh, n_total, _ = layer_info[i]
                v_i = v_layers[i]
                refrac_i = refrac_layers[i]
                prev_sp_i = prev_spike_layers[i]

                # Feed-forward input
                if i == 0:
                    I_ff = I_input_0
                else:
                    # Excitatory spikes from previous layer
                    prev_exc_spikes = new_spike_parts[i - 1][:, :layer_info[i - 1][0]]
                    I_ff = prev_exc_spikes @ W_ins[i] + b_ins[i]

                # Lateral E/I currents
                sp_e = prev_sp_i[:, :n_exc]
                sp_i = prev_sp_i[:, n_exc:]
                I_lat_e = -sp_i @ jp.abs(W_ies[i])
                I_lat_i = sp_e @ jp.abs(W_eis[i])
                I_lateral = jp.concatenate([I_lat_e, I_lat_i], axis=-1)
                I_total = I_ff + I_lateral

                # LIF membrane dynamics
                is_refractory = (refrac_i > 0.0).astype(v_i.dtype)
                v_new = alphas[i] * v_i + (1.0 - alphas[i]) * I_total
                v_new = v_new * (1.0 - is_refractory)

                # Surrogate gradient spike
                spike_hard = (v_new >= v_th).astype(v_new.dtype)
                spike_soft = jax.nn.sigmoid(beta * (v_new - v_th))
                spike = jax.lax.stop_gradient(spike_hard - spike_soft) + spike_soft
                spike = spike * (1.0 - is_refractory)

                # Reset and refractory
                v_after = v_new * (1.0 - spike) + v_reset * spike
                refrac_new = jp.where(
                    spike > 0.5, n_refrac, jp.maximum(refrac_i - 1.0, 0.0)
                )

                new_v_parts.append(v_after)
                new_refrac_parts.append(refrac_new)
                new_spike_parts.append(spike)

            # Emit final layer excitatory spikes
            final_exc_spikes = new_spike_parts[-1][:, :n_exc_last]

            new_v_f = jp.concatenate(new_v_parts, axis=-1)
            new_refrac_f = jp.concatenate(new_refrac_parts, axis=-1)
            new_spike_f = jp.concatenate(new_spike_parts, axis=-1)
            return (new_v_f, new_refrac_f, new_spike_f), final_exc_spikes

        (v_final, refrac_final, _), all_final_spikes = jax.lax.scan(
            micro_step,
            (v_flat, refrac_flat, init_prev_spike),
            None,
            length=self.n_micro_steps,
        )

        # Last micro-step's spikes -> readout (no averaging)
        last_spikes = all_final_spikes[-1]  # (B, n_exc_last)
        logits = last_spikes @ W_readout + b_readout

        new_carry_flat = jp.concatenate([v_final, refrac_final], axis=-1)
        return logits, new_carry_flat


class SpikePropDiagnosticPolicy(linen.Module):
    """Same as SpikePropagatingPolicy but returns per-layer E/I traces."""

    layer_sizes: Sequence[int]
    output_size: int
    exc_ratio: float = 0.8
    n_micro_steps: int = 16
    tau_min: float = 3.0
    tau_max: float = 15.0
    v_th: float = 0.3
    v_reset: float = 0.0
    beta_surrogate: float = 5.0
    n_refractory: float = 2.0

    @linen.compact
    def __call__(self, obs, carry_flat):
        n_layers = len(self.layer_sizes)
        total_neurons = sum(self.layer_sizes)
        v_th = self.v_th
        v_reset = self.v_reset
        n_refrac = self.n_refractory

        v_flat = carry_flat[:, :total_neurons]
        refrac_flat = carry_flat[:, total_neurons:]
        splits = list(np.cumsum(self.layer_sizes[:-1]))

        layer_info = []
        for i, n_total in enumerate(self.layer_sizes):
            n_exc = round(n_total * self.exc_ratio)
            n_inh = n_total - n_exc
            if i == 0:
                in_dim = obs.shape[-1]
            else:
                in_dim = round(self.layer_sizes[i - 1] * self.exc_ratio)
            layer_info.append((n_exc, n_inh, n_total, in_dim))

        W_ins, b_ins = [], []
        W_ies, W_eis = [], []
        alphas = []
        for i, (n_exc, n_inh, n_total, in_dim) in enumerate(layer_info):
            W_ins.append(self.param(
                f'lif_{i}_W_in',
                jax.nn.initializers.lecun_uniform(),
                (in_dim, n_total),
            ))
            b_ins.append(self.param(
                f'lif_{i}_b_in',
                jax.nn.initializers.zeros,
                (n_total,),
            ))
            tau_m = self.param(
                f'lif_{i}_tau_m',
                lambda key, shape: jp.exp(jax.random.uniform(
                    key, shape,
                    minval=jp.log(self.tau_min), maxval=jp.log(self.tau_max),
                )),
                (n_total,),
            )
            tau_m = jax.lax.stop_gradient(tau_m)
            alphas.append(jp.exp(-1.0 / tau_m))
            W_ies.append(self.param(
                f'lif_{i}_W_ie',
                jax.nn.initializers.lecun_uniform(),
                (n_inh, n_exc),
            ))
            W_eis.append(self.param(
                f'lif_{i}_W_ei',
                jax.nn.initializers.lecun_uniform(),
                (n_exc, n_inh),
            ))

        n_exc_last = layer_info[-1][0]
        W_readout = self.param(
            'readout_kernel',
            jax.nn.initializers.lecun_uniform(),
            (n_exc_last, self.output_size),
        )
        b_readout = self.param(
            'readout_bias',
            jax.nn.initializers.zeros,
            (self.output_size,),
        )

        I_input_0 = obs @ W_ins[0] + b_ins[0]

        init_prev_spike = jp.zeros_like(v_flat)

        def micro_step(carry, _):
            v_f, refrac_f, prev_spike_f = carry

            v_layers = jp.split(v_f, splits, axis=-1) if splits else [v_f]
            refrac_layers = jp.split(refrac_f, splits, axis=-1) if splits else [refrac_f]
            prev_spike_layers = jp.split(prev_spike_f, splits, axis=-1) if splits else [prev_spike_f]

            new_v_parts = []
            new_refrac_parts = []
            new_spike_parts = []

            for i in range(n_layers):
                n_exc, n_inh, n_total, _ = layer_info[i]
                v_i = v_layers[i]
                refrac_i = refrac_layers[i]
                prev_sp_i = prev_spike_layers[i]

                if i == 0:
                    I_ff = I_input_0
                else:
                    prev_exc_spikes = new_spike_parts[i - 1][:, :layer_info[i - 1][0]]
                    I_ff = prev_exc_spikes @ W_ins[i] + b_ins[i]

                sp_e = prev_sp_i[:, :n_exc]
                sp_i = prev_sp_i[:, n_exc:]
                I_lat_e = -sp_i @ jp.abs(W_ies[i])
                I_lat_i = sp_e @ jp.abs(W_eis[i])
                I_lateral = jp.concatenate([I_lat_e, I_lat_i], axis=-1)
                I_total = I_ff + I_lateral

                is_refractory = (refrac_i > 0.0).astype(v_i.dtype)
                v_new = alphas[i] * v_i + (1.0 - alphas[i]) * I_total
                v_new = v_new * (1.0 - is_refractory)

                # Diagnostic: no surrogate, just hard spikes
                spike = (v_new >= v_th).astype(v_new.dtype)
                spike = spike * (1.0 - is_refractory)

                v_after = v_new * (1.0 - spike) + v_reset * spike
                refrac_new = jp.where(
                    spike > 0.5, n_refrac, jp.maximum(refrac_i - 1.0, 0.0)
                )

                new_v_parts.append(v_after)
                new_refrac_parts.append(refrac_new)
                new_spike_parts.append(spike)

            final_exc_spikes = new_spike_parts[-1][:, :n_exc_last]

            new_v_f = jp.concatenate(new_v_parts, axis=-1)
            new_refrac_f = jp.concatenate(new_refrac_parts, axis=-1)
            new_spike_f = jp.concatenate(new_spike_parts, axis=-1)

            # Emit per-step diagnostics
            all_spikes_flat = jp.concatenate(new_spike_parts, axis=-1)
            all_voltages_flat = jp.concatenate(new_v_parts, axis=-1)
            return (new_v_f, new_refrac_f, new_spike_f), (all_spikes_flat, all_voltages_flat, final_exc_spikes)

        (v_final, refrac_final, _), (all_spikes, all_voltages, all_final_spikes) = jax.lax.scan(
            micro_step,
            (v_flat, refrac_flat, init_prev_spike),
            None,
            length=self.n_micro_steps,
        )

        # Last micro-step's spikes -> readout (no averaging)
        last_spikes = all_final_spikes[-1]  # (B, n_exc_last)
        logits = last_spikes @ W_readout + b_readout
        new_carry_flat = jp.concatenate([v_final, refrac_final], axis=-1)

        # Split diagnostics per layer into E/I
        # all_spikes: (n_micro_steps, B, total_neurons)
        layer_ranges = []
        offset = 0
        for n_exc, n_inh, n_total, _ in layer_info:
            layer_ranges.append((offset, offset + n_total, n_exc))
            offset += n_total

        layer_diag = {}
        for i, (start, end, n_exc) in enumerate(layer_ranges):
            layer_diag[f"lif_{i}"] = {
                "spikes_exc": all_spikes[:, :, start:start + n_exc],
                "spikes_inh": all_spikes[:, :, start + n_exc:end],
                "voltages_exc": all_voltages[:, :, start:start + n_exc],
                "voltages_inh": all_voltages[:, :, start + n_exc:end],
            }

        return logits, new_carry_flat, layer_diag


# =============================================================================
# Custom PPO with recurrent carry
# =============================================================================


class Transition(NamedTuple):
    obs: Any           # (B, obs_dim) unnormalized
    action: Any        # (B, act_dim)
    raw_action: Any    # (B, act_dim)
    log_prob: Any      # (B,)
    value: Any         # (B,)
    reward: Any        # (B,)
    done: Any          # (B,)
    carry: Any         # (B, carry_dim)  carry used as INPUT for this step


def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    """Vectorised GAE via reverse scan.  All inputs are (T, B)."""
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
    advantages = advantages_rev[::-1]  # (T, B)
    returns = advantages + values
    return advantages, returns


def flatten_obs(obs):
    """Flatten nested observation dict to a single array."""
    flat_parts = []
    for key in sorted(obs.keys()):
        val = obs[key]
        if isinstance(val, dict):
            flat_parts.append(flatten_obs(val))
        else:
            flat_parts.append(val.flatten())
    return jp.concatenate(flat_parts)


# =============================================================================
# Spike diagnostic plotting
# =============================================================================


def plot_spike_diagnostics(layer_diag, v_th, env_step_idx, save_path=None):
    """Plot raster + voltage traces + rate histogram for each LIF layer.

    Args:
        layer_diag: dict  {layer_name: {spikes: (K,B,N), voltages: (K,B,N)}}
        v_th: firing threshold (for reference line)
        env_step_idx: for the title
        save_path: if provided, save figure
    Returns:
        matplotlib Figure
    """
    n_layers = len(layer_diag)
    fig, axes = plt.subplots(n_layers, 3, figsize=(15, 4 * n_layers), squeeze=False)

    for row, (name, diag) in enumerate(layer_diag.items()):
        spikes = np.array(diag["spikes"][:, 0, :])    # (K, N) first batch
        voltages = np.array(diag["voltages"][:, 0, :])
        K, N = spikes.shape

        # Raster
        ax = axes[row, 0]
        t_idx, n_idx = np.where(spikes > 0.5)
        n_show = min(N, 100)
        mask = n_idx < n_show
        ax.scatter(t_idx[mask], n_idx[mask], s=2, c="black", marker="|")
        ax.set_xlim(-0.5, K - 0.5)
        ax.set_ylim(-0.5, n_show - 0.5)
        ax.set_xlabel("micro-step")
        ax.set_ylabel("neuron")
        ax.set_title(f"{name} raster (t={env_step_idx})")

        # Voltage traces
        ax = axes[row, 1]
        for j in np.linspace(0, N - 1, min(10, N), dtype=int):
            ax.plot(voltages[:, j], alpha=0.7, linewidth=1)
        ax.axhline(v_th, color="red", ls="--", alpha=0.5, label=f"v_th={v_th}")
        ax.set_xlabel("micro-step")
        ax.set_ylabel("v")
        ax.set_title(f"{name} voltages")
        ax.legend(fontsize=7)

        # Rate histogram
        ax = axes[row, 2]
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
    batch_size=256,
    max_grad_norm=1.0,
    gae_lambda=0.95,
    clip_eps=0.3,
    vf_coef=0.5,
    network_factory=config_dict.create(
        policy_hidden_layer_sizes=(512, 512),
        value_hidden_layer_sizes=(512, 512, 512),
        exc_ratio=0.8,
        n_micro_steps=16,
        tau_min=2.0,
        tau_max=8.0,
        v_th=0.3,
        v_reset=0.0,
        beta_surrogate=5.0,
        n_refractory=2,
    ),
)

env_name = "mouse-imitation-spike-propagation"
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
               id=f"spike-prop-{exp_name}")
    wandb.config.update({
        "env_name": env_name,
        "policy_type": "spike_propagating_lif",
        **dict(ppo_params.network_factory),
    })


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Mouse Arm Imitation -- Spike-Propagating LIF Policy (E/I)")
    print("=" * 80)
    nf = ppo_params.network_factory
    n_exc_per = round(nf.policy_hidden_layer_sizes[0] * nf.exc_ratio)
    n_inh_per = nf.policy_hidden_layer_sizes[0] - n_exc_per
    print(f"LIF: K={nf.n_micro_steps}, tau=[{nf.tau_min},{nf.tau_max}], "
          f"v_th={nf.v_th}, beta={nf.beta_surrogate}, refrac={nf.n_refractory}")
    print(f"E/I: {n_exc_per}/{n_inh_per} per layer ({nf.exc_ratio:.0%} exc)")
    print(f"Policy layers: {nf.policy_hidden_layer_sizes}")

    # ------------------------------------------------------------------
    # Environment setup (identical to before)
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

    obs_size = env.observation_size
    if isinstance(obs_size, (tuple, list)):
        obs_size = obs_size[-1]
    act_size = env.action_size

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

    policy_module = SpikePropagatingPolicy(
        layer_sizes=nf.policy_hidden_layer_sizes,
        output_size=param_size,
        exc_ratio=nf.exc_ratio,
        n_micro_steps=nf.n_micro_steps,
        tau_min=nf.tau_min,
        tau_max=nf.tau_max,
        v_th=nf.v_th,
        v_reset=nf.v_reset,
        beta_surrogate=nf.beta_surrogate,
        n_refractory=nf.n_refractory,
    )
    diag_policy_module = SpikePropDiagnosticPolicy(
        layer_sizes=nf.policy_hidden_layer_sizes,
        output_size=param_size,
        exc_ratio=nf.exc_ratio,
        n_micro_steps=nf.n_micro_steps,
        tau_min=nf.tau_min,
        tau_max=nf.tau_max,
        v_th=nf.v_th,
        v_reset=nf.v_reset,
        beta_surrogate=nf.beta_surrogate,
        n_refractory=nf.n_refractory,
    )
    value_module = networks.MLP(
        layer_sizes=list(nf.value_hidden_layer_sizes) + [1],
        activation=linen.swish,
        kernel_init=jax.nn.initializers.lecun_uniform(),
    )

    carry_dim = 2 * sum(nf.policy_hidden_layer_sizes)  # v + refrac
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

    # ------------------------------------------------------------------
    # JIT-compiled core functions
    # ------------------------------------------------------------------

    @jax.jit
    def collect_rollout(policy_params, value_params, normalizer_params,
                        env_state, membrane_carry, rng):
        """Collect unroll_length transitions with threaded carry."""

        def step_fn(carry, _):
            state, mem_carry, k = carry
            k, ak = jax.random.split(k)

            obs_norm = running_statistics.normalize(state.obs, normalizer_params)
            logits, new_carry = policy_module.apply(
                policy_params, obs_norm, mem_carry
            )
            raw_action = action_dist.sample_no_postprocessing(logits, ak)
            log_prob = action_dist.log_prob(logits, raw_action)
            action = action_dist.postprocess(raw_action)
            value = jp.squeeze(
                value_module.apply(value_params, obs_norm), axis=-1
            )

            next_state = train_env.step(state, action)

            # Reset carry on episode done
            done_mask = next_state.done.reshape(-1, 1)
            new_carry = new_carry * (1.0 - done_mask)

            transition = Transition(
                obs=state.obs,
                action=action,
                raw_action=raw_action,
                log_prob=log_prob,
                value=value,
                reward=next_state.reward,
                done=next_state.done,
                carry=mem_carry,
            )
            return (next_state, new_carry, k), transition

        (final_state, final_carry, _), rollout = jax.lax.scan(
            step_fn,
            (env_state, membrane_carry, rng),
            None,
            length=ppo_params.unroll_length,
        )
        return final_state, final_carry, rollout

    def _sgd_step(policy_params, value_params, opt_state, normalizer_params,
                  batch_obs, batch_raw_action, batch_log_prob, batch_advantage,
                  batch_target, batch_carry, rng):
        """Single gradient update on one minibatch (called inside JIT)."""

        def loss_fn(params):
            pp, vp = params
            obs_norm = running_statistics.normalize(batch_obs, normalizer_params)

            logits, _ = policy_module.apply(pp, obs_norm, batch_carry)
            new_log_prob = action_dist.log_prob(logits, batch_raw_action)

            ratio = jp.exp(new_log_prob - batch_log_prob)
            adv = batch_advantage
            adv = (adv - jp.mean(adv)) / (jp.std(adv) + 1e-8)

            pg1 = -adv * ratio
            pg2 = -adv * jp.clip(ratio, 1.0 - ppo_params.clip_eps,
                                 1.0 + ppo_params.clip_eps)
            policy_loss = jp.mean(jp.maximum(pg1, pg2))

            new_value = jp.squeeze(value_module.apply(vp, obs_norm), axis=-1)
            value_loss = jp.mean(jp.square(new_value - batch_target))

            entropy = jp.mean(action_dist.entropy(logits, rng))

            total = (policy_loss
                     + ppo_params.vf_coef * value_loss
                     - ppo_params.entropy_cost * entropy)
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
        new_pp, new_vp = optax.apply_updates(
            (policy_params, value_params), updates
        )
        return new_pp, new_vp, new_opt_state, loss, metrics

    @jax.jit
    def prepare_ppo_data(normalizer_params, rollout, env_state_obs, value_params):
        """JIT'd normalizer update + GAE computation."""
        normalizer_params = running_statistics.update(
            normalizer_params, rollout.obs.reshape(-1, obs_size),
        )
        last_obs_norm = running_statistics.normalize(env_state_obs, normalizer_params)
        last_value = jp.squeeze(
            value_module.apply(value_params, last_obs_norm), axis=-1
        )
        rewards = rollout.reward * reward_scaling
        advantages, returns = compute_gae(
            rewards, rollout.value, rollout.done, last_value, gamma, gae_lambda,
        )
        return normalizer_params, advantages, returns

    @jax.jit
    def run_ppo_epochs(policy_params, value_params, opt_state, normalizer_params,
                       f_obs, f_raw, f_lp, f_adv, f_ret, f_carry, key):
        """Full PPO update: num_updates epochs x num_minibatches, all via lax.scan."""
        n_samples = f_obs.shape[0]

        def epoch_step(carry, _):
            pp, vp, os, k = carry
            k, perm_key = jax.random.split(k)
            perm = jax.random.permutation(perm_key, n_samples)

            def mb_step(carry2, mb_idx):
                pp2, vp2, os2, k2 = carry2
                k2, ek = jax.random.split(k2)
                start = mb_idx * batch_size
                idx = jax.lax.dynamic_slice(perm, (start,), (batch_size,))
                pp2, vp2, os2, loss, metrics = _sgd_step(
                    pp2, vp2, os2, normalizer_params,
                    f_obs[idx], f_raw[idx], f_lp[idx], f_adv[idx],
                    f_ret[idx], f_carry[idx], ek,
                )
                return (pp2, vp2, os2, k2), (loss, metrics)

            (pp, vp, os, k), (losses, all_metrics) = jax.lax.scan(
                mb_step, (pp, vp, os, k), jp.arange(num_minibatches)
            )
            # Keep last minibatch metrics from this epoch
            last_metrics = jax.tree.map(lambda x: x[-1], all_metrics)
            return (pp, vp, os, k), last_metrics

        (pp, vp, os, k), epoch_metrics = jax.lax.scan(
            epoch_step,
            (policy_params, value_params, opt_state, key),
            None, length=num_updates,
        )
        # Return metrics from last epoch's last minibatch
        final_metrics = jax.tree.map(lambda x: x[-1], epoch_metrics)
        return pp, vp, os, k, final_metrics

    # ------------------------------------------------------------------
    # Eval helpers
    # ------------------------------------------------------------------

    num_eval_envs = 128

    @jax.jit
    def jit_eval_rollout(policy_params, normalizer_params, eval_state, rng):
        """Fast JIT'd eval rollout on test_env with lax.scan + carry."""
        eval_carry = jp.zeros((num_eval_envs, carry_dim))

        def step_fn(carry, _):
            state, mem_carry, k = carry
            k, _ = jax.random.split(k)
            obs_norm = running_statistics.normalize(state.obs, normalizer_params)
            logits, new_carry = policy_module.apply(
                policy_params, obs_norm, mem_carry
            )
            action = action_dist.mode(logits)  # deterministic
            next_state = test_env.step(state, action)
            done_mask = next_state.done.reshape(-1, 1)
            new_carry = new_carry * (1.0 - done_mask)
            return (next_state, new_carry, k), (next_state.reward, next_state.done)

        (final_state, _, _), (rewards, _) = jax.lax.scan(
            step_fn, (eval_state, eval_carry, rng), None, length=episode_length,
        )
        # Compute per-env episode reward (sum until first done)
        # For simplicity, use mean reward across the whole rollout
        mean_reward = jp.mean(jp.sum(rewards, axis=0))
        std_reward = jp.std(jp.sum(rewards, axis=0))
        return final_state, {
            "eval/episode_reward": mean_reward,
            "eval/episode_reward_std": std_reward,
            "eval/mean_step_reward": jp.mean(rewards),
        }

    # JIT'd helpers for video rollout on raw eval_env (single-env, unbatched)
    jit_eval_reset = jax.jit(eval_env.reset)
    jit_eval_step = jax.jit(eval_env.step)
    jit_policy_apply = jax.jit(policy_module.apply)

    def video_rollout(policy_params, normalizer_params, seed=0):
        """Fast video rollout: Python loop but all ops are pre-JIT'd."""
        rng = jax.random.PRNGKey(seed)
        state = jit_eval_reset(rng)
        carry = jp.zeros((1, carry_dim))
        rollout_states = [state]
        for _ in range(episode_length):
            flat_obs = flatten_obs(state.obs)
            obs_norm = running_statistics.normalize(
                flat_obs[None], normalizer_params
            )
            logits, new_carry = jit_policy_apply(
                policy_params, obs_norm, carry
            )
            action = jp.squeeze(action_dist.mode(logits), axis=0)
            state = jit_eval_step(state, action)
            carry = new_carry * (1.0 - state.done.reshape(1, 1))
            rollout_states.append(state)
        return rollout_states

    def eval_and_log(step, policy_params, value_params, normalizer_params):
        """Eval metrics + video + checkpoint."""
        import time as _time
        t_eval = _time.time()

        # -- Fast JIT'd eval metrics --
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

        # -- Video (Python loop, JIT'd ops) --
        try:
            states = video_rollout(policy_params, normalizer_params, seed=step)
            fps = int(1.0 / eval_env.dt)
            frames = eval_env.render(
                states, height=512, width=512, render_ghost=True
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

        # -- Checkpoint --
        params_to_save = (normalizer_params, policy_params, value_params)
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params_to_save)
        path = ckpt_path / f"{step}"
        orbax_checkpointer.save(path, params_to_save, force=True,
                                save_args=save_args)
        print(f"  checkpoint -> {path}")

    # ------------------------------------------------------------------
    # Post-training spike diagnostics (one episode, one neuron per layer)
    # ------------------------------------------------------------------

    def run_spike_diagnostics(policy_params, normalizer_params):
        """Run one episode with diagnostic policy and plot E/I spike data."""
        print("Running post-training spike diagnostics...")
        rng = jax.random.PRNGKey(999)
        state = jit_eval_reset(rng)
        carry = jp.zeros((1, carry_dim))

        jit_diag_apply = jax.jit(diag_policy_module.apply)
        n_lif = len(nf.policy_hidden_layer_sizes)
        all_data = {f"lif_{i}": {"spikes_exc": [], "spikes_inh": [],
                                  "voltages_exc": [], "voltages_inh": []}
                    for i in range(n_lif)}

        for t in range(episode_length):
            flat_obs = flatten_obs(state.obs)
            obs_norm = running_statistics.normalize(flat_obs[None], normalizer_params)
            logits, new_carry, layer_diag = jit_diag_apply(
                policy_params, obs_norm, carry
            )
            action = jp.squeeze(action_dist.mode(logits), axis=0)
            state = jit_eval_step(state, action)
            carry = new_carry * (1.0 - state.done.reshape(1, 1))

            for lname, diag in layer_diag.items():
                all_data[lname]["spikes_exc"].append(np.array(diag["spikes_exc"][:, 0, :]))
                all_data[lname]["spikes_inh"].append(np.array(diag["spikes_inh"][:, 0, :]))
                all_data[lname]["voltages_exc"].append(np.array(diag["voltages_exc"][:, 0, :]))
                all_data[lname]["voltages_inh"].append(np.array(diag["voltages_inh"][:, 0, :]))

        # 6 panels per layer: E voltage, I voltage, E raster, I raster, E hist, I hist
        n_layers = len(all_data)
        fig, axes = plt.subplots(n_layers, 6, figsize=(36, 4 * n_layers),
                                 squeeze=False)

        for row, lname in enumerate(sorted(all_data.keys())):
            d = all_data[lname]
            sp_e = np.stack(d["spikes_exc"])     # (T, K, n_exc)
            sp_i = np.stack(d["spikes_inh"])     # (T, K, n_inh)
            vo_e = np.stack(d["voltages_exc"])   # (T, K, n_exc)
            vo_i = np.stack(d["voltages_inh"])   # (T, K, n_inh)

            # Panel 1: E neuron 0 voltage trace
            ax = axes[row, 0]
            ax.plot(vo_e[:, :, 0].reshape(-1), linewidth=0.3, color="steelblue")
            ax.axhline(nf.v_th, color="red", ls="--", alpha=0.5)
            ax.set_xlabel("micro-step")
            ax.set_ylabel("v")
            ax.set_title(f"{lname} E neuron 0 voltage")

            # Panel 2: I neuron 0 voltage trace
            ax = axes[row, 1]
            ax.plot(vo_i[:, :, 0].reshape(-1), linewidth=0.3, color="darkorange")
            ax.axhline(nf.v_th, color="red", ls="--", alpha=0.5)
            ax.set_xlabel("micro-step")
            ax.set_ylabel("v")
            ax.set_title(f"{lname} I neuron 0 voltage")

            # Panel 3: E spike raster (first 64)
            n_show_e = min(sp_e.shape[2], 64)
            ax = axes[row, 2]
            rates_e = np.mean(sp_e, axis=1)  # (T, n_exc)
            ax.imshow(rates_e[:, :n_show_e].T, aspect="auto", cmap="Blues",
                      interpolation="nearest", vmin=0, vmax=1)
            ax.set_xlabel("env step")
            ax.set_ylabel("E neuron")
            ax.set_title(f"{lname} E raster (first {n_show_e})")

            # Panel 4: I spike raster (first 32)
            n_show_i = min(sp_i.shape[2], 32)
            ax = axes[row, 3]
            rates_i = np.mean(sp_i, axis=1)  # (T, n_inh)
            ax.imshow(rates_i[:, :n_show_i].T, aspect="auto", cmap="Oranges",
                      interpolation="nearest", vmin=0, vmax=1)
            ax.set_xlabel("env step")
            ax.set_ylabel("I neuron")
            ax.set_title(f"{lname} I raster (first {n_show_i})")

            # Panel 5: E firing rate histogram
            ax = axes[row, 4]
            ep_rates_e = np.mean(rates_e, axis=0)
            ax.hist(ep_rates_e, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
            ax.axvline(np.mean(ep_rates_e), color="red", ls="--",
                       label=f"mean={np.mean(ep_rates_e):.3f}")
            ax.set_xlabel("spike rate")
            ax.set_title(f"{lname} E rate dist")
            ax.legend(fontsize=7)

            # Panel 6: I firing rate histogram
            ax = axes[row, 5]
            ep_rates_i = np.mean(rates_i, axis=0)
            ax.hist(ep_rates_i, bins=30, edgecolor="black", alpha=0.7, color="darkorange")
            ax.axvline(np.mean(ep_rates_i), color="red", ls="--",
                       label=f"mean={np.mean(ep_rates_i):.3f}")
            ax.set_xlabel("spike rate")
            ax.set_title(f"{lname} I rate dist")
            ax.legend(fontsize=7)

        fig.tight_layout()
        save_path = f"{ckpt_path}/final_spike_diagnostics.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  spike diagnostics -> {save_path}")
        if USE_WANDB:
            wandb.log({"eval/final_spike_diagnostics": wandb.Image(fig)},
                      commit=True)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    print("Initialising training env state...")
    env_state = train_env.reset(jax.random.split(ek, num_envs))
    membrane_carry = jp.zeros((num_envs, carry_dim))

    total_steps = 0
    unroll_length = ppo_params.unroll_length
    num_updates = ppo_params.num_updates_per_batch
    num_minibatches = ppo_params.num_minibatches
    batch_size = ppo_params.batch_size
    gamma = ppo_params.discounting
    gae_lambda = ppo_params.gae_lambda
    reward_scaling = ppo_params.reward_scaling

    steps_per_unroll = unroll_length * num_envs
    num_timesteps = ppo_params.num_timesteps
    num_evals = ppo_params.num_evals
    steps_per_eval = num_timesteps // num_evals
    next_eval_at = steps_per_eval

    print(f"Training for {num_timesteps:,} steps  "
          f"({steps_per_unroll:,} per unroll, eval every {steps_per_eval:,})")
    print("=" * 80)

    import time as _time
    t0 = _time.time()

    while total_steps < num_timesteps:
        # -- collect rollout --
        key, rollout_key = jax.random.split(key)
        env_state, membrane_carry, rollout = collect_rollout(
            policy_params, value_params, normalizer_params,
            env_state, membrane_carry, rollout_key,
        )
        total_steps += steps_per_unroll

        # -- normalizer update + GAE (JIT'd) --
        normalizer_params, advantages, returns = prepare_ppo_data(
            normalizer_params, rollout, env_state.obs, value_params,
        )

        # -- flatten (T, B, ...) -> (T*B, ...) --
        T, B = rollout.obs.shape[:2]
        flat = lambda x: x.reshape(T * B, *x.shape[2:])
        f_obs = flat(rollout.obs)
        f_raw = flat(rollout.raw_action)
        f_lp = flat(rollout.log_prob)
        f_adv = flat(advantages)
        f_ret = flat(returns)
        f_carry = flat(rollout.carry)

        # -- PPO updates (JIT'd lax.scan) --
        key, update_key = jax.random.split(key)
        policy_params, value_params, opt_state, _, metrics = run_ppo_epochs(
            policy_params, value_params, opt_state, normalizer_params,
            f_obs, f_raw, f_lp, f_adv, f_ret, f_carry, update_key,
        )

        # -- logging --
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

    # Run spike diagnostics once at the end
    run_spike_diagnostics(policy_params, normalizer_params)
