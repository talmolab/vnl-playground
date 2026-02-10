"""Detailed spike diagnostics for LRN-Cerebellum spiking circuit.

Generates publication-quality neuroscience figures analyzing:
  - ProprioSpinal E/I populations
  - LRN relay neurons (E only)
  - Cerebellum Kalman filter internals
  - Firing statistics, network parameters

Outputs:
  - population_activity.png : spike rasters + smoothed PSTH + E/I balance
  - single_neuron_traces.png : intracellular-style voltage traces
  - firing_statistics.png : ISI distributions, rate distributions, CV, Fano factor
  - network_params.png : tau_m, weight distributions, F spectrum, B/H weights
  - cerebellum_diagnostics.png : innovation, state, correction, covariance
  - firing_rate_spectral.png : population rate PSD (Welch), band power, spectrogram
  - spike_synchrony.png : PLV (Varela/Lachaux), coherence, cross-correlograms

Usage:
    python spike_diagnostics.py [checkpoint_path] [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch, coherence, spectrogram, hilbert, butter, filtfilt

import jax
import jax.numpy as jp
from brax.training import distribution
from brax.training.acme import running_statistics
from orbax import checkpoint as ocp

from vnl_playground.tasks.mouse.imitation import MouseImitation, default_config
from vnl_playground.tasks.mouse.reference_clips import MouseReferenceClips
from vnl_playground.tasks.mouse import consts

from train_mouse_lrn_cerebellum import (
    LRNCerebellumPolicy,
    DiagnosticLRNCerebellumPolicy,
    flatten_obs,
    ppo_params,
)

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

# =============================================================================
# Publication-quality plot style — seaborn E/I colour scheme
# =============================================================================

C_EXC = '#1f77b4'   # blue for excitatory
C_INH = '#ff7f0e'   # orange for inhibitory
C_TH  = '#2ca02c'   # green for threshold
C_REF = '#d9d9d9'   # light gray for refractory shading
C_BAL = '#9467bd'   # purple for E/I balance
C_CB  = '#d62728'   # red for cerebellum signals

CMAP_EXC = LinearSegmentedColormap.from_list('exc', ['#ffffff', C_EXC])
CMAP_INH = LinearSegmentedColormap.from_list('inh', ['#ffffff', C_INH])

plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.family': 'sans-serif',
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
})

# =============================================================================
# Args
# =============================================================================

DEFAULT_CKPT = (
    "/root/vast/eric/vnl-playground/checkpoints/"
    "mouse-lrn-cerebellum-20260203-033949/160030720"
)
CHECKPOINT_PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT
OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "diagnostic_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"Output dir: {OUTPUT_DIR}")

# =============================================================================
# Environment setup
# =============================================================================

nf = ppo_params.network_factory
env_cfg = default_config()
print("Loading reference clips...")
reference_clips = MouseReferenceClips(
    str(consts.MOUSE_REFERENCE_DATA_PATH),
    n_frames_per_clip=env_cfg.clip_length,
)
_, test_clips = reference_clips.split(train_ratio=0.8, seed=42)
eval_env = MouseImitation(config=env_cfg, clips=test_clips)

steps_per_frame = (1 / env_cfg.mocap_hz) / env_cfg.ctrl_dt
episode_length = int(
    (env_cfg.clip_length - env_cfg.start_frame_range[-1]
     - env_cfg.reference_length) * steps_per_frame
)

obs_size = eval_env.observation_size
if isinstance(obs_size, (tuple, list)):
    obs_size = obs_size[-1]
act_size = eval_env.action_size

# =============================================================================
# Network setup & checkpoint
# =============================================================================
#
# The checkpoint may have been saved with a different exc_ratio than the
# current ppo_params.  We do a raw restore first to discover the actual
# W_ie shape, infer the true exc_ratio, then build modules that match.
# =============================================================================

action_dist = distribution.NormalTanhDistribution(event_size=act_size)
param_size = action_dist.param_size

# Phase 1: Raw restore to discover the actual architecture
print("Loading checkpoint (phase 1: discover architecture)...")
raw_restored = ocp.PyTreeCheckpointer().restore(CHECKPOINT_PATH)
_, raw_policy, _ = raw_restored

W_ie_shape = np.array(raw_policy['params']['propriospinal']['W_ie']).shape
ckpt_n_inh = W_ie_shape[0]
ckpt_n_exc = W_ie_shape[1]
ckpt_ps_size = ckpt_n_exc + ckpt_n_inh
ckpt_exc_ratio = ckpt_n_exc / ckpt_ps_size
print(f"Checkpoint arch: PS {ckpt_n_exc}E/{ckpt_n_inh}I "
      f"(exc_ratio={ckpt_exc_ratio:.4f}, total={ckpt_ps_size})")
del raw_restored, raw_policy

# Phase 2: Build modules with correct architecture, then do typed restore
_module_kwargs = dict(
    propriospinal_size=ckpt_ps_size,
    exc_ratio=ckpt_exc_ratio,
    lrn_size=nf.lrn_size,
    cerebellum_state_dim=nf.cerebellum_state_dim,
    n_micro_steps=nf.n_micro_steps,
    tau_min=nf.tau_min, tau_max=nf.tau_max,
    v_th=nf.v_th, v_reset=nf.v_reset,
    beta_surrogate=nf.beta_surrogate,
    n_refractory=nf.n_refractory,
    output_size=param_size,
)

policy_module = LRNCerebellumPolicy(**_module_kwargs)
diag_policy_module = DiagnosticLRNCerebellumPolicy(**_module_kwargs)

carry_dim = (
    2 * ckpt_ps_size                # ps_v + ps_r
    + 2 * nf.lrn_size              # lrn_v + lrn_r
    + 2 * nf.cerebellum_state_dim  # x_hat + P_diag
)

key = jax.random.PRNGKey(0)
key, pk, vk = jax.random.split(key, 3)
dummy_obs = jp.zeros((1, obs_size))
dummy_carry = jp.zeros((1, carry_dim))
policy_params_init = policy_module.init(pk, dummy_obs, dummy_carry)
normalizer_params_init = running_statistics.init_state(jp.zeros(obs_size))

from brax.training import networks as brax_networks
from flax import linen
value_module = brax_networks.MLP(
    layer_sizes=list(nf.value_hidden_layer_sizes) + [1],
    activation=linen.swish,
    kernel_init=jax.nn.initializers.lecun_uniform(),
)
value_params_init = value_module.init(vk, dummy_obs)

print("Loading checkpoint (phase 2: typed restore)...")
target = (normalizer_params_init, policy_params_init, value_params_init)
normalizer_params, policy_params, value_params = (
    ocp.PyTreeCheckpointer().restore(CHECKPOINT_PATH, item=target)
)
print("Checkpoint loaded.")

# =============================================================================
# Constants
# =============================================================================

n_exc = ckpt_n_exc
n_inh = ckpt_n_inh
n_lrn = nf.lrn_size
K = nf.n_micro_steps
n_refrac = nf.n_refractory
cb_state_dim = nf.cerebellum_state_dim
print(f"\nProprioSpinal: {n_exc}E/{n_inh}I, LRN: {n_lrn}E, "
      f"Cerebellum: state_dim={cb_state_dim}")
print(f"K={K} micro-steps, refrac={n_refrac}")

# Extract learned parameters
ps_params = policy_params['params']['propriospinal']
lrn_params = policy_params['params']['lrn_relay']
cb_params = policy_params['params']['cerebellum']

learned = {
    'ps_tau_m': np.array(ps_params['tau_m']),
    'ps_W_ie': np.array(ps_params['W_ie']),
    'ps_W_ei': np.array(ps_params['W_ei']),
    'ps_input_kernel': np.array(ps_params['input_proj']['kernel']),
    'lrn_tau_m': np.array(lrn_params['tau_m']),
    'lrn_input_kernel': np.array(lrn_params['input_proj']['kernel']),
    'cb_F': np.array(cb_params['F']),
    'cb_B': np.array(cb_params['B']),
    'cb_H': np.array(cb_params['H']),
}

w_raw = np.array(policy_params['params']['correction_weight_raw'])
correction_weight = 1.0 / (1.0 + np.exp(-w_raw))  # sigmoid
print(f"Correction weight: {correction_weight.item():.6f}")

# =============================================================================
# Diagnostic rollout
# =============================================================================

print("Running diagnostic rollout...")
jit_eval_reset = jax.jit(eval_env.reset)
jit_eval_step = jax.jit(eval_env.step)
jit_diag_apply = jax.jit(diag_policy_module.apply)

state = jit_eval_reset(jax.random.PRNGKey(999))
carry = jp.zeros((1, carry_dim))

all_data = {
    'ps_spikes_exc': [], 'ps_spikes_inh': [],
    'ps_voltages_exc': [], 'ps_voltages_inh': [],
    'lrn_spikes': [], 'lrn_voltages': [],
    'cb_innovation': [], 'cb_innovation_norm': [],
    'cb_x_hat_new': [], 'cb_correction': [],
    'cb_P_diag_new': [],
    'raw_motor_cmd': [], 'reward': [],
}

for t in range(episode_length):
    flat_obs = flatten_obs(state.obs)
    obs_norm = running_statistics.normalize(flat_obs[None], normalizer_params)
    logits, new_carry, diagnostics = jit_diag_apply(
        policy_params, obs_norm, carry)
    action = jp.squeeze(action_dist.mode(logits), axis=0)
    state = jit_eval_step(state, action)
    carry = new_carry * (1.0 - state.done.reshape(1, 1))

    ps_diag = diagnostics['propriospinal']
    lrn_diag = diagnostics['lrn']
    cb_diag = diagnostics['cerebellum']

    # PS: (K, batch, N) -> take batch 0
    all_data['ps_spikes_exc'].append(np.array(ps_diag['spikes_exc'][:, 0, :]))
    all_data['ps_spikes_inh'].append(np.array(ps_diag['spikes_inh'][:, 0, :]))
    all_data['ps_voltages_exc'].append(np.array(ps_diag['voltages_exc'][:, 0, :]))
    all_data['ps_voltages_inh'].append(np.array(ps_diag['voltages_inh'][:, 0, :]))

    # LRN: (K, batch, N) -> take batch 0
    all_data['lrn_spikes'].append(np.array(lrn_diag['spikes'][:, 0, :]))
    all_data['lrn_voltages'].append(np.array(lrn_diag['voltages'][:, 0, :]))

    # Cerebellum: (batch, dim) -> take batch 0
    all_data['cb_innovation'].append(np.array(cb_diag['innovation'][0]))
    all_data['cb_innovation_norm'].append(float(cb_diag['innovation_norm'][0]))
    all_data['cb_x_hat_new'].append(np.array(cb_diag['x_hat_new'][0]))
    all_data['cb_correction'].append(np.array(cb_diag['correction'][0]))
    all_data['cb_P_diag_new'].append(np.array(cb_diag['P_diag_new'][0]))

    all_data['raw_motor_cmd'].append(np.array(diagnostics['raw_motor_cmd'][0]))
    all_data['reward'].append(float(state.reward))

print("Stacking data...")
ps_sp_e = np.stack(all_data['ps_spikes_exc'])    # (T, K, n_exc)
ps_sp_i = np.stack(all_data['ps_spikes_inh'])    # (T, K, n_inh)
ps_vo_e = np.stack(all_data['ps_voltages_exc'])
ps_vo_i = np.stack(all_data['ps_voltages_inh'])
lrn_sp  = np.stack(all_data['lrn_spikes'])       # (T, K, n_lrn)
lrn_vo  = np.stack(all_data['lrn_voltages'])

cb_innov      = np.stack(all_data['cb_innovation'])       # (T, obs_dim)
cb_innov_norm = np.array(all_data['cb_innovation_norm'])  # (T,)
cb_x_hat      = np.stack(all_data['cb_x_hat_new'])        # (T, state_dim)
cb_corr       = np.stack(all_data['cb_correction'])       # (T, output_dim)
cb_P          = np.stack(all_data['cb_P_diag_new'])       # (T, state_dim)
raw_motor     = np.stack(all_data['raw_motor_cmd'])
rewards       = np.array(all_data['reward'])

T_ep = ps_sp_e.shape[0]

# Per-env-step rates (average over micro-steps)
rate_e   = np.mean(ps_sp_e, axis=1)   # (T, n_exc)
rate_i   = np.mean(ps_sp_i, axis=1)   # (T, n_inh)
rate_lrn = np.mean(lrn_sp, axis=1)    # (T, n_lrn)


def panel_label(ax, label, x=-0.08, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12,
            fontweight='bold', va='top', ha='right')


# =============================================================================
# Figure 1: Population Activity — Rasters + PSTH + E/I Balance
# =============================================================================

print("Plotting population activity...")
fig, axes = plt.subplots(2, 3, figsize=(16, 9), squeeze=False)
fig.suptitle("Population Activity — LRN-Cerebellum Circuit",
             fontsize=13, fontweight='bold', y=0.995)

# Sort neurons by mean rate for cleaner rasters
sort_e   = np.argsort(np.mean(rate_e, axis=0))[::-1]
sort_i   = np.argsort(np.mean(rate_i, axis=0))[::-1]
sort_lrn = np.argsort(np.mean(rate_lrn, axis=0))[::-1]

# --- Row 0: ProprioSpinal ---
# Panel A: Spike raster (E top, I bottom)
ax = axes[0, 0]
n_show_e = min(n_exc, 80)
n_show_i = min(n_inh, 40)
for ni_idx, neuron in enumerate(sort_e[:n_show_e]):
    spike_times = np.where(rate_e[:, neuron] > 0.05)[0]
    ax.scatter(spike_times, np.full_like(spike_times, ni_idx),
               s=0.3, c=C_EXC, marker='|', linewidths=0.4)
gap = 3
for ni_idx, neuron in enumerate(sort_i[:n_show_i]):
    spike_times = np.where(rate_i[:, neuron] > 0.05)[0]
    ax.scatter(spike_times,
               np.full_like(spike_times, n_show_e + gap + ni_idx),
               s=0.3, c=C_INH, marker='|', linewidths=0.4)
ax.axhline(n_show_e + gap / 2, color='black', lw=0.5, ls='-', alpha=0.3)
ax.set_xlim(0, T_ep)
ax.set_ylim(-1, n_show_e + gap + n_show_i)
ax.set_ylabel('Neuron index')
ax.set_xlabel('Time (env steps)')
ax.set_title('ProprioSpinal — Spike Raster')
legend_e = Line2D([0], [0], color=C_EXC, lw=2, label=f'E (n={n_exc})')
legend_i = Line2D([0], [0], color=C_INH, lw=2, label=f'I (n={n_inh})')
ax.legend(handles=[legend_e, legend_i], loc='upper right', framealpha=0.8)
panel_label(ax, 'a')

# Panel B: Smoothed PSTH
ax = axes[0, 1]
pop_e = np.mean(rate_e, axis=1)
pop_i = np.mean(rate_i, axis=1)
sigma = max(1, T_ep // 40)
pop_e_s = gaussian_filter1d(pop_e, sigma)
pop_i_s = gaussian_filter1d(pop_i, sigma)
t = np.arange(T_ep)
ax.fill_between(t, pop_e_s, alpha=0.2, color=C_EXC)
ax.fill_between(t, pop_i_s, alpha=0.2, color=C_INH)
ax.plot(t, pop_e_s, color=C_EXC, lw=1.2,
        label=f'E (mean={np.mean(pop_e):.3f})')
ax.plot(t, pop_i_s, color=C_INH, lw=1.2,
        label=f'I (mean={np.mean(pop_i):.3f})')
ax.set_xlabel('Time (env steps)')
ax.set_ylabel('Population firing rate')
ax.set_title('ProprioSpinal — PSTH')
ax.legend(loc='upper right', framealpha=0.8)
ax.set_xlim(0, T_ep)
panel_label(ax, 'b')

# Panel C: E/I Balance
ax = axes[0, 2]
balance = pop_e_s / np.maximum(pop_i_s, 1e-8)
ax.plot(t, balance, color=C_BAL, lw=1.0)
ax.axhline(1.0, color='black', ls='--', lw=0.6, alpha=0.4)
ax.fill_between(t, 1.0, balance, where=balance > 1.0,
                alpha=0.15, color=C_EXC, label='E dominant')
ax.fill_between(t, 1.0, balance, where=balance < 1.0,
                alpha=0.15, color=C_INH, label='I dominant')
ax.set_xlabel('Time (env steps)')
ax.set_ylabel('E/I rate ratio')
ax.set_title('ProprioSpinal — E/I Balance')
ax.legend(loc='upper right', framealpha=0.8)
ax.set_xlim(0, T_ep)
panel_label(ax, 'c')

# --- Row 1: LRN Relay ---
# Panel D: LRN spike raster
ax = axes[1, 0]
n_show_lrn = min(n_lrn, 80)
for ni_idx, neuron in enumerate(sort_lrn[:n_show_lrn]):
    spike_times = np.where(rate_lrn[:, neuron] > 0.05)[0]
    ax.scatter(spike_times, np.full_like(spike_times, ni_idx),
               s=0.3, c=C_EXC, marker='|', linewidths=0.4)
ax.set_xlim(0, T_ep)
ax.set_ylim(-1, n_show_lrn)
ax.set_ylabel('Neuron index')
ax.set_xlabel('Time (env steps)')
ax.set_title('LRN Relay — Spike Raster (E only)', color=C_EXC)
panel_label(ax, 'd')

# Panel E: LRN PSTH
ax = axes[1, 1]
pop_lrn = np.mean(rate_lrn, axis=1)
pop_lrn_s = gaussian_filter1d(pop_lrn, sigma)
ax.fill_between(t, pop_lrn_s, alpha=0.2, color=C_EXC)
ax.plot(t, pop_lrn_s, color=C_EXC, lw=1.2,
        label=f'LRN (mean={np.mean(pop_lrn):.3f})')
ax.set_xlabel('Time (env steps)')
ax.set_ylabel('Population firing rate')
ax.set_title('LRN Relay — PSTH')
ax.legend(loc='upper right', framealpha=0.8)
ax.set_xlim(0, T_ep)
panel_label(ax, 'e')

# Panel F: PN→LRN relay fidelity
ax = axes[1, 2]
ax.scatter(pop_e, pop_lrn, s=3, alpha=0.5, color=C_EXC)
r_corr = np.corrcoef(pop_e, pop_lrn)[0, 1]
ax.set_xlabel('PS E mean rate')
ax.set_ylabel('LRN mean rate')
ax.set_title(f'PN→LRN Relay Fidelity (r={r_corr:.3f})')
panel_label(ax, 'f')

fig.tight_layout()
path = os.path.join(OUTPUT_DIR, "population_activity.png")
fig.savefig(path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path}")


# =============================================================================
# Figure 2: Single-Neuron Electrophysiology
# =============================================================================

print("Plotting single-neuron traces...")
fig, axes = plt.subplots(2, 3, figsize=(18, 8), squeeze=False)
fig.suptitle("Single-Neuron Membrane Dynamics",
             fontsize=13, fontweight='bold', y=0.995)

# Select representative neurons
rate_per_e   = np.mean(ps_sp_e, axis=(0, 1))
rate_per_i   = np.mean(ps_sp_i, axis=(0, 1))
rate_per_lrn = np.mean(lrn_sp, axis=(0, 1))

active_e = np.where(rate_per_e > 0.01)[0]
high_e = active_e[np.argmax(rate_per_e[active_e])] if len(active_e) else 0
sorted_e = np.argsort(rate_per_e)
mid_e = sorted_e[len(sorted_e) // 2]

active_i = np.where(rate_per_i > 0.01)[0]
high_i = active_i[np.argmax(rate_per_i[active_i])] if len(active_i) else 0

active_lrn = np.where(rate_per_lrn > 0.01)[0]
high_lrn = active_lrn[np.argmax(rate_per_lrn[active_lrn])] if len(active_lrn) else 0
sorted_lrn = np.argsort(rate_per_lrn)
mid_lrn = sorted_lrn[len(sorted_lrn) // 2]

n_show_steps = min(20, T_ep)

# Row 0: ProprioSpinal
for col, (neuron_idx, pop_sp, pop_vo, color, title) in enumerate([
    (high_e, ps_sp_e, ps_vo_e, C_EXC,
     f'PS E neuron {high_e} (rate={rate_per_e[high_e]:.3f})'),
    (mid_e, ps_sp_e, ps_vo_e, C_EXC,
     f'PS E neuron {mid_e} (rate={rate_per_e[mid_e]:.3f})'),
    (high_i, ps_sp_i, ps_vo_i, C_INH,
     f'PS I neuron {high_i} (rate={rate_per_i[high_i]:.3f})'),
]):
    ax = axes[0, col]
    v_data = pop_vo[:n_show_steps, :, neuron_idx].flatten()
    s_data = pop_sp[:n_show_steps, :, neuron_idx].flatten()
    t_micro = np.arange(len(v_data))

    ax.plot(t_micro, v_data, color=color, lw=0.4, alpha=0.9)
    ax.axhline(nf.v_th, color=C_TH, ls='--', lw=0.7, alpha=0.6,
               label=f'$V_{{th}}$={nf.v_th}')
    spike_idx = np.where(s_data > 0.5)[0]
    if len(spike_idx) > 0:
        ax.scatter(spike_idx, np.full_like(spike_idx, nf.v_th, dtype=float),
                   color='red', s=8, zorder=5, marker='v',
                   label=f'Spikes (n={len(spike_idx)})')
        for si in spike_idx:
            ax.axvspan(si, min(si + n_refrac, len(v_data)),
                       alpha=0.15, color=C_REF, lw=0)
    for s in range(1, n_show_steps):
        ax.axvline(s * K, color='black', ls=':', alpha=0.15, lw=0.4)

    ax.set_xlabel('Micro-step')
    ax.set_ylabel('$V_m$')
    ax.set_title(title)
    ax.legend(loc='lower right', framealpha=0.8, fontsize=7)
    panel_label(ax, chr(ord('a') + col))

# Row 1: LRN
for col, (neuron_idx, title) in enumerate([
    (high_lrn,
     f'LRN neuron {high_lrn} (rate={rate_per_lrn[high_lrn]:.3f})'),
    (mid_lrn,
     f'LRN neuron {mid_lrn} (rate={rate_per_lrn[mid_lrn]:.3f})'),
]):
    ax = axes[1, col]
    v_data = lrn_vo[:n_show_steps, :, neuron_idx].flatten()
    s_data = lrn_sp[:n_show_steps, :, neuron_idx].flatten()
    t_micro = np.arange(len(v_data))

    ax.plot(t_micro, v_data, color=C_EXC, lw=0.4, alpha=0.9)
    ax.axhline(nf.v_th, color=C_TH, ls='--', lw=0.7, alpha=0.6,
               label=f'$V_{{th}}$={nf.v_th}')
    spike_idx = np.where(s_data > 0.5)[0]
    if len(spike_idx) > 0:
        ax.scatter(spike_idx, np.full_like(spike_idx, nf.v_th, dtype=float),
                   color='red', s=8, zorder=5, marker='v',
                   label=f'Spikes (n={len(spike_idx)})')
        for si in spike_idx:
            ax.axvspan(si, min(si + n_refrac, len(v_data)),
                       alpha=0.15, color=C_REF, lw=0)
    for s in range(1, n_show_steps):
        ax.axvline(s * K, color='black', ls=':', alpha=0.15, lw=0.4)

    ax.set_xlabel('Micro-step')
    ax.set_ylabel('$V_m$')
    ax.set_title(title, color=C_EXC)
    ax.legend(loc='lower right', framealpha=0.8, fontsize=7)
    panel_label(ax, chr(ord('d') + col))

# Row 1, col 2: LRN input–output relationship
ax = axes[1, 2]
ps_e_mean_rate = np.mean(rate_e, axis=0)   # (n_exc,)
lrn_mean_rate  = np.mean(rate_lrn, axis=0) # (n_lrn,)
lrn_input_w = learned['lrn_input_kernel']   # (n_exc, n_lrn)
eff_input = ps_e_mean_rate @ np.abs(lrn_input_w)  # (n_lrn,)
ax.scatter(eff_input, lrn_mean_rate, s=5, alpha=0.6, color=C_EXC)
ax.set_xlabel('Effective PS→LRN input')
ax.set_ylabel('LRN mean firing rate')
ax.set_title('LRN Input–Output Relationship')
panel_label(ax, 'f')

fig.tight_layout()
path = os.path.join(OUTPUT_DIR, "single_neuron_traces.png")
fig.savefig(path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path}")


# =============================================================================
# Figure 3: Firing Statistics
# =============================================================================

print("Plotting firing statistics...")
fig, axes = plt.subplots(2, 4, figsize=(20, 9), squeeze=False)
fig.suptitle("Firing Statistics", fontsize=13, fontweight='bold', y=0.995)

for row, (sp_e_arr, sp_i_arr, ne_count, ni_count, pop_label) in enumerate([
    (ps_sp_e, ps_sp_i, n_exc, n_inh, "ProprioSpinal"),
    (lrn_sp, None, n_lrn, 0, "LRN Relay"),
]):
    sp_e_flat = sp_e_arr.reshape(T_ep * K, ne_count)
    sp_i_flat = (sp_i_arr.reshape(T_ep * K, ni_count)
                 if sp_i_arr is not None else None)

    # --- A: ISI distribution ---
    ax = axes[row, 0]
    isis_e, isis_i = [], []
    for n_idx in range(min(100, ne_count)):
        times = np.where(sp_e_flat[:, n_idx] > 0.5)[0]
        if len(times) > 1:
            isis_e.extend(np.diff(times).tolist())
    if sp_i_flat is not None:
        for n_idx in range(min(100, ni_count)):
            times = np.where(sp_i_flat[:, n_idx] > 0.5)[0]
            if len(times) > 1:
                isis_i.extend(np.diff(times).tolist())

    all_isis = isis_e + isis_i
    if all_isis:
        max_isi = int(np.percentile(all_isis, 98))
        bins_isi = np.arange(1, min(max_isi, 80) + 1)
        if isis_e:
            cv_e = np.std(isis_e) / max(np.mean(isis_e), 1e-8)
            ax.hist(isis_e, bins=bins_isi, alpha=0.6, color=C_EXC,
                    density=True, label=f'E (CV={cv_e:.2f})')
        if isis_i:
            cv_i = np.std(isis_i) / max(np.mean(isis_i), 1e-8)
            ax.hist(isis_i, bins=bins_isi, alpha=0.6, color=C_INH,
                    density=True, label=f'I (CV={cv_i:.2f})')
        ax.axvline(n_refrac + 1, color='black', ls='--', lw=0.8,
                   label=f'Refractory limit ({n_refrac}+1)')
        ax.legend(framealpha=0.8)
    ax.set_xlabel('Inter-spike interval (micro-steps)')
    ax.set_ylabel('Density')
    ax.set_title(f'{pop_label} — ISI Distribution')
    panel_label(ax, chr(ord('a') + row * 4))

    # --- B: Rate distribution ---
    ax = axes[row, 1]
    rate_per_e_pop = np.mean(sp_e_flat, axis=0)
    max_rate = (np.percentile(rate_per_e_pop, 99)
                if ne_count > 0 else 0.01)
    if sp_i_flat is not None:
        rate_per_i_pop = np.mean(sp_i_flat, axis=0)
        max_rate = max(max_rate,
                       np.percentile(rate_per_i_pop, 99)
                       if ni_count > 0 else 0.01)
    max_rate = max(max_rate, 0.01)
    bins_r = np.linspace(0, max_rate, 40)
    ax.hist(rate_per_e_pop, bins=bins_r, alpha=0.6, color=C_EXC,
            label=f'E ($\\mu$={np.mean(rate_per_e_pop):.3f})')
    if sp_i_flat is not None:
        ax.hist(rate_per_i_pop, bins=bins_r, alpha=0.6, color=C_INH,
                label=f'I ($\\mu$={np.mean(rate_per_i_pop):.3f})')
    ax.set_xlabel('Mean firing rate')
    ax.set_ylabel('Neuron count')
    ax.set_title(f'{pop_label} — Rate Distribution')
    silent_e = np.mean(rate_per_e_pop < 1e-6) * 100
    info = f'Silent: E={silent_e:.0f}%'
    if sp_i_flat is not None:
        silent_i = np.mean(rate_per_i_pop < 1e-6) * 100
        info += f'  I={silent_i:.0f}%'
    ax.text(0.95, 0.95, info, transform=ax.transAxes, ha='right', va='top',
            fontsize=7,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray',
                      alpha=0.8))
    ax.legend(framealpha=0.8)
    panel_label(ax, chr(ord('b') + row * 4))

    # --- C: CV of ISI ---
    ax = axes[row, 2]
    cvs_e, cvs_i = [], []
    for n_idx in range(ne_count):
        times = np.where(sp_e_flat[:, n_idx] > 0.5)[0]
        if len(times) > 2:
            isis = np.diff(times).astype(float)
            cvs_e.append(np.std(isis) / max(np.mean(isis), 1e-8))
    if sp_i_flat is not None:
        for n_idx in range(ni_count):
            times = np.where(sp_i_flat[:, n_idx] > 0.5)[0]
            if len(times) > 2:
                isis = np.diff(times).astype(float)
                cvs_i.append(np.std(isis) / max(np.mean(isis), 1e-8))

    if cvs_e or cvs_i:
        max_cv = max(np.percentile(cvs_e, 99) if cvs_e else 2,
                     np.percentile(cvs_i, 99) if cvs_i else 2, 0.1)
        bins_cv = np.linspace(0, min(max_cv, 4), 35)
        if cvs_e:
            ax.hist(cvs_e, bins=bins_cv, alpha=0.6, color=C_EXC,
                    label=f'E ($\\mu$={np.mean(cvs_e):.2f})')
        if cvs_i:
            ax.hist(cvs_i, bins=bins_cv, alpha=0.6, color=C_INH,
                    label=f'I ($\\mu$={np.mean(cvs_i):.2f})')
        ax.axvline(1.0, color='black', ls=':', lw=0.8,
                   label='Poisson (CV=1)')
        ax.legend(framealpha=0.8)
    ax.set_xlabel('CV of ISI')
    ax.set_ylabel('Neuron count')
    ax.set_title(f'{pop_label} — Regularity (CV$_{{ISI}}$)')
    panel_label(ax, chr(ord('c') + row * 4))

    # --- D: Fano factor ---
    ax = axes[row, 3]
    window = max(K * 5, 20)
    n_windows = T_ep * K // window
    if n_windows >= 2:
        counts_e = sp_e_flat[:n_windows * window, :].reshape(
            n_windows, window, ne_count)
        counts_e = np.sum(counts_e > 0.5, axis=1).astype(float)
        fano_e = (np.var(counts_e, axis=0)
                  / np.maximum(np.mean(counts_e, axis=0), 1e-8))
        fano_e_valid = fano_e[np.mean(counts_e, axis=0) > 0.1]

        fano_i_valid = np.array([])
        if sp_i_flat is not None:
            counts_i = sp_i_flat[:n_windows * window, :].reshape(
                n_windows, window, ni_count)
            counts_i = np.sum(counts_i > 0.5, axis=1).astype(float)
            fano_i = (np.var(counts_i, axis=0)
                      / np.maximum(np.mean(counts_i, axis=0), 1e-8))
            fano_i_valid = fano_i[np.mean(counts_i, axis=0) > 0.1]

        max_fano = max(
            np.percentile(fano_e_valid, 99) if len(fano_e_valid) else 2,
            np.percentile(fano_i_valid, 99) if len(fano_i_valid) else 2,
            0.1)
        bins_f = np.linspace(0, min(max_fano, 5), 35)
        if len(fano_e_valid):
            ax.hist(fano_e_valid, bins=bins_f, alpha=0.6, color=C_EXC,
                    label=f'E ($\\mu$={np.mean(fano_e_valid):.2f})')
        if len(fano_i_valid):
            ax.hist(fano_i_valid, bins=bins_f, alpha=0.6, color=C_INH,
                    label=f'I ($\\mu$={np.mean(fano_i_valid):.2f})')
        ax.axvline(1.0, color='black', ls=':', lw=0.8,
                   label='Poisson (FF=1)')
        ax.legend(framealpha=0.8)
    ax.set_xlabel('Fano factor')
    ax.set_ylabel('Neuron count')
    ax.set_title(f'{pop_label} — Fano Factor (window={window} steps)')
    panel_label(ax, chr(ord('d') + row * 4))

fig.tight_layout()
path = os.path.join(OUTPUT_DIR, "firing_statistics.png")
fig.savefig(path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path}")


# =============================================================================
# Figure 4: Network Parameters & E-I Coupling
# =============================================================================

print("Plotting network parameters...")
fig, axes = plt.subplots(2, 4, figsize=(20, 9), squeeze=False)
fig.suptitle("Network Parameters & E–I Coupling",
             fontsize=13, fontweight='bold', y=0.995)

# --- Row 0: ProprioSpinal ---
ps_tau_e = learned['ps_tau_m'][:n_exc]
ps_tau_i = learned['ps_tau_m'][n_exc:]

# A: tau_m distributions
ax = axes[0, 0]
bins_tau = np.linspace(min(ps_tau_e.min(), ps_tau_i.min()),
                       max(ps_tau_e.max(), ps_tau_i.max()), 35)
ax.hist(ps_tau_e, bins=bins_tau, alpha=0.6, color=C_EXC,
        label=f'E ($\\mu$={np.mean(ps_tau_e):.2f})')
ax.hist(ps_tau_i, bins=bins_tau, alpha=0.6, color=C_INH,
        label=f'I ($\\mu$={np.mean(ps_tau_i):.2f})')
ax.set_xlabel('$\\tau_m$ (time constant)')
ax.set_ylabel('Neuron count')
ax.set_title('ProprioSpinal — Membrane Time Constants')
ax.legend(framealpha=0.8)
panel_label(ax, 'a')

# B: Lateral weight magnitudes
ax = axes[0, 1]
w_ie = np.abs(learned['ps_W_ie']).flatten()
w_ei = np.abs(learned['ps_W_ei']).flatten()
max_w = max(np.percentile(w_ie, 99), np.percentile(w_ei, 99))
bins_w = np.linspace(0, max_w, 40)
ax.hist(w_ie, bins=bins_w, alpha=0.6, color=C_INH,
        label=f'$|W_{{I \\to E}}|$ ($\\mu$={np.mean(w_ie):.4f})')
ax.hist(w_ei, bins=bins_w, alpha=0.6, color=C_EXC,
        label=f'$|W_{{E \\to I}}|$ ($\\mu$={np.mean(w_ei):.4f})')
ax.set_xlabel('Weight magnitude')
ax.set_ylabel('Count')
ax.set_title('ProprioSpinal — Lateral Weight Distributions')
ax.legend(framealpha=0.8)
panel_label(ax, 'b')

# C: Effective synaptic drive
ax = axes[0, 2]
total_inh_per_e = np.sum(np.abs(learned['ps_W_ie']), axis=0)
total_exc_per_i = np.sum(np.abs(learned['ps_W_ei']), axis=0)
bp = ax.boxplot([total_inh_per_e, total_exc_per_i],
                labels=['Total I→E\nper E neuron', 'Total E→I\nper I neuron'],
                patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor(C_INH)
bp['boxes'][0].set_alpha(0.4)
bp['boxes'][1].set_facecolor(C_EXC)
bp['boxes'][1].set_alpha(0.4)
ax.set_ylabel('Total synaptic weight')
ax.set_title('ProprioSpinal — Effective Connectivity')
panel_label(ax, 'c')

# D: E-I Cross-Correlogram
ax = axes[0, 3]
rate_e_ts = np.mean(ps_sp_e, axis=(1, 2))  # (T,)
rate_i_ts = np.mean(ps_sp_i, axis=(1, 2))  # (T,)
rate_e_z = (rate_e_ts - np.mean(rate_e_ts)) / max(np.std(rate_e_ts), 1e-8)
rate_i_z = (rate_i_ts - np.mean(rate_i_ts)) / max(np.std(rate_i_ts), 1e-8)
max_lag = min(20, T_ep // 4)
lags = np.arange(-max_lag, max_lag + 1)
xcorr = np.correlate(rate_e_z, rate_i_z, mode='full')
xcorr = xcorr / T_ep
mid_idx = len(xcorr) // 2
xcorr_window = xcorr[mid_idx - max_lag: mid_idx + max_lag + 1]
ax.bar(lags, xcorr_window, color=C_BAL, alpha=0.7, width=0.8)
ax.axvline(0, color='black', ls='--', lw=0.6, alpha=0.4)
ax.axhline(0, color='black', lw=0.4)
peak_lag = lags[np.argmax(xcorr_window)]
ax.set_xlabel('Lag (env steps, E leads →)')
ax.set_ylabel('Cross-correlation')
ax.set_title(f'PS — E–I Cross-Correlogram (peak at lag {peak_lag})')
panel_label(ax, 'd')

# --- Row 1: LRN + Cerebellum parameters ---
# E: LRN tau_m
ax = axes[1, 0]
lrn_tau = learned['lrn_tau_m']
ax.hist(lrn_tau, bins=35, alpha=0.7, color=C_EXC,
        label=f'LRN ($\\mu$={np.mean(lrn_tau):.2f})')
ax.set_xlabel('$\\tau_m$ (time constant)')
ax.set_ylabel('Neuron count')
ax.set_title('LRN — Membrane Time Constants')
ax.legend(framealpha=0.8)
panel_label(ax, 'e')

# F: Cerebellum F matrix eigenvalue spectrum
ax = axes[1, 1]
F_mat = learned['cb_F']
eigvals = np.linalg.eigvals(F_mat)
ax.scatter(eigvals.real, eigvals.imag, s=10, alpha=0.7, color=C_CB)
theta = np.linspace(0, 2 * np.pi, 100)
ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=0.6, alpha=0.4,
        label='unit circle')
ax.set_xlabel('Re($\\lambda$)')
ax.set_ylabel('Im($\\lambda$)')
ax.set_title('Cerebellum F — Eigenvalue Spectrum')
ax.set_aspect('equal')
ax.legend(framealpha=0.8, fontsize=7)
panel_label(ax, 'f')

# G: Cerebellum B and H weight distributions
ax = axes[1, 2]
b_flat = np.abs(learned['cb_B']).flatten()
h_flat = np.abs(learned['cb_H']).flatten()
max_bh = max(np.percentile(b_flat, 99), np.percentile(h_flat, 99))
bins_bh = np.linspace(0, max_bh, 40)
ax.hist(b_flat, bins=bins_bh, alpha=0.6, color=C_CB,
        label=f'$|B|$ (motor→state, $\\mu$={np.mean(b_flat):.4f})')
ax.hist(h_flat, bins=bins_bh, alpha=0.6, color=C_BAL,
        label=f'$|H|$ (state→obs, $\\mu$={np.mean(h_flat):.4f})')
ax.set_xlabel('Weight magnitude')
ax.set_ylabel('Count')
ax.set_title('Cerebellum — B and H Matrices')
ax.legend(framealpha=0.8, fontsize=7)
panel_label(ax, 'g')

# H: Innovation and correction norms over episode
ax = axes[1, 3]
corr_norm = np.sqrt(np.sum(cb_corr ** 2, axis=-1))
ax.plot(cb_innov_norm, color=C_BAL, lw=1.0, alpha=0.7,
        label='||innovation||')
ax.plot(corr_norm, color=C_CB, lw=1.0,
        label='||correction||')
ax.set_xlabel('Time (env steps)')
ax.set_ylabel('Magnitude')
ax.set_title(f'Cerebellum — Signals (w={correction_weight.item():.4f})')
ax.legend(framealpha=0.8, fontsize=7)
panel_label(ax, 'h')

fig.tight_layout()
path = os.path.join(OUTPUT_DIR, "network_params.png")
fig.savefig(path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path}")


# =============================================================================
# Figure 5: Cerebellum (Kalman Filter) Diagnostics
# =============================================================================

print("Plotting cerebellum diagnostics...")
fig, axes = plt.subplots(2, 3, figsize=(18, 9), squeeze=False)
fig.suptitle("Cerebellum (Kalman Filter) Diagnostics",
             fontsize=13, fontweight='bold', y=0.995)

# A: Innovation norm
ax = axes[0, 0]
ax.plot(cb_innov_norm, color=C_CB, lw=1.0)
ax.set_xlabel('Time (env steps)')
ax.set_ylabel('||innovation||')
ax.set_title('Prediction Error (Innovation Norm)')
panel_label(ax, 'a')

# B: State estimate traces
ax = axes[0, 1]
n_dims = min(8, cb_x_hat.shape[1])
for d in range(n_dims):
    ax.plot(cb_x_hat[:, d], alpha=0.6, lw=0.8)
ax.set_xlabel('Time (env steps)')
ax.set_ylabel('$\\hat{x}$')
ax.set_title(f'State Estimate (first {n_dims} dims)')
panel_label(ax, 'b')

# C: Correction per actuator
ax = axes[0, 2]
n_act = min(8, cb_corr.shape[1])
for d in range(n_act):
    ax.plot(cb_corr[:, d], alpha=0.6, lw=0.8)
ax.set_xlabel('Time (env steps)')
ax.set_ylabel('Correction')
ax.set_title(f'Correction Signals (first {n_act} actuators)')
panel_label(ax, 'c')

# D: Covariance diagonal traces
ax = axes[1, 0]
for d in range(n_dims):
    ax.plot(cb_P[:, d], alpha=0.6, lw=0.8)
ax.set_xlabel('Time (env steps)')
ax.set_ylabel('$P_{diag}$')
ax.set_title(f'Covariance Diagonal (first {n_dims} dims)')
panel_label(ax, 'd')

# E: Raw motor command vs cerebellar correction contribution
ax = axes[1, 1]
raw_norm = np.sqrt(np.sum(raw_motor ** 2, axis=-1))
corr_contrib = correction_weight.item() * corr_norm
ax.plot(raw_norm, color=C_EXC, lw=1.0, label='||raw motor||')
ax.plot(corr_contrib, color=C_CB, lw=1.0,
        label=f'w*||correction|| (w={correction_weight.item():.4f})')
ax.set_xlabel('Time (env steps)')
ax.set_ylabel('Magnitude')
ax.set_title('Motor Command vs Cerebellar Correction')
ax.legend(framealpha=0.8, fontsize=7)
panel_label(ax, 'e')

# F: Episode reward
ax = axes[1, 2]
ax.plot(rewards, color='green', lw=1.0)
ax.set_xlabel('Time (env steps)')
ax.set_ylabel('Reward')
ax.set_title(f'Episode Reward (total={np.sum(rewards):.1f})')
panel_label(ax, 'f')

fig.tight_layout()
path = os.path.join(OUTPUT_DIR, "cerebellum_diagnostics.png")
fig.savefig(path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path}")


# =============================================================================
# Spike-train spectral & synchrony analysis (Varela-style)
# =============================================================================

print("Computing spike-train firing rate signals at micro-step resolution...")

# Population spike rates at micro-step resolution: mean across neurons
# ps_sp_e: (T, K, n_exc) → flatten to (T*K, n_exc) → mean across neurons
sp_e_flat = ps_sp_e.reshape(-1, n_exc)   # (T*K, n_exc)  binary spikes
sp_i_flat = ps_sp_i.reshape(-1, n_inh)   # (T*K, n_inh)
sp_l_flat = lrn_sp.reshape(-1, n_lrn)    # (T*K, n_lrn)

pop_rate_e = np.mean(sp_e_flat, axis=1)  # population firing rate (T*K,)
pop_rate_i = np.mean(sp_i_flat, axis=1)
pop_rate_l = np.mean(sp_l_flat, axis=1)

# Sampling frequency = 1 / micro-step dt
env_ctrl_dt = 0.0025   # from base config ctrl_dt
dt_micro = env_ctrl_dt / K  # 0.0003125 s per micro-step
fs = 1.0 / dt_micro         # 3200 Hz
nyquist = fs / 2.0

n_total = len(pop_rate_e)  # T_ep * K
t_micro_ms = np.arange(n_total) * dt_micro * 1000  # time in ms

print(f"  dt_micro = {dt_micro*1e6:.1f} us, fs = {fs:.0f} Hz, "
      f"Nyquist = {nyquist:.0f} Hz")
print(f"  Signal length: {n_total} samples ({n_total * dt_micro * 1000:.1f} ms)")

# Welch PSD of population firing rates
nperseg = min(2048, n_total // 4)
nperseg = max(nperseg, 256)
noverlap = nperseg // 2

freq_e, psd_e = welch(pop_rate_e, fs=fs, nperseg=nperseg, noverlap=noverlap)
freq_i, psd_i = welch(pop_rate_i, fs=fs, nperseg=nperseg, noverlap=noverlap)
freq_l, psd_l = welch(pop_rate_l, fs=fs, nperseg=nperseg, noverlap=noverlap)


# --- Helper: bandpass filter + Hilbert for Phase Locking Value (PLV) ---

def bandpass(signal, f_lo, f_hi, fs, order=3):
    """Zero-phase Butterworth bandpass filter."""
    fny = fs / 2.0
    lo = max(f_lo / fny, 0.001)
    hi = min(f_hi / fny, 0.999)
    if lo >= hi:
        return np.zeros_like(signal)
    b, a = butter(order, [lo, hi], btype='band')
    return filtfilt(b, a, signal)


def phase_locking_value(sig_a, sig_b, fs, f_lo, f_hi):
    """Compute PLV between two signals in a given frequency band.

    Filters both signals in [f_lo, f_hi], extracts instantaneous phase
    via Hilbert transform, then PLV = |mean(exp(j * dphi))|.
    Reference: Lachaux et al. (1999), as used in Varela et al. (2001).
    """
    fa = bandpass(sig_a, f_lo, f_hi, fs)
    fb = bandpass(sig_b, f_lo, f_hi, fs)
    phase_a = np.angle(hilbert(fa))
    phase_b = np.angle(hilbert(fb))
    dphi = phase_a - phase_b
    plv = np.abs(np.mean(np.exp(1j * dphi)))
    return plv, dphi


def windowed_plv(sig_a, sig_b, fs, f_lo, f_hi, win_samples):
    """Compute PLV in sliding windows for time-resolved synchrony."""
    fa = bandpass(sig_a, f_lo, f_hi, fs)
    fb = bandpass(sig_b, f_lo, f_hi, fs)
    phase_a = np.angle(hilbert(fa))
    phase_b = np.angle(hilbert(fb))
    dphi = phase_a - phase_b
    n_wins = len(dphi) // win_samples
    plv_t = np.zeros(n_wins)
    for w in range(n_wins):
        s = w * win_samples
        e = s + win_samples
        plv_t[w] = np.abs(np.mean(np.exp(1j * dphi[s:e])))
    return plv_t


# =============================================================================
# Figure 6: Firing Rate Spectral Analysis
# =============================================================================

print("Plotting firing rate spectral analysis...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)
fig.suptitle("Population Firing Rate — Spectral Analysis",
             fontsize=13, fontweight='bold', y=0.995)

# --- Row 0: Population firing rate time series ---

# Panel A: PS E population rate
ax = axes[0, 0]
sigma_smooth = max(K * 2, 16)
pop_rate_e_s = gaussian_filter1d(pop_rate_e, sigma_smooth)
ax.plot(t_micro_ms, pop_rate_e, color=C_EXC, lw=0.2, alpha=0.4)
ax.plot(t_micro_ms, pop_rate_e_s, color=C_EXC, lw=1.2)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Population rate')
ax.set_title('PS Excitatory — Firing Rate', color=C_EXC)
ax.set_xlim(t_micro_ms[0], t_micro_ms[-1])
panel_label(ax, 'a')

# Panel B: PS I population rate
ax = axes[0, 1]
pop_rate_i_s = gaussian_filter1d(pop_rate_i, sigma_smooth)
ax.plot(t_micro_ms, pop_rate_i, color=C_INH, lw=0.2, alpha=0.4)
ax.plot(t_micro_ms, pop_rate_i_s, color=C_INH, lw=1.2)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Population rate')
ax.set_title('PS Inhibitory — Firing Rate', color=C_INH)
ax.set_xlim(t_micro_ms[0], t_micro_ms[-1])
panel_label(ax, 'b')

# Panel C: LRN population rate
ax = axes[0, 2]
pop_rate_l_s = gaussian_filter1d(pop_rate_l, sigma_smooth)
ax.plot(t_micro_ms, pop_rate_l, color='#2ca02c', lw=0.2, alpha=0.4)
ax.plot(t_micro_ms, pop_rate_l_s, color='#2ca02c', lw=1.2)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Population rate')
ax.set_title('LRN Relay — Firing Rate', color='#2ca02c')
ax.set_xlim(t_micro_ms[0], t_micro_ms[-1])
panel_label(ax, 'c')

# --- Row 1: Spectral analysis ---

# Panel D: PSD overlay (loglog)
ax = axes[1, 0]
ax.loglog(freq_e[1:], psd_e[1:], color=C_EXC, lw=1.2, label='PS E')
ax.loglog(freq_i[1:], psd_i[1:], color=C_INH, lw=1.2, label='PS I')
ax.loglog(freq_l[1:], psd_l[1:], color='#2ca02c', lw=1.2, label='LRN')
ax.axvline(nyquist, color='red', ls=':', lw=0.8, alpha=0.5,
           label=f'Nyquist ({nyquist:.0f} Hz)')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD (rate$^2$/Hz)')
ax.set_title('Power Spectral Density (Welch)')
ax.legend(framealpha=0.8, fontsize=7)
ax.grid(True, which='both', alpha=0.2)
panel_label(ax, 'd')

# Panel E: Band power decomposition
ax = axes[1, 1]
bands = {
    'Delta\n(1-4)':    (1, 4),
    'Theta\n(4-8)':    (4, 8),
    'Alpha\n(8-13)':   (8, 13),
    'Beta\n(13-30)':   (13, 30),
    'Low-$\\gamma$\n(30-80)':  (30, 80),
    'High-$\\gamma$\n(80-200)': (80, 200),
}
x_pos = np.arange(len(bands))
width = 0.25

for idx, (label, freq_arr, psd_arr, color) in enumerate([
    ('PS E', freq_e, psd_e, C_EXC),
    ('PS I', freq_i, psd_i, C_INH),
    ('LRN',  freq_l, psd_l, '#2ca02c'),
]):
    band_powers = []
    for band_name, (f_lo, f_hi) in bands.items():
        mask = (freq_arr >= f_lo) & (freq_arr <= f_hi)
        if np.any(mask):
            band_powers.append(np.trapz(psd_arr[mask], freq_arr[mask]))
        else:
            band_powers.append(0.0)
    ax.bar(x_pos + (idx - 1) * width, np.maximum(band_powers, 1e-20),
           width, color=color, alpha=0.7, label=label)

ax.set_xticks(x_pos)
ax.set_xticklabels(list(bands.keys()), fontsize=7)
ax.set_ylabel('Band Power')
ax.set_title('Frequency Band Decomposition')
ax.set_yscale('log')
ax.legend(framealpha=0.8, fontsize=7)
panel_label(ax, 'e')

# Panel F: Spectrogram of PS E firing rate
ax = axes[1, 2]
nperseg_sg = min(256, n_total // 8)
nperseg_sg = max(nperseg_sg, 64)
f_sg, t_sg, Sxx = spectrogram(pop_rate_e, fs=fs,
                                nperseg=nperseg_sg,
                                noverlap=nperseg_sg // 2)
f_mask = f_sg <= 500
Sxx_db = 10 * np.log10(Sxx[f_mask] + 1e-20)
im = ax.pcolormesh(t_sg * 1000, f_sg[f_mask], Sxx_db,
                    shading='gouraud', cmap='viridis')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Frequency (Hz)')
ax.set_title('PS E — Rate Spectrogram (dB)')
fig.colorbar(im, ax=ax, label='Power (dB)', pad=0.02, aspect=30)
panel_label(ax, 'f')

fig.tight_layout()
path = os.path.join(OUTPUT_DIR, "firing_rate_spectral.png")
fig.savefig(path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path}")


# =============================================================================
# Figure 7: Spike-Train Synchrony (Varela-style)
#   - PLV (Lachaux et al. 1999) across frequency bands
#   - Spike-train coherence spectra
#   - Cross-correlograms at micro-step resolution
#   - Windowed PLV for dynamic synchrony
# =============================================================================

print("Computing Varela-style spike-train synchrony...")

# PLV across canonical frequency bands
plv_bands = {
    'Delta (1-4)':    (1, 4),
    'Theta (4-8)':    (4, 8),
    'Alpha (8-13)':   (8, 13),
    'Beta (13-30)':   (13, 30),
    'Low-g (30-80)':  (30, 80),
    'High-g (80-200)': (80, 200),
}

plv_results = {}  # (pair, band) -> plv
for pair_name, (sig_a, sig_b) in [
    ('PS E-PS I', (pop_rate_e, pop_rate_i)),
    ('PS E-LRN',  (pop_rate_e, pop_rate_l)),
    ('PS I-LRN',  (pop_rate_i, pop_rate_l)),
]:
    for band_name, (f_lo, f_hi) in plv_bands.items():
        if f_hi < nyquist:
            plv_val, _ = phase_locking_value(sig_a, sig_b, fs, f_lo, f_hi)
        else:
            plv_val = 0.0
        plv_results[(pair_name, band_name)] = plv_val

print("Plotting spike-train synchrony...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)
fig.suptitle("Spike-Train Synchrony (Varela / Lachaux)",
             fontsize=13, fontweight='bold', y=0.995)

# Panel A: PLV bar chart across bands for all pairs
ax = axes[0, 0]
x_pos = np.arange(len(plv_bands))
width = 0.25
pair_colors = {'PS E-PS I': C_BAL, 'PS E-LRN': C_EXC, 'PS I-LRN': C_INH}
for idx, pair_name in enumerate(['PS E-PS I', 'PS E-LRN', 'PS I-LRN']):
    plv_vals = [plv_results[(pair_name, b)] for b in plv_bands.keys()]
    ax.bar(x_pos + (idx - 1) * width, plv_vals, width,
           color=pair_colors[pair_name], alpha=0.7, label=pair_name)
ax.set_xticks(x_pos)
ax.set_xticklabels(list(plv_bands.keys()), fontsize=7, rotation=30, ha='right')
ax.set_ylabel('Phase Locking Value')
ax.set_ylim(0, 1)
ax.axhline(0.1, color='gray', ls='--', lw=0.6, alpha=0.5,
           label='Weak sync threshold')
ax.set_title('PLV Across Frequency Bands')
ax.legend(framealpha=0.8, fontsize=7, loc='upper right')
panel_label(ax, 'a')

# Panel B: Spike-train coherence PS E vs PS I
ax = axes[0, 1]
nperseg_coh = min(1024, n_total // 4)
nperseg_coh = max(nperseg_coh, 256)
f_coh, coh_ei = coherence(pop_rate_e, pop_rate_i, fs=fs,
                           nperseg=nperseg_coh, noverlap=nperseg_coh // 2)
f_coh, coh_el = coherence(pop_rate_e, pop_rate_l, fs=fs,
                           nperseg=nperseg_coh, noverlap=nperseg_coh // 2)
f_coh, coh_il = coherence(pop_rate_i, pop_rate_l, fs=fs,
                           nperseg=nperseg_coh, noverlap=nperseg_coh // 2)
ax.semilogy(f_coh, coh_ei, color=C_BAL, lw=1.2, label='PS E-PS I')
ax.semilogy(f_coh, coh_el, color=C_EXC, lw=1.2, label='PS E-LRN')
ax.semilogy(f_coh, coh_il, color=C_INH, lw=1.2, label='PS I-LRN')
ax.set_xlim(0, min(500, nyquist))
ax.set_ylim(1e-3, 1)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Coherence')
ax.set_title('Spike-Train Coherence Spectra')
ax.axhline(0.05, color='gray', ls='--', lw=0.6, alpha=0.5)
ax.legend(framealpha=0.8, fontsize=7)
ax.grid(True, which='both', alpha=0.2)
panel_label(ax, 'b')

# Panel C: Cross-correlograms at micro-step resolution
ax = axes[0, 2]
max_lag_sp = min(K * 20, n_total // 4)
re_z = (pop_rate_e - np.mean(pop_rate_e)) / max(np.std(pop_rate_e), 1e-8)
ri_z = (pop_rate_i - np.mean(pop_rate_i)) / max(np.std(pop_rate_i), 1e-8)
rl_z = (pop_rate_l - np.mean(pop_rate_l)) / max(np.std(pop_rate_l), 1e-8)

xcorr_ei = np.correlate(re_z, ri_z, mode='full') / n_total
xcorr_el = np.correlate(re_z, rl_z, mode='full') / n_total
mid_x = len(xcorr_ei) // 2
lags = np.arange(-max_lag_sp, max_lag_sp + 1)
lag_ms = lags * dt_micro * 1000

ax.plot(lag_ms, xcorr_ei[mid_x - max_lag_sp: mid_x + max_lag_sp + 1],
        color=C_BAL, lw=1.0, label='PS E-PS I')
ax.plot(lag_ms, xcorr_el[mid_x - max_lag_sp: mid_x + max_lag_sp + 1],
        color=C_EXC, lw=1.0, label='PS E-LRN')
ax.axvline(0, color='black', ls='--', lw=0.5, alpha=0.4)
ax.axhline(0, color='black', lw=0.3)
ax.set_xlabel('Lag (ms)')
ax.set_ylabel('Cross-correlation')
ax.set_title('Spike-Train Cross-Correlograms')
ax.legend(framealpha=0.8, fontsize=7)
panel_label(ax, 'c')

# Panel D: Time-resolved PLV (windowed)
ax = axes[1, 0]
win_plv = int(0.050 * fs)  # 50 ms windows
win_plv = max(win_plv, K * 4)

plv_time_data = {}
for band_label, (f_lo, f_hi) in [('Beta', (13, 30)), ('Low-$\\gamma$', (30, 80))]:
    if f_hi < nyquist:
        plv_t_ei = windowed_plv(pop_rate_e, pop_rate_i, fs, f_lo, f_hi, win_plv)
        plv_time_data[(band_label, 'PS E-PS I')] = plv_t_ei

n_plv_wins = n_total // win_plv
plv_t_ms = np.arange(n_plv_wins) * win_plv * dt_micro * 1000

for (band_label, pair), plv_t in plv_time_data.items():
    n_show = min(len(plv_t), len(plv_t_ms))
    ax.plot(plv_t_ms[:n_show], plv_t[:n_show], lw=1.0,
            label=f'{band_label}')
ax.axhline(0.3, color='gray', ls='--', lw=0.6, alpha=0.5,
           label='Moderate sync')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('PLV')
ax.set_ylim(0, 1)
ax.set_title('PS E-PS I — Windowed PLV Over Time')
ax.legend(framealpha=0.8, fontsize=7)
if len(plv_t_ms) > 0:
    ax.set_xlim(plv_t_ms[0], plv_t_ms[-1])
panel_label(ax, 'd')

# Panel E: Pairwise spike-train synchrony over time
# Use windowed correlation of population rates as a model-free synchrony index
ax = axes[1, 1]
win_sync = K * 8  # ~2.5 ms window
n_sync_wins = n_total // win_sync

def _windowed_corr(a, b):
    """Pearson correlation between two short windows."""
    a_z = a - np.mean(a)
    b_z = b - np.mean(b)
    denom = max(np.std(a), 1e-8) * max(np.std(b), 1e-8) * len(a)
    return np.dot(a_z, b_z) / denom

sync_ei, sync_el, sync_il = [], [], []
for w in range(n_sync_wins):
    s = w * win_sync
    e_idx = s + win_sync
    chunk_re = pop_rate_e[s:e_idx]
    chunk_ri = pop_rate_i[s:e_idx]
    chunk_rl = pop_rate_l[s:e_idx]
    sync_ei.append(_windowed_corr(chunk_re, chunk_ri))
    sync_el.append(_windowed_corr(chunk_re, chunk_rl))
    sync_il.append(_windowed_corr(chunk_ri, chunk_rl))

sync_t_ms = np.arange(n_sync_wins) * win_sync * dt_micro * 1000
ax.plot(sync_t_ms, gaussian_filter1d(np.array(sync_ei), 3),
        color=C_BAL, lw=1.0, label='PS E-PS I')
ax.plot(sync_t_ms, gaussian_filter1d(np.array(sync_el), 3),
        color=C_EXC, lw=1.0, label='PS E-LRN')
ax.plot(sync_t_ms, gaussian_filter1d(np.array(sync_il), 3),
        color=C_INH, lw=1.0, label='PS I-LRN')
ax.axhline(0, color='black', lw=0.3)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Windowed correlation')
ax.set_title('Pairwise Rate Synchrony Over Time')
ax.legend(framealpha=0.8, fontsize=7)
if len(sync_t_ms) > 0:
    ax.set_xlim(sync_t_ms[0], sync_t_ms[-1])
panel_label(ax, 'e')

# Panel F: Rate distributions — per neuron and per timestep
ax = axes[1, 2]
# Per-neuron mean rate (across entire episode)
rate_per_neuron_e = np.mean(sp_e_flat, axis=0)  # (n_exc,)
rate_per_neuron_i = np.mean(sp_i_flat, axis=0)  # (n_inh,)
rate_per_neuron_l = np.mean(sp_l_flat, axis=0)  # (n_lrn,)
all_rates = np.concatenate([rate_per_neuron_e, rate_per_neuron_i,
                            rate_per_neuron_l])
max_r = max(np.percentile(all_rates, 99), 0.01)
bins_r = np.linspace(0, max_r, 50)
ax.hist(rate_per_neuron_e, bins=bins_r, alpha=0.5, color=C_EXC, density=True,
        label=f'PS E ($\\mu$={np.mean(rate_per_neuron_e):.4f})')
ax.hist(rate_per_neuron_i, bins=bins_r, alpha=0.5, color=C_INH, density=True,
        label=f'PS I ($\\mu$={np.mean(rate_per_neuron_i):.4f})')
ax.hist(rate_per_neuron_l, bins=bins_r, alpha=0.5, color='#2ca02c', density=True,
        label=f'LRN ($\\mu$={np.mean(rate_per_neuron_l):.4f})')
ax.set_xlabel('Mean firing rate (spikes/micro-step)')
ax.set_ylabel('Density')
ax.set_title('Per-Neuron Rate Distributions')
ax.legend(framealpha=0.8, fontsize=7)
panel_label(ax, 'f')

fig.tight_layout()
path = os.path.join(OUTPUT_DIR, "spike_synchrony.png")
fig.savefig(path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path}")


# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY — LRN-Cerebellum Circuit")
print("=" * 70)

# ProprioSpinal
rate_e_mean = np.mean(ps_sp_e)
rate_i_mean = np.mean(ps_sp_i)
v_e_mean = np.mean(ps_vo_e)
v_i_mean = np.mean(ps_vo_i)
ps_tau_e_mean = np.mean(ps_tau_e)
ps_tau_i_mean = np.mean(ps_tau_i)
rate_per_e_all = np.mean(ps_sp_e, axis=(0, 1))
rate_per_i_all = np.mean(ps_sp_i, axis=(0, 1))
silent_e_pct = np.mean(rate_per_e_all < 1e-6) * 100
silent_i_pct = np.mean(rate_per_i_all < 1e-6) * 100

print(f"\nProprioSpinal ({n_exc}E / {n_inh}I):")
print(f"  Firing rates:  E={rate_e_mean:.4f}  I={rate_i_mean:.4f}  "
      f"ratio={rate_e_mean / max(rate_i_mean, 1e-8):.2f}")
print(f"  Mean voltage:  E={v_e_mean:.4f}  I={v_i_mean:.4f}")
print(f"  Mean tau_m:    E={ps_tau_e_mean:.3f}  I={ps_tau_i_mean:.3f}")
print(f"  Mean |W_ie|:   {np.mean(np.abs(learned['ps_W_ie'])):.5f}")
print(f"  Mean |W_ei|:   {np.mean(np.abs(learned['ps_W_ei'])):.5f}")
print(f"  Silent:        E={silent_e_pct:.1f}%  I={silent_i_pct:.1f}%")

# LRN
lrn_rate_mean = np.mean(lrn_sp)
lrn_v_mean = np.mean(lrn_vo)
lrn_tau_mean = np.mean(learned['lrn_tau_m'])
rate_per_lrn_all = np.mean(lrn_sp, axis=(0, 1))
silent_lrn_pct = np.mean(rate_per_lrn_all < 1e-6) * 100

print(f"\nLRN Relay ({n_lrn}E):")
print(f"  Firing rate:   {lrn_rate_mean:.4f}")
print(f"  Mean voltage:  {lrn_v_mean:.4f}")
print(f"  Mean tau_m:    {lrn_tau_mean:.3f}")
print(f"  Silent:        {silent_lrn_pct:.1f}%")
print(f"  PN→LRN corr:   {r_corr:.3f}")

# Cerebellum
spectral_radius = np.max(np.abs(eigvals))
print(f"\nCerebellum (state_dim={cb_state_dim}):")
print(f"  Correction weight: {correction_weight.item():.6f}")
print(f"  Mean ||innovation||: {np.mean(cb_innov_norm):.4f}")
print(f"  Mean ||correction||: {np.mean(corr_norm):.4f}")
print(f"  F spectral radius: {spectral_radius:.4f}")
print(f"  Mean P_diag: {np.mean(cb_P):.4f}")

print(f"\nEpisode reward: {np.sum(rewards):.1f}")

# Spectral stats
print(f"\nFiring Rate Spectral Analysis:")
print(f"  Sampling: dt_micro={dt_micro*1e6:.1f} us, fs={fs:.0f} Hz, "
      f"Nyquist={nyquist:.0f} Hz")
print(f"  Signal length: {n_total} samples ({n_total * dt_micro * 1000:.1f} ms)")
print(f"  Welch nperseg={nperseg}, noverlap={noverlap}")
peak_f_e = freq_e[1:][np.argmax(psd_e[1:])]
peak_f_i = freq_i[1:][np.argmax(psd_i[1:])]
peak_f_l = freq_l[1:][np.argmax(psd_l[1:])]
print(f"  Peak PSD freq:  PS E={peak_f_e:.1f} Hz, "
      f"PS I={peak_f_i:.1f} Hz, LRN={peak_f_l:.1f} Hz")

# PLV stats
print(f"\nPhase Locking Values (Varela/Lachaux):")
for pair_name in ['PS E-PS I', 'PS E-LRN', 'PS I-LRN']:
    vals = [f"{b}={plv_results[(pair_name, b)]:.3f}"
            for b in plv_bands.keys()
            if plv_results[(pair_name, b)] > 0]
    print(f"  {pair_name}: {', '.join(vals)}")

# Coherence stats
peak_coh_ei = np.max(coh_ei)
peak_coh_el = np.max(coh_el)
peak_coh_il = np.max(coh_il)
peak_coh_ei_f = f_coh[np.argmax(coh_ei)]
peak_coh_el_f = f_coh[np.argmax(coh_el)]
peak_coh_il_f = f_coh[np.argmax(coh_il)]
print(f"\nSpike-train coherence peaks:")
print(f"  PS E-PS I: {peak_coh_ei:.3f} at {peak_coh_ei_f:.1f} Hz")
print(f"  PS E-LRN:  {peak_coh_el:.3f} at {peak_coh_el_f:.1f} Hz")
print(f"  PS I-LRN:  {peak_coh_il:.3f} at {peak_coh_il_f:.1f} Hz")

print(f"\nAll figures saved to: {OUTPUT_DIR}/")
print("  population_activity.png — rasters, PSTH, E/I balance")
print("  single_neuron_traces.png — intracellular-style voltage traces")
print("  firing_statistics.png — ISI, rate distributions, CV, Fano factor")
print("  network_params.png — tau_m, weights, F spectrum, B/H distributions")
print("  cerebellum_diagnostics.png — innovation, state, correction, covariance")
print("  firing_rate_spectral.png — population rate PSD (Welch), band power, spectrogram")
print("  spike_synchrony.png — PLV, coherence, cross-correlograms, windowed PLV")
print("=" * 70)
