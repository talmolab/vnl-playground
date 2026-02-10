"""Inhibitory stimulation study: dose-response of amplified I→E inhibition.

Loads a trained checkpoint, scales up the I→E lateral weights (W_ie) by
several factors, and measures how behavior degrades as inhibition increases.
Analogous to a GABAergic agonist dose-response experiment.

Outputs:
  - dose_response.png : reward, E/I rates, balance vs inhibition strength
  - population_comparison.png : per-layer E/I activity across conditions
  - comparison.mp4 : side-by-side video (normal vs strongest stimulation)

Usage:
    python stimulation.py [checkpoint_path] [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d

import jax
import jax.numpy as jp
from brax.training import distribution, networks as brax_networks
from brax.training.acme import running_statistics
from flax import linen
from flax.core import unfreeze, freeze
from orbax import checkpoint as ocp

from vnl_playground.tasks.mouse.imitation import MouseImitation, default_config
from vnl_playground.tasks.mouse.reference_clips import MouseReferenceClips
from vnl_playground.tasks.mouse import consts

from train_mouse_LIF import (
    BiologicalSpikingPolicy,
    DiagnosticBiologicalPolicy,
    flatten_obs,
    ppo_params,
)

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

# =============================================================================
# Plot style
# =============================================================================

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.family': 'sans-serif', 'axes.linewidth': 0.8,
})

# Dose-response color map: darker = more inhibition
SCALE_FACTORS = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
CMAP = plt.cm.RdPu  # light pink → deep magenta
COLORS = [CMAP(0.15 + 0.8 * i / (len(SCALE_FACTORS) - 1)) for i in range(len(SCALE_FACTORS))]
C_EXC = '#1f77b4'
C_INH = '#d62728'

# =============================================================================
# Args
# =============================================================================

DEFAULT_CKPT = (
    "/root/vast/eric/vnl-playground/checkpoints/"
    "mouse-imitation-biological-lif-20260202-072917/100024320"
)
CHECKPOINT_PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT
OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "stimulation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"Output dir: {OUTPUT_DIR}")

# =============================================================================
# Environment setup
# =============================================================================

nf = ppo_params.network_factory
env_cfg = default_config()
print(f"Loading reference clips...")
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
print(f"Episode length: {episode_length}")

obs_size = eval_env.observation_size
if isinstance(obs_size, (tuple, list)):
    obs_size = obs_size[-1]
act_size = eval_env.action_size

# =============================================================================
# Network setup & checkpoint
# =============================================================================

action_dist = distribution.NormalTanhDistribution(event_size=act_size)
param_size = action_dist.param_size

policy_module = BiologicalSpikingPolicy(
    layer_sizes=nf.policy_hidden_layer_sizes,
    output_size=param_size, exc_ratio=nf.exc_ratio,
    n_micro_steps=nf.n_micro_steps, tau_min=nf.tau_min, tau_max=nf.tau_max,
    v_th=nf.v_th, v_reset=nf.v_reset, beta_surrogate=nf.beta_surrogate,
    n_refractory=nf.n_refractory,
)
diag_policy_module = DiagnosticBiologicalPolicy(
    layer_sizes=nf.policy_hidden_layer_sizes,
    output_size=param_size, exc_ratio=nf.exc_ratio,
    n_micro_steps=nf.n_micro_steps, tau_min=nf.tau_min, tau_max=nf.tau_max,
    v_th=nf.v_th, v_reset=nf.v_reset, beta_surrogate=nf.beta_surrogate,
    n_refractory=nf.n_refractory,
)
value_module = brax_networks.MLP(
    layer_sizes=list(nf.value_hidden_layer_sizes) + [1],
    activation=linen.swish, kernel_init=jax.nn.initializers.lecun_uniform(),
)

carry_dim = 2 * sum(nf.policy_hidden_layer_sizes)
dummy_obs = jp.zeros((1, obs_size))
dummy_carry = jp.zeros((1, carry_dim))

key = jax.random.PRNGKey(0)
key, pk, vk = jax.random.split(key, 3)
policy_params_init = policy_module.init(pk, dummy_obs, dummy_carry)
normalizer_params_init = running_statistics.init_state(jp.zeros(obs_size))
value_params_init = value_module.init(vk, dummy_obs)

print(f"Loading checkpoint...")
target = (normalizer_params_init, policy_params_init, value_params_init)
restored = ocp.PyTreeCheckpointer().restore(CHECKPOINT_PATH, item=target)
normalizer_params, policy_params, value_params = restored
print("Checkpoint loaded.")

# =============================================================================
# Constants
# =============================================================================

n_layers = len(nf.policy_hidden_layer_sizes)
n_exc = round(nf.policy_hidden_layer_sizes[0] * nf.exc_ratio)
n_inh = nf.policy_hidden_layer_sizes[0] - n_exc

# =============================================================================
# Rollout helpers
# =============================================================================

jit_eval_reset = jax.jit(eval_env.reset)
jit_eval_step = jax.jit(eval_env.step)
jit_policy_apply = jax.jit(policy_module.apply)
jit_diag_apply = jax.jit(diag_policy_module.apply)


def make_stimulated_params(scale_factor):
    """Scale W_ie (I→E weights) by the given factor in all layers."""
    stim = unfreeze(policy_params)
    for i in range(n_layers):
        lif = stim['params'][f'lif_{i}']
        lif['W_ie'] = lif['W_ie'] * scale_factor
    return freeze(stim)


def diag_rollout(params, norm_params, seed=0):
    """Run one episode with diagnostic policy, return per-layer spike data + rewards."""
    rng = jax.random.PRNGKey(seed)
    state = jit_eval_reset(rng)
    carry = jp.zeros((1, carry_dim))
    data = {f"lif_{i}": {"sp_e": [], "sp_i": []} for i in range(n_layers)}
    rewards = []
    for _ in range(episode_length):
        flat_obs = flatten_obs(state.obs)
        obs_norm = running_statistics.normalize(flat_obs[None], norm_params)
        logits, new_carry, layer_diag = jit_diag_apply(params, obs_norm, carry)
        action = jp.squeeze(action_dist.mode(logits), axis=0)
        state = jit_eval_step(state, action)
        carry = new_carry * (1.0 - state.done.reshape(1, 1))
        rewards.append(float(state.reward))
        for lname, diag in layer_diag.items():
            data[lname]["sp_e"].append(np.array(diag["spikes_exc"][:, 0, :]))
            data[lname]["sp_i"].append(np.array(diag["spikes_inh"][:, 0, :]))
    stacked = {}
    for lname in sorted(data.keys()):
        stacked[lname] = {
            'sp_e': np.stack(data[lname]["sp_e"]),
            'sp_i': np.stack(data[lname]["sp_i"]),
        }
    return stacked, np.array(rewards)


def video_rollout(params, norm_params, seed=0):
    """Run one episode, return (states_list, total_reward)."""
    rng = jax.random.PRNGKey(seed)
    state = jit_eval_reset(rng)
    carry = jp.zeros((1, carry_dim))
    rollout_states = [state]
    total_reward = 0.0
    for _ in range(episode_length):
        flat_obs = flatten_obs(state.obs)
        obs_norm = running_statistics.normalize(flat_obs[None], norm_params)
        logits, new_carry = jit_policy_apply(params, obs_norm, carry)
        action = jp.squeeze(action_dist.mode(logits), axis=0)
        state = jit_eval_step(state, action)
        carry = new_carry * (1.0 - state.done.reshape(1, 1))
        rollout_states.append(state)
        total_reward += float(state.reward)
    return rollout_states, total_reward


# =============================================================================
# Run dose-response: diagnostic rollout for each scale factor
# =============================================================================

SEED = 42
results = {}

for sf in SCALE_FACTORS:
    label = f"{sf:.0f}x" if sf >= 1 else f"{sf}x"
    print(f"\n--- W_ie scale = {label} ---")
    if sf == 1.0:
        params = policy_params
    else:
        params = make_stimulated_params(sf)
    diag, rewards = diag_rollout(params, normalizer_params, seed=SEED)
    total_r = float(np.sum(rewards))
    print(f"  Total reward: {total_r:.2f}")
    # Per-layer summary
    layer_stats = {}
    for lname in sorted(diag.keys()):
        sd = diag[lname]
        mean_e = float(np.mean(sd['sp_e']))
        mean_i = float(np.mean(sd['sp_i']))
        print(f"  {lname}: E rate={mean_e:.4f}, I rate={mean_i:.4f}")
        layer_stats[lname] = {'mean_e': mean_e, 'mean_i': mean_i}
    results[sf] = {
        'diag': diag,
        'rewards': rewards,
        'total_reward': total_r,
        'layer_stats': layer_stats,
    }

# =============================================================================
# Figure 1: Dose-Response Curves
# =============================================================================

print("\nPlotting dose-response curves...")
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
fig.suptitle("Inhibitory Stimulation Dose-Response (W$_{I→E}$ scaling)",
             fontsize=13, fontweight='bold')

sfs = np.array(SCALE_FACTORS)
total_rewards = np.array([results[sf]['total_reward'] for sf in SCALE_FACTORS])

# — Panel A: Total reward vs scale —
ax = axes[0]
ax.plot(sfs, total_rewards, 'o-', color='black', lw=1.5, markersize=6)
ax.axhline(total_rewards[0], color='gray', ls=':', lw=0.6, alpha=0.5)
for i, sf in enumerate(SCALE_FACTORS):
    pct = (total_rewards[i] - total_rewards[0]) / max(abs(total_rewards[0]), 1e-6) * 100
    ax.annotate(f'{pct:+.1f}%', (sfs[i], total_rewards[i]),
                textcoords='offset points', xytext=(0, 10), ha='center', fontsize=7)
ax.set_xscale('log')
ax.set_xlabel('W$_{I→E}$ scale factor')
ax.set_ylabel('Total episode reward')
ax.set_title('Reward vs Inhibition Strength')
ax.set_xticks(sfs)
ax.set_xticklabels([f'{s:.0f}x' for s in sfs])

# — Panel B: Mean E firing rate vs scale (per layer) —
ax = axes[1]
for li, lname in enumerate(sorted(results[1.0]['layer_stats'].keys())):
    rates = [results[sf]['layer_stats'][lname]['mean_e'] for sf in SCALE_FACTORS]
    ax.plot(sfs, rates, 'o-', lw=1.2, markersize=5, label=lname, color=f'C{li}')
ax.set_xscale('log')
ax.set_xlabel('W$_{I→E}$ scale factor')
ax.set_ylabel('Mean E firing rate')
ax.set_title('Excitatory Rate vs Inhibition')
ax.set_xticks(sfs)
ax.set_xticklabels([f'{s:.0f}x' for s in sfs])
ax.legend(framealpha=0.8)

# — Panel C: Mean I firing rate vs scale (per layer) —
ax = axes[2]
for li, lname in enumerate(sorted(results[1.0]['layer_stats'].keys())):
    rates = [results[sf]['layer_stats'][lname]['mean_i'] for sf in SCALE_FACTORS]
    ax.plot(sfs, rates, 's-', lw=1.2, markersize=5, label=lname, color=f'C{li}')
ax.set_xscale('log')
ax.set_xlabel('W$_{I→E}$ scale factor')
ax.set_ylabel('Mean I firing rate')
ax.set_title('Inhibitory Rate vs Inhibition')
ax.set_xticks(sfs)
ax.set_xticklabels([f'{s:.0f}x' for s in sfs])
ax.legend(framealpha=0.8)

# — Panel D: E/I balance vs scale —
ax = axes[3]
for li, lname in enumerate(sorted(results[1.0]['layer_stats'].keys())):
    balance = [results[sf]['layer_stats'][lname]['mean_e'] /
               max(results[sf]['layer_stats'][lname]['mean_i'], 1e-8)
               for sf in SCALE_FACTORS]
    ax.plot(sfs, balance, 'D-', lw=1.2, markersize=5, label=lname, color=f'C{li}')
ax.axhline(1.0, color='black', ls=':', lw=0.6, alpha=0.4)
ax.set_xscale('log')
ax.set_xlabel('W$_{I→E}$ scale factor')
ax.set_ylabel('E/I rate ratio')
ax.set_title('E/I Balance vs Inhibition')
ax.set_xticks(sfs)
ax.set_xticklabels([f'{s:.0f}x' for s in sfs])
ax.legend(framealpha=0.8)

fig.tight_layout()
path = os.path.join(OUTPUT_DIR, "dose_response.png")
fig.savefig(path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path}")

# =============================================================================
# Figure 2: Population Activity Comparison Across Conditions
# =============================================================================

print("Plotting population activity comparison...")
fig, axes = plt.subplots(n_layers, 3, figsize=(18, 4.5 * n_layers), squeeze=False)
fig.suptitle("E/I Population Rates Across Inhibition Levels",
             fontsize=13, fontweight='bold', y=0.995)

for row, lname in enumerate(sorted(results[1.0]['diag'].keys())):
    T_ep = results[1.0]['diag'][lname]['sp_e'].shape[0]
    t = np.arange(T_ep)
    sigma = max(1, T_ep // 40)

    # — Col 0: E population rate for each condition —
    ax = axes[row, 0]
    for idx, sf in enumerate(SCALE_FACTORS):
        rate = gaussian_filter1d(
            np.mean(results[sf]['diag'][lname]['sp_e'], axis=(1, 2)), sigma)
        ax.plot(t, rate, color=COLORS[idx], lw=0.9,
                label=f'{sf:.0f}x', alpha=0.85)
    ax.set_xlabel('Time (env steps)')
    ax.set_ylabel('E firing rate')
    ax.set_title(f'{lname} — Excitatory Population')
    ax.legend(title='W$_{I→E}$', framealpha=0.8, ncol=2, fontsize=7)
    ax.set_xlim(0, T_ep)

    # — Col 1: I population rate for each condition —
    ax = axes[row, 1]
    for idx, sf in enumerate(SCALE_FACTORS):
        rate = gaussian_filter1d(
            np.mean(results[sf]['diag'][lname]['sp_i'], axis=(1, 2)), sigma)
        ax.plot(t, rate, color=COLORS[idx], lw=0.9,
                label=f'{sf:.0f}x', alpha=0.85)
    ax.set_xlabel('Time (env steps)')
    ax.set_ylabel('I firing rate')
    ax.set_title(f'{lname} — Inhibitory Population')
    ax.legend(title='W$_{I→E}$', framealpha=0.8, ncol=2, fontsize=7)
    ax.set_xlim(0, T_ep)

    # — Col 2: Reward trajectory comparison —
    ax = axes[row, 2]
    for idx, sf in enumerate(SCALE_FACTORS):
        cum_r = np.cumsum(results[sf]['rewards'])
        ax.plot(cum_r, color=COLORS[idx], lw=0.9, label=f'{sf:.0f}x', alpha=0.85)
    ax.set_xlabel('Time (env steps)')
    ax.set_ylabel('Cumulative reward')
    ax.set_title('Cumulative Reward')
    ax.legend(title='W$_{I→E}$', framealpha=0.8, ncol=2, fontsize=7)
    # Only draw this once (same data for all layers)
    if row > 0:
        ax.set_visible(False)

fig.tight_layout()
path = os.path.join(OUTPUT_DIR, "population_comparison.png")
fig.savefig(path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path}")

# =============================================================================
# Figure 3: Per-neuron rate distribution shift
# =============================================================================

print("Plotting rate distribution shift...")
fig, axes = plt.subplots(n_layers, 2, figsize=(14, 4.5 * n_layers), squeeze=False)
fig.suptitle("Per-Neuron Firing Rate Shift Under Inhibitory Stimulation",
             fontsize=13, fontweight='bold', y=0.995)

# Pick a few key conditions to compare
key_sfs = [1.0, SCALE_FACTORS[len(SCALE_FACTORS) // 2], SCALE_FACTORS[-1]]

for row, lname in enumerate(sorted(results[1.0]['diag'].keys())):
    # E neurons
    ax = axes[row, 0]
    for sf in key_sfs:
        sd = results[sf]['diag'][lname]
        rates = np.mean(sd['sp_e'], axis=(0, 1))  # per-neuron mean rate
        idx = SCALE_FACTORS.index(sf)
        ax.hist(rates, bins=40, alpha=0.5, color=COLORS[idx],
                label=f'{sf:.0f}x ($\\mu$={np.mean(rates):.4f})', density=True)
    ax.set_xlabel('Mean firing rate')
    ax.set_ylabel('Density')
    ax.set_title(f'{lname} — E Neuron Rate Distributions')
    ax.legend(title='W$_{I→E}$', framealpha=0.8, fontsize=7)

    # I neurons
    ax = axes[row, 1]
    for sf in key_sfs:
        sd = results[sf]['diag'][lname]
        rates = np.mean(sd['sp_i'], axis=(0, 1))
        idx = SCALE_FACTORS.index(sf)
        ax.hist(rates, bins=40, alpha=0.5, color=COLORS[idx],
                label=f'{sf:.0f}x ($\\mu$={np.mean(rates):.4f})', density=True)
    ax.set_xlabel('Mean firing rate')
    ax.set_ylabel('Density')
    ax.set_title(f'{lname} — I Neuron Rate Distributions')
    ax.legend(title='W$_{I→E}$', framealpha=0.8, fontsize=7)

fig.tight_layout()
path = os.path.join(OUTPUT_DIR, "rate_distributions.png")
fig.savefig(path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path}")

# =============================================================================
# Videos: Normal (1x) vs strongest stimulation
# =============================================================================

strongest_sf = SCALE_FACTORS[-1]
print(f"\nRendering videos: 1x normal vs {strongest_sf:.0f}x stimulation...")

normal_states, _ = video_rollout(policy_params, normalizer_params, seed=SEED)
stim_params = make_stimulated_params(strongest_sf)
stim_states, _ = video_rollout(stim_params, normalizer_params, seed=SEED)

fps = int(1.0 / eval_env.dt)
normal_frames = eval_env.render(normal_states, height=512, width=512, render_ghost=True)
stim_frames = eval_env.render(stim_states, height=512, width=512, render_ghost=True)

# Save individual videos
for name, frames in [("normal_rollout.mp4", normal_frames),
                     (f"stimulated_{strongest_sf:.0f}x_rollout.mp4", stim_frames)]:
    p = os.path.join(OUTPUT_DIR, name)
    with imageio.get_writer(p, fps=fps) as vid:
        for f in frames:
            vid.append_data(f)
    print(f"  Saved: {p}")

# Side-by-side with labels


def render_text_bar(text, width, height=48, bg_color=(30, 30, 30),
                    text_color='white', fontsize=18):
    """Render a text label as a numpy array."""
    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax.text(0.5, 0.5, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=text_color)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.patch.set_facecolor(np.array(bg_color) / 255.0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)
    from PIL import Image
    img = np.array(Image.fromarray(img).resize((width, height), Image.LANCZOS))
    return img


print("Creating comparison video...")
n_frames = min(len(normal_frames), len(stim_frames))
frame_h, frame_w = normal_frames[0].shape[:2]
sep_w = 6

try:
    label_l = render_text_bar("NORMAL (1x)", frame_w, 48,
                              bg_color=(44, 160, 44))
    label_r = render_text_bar(f"STIMULATED ({strongest_sf:.0f}x I→E)", frame_w, 48,
                              bg_color=(180, 40, 100))
    sep_label = np.zeros((48, sep_w, 3), dtype=np.uint8)
    header = np.concatenate([label_l, sep_label, label_r], axis=1)
    has_labels = True
except Exception as e:
    print(f"  Label rendering failed ({e})")
    has_labels = False

separator = np.zeros((frame_h, sep_w, 3), dtype=np.uint8)
comp_path = os.path.join(OUTPUT_DIR, "comparison.mp4")
with imageio.get_writer(comp_path, fps=fps) as vid:
    for i in range(n_frames):
        combined = np.concatenate(
            [normal_frames[i], separator, stim_frames[i]], axis=1)
        if has_labels:
            combined = np.concatenate([header, combined], axis=0)
        vid.append_data(combined)
print(f"  Saved: {comp_path}")

# =============================================================================
# Summary table
# =============================================================================

print("\n" + "=" * 70)
print("INHIBITORY STIMULATION RESULTS")
print("=" * 70)
print(f"{'Scale':>8s}  {'Reward':>10s}  {'Δ%':>8s}", end="")
for lname in sorted(results[1.0]['layer_stats'].keys()):
    print(f"  {lname} E rate  {lname} I rate", end="")
print()
print("-" * 70)
baseline_r = results[1.0]['total_reward']
for sf in SCALE_FACTORS:
    r = results[sf]
    pct = (r['total_reward'] - baseline_r) / max(abs(baseline_r), 1e-6) * 100
    print(f"{sf:>7.0f}x  {r['total_reward']:>10.2f}  {pct:>+7.1f}%", end="")
    for lname in sorted(r['layer_stats'].keys()):
        ls = r['layer_stats'][lname]
        print(f"    {ls['mean_e']:.4f}      {ls['mean_i']:.4f}", end="")
    print()

print(f"\nOutputs in {OUTPUT_DIR}/:")
print("  dose_response.png          — reward & rates vs inhibition strength")
print("  population_comparison.png  — E/I activity traces across conditions")
print("  rate_distributions.png     — per-neuron rate shift under stimulation")
print(f"  normal_rollout.mp4         — baseline video")
print(f"  stimulated_{strongest_sf:.0f}x_rollout.mp4  — strongest stimulation video")
print("  comparison.mp4             — side-by-side video")
print("=" * 70)
