"""Lab meeting 2/23/26 — synchronized neural + behavior animation suite.

Generates MP4 videos from the propriospinal-MN checkpoint with:
  - MuJoCo ghost overlay
  - Spike rasters (PS-E, PS-I, MN) with time cursors
  - Membrane voltage heatmaps
  - PCA of spike rates and voltages
  - Action heatmaps
  - Stimulation and ablation comparisons

Usage:
    python lab_meeting_2_23_26_animation.py [checkpoint_path] [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["WANDB_MODE"] = "disabled"

import numpy as np
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
from PIL import Image

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

from train_mouse_propriospinal_mn import (
    ProprioSpinalMNPolicy,
    DiagnosticProprioSpinalMNPolicy,
    flatten_obs,
    ppo_params,
)

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

# =============================================================================
# Colors & style
# =============================================================================
plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.family': 'sans-serif', 'axes.linewidth': 0.8,
})

C_EXC = '#1f77b4'
C_INH = '#ff7f0e'
C_MN  = '#2ca02c'
CMAP_EXC = LinearSegmentedColormap.from_list('exc', ['#ffffff', C_EXC])
CMAP_INH = LinearSegmentedColormap.from_list('inh', ['#ffffff', C_INH])
CMAP_MN  = LinearSegmentedColormap.from_list('mn',  ['#ffffff', C_MN])

# =============================================================================
# Args & checkpoint
# =============================================================================
DEFAULT_CKPT = (
    "/root/vast/eric/vnl-playground/checkpoints/mouse-propriospinal-mn-20260224-075820/80035840"
)
CHECKPOINT_PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT
OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "outputs/lab_meeting_2_23_26"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"Output dir: {OUTPUT_DIR}")

# --- Network config ---
nf = ppo_params.network_factory
n_ps = nf.propriospinal_size           # 512
n_mn = nf.motor_neuron_size             # 128
n_exc = round(n_ps * nf.propriospinal_exc_ratio)  # 410
n_inh = n_ps - n_exc                    # 102
v_th = nf.v_th                          # 0.3

# carry = [ps_v(512) | mn_v(128) | ps_r(512) | mn_r(128)] = 1280
carry_dim = 2 * (n_ps + n_mn)

# --- Environment ---
env_cfg = default_config()
print("Loading reference clips...")
reference_clips = MouseReferenceClips(
    str(consts.MOUSE_REFERENCE_DATA_PATH),
    n_frames_per_clip=env_cfg.clip_length,
)
_, test_clips = reference_clips.split(train_ratio=0.8, seed=42)
eval_env = MouseImitation(config=env_cfg, clips=test_clips)

dt_env = env_cfg.ctrl_dt  # 0.0025s
fps_physics = int(1.0 / dt_env)  # 400
VIDEO_FPS = 50  # playback speed

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

# --- Network modules ---
action_dist = distribution.NormalTanhDistribution(event_size=act_size)
param_size = action_dist.param_size

policy_module = ProprioSpinalMNPolicy(
    propriospinal_size=n_ps,
    exc_ratio=nf.propriospinal_exc_ratio,
    motor_neuron_size=n_mn,
    n_micro_steps=nf.n_micro_steps,
    tau_min=nf.tau_min, tau_max=nf.tau_max,
    v_th=nf.v_th, v_reset=nf.v_reset,
    beta_surrogate=nf.beta_surrogate,
    n_refractory=nf.n_refractory,
    output_size=param_size,
)
diag_policy_module = DiagnosticProprioSpinalMNPolicy(
    propriospinal_size=n_ps,
    exc_ratio=nf.propriospinal_exc_ratio,
    motor_neuron_size=n_mn,
    n_micro_steps=nf.n_micro_steps,
    tau_min=nf.tau_min, tau_max=nf.tau_max,
    v_th=nf.v_th, v_reset=nf.v_reset,
    beta_surrogate=nf.beta_surrogate,
    n_refractory=nf.n_refractory,
    output_size=param_size,
)
value_module = brax_networks.MLP(
    layer_sizes=list(nf.value_hidden_layer_sizes) + [1],
    activation=linen.swish,
    kernel_init=jax.nn.initializers.lecun_uniform(),
)

# --- Load checkpoint ---
dummy_obs = jp.zeros((1, obs_size))
dummy_carry = jp.zeros((1, carry_dim))
key = jax.random.PRNGKey(0)
key, pk, vk = jax.random.split(key, 3)
policy_params_init = policy_module.init(pk, dummy_obs, dummy_carry)
normalizer_params_init = running_statistics.init_state(jp.zeros(obs_size))
value_params_init = value_module.init(vk, dummy_obs)

print("Loading checkpoint...")
target = (normalizer_params_init, policy_params_init, value_params_init)
restored = ocp.PyTreeCheckpointer().restore(CHECKPOINT_PATH, item=target)
normalizer_params, policy_params, value_params = restored
print("Checkpoint loaded.")

# JIT compile
jit_eval_reset = jax.jit(eval_env.reset)
jit_eval_step = jax.jit(eval_env.step)
jit_diag_apply = jax.jit(diag_policy_module.apply)

# =============================================================================
# Rollout helpers
# =============================================================================



def collect_full_data(params, norm_params, label, seed=42):
    """Single rollout that collects BOTH diagnostic data AND states for rendering.

    Uses the diagnostic policy for everything so neural data and behavior
    are from the exact same trajectory.
    """
    print(f"  [{label}] Rollout (diag + states)...")
    rng = jax.random.PRNGKey(seed)
    state = jit_eval_reset(rng)
    carry = jp.zeros((1, carry_dim))
    rollout_states = [state]
    total_reward = 0.0

    data = {k: [] for k in [
        'ps_spikes_exc', 'ps_spikes_inh', 'mn_spikes',
        'ps_voltages_exc', 'ps_voltages_inh', 'mn_voltages',
        'mn_input_exc', 'mn_input_inh', 'actions', 'rewards',
    ]}

    for _ in range(episode_length):
        flat_obs = flatten_obs(state.obs)
        obs_norm = running_statistics.normalize(flat_obs[None], norm_params)
        logits, new_carry, diagnostics = jit_diag_apply(params, obs_norm, carry)
        action = jp.squeeze(action_dist.mode(logits), axis=0)
        state = jit_eval_step(state, action)
        carry = new_carry * (1.0 - state.done.reshape(1, 1))
        rollout_states.append(state)
        total_reward += float(state.reward)

        ps_diag = diagnostics['propriospinal']
        mn_diag = diagnostics['motor_neurons']

        data['ps_spikes_exc'].append(np.array(ps_diag['spikes_exc'][:, 0, :]))
        data['ps_spikes_inh'].append(np.array(ps_diag['spikes_inh'][:, 0, :]))
        data['mn_spikes'].append(np.array(mn_diag['spikes'][:, 0, :]))
        data['ps_voltages_exc'].append(np.array(ps_diag['voltages_exc'][:, 0, :]))
        data['ps_voltages_inh'].append(np.array(ps_diag['voltages_inh'][:, 0, :]))
        data['mn_voltages'].append(np.array(mn_diag['voltages'][:, 0, :]))
        data['mn_input_exc'].append(np.array(diagnostics['mn_input_exc'][0]))
        data['mn_input_inh'].append(np.array(diagnostics['mn_input_inh'][0]))
        data['actions'].append(np.array(action))
        data['rewards'].append(float(state.reward))

    # Stack arrays
    result = {k: np.stack(v) if k != 'rewards' else np.array(v)
              for k, v in data.items()}

    print(f"  [{label}] Rendering frames...")
    frames = eval_env.render(rollout_states, height=480, width=480, render_ghost=True)
    result['frames'] = [np.array(f) for f in frames]
    result['total_reward'] = total_reward
    print(f"  [{label}] Done. Reward={total_reward:.2f}")
    return result


# =============================================================================
# Stimulation & Ablation param modifiers
# =============================================================================

STIM_FACTORS = [2.0, 5.0, 10.0, 20.0, 50.0]
ABLATION_FRACS = [0.25, 0.50, 0.75, 1.00]


_baseline_wie_norm = None  # set once to verify control is never mutated


def make_stimulated_params(scale_factor):
    """Scale W_ie (I->E lateral weights) in the propriospinal module."""
    stim = unfreeze(policy_params)
    ps = stim['params']['propriospinal']
    ps['W_ie'] = ps['W_ie'] * scale_factor
    result = freeze(stim)
    # Verify original params untouched
    orig_norm = float(jp.linalg.norm(policy_params['params']['propriospinal']['W_ie']))
    new_norm = float(jp.linalg.norm(result['params']['propriospinal']['W_ie']))
    print(f"    [verify] control W_ie norm={orig_norm:.4f}, "
          f"stim {scale_factor}x W_ie norm={new_norm:.4f}")
    return result


def make_ablated_params(frac):
    """Silence a fraction of E neurons in the propriospinal module."""
    if frac == 0.0:
        return policy_params
    n_ablate = round(n_exc * frac)
    rng_abl = np.random.RandomState(0)
    ablate_idx = np.sort(rng_abl.choice(n_exc, size=n_ablate, replace=False))
    abl = unfreeze(policy_params)
    ps = abl['params']['propriospinal']
    # Zero feedforward input to ablated E neurons
    ps['input_proj']['kernel'] = ps['input_proj']['kernel'].at[:, ablate_idx].set(0.0)
    ps['input_proj']['bias'] = ps['input_proj']['bias'].at[ablate_idx].set(0.0)
    # Zero ablated E -> I output
    ps['W_ei'] = ps['W_ei'].at[ablate_idx, :].set(0.0)
    # Zero I -> ablated-E input
    ps['W_ie'] = ps['W_ie'].at[:, ablate_idx].set(0.0)
    # Also zero ablated E -> MN output
    abl['params']['W_exc_mn'] = abl['params']['W_exc_mn'].at[:, ablate_idx].set(0.0)
    result = freeze(abl)
    # Verify original params untouched
    orig_k = float(jp.sum(jp.abs(policy_params['params']['propriospinal']['input_proj']['kernel'])))
    new_k = float(jp.sum(jp.abs(result['params']['propriospinal']['input_proj']['kernel'])))
    print(f"    [verify] control kernel sum={orig_k:.4f}, "
          f"ablation {frac*100:.0f}% kernel sum={new_k:.4f}")
    return result


# =============================================================================
# Collect all data
# =============================================================================
SEED = 42

print("\n=== Collecting baseline data ===")
baseline_data = collect_full_data(policy_params, normalizer_params, "baseline", SEED)

print("\n=== Collecting stimulation data ===")
stim_data = {}
for sf in STIM_FACTORS:
    label = f"stim_{sf:.0f}x"
    stim_data[sf] = collect_full_data(
        make_stimulated_params(sf), normalizer_params, label, SEED)

print("\n=== Collecting ablation data ===")
ablation_data = {}
for frac in ABLATION_FRACS:
    label = f"ablation_{int(frac*100)}pct"
    ablation_data[frac] = collect_full_data(
        make_ablated_params(frac), normalizer_params, label, SEED)

print("\nAll data collected.")


# =============================================================================
# Panel rendering utilities
# =============================================================================

PANEL_W, PANEL_H = 480, 320  # pixels per panel
PANEL_DPI = 100
FIG_W = PANEL_W / PANEL_DPI
FIG_H = PANEL_H / PANEL_DPI


def fig_to_array(fig):
    """Convert matplotlib figure to numpy RGB array at PANEL_W x PANEL_H."""
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)
    # Resize to exact panel size
    img = np.array(Image.fromarray(img).resize((PANEL_W, PANEL_H), Image.LANCZOS))
    return img


def render_raster_panel(spikes, title, cmap, n_show=None):
    """Pre-render a spike raster as a static image.

    Args:
        spikes: (T, K, N) binary spikes. Averaged across K micro-steps.
        title: panel title
        cmap: colormap for spike intensity
        n_show: if set, subsample to this many neurons (sorted by rate)

    Returns:
        numpy array (PANEL_H, PANEL_W, 3)
    """
    # Average over micro-steps: (T, K, N) -> (T, N)
    rates = np.mean(spikes, axis=1)
    T, N = rates.shape

    # Sort by mean firing rate (descending)
    order = np.argsort(-np.mean(rates, axis=0))
    if n_show is not None and n_show < N:
        order = order[:n_show]
    rates = rates[:, order]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=PANEL_DPI)
    ax.imshow(rates.T, aspect='auto', cmap=cmap, interpolation='nearest',
              extent=[0, T * dt_env, rates.shape[1], 0], vmin=0, vmax=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neuron #')
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig_to_array(fig)


def render_voltage_heatmap(voltages, title, cmap='RdBu_r', n_show=None):
    """Pre-render membrane voltage heatmap.

    Args:
        voltages: (T, K, N). Take last micro-step as representative.
        title: panel title
        cmap: colormap
        n_show: subsample neuron count
    Returns:
        numpy array (PANEL_H, PANEL_W, 3)
    """
    # Take last micro-step voltage: (T, N)
    v = voltages[:, -1, :]
    T, N = v.shape

    order = np.argsort(-np.mean(np.mean(voltages, axis=1), axis=0))
    if n_show is not None and n_show < N:
        order = order[:n_show]
    v = v[:, order]

    vmax = max(abs(float(v.min())), abs(float(v.max())), v_th)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=PANEL_DPI)
    im = ax.imshow(v.T, aspect='auto', cmap=cmap, interpolation='nearest',
                   extent=[0, T * dt_env, v.shape[1], 0],
                   vmin=-vmax, vmax=vmax)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neuron #')
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, label='V_m', shrink=0.8)
    fig.tight_layout()
    return fig_to_array(fig)


def render_action_heatmap(actions, title='Actions'):
    """Pre-render action heatmap.

    Args:
        actions: (T, n_act)
    Returns:
        numpy array (PANEL_H, PANEL_W, 3)
    """
    T, n_act = actions.shape
    vmax = float(np.abs(actions).max())

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=PANEL_DPI)
    im = ax.imshow(actions.T, aspect='auto', cmap='RdBu_r',
                   interpolation='nearest',
                   extent=[0, T * dt_env, n_act, 0],
                   vmin=-vmax, vmax=vmax)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Muscle #')
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, label='Activation', shrink=0.8)
    fig.tight_layout()
    return fig_to_array(fig)


def render_pca_panel(signal, title, n_components=3):
    """Pre-render PCA of population activity.

    Args:
        signal: (T, K, N) — averaged across K before PCA, or (T, N) already averaged
        title: panel title
        n_components: number of PCs to show
    Returns:
        numpy array (PANEL_H, PANEL_W, 3)
    """
    if signal.ndim == 3:
        x = np.mean(signal, axis=1)  # (T, N)
    else:
        x = signal
    T, N = x.shape
    n_components = min(n_components, N, T)

    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(x)  # (T, n_components)

    t_sec = np.arange(T) * dt_env
    colors = ['#e41a1c', '#377eb8', '#4daf4a']

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=PANEL_DPI)
    for i in range(n_components):
        var = pca.explained_variance_ratio_[i] * 100
        ax.plot(t_sec, pcs[:, i], color=colors[i % len(colors)], lw=1.2,
                label=f'PC{i+1} ({var:.1f}%)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('PC projection')
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, framealpha=0.8)
    fig.tight_layout()
    return fig_to_array(fig)


def render_drive_panel(mn_exc, mn_inh, title='MN Excitatory/Inhibitory Drive'):
    """Pre-render motor neuron E/I drive traces.

    Args:
        mn_exc: (T, n_mn) excitatory drive
        mn_inh: (T, n_mn) inhibitory drive (negative values)
    Returns:
        numpy array (PANEL_H, PANEL_W, 3)
    """
    T = mn_exc.shape[0]
    t_sec = np.arange(T) * dt_env

    mean_exc = np.mean(mn_exc, axis=1)
    mean_inh = np.mean(mn_inh, axis=1)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=PANEL_DPI)
    ax.plot(t_sec, mean_exc, color=C_EXC, lw=1.2, label='Exc drive')
    ax.plot(t_sec, mean_inh, color=C_INH, lw=1.2, label='Inh drive')
    ax.plot(t_sec, mean_exc + mean_inh, color='black', lw=0.8, ls='--',
            label='Net drive', alpha=0.7)
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mean drive')
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, framealpha=0.8)
    fig.tight_layout()
    return fig_to_array(fig)


def render_firing_rate_traces(data, title='Population Firing Rates'):
    """Pre-render mean firing rate traces for PS-E, PS-I, MN.

    Args:
        data: dict with ps_spikes_exc, ps_spikes_inh, mn_spikes (T, K, N)
    Returns:
        numpy array (PANEL_H, PANEL_W, 3)
    """
    T = data['ps_spikes_exc'].shape[0]
    t_sec = np.arange(T) * dt_env
    sigma = max(1, T // 40)

    ps_e_rate = gaussian_filter1d(np.mean(data['ps_spikes_exc'], axis=(1, 2)), sigma)
    ps_i_rate = gaussian_filter1d(np.mean(data['ps_spikes_inh'], axis=(1, 2)), sigma)
    mn_rate = gaussian_filter1d(np.mean(data['mn_spikes'], axis=(1, 2)), sigma)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=PANEL_DPI)
    ax.plot(t_sec, ps_e_rate, color=C_EXC, lw=1.2, label=f'PS-E (n={n_exc})')
    ax.plot(t_sec, ps_i_rate, color=C_INH, lw=1.2, label=f'PS-I (n={n_inh})')
    ax.plot(t_sec, mn_rate, color=C_MN, lw=1.2, label=f'MN (n={n_mn})')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mean spike rate')
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, framealpha=0.8)
    fig.tight_layout()
    return fig_to_array(fig)


def draw_time_cursor(panel_img, t_idx, T_total):
    """Overlay a red vertical time cursor on a panel image.

    Args:
        panel_img: (H, W, 3) numpy array
        t_idx: current timestep index (0 to T_total-1)
        T_total: total number of timesteps
    Returns:
        new (H, W, 3) array with red line
    """
    img = panel_img.copy()
    H, W = img.shape[:2]
    # Matplotlib axes margins (approximate from tight_layout)
    margin_left = int(0.14 * W)
    margin_right = int(0.04 * W)
    plot_w = W - margin_left - margin_right

    x = margin_left + int(t_idx / max(T_total - 1, 1) * plot_w)
    x = np.clip(x, 0, W - 1)

    # Draw 2-pixel wide red line
    x_min = max(0, x - 1)
    x_max = min(W, x + 1)
    img[:, x_min:x_max, :] = [255, 0, 0]
    return img


def render_text_bar(text, width, height=40, bg_color=(30, 30, 30),
                    text_color='white', fontsize=14):
    """Render a text label bar."""
    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax.text(0.5, 0.5, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=text_color)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    fig.patch.set_facecolor(np.array(bg_color) / 255.0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)
    img = np.array(Image.fromarray(img).resize((width, height), Image.LANCZOS))
    return img


# =============================================================================
# Video generators
# =============================================================================


def make_video(frames_list, path, fps=VIDEO_FPS):
    """Write a list of numpy frames to MP4."""
    with imageio.get_writer(path, fps=fps) as vid:
        for f in frames_list:
            vid.append_data(f)
    print(f"  Saved: {path}")


def make_baseline_full_video(data, output_path):
    """2x4 grid: MuJoCo | PS-E raster | PS-I raster | MN raster
                  Actions | PCA spikes  | PCA voltages | MN voltage heatmap

    All panels have synchronized time cursor.
    """
    print("Generating baseline_full.mp4...")
    T = data['ps_spikes_exc'].shape[0]
    mujoco_frames = data['frames']

    # Pre-render static panels
    panel_pse = render_raster_panel(data['ps_spikes_exc'], 'PS Excitatory (100/410)',
                                     CMAP_EXC, n_show=100)
    panel_psi = render_raster_panel(data['ps_spikes_inh'], 'PS Inhibitory (102)',
                                     CMAP_INH)
    panel_mn  = render_raster_panel(data['mn_spikes'], 'Motor Neurons (128)',
                                     CMAP_MN)
    panel_act = render_action_heatmap(data['actions'])
    panel_pca_sp = render_pca_panel(
        np.concatenate([data['ps_spikes_exc'], data['ps_spikes_inh'],
                        data['mn_spikes']], axis=-1),
        'PCA — Spike Rates (all populations)')
    panel_pca_v = render_pca_panel(
        np.concatenate([data['ps_voltages_exc'], data['ps_voltages_inh'],
                        data['mn_voltages']], axis=-1),
        'PCA — Membrane Voltages')
    panel_mn_v = render_voltage_heatmap(data['mn_voltages'], 'MN Membrane Voltages',
                                         cmap='RdBu_r')

    # Resize mujoco frames to panel size
    def resize_frame(frame):
        return np.array(Image.fromarray(frame).resize((PANEL_W, PANEL_H), Image.LANCZOS))

    composed = []
    for t in range(min(T, len(mujoco_frames))):
        mj = resize_frame(mujoco_frames[t])

        # Draw time cursors on all temporal panels
        row0 = np.concatenate([
            mj,
            draw_time_cursor(panel_pse, t, T),
            draw_time_cursor(panel_psi, t, T),
            draw_time_cursor(panel_mn, t, T),
        ], axis=1)
        row1 = np.concatenate([
            draw_time_cursor(panel_act, t, T),
            draw_time_cursor(panel_pca_sp, t, T),
            draw_time_cursor(panel_pca_v, t, T),
            draw_time_cursor(panel_mn_v, t, T),
        ], axis=1)
        composed.append(np.concatenate([row0, row1], axis=0))

    make_video(composed, output_path)


def make_baseline_membrane_video(data, output_path):
    """2x3 grid: MuJoCo     | PS-E voltage | PS-I voltage
                  MN voltage | MN drive     | Actions
    """
    print("Generating baseline_membrane_potentials.mp4...")
    T = data['ps_voltages_exc'].shape[0]
    mujoco_frames = data['frames']

    panel_pse_v = render_voltage_heatmap(data['ps_voltages_exc'],
                                          'PS-E Membrane Voltages (100/410)',
                                          n_show=100)
    panel_psi_v = render_voltage_heatmap(data['ps_voltages_inh'],
                                          'PS-I Membrane Voltages (102)')
    panel_mn_v  = render_voltage_heatmap(data['mn_voltages'],
                                          'MN Membrane Voltages (128)')
    panel_drive = render_drive_panel(data['mn_input_exc'], data['mn_input_inh'])
    panel_act = render_action_heatmap(data['actions'])

    def resize_frame(frame):
        return np.array(Image.fromarray(frame).resize((PANEL_W, PANEL_H), Image.LANCZOS))

    composed = []
    for t in range(min(T, len(mujoco_frames))):
        mj = resize_frame(mujoco_frames[t])
        row0 = np.concatenate([
            mj,
            draw_time_cursor(panel_pse_v, t, T),
            draw_time_cursor(panel_psi_v, t, T),
        ], axis=1)
        row1 = np.concatenate([
            draw_time_cursor(panel_mn_v, t, T),
            draw_time_cursor(panel_drive, t, T),
            draw_time_cursor(panel_act, t, T),
        ], axis=1)
        composed.append(np.concatenate([row0, row1], axis=0))

    make_video(composed, output_path)


def make_spike_diagnostics_video(data, output_path):
    """4x2 neural focus: MuJoCo        | PS-E raster
                          PS-I raster   | MN raster
                          Firing rates  | PCA spikes
                          PCA voltages  | Actions
    """
    print("Generating spike_diagnostics_baseline.mp4...")
    T = data['ps_spikes_exc'].shape[0]
    mujoco_frames = data['frames']

    panel_pse = render_raster_panel(data['ps_spikes_exc'], 'PS Excitatory (100/410)',
                                     CMAP_EXC, n_show=100)
    panel_psi = render_raster_panel(data['ps_spikes_inh'], 'PS Inhibitory (102)',
                                     CMAP_INH)
    panel_mn  = render_raster_panel(data['mn_spikes'], 'Motor Neurons (128)', CMAP_MN)
    panel_rates = render_firing_rate_traces(data)
    panel_pca_sp = render_pca_panel(
        np.concatenate([data['ps_spikes_exc'], data['ps_spikes_inh'],
                        data['mn_spikes']], axis=-1),
        'PCA — Spike Rates')
    panel_pca_v = render_pca_panel(
        np.concatenate([data['ps_voltages_exc'], data['ps_voltages_inh'],
                        data['mn_voltages']], axis=-1),
        'PCA — Voltages')
    panel_act = render_action_heatmap(data['actions'])

    def resize_frame(frame):
        return np.array(Image.fromarray(frame).resize((PANEL_W, PANEL_H), Image.LANCZOS))

    composed = []
    for t in range(min(T, len(mujoco_frames))):
        mj = resize_frame(mujoco_frames[t])
        row0 = np.concatenate([
            mj,
            draw_time_cursor(panel_pse, t, T),
        ], axis=1)
        row1 = np.concatenate([
            draw_time_cursor(panel_psi, t, T),
            draw_time_cursor(panel_mn, t, T),
        ], axis=1)
        row2 = np.concatenate([
            draw_time_cursor(panel_rates, t, T),
            draw_time_cursor(panel_pca_sp, t, T),
        ], axis=1)
        row3 = np.concatenate([
            draw_time_cursor(panel_pca_v, t, T),
            draw_time_cursor(panel_act, t, T),
        ], axis=1)
        composed.append(np.concatenate([row0, row1, row2, row3], axis=0))

    make_video(composed, output_path)


def make_comparison_video(data_normal, data_condition, label_normal, label_condition,
                           output_path):
    """Left/Right comparison video.

    Each side (5 rows): MuJoCo | PS-E raster | PS-I raster | MN raster | Actions | MN Voltages
    Separator in between. Labels on top.
    """
    print(f"Generating {os.path.basename(output_path)}...")
    T = min(data_normal['ps_spikes_exc'].shape[0],
            data_condition['ps_spikes_exc'].shape[0])

    def render_side_panels(data):
        return {
            'pse': render_raster_panel(data['ps_spikes_exc'], 'PS-E (100)',
                                        CMAP_EXC, n_show=100),
            'psi': render_raster_panel(data['ps_spikes_inh'], 'PS-I (102)',
                                        CMAP_INH),
            'mn':  render_raster_panel(data['mn_spikes'], 'MN (128)', CMAP_MN),
            'act': render_action_heatmap(data['actions']),
            'mn_v': render_voltage_heatmap(data['mn_voltages'], 'MN Voltages'),
        }

    panels_l = render_side_panels(data_normal)
    panels_r = render_side_panels(data_condition)

    # Labels (sizes chosen so total is divisible by 16 for ffmpeg)
    label_h = 48
    sep_w = 16
    lbl_l = render_text_bar(label_normal, PANEL_W, label_h, bg_color=(44, 160, 44))
    lbl_r = render_text_bar(label_condition, PANEL_W, label_h, bg_color=(180, 40, 100))
    sep_label = np.zeros((label_h, sep_w, 3), dtype=np.uint8)
    header = np.concatenate([lbl_l, sep_label, lbl_r], axis=1)

    separator = np.zeros((PANEL_H, sep_w, 3), dtype=np.uint8)

    def resize_frame(frame):
        return np.array(Image.fromarray(frame).resize((PANEL_W, PANEL_H), Image.LANCZOS))

    composed = []
    frames_l = data_normal['frames']
    frames_r = data_condition['frames']

    for t in range(min(T, len(frames_l), len(frames_r))):
        rows = []
        # Row 0: MuJoCo
        mj_l = resize_frame(frames_l[t])
        mj_r = resize_frame(frames_r[t])
        rows.append(np.concatenate([mj_l, separator, mj_r], axis=1))

        # Row 1-5: rasters + actions + voltages
        for key in ['pse', 'psi', 'mn', 'act', 'mn_v']:
            pl = draw_time_cursor(panels_l[key], t, T)
            pr = draw_time_cursor(panels_r[key], t, T)
            rows.append(np.concatenate([pl, separator, pr], axis=1))

        frame = np.concatenate([header] + rows, axis=0)
        composed.append(frame)

    make_video(composed, output_path)


def make_stim_neural_detail_video(data_normal, data_stim, sf, output_path):
    """Neural detail comparison for stimulation.

    2x5 grid:
      Normal MuJoCo       | Stim MuJoCo
      Normal PS-E raster  | Stim PS-E raster
      Normal PS-I raster  | Stim PS-I raster
      Normal MN raster    | Stim MN raster
      Normal MN Drive     | Stim MN Drive
    """
    print(f"Generating {os.path.basename(output_path)}...")
    T = data_normal['ps_spikes_exc'].shape[0]
    frames_n = data_normal['frames']
    frames_s = data_stim['frames']

    # Normal side
    n_pse = render_raster_panel(data_normal['ps_spikes_exc'], 'Normal PS-E',
                                 CMAP_EXC, n_show=100)
    n_psi = render_raster_panel(data_normal['ps_spikes_inh'], 'Normal PS-I',
                                 CMAP_INH)
    n_mn  = render_raster_panel(data_normal['mn_spikes'], 'Normal MN', CMAP_MN)
    n_drive = render_drive_panel(data_normal['mn_input_exc'],
                                  data_normal['mn_input_inh'], 'Normal MN Drive')

    # Stim side
    s_pse = render_raster_panel(data_stim['ps_spikes_exc'],
                                 f'Stim {sf:.0f}x PS-E', CMAP_EXC, n_show=100)
    s_psi = render_raster_panel(data_stim['ps_spikes_inh'],
                                 f'Stim {sf:.0f}x PS-I', CMAP_INH)
    s_mn  = render_raster_panel(data_stim['mn_spikes'],
                                 f'Stim {sf:.0f}x MN', CMAP_MN)
    s_drive = render_drive_panel(data_stim['mn_input_exc'],
                                  data_stim['mn_input_inh'],
                                  f'Stim {sf:.0f}x MN Drive')

    def resize_frame(frame):
        return np.array(Image.fromarray(frame).resize((PANEL_W, PANEL_H), Image.LANCZOS))

    composed = []
    for t in range(min(T, len(frames_n), len(frames_s))):
        mj_n = resize_frame(frames_n[t])
        mj_s = resize_frame(frames_s[t])
        row_mj = np.concatenate([mj_n, mj_s], axis=1)
        row0 = np.concatenate([
            draw_time_cursor(n_pse, t, T),
            draw_time_cursor(s_pse, t, T),
        ], axis=1)
        row1 = np.concatenate([
            draw_time_cursor(n_psi, t, T),
            draw_time_cursor(s_psi, t, T),
        ], axis=1)
        row2 = np.concatenate([
            draw_time_cursor(n_mn, t, T),
            draw_time_cursor(s_mn, t, T),
        ], axis=1)
        row3 = np.concatenate([
            draw_time_cursor(n_drive, t, T),
            draw_time_cursor(s_drive, t, T),
        ], axis=1)
        composed.append(np.concatenate([row_mj, row0, row1, row2, row3], axis=0))

    make_video(composed, output_path)


# =============================================================================
# Generate all videos
# =============================================================================

print("\n=== Generating baseline videos ===")
make_baseline_full_video(baseline_data,
                         os.path.join(OUTPUT_DIR, "baseline_full.mp4"))
make_baseline_membrane_video(baseline_data,
                              os.path.join(OUTPUT_DIR, "baseline_membrane_potentials.mp4"))
make_spike_diagnostics_video(baseline_data,
                              os.path.join(OUTPUT_DIR, "spike_diagnostics_baseline.mp4"))

print("\n=== Generating stimulation comparison videos ===")
for sf in STIM_FACTORS:
    make_comparison_video(
        baseline_data, stim_data[sf],
        f"NORMAL (reward={baseline_data['total_reward']:.1f})",
        f"STIM {sf:.0f}x I->E (reward={stim_data[sf]['total_reward']:.1f})",
        os.path.join(OUTPUT_DIR, f"stim_{sf:.0f}x.mp4"),
    )

print("\n=== Generating stimulation neural detail videos ===")
for sf in STIM_FACTORS:
    make_stim_neural_detail_video(
        baseline_data, stim_data[sf], sf,
        os.path.join(OUTPUT_DIR, f"stim_{sf:.0f}x_neural_detail.mp4"),
    )

print("\n=== Generating ablation comparison videos ===")
for frac in ABLATION_FRACS:
    pct = int(frac * 100)
    make_comparison_video(
        baseline_data, ablation_data[frac],
        f"NORMAL (reward={baseline_data['total_reward']:.1f})",
        f"ABLATION {pct}% E silenced (reward={ablation_data[frac]['total_reward']:.1f})",
        os.path.join(OUTPUT_DIR, f"ablation_{pct}pct.mp4"),
    )

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("LAB MEETING ANIMATION SUITE — COMPLETE")
print("=" * 70)
print(f"Output directory: {OUTPUT_DIR}")
print(f"\nBaseline videos:")
print(f"  baseline_full.mp4                  — 2x4 grid (behavior + rasters + PCA + actions)")
print(f"  baseline_membrane_potentials.mp4   — 2x2 voltage heatmaps + MN drive")
print(f"  spike_diagnostics_baseline.mp4     — 3x2 neural focus (rasters + rates + PCA)")
print(f"\nStimulation comparison videos (normal vs stimulated):")
for sf in STIM_FACTORS:
    r = stim_data[sf]['total_reward']
    print(f"  stim_{sf:.0f}x.mp4                        — behavior + rasters (reward={r:.1f})")
    print(f"  stim_{sf:.0f}x_neural_detail.mp4           — population rasters + drive")
print(f"\nAblation comparison videos (normal vs ablated):")
for frac in ABLATION_FRACS:
    pct = int(frac * 100)
    r = ablation_data[frac]['total_reward']
    print(f"  ablation_{pct}pct.mp4                    — behavior + rasters (reward={r:.1f})")
print(f"\nBaseline reward: {baseline_data['total_reward']:.2f}")
print("=" * 70)
