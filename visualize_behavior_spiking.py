"""Synchronized video + neural diagnostics for LRN-Cerebellum circuit.

Creates a grid video where each frame shows:
  - Top-left:    MuJoCo rollout (perturbed)
  - Top-right:   Action heatmap with time cursor
  - Mid-left:    PS E population raster with time cursor
  - Mid-right:   PS I population raster with time cursor
  - Bottom-left: Cerebellum correction + prediction + innovation traces
  - Bottom-right: Firing-rate spectrogram (PS E) with time cursor

A vertical line tracks the current timestep across all panels, synced
to the video frames.

Usage:
    python visualize_behavior_spiking.py [checkpoint_path] [output_path]

    Default checkpoint:
        checkpoints/mouse-lrn-cerebellum-20260203-033949/160030720
    Default output:
        diagnostic_results/behavior_spiking.mp4
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
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import spectrogram
import imageio
import cv2

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
# Colours — same seaborn palette as spike_diagnostics.py
# =============================================================================

C_EXC = '#1f77b4'
C_INH = '#ff7f0e'
C_CB  = '#d62728'
C_BAL = '#9467bd'

CMAP_EXC = LinearSegmentedColormap.from_list('exc', ['#ffffff', C_EXC])
CMAP_INH = LinearSegmentedColormap.from_list('inh', ['#ffffff', C_INH])

# =============================================================================
# Args
# =============================================================================

DEFAULT_CKPT = (
    "/root/vast/eric/vnl-playground/checkpoints/"
    "mouse-lrn-cerebellum-20260203-033949/160030720"
)
CHECKPOINT_PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT
OUTPUT_DIR = "diagnostic_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    OUTPUT_DIR, "behavior_spiking.mp4"
)
print(f"Checkpoint : {CHECKPOINT_PATH}")
print(f"Output     : {OUTPUT_PATH}")

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
# current ppo_params.  We first do a raw restore to discover the actual
# W_ie shape, infer the true exc_ratio, then build modules that match.
# =============================================================================

action_dist = distribution.NormalTanhDistribution(event_size=act_size)
param_size = action_dist.param_size

key = jax.random.PRNGKey(0)

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
carry_dim = (
    2 * ckpt_ps_size
    + 2 * nf.lrn_size
    + 2 * nf.cerebellum_state_dim
)

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

# Init skeletons for typed restore
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
# Derived constants
# =============================================================================

n_exc = ckpt_n_exc
n_inh = ckpt_n_inh
n_lrn = nf.lrn_size
K = nf.n_micro_steps
cb_state_dim = nf.cerebellum_state_dim

w_raw = np.array(policy_params['params']['correction_weight_raw'])
correction_weight = 1.0 / (1.0 + np.exp(-w_raw))

# Perturbation window
perturb_start = int(episode_length * nf.perturb_start_frac)
perturb_end = int(episode_length * nf.perturb_end_frac)
perturb_scale = nf.perturb_scale

print(f"Episode length : {episode_length}")
print(f"Perturbation   : steps [{perturb_start}, {perturb_end}), "
      f"scale={perturb_scale}")
print(f"Network        : PS {n_exc}E/{n_inh}I, LRN {n_lrn}E, "
      f"CB state_dim={cb_state_dim}")
print(f"Correction w   : {correction_weight.item():.6f}")

# =============================================================================
# Diagnostic rollout (with perturbation)
# =============================================================================

print("\nRunning diagnostic rollout with perturbation...")

jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)
jit_diag = jax.jit(diag_policy_module.apply)
jit_policy = jax.jit(policy_module.apply)

# Sample a perturbation direction
key, force_rng = jax.random.split(key)
perturb_dir = jax.random.normal(force_rng, (act_size,))
perturb_force = np.array(
    (perturb_dir / (jp.linalg.norm(perturb_dir) + 1e-8))
    * perturb_scale * jp.sqrt(jp.float32(act_size))
)

state = jit_reset(jax.random.PRNGKey(999))
carry = jp.zeros((1, carry_dim))

# Collect rollout states (for rendering) AND diagnostic data
rollout_states = [state]
all_data = {
    'ps_spikes_exc': [], 'ps_spikes_inh': [],
    'ps_voltages_exc': [], 'ps_voltages_inh': [],
    'lrn_spikes': [], 'lrn_voltages': [],
    'cb_innovation_norm': [], 'cb_x_hat_new': [],
    'cb_correction': [], 'cb_x_hat_pred': [],
    'cb_P_diag_new': [],
    'raw_motor_cmd': [], 'actions': [], 'reward': [],
}

for t in range(episode_length):
    flat_obs = flatten_obs(state.obs)
    obs_norm = running_statistics.normalize(flat_obs[None], normalizer_params)

    logits, new_carry, diagnostics = jit_diag(
        policy_params, obs_norm, carry)

    action_mode = jp.squeeze(action_dist.mode(logits), axis=0)

    # Apply perturbation in window
    if perturb_start <= t < perturb_end:
        action = action_mode + jp.array(perturb_force)
    else:
        action = action_mode

    state = jit_step(state, action)
    carry = new_carry * (1.0 - state.done.reshape(1, 1))
    rollout_states.append(state)

    ps_diag = diagnostics['propriospinal']
    lrn_diag = diagnostics['lrn']
    cb_diag = diagnostics['cerebellum']

    all_data['ps_spikes_exc'].append(np.array(ps_diag['spikes_exc'][:, 0, :]))
    all_data['ps_spikes_inh'].append(np.array(ps_diag['spikes_inh'][:, 0, :]))
    all_data['ps_voltages_exc'].append(np.array(ps_diag['voltages_exc'][:, 0, :]))
    all_data['ps_voltages_inh'].append(np.array(ps_diag['voltages_inh'][:, 0, :]))
    all_data['lrn_spikes'].append(np.array(lrn_diag['spikes'][:, 0, :]))
    all_data['lrn_voltages'].append(np.array(lrn_diag['voltages'][:, 0, :]))
    all_data['cb_innovation_norm'].append(float(cb_diag['innovation_norm'][0]))
    all_data['cb_x_hat_new'].append(np.array(cb_diag['x_hat_new'][0]))
    all_data['cb_x_hat_pred'].append(np.array(cb_diag['x_hat_pred'][0]))
    all_data['cb_correction'].append(np.array(cb_diag['correction'][0]))
    all_data['cb_P_diag_new'].append(np.array(cb_diag['P_diag_new'][0]))
    all_data['raw_motor_cmd'].append(np.array(diagnostics['raw_motor_cmd'][0]))
    all_data['actions'].append(np.array(action))
    all_data['reward'].append(float(state.reward))

print("Stacking arrays...")
ps_sp_e = np.stack(all_data['ps_spikes_exc'])    # (T, K, n_exc)
ps_sp_i = np.stack(all_data['ps_spikes_inh'])    # (T, K, n_inh)
lrn_sp  = np.stack(all_data['lrn_spikes'])       # (T, K, n_lrn)

cb_innov_norm = np.array(all_data['cb_innovation_norm'])  # (T,)
cb_x_hat      = np.stack(all_data['cb_x_hat_new'])        # (T, state_dim)
cb_x_hat_pred = np.stack(all_data['cb_x_hat_pred'])       # (T, state_dim)
cb_corr       = np.stack(all_data['cb_correction'])       # (T, output_dim)
cb_P          = np.stack(all_data['cb_P_diag_new'])       # (T, state_dim)
raw_motor     = np.stack(all_data['raw_motor_cmd'])       # (T, output_dim)
actions       = np.stack(all_data['actions'])              # (T, act_size)
rewards       = np.array(all_data['reward'])               # (T,)

T_ep = ps_sp_e.shape[0]

# Per-env-step rates (average over micro-steps)
rate_e   = np.mean(ps_sp_e, axis=1)   # (T, n_exc)
rate_i   = np.mean(ps_sp_i, axis=1)   # (T, n_inh)
rate_lrn = np.mean(lrn_sp, axis=1)    # (T, n_lrn)

# Sort neurons by mean rate for cleaner rasters
sort_e   = np.argsort(np.mean(rate_e, axis=0))[::-1]
sort_i   = np.argsort(np.mean(rate_i, axis=0))[::-1]

# =============================================================================
# Render video frames
# =============================================================================

print("Rendering MuJoCo frames...")
video_frames = eval_env.render(
    rollout_states, height=480, width=480, render_ghost=True
)
# video_frames has episode_length+1 frames (includes initial state)
# We want T_ep frames corresponding to steps 0..T_ep-1
# Frame i corresponds to state after step i, so use frames[1:]
video_frames = [np.array(f) for f in video_frames[1:]]
n_vid_frames = min(len(video_frames), T_ep)
fps = int(1.0 / eval_env.dt)
print(f"  {n_vid_frames} frames at {fps} fps")

# =============================================================================
# Pre-compute static data for panels
# =============================================================================

# --- Raster data: (T, N_show) mean-rate images, sorted by activity ---
N_SHOW_E = min(n_exc, 100)
N_SHOW_I = min(n_inh, 60)

raster_e = rate_e[:, sort_e[:N_SHOW_E]].T   # (N_show, T)
raster_i = rate_i[:, sort_i[:N_SHOW_I]].T   # (N_show, T)

# --- Action heatmap: (act_size, T) ---
action_img = actions.T  # (act_size, T)

# --- Cerebellum traces ---
corr_norm = np.sqrt(np.sum(cb_corr ** 2, axis=-1))      # (T,)
corr_contrib = correction_weight.item() * corr_norm
raw_motor_norm = np.sqrt(np.sum(raw_motor ** 2, axis=-1))  # (T,)

# --- Spectrogram of PS E population rate ---
pop_rate_e = np.mean(rate_e, axis=1)  # (T,)
env_ctrl_dt = 0.0025
dt_env = env_ctrl_dt  # one env step
fs_env = 1.0 / dt_env  # 400 Hz
nperseg_sg = min(32, T_ep // 4)
nperseg_sg = max(nperseg_sg, 8)
f_sg, t_sg, Sxx = spectrogram(
    pop_rate_e, fs=fs_env, nperseg=nperseg_sg,
    noverlap=nperseg_sg // 2, mode='psd',
)
Sxx_db = 10 * np.log10(Sxx + 1e-20)

# Time axis for env steps (seconds)
t_env = np.arange(T_ep) * dt_env

# =============================================================================
# Panel rendering helpers
# =============================================================================

# We'll create a fixed-layout figure with 6 panels in a 3x2 grid:
#   [Video]     [Actions heatmap]
#   [PS E]      [PS I]
#   [CB traces] [Spectrogram]
#
# Each "frame" of the output video:
#   1. Draw all static panels with data
#   2. Add vertical time-cursor line at current step
#   3. Composite video frame into top-left panel
#   4. Rasterize to numpy array

PANEL_W = 480   # pixels per panel
PANEL_H = 320
GRID_COLS = 2
GRID_ROWS = 3
TOTAL_W = PANEL_W * GRID_COLS
TOTAL_H = PANEL_H * GRID_ROWS

DPI = 100
FIG_W = TOTAL_W / DPI
FIG_H = TOTAL_H / DPI


def fig_to_array(fig):
    """Rasterize matplotlib figure to RGB numpy array."""
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    w, h = fig.canvas.get_width_height()
    return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)


def draw_perturb_span(ax):
    """Draw a red shaded region for the perturbation window."""
    ax.axvspan(perturb_start * dt_env, perturb_end * dt_env,
               alpha=0.15, color='red', zorder=0)


# Pre-render the static parts (everything except the video and time cursor)
# We'll create static images for each panel, then composite per frame.

print("Pre-rendering static panels...")

# --- Panel: Action heatmap ---
fig_act, ax_act = plt.subplots(figsize=(PANEL_W/DPI, PANEL_H/DPI), dpi=DPI)
vmax_act = max(np.percentile(np.abs(action_img), 98), 0.01)
ax_act.imshow(action_img, aspect='auto', cmap='RdBu_r',
              vmin=-vmax_act, vmax=vmax_act,
              extent=[0, T_ep * dt_env, act_size - 0.5, -0.5])
draw_perturb_span(ax_act)
ax_act.set_xlabel('Time (s)', fontsize=8)
ax_act.set_ylabel('Actuator', fontsize=8)
ax_act.set_title('Actions (perturbed)', fontsize=9, fontweight='bold')
ax_act.tick_params(labelsize=7)
fig_act.tight_layout(pad=0.5)
static_act = fig_to_array(fig_act)
plt.close(fig_act)

# --- Panel: PS E raster ---
fig_rE, ax_rE = plt.subplots(figsize=(PANEL_W/DPI, PANEL_H/DPI), dpi=DPI)
vmax_e = max(np.percentile(raster_e, 98), 0.01)
ax_rE.imshow(raster_e, aspect='auto', cmap=CMAP_EXC,
             vmin=0, vmax=vmax_e,
             extent=[0, T_ep * dt_env, N_SHOW_E - 0.5, -0.5])
draw_perturb_span(ax_rE)
ax_rE.set_xlabel('Time (s)', fontsize=8)
ax_rE.set_ylabel('Neuron (sorted)', fontsize=8)
ax_rE.set_title(f'PS Excitatory Raster (n={n_exc})', fontsize=9,
                fontweight='bold', color=C_EXC)
ax_rE.tick_params(labelsize=7)
fig_rE.tight_layout(pad=0.5)
static_rE = fig_to_array(fig_rE)
plt.close(fig_rE)

# --- Panel: PS I raster ---
fig_rI, ax_rI = plt.subplots(figsize=(PANEL_W/DPI, PANEL_H/DPI), dpi=DPI)
vmax_i = max(np.percentile(raster_i, 98), 0.01)
ax_rI.imshow(raster_i, aspect='auto', cmap=CMAP_INH,
             vmin=0, vmax=vmax_i,
             extent=[0, T_ep * dt_env, N_SHOW_I - 0.5, -0.5])
draw_perturb_span(ax_rI)
ax_rI.set_xlabel('Time (s)', fontsize=8)
ax_rI.set_ylabel('Neuron (sorted)', fontsize=8)
ax_rI.set_title(f'PS Inhibitory Raster (n={n_inh})', fontsize=9,
                fontweight='bold', color=C_INH)
ax_rI.tick_params(labelsize=7)
fig_rI.tight_layout(pad=0.5)
static_rI = fig_to_array(fig_rI)
plt.close(fig_rI)

# --- Panel: Cerebellum traces ---
fig_cb, ax_cb = plt.subplots(figsize=(PANEL_W/DPI, PANEL_H/DPI), dpi=DPI)
ax_cb.plot(t_env, cb_innov_norm, color=C_BAL, lw=1.0, alpha=0.8,
           label='||innovation||')
ax_cb.plot(t_env, corr_contrib, color=C_CB, lw=1.2,
           label=f'w*||correction|| (w={correction_weight.item():.4f})')
ax_cb.plot(t_env, raw_motor_norm, color=C_EXC, lw=0.8, alpha=0.6,
           label='||raw motor||')
draw_perturb_span(ax_cb)
ax_cb.set_xlabel('Time (s)', fontsize=8)
ax_cb.set_ylabel('Magnitude', fontsize=8)
ax_cb.set_title('Cerebellum: Prediction Error & Correction', fontsize=9,
                fontweight='bold')
ax_cb.legend(fontsize=6, loc='upper right', framealpha=0.8)
ax_cb.tick_params(labelsize=7)
ax_cb.set_xlim(0, T_ep * dt_env)
fig_cb.tight_layout(pad=0.5)
static_cb = fig_to_array(fig_cb)
plt.close(fig_cb)

# --- Panel: Spectrogram ---
fig_sg, ax_sg = plt.subplots(figsize=(PANEL_W/DPI, PANEL_H/DPI), dpi=DPI)
f_mask = f_sg <= 150  # show up to 150 Hz
if np.any(f_mask):
    im = ax_sg.pcolormesh(t_sg, f_sg[f_mask], Sxx_db[f_mask],
                          shading='gouraud', cmap='viridis')
    fig_sg.colorbar(im, ax=ax_sg, label='dB', pad=0.02, aspect=20)
draw_perturb_span(ax_sg)
ax_sg.set_xlabel('Time (s)', fontsize=8)
ax_sg.set_ylabel('Frequency (Hz)', fontsize=8)
ax_sg.set_title('PS E Rate Spectrogram', fontsize=9, fontweight='bold')
ax_sg.tick_params(labelsize=7)
fig_sg.tight_layout(pad=0.5)
static_sg = fig_to_array(fig_sg)
plt.close(fig_sg)

# Resize static panels to exact target size
def resize_panel(img, w, h):
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

static_act = resize_panel(static_act, PANEL_W, PANEL_H)
static_rE  = resize_panel(static_rE,  PANEL_W, PANEL_H)
static_rI  = resize_panel(static_rI,  PANEL_W, PANEL_H)
static_cb  = resize_panel(static_cb,  PANEL_W, PANEL_H)
static_sg  = resize_panel(static_sg,  PANEL_W, PANEL_H)


def draw_time_cursor(panel_img, t_step, t_min, t_max, color=(255, 0, 0)):
    """Draw a vertical time cursor line on a panel image."""
    img = panel_img.copy()
    frac = (t_step - t_min) / max(t_max - t_min, 1)
    # Account for axis margins (~12% left, ~5% right for typical tight_layout)
    margin_left = 0.14
    margin_right = 0.04
    plot_frac = margin_left + frac * (1.0 - margin_left - margin_right)
    x_px = int(plot_frac * img.shape[1])
    x_px = max(0, min(x_px, img.shape[1] - 1))
    cv2.line(img, (x_px, 0), (x_px, img.shape[0]), color, 2)
    return img


# =============================================================================
# Compose video frames
# =============================================================================

print(f"Compositing {n_vid_frames} video frames...")

# Time range for cursor
t_min_env = 0.0
t_max_env = T_ep * dt_env

# Spectrogram time axis is in the same units (seconds)
t_min_sg = t_sg[0] if len(t_sg) > 0 else 0
t_max_sg = t_sg[-1] if len(t_sg) > 0 else t_max_env

CURSOR_COLOR = (255, 50, 50)  # bright red

with imageio.get_writer(OUTPUT_PATH, fps=fps, quality=8) as vid:
    for frame_idx in range(n_vid_frames):
        t_sec = frame_idx * dt_env  # current time in seconds

        # -- Top-left: video frame (resize to panel size) --
        vf = video_frames[frame_idx]
        vf = resize_panel(vf, PANEL_W, PANEL_H)

        # Add perturbation label
        if perturb_start <= frame_idx < perturb_end:
            cv2.putText(vf, "PERTURBING", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(vf, f"Step {frame_idx}/{T_ep}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # -- Other panels with time cursor --
        act_panel = draw_time_cursor(static_act, t_sec, t_min_env, t_max_env,
                                     CURSOR_COLOR)
        rE_panel  = draw_time_cursor(static_rE,  t_sec, t_min_env, t_max_env,
                                     CURSOR_COLOR)
        rI_panel  = draw_time_cursor(static_rI,  t_sec, t_min_env, t_max_env,
                                     CURSOR_COLOR)
        cb_panel  = draw_time_cursor(static_cb,  t_sec, t_min_env, t_max_env,
                                     CURSOR_COLOR)
        sg_panel  = draw_time_cursor(static_sg,  t_sec, t_min_sg, t_max_sg,
                                     CURSOR_COLOR)

        # -- Assemble grid --
        row0 = np.concatenate([vf, act_panel], axis=1)
        row1 = np.concatenate([rE_panel, rI_panel], axis=1)
        row2 = np.concatenate([cb_panel, sg_panel], axis=1)
        grid = np.concatenate([row0, row1, row2], axis=0)

        vid.append_data(grid)

        if (frame_idx + 1) % 20 == 0 or frame_idx == 0:
            print(f"  frame {frame_idx + 1}/{n_vid_frames}")

print(f"\nVideo saved: {OUTPUT_PATH}")
print(f"  Resolution : {TOTAL_W} x {TOTAL_H}")
print(f"  Frames     : {n_vid_frames}")
print(f"  FPS        : {fps}")
print(f"  Duration   : {n_vid_frames / fps:.1f} s")

# =============================================================================
# Also save a high-res static snapshot at the perturbation midpoint
# =============================================================================

snap_step = (perturb_start + perturb_end) // 2
snap_sec = snap_step * dt_env

print(f"\nSaving static snapshot at step {snap_step} (mid-perturbation)...")

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle(f"Behavior + Neural Snapshot — Step {snap_step} "
             f"(t={snap_sec:.3f}s, mid-perturbation)",
             fontsize=13, fontweight='bold')

# (0,0): Video frame
ax = axes[0, 0]
ax.imshow(video_frames[min(snap_step, len(video_frames) - 1)])
ax.set_title('Rollout (perturbed)', fontsize=10)
ax.axis('off')

# (0,1): Action heatmap
ax = axes[0, 1]
ax.imshow(action_img, aspect='auto', cmap='RdBu_r',
          vmin=-vmax_act, vmax=vmax_act,
          extent=[0, T_ep, act_size - 0.5, -0.5])
ax.axvline(snap_step, color='red', lw=2)
ax.axvspan(perturb_start, perturb_end, alpha=0.15, color='red')
ax.set_xlabel('Env step')
ax.set_ylabel('Actuator')
ax.set_title('Actions', fontsize=10, fontweight='bold')

# (1,0): PS E raster
ax = axes[1, 0]
ax.imshow(raster_e, aspect='auto', cmap=CMAP_EXC, vmin=0, vmax=vmax_e,
          extent=[0, T_ep, N_SHOW_E - 0.5, -0.5])
ax.axvline(snap_step, color='red', lw=2)
ax.axvspan(perturb_start, perturb_end, alpha=0.15, color='red')
ax.set_xlabel('Env step')
ax.set_ylabel('Neuron')
ax.set_title(f'PS Excitatory (n={n_exc})', fontsize=10,
             fontweight='bold', color=C_EXC)

# (1,1): PS I raster
ax = axes[1, 1]
ax.imshow(raster_i, aspect='auto', cmap=CMAP_INH, vmin=0, vmax=vmax_i,
          extent=[0, T_ep, N_SHOW_I - 0.5, -0.5])
ax.axvline(snap_step, color='red', lw=2)
ax.axvspan(perturb_start, perturb_end, alpha=0.15, color='red')
ax.set_xlabel('Env step')
ax.set_ylabel('Neuron')
ax.set_title(f'PS Inhibitory (n={n_inh})', fontsize=10,
             fontweight='bold', color=C_INH)

# (2,0): Cerebellum traces
ax = axes[2, 0]
ax.plot(cb_innov_norm, color=C_BAL, lw=1.0, label='||innovation||')
ax.plot(corr_contrib, color=C_CB, lw=1.2,
        label=f'w*||correction||')
ax.plot(raw_motor_norm, color=C_EXC, lw=0.8, alpha=0.6,
        label='||raw motor||')
ax.axvline(snap_step, color='red', lw=2)
ax.axvspan(perturb_start, perturb_end, alpha=0.15, color='red')
ax.set_xlabel('Env step')
ax.set_ylabel('Magnitude')
ax.set_title('Cerebellum Signals', fontsize=10, fontweight='bold')
ax.legend(fontsize=7, framealpha=0.8)
ax.set_xlim(0, T_ep)

# (2,1): Reward + cumulative
ax = axes[2, 1]
ax.plot(rewards, color='green', lw=1.0, label='Step reward')
ax2 = ax.twinx()
ax2.plot(np.cumsum(rewards), color='darkgreen', lw=1.0, ls='--',
         alpha=0.7, label='Cumulative')
ax.axvline(snap_step, color='red', lw=2)
ax.axvspan(perturb_start, perturb_end, alpha=0.15, color='red')
ax.set_xlabel('Env step')
ax.set_ylabel('Reward')
ax2.set_ylabel('Cumulative reward')
ax.set_title(f'Reward (total={np.sum(rewards):.1f})', fontsize=10,
             fontweight='bold')
ax.legend(fontsize=7, loc='upper left', framealpha=0.8)
ax2.legend(fontsize=7, loc='upper right', framealpha=0.8)
ax.set_xlim(0, T_ep)

fig.tight_layout()
snap_path = os.path.join(OUTPUT_DIR, "behavior_snapshot.png")
fig.savefig(snap_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {snap_path}")

print("\nDone.")
