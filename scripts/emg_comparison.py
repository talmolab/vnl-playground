"""EMG comparison pipeline for Janelia mouse forelimb checkpoints.

Loads any Brax PPO checkpoint, runs deterministic rollouts over all 46 reference
clips, and compares simulated muscle activations against biological EMG recordings.

Usage:
    python scripts/emg_comparison.py --checkpoint checkpoints/S1-17-d1e-7-arm4e-10
    python scripts/emg_comparison.py --checkpoint checkpoints/S1-17-d1e-7-arm4e-10 --step 500695040
"""

import argparse
import json
import os
import sys

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import functools
from pathlib import Path

import jax
import jax.numpy as jp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from brax.training import distribution
from brax.training import networks
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from orbax import checkpoint as ocp

import collections

from vnl_playground.tasks.mouse.imitation import MouseImitation, default_config
from vnl_playground.tasks.mouse.reference_clips import MouseReferenceClips
from vnl_playground.tasks.mouse.consts import JANELIA_MOUSE_XML_PATH, MOUSE_REFERENCE_DATA_PATH
from vnl_playground.eval_metrics import emg as emg_metrics


class MouseImitationV1Obs(MouseImitation):
    """MouseImitation with the old obs structure used for S1/S2 training.

    The S1/S2 checkpoints were trained with _get_imitation_target returning
    OrderedDict(joint=joint_targets, wrist=wrist_targets) rather than the
    current flat concatenation of body deltas for all tracked bodies.
    """

    def _get_imitation_target(self, data, info):
        reference = self._get_imitation_reference(data, info)
        joint_targets = reference.joints - data.qpos
        wrist_pos = data.xpos[self._wrist_body_id]
        wrist_targets = jax.vmap(lambda ref_pos: ref_pos - wrist_pos)(
            reference.body_xpos(self._config.end_effector)
        )
        return collections.OrderedDict(
            joint=joint_targets,
            wrist=wrist_targets,
        )


# ── CLI ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="EMG comparison for Janelia checkpoints")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to checkpoint directory (e.g., checkpoints/S1-17-d1e-7-arm4e-10)")
    p.add_argument("--step", type=int, default=None,
                   help="Checkpoint step to load (default: latest)")
    p.add_argument("--output-dir", type=str, default="results/emg_comparison",
                   help="Output directory for metrics and figures")
    p.add_argument("--n-clips", type=int, default=46,
                   help="Number of reference clips to evaluate")
    p.add_argument("--save-rollouts", action="store_true",
                   help="Save raw rollout data (ctrl, qpos) as .npz")
    p.add_argument("--xml", type=str, default=None,
                   help="Override walker XML path (for testing XML variants)")
    p.add_argument("--latent-size", type=int, default=4,
                   help="Latent size for IntentionPolicy (default: 4)")
    p.add_argument("--emg-norm-percentile", type=float, default=100.0,
                   help="Percentile for reference EMG normalization. 100 matches s15 trainer default; "
                        "use 98 to reproduce pre-s15 metrics.")
    p.add_argument("--output-json", type=str, default=None,
                   help="If set, write metrics_by_muscle as JSON to this path (for s15 Stage 2 aggregation).")
    return p.parse_args()


# ── EMG Processing (matches SCAMPER emg_comparison.ipynb pipeline) ──────

EMG_DIR = "/root/vast/eric/mouse-reach-mjx-neurips/emg"
TRIAL_CSV = "/root/vast/eric/mouse-reach-mjx-neurips/trial_info/A36-1_2023-07-18_16-54-01_lightOff_tone_on_off_trials_edited.csv"

MUSCLE_CONFIGS = [
    (5, "Triceps_Lateral", f"{EMG_DIR}/emg_triceps_fixed_A36-1_2023-07-18_16-54-01_lightOff_tone_on.csv", "Triceps"),
    (8, "Biceps_Long", f"{EMG_DIR}/emg_biceps_fixed_A36-1_2023-07-18_16-54-01_lightOff_tone_on.csv", "Biceps"),
]

TARGET_TIMESTEPS = 60
DURATION_MS = 250
CTRL_DT = 0.0025

ALL_MUSCLE_NAMES = [
    "Pec_C", "Lat", "AD", "PD", "MD",
    "Triceps_Lateral", "Triceps_Long", "Brachialis", "Biceps_Long",
    "Supraspinatus", "Infraspinatus", "Subscapularis",
]


def load_trial_info():
    """Load trial info and return valid trial indices."""
    trial_info = pd.read_csv(TRIAL_CSV)
    valid_mask = ~((trial_info["start"] == 0) & (trial_info["end"] == 0))
    return trial_info[valid_mask]


def process_emg_data(emg_file_path, valid_trials_df, n_clips, target_samples=TARGET_TIMESTEPS,
                     percentile=100.0):
    """Process biological EMG: bandpass -> rectify -> lowpass -> percentile norm.

    Matches the SCAMPER emg_comparison.ipynb pipeline exactly. The percentile used for
    normalization is configurable (default 100.0 to match s15 trainer; use 98 for the
    pre-s15 reproduction).
    """
    fs = 30000
    emg_duration_samples = int(DURATION_MS / 1000 * fs)

    emg_data = pd.read_csv(emg_file_path, header=None)
    envelopes = []

    for i, (idx, row) in enumerate(valid_trials_df.iterrows()):
        if i >= n_clips:
            break
        trial_num = idx
        emg_start = int(1 / 200 * row["start"] * 30000)
        emg_end = emg_start + emg_duration_samples

        if trial_num >= len(emg_data):
            continue

        if emg_start >= 90000 or emg_end > 90000:
            continue

        trial_emg = emg_data.iloc[trial_num, :].values.astype(float)

        # Bandpass 20-1000 Hz
        b, a = signal.butter(4, [20, 1000], btype="bandpass", fs=fs)
        filtered = signal.filtfilt(b, a, trial_emg)

        # Full-wave rectification + 50 Hz lowpass envelope
        b_env, a_env = signal.butter(4, 50, btype="lowpass", fs=fs)
        envelope = signal.filtfilt(b_env, a_env, np.abs(filtered))

        reach_env = envelope[emg_start:emg_end]
        if len(reach_env) > 0:
            resampled = np.interp(
                np.linspace(0, 1, target_samples),
                np.linspace(0, 1, len(reach_env)),
                reach_env,
            )
            envelopes.append(resampled)

    if not envelopes:
        return None

    arr = np.array(envelopes)
    return arr / np.percentile(arr, percentile)


def process_sim_actions(ctrl, target_timesteps=TARGET_TIMESTEPS):
    """Clip sim actions to [0,1], take first 250ms, resample to target_timesteps."""
    actions = np.clip(ctrl, 0.0, 1.0)
    n_clips, T, n_act = actions.shape

    # Take first 250ms worth of steps
    n_steps_250ms = min(int(DURATION_MS / 1000 / CTRL_DT), T)
    actions = actions[:, :n_steps_250ms, :]

    if n_steps_250ms != target_timesteps:
        original_t = np.linspace(0, 1, n_steps_250ms)
        target_t = np.linspace(0, 1, target_timesteps)
        resampled = np.zeros((n_clips, target_timesteps, n_act))
        for c in range(n_clips):
            for m in range(n_act):
                resampled[c, :, m] = np.interp(target_t, original_t, actions[c, :, m])
        actions = resampled

    return actions


# ── Checkpoint Loading ──────────────────────────────────────────────────


def load_config(checkpoint_dir):
    """Load config.json from checkpoint directory."""
    config_path = Path(checkpoint_dir) / "config.json"
    with open(config_path) as f:
        return json.load(f)


def find_latest_step(checkpoint_dir):
    """Find the latest checkpoint step (largest integer subdirectory)."""
    ckpt_path = Path(checkpoint_dir)
    steps = []
    for d in ckpt_path.iterdir():
        if d.is_dir():
            try:
                steps.append(int(d.name))
            except ValueError:
                continue
    if not steps:
        raise ValueError(f"No checkpoint steps found in {checkpoint_dir}")
    return max(steps)


def create_env_from_config(config, xml_override=None):
    """Create MouseImitation env with physics from config.json."""
    env_cfg = default_config()
    from etils import epath
    xml_path = xml_override if xml_override else str(JANELIA_MOUSE_XML_PATH)
    env_cfg.walker_xml_path = epath.Path(xml_path)

    env_cfg.tracked_bodies = config.get("tracked_bodies",
                                         ["scapula", "humerus", "ulna", "wrist"])
    env_cfg.end_effector = config.get("end_effector", "wrist")
    env_cfg.recompute_kinematics = config.get("recompute_kinematics", False)
    env_cfg.qvel_init = config.get("qvel_init", "zeros")

    # Apply physics from config
    if config.get("joint_damping") is not None:
        env_cfg.joint_damping = config["joint_damping"]
    if config.get("joint_armature") is not None:
        env_cfg.joint_armature = config["joint_armature"]
    if config.get("joint_stiffness") is not None:
        env_cfg.joint_stiffness = config["joint_stiffness"]
    if config.get("force_scale") is not None:
        env_cfg.force_scale = config["force_scale"]
    if config.get("reference_length") is not None:
        env_cfg.reference_length = config["reference_length"]
    if config.get("ctrl_dt") is not None:
        env_cfg.ctrl_dt = config["ctrl_dt"]

    # Apply reward terms if present
    if "reward_terms" in config:
        for term, params in config["reward_terms"].items():
            if term in env_cfg.reward_terms:
                for k, v in params.items():
                    env_cfg.reward_terms[term][k] = v

    env = MouseImitation(config=env_cfg)
    return env, env_cfg


# ── IntentionPolicy (must match train_mouse_janelia_intention.py) ──────

from flax import linen


class _Encoder(linen.Module):
    layer_sizes: tuple
    latents: int
    @linen.compact
    def __call__(self, x):
        for i, h in enumerate(self.layer_sizes):
            x = linen.Dense(h, name=f"hidden_{i}", kernel_init=jax.nn.initializers.lecun_uniform())(x)
            x = linen.silu(x)
            x = linen.LayerNorm()(x)
        mean = linen.Dense(self.latents, name="fc_mean")(x)
        logvar = linen.Dense(self.latents, name="fc_logvar")(x)
        return mean, logvar


class _Decoder(linen.Module):
    layer_sizes: tuple
    @linen.compact
    def __call__(self, x):
        for i, h in enumerate(self.layer_sizes):
            x = linen.Dense(h, name=f"hidden_{i}", kernel_init=jax.nn.initializers.lecun_uniform())(x)
            if i != len(self.layer_sizes) - 1:
                x = linen.silu(x)
                x = linen.LayerNorm()(x)
        return x


class _IntentionPolicy(linen.Module):
    encoder_layers: tuple
    decoder_layers: tuple
    latents: int
    proprio_size: int
    def setup(self):
        self.encoder = _Encoder(layer_sizes=self.encoder_layers, latents=self.latents)
        self.decoder = _Decoder(layer_sizes=self.decoder_layers)
    def __call__(self, obs_flat, key, deterministic=False):
        proprio = obs_flat[..., :self.proprio_size]
        task_obs = obs_flat[..., self.proprio_size:]
        mean, logvar = self.encoder(task_obs)
        std = jp.exp(0.5 * logvar)
        eps = jax.random.normal(key, logvar.shape)
        z = jp.where(deterministic, mean, mean + eps * std)
        decoder_input = jp.concatenate([z, proprio], axis=-1)
        logits = self.decoder(decoder_input)
        return logits, mean, logvar


def load_intention_checkpoint(checkpoint_dir, step, obs_size, act_size, proprio_size,
                               latent_size=4, hidden=(512, 512, 512)):
    """Load an IntentionPolicy checkpoint and return inference function + params."""
    action_dist = distribution.NormalTanhDistribution(event_size=act_size)
    param_size = action_dist.param_size

    policy_module = _IntentionPolicy(
        encoder_layers=hidden,
        decoder_layers=tuple(hidden) + (param_size,),
        latents=latent_size,
        proprio_size=proprio_size,
    )
    value_module = networks.MLP(
        layer_sizes=list(hidden) + [1],
        activation=linen.swish,
        kernel_init=jax.nn.initializers.lecun_uniform(),
    )

    dummy_key = jax.random.PRNGKey(0)
    dummy_obs = jp.zeros((1, obs_size))
    normalizer_params = running_statistics.init_state(jp.zeros(obs_size))
    policy_params = policy_module.init(dummy_key, dummy_obs, dummy_key)
    value_params = value_module.init(dummy_key, dummy_obs)
    dummy_params = (normalizer_params, policy_params, value_params)

    ckptr = ocp.PyTreeCheckpointer()
    ckpt_abs = str((Path(checkpoint_dir) / str(step)).resolve())
    params = ckptr.restore(ckpt_abs, item=dummy_params)

    def policy_fn(params, obs, key):
        norm_params, pol_params, _ = params
        obs_norm = running_statistics.normalize(obs, norm_params)
        logits, _, _ = policy_module.apply(pol_params, obs_norm, key, deterministic=True)
        return jp.array(action_dist.mode(logits)), {}

    return params, jax.jit(policy_fn)


# ── Rollout Execution ───────────────────────────────────────────────────


def _flatten_obs(obs):
    """Flatten nested observation dict to a single array (sorted keys)."""
    flat_parts = []
    for key in sorted(obs.keys()):
        val = obs[key]
        if isinstance(val, dict):
            flat_parts.append(_flatten_obs(val))
        else:
            flat_parts.append(val.flatten())
    return jp.concatenate(flat_parts)


def run_rollouts(params, policy_fn, env, n_clips, episode_length):
    """Run deterministic rollouts for all clips using scan + vmap."""

    def single_clip_rollout(clip_idx):
        """Run one clip using lax.scan (no Python loop)."""
        rng = jax.random.PRNGKey(clip_idx)
        state = env.reset(rng, clip_idx=clip_idx, start_frame=0)
        state = state.replace(obs=_flatten_obs(state.obs))

        def step_fn(carry, _):
            state, rng = carry
            rng, step_rng = jax.random.split(rng)
            action, _ = policy_fn(params, state.obs[None], step_rng)
            action = jp.squeeze(action, axis=0)
            next_state = env.step(state, action)
            next_state = next_state.replace(obs=_flatten_obs(next_state.obs))
            return (next_state, rng), (action, next_state.reward)

        _, (actions, rewards) = jax.lax.scan(
            step_fn, (state, rng), None, length=episode_length
        )
        return actions, rewards

    # vmap over all clips
    batched_rollout = jax.jit(jax.vmap(single_clip_rollout))
    clip_indices = jp.arange(n_clips)

    print(f"  JIT compiling batched rollout ({n_clips} clips x {episode_length} steps)...")
    actions, rewards = batched_rollout(clip_indices)
    actions.block_until_ready()
    print(f"  Rollouts complete.")

    return {
        "ctrl": np.array(actions),
        "rewards": np.array(rewards),
    }


# ── Plotting ────────────────────────────────────────────────────────────


def plot_emg_comparison(sim_actions, emg_by_muscle, metrics_by_muscle, checkpoint_name, save_path):
    """Plot EMG comparison figure (matching SCAMPER notebook style)."""
    time_axis = np.linspace(0, 0.25, TARGET_TIMESTEPS)
    colors = ["#1f77b4", "#ef7307"]

    fig, axes = plt.subplots(1, len(MUSCLE_CONFIGS), figsize=(6 * len(MUSCLE_CONFIGS), 5))
    if len(MUSCLE_CONFIGS) == 1:
        axes = [axes]

    for ax, (sim_idx, sim_name, _, muscle_name) in zip(axes, MUSCLE_CONFIGS):
        emg_traces = emg_by_muscle.get(muscle_name)
        if emg_traces is None:
            ax.set_title(f"{muscle_name} - no EMG data")
            continue

        sim_muscle = sim_actions[:, :, sim_idx]
        n_trials = min(sim_muscle.shape[0], emg_traces.shape[0])

        # Individual trials
        for i in range(n_trials):
            ax.plot(time_axis, sim_muscle[i], color=colors[0], alpha=0.1, linewidth=0.5)
            ax.plot(time_axis, emg_traces[i], color=colors[1], alpha=0.1, linewidth=0.5)

        # Mean +/- SEM
        sim_mean = sim_muscle[:n_trials].mean(axis=0)
        sim_sem = sim_muscle[:n_trials].std(axis=0) / np.sqrt(n_trials)
        emg_mean = emg_traces[:n_trials].mean(axis=0)
        emg_sem = emg_traces[:n_trials].std(axis=0) / np.sqrt(n_trials)

        ax.plot(time_axis, sim_mean, color=colors[0], linewidth=2.5, label="Simulated")
        ax.fill_between(time_axis, sim_mean - sim_sem, sim_mean + sim_sem, color=colors[0], alpha=0.25)
        ax.plot(time_axis, emg_mean, color=colors[1], linewidth=2.5, label="Biological EMG")
        ax.fill_between(time_axis, emg_mean - emg_sem, emg_mean + emg_sem, color=colors[1], alpha=0.25)

        m = metrics_by_muscle[muscle_name]
        ax.set_title(f"{muscle_name} (r={m['mean_corr']:.3f}, MAE={m['mean_mae']:.4f}, trial MAE={m['trial_mae']:.4f})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Normalized activation")
        ax.set_ylim(0, 1.2)
        ax.legend(loc="upper right", fontsize=10)

    plt.suptitle(checkpoint_name, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_trial_mae(sim_actions, emg_by_muscle, metrics_by_muscle, checkpoint_name, save_path):
    """Plot per-trial MAE bar chart."""
    fig, axes = plt.subplots(1, len(MUSCLE_CONFIGS), figsize=(7 * len(MUSCLE_CONFIGS), 5))
    if len(MUSCLE_CONFIGS) == 1:
        axes = [axes]

    for ax, (sim_idx, sim_name, _, muscle_name) in zip(axes, MUSCLE_CONFIGS):
        m = metrics_by_muscle.get(muscle_name)
        if m is None:
            continue

        trial_maes = m["trial_maes"]
        x = np.arange(len(trial_maes))
        ax.bar(x, trial_maes, color="#1f77b4", alpha=0.7)
        ax.axhline(m["trial_mae"], color="red", linewidth=1.5, linestyle="--",
                    label=f"Mean MAE = {m['trial_mae']:.4f}")
        ax.set_xlabel("Trial")
        ax.set_ylabel("MAE")
        ax.set_title(f"{muscle_name} - Per-Trial MAE", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)

    plt.suptitle(checkpoint_name, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_mae_over_time(sim_actions, emg_by_muscle, checkpoint_name, save_path):
    """Plot time-resolved MAE (MAE at each timestep, averaged across trials)."""
    time_axis = np.linspace(0, 0.25, TARGET_TIMESTEPS)

    fig, axes = plt.subplots(1, len(MUSCLE_CONFIGS), figsize=(6 * len(MUSCLE_CONFIGS), 4))
    if len(MUSCLE_CONFIGS) == 1:
        axes = [axes]

    for ax, (sim_idx, sim_name, _, muscle_name) in zip(axes, MUSCLE_CONFIGS):
        emg_traces = emg_by_muscle.get(muscle_name)
        if emg_traces is None:
            continue

        sim_muscle = sim_actions[:, :, sim_idx]
        n_trials = min(sim_muscle.shape[0], emg_traces.shape[0])

        # MAE at each timestep, averaged across trials
        abs_err = np.abs(sim_muscle[:n_trials] - emg_traces[:n_trials])
        time_mae = abs_err.mean(axis=0)
        time_sem = abs_err.std(axis=0) / np.sqrt(n_trials)

        ax.plot(time_axis, time_mae, color="#1f77b4", linewidth=2)
        ax.fill_between(time_axis, time_mae - time_sem, time_mae + time_sem,
                         color="#1f77b4", alpha=0.2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("MAE")
        ax.set_title(f"{muscle_name} - MAE Over Time", fontsize=12, fontweight="bold")
        ax.set_ylim(0, None)

    plt.suptitle(checkpoint_name, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_all_muscles(sim_actions, checkpoint_name, save_path):
    """Plot all 12 simulated muscle activations."""
    time_axis = np.linspace(0, 0.25, sim_actions.shape[1])
    n_muscles = sim_actions.shape[2]

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    cmap = plt.cm.tab20

    for m_idx in range(min(n_muscles, 12)):
        ax = axes.flat[m_idx]
        for trial in range(sim_actions.shape[0]):
            ax.plot(time_axis, sim_actions[trial, :, m_idx],
                    color=cmap(trial % 20), alpha=0.3, linewidth=0.5)
        mean = sim_actions[:, :, m_idx].mean(axis=0)
        ax.plot(time_axis, mean, color="black", linewidth=2)
        ax.set_title(ALL_MUSCLE_NAMES[m_idx] if m_idx < len(ALL_MUSCLE_NAMES) else f"Muscle {m_idx}", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Time (s)", fontsize=8)

    for m_idx in range(n_muscles, 12):
        axes.flat[m_idx].set_visible(False)

    plt.suptitle(f"{checkpoint_name} - All Muscles", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Main ────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    checkpoint_dir = args.checkpoint
    checkpoint_name = Path(checkpoint_dir).name

    print(f"=" * 70)
    print(f"EMG Comparison: {checkpoint_name}")
    print(f"=" * 70)

    # Load config
    config = load_config(checkpoint_dir)
    ref_length = config.get("reference_length", 1)
    print(f"  Physics: damping={config.get('joint_damping')}, "
          f"armature={config.get('joint_armature')}, "
          f"force_scale={config.get('force_scale')}, "
          f"ref_length={ref_length}")
    print(f"  Control cost: {config.get('reward_terms', {}).get('control_cost', {}).get('weight', 'N/A')}")

    # Find step
    step = args.step if args.step is not None else find_latest_step(checkpoint_dir)
    print(f"  Step: {step}")

    # Create environment
    env, env_cfg = create_env_from_config(config, xml_override=args.xml)
    print(f"  Env: action_size={env.action_size}, obs_size={env.observation_size}")

    # Determine obs sizes by resetting env
    _rng = jax.random.PRNGKey(99)
    _state = env.reset(_rng)
    _obs = _state.obs["state"]
    _proprio = _obs["proprioception"].flatten()
    _task = _obs["task_obs"]
    if isinstance(_task, dict):
        _parts = []
        for k in sorted(_task.keys()):
            _parts.append(_task[k].flatten())
        _task_flat = jp.concatenate(_parts)
    else:
        _task_flat = _task.flatten()
    proprio_size = _proprio.shape[0]
    obs_size = proprio_size + _task_flat.shape[0]
    act_size = env.action_size
    print(f"  Proprio size: {proprio_size}, Total obs: {obs_size}")

    # Compute episode length from config
    episode_length = config.get("episode_length", 50)
    print(f"  Episode length: {episode_length}")

    # Load IntentionPolicy checkpoint
    latent_size = args.latent_size
    params, policy_fn = load_intention_checkpoint(
        checkpoint_dir, step, obs_size, act_size, proprio_size,
        latent_size=latent_size,
    )
    print(f"  Checkpoint loaded (IntentionPolicy, latent={latent_size}).")

    # Run rollouts
    print(f"  Running {args.n_clips} rollouts...")
    data = run_rollouts(params, policy_fn, env, args.n_clips, episode_length)

    ctrl = data["ctrl"]
    print(f"  Rollout ctrl shape: {ctrl.shape}")

    # Process sim actions
    sim_actions = process_sim_actions(ctrl, TARGET_TIMESTEPS)
    print(f"  Processed sim actions shape: {sim_actions.shape}")

    # Episode reward
    rewards = data["rewards"]
    episode_reward = rewards.sum(axis=1).mean()
    print(f"  Mean episode reward: {episode_reward:.2f}")

    # Load and process biological EMG
    valid_trials = load_trial_info()
    emg_by_muscle = {}
    for _, _, emg_file, muscle_name in MUSCLE_CONFIGS:
        emg_traces = process_emg_data(emg_file, valid_trials, args.n_clips,
                                      percentile=args.emg_norm_percentile)
        if emg_traces is not None:
            emg_by_muscle[muscle_name] = emg_traces

    # Derive ctrl_dt_ms for the shared EMG metrics module.
    ctrl_dt = float(config.get("ctrl_dt", 0.0025))
    ctrl_dt_ms = ctrl_dt * 1000.0

    # Compute metrics
    metrics_by_muscle = {}
    for sim_idx, sim_name, _, muscle_name in MUSCLE_CONFIGS:
        if muscle_name in emg_by_muscle:
            sim_muscle = sim_actions[:, :, sim_idx]
            emg_traces = emg_by_muscle[muscle_name]
            metrics_by_muscle[muscle_name] = emg_metrics.compute_all_emg_metrics(
                sim_muscle, bio_traces=emg_traces, ctrl_dt_ms=ctrl_dt_ms,
            )
            # Reconstruct per-trial arrays for downstream plotting consumers.
            n = min(sim_muscle.shape[0], emg_traces.shape[0])
            trial_corrs_arr = np.array([
                np.corrcoef(sim_muscle[i], emg_traces[i])[0, 1]
                if sim_muscle[i].std() > 0 and emg_traces[i].std() > 0 else np.nan
                for i in range(n)
            ])
            trial_maes_arr = np.array([
                np.mean(np.abs(sim_muscle[i] - emg_traces[i])) for i in range(n)
            ])
            metrics_by_muscle[muscle_name]["trial_corrs"] = trial_corrs_arr
            metrics_by_muscle[muscle_name]["trial_maes"] = trial_maes_arr
            m = metrics_by_muscle[muscle_name]
            trial_mae_sem = float(np.std(trial_maes_arr) / np.sqrt(len(trial_maes_arr))) \
                if len(trial_maes_arr) > 0 else float("nan")
            print(f"  {muscle_name}: mean_corr={m['mean_corr']:.4f}, "
                  f"trial_corr={m['trial_corr_mean']:.4f}, "
                  f"mean_mae={m['mean_mae']:.4f}, "
                  f"trial_mae={m['trial_mae']:.4f} +/- {trial_mae_sem:.4f}")

    # Compute co-contraction index
    biceps_idx = 8
    triceps_idx = 5
    cocontraction = float(np.mean(sim_actions[:, :, biceps_idx] * sim_actions[:, :, triceps_idx]))
    mean_biceps_act = float(np.mean(sim_actions[:, :, biceps_idx]))
    mean_triceps_act = float(np.mean(sim_actions[:, :, triceps_idx]))
    print(f"  Co-contraction index: {cocontraction:.4f}")
    print(f"  Mean biceps activation: {mean_biceps_act:.4f}")
    print(f"  Mean triceps activation: {mean_triceps_act:.4f}")

    # Save outputs
    output_dir = Path(args.output_dir) / checkpoint_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Metrics JSON
    summary = {
        "checkpoint": checkpoint_name,
        "step": step,
        "reference_length": ref_length,
        "joint_damping": config.get("joint_damping"),
        "joint_armature": config.get("joint_armature"),
        "force_scale": config.get("force_scale"),
        "joint_stiffness": config.get("joint_stiffness"),
        "control_cost": config.get("reward_terms", {}).get("control_cost", {}).get("weight"),
        "episode_reward": float(episode_reward),
        "cocontraction_index": cocontraction,
        "mean_biceps_activation": mean_biceps_act,
        "mean_triceps_activation": mean_triceps_act,
    }
    for muscle_name, m in metrics_by_muscle.items():
        summary[f"{muscle_name}_mean_corr"] = m["mean_corr"]
        summary[f"{muscle_name}_mean_trial_corr"] = m["trial_corr_mean"]
        summary[f"{muscle_name}_mean_mae"] = m["mean_mae"]
        summary[f"{muscle_name}_mean_trial_mae"] = m["trial_mae"]
        trial_maes_arr = m.get("trial_maes")
        if trial_maes_arr is not None and len(trial_maes_arr) > 0:
            summary[f"{muscle_name}_trial_mae_sem"] = float(
                np.std(trial_maes_arr) / np.sqrt(len(trial_maes_arr))
            )
        else:
            summary[f"{muscle_name}_trial_mae_sem"] = float("nan")

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    if args.output_json:
        # Convert numpy types to plain Python for JSON serialization.
        def _serialize(obj):
            if isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_serialize(x) for x in obj]
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if hasattr(obj, "item"):
                return obj.item()
            return obj
        with open(args.output_json, "w") as f:
            json.dump({"metrics_by_muscle": _serialize(metrics_by_muscle)}, f, indent=2)
        print(f"  wrote metrics JSON to {args.output_json}")

    # Plots
    plot_emg_comparison(sim_actions, emg_by_muscle, metrics_by_muscle,
                        checkpoint_name, output_dir / "emg_comparison.png")
    plot_per_trial_mae(sim_actions, emg_by_muscle, metrics_by_muscle,
                       checkpoint_name, output_dir / "per_trial_mae.png")
    plot_mae_over_time(sim_actions, emg_by_muscle,
                       checkpoint_name, output_dir / "mae_over_time.png")
    plot_all_muscles(sim_actions, checkpoint_name, output_dir / "all_muscles.png")

    # Save rollout data
    if args.save_rollouts:
        np.savez(output_dir / "rollouts.npz", **{k: v for k, v in data.items()})

    print(f"\n  Results saved to {output_dir}")
    return summary


if __name__ == "__main__":
    main()
