# Janelia EMG-Guided Parameter Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an EMG comparison pipeline for existing checkpoints, then run new parameter sweeps (control cost, damping, force scale, latent size x reference length) optimizing for both task reward AND biological EMG fidelity.

**Architecture:** A standalone Python script loads any checkpoint (Brax PPO or Intention VAE), reconstructs the correct physics environment from the stored `config.json`, runs deterministic rollouts over all 46 reference clips, and computes per-trial and average EMG correlation/error metrics against biological recordings. Sweep shell scripts orchestrate training runs. A final analysis notebook produces heatmaps.

**Tech Stack:** JAX, MuJoCo/MJX, Brax PPO, orbax checkpointing, scipy.signal (EMG processing), pandas, matplotlib/seaborn

---

## Context: The Co-Contraction Problem

The current Janelia mouse forelimb model (12 muscles, 4 DOF) tracks reference motion kinematics well, but simulated biceps and triceps activate near 100% simultaneously throughout reaches. Biological EMG shows modulated, reciprocal activation patterns. The key physical differences from the original (working) model:

| Parameter | Old Model (good EMG) | New Model (co-contraction) |
|-----------|---------------------|---------------------------|
| Joint damping | 1e-5 | 1e-6 |
| Joint armature | 4e-8 | 4e-10 |
| Elbow muscle forces | 0.52-0.9 | 0.2 |
| Shoulder muscle forces | 0.8-1.2 | 0.2-0.4 |
| Control cost | 0.01 | 0.01 |

**Hypothesis:** With very low muscle forces and low joint damping/armature, the policy must max out activations to produce any torque. Increasing control cost penalizes this; increasing damping adds passive resistance; increasing force scale makes muscles stronger so they don't need max activation.

## Checkpoint Inventory

Three generations of checkpoints exist, all at `/root/vast/eric/vnl-playground/checkpoints/`:

| Generation | Architecture | Env | ref_length | Count | Physics |
|------------|-------------|-----|------------|-------|---------|
| S1-01 to S1-30 | Brax PPO MLP | MouseImitation | 1 | 30 | damping/armature/stiffness/force sweep |
| S2-01 to S2-24 | Brax PPO MLP | MouseImitation | 1 | 24 | damping x force_scale (finer grid) |
| janelia-intention-* | Intention VAE | MouseImitation | 1 or 2 | 8 | S2-05 physics (d1e-6, arm4e-10, f1.0) |
| sweep-lat*-ref* | Intention VAE | MouseImitation | varies | 5 | S2-05 physics |

**Important:** S1/S2 all used `control_cost=0.01`. The intention runs also use 0.01. None have explored higher control cost yet.

## File Structure

```
scripts/
  emg_comparison.py          # CREATE - Core EMG comparison pipeline (loads any checkpoint)
  emg_compare_all.sh         # CREATE - Batch EMG eval over S1/S2/intention checkpoints
  sweep_control_cost.sh      # CREATE - Control cost sweep (0.01-0.3)
  sweep_damping.sh           # CREATE - Damping sweep (1e-5, 1e-6, 1e-7)
  sweep_force_scale.sh       # CREATE - Force scale sweep
  sweep_latent_reflen.sh     # CREATE - Latent x ref_length grid (replace existing)
  run_all_sweeps.sh          # CREATE - Master sweep orchestrator
results/
  emg_comparison/             # OUTPUT - Per-checkpoint EMG metrics + figures
  emg_summary.csv             # OUTPUT - Aggregated metrics across all checkpoints
  sweep_heatmaps/             # OUTPUT - Heatmap figures from analysis
```

## EMG Data Locations

- Biological EMG CSVs: `/root/vast/eric/mouse-reach-mjx-neurips/emg/`
  - `emg_triceps_fixed_A36-1_2023-07-18_16-54-01_lightOff_tone_on.csv`
  - `emg_biceps_fixed_A36-1_2023-07-18_16-54-01_lightOff_tone_on.csv`
- Trial info: `/root/vast/eric/mouse-reach-mjx-neurips/trial_info/A36-1_2023-07-18_16-54-01_lightOff_tone_on_off_trials_edited.csv`
- EMG processing pipeline reference: `/root/vast/eric/SCAMPER/notebooks/neurips_figures/emg_comparison.ipynb`

---

## Phase 1: EMG Comparison Pipeline

### Task 1: Build the EMG comparison script

**Files:**
- Create: `scripts/emg_comparison.py`

This is the core workhorse. It loads a single checkpoint, runs rollouts, compares against biological EMG, and outputs metrics + figures.

- [ ] **Step 1: Create `scripts/emg_comparison.py` with CLI and checkpoint loading**

```python
"""EMG comparison pipeline for Janelia mouse forelimb checkpoints.

Loads any checkpoint (Brax PPO or Intention VAE), runs deterministic rollouts
over all 46 reference clips, and compares simulated muscle activations against
biological EMG recordings.

Usage:
    python scripts/emg_comparison.py --checkpoint checkpoints/S1-17-d1e-7-arm4e-10
    python scripts/emg_comparison.py --checkpoint checkpoints/S1-17-d1e-7-arm4e-10 --step 500695040
    python scripts/emg_comparison.py --checkpoint checkpoints/janelia-mouse-intention-20260406-232324 --arch intention
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
import seaborn as sns
from scipy import signal
from scipy.interpolate import interp1d

from brax.training import distribution
from brax.training import networks
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from orbax import checkpoint as ocp

from mujoco_playground import wrapper
from vnl_playground.tasks.mouse.imitation import MouseImitation, default_config
from vnl_playground.tasks.mouse.reference_clips import MouseReferenceClips
from vnl_playground.tasks.mouse.consts import JANELIA_MOUSE_XML_PATH, MOUSE_REFERENCE_DATA_PATH


# ── CLI ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="EMG comparison for Janelia checkpoints")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to checkpoint directory (e.g., checkpoints/S1-17-d1e-7-arm4e-10)")
    p.add_argument("--step", type=int, default=None,
                   help="Checkpoint step to load (default: latest)")
    p.add_argument("--arch", type=str, default="auto",
                   choices=["auto", "brax_ppo", "intention"],
                   help="Network architecture (auto-detect from checkpoint)")
    p.add_argument("--output-dir", type=str, default="results/emg_comparison",
                   help="Output directory for metrics and figures")
    p.add_argument("--n-clips", type=int, default=46,
                   help="Number of reference clips to evaluate")
    p.add_argument("--save-rollouts", action="store_true",
                   help="Save raw rollout data (ctrl, qpos) as .npz")
    return p.parse_args()


# ── EMG Processing (matches emg_comparison.ipynb pipeline) ──────────────

EMG_DIR = "/root/vast/eric/mouse-reach-mjx-neurips/emg"
TRIAL_CSV = "/root/vast/eric/mouse-reach-mjx-neurips/trial_info/A36-1_2023-07-18_16-54-01_lightOff_tone_on_off_trials_edited.csv"

MUSCLE_CONFIGS = [
    (5, "Triceps_Lateral", f"{EMG_DIR}/emg_triceps_fixed_A36-1_2023-07-18_16-54-01_lightOff_tone_on.csv", "Triceps"),
    (8, "Biceps_Long", f"{EMG_DIR}/emg_biceps_fixed_A36-1_2023-07-18_16-54-01_lightOff_tone_on.csv", "Biceps"),
]

EMG_RATE = 30000
MOCAP_HZ = 200
CLIP_LENGTH = 50
TARGET_TIMESTEPS = 60
DURATION_MS = 250

ALL_MUSCLE_NAMES = [
    "Pec_C", "Lat", "AD", "PD", "MD",
    "Triceps_Lateral", "Triceps_Long", "Brachialis", "Biceps_Long",
    "Supraspinatus", "Infraspinatus", "Subscapularis",
]


def load_trial_info():
    """Load trial info and return valid trial indices."""
    trial_info = pd.read_csv(TRIAL_CSV)
    valid_mask = ~((trial_info["start"] == 0) & (trial_info["end"] == 0))
    valid_trials_df = trial_info[valid_mask]
    return valid_trials_df


def process_emg_data(emg_file_path, valid_trials_df, n_clips, target_samples=TARGET_TIMESTEPS):
    """Process biological EMG: bandpass -> rectify -> lowpass -> 98th percentile norm.

    Matches the pipeline from emg_comparison.ipynb exactly.
    """
    fs = 30000
    emg_duration_samples = int(DURATION_MS / 1000 * fs)

    emg_data = pd.read_csv(emg_file_path, header=None)
    envelopes = []

    for i, (idx, row) in enumerate(valid_trials_df.iterrows()):
        if i >= n_clips:
            break
        trial_num = idx
        emg_reach_start = int(1 / 200 * row["start"] * 30000)
        emg_reach_end = emg_reach_start + emg_duration_samples

        if trial_num >= len(emg_data):
            continue

        trial_emg = emg_data.iloc[trial_num, :].values.astype(float)

        # Bandpass 20-1000 Hz
        b, a = signal.butter(4, [20, 1000], btype="bandpass", fs=fs)
        filtered = signal.filtfilt(b, a, trial_emg)

        # Full-wave rectification
        rectified = np.abs(filtered)

        # Lowpass envelope 50 Hz
        b_env, a_env = signal.butter(4, 50, btype="lowpass", fs=fs)
        envelope = signal.filtfilt(b_env, a_env, rectified)

        # Extract reach window and resample
        if emg_reach_start < len(envelope) and emg_reach_end <= len(envelope):
            reach_env = envelope[emg_reach_start:emg_reach_end]
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
    norm_val = np.percentile(arr, 98)
    return arr / norm_val


def process_sim_actions(ctrl, target_timesteps=TARGET_TIMESTEPS):
    """Clip sim actions to [0,1], take first half (250ms), resample."""
    actions = np.clip(ctrl, 0.0, 1.0)
    n_clips, T, n_act = actions.shape
    T_half = T // 2
    actions = actions[:, :T_half, :]

    if T_half != target_timesteps:
        original_t = np.linspace(0, 1, T_half)
        target_t = np.linspace(0, 1, target_timesteps)
        resampled = np.zeros((n_clips, target_timesteps, n_act))
        for c in range(n_clips):
            for m in range(n_act):
                resampled[c, :, m] = np.interp(target_t, original_t, actions[c, :, m])
        actions = resampled

    return actions


def compute_emg_metrics(sim_muscle, emg_traces):
    """Compute correlation and MSE between simulated and biological EMG.

    Returns dict with:
        - mean_corr: correlation between trial-averaged traces
        - trial_corrs: per-trial correlations (array)
        - mean_trial_corr: mean of per-trial correlations
        - mean_mse: MSE between trial-averaged traces
        - trial_mses: per-trial MSEs (array)
        - mean_trial_mse: mean of per-trial MSEs
    """
    n_trials = min(sim_muscle.shape[0], emg_traces.shape[0])
    sim = sim_muscle[:n_trials]
    emg = emg_traces[:n_trials]

    # Mean traces
    sim_mean = sim.mean(axis=0)
    emg_mean = emg.mean(axis=0)

    # Average correlation
    mean_corr = np.corrcoef(sim_mean, emg_mean)[0, 1]

    # Average MSE
    mean_mse = np.mean((sim_mean - emg_mean) ** 2)

    # Per-trial metrics
    trial_corrs = []
    trial_mses = []
    for i in range(n_trials):
        r = np.corrcoef(sim[i], emg[i])[0, 1]
        trial_corrs.append(r if np.isfinite(r) else 0.0)
        trial_mses.append(np.mean((sim[i] - emg[i]) ** 2))

    return {
        "mean_corr": float(mean_corr),
        "trial_corrs": np.array(trial_corrs),
        "mean_trial_corr": float(np.mean(trial_corrs)),
        "mean_mse": float(mean_mse),
        "trial_mses": np.array(trial_mses),
        "mean_trial_mse": float(np.mean(trial_mses)),
    }


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


def detect_architecture(checkpoint_dir, step):
    """Auto-detect whether checkpoint is Brax PPO or Intention VAE.

    Brax PPO: params tuple is (normalizer, policy_network_params, value_network_params)
        where policy_network_params has 'params' -> 'hidden_N' structure
    Intention VAE: params tuple is (normalizer, intention_policy_params, value_params)
        where intention_policy_params has 'params' -> 'encoder' and 'params' -> 'decoder'
    """
    meta_path = Path(checkpoint_dir) / str(step) / "_METADATA"
    with open(meta_path, "rb") as f:
        meta = json.loads(f.read())

    keys = list(meta["tree_metadata"].keys())
    # Intention VAE has encoder/decoder in the key paths
    has_encoder = any("encoder" in k for k in keys)
    has_decoder = any("decoder" in k for k in keys)

    if has_encoder and has_decoder:
        return "intention"
    return "brax_ppo"


def infer_brax_ppo_layer_sizes(checkpoint_dir, step):
    """Infer policy/value hidden layer sizes from checkpoint metadata (bias shapes)."""
    meta_path = Path(checkpoint_dir) / str(step) / "_METADATA"
    with open(meta_path, "rb") as f:
        meta = json.loads(f.read())

    # Policy layers are under key '1', value under '2'
    policy_biases = {}
    value_biases = {}
    for key_str, info in meta["tree_metadata"].items():
        shape = info["value_metadata"].get("write_shape", [])
        if "'1'" in key_str and "'bias'" in key_str:
            # Extract layer index from 'hidden_N'
            for part in key_str.split("'"):
                if part.startswith("hidden_"):
                    idx = int(part.split("_")[1])
                    policy_biases[idx] = shape[0] if shape else 0
        if "'2'" in key_str and "'bias'" in key_str:
            for part in key_str.split("'"):
                if part.startswith("hidden_"):
                    idx = int(part.split("_")[1])
                    value_biases[idx] = shape[0] if shape else 0

    # Hidden layers are all except the last (which is the output layer)
    max_policy_idx = max(policy_biases.keys())
    policy_hidden = tuple(policy_biases[i] for i in range(max_policy_idx))
    max_value_idx = max(value_biases.keys())
    value_hidden = tuple(value_biases[i] for i in range(max_value_idx))

    return policy_hidden, value_hidden


def create_env_from_config(config, n_clips=46):
    """Create MouseImitation env with physics from config.json."""
    env_cfg = default_config()
    env_cfg.walker_xml_path = JANELIA_MOUSE_XML_PATH
    env_cfg.tracked_bodies = ["scapula", "humerus", "ulna", "wrist"]
    env_cfg.end_effector = "wrist"
    env_cfg.recompute_kinematics = False

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

    reference_clips = MouseReferenceClips(
        str(env_cfg.reference_data_path),
        n_frames_per_clip=env_cfg.clip_length,
    )
    env = MouseImitation(config=env_cfg, clips=reference_clips)
    return env, env_cfg, reference_clips


def load_brax_ppo_checkpoint(checkpoint_dir, step, env, env_cfg):
    """Load a Brax PPO checkpoint and return inference function + params."""
    from vnl_playground.tasks.wrappers import FlattenObsWrapper

    flat_env = FlattenObsWrapper(env)

    obs_size = flat_env.observation_size
    act_size = flat_env.action_size

    policy_hidden, value_hidden = infer_brax_ppo_layer_sizes(checkpoint_dir, step)
    print(f"  Inferred policy layers: {policy_hidden}, value layers: {value_hidden}")

    normalize = running_statistics.normalize
    ppo_network = ppo_networks.make_ppo_networks(
        observation_size=obs_size,
        action_size=act_size,
        preprocess_observations_fn=normalize,
        policy_hidden_layer_sizes=policy_hidden,
        value_hidden_layer_sizes=value_hidden,
    )

    # Create dummy params for restore tree structure
    dummy_obs = jp.zeros((1, obs_size))
    dummy_key = jax.random.PRNGKey(0)
    normalizer_params = running_statistics.init_state(jp.zeros(obs_size))
    policy_params = ppo_network.policy_network.init(dummy_key)
    value_params = ppo_network.value_network.init(dummy_key)
    dummy_params = (normalizer_params, policy_params, value_params)

    # Restore
    ckptr = ocp.PyTreeCheckpointer()
    params = ckptr.restore(str(Path(checkpoint_dir) / str(step)), item=dummy_params)

    # Build inference function
    def make_policy(deterministic=True):
        policy_network = ppo_network.policy_network
        action_dist = ppo_network.parametric_action_distribution

        def policy_fn(params, obs, key):
            param_subset = (params[0], params[1])
            logits = policy_network.apply(*param_subset, obs)
            if deterministic:
                return action_dist.mode(logits), {}
            raw = action_dist.sample_no_postprocessing(logits, key)
            return action_dist.postprocess(raw), {}

        return policy_fn

    return params, make_policy, flat_env


def load_intention_checkpoint(checkpoint_dir, step, env, env_cfg):
    """Load an Intention VAE checkpoint and return inference function + params."""
    from flax import linen

    # Import IntentionPolicy from training script
    # We inline the essential parts to avoid importing the full training script
    from brax.training import networks as brax_networks

    class Encoder(linen.Module):
        layer_sizes: tuple
        latents: int
        activation = linen.silu

        @linen.compact
        def __call__(self, x):
            for i, size in enumerate(self.layer_sizes):
                x = linen.Dense(size, name=f"hidden_{i}")(x)
                x = self.activation(x)
                x = linen.LayerNorm()(x)
            mean = linen.Dense(self.latents, name="fc_mean")(x)
            logvar = linen.Dense(self.latents, name="fc_logvar")(x)
            return mean, logvar

    class Decoder(linen.Module):
        layer_sizes: tuple
        activation = linen.silu

        @linen.compact
        def __call__(self, x):
            for i, size in enumerate(self.layer_sizes):
                x = linen.Dense(size, name=f"hidden_{i}")(x)
                if i != len(self.layer_sizes) - 1:
                    x = self.activation(x)
                    x = linen.LayerNorm()(x)
            return x

    class IntentionPolicy(linen.Module):
        encoder_layers: tuple
        decoder_layers: tuple
        latents: int
        proprio_size: int

        def setup(self):
            self.encoder = Encoder(layer_sizes=self.encoder_layers, latents=self.latents)
            self.decoder = Decoder(layer_sizes=self.decoder_layers)

        def __call__(self, obs_flat, key, deterministic=False):
            proprio = obs_flat[..., :self.proprio_size]
            task_obs = obs_flat[..., self.proprio_size:]
            mean, logvar = self.encoder(task_obs)
            std = jp.exp(0.5 * logvar)
            eps = jax.random.normal(key, logvar.shape)
            z = jp.where(deterministic, mean, mean + eps * std)
            decoder_input = jp.concatenate([z, proprio], axis=-1)
            return self.decoder(decoder_input), mean, logvar

    # Flatten obs to get sizes
    dummy_rng = jax.random.PRNGKey(99)
    dummy_state = env.reset(dummy_rng)
    obs_dict = dummy_state.obs["state"]
    proprio_flat = obs_dict["proprioception"].flatten()
    task_obs = obs_dict["task_obs"]
    if isinstance(task_obs, dict):
        parts = [task_obs[k].flatten() for k in sorted(task_obs.keys())]
        task_flat = jp.concatenate(parts)
    else:
        task_flat = task_obs.flatten()
    proprio_size = proprio_flat.shape[0]
    obs_size = proprio_size + task_flat.shape[0]
    act_size = env.action_size

    # Infer latent size and layer sizes from checkpoint metadata
    meta_path = Path(checkpoint_dir) / str(step) / "_METADATA"
    with open(meta_path, "rb") as f:
        meta = json.loads(f.read())

    # Find latent size from encoder fc_mean bias
    latent_size = None
    encoder_layers = []
    decoder_layers = []
    for key_str, info in meta["tree_metadata"].items():
        shape = info["value_metadata"].get("write_shape", [])
        if "'1'" in key_str and "'encoder'" in key_str and "'fc_mean'" in key_str and "'bias'" in key_str:
            latent_size = shape[0]
        if "'1'" in key_str and "'encoder'" in key_str and "'bias'" in key_str and "hidden_" in key_str:
            for part in key_str.split("'"):
                if part.startswith("hidden_"):
                    idx = int(part.split("_")[1])
                    encoder_layers.append((idx, shape[0]))
        if "'1'" in key_str and "'decoder'" in key_str and "'bias'" in key_str and "hidden_" in key_str:
            for part in key_str.split("'"):
                if part.startswith("hidden_"):
                    idx = int(part.split("_")[1])
                    decoder_layers.append((idx, shape[0]))

    encoder_layers.sort()
    decoder_layers.sort()
    encoder_hidden = tuple(s for _, s in encoder_layers)
    decoder_hidden = tuple(s for _, s in decoder_layers)

    print(f"  Inferred: encoder={encoder_hidden}, decoder={decoder_hidden}, latent={latent_size}")

    action_dist = distribution.NormalTanhDistribution(event_size=act_size)
    param_size = action_dist.param_size

    policy_module = IntentionPolicy(
        encoder_layers=encoder_hidden,
        decoder_layers=decoder_hidden + (param_size,),
        latents=latent_size,
        proprio_size=proprio_size,
    )
    value_module = brax_networks.MLP(
        layer_sizes=(512, 512, 512, 1),
        activation=linen.swish,
        kernel_init=jax.nn.initializers.lecun_uniform(),
    )

    # Init dummy params for restore
    dummy_obs = jp.zeros((1, obs_size))
    dummy_key = jax.random.PRNGKey(0)
    normalizer_params = running_statistics.init_state(jp.zeros(obs_size))
    policy_params = policy_module.init(dummy_key, dummy_obs, dummy_key)
    value_params = value_module.init(dummy_key, dummy_obs)
    dummy_params = (normalizer_params, policy_params, value_params)

    ckptr = ocp.PyTreeCheckpointer()
    params = ckptr.restore(str(Path(checkpoint_dir) / str(step)), item=dummy_params)

    # Flatten obs helper (sorted keys, same as training script)
    def flatten_obs(obs):
        flat_parts = []
        for key in sorted(obs.keys()):
            val = obs[key]
            if isinstance(val, dict):
                flat_parts.append(flatten_obs(val))
            else:
                flat_parts.append(val.flatten())
        return jp.concatenate(flat_parts)

    def make_policy(deterministic=True):
        def policy_fn(params, obs, key):
            norm_params, pp, vp = params
            obs_norm = running_statistics.normalize(obs, norm_params)
            logits, mean, logvar = policy_module.apply(pp, obs_norm, key, deterministic=deterministic)
            action = action_dist.mode(logits) if deterministic else action_dist.postprocess(
                action_dist.sample_no_postprocessing(logits, key))
            return action, {"latent_mean": mean, "latent_logvar": logvar}

        return policy_fn

    return params, make_policy, flatten_obs, obs_size


# ── Rollout Execution ───────────────────────────────────────────────────


def run_rollouts_brax_ppo(params, make_policy, flat_env, env, reference_clips, n_clips, env_cfg):
    """Run deterministic rollouts for Brax PPO checkpoints."""
    from mujoco_playground import wrapper as wp

    episode_length = int(
        (env_cfg.clip_length - env_cfg.start_frame_range[-1]
         - env_cfg.reference_length) * ((1 / env_cfg.mocap_hz) / env_cfg.ctrl_dt)
    )

    wrapped = wp.wrap_for_brax_training(flat_env, episode_length=episode_length, action_repeat=1)
    policy_fn = make_policy(deterministic=True)
    jit_policy = jax.jit(lambda p, o, k: policy_fn(p, o, k))
    jit_reset = jax.jit(wrapped.reset)
    jit_step = jax.jit(wrapped.step)

    all_ctrl = []
    all_qpos = []
    all_ref_qpos = []
    all_rewards = []

    for clip_idx in range(n_clips):
        rng = jax.random.PRNGKey(clip_idx)
        # Reset to specific clip
        state = flat_env.reset(rng, clip_idx=clip_idx, start_frame=0)
        # Wrap state for brax training wrapper
        from mujoco_playground._src import mjx_env
        state = mjx_env.State(state.data, flat_env._get_obs(state.data, state.info),
                              state.reward, state.done, state.metrics, state.info)
        # Flatten obs
        state = state.replace(obs=jax.flatten_util.ravel_pytree(flat_env.env._get_obs(state.data, state.info))[0])

        ctrl_traj = []
        qpos_traj = []
        ref_qpos_traj = []
        rewards_traj = []

        for t in range(episode_length):
            rng, step_rng = jax.random.split(rng)
            action, _ = jit_policy(params, state.obs[None], step_rng)
            action = jp.squeeze(action, axis=0)

            ctrl_traj.append(np.array(action))
            qpos_traj.append(np.array(state.data.qpos))

            frame_idx = env._get_cur_frame(state.data, state.info)
            clip_i = state.info["reference_clip"]
            ref = reference_clips.at(clip=clip_i, frame=frame_idx)
            ref_qpos_traj.append(np.array(ref.qpos))

            next_state = flat_env.step(state, action)
            next_state = next_state.replace(
                obs=jax.flatten_util.ravel_pytree(
                    flat_env.env._get_obs(next_state.data, next_state.info)
                )[0]
            )
            rewards_traj.append(float(next_state.reward))
            state = next_state

        all_ctrl.append(np.stack(ctrl_traj))
        all_qpos.append(np.stack(qpos_traj))
        all_ref_qpos.append(np.stack(ref_qpos_traj))
        all_rewards.append(np.array(rewards_traj))

        if (clip_idx + 1) % 10 == 0:
            print(f"  Completed {clip_idx + 1}/{n_clips} clips")

    return {
        "ctrl": np.stack(all_ctrl),
        "qpos": np.stack(all_qpos),
        "ref_qpos": np.stack(all_ref_qpos),
        "rewards": np.stack(all_rewards),
    }


def run_rollouts_intention(params, make_policy, flatten_obs_fn, env, reference_clips, n_clips, env_cfg, obs_size):
    """Run deterministic rollouts for Intention VAE checkpoints."""
    episode_length = int(
        (env_cfg.clip_length - env_cfg.start_frame_range[-1]
         - env_cfg.reference_length) * ((1 / env_cfg.mocap_hz) / env_cfg.ctrl_dt)
    )

    policy_fn = make_policy(deterministic=True)
    jit_policy = jax.jit(lambda p, o, k: policy_fn(p, o, k))
    jit_step = jax.jit(env.step)
    jit_reset = jax.jit(env.reset)

    action_dist = distribution.NormalTanhDistribution(event_size=env.action_size)

    all_ctrl = []
    all_qpos = []
    all_ref_qpos = []
    all_rewards = []

    for clip_idx in range(n_clips):
        rng = jax.random.PRNGKey(clip_idx)
        state = env.reset(rng, clip_idx=clip_idx, start_frame=0)

        ctrl_traj = []
        qpos_traj = []
        ref_qpos_traj = []
        rewards_traj = []

        for t in range(episode_length):
            rng, step_rng = jax.random.split(rng)
            flat = flatten_obs_fn(state.obs)
            action, extras = jit_policy(params, flat[None], step_rng)
            action = jp.squeeze(action, axis=0)

            ctrl_traj.append(np.array(action))
            qpos_traj.append(np.array(state.data.qpos))

            frame_idx = env._get_cur_frame(state.data, state.info)
            clip_i = state.info["reference_clip"]
            ref = reference_clips.at(clip=clip_i, frame=frame_idx)
            ref_qpos_traj.append(np.array(ref.qpos))

            state = env.step(state, action)
            rewards_traj.append(float(state.reward))

        all_ctrl.append(np.stack(ctrl_traj))
        all_qpos.append(np.stack(qpos_traj))
        all_ref_qpos.append(np.stack(ref_qpos_traj))
        all_rewards.append(np.array(rewards_traj))

        if (clip_idx + 1) % 10 == 0:
            print(f"  Completed {clip_idx + 1}/{n_clips} clips")

    return {
        "ctrl": np.stack(all_ctrl),
        "qpos": np.stack(all_qpos),
        "ref_qpos": np.stack(all_ref_qpos),
        "rewards": np.stack(all_rewards),
    }


# ── Plotting ────────────────────────────────────────────────────────────


def plot_emg_comparison(sim_actions, emg_by_muscle, metrics_by_muscle, checkpoint_name, save_path):
    """Plot EMG comparison figure (matching style from emg_comparison.ipynb)."""
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
        ax.set_title(f"{muscle_name} (r={m['mean_corr']:.3f}, MSE={m['mean_mse']:.4f})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Normalized activation")
        ax.set_ylim(0, 1.2)
        ax.legend(loc="upper right", fontsize=10)

    plt.suptitle(checkpoint_name, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_all_muscles(sim_actions, checkpoint_name, save_path):
    """Plot all 12 simulated muscle activations (no biological EMG needed)."""
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

    # Detect architecture
    arch = args.arch
    if arch == "auto":
        arch = detect_architecture(checkpoint_dir, step)
    print(f"  Architecture: {arch}")

    # Create environment
    env, env_cfg, reference_clips = create_env_from_config(config, n_clips=args.n_clips)
    print(f"  Env: action_size={env.action_size}, obs_size={env.observation_size}")

    # Load checkpoint and run rollouts
    if arch == "brax_ppo":
        params, make_policy, flat_env = load_brax_ppo_checkpoint(checkpoint_dir, step, env, env_cfg)
        data = run_rollouts_brax_ppo(params, make_policy, flat_env, env, reference_clips, args.n_clips, env_cfg)
    else:
        params, make_policy, flatten_obs_fn, obs_size = load_intention_checkpoint(
            checkpoint_dir, step, env, env_cfg)
        data = run_rollouts_intention(
            params, make_policy, flatten_obs_fn, env, reference_clips, args.n_clips, env_cfg, obs_size)

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
        emg_traces = process_emg_data(emg_file, valid_trials, args.n_clips)
        if emg_traces is not None:
            emg_by_muscle[muscle_name] = emg_traces

    # Compute metrics
    metrics_by_muscle = {}
    for sim_idx, sim_name, _, muscle_name in MUSCLE_CONFIGS:
        if muscle_name in emg_by_muscle:
            metrics_by_muscle[muscle_name] = compute_emg_metrics(
                sim_actions[:, :, sim_idx], emg_by_muscle[muscle_name]
            )
            m = metrics_by_muscle[muscle_name]
            print(f"  {muscle_name}: mean_corr={m['mean_corr']:.4f}, "
                  f"trial_corr={m['mean_trial_corr']:.4f}, "
                  f"mean_mse={m['mean_mse']:.4f}")

    # Compute co-contraction index (mean biceps * triceps activation)
    biceps_idx = 8   # Biceps_Long
    triceps_idx = 5  # Triceps_Lateral
    cocontraction = np.mean(sim_actions[:, :, biceps_idx] * sim_actions[:, :, triceps_idx])
    mean_biceps_act = np.mean(sim_actions[:, :, biceps_idx])
    mean_triceps_act = np.mean(sim_actions[:, :, triceps_idx])
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
        "architecture": arch,
        "reference_length": ref_length,
        "joint_damping": config.get("joint_damping"),
        "joint_armature": config.get("joint_armature"),
        "force_scale": config.get("force_scale"),
        "joint_stiffness": config.get("joint_stiffness"),
        "control_cost": config.get("reward_terms", {}).get("control_cost", {}).get("weight"),
        "episode_reward": float(episode_reward),
        "cocontraction_index": float(cocontraction),
        "mean_biceps_activation": float(mean_biceps_act),
        "mean_triceps_activation": float(mean_triceps_act),
    }
    for muscle_name, m in metrics_by_muscle.items():
        summary[f"{muscle_name}_mean_corr"] = m["mean_corr"]
        summary[f"{muscle_name}_mean_trial_corr"] = m["mean_trial_corr"]
        summary[f"{muscle_name}_mean_mse"] = m["mean_mse"]
        summary[f"{muscle_name}_mean_trial_mse"] = m["mean_trial_mse"]

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plots
    plot_emg_comparison(sim_actions, emg_by_muscle, metrics_by_muscle,
                        checkpoint_name, output_dir / "emg_comparison.png")
    plot_all_muscles(sim_actions, checkpoint_name, output_dir / "all_muscles.png")

    # Save rollout data
    if args.save_rollouts:
        np.savez(output_dir / "rollouts.npz", **{k: v for k, v in data.items()})

    print(f"\n  Results saved to {output_dir}")
    return summary


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test with one S1 checkpoint**

Run:
```bash
cd /root/vast/eric/vnl-playground
python scripts/emg_comparison.py --checkpoint checkpoints/S1-17-d1e-7-arm4e-10
```

Expected: Script loads S1-17, runs 46 rollouts, prints correlation/MSE metrics for Triceps and Biceps, saves figures to `results/emg_comparison/S1-17-d1e-7-arm4e-10/`.

Debug if needed — the Brax PPO checkpoint loading is the trickiest part (matching network shapes to checkpoint). If shapes don't match, adjust `infer_brax_ppo_layer_sizes` logic.

- [ ] **Step 3: Test with one intention checkpoint**

Run:
```bash
python scripts/emg_comparison.py --checkpoint checkpoints/janelia-mouse-intention-20260406-232324 --arch intention
```

Expected: Loads intention VAE checkpoint, runs rollouts, prints metrics. Architecture auto-detection should also work (encoder/decoder keys in metadata).

- [ ] **Step 4: Commit**

```bash
git add scripts/emg_comparison.py
git commit -m "feat: add EMG comparison pipeline for Janelia checkpoints

Loads Brax PPO or Intention VAE checkpoints, runs deterministic rollouts,
compares simulated muscle activations against biological EMG recordings.
Outputs per-trial/average correlation, MSE, co-contraction metrics."
```

---

### Task 2: Batch EMG evaluation across all existing checkpoints

**Files:**
- Create: `scripts/emg_compare_all.sh`

- [ ] **Step 1: Create the batch evaluation script**

```bash
#!/bin/bash
# Batch EMG comparison across all S1, S2, and intention checkpoints.
# Outputs individual results + aggregated CSV summary.

set -e

SCRIPT="scripts/emg_comparison.py"
OUTPUT_BASE="results/emg_comparison"

echo "=============================="
echo "Batch EMG Comparison"
echo "=============================="

# ── S1 checkpoints (Brax PPO, ref_length=1) ──
for ckpt in checkpoints/S1-*/; do
    name=$(basename "$ckpt")
    if [ -f "$OUTPUT_BASE/$name/metrics.json" ]; then
        echo "SKIP $name (already done)"
        continue
    fi
    echo "--- $name ---"
    python $SCRIPT --checkpoint "$ckpt" --arch brax_ppo || echo "FAILED: $name"
done

# ── S2 checkpoints (Brax PPO, ref_length=1) ──
for ckpt in checkpoints/S2-*/; do
    name=$(basename "$ckpt")
    if [ -f "$OUTPUT_BASE/$name/metrics.json" ]; then
        echo "SKIP $name (already done)"
        continue
    fi
    echo "--- $name ---"
    python $SCRIPT --checkpoint "$ckpt" --arch brax_ppo || echo "FAILED: $name"
done

# ── Intention checkpoints (Intention VAE, ref_length varies) ──
for ckpt in checkpoints/janelia-mouse-intention-*/; do
    name=$(basename "$ckpt")
    if [ -f "$OUTPUT_BASE/$name/metrics.json" ]; then
        echo "SKIP $name (already done)"
        continue
    fi
    echo "--- $name ---"
    python $SCRIPT --checkpoint "$ckpt" --arch intention || echo "FAILED: $name"
done

# ── Latent x ref_length sweep checkpoints ──
for ckpt in checkpoints/sweep-lat*/; do
    name=$(basename "$ckpt")
    if [ -f "$OUTPUT_BASE/$name/metrics.json" ]; then
        echo "SKIP $name (already done)"
        continue
    fi
    echo "--- $name ---"
    python $SCRIPT --checkpoint "$ckpt" --arch intention || echo "FAILED: $name"
done

# ── Aggregate results into CSV ──
echo ""
echo "Aggregating results..."
python3 -c "
import json, csv, os
from pathlib import Path

output_dir = Path('$OUTPUT_BASE')
rows = []
for metrics_path in sorted(output_dir.glob('*/metrics.json')):
    with open(metrics_path) as f:
        rows.append(json.load(f))

if rows:
    fieldnames = list(rows[0].keys())
    csv_path = output_dir / 'emg_summary.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {len(rows)} results to {csv_path}')
else:
    print('No results found.')
"

echo "=============================="
echo "Batch EMG comparison complete."
echo "=============================="
```

- [ ] **Step 2: Run the batch evaluation**

```bash
chmod +x scripts/emg_compare_all.sh
bash scripts/emg_compare_all.sh
```

This will take a while (54+ S1/S2 checkpoints + intention checkpoints). Each checkpoint takes ~1-3 minutes for rollouts. Can be interrupted and resumed (skips existing results).

Expected output: `results/emg_comparison/emg_summary.csv` with one row per checkpoint.

- [ ] **Step 3: Commit**

```bash
git add scripts/emg_compare_all.sh
git commit -m "feat: add batch EMG comparison script for all checkpoints"
```

---

## Phase 2: New Parameter Sweeps

### Task 3: Control cost sweep (PRIORITY - run first)

**Files:**
- Create: `scripts/sweep_control_cost.sh`

This is the highest priority sweep. The hypothesis is that increasing control cost will penalize the constant-maxed-out muscle activations.

- [ ] **Step 1: Create control cost sweep script**

```bash
#!/bin/bash
# Control cost sweep: 0.01, 0.02, 0.05, 0.1, 0.2, 0.3
# Uses intention architecture with current best physics (S2-05: d1e-6, arm4e-10, f1.0)
# Trains to 600M steps (reward plateaus before this)

set -e

CTRL_COSTS=(0.01 0.02 0.05 0.1 0.2 0.3)

for CC in "${CTRL_COSTS[@]}"; do
    TAG="ctrl${CC}"
    NAME="sweep-ctrl-cost-${CC}"
    echo "=== Launching control_cost=${CC} ==="
    python train_mouse_janelia_intention.py \
        --control-cost ${CC} \
        --num-timesteps 600000000 \
        --tag "${TAG}" \
        --run-name "${NAME}" \
        --wandb-group "janelia-intention-ctrl-cost-sweep" \
        --wandb-tags "ctrl-cost-sweep" "ctrl${CC}"

    echo "=== Finished control_cost=${CC} ==="
done

echo "Control cost sweep complete (6 runs)."
```

- [ ] **Step 2: Launch the sweep**

```bash
chmod +x scripts/sweep_control_cost.sh
nohup bash scripts/sweep_control_cost.sh > logs/sweep_ctrl_cost.log 2>&1 &
```

- [ ] **Step 3: Commit**

```bash
git add scripts/sweep_control_cost.sh
git commit -m "feat: add control cost sweep script (0.01-0.3)"
```

---

### Task 4: Damping sweep

**Files:**
- Create: `scripts/sweep_damping.sh`

- [ ] **Step 1: Create damping sweep script**

```bash
#!/bin/bash
# Damping sweep: 1e-5, 1e-6, 1e-7
# Uses intention architecture, control_cost=0.01 (baseline)
# armature=4e-10 (current default)

set -e

DAMPINGS=("1e-5" "1e-6" "1e-7")

for DAMP in "${DAMPINGS[@]}"; do
    TAG="damp${DAMP}"
    NAME="sweep-damping-${DAMP}"
    echo "=== Launching damping=${DAMP} ==="
    python train_mouse_janelia_intention.py \
        --joint-damping ${DAMP} \
        --num-timesteps 600000000 \
        --tag "${TAG}" \
        --run-name "${NAME}" \
        --wandb-group "janelia-intention-damping-sweep" \
        --wandb-tags "damping-sweep" "damp${DAMP}"

    echo "=== Finished damping=${DAMP} ==="
done

echo "Damping sweep complete (3 runs)."
```

- [ ] **Step 2: Commit**

```bash
git add scripts/sweep_damping.sh
git commit -m "feat: add damping sweep script (1e-5, 1e-6, 1e-7)"
```

---

### Task 5: Force scale sweep

**Files:**
- Create: `scripts/sweep_force_scale.sh`

- [ ] **Step 1: Create force scale sweep script**

```bash
#!/bin/bash
# Force scale sweep: 0.5, 1.0, 2.0, 3.0
# Tests whether stronger muscles reduce co-contraction
# (if muscles are stronger, they don't need to fire at 100%)

set -e

FSCALES=(0.5 1.0 2.0 3.0)

for FS in "${FSCALES[@]}"; do
    TAG="fscale${FS}"
    NAME="sweep-force-scale-${FS}"
    echo "=== Launching force_scale=${FS} ==="
    python train_mouse_janelia_intention.py \
        --force-scale ${FS} \
        --num-timesteps 600000000 \
        --tag "${TAG}" \
        --run-name "${NAME}" \
        --wandb-group "janelia-intention-force-scale-sweep" \
        --wandb-tags "force-scale-sweep" "fscale${FS}"

    echo "=== Finished force_scale=${FS} ==="
done

echo "Force scale sweep complete (4 runs)."
```

- [ ] **Step 2: Commit**

```bash
git add scripts/sweep_force_scale.sh
git commit -m "feat: add force scale sweep script (0.5-3.0)"
```

---

### Task 6: Top-5 S1 physics retrain with intention architecture

**Files:**
- Create: `scripts/sweep_top5_s1_physics.sh`

Based on the learning curve CSV, these are the top-performing S1 parameter sets that need to be retrained with the intention architecture. The top 5 by reward are:

1. **S1-17** `damp=1e-7, arm=4e-10` (highest at ~240 reward)
2. **S1-07** `damp=1e-7, fscale=1.0` (from the original XML physics, damp 1e-7)
3. **S1-22** `damp=1e-7, stiff=1e-6`
4. **S1-05** `damp=1e-6, fscale=2.0`
5. **S1-09** `damp=1e-7, fscale=3.0`

> **Note:** Verify the actual top 5 from `emg_summary.csv` after Phase 1 completes. The ranking may change when EMG fit (not just reward) is considered.

- [ ] **Step 1: Create the top-5 retrain script**

```bash
#!/bin/bash
# Retrain top-5 S1 physics configs with the intention architecture.
# Physics params come from each S1 run's config.json.
# Trains to 600M steps with intention VAE.

set -e

echo "=== Top-5 S1 physics retrain (intention architecture) ==="

# S1-17: damp=1e-7, arm=4e-10
echo "--- S1-17 physics ---"
python train_mouse_janelia_intention.py \
    --joint-damping 1e-7 \
    --joint-armature 4e-10 \
    --num-timesteps 600000000 \
    --tag "S1-17-retrain" \
    --run-name "intention-S1-17-physics" \
    --wandb-group "janelia-intention-s1-retrain" \
    --wandb-tags "s1-retrain" "S1-17"

# S1-07: damp=1e-7, fscale=1.0 (XML default physics + low damping)
echo "--- S1-07 physics ---"
python train_mouse_janelia_intention.py \
    --joint-damping 1e-7 \
    --num-timesteps 600000000 \
    --tag "S1-07-retrain" \
    --run-name "intention-S1-07-physics" \
    --wandb-group "janelia-intention-s1-retrain" \
    --wandb-tags "s1-retrain" "S1-07"

# S1-22: damp=1e-7, stiff=1e-6
echo "--- S1-22 physics ---"
python train_mouse_janelia_intention.py \
    --joint-damping 1e-7 \
    --joint-stiffness 1e-6 \
    --num-timesteps 600000000 \
    --tag "S1-22-retrain" \
    --run-name "intention-S1-22-physics" \
    --wandb-group "janelia-intention-s1-retrain" \
    --wandb-tags "s1-retrain" "S1-22"

# S1-05: damp=1e-6, fscale=2.0
echo "--- S1-05 physics ---"
python train_mouse_janelia_intention.py \
    --joint-damping 1e-6 \
    --force-scale 2.0 \
    --num-timesteps 600000000 \
    --tag "S1-05-retrain" \
    --run-name "intention-S1-05-physics" \
    --wandb-group "janelia-intention-s1-retrain" \
    --wandb-tags "s1-retrain" "S1-05"

# S1-09: damp=1e-7, fscale=3.0
echo "--- S1-09 physics ---"
python train_mouse_janelia_intention.py \
    --joint-damping 1e-7 \
    --force-scale 3.0 \
    --num-timesteps 600000000 \
    --tag "S1-09-retrain" \
    --run-name "intention-S1-09-physics" \
    --wandb-group "janelia-intention-s1-retrain" \
    --wandb-tags "s1-retrain" "S1-09"

echo "=== Top-5 retrain complete ==="
```

- [ ] **Step 2: Commit**

```bash
git add scripts/sweep_top5_s1_physics.sh
git commit -m "feat: add top-5 S1 physics retrain with intention architecture"
```

---

### Task 7: Latent size x reference length grid

**Files:**
- Create: `scripts/sweep_latent_reflen_grid.sh`

The existing `sweep_latent_reflen.sh` only got through 5 of 12 runs. This creates a clean grid.

- [ ] **Step 1: Create latent x ref_length sweep**

```bash
#!/bin/bash
# Latent size x reference length grid sweep.
# latent: 4, 5, 6, 7
# reference_length: 1, 2, 3
# Total: 12 runs
# Skip any that already exist from the partial sweep.

set -e

for LAT in 4 5 6 7; do
    for REF in 1 2 3; do
        NAME="sweep-lat${LAT}-ref${REF}"
        CKPT_DIR="checkpoints/${NAME}"

        if [ -d "$CKPT_DIR" ] && [ "$(ls -d "$CKPT_DIR"/[0-9]* 2>/dev/null | wc -l)" -gt 2 ]; then
            echo "SKIP ${NAME} (checkpoint exists with >2 steps)"
            continue
        fi

        echo "=== Launching ${NAME} ==="
        python train_mouse_janelia_intention.py \
            --latent-size ${LAT} \
            --reference-length ${REF} \
            --num-timesteps 600000000 \
            --tag "${NAME}" \
            --run-name "${NAME}" \
            --wandb-group "janelia-intention-lat-ref-sweep" \
            --wandb-tags "lat-ref-sweep" "lat${LAT}" "ref${REF}"

        echo "=== Finished ${NAME} ==="
    done
done

echo "Latent x ref_length grid complete."
```

- [ ] **Step 2: Commit**

```bash
git add scripts/sweep_latent_reflen_grid.sh
git commit -m "feat: add latent x ref_length grid sweep (4x3 = 12 runs)"
```

---

### Task 8: Master sweep orchestrator

**Files:**
- Create: `scripts/run_all_sweeps.sh`

- [ ] **Step 1: Create master orchestrator**

```bash
#!/bin/bash
# Master sweep orchestrator.
# Run sweeps in priority order:
#   1. Control cost (most likely to fix co-contraction)
#   2. Damping (passive resistance helps)
#   3. Force scale (muscle strength)
#   4. Top-5 S1 physics retrain
#   5. Latent x ref_length grid
#
# Each sweep can be run independently. This script runs them sequentially.

set -e

mkdir -p logs

echo "============================================"
echo "Janelia EMG Parameter Sweep - Full Pipeline"
echo "============================================"
echo ""

echo "[1/5] Control cost sweep..."
bash scripts/sweep_control_cost.sh 2>&1 | tee logs/sweep_ctrl_cost.log
echo ""

echo "[2/5] Damping sweep..."
bash scripts/sweep_damping.sh 2>&1 | tee logs/sweep_damping.log
echo ""

echo "[3/5] Force scale sweep..."
bash scripts/sweep_force_scale.sh 2>&1 | tee logs/sweep_force_scale.log
echo ""

echo "[4/5] Top-5 S1 physics retrain..."
bash scripts/sweep_top5_s1_physics.sh 2>&1 | tee logs/sweep_top5_retrain.log
echo ""

echo "[5/5] Latent x ref_length grid..."
bash scripts/sweep_latent_reflen_grid.sh 2>&1 | tee logs/sweep_lat_ref.log
echo ""

echo "============================================"
echo "All sweeps complete. Running EMG comparison..."
echo "============================================"

# Re-run batch EMG comparison to include new checkpoints
bash scripts/emg_compare_all.sh 2>&1 | tee logs/emg_compare_all.log

echo "Done. Check results/emg_comparison/emg_summary.csv"
```

- [ ] **Step 2: Commit**

```bash
git add scripts/run_all_sweeps.sh
git commit -m "feat: add master sweep orchestrator"
```

---

## Phase 3: Analysis and Heatmaps

### Task 9: Analysis script for heatmaps and summary

**Files:**
- Create: `scripts/analyze_sweep_results.py`

This reads `results/emg_comparison/emg_summary.csv` and produces heatmaps across parameter axes.

- [ ] **Step 1: Create analysis script**

```python
"""Analyze EMG sweep results and produce heatmaps.

Reads results/emg_comparison/emg_summary.csv and generates:
1. Reward heatmaps (damping x force_scale, damping x control_cost, etc.)
2. EMG correlation heatmaps (same axes)
3. EMG MSE heatmaps
4. Co-contraction index heatmaps
5. Combined ranking table

Usage:
    python scripts/analyze_sweep_results.py
    python scripts/analyze_sweep_results.py --csv results/emg_comparison/emg_summary.csv
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="results/emg_comparison/emg_summary.csv")
    p.add_argument("--output-dir", type=str, default="results/sweep_heatmaps")
    return p.parse_args()


def make_heatmap(df, row_col, col_col, val_col, title, save_path, cmap="viridis", fmt=".3f"):
    """Create a heatmap from a dataframe pivot."""
    if df.empty or row_col not in df.columns or col_col not in df.columns:
        print(f"  SKIP {title}: missing columns")
        return

    pivot = df.pivot_table(index=row_col, columns=col_col, values=val_col, aggfunc="mean")
    if pivot.empty:
        print(f"  SKIP {title}: empty pivot")
        return

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.5), max(4, len(pivot.index) * 0.8)))
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, ax=ax, linewidths=0.5)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} checkpoints from {args.csv}")

    # Separate by architecture for fair comparison
    brax_df = df[df["architecture"] == "brax_ppo"].copy()
    intention_df = df[df["architecture"] == "intention"].copy()

    print(f"  Brax PPO checkpoints: {len(brax_df)}")
    print(f"  Intention checkpoints: {len(intention_df)}")

    # ── Compute composite EMG score ──
    for d in [df, brax_df, intention_df]:
        if "Triceps_mean_corr" in d.columns and "Biceps_mean_corr" in d.columns:
            d["avg_emg_corr"] = (d["Triceps_mean_corr"] + d["Biceps_mean_corr"]) / 2
            d["avg_emg_mse"] = (d["Triceps_mean_mse"] + d["Biceps_mean_mse"]) / 2
            d["avg_trial_corr"] = (d["Triceps_mean_trial_corr"] + d["Biceps_mean_trial_corr"]) / 2

    # ── S1/S2 heatmaps (damping x force_scale) ──
    # Parse physics params from checkpoint names for S1/S2
    for d in [brax_df]:
        d["damping_str"] = d["joint_damping"].apply(lambda x: f"{x:.0e}" if pd.notna(x) else "XML default")
        d["fscale_str"] = d["force_scale"].apply(lambda x: f"{x}" if pd.notna(x) else "1.0")
        d["armature_str"] = d["joint_armature"].apply(lambda x: f"{x:.0e}" if pd.notna(x) else "XML default")

    # Heatmap: reward
    metrics_to_plot = [
        ("episode_reward", "Episode Reward", "viridis"),
        ("avg_emg_corr", "Avg EMG Correlation (mean traces)", "RdYlGn"),
        ("avg_trial_corr", "Avg EMG Correlation (trial-by-trial)", "RdYlGn"),
        ("avg_emg_mse", "Avg EMG MSE (mean traces)", "viridis_r"),
        ("cocontraction_index", "Co-contraction Index", "viridis_r"),
        ("mean_biceps_activation", "Mean Biceps Activation", "viridis_r"),
        ("mean_triceps_activation", "Mean Triceps Activation", "viridis_r"),
    ]

    # Brax PPO: damping x force_scale
    for metric, title, cmap in metrics_to_plot:
        if metric in brax_df.columns:
            make_heatmap(
                brax_df, "damping_str", "fscale_str", metric,
                f"S1/S2: {title}\n(damping x force_scale)",
                output_dir / f"brax_damp_x_fscale_{metric}.png",
                cmap=cmap,
            )

    # ── Intention: control cost analysis (once sweep completes) ──
    ctrl_cost_df = intention_df[intention_df["checkpoint"].str.contains("ctrl-cost", na=False)]
    if len(ctrl_cost_df) > 0:
        for metric, title, cmap in metrics_to_plot:
            if metric in ctrl_cost_df.columns:
                fig, ax = plt.subplots(figsize=(8, 5))
                ctrl_cost_df_sorted = ctrl_cost_df.sort_values("control_cost")
                ax.plot(ctrl_cost_df_sorted["control_cost"], ctrl_cost_df_sorted[metric], "o-", linewidth=2)
                ax.set_xlabel("Control Cost Weight")
                ax.set_ylabel(title)
                ax.set_title(f"Control Cost Sweep: {title}")
                ax.set_xscale("log")
                plt.tight_layout()
                plt.savefig(output_dir / f"intention_ctrl_cost_{metric}.png", dpi=150, bbox_inches="tight")
                plt.close()

    # ── Ranking table ──
    print("\n" + "=" * 80)
    print("TOP 10 BY EMG CORRELATION (avg of Triceps + Biceps mean-trace corr)")
    print("=" * 80)

    if "avg_emg_corr" in df.columns:
        ranking = df.nlargest(10, "avg_emg_corr")[
            ["checkpoint", "architecture", "episode_reward", "avg_emg_corr",
             "avg_trial_corr", "cocontraction_index", "joint_damping", "force_scale", "control_cost"]
        ]
        print(ranking.to_string(index=False))
        ranking.to_csv(output_dir / "top10_emg_corr.csv", index=False)

    print("\n" + "=" * 80)
    print("TOP 10 BY REWARD")
    print("=" * 80)

    ranking_reward = df.nlargest(10, "episode_reward")[
        ["checkpoint", "architecture", "episode_reward", "avg_emg_corr",
         "cocontraction_index", "joint_damping", "force_scale", "control_cost"]
    ]
    print(ranking_reward.to_string(index=False))
    ranking_reward.to_csv(output_dir / "top10_reward.csv", index=False)

    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run after batch EMG comparison completes**

```bash
python scripts/analyze_sweep_results.py
```

Expected: Heatmaps in `results/sweep_heatmaps/`, ranking tables printed to stdout.

- [ ] **Step 3: Commit**

```bash
git add scripts/analyze_sweep_results.py
git commit -m "feat: add sweep analysis script with heatmaps and rankings"
```

---

## Execution Order

The tasks above should be executed in this order, but some can run in parallel:

```
Phase 1 (immediate — no training needed):
  Task 1: Build EMG comparison script
  Task 2: Batch eval all existing S1/S2/intention checkpoints
    → Produces emg_summary.csv
    → Identifies top-5 S1 physics for retrain
    → Gives immediate insight into co-contraction across physics settings

Phase 2 (training sweeps — sequential per GPU, parallel across GPUs):
  Task 3: Control cost sweep (RUN FIRST — highest priority hypothesis)
  Task 4: Damping sweep
  Task 5: Force scale sweep
  Task 6: Top-5 S1 physics retrain (update after Phase 1 results!)
  Task 7: Latent x ref_length grid
  Task 8: Master orchestrator (ties them together)

Phase 3 (analysis — after sweeps complete):
  Task 9: Heatmap analysis
    → Re-run batch EMG comparison for new checkpoints
    → Generate heatmaps and rankings
```

## Key Notes

1. **S1/S2 vs Intention comparison:** S1/S2 use Brax PPO (no latent bottleneck), while intention runs use encoder-decoder VAE. EMG metrics are comparable (both output 12-muscle activations), but note `reference_length` differences in any tables. Do NOT directly compare reward across architectures — only compare EMG metrics.

2. **Test/train split:** The intention training script already splits clips 80/20 (`reference_clips.split(train_ratio=0.8, seed=42)`). The EMG comparison should use ALL 46 clips since we're comparing against biological EMG (not evaluating generalization). The S1/S2 training did NOT split clips.

3. **Top-5 S1 selection:** The initial list in Task 6 is based on reward from the learning curve CSV. After Phase 1 produces EMG metrics, UPDATE the top-5 to rank by EMG correlation (or a composite of reward + EMG fit). The script is editable.

4. **Stopping criterion:** Reward typically plateaus before 500-600M steps. 600M is sufficient. Check wandb curves — if a run has clearly converged by 400M, consider stopping early to save GPU time.

5. **Co-contraction diagnosis:** Beyond the sweeps, watch for these in the EMG comparison output:
   - `mean_biceps_activation > 0.8` AND `mean_triceps_activation > 0.8` = severe co-contraction
   - `cocontraction_index > 0.5` = problematic
   - Biological pattern: biceps and triceps should show **reciprocal** activation (one high, other low, alternating)
