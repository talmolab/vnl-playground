"""Evaluation and visualization for trained gap-jump agents.

Loads a trained checkpoint, runs experimental conditions from Liska et al.,
generates behavioral and neural analysis figures, and renders comparison videos.

Usage:
    python -m track_mjx.scripts.eval_gap_jump \\
        --checkpoint_path /path/to/checkpoint \\
        --mimic_run_id XXXXXX \\
        --output_dir ./eval_results \\
        --n_trials 50 \\
        --render_video

Arguments:
    --checkpoint_path: Path to trained high-level policy checkpoint
    --mimic_run_id: Run ID for the imitation decoder checkpoint
    --output_dir: Directory for output files (figures, data, videos)
    --n_trials: Number of trials per gap distance per condition
    --render_video: Whether to render comparison videos
    --conditions: Which conditions to run (default: all)
"""

import argparse
import functools
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

# Environment flags must be set before importing JAX/MuJoCo.
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import imageio
import jax
import jax.numpy as jp
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from flax.training import orbax_utils
from ml_collections import config_dict
from omegaconf import OmegaConf
from orbax import checkpoint as ocp

import hydra
from mujoco_playground._src import mjx_env
from vnl_playground import registry
from vnl_playground.tasks import wrappers as rodent_wrappers
from vnl_playground.tasks.rodent import run_gap_vision

from track_mjx.agent import checkpointing
from track_mjx.agent.ff_ppo import ppo_networks as ff_ppo_networks
from vnl_playground.tasks.rodent.analysis.gap_jump_experiments import (
    ALL_CONDITIONS,
    BINOCULAR,
    MONOCULAR_LEFT,
    MONOCULAR_RIGHT,
    V1_SUPPRESSION,
    ExperimentConfig,
    compute_mean_decision_time,
    compute_success_rate,
    run_all_conditions,
    save_experiment_results,
)
from vnl_playground.tasks.rodent.analysis.gap_jump_neural_analysis import (
    analyze_cnn_features,
    analyze_latent_intentions,
    analyze_rnn_confidence,
    compute_decision_time_data,
    compute_psychometric_data,
    generate_analysis_report,
    plot_decision_times,
    plot_psychometric_curves,
    plot_rnn_trajectories,
    print_report_summary,
)

# Enable persistent compilation cache.
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def parse_args():
    """Parse command-line arguments for gap-jump evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate gap-jump agent under experimental conditions"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to trained high-level policy checkpoint directory",
    )
    parser.add_argument(
        "--mimic_run_id",
        type=str,
        required=True,
        help="Run ID for the imitation decoder checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Directory for output files (figures, data, videos)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="Number of trials per gap distance per condition",
    )
    parser.add_argument(
        "--render_video",
        action="store_true",
        help="Whether to render comparison videos",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[
            "binocular",
            "monocular_left",
            "monocular_right",
            "v1_suppression",
        ],
        help="Which experimental conditions to run",
    )
    parser.add_argument(
        "--gap_distances",
        nargs="+",
        type=float,
        default=[0.06, 0.08, 0.10, 0.12, 0.14],
        help="Gap distances to evaluate (in meters)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Maximum episode steps per trial",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--episode_length",
        type=int,
        default=1000,
        help="Episode length for rendering rollouts",
    )
    parser.add_argument(
        "--render_height",
        type=int,
        default=480,
        help="Video render height in pixels",
    )
    parser.add_argument(
        "--render_width",
        type=int,
        default=640,
        help="Video render width in pixels",
    )
    parser.add_argument(
        "--mimic_checkpoint_dir",
        type=str,
        default="./model_checkpoints",
        help="Base directory containing mimic checkpoints",
    )
    return parser.parse_args()


def load_policy(
    checkpoint_path: str,
    mimic_run_id: str,
    mimic_checkpoint_dir: str = "./model_checkpoints",
) -> dict[str, Any]:
    """Load trained high-level policy and frozen decoder.

    The high-level policy is a standard Brax PPO MLP saved via
    ``ocp.PyTreeCheckpointer``. The decoder comes from a separate
    imitation training checkpoint.

    Args:
        checkpoint_path: Path to the high-level policy checkpoint directory.
            This should be the directory containing a step subdirectory, e.g.
            ``highlvl_checkpoints/RodentRunGapVision-highlvl-20260101-120000/``
            with a subdirectory like ``250000000/``.
        mimic_run_id: Run ID for the imitation decoder checkpoint.
        mimic_checkpoint_dir: Base directory containing imitation checkpoints.

    Returns:
        Dictionary with keys:
            - "highlvl_params": The high-level policy parameters (tuple).
            - "decoder_policy_fn": Frozen decoder function.
            - "mimic_cfg": Imitation checkpoint configuration.
            - "env_cfg": RunGapVision environment configuration.
            - "ppo_network": Brax PPO network for constructing inference fn.
            - "normalize_fn": Observation normalization function.
    """
    # Load imitation decoder
    mimic_checkpoint_path = hydra.utils.to_absolute_path(
        os.path.join(mimic_checkpoint_dir, mimic_run_id)
    )
    mimic_cfg = OmegaConf.create(
        checkpointing.load_config_from_checkpoint(mimic_checkpoint_path)
    )
    decoder_policy_fn = ff_ppo_networks.make_decoder_policy_fn(mimic_checkpoint_path)

    # RunGapVision environment config
    env_cfg = run_gap_vision.default_config()
    # Match ctrl_dt from imitation training
    env_cfg.ctrl_dt = mimic_cfg.env_config.ctrl_dt

    # Load high-level policy params from checkpoint.
    # The training script saves params via ocp.PyTreeCheckpointer at
    # ``checkpoint_path/<step>/``. We find the latest step subdirectory
    # and restore from it.
    ckpt_path = Path(checkpoint_path)

    # Find the latest checkpoint step directory
    step_dirs = sorted(
        [d for d in ckpt_path.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    if not step_dirs:
        raise FileNotFoundError(f"No checkpoint step directories found in {ckpt_path}")
    latest_step_dir = step_dirs[-1]
    print(f"Loading high-level policy from: {latest_step_dir}")

    # Build a dummy environment + network to get the pytree structure
    # for restoring the checkpoint.
    base_env = registry.load(
        "RodentRunGapVision", config=env_cfg, clips=None, flatten_obs=False
    )
    dummy_env = rodent_wrappers.HighLevelWrapper(
        base_env,
        decoder_policy_fn,
        mimic_cfg.network_config.intention_size,
        highlvl_obs_key="imitation_target",
        decoder_obs_key="proprioception",
    )

    # Standard Brax PPO network (MLP, not intention network)
    normalize_fn = running_statistics.normalize
    rng = jax.random.PRNGKey(0)
    jit_reset = jax.jit(dummy_env.reset)
    start_state = jit_reset(rng)

    network_factory_kwargs = dict(
        policy_hidden_layer_sizes=(1024, 512, 256),
        value_hidden_layer_sizes=(1024, 512, 256),
    )
    ppo_network = ppo_networks.make_ppo_networks(
        start_state.obs.shape[-1],
        dummy_env.action_size,
        preprocess_observations_fn=normalize_fn,
        **network_factory_kwargs,
    )

    # Restore the params.
    # Brax PPO train returns params = (normalizer_params, policy_params),
    # which is saved as a flat pytree via PyTreeCheckpointer.
    orbax_checkpointer = ocp.PyTreeCheckpointer()

    # Create abstract target for restore
    dummy_params = jax.eval_shape(
        lambda: ppo_network.policy_network.init(jax.random.PRNGKey(0))
    )
    # Build a normalizer state template
    normalizer_state = running_statistics.init_state(
        jax.eval_shape(lambda: start_state.obs)
    )
    abstract_params = (normalizer_state, dummy_params)

    # Restore checkpoint
    highlvl_params = orbax_checkpointer.restore(str(latest_step_dir))
    print(f"High-level policy loaded successfully (step {latest_step_dir.name})")

    return {
        "highlvl_params": highlvl_params,
        "decoder_policy_fn": decoder_policy_fn,
        "mimic_cfg": mimic_cfg,
        "env_cfg": env_cfg,
        "ppo_network": ppo_network,
        "normalize_fn": normalize_fn,
    }


def setup_environment(
    env_cfg: config_dict.ConfigDict,
    mimic_cfg: Any,
    decoder_policy_fn: Callable,
) -> rodent_wrappers.HighLevelWrapper:
    """Create evaluation environment with HighLevelWrapper.

    Args:
        env_cfg: RunGapVision environment configuration.
        mimic_cfg: Imitation checkpoint configuration (OmegaConf).
        decoder_policy_fn: Frozen decoder policy function.

    Returns:
        HighLevelWrapper environment ready for evaluation.
    """
    base_env = registry.load(
        "RodentRunGapVision", config=env_cfg, clips=None, flatten_obs=False
    )
    env = rodent_wrappers.HighLevelWrapper(
        base_env,
        decoder_policy_fn,
        mimic_cfg.network_config.intention_size,
        highlvl_obs_key="imitation_target",
        decoder_obs_key="proprioception",
    )
    return env


def make_eval_inference_fn(
    ppo_network: Any,
    highlvl_params: Any,
    deterministic: bool = True,
) -> Callable:
    """Create a deterministic inference function for evaluation.

    Constructs a policy function that takes (observations, rng_key) and
    returns (action, extras). Uses the standard Brax PPO inference pattern
    with fixed parameters.

    Args:
        ppo_network: Brax PPO network (from ``ppo_networks.make_ppo_networks``).
        highlvl_params: Loaded high-level policy parameters.
        deterministic: If True, use the mode of the action distribution.

    Returns:
        JIT-compiled policy function: (obs, rng) -> (action, extras).
    """
    policy_network = ppo_network.policy_network
    parametric_action_distribution = ppo_network.parametric_action_distribution

    def policy_fn(params, observations, key_sample):
        param_subset = (params[0], params[1])
        logits = policy_network.apply(*param_subset, observations)
        if deterministic:
            return (
                jp.array(parametric_action_distribution.mode(logits)),
                {},
            )
        raw_actions = parametric_action_distribution.sample_no_postprocessing(
            logits, key_sample
        )
        log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
        postprocessed_actions = parametric_action_distribution.postprocess(raw_actions)
        return jp.array(postprocessed_actions), {
            "log_prob": log_prob,
            "raw_action": raw_actions,
        }

    jit_policy_fn = jax.jit(policy_fn)

    # Bind the params so the returned function matches the signature
    # expected by run_single_trial: (obs, rng) -> (action, extras)
    def bound_policy(observations, key_sample):
        return jit_policy_fn(highlvl_params, observations, key_sample)

    return bound_policy


def run_eval_rollout(
    env: rodent_wrappers.HighLevelWrapper,
    policy_fn: Callable,
    rng: jax.Array,
    episode_length: int = 1000,
) -> list[mjx_env.State]:
    """Run a single evaluation rollout collecting states.

    Args:
        env: The HighLevelWrapper environment.
        policy_fn: Policy function (obs, rng) -> (action, extras).
        rng: JAX random key.
        episode_length: Number of steps to run.

    Returns:
        List of environment states forming the trajectory.
    """
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng, reset_rng = jax.random.split(rng)
    state = jit_reset(reset_rng)
    rollout = [state]

    for _ in range(episode_length):
        rng, act_rng = jax.random.split(rng)
        action, _ = policy_fn(state.obs, act_rng)
        state = jit_step(state, action)
        rollout.append(state)
        # Stop early if the episode ended
        if float(state.done) > 0.5:
            break

    return rollout


PHASE_NAMES = {0: "HOLD", 1: "DECISION", 2: "JUMP"}
PHASE_COLORS = {
    0: (100, 149, 237),
    1: (255, 165, 0),
    2: (50, 205, 50),
}  # blue, orange, green
OUTCOME_NAMES = {0: "ONGOING", 1: "SUCCESS", 2: "FAILURE", 3: "ABORT", 4: "TIMEOUT"}
OUTCOME_COLORS = {
    0: (200, 200, 200),
    1: (50, 205, 50),
    2: (255, 60, 60),
    3: (255, 165, 0),
    4: (180, 180, 180),
}


def _overlay_trial_info(
    frame: np.ndarray,
    state: mjx_env.State,
    step_idx: int,
    cumulative_reward: float = 0.0,
) -> np.ndarray:
    """Draw rich trial state overlay on a rendered frame.

    Shows trial phase with color indicator, gap distance, per-step and
    cumulative reward, individual reward term breakdown, and trial outcome.

    Args:
        frame: (H, W, 3) uint8 numpy array.
        state: Environment state with info dict and metrics.
        step_idx: Current step index in the episode.
        cumulative_reward: Running total of rewards up to this step.

    Returns:
        Frame with overlay drawn on it.
    """
    img = Image.fromarray(frame)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13
        )
        font_bold = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 13
        )
        font_small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 11
        )
    except (IOError, OSError):
        font = ImageFont.load_default()
        font_bold = font
        font_small = font

    info = state.info
    metrics = state.metrics if hasattr(state, "metrics") else {}

    phase_code = int(info.get("trial_phase", 0))
    phase_name = PHASE_NAMES.get(phase_code, f"?({phase_code})")
    phase_color = PHASE_COLORS.get(phase_code, (200, 200, 200))
    gap_dist = float(info.get("gap_distance", 0.0))
    reward = float(state.reward)
    outcome_code = int(info.get("trial_outcome", 0))
    outcome_name = OUTCOME_NAMES.get(outcome_code, f"?({outcome_code})")
    outcome_color = OUTCOME_COLORS.get(outcome_code, (200, 200, 200))
    done = float(state.done) > 0.5

    # Build lines for the overlay
    lines = []
    colors = []

    # Header: step and phase with color
    lines.append(f"Step {step_idx:4d}  |  {phase_name}")
    colors.append(phase_color)

    # Gap distance
    lines.append(f"Gap: {gap_dist * 100:.1f} cm")
    colors.append((255, 255, 255))

    # Rewards
    lines.append(f"Reward:  {reward:+.3f}")
    colors.append(
        (255, 255, 100)
        if reward > 0
        else (255, 100, 100) if reward < 0 else (200, 200, 200)
    )

    lines.append(f"Cumul:   {cumulative_reward:+.2f}")
    colors.append((255, 255, 255))

    # Separator
    lines.append("\u2500" * 24)
    colors.append((120, 120, 120))

    # Per-term reward breakdown from metrics
    reward_terms = sorted(
        [(k, float(v)) for k, v in metrics.items() if k.startswith("rewards/")],
        key=lambda x: -abs(x[1]),
    )
    for term_key, term_val in reward_terms:
        term_name = term_key.replace("rewards/", "")
        if abs(term_val) > 1e-6:
            lines.append(f"  {term_name[:16]:<16s} {term_val:+.3f}")
            colors.append((180, 220, 255) if term_val > 0 else (255, 180, 180))

    # Separator
    lines.append("\u2500" * 24)
    colors.append((120, 120, 120))

    # Outcome
    lines.append(f"Outcome: {outcome_name}")
    colors.append(outcome_color)

    if done:
        lines.append(">>> DONE <<<")
        colors.append((255, 80, 80))

    # Draw semi-transparent background box
    line_height = 16
    padding = 6
    box_width = 220
    box_height = len(lines) * line_height + 2 * padding
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        [4, 4, 4 + box_width, 4 + box_height],
        fill=(0, 0, 0, 180),
    )

    # Phase indicator bar at top of box
    bar_height = 3
    overlay_draw.rectangle(
        [4, 4, 4 + box_width, 4 + bar_height],
        fill=phase_color + (255,),
    )

    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    y = padding + 4 + bar_height
    for i, (line, color) in enumerate(zip(lines, colors)):
        use_font = (
            font_bold if i == 0 else (font_small if line.startswith("  ") else font)
        )
        draw.text((padding + 4, y), line, fill=color, font=use_font)
        y += line_height

    return np.array(img)


def render_trial_video(
    env: rodent_wrappers.HighLevelWrapper,
    policy_fn: Callable,
    rng: jax.Array,
    output_path: str,
    episode_length: int = 1000,
    camera: str = "close_profile-rodent",
    height: int = 480,
    width: int = 640,
) -> None:
    """Render a single trial as video.

    Runs a rollout, renders each frame using the MuJoCo renderer, and
    saves the resulting video to ``output_path``.

    Args:
        env: The HighLevelWrapper environment.
        policy_fn: Policy function (obs, rng) -> (action, extras).
        rng: JAX random key.
        output_path: File path for the output .mp4 video.
        episode_length: Maximum number of steps.
        camera: MuJoCo camera name to render from.
        height: Render height in pixels.
        width: Render width in pixels.
    """
    rollout = run_eval_rollout(env, policy_fn, rng, episode_length)

    # Use the environment's built-in render method
    # Access the base environment for its render method
    base_env = env
    while hasattr(base_env, "env") or hasattr(base_env, "_env"):
        if hasattr(base_env, "_env"):
            base_env = base_env._env
        elif hasattr(base_env, "env"):
            base_env = base_env.env
        else:
            break

    frames = base_env.render(rollout, height=height, width=width, camera=camera)

    # Track cumulative reward across the episode
    cumulative_reward = 0.0
    overlaid_frames = []
    for i, (frame, state) in enumerate(zip(frames, rollout)):
        cumulative_reward += float(state.reward)
        overlaid_frames.append(
            _overlay_trial_info(
                frame, state, step_idx=i, cumulative_reward=cumulative_reward
            )
        )

    fps = int(1.0 / env.dt)
    imageio.mimsave(output_path, overlaid_frames, fps=fps)
    print(f"  Saved video: {output_path} ({len(overlaid_frames)} frames, {fps} fps)")


def render_comparison_video(
    env: rodent_wrappers.HighLevelWrapper,
    policy_fn: Callable,
    conditions: list[str],
    output_path: str,
    episode_length: int = 1000,
    camera: str = "close_profile-rodent",
    height: int = 480,
    width: int = 640,
    seed: int = 42,
) -> None:
    """Render side-by-side comparison of conditions.

    Generates separate rollouts for each condition using the same seed,
    renders them, and tiles them horizontally into a single video.

    Note: Since vision manipulations happen at the trial runner level
    (via ExperimentConfig), this function renders the same policy under
    the same environment but logs separate videos per condition. A true
    side-by-side comparison is assembled by tiling frames from each
    condition horizontally.

    Args:
        env: The HighLevelWrapper environment.
        policy_fn: Policy function (obs, rng) -> (action, extras).
        conditions: List of condition names for labeling.
        output_path: File path for the output .mp4 video.
        episode_length: Maximum number of steps.
        camera: MuJoCo camera name.
        height: Per-panel render height.
        width: Per-panel render width.
        seed: Random seed (shared across conditions for comparable rollouts).
    """
    # Access the base environment for rendering
    base_env = env
    while hasattr(base_env, "env") or hasattr(base_env, "_env"):
        if hasattr(base_env, "_env"):
            base_env = base_env._env
        elif hasattr(base_env, "env"):
            base_env = base_env.env
        else:
            break

    condition_frames = []
    condition_rollouts = []
    min_length = episode_length

    for i, cond_name in enumerate(conditions):
        rng = jax.random.PRNGKey(seed + i)
        rollout = run_eval_rollout(env, policy_fn, rng, episode_length)
        frames = base_env.render(rollout, height=height, width=width, camera=camera)
        condition_frames.append(frames)
        condition_rollouts.append(rollout)
        min_length = min(min_length, len(frames))

    # Trim all frame sequences to the same length
    condition_frames = [frames[:min_length] for frames in condition_frames]

    # Try to load font once for condition labels
    try:
        label_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 16
        )
    except (IOError, OSError):
        label_font = ImageFont.load_default()

    # Tile frames horizontally with condition labels and trial info overlay
    cumulative_rewards = [0.0] * len(conditions)
    combined_frames = []
    for frame_idx in range(min_length):
        panels = []
        for c in range(len(conditions)):
            frame = condition_frames[c][frame_idx]
            state = condition_rollouts[c][frame_idx]
            cumulative_rewards[c] += float(state.reward)
            # Apply trial info overlay with cumulative reward
            frame = _overlay_trial_info(
                frame,
                state,
                step_idx=frame_idx,
                cumulative_reward=cumulative_rewards[c],
            )
            # Add condition label at bottom-left
            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)
            draw.text(
                (8, height - 22),
                conditions[c],
                fill=(255, 255, 0),
                font=label_font,
            )
            panels.append(np.array(img))
        combined = np.concatenate(panels, axis=1)
        combined_frames.append(combined)

    fps = int(1.0 / env.dt)
    imageio.mimsave(output_path, combined_frames, fps=fps)
    print(
        f"  Saved comparison video: {output_path} "
        f"({len(combined_frames)} frames, {len(conditions)} conditions)"
    )


def generate_figures(report: dict, output_dir: str) -> list[str]:
    """Generate all analysis figures and save to output directory.

    Generates psychometric curves, decision time plots, and RNN trajectory
    visualizations based on the analysis report. Each figure is saved in
    both PNG (150 dpi) and PDF formats.

    Args:
        report: Analysis report dictionary from ``generate_analysis_report``.
        output_dir: Directory to save figure files.

    Returns:
        List of saved file paths.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    saved_files = []

    # 1. Psychometric curves
    if "psychometric" in report and report["psychometric"]:
        fig = plot_psychometric_curves(report["psychometric"])
        png_path = os.path.join(output_dir, "psychometric_curves.png")
        pdf_path = os.path.join(output_dir, "psychometric_curves.pdf")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        saved_files.extend([png_path, pdf_path])
        print(f"  Saved: psychometric_curves.png/pdf")

    # 2. Decision times
    if "decision_times" in report and report["decision_times"]:
        fig = plot_decision_times(report["decision_times"])
        png_path = os.path.join(output_dir, "decision_times.png")
        pdf_path = os.path.join(output_dir, "decision_times.pdf")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        saved_files.extend([png_path, pdf_path])
        print(f"  Saved: decision_times.png/pdf")

    # 3. RNN trajectories (colored by gap distance)
    if "rnn_confidence" in report and "error" not in report["rnn_confidence"]:
        fig = plot_rnn_trajectories(report["rnn_confidence"], color_by="gap_distance")
        png_path = os.path.join(output_dir, "rnn_trajectories_by_distance.png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(png_path)
        print(f"  Saved: rnn_trajectories_by_distance.png")

        # 4. RNN trajectories (colored by time)
        fig = plot_rnn_trajectories(report["rnn_confidence"], color_by="time")
        png_path = os.path.join(output_dir, "rnn_trajectories_by_time.png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(png_path)
        print(f"  Saved: rnn_trajectories_by_time.png")

    # 5. Summary statistics figure
    if "psychometric" in report and report["psychometric"]:
        fig = _plot_summary_panel(report)
        png_path = os.path.join(output_dir, "summary_panel.png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(png_path)
        print(f"  Saved: summary_panel.png")

    return saved_files


def _plot_summary_panel(report: dict) -> "matplotlib.figure.Figure":
    """Create a multi-panel summary figure.

    Combines psychometric curves, decision times, and key statistics
    into a single publication-ready figure.

    Args:
        report: Full analysis report dictionary.

    Returns:
        Matplotlib figure with 2-3 subpanels.
    """
    import matplotlib.pyplot as plt

    n_panels = 2
    if "rnn_confidence" in report and "error" not in report.get("rnn_confidence", {}):
        n_panels = 3

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    colors = {
        "binocular": "black",
        "monocular_left": "blue",
        "monocular_right": "cyan",
        "v1_suppression": "red",
        "v1_suppression_50": "orange",
        "monocular_left_v1": "purple",
    }

    # Panel A: Psychometric curves
    ax = axes[0]
    if "psychometric" in report:
        for condition, data in report["psychometric"].items():
            color = colors.get(condition, "gray")
            ax.errorbar(
                data["distances"] * 100,
                data["success_rates"],
                yerr=[
                    data["success_rates"] - data["ci_lower"],
                    data["ci_upper"] - data["success_rates"],
                ],
                fmt="o-",
                color=color,
                label=condition,
                capsize=3,
                markersize=5,
            )
    ax.set_xlabel("Gap Distance (cm)")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("A. Psychometric Curves")
    ax.legend(frameon=False, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B: Decision times
    ax = axes[1]
    if "decision_times" in report:
        for condition, data in report["decision_times"].items():
            color = colors.get(condition, "gray")
            ax.errorbar(
                data["distances"] * 100,
                data["mean_times"],
                yerr=data["sem_times"],
                fmt="o-",
                color=color,
                label=condition,
                capsize=3,
                markersize=5,
            )
    ax.set_xlabel("Gap Distance (cm)")
    ax.set_ylabel("Decision Time (s)")
    ax.set_title("B. Decision Times")
    ax.legend(frameon=False, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel C: RNN PCA (if available)
    if n_panels >= 3:
        ax = axes[2]
        rnn = report["rnn_confidence"]
        X_pca = rnn["X_pca"]
        trial_ids = rnn["trial_ids"]
        gap_dists = rnn["gap_distances"]
        unique_trials = np.unique(trial_ids)
        for tid in unique_trials:
            mask = trial_ids == tid
            trial_pca = X_pca[mask]
            trial_colors = gap_dists[mask]
            sc = ax.scatter(
                trial_pca[:, 0],
                trial_pca[:, 1],
                c=trial_colors,
                cmap="viridis",
                s=5,
                alpha=0.5,
            )
            ax.plot(
                trial_pca[:, 0],
                trial_pca[:, 1],
                alpha=0.15,
                linewidth=0.5,
                color="gray",
            )
        ev = rnn["explained_variance"]
        ax.set_xlabel(f"PC1 ({ev[0] * 100:.1f}% var)")
        ax.set_ylabel(f"PC2 ({ev[1] * 100:.1f}% var)")
        ax.set_title("C. GRU Trajectories")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


def save_summary_json(report: dict, output_dir: str, args: argparse.Namespace) -> str:
    """Save a JSON-serializable summary of the analysis report.

    Extracts scalar statistics from the full report (which may contain
    non-serializable objects like PCA models) and writes them to a
    JSON file.

    Args:
        report: Full analysis report dictionary.
        output_dir: Output directory.
        args: Parsed command-line arguments.

    Returns:
        Path to the saved JSON file.
    """
    summary = {
        "eval_config": vars(args),
        "timestamp": datetime.now().isoformat(),
    }

    # Psychometric summary
    if "psychometric" in report:
        psych_summary = {}
        for cond, data in report["psychometric"].items():
            psych_summary[cond] = {
                "distances_cm": (data["distances"] * 100).tolist(),
                "success_rates": data["success_rates"].tolist(),
                "n_trials": data["n_trials"].tolist(),
                "mean_success_rate": float(np.mean(data["success_rates"])),
            }
        summary["psychometric"] = psych_summary

    # Decision time summary
    if "decision_times" in report:
        dt_summary = {}
        for cond, data in report["decision_times"].items():
            dt_summary[cond] = {
                "distances_cm": (data["distances"] * 100).tolist(),
                "mean_times_s": data["mean_times"].tolist(),
                "overall_mean_time_s": float(np.mean(data["mean_times"])),
            }
        summary["decision_times"] = dt_summary

    # RNN analysis summary
    if "rnn_confidence" in report and "error" not in report["rnn_confidence"]:
        rnn = report["rnn_confidence"]
        summary["rnn_analysis"] = {
            "distance_decoding_r2": float(rnn["distance_decoding_r2"]),
            "confidence_dim_pc": int(rnn["confidence_dim"]["pc"]),
            "confidence_dim_r": float(rnn["confidence_dim"]["r"]),
            "distance_dim_pc": int(rnn["distance_dim"]["pc"]),
            "distance_dim_r": float(rnn["distance_dim"]["r"]),
            "explained_variance_top3": rnn["explained_variance"][:3].tolist(),
        }

    # CNN analysis summary
    if "cnn_features" in report and "error" not in report["cnn_features"]:
        cnn = report["cnn_features"]
        summary["cnn_analysis"] = {
            "distance_decoding_r2": float(cnn["distance_decoding_r2"]),
            "rsa_correlation": float(cnn["rsa_correlation"]),
            "rsa_p_value": float(cnn["rsa_p_value"]),
        }

    # Latent intention summary
    if "latent_intentions" in report and "error" not in report["latent_intentions"]:
        lat = report["latent_intentions"]
        summary["latent_intentions"] = {
            "distance_prediction_r2": float(lat["distance_prediction_r2"]),
        }

    json_path = os.path.join(output_dir, "analysis_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return json_path


def main():
    """Main evaluation pipeline.

    1. Parse arguments and set up output directory.
    2. Load the trained high-level policy and frozen decoder.
    3. Create the evaluation environment.
    4. Build experimental conditions from CLI arguments.
    5. Run all experimental conditions, collecting trial data.
    6. Generate the full analysis report.
    7. Save figures, structured data, and summary statistics.
    8. Optionally render comparison videos.
    """
    args = parse_args()

    # Setup output directory with timestamp
    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save eval config
    config_path = output_dir / "eval_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Output directory: {output_dir}")
    print(f"Saved eval config to: {config_path}")

    # ----------------------------------------------------------------
    # 1. Load policy
    # ----------------------------------------------------------------
    print("\n[1/6] Loading policy...")
    policy_data = load_policy(
        args.checkpoint_path,
        args.mimic_run_id,
        args.mimic_checkpoint_dir,
    )
    highlvl_params = policy_data["highlvl_params"]
    decoder_policy_fn = policy_data["decoder_policy_fn"]
    mimic_cfg = policy_data["mimic_cfg"]
    env_cfg = policy_data["env_cfg"]
    ppo_network = policy_data["ppo_network"]

    # ----------------------------------------------------------------
    # 2. Setup environment
    # ----------------------------------------------------------------
    print("\n[2/6] Setting up environment...")
    env = setup_environment(env_cfg, mimic_cfg, decoder_policy_fn)
    print(f"  Environment action size: {env.action_size}")
    print(f"  Environment observation size: {env.observation_size}")
    print(f"  Control dt: {env.dt}s")

    # Create inference function
    policy_fn = make_eval_inference_fn(ppo_network, highlvl_params, deterministic=True)

    # ----------------------------------------------------------------
    # 3. Build experiment configs
    # ----------------------------------------------------------------
    print("\n[3/6] Building experiment configurations...")
    conditions = []
    for cond_name in args.conditions:
        config = ExperimentConfig(
            condition=cond_name,
            gap_distances=tuple(args.gap_distances),
            n_trials_per_distance=args.n_trials,
            ctrl_dt=env_cfg.ctrl_dt,
        )
        if "monocular_left" in cond_name:
            config.monocular_side = "left"
        elif "monocular_right" in cond_name:
            config.monocular_side = "right"
        if "v1_suppression" in cond_name:
            config.v1_suppression_fraction = 1.0
        conditions.append(config)
        print(
            f"  Condition: {cond_name} | "
            f"{len(args.gap_distances)} distances x {args.n_trials} trials"
        )

    # ----------------------------------------------------------------
    # 4. Run experiments
    # ----------------------------------------------------------------
    print("\n[4/6] Running experiments...")
    rng = jax.random.PRNGKey(args.seed)
    results = run_all_conditions(
        env,
        policy_fn,
        conditions,
        base_rng=rng,
        max_steps=args.max_steps,
        record_neural=True,
        verbose=True,
    )

    # Print per-condition success rates
    print("\n--- Per-condition success rates ---")
    for cond_name, trials in results.items():
        success_rates = compute_success_rate(trials)
        mean_dt = compute_mean_decision_time(trials)
        n_success = sum(1 for t in trials if t.outcome == "success")
        n_total = len(trials)
        print(
            f"  {cond_name}: {n_success}/{n_total} "
            f"({100 * n_success / max(n_total, 1):.1f}%)"
        )
        for dist, rate in success_rates.items():
            dt = mean_dt.get(dist, 0.0)
            print(
                f"    gap={dist*100:.0f}cm: " f"success={rate:.1%}, mean_dt={dt:.3f}s"
            )

    # ----------------------------------------------------------------
    # 5. Analysis and figures
    # ----------------------------------------------------------------
    print("\n[5/6] Generating analysis report and figures...")

    # Flatten all trials for combined analysis
    all_trials = [t for trials in results.values() for t in trials]

    # Generate full analysis report
    report = generate_analysis_report(all_trials)
    print_report_summary(report)

    # Generate and save figures
    figures_dir = str(output_dir / "figures")
    os.makedirs(figures_dir, exist_ok=True)
    saved_figures = generate_figures(report, figures_dir)
    print(f"  Generated {len(saved_figures)} figure files")

    # Save summary JSON
    json_path = save_summary_json(report, str(output_dir), args)
    print(f"  Saved analysis summary: {json_path}")

    # Save raw experiment results
    results_path = str(output_dir / "experiment_results.npz")
    save_experiment_results(results, results_path)

    # ----------------------------------------------------------------
    # 6. Render videos (optional)
    # ----------------------------------------------------------------
    if args.render_video:
        print("\n[6/6] Rendering videos...")
        videos_dir = output_dir / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)

        # Render a single trial video for each condition
        for i, cond_name in enumerate(args.conditions):
            video_rng = jax.random.PRNGKey(args.seed + 1000 + i)
            video_path = str(videos_dir / f"trial_{cond_name}.mp4")
            try:
                render_trial_video(
                    env,
                    policy_fn,
                    video_rng,
                    output_path=video_path,
                    episode_length=args.episode_length,
                    height=args.render_height,
                    width=args.render_width,
                )
            except Exception as e:
                print(f"  Warning: Failed to render video for {cond_name}: {e}")

        # Render comparison video (all conditions side-by-side)
        comparison_path = str(videos_dir / "comparison_all_conditions.mp4")
        try:
            render_comparison_video(
                env,
                policy_fn,
                args.conditions,
                output_path=comparison_path,
                episode_length=args.episode_length,
                height=args.render_height,
                width=args.render_width,
                seed=args.seed,
            )
        except Exception as e:
            print(f"  Warning: Failed to render comparison video: {e}")
    else:
        print("\n[6/6] Skipping video rendering (use --render_video to enable)")

    # ----------------------------------------------------------------
    # Done
    # ----------------------------------------------------------------
    print(f"\nAll results saved to: {output_dir}")
    print("Contents:")
    for item in sorted(output_dir.rglob("*")):
        if item.is_file():
            rel = item.relative_to(output_dir)
            size_kb = item.stat().st_size / 1024
            print(f"  {rel} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
