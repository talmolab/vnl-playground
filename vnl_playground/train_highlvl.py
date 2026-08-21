"""Unified high-level transfer training script for VNL tasks.

Trains a high-level policy that outputs latent intentions to a frozen
Phase 1 decoder. Supports two modes:

MLP mode (arch_name="mlp"):
    Standard Brax PPO with flat observations (task_obs body signals).
    HighLevelWrapper produces state/privileged_state flat arrays.

Vision+TaskObs mode (arch_name="vision_task_obs"):
    ff_ppo with CNN encoder fused with task observation (body signals).
    HighLevelWrapper produces both vision and task_obs observations.
    VisionRenderWrapper provides GPU-rendered egocentric frames.

Usage:
    cd vnl-playground
    python -m vnl_playground.train_highlvl
    python -m vnl_playground.train_highlvl --config-name=rodent_run_gap/task_obs_transfer
    python -m vnl_playground.train_highlvl --config-name=rodent_run_gap/vision_task_obs_transfer
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import vnl_playground.naccdmax_patch  # noqa: F401  # monkey-patch naccdmax default

import functools
import gc
import json
import logging
from datetime import datetime
from pathlib import Path

import cv2
import hydra
import imageio
import jax
import jax.numpy as jp
import mujoco
from mujoco.mjx.warp.types import DATA_NON_VMAP
import numpy as np
import psutil
import warp as wp
import wandb
from brax.training.agents.ppo import networks as brax_ppo_networks
from brax.training.agents.ppo import train as brax_ppo_train
from brax.training.acme import running_statistics
from mujoco_playground import wrapper as mp_wrapper
from omegaconf import DictConfig, OmegaConf
from orbax import checkpoint as ocp

from track_mjx.agent import checkpointing
from track_mjx.agent.ff_ppo import ppo as ff_ppo_train
from track_mjx.agent.ff_ppo import ppo_networks as ff_ppo_networks

from vnl_playground import tasks
from vnl_playground.tasks.wrappers import (
    EndToEndWrapper,
    HighLevelWrapper,
    PriorHighLevelWrapper,
)
from vnl_playground import run_state
from vnl_playground.wandb_state import load_wandb_state, save_wandb_state


def _add_batch_dim_for_warp(data):
    """Add leading batch dim to MJX Data, skipping non-vmap fields.

    MJX Warp's FFI layer expects certain fields (contact, efc, etc.) to remain
    unbatched.  A naive ``jax.tree.map(lambda x: x[None, ...], data)`` would
    add a dimension to *every* leaf, violating the Warp type contract and
    causing an AssertionError in ``_expand_dim_from_path``.
    """

    def _maybe_expand(path, x):
        parts = [p.name for p in path if hasattr(p, "name") and p.name != "_impl"]
        attr = "__".join(parts)
        if attr in DATA_NON_VMAP:
            return x
        return x[None, ...]

    return jax.tree.map_with_path(_maybe_expand, data)


# ---------------------------------------------------------------------------
# Optional top-level `wrappers:` block
# ---------------------------------------------------------------------------
# Two knobs, both default-off so every existing config keeps its exact
# behaviour:
#
#   wrappers:
#     full_reset: true                 # BraxAutoResetWrapper calls env.reset()
#     info_reset_on_done: true         # InfoResetOnDoneWrapper (info keys only)
#     info_reset_keys: [a, b, ...]
#
# `full_reset: false` (the historical hardcoded value) makes
# BraxAutoResetWrapper restore `data`/`obs` from the FIRST reset and never
# touch `state.info`.  For an env whose per-episode variation lives in `Data`
# (spawn pose, object positions written into qpos) that means `env.reset` runs
# exactly once per env index for the whole run: every worker replays one frozen
# layout, and info-derived masks/counters ratchet forever.  See
# tasks/wrappers_info_reset.py for the ratchet half of that story, and
# tasks/rodent/maze_forage_vision.py for the frozen-layout half.


def _wrapper_settings(cfg):
    """Reads the optional top-level ``wrappers:`` block.

    Args:
        cfg: The hydra config.

    Returns:
        ``(full_reset, info_reset_on_done, info_reset_keys)``; the keys tuple is
        ``None`` when the config does not name any.
    """
    wrap_cfg = cfg.get("wrappers", None)
    if not wrap_cfg:
        return False, False, None
    keys = wrap_cfg.get("info_reset_keys", None)
    return (
        bool(wrap_cfg.get("full_reset", False)),
        bool(wrap_cfg.get("info_reset_on_done", False)),
        tuple(keys) if keys else None,
    )


def _wrap_for_brax_training(
    environment,
    cfg,
    *,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn=None,
):
    """``mp_wrapper.wrap_for_brax_training`` honouring the ``wrappers:`` block.

    ``InfoResetOnDoneWrapper`` is applied OUTSIDE the brax wrappers on purpose:
    it has to see the post-swap ``done`` flag.

    Args:
        environment: Env to wrap.
        cfg: The hydra config (read for the ``wrappers:`` block).
        episode_length: Truncation horizon for ``EpisodeWrapper``.
        action_repeat: Action repeat for ``EpisodeWrapper``.
        randomization_fn: Optional domain-randomization function.

    Returns:
        The wrapped environment.
    """
    full_reset, info_reset_on_done, info_reset_keys = _wrapper_settings(cfg)
    brax_env = mp_wrapper.wrap_for_brax_training(
        environment,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=randomization_fn,
        full_reset=full_reset,
    )
    if full_reset:
        logging.info(
            "BraxAutoResetWrapper full_reset=True: env.reset() runs on every "
            "done, so Data-side per-episode randomisation and state.info both "
            "reset properly (costs one extra reset per step)."
        )
    if info_reset_on_done:
        from vnl_playground.tasks.wrappers_info_reset import (
            DEFAULT_RUN_GAP_KEYS,
            InfoResetOnDoneWrapper,
        )

        keys = info_reset_keys or DEFAULT_RUN_GAP_KEYS
        brax_env = InfoResetOnDoneWrapper(brax_env, keys=keys)
        logging.info(
            "InfoResetOnDoneWrapper ACTIVE: restoring info keys %s to their "
            "reset-time values on every done.",
            keys,
        )
    return brax_env


def _validate_reset_requirements(cfg, env, env_name: str) -> None:
    """Refuses to launch an env that needs real resets inside a cached-reset stack.

    An env opts in by setting ``requires_per_episode_reset = True`` (and,
    optionally, ``info_reset_keys``).  Without ``wrappers.full_reset`` such an
    env trains on one frozen layout per worker and its ``info`` masks ratchet --
    a silently dead run rather than a crash, which is exactly the failure this
    guard exists to convert into an exception.

    Args:
        cfg: The hydra config.
        env: The loaded (possibly wrapped) environment.
        env_name: Env name, for the error message.

    Raises:
        ValueError: If the env demands per-episode resets and the config does
            not provide them.
    """
    raw_env = env
    while hasattr(raw_env, "env"):
        raw_env = raw_env.env
    if not bool(getattr(raw_env, "requires_per_episode_reset", False)):
        return

    full_reset, info_reset_on_done, info_reset_keys = _wrapper_settings(cfg)
    if full_reset:
        return

    required_keys = tuple(getattr(raw_env, "info_reset_keys", ()) or ())
    wrap_cfg = cfg.get("wrappers", None) or {}
    allow_frozen = bool(wrap_cfg.get("allow_frozen_layout", False))
    covered = info_reset_on_done and not (
        set(required_keys) - set(info_reset_keys or ())
    )
    if allow_frozen and covered:
        logging.warning(
            "%s: wrappers.allow_frozen_layout=true -- env.reset() will run "
            "ONCE per env index for the whole run, so every worker replays a "
            "single frozen layout. Only the info ratchet is fixed. This is an "
            "ablation setting, not the intended training recipe.",
            env_name,
        )
        return

    raise ValueError(
        f"{env_name} sets requires_per_episode_reset=True, but this config "
        "does not enable per-episode resets.\n"
        "BraxAutoResetWrapper(full_reset=False) restores data/obs from the "
        "FIRST reset and never clears state.info, so env.reset() would run "
        "exactly once per env index for the entire run: every worker replays "
        "one frozen spawn/object layout and info masks ratchet until the env "
        "emits 1-step, zero-reward episodes forever.\n"
        "Add to the config:\n"
        "  wrappers:\n"
        "    full_reset: true\n"
        "Deliberate frozen-layout ablation instead? Set\n"
        "  wrappers:\n"
        "    allow_frozen_layout: true\n"
        "    info_reset_on_done: true\n"
        f"    info_reset_keys: {list(required_keys)}"
    )


@functools.lru_cache(maxsize=4)
def _make_render_all_fn(renderer_id, renderer):
    """Return a cached JIT-compiled scan-render function for a given renderer.

    ``renderer_id`` = ``id(renderer)`` serves as a hashable cache key so that
    the same renderer always reuses its compiled XLA/Warp kernel instead of
    leaking a new one on every call.
    """

    @jax.jit
    def _render_all(stacked_data):
        def body(carry, data_slice):
            batched = _add_batch_dim_for_warp(data_slice)
            img = renderer.render(batched)
            return carry, img[0]

        _, all_imgs = jax.lax.scan(body, None, stacked_data)
        return all_imgs

    return _render_all


class _FlatObsAdapter:
    """Adapts HighLevelWrapper's dict observations to flat arrays for Brax PPO.

    In MLP mode, HighLevelWrapper returns {"state": flat, "privileged_state": flat}.
    Brax PPO expects flat array observations and int observation_size.
    This adapter extracts obs["state"], making the interface Brax-compatible.

    Note: HighLevelWrapper.step() reads proprioception from info["_full_obs"],
    not from state.obs, so replacing obs with a flat array is safe.
    """

    def __init__(self, env, obs_key="state"):
        self._env = env
        self._obs_key = obs_key

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, rng, **kwargs):
        state = self._env.reset(rng, **kwargs)
        return state.replace(obs=state.obs[self._obs_key])

    def step(self, state, action):
        state = self._env.step(state, action)
        return state.replace(obs=state.obs[self._obs_key])

    @property
    def observation_size(self):
        return self._env.observation_size[self._obs_key]

    @property
    def action_size(self):
        return self._env.action_size


def _log_memory(label: str):
    """Log current process RSS memory usage."""
    proc = psutil.Process()
    rss_gb = proc.memory_info().rss / (1024**3)
    logging.info(f"[MEM] {label}: {rss_gb:.2f} GB RSS")
    return rss_gb


def _log_gpu_memory(label: str):
    """Log GPU memory usage from both JAX and CUDA perspectives."""
    try:
        backend = jax.lib.xla_bridge.get_backend()
        for device in backend.devices():
            stats = device.memory_stats()
            if stats:
                used_mb = stats.get("bytes_in_use", 0) / 1e6
                peak_mb = stats.get("peak_bytes_in_use", 0) / 1e6
                logging.info(
                    f"[GPU-MEM] {label}: JAX used={used_mb:.0f}MB "
                    f"peak={peak_mb:.0f}MB"
                )
    except Exception:
        pass
    try:
        import subprocess

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free",
                "--format=csv,nounits,noheader",
                "--id=0",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            used, free = result.stdout.strip().split(", ")
            logging.info(f"[GPU-MEM] {label}: CUDA used={used}MB free={free}MB")
    except Exception:
        pass


def _prepare_ego_overlay(ego_frames_np, scale=2):
    """Prepare egocentric frames for overlay compositing.

    Takes (T, H, W, C) float32 [0,1] array from warp GPU render and returns
    (T, H*scale, W*scale, 3) uint8 array suitable for overlay.
    """
    # If grayscale (C=1), expand to RGB
    if ego_frames_np.shape[-1] == 1:
        ego_frames_np = np.repeat(ego_frames_np, 3, axis=-1)
    # Convert to uint8
    ego_uint8 = np.clip(ego_frames_np * 255, 0, 255).astype(np.uint8)
    # Scale up via np.repeat on spatial axes
    ego_scaled = np.repeat(np.repeat(ego_uint8, scale, axis=1), scale, axis=2)
    return ego_scaled


def _get_termination_reason(state):
    """Extract termination reason string from state metrics.

    Reads ``terminations/<name>`` metrics populated by the task registry's
    ``_is_done`` method. Returns the name of the first active termination
    criterion, or ``"unknown"`` if none found.
    """
    for key in state.metrics:
        if key.startswith("terminations/") and key != "terminations/any":
            if float(state.metrics[key]) > 0.5:
                return key.replace("terminations/", "")
    return "unknown"


def _run_eval_rollout(jit_reset, jit_step, inference_fn, params, episode_length, rng):
    """Run an evaluation rollout with termination detection and auto-reset.

    Instead of blindly stepping for ``episode_length`` steps (which continues
    rendering a dead/fallen agent), this detects ``state.done`` and resets the
    environment to start a new episode.

    Returns:
        rollout: list of states (may span multiple episodes).
        termination_events: list of ``(frame_index, reason_string)`` tuples
            marking where each episode ended and why.
    """
    _, reset_rng, act_rng = jax.random.split(rng, 3)
    state = jit_reset(reset_rng)
    rollout = [state]
    termination_events = []

    for _ in range(episode_length):
        _, act_rng = jax.random.split(act_rng)
        action, _ = inference_fn(params, state.obs, act_rng)
        state = jit_step(state, action)
        rollout.append(state)

        if float(state.done) > 0.5:
            reason = _get_termination_reason(state)
            termination_events.append((len(rollout) - 1, reason))
            # Reset for a new episode
            _, reset_rng = jax.random.split(act_rng)
            state = jit_reset(reset_rng)
            rollout.append(state)

    return rollout, termination_events


def _draw_hud(
    frame, lines, x=10, y_start=20, line_height=22, font_scale=0.5, thickness=1
):
    """Draw multiple lines of HUD text with black shadow for readability.

    Each entry in ``lines`` is ``(text, color_bgr)``.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (text, color) in enumerate(lines):
        y = y_start + i * line_height
        cv2.putText(
            frame,
            text,
            (x + 1, y + 1),
            font,
            font_scale,
            (0, 0, 0),
            thickness + 1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA
        )


def render_video(
    rollout,
    mj_model,
    mj_data,
    renderer,
    video_path,
    fps=50,
    vision_renderer=None,
    right_vision_renderer=None,
    termination_events=None,
    termination_fade_seconds=1.0,
    hud_config=None,
    reward_config=None,
    use_obs_vision=False,
    eye_qpos_indices=None,
    reward_remix=None,
):
    """Render a rollout to an MP4 video file with tracking camera.

    If ``vision_renderer`` (a ``JaxVisionRenderer`` with nworld=1) is
    provided, the agent's egocentric view rendered by the warp GPU
    ray-tracer is overlaid in the upper-left corner of each frame.

    If ``right_vision_renderer`` is also provided, both left and right
    eye views are rendered and displayed side-by-side in the overlay
    (binocular stereo visualization).

    Egocentric renders are batched into a single JAX call via
    ``jax.lax.scan`` to avoid per-call host-memory leaks from the
    Warp FFI ``jax_callable`` bridge.  Ego overlay preparation
    (grayscale->RGB, float->uint8, 2x upscale) is vectorized across
    all frames before the per-frame rendering loop.

    If ``termination_events`` is provided (a list of ``(frame_index,
    reason_string)`` tuples from ``_run_eval_rollout``), frames at
    termination points receive a text overlay showing the termination
    reason followed by a logistic fade-out effect lasting
    ``termination_fade_seconds``.

    If ``hud_config`` is provided (a dict from ``render_config.hud``),
    a heads-up display is drawn in the bottom-left corner with per-frame
    metrics (speed, reward breakdown, cumulative reward, gap crossing
    indicator, torso height, heading, etc.).  ``reward_config`` supplies
    the reward term parameters (e.g. target_speed) for display.

    ``reward_remix`` is an optional ``{"sparse_key": str, "lambda": float}``.
    When given, the HUD's reward and cumulative-reward lines report the reward
    the replay buffer STORED, ``sparse + lambda*(total - sparse)``
    (rollout.py:200-207), with the raw env total shown alongside. The per-term
    breakdown stays raw -- lambda scales the whole dense remainder uniformly,
    so the single displayed lambda covers it.
    """
    import math

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING

    # Try common body names across walkers (rodent, sprout, stick, etc.)
    track_body_names = [
        "torso-rodent",
        "torso_link-sprout",
        "torso",
        "torso_link",
    ]
    for name in track_body_names:
        try:
            camera.trackbodyid = mj_model.body(name).id
            break
        except Exception:
            continue
    else:
        camera.trackbodyid = 1

    camera.distance = 1.0
    camera.azimuth = 90
    camera.elevation = -20
    camera.lookat[:] = [0, 0, 0.3]

    scene_option = mujoco.MjvOption()

    # -- HUD setup -------------------------------------------------------------
    hud_enabled = False
    if hud_config is not None and hud_config.get("enabled", True):
        hud_enabled = True

    _torso_id = None
    if hud_enabled:
        for name in track_body_names:
            try:
                _torso_id = mj_model.body(name).id
                break
            except Exception:
                continue

    # Speed target info: either a single target_speed (forward_velocity) or a
    # range [min_speed, max_speed] (forward_velocity_range).
    speed_target_type = None  # None | "single" | "range"
    speed_target_info = None
    if reward_config is not None:
        fv = reward_config.get("forward_velocity", {})
        if isinstance(fv, dict) and fv.get("target_speed") is not None:
            speed_target_type = "single"
            speed_target_info = (float(fv.get("target_speed")),)
        else:
            fvr = reward_config.get("forward_velocity_range", {})
            if isinstance(fvr, dict):
                min_s = fvr.get("min_speed")
                max_s = fvr.get("max_speed")
                if min_s is not None and max_s is not None:
                    speed_target_type = "range"
                    speed_target_info = (float(min_s), float(max_s))

    # -- Ego vision overlay ---------------------------------------------------
    ego_overlay_np = None
    # After jax.device_get(), Brax State preserves its pytree structure
    # (including .obs dict attribute), so hasattr/key checks work on CPU states.
    if use_obs_vision and hasattr(rollout[0], "obs") and "vision" in rollout[0].obs:
        # Use vision directly from rollout obs — shows what the policy
        # actually saw, including any eye ablations (left_only / right_only).
        # Vision shape per frame: (H, W, 2*C) for binocular grayscale → (H, W, 2)
        vision_stack = np.stack([np.asarray(s.obs["vision"]) for s in rollout])
        n_channels = vision_stack.shape[-1]
        mono_c = n_channels // 2  # e.g., 1 for grayscale binocular

        left_frames = vision_stack[..., :mono_c]   # (T, H, W, C)
        right_frames = vision_stack[..., mono_c:]   # (T, H, W, C)

        # Side-by-side with 2px white gap (same layout as re-render path)
        gap = np.ones(
            (len(rollout), vision_stack.shape[1], 2, mono_c), dtype=np.float32
        )
        ego_frames_np = np.concatenate([left_frames, gap, right_frames], axis=2)
        del vision_stack, left_frames, right_frames, gap

        ego_overlay_np = _prepare_ego_overlay(ego_frames_np)
        del ego_frames_np
        gc.collect()

    elif vision_renderer is not None:
        # Fallback: re-render from physics data (original path)
        all_data = jax.tree.map(
            lambda *xs: jax.numpy.stack(xs), *[s.data for s in rollout]
        )

        _render_all_ego = _make_render_all_fn(id(vision_renderer), vision_renderer)
        ego_imgs_jax = _render_all_ego(all_data)
        ego_frames_np = np.array(ego_imgs_jax)
        del ego_imgs_jax

        if right_vision_renderer is not None:
            _render_all_right = _make_render_all_fn(
                id(right_vision_renderer), right_vision_renderer
            )
            right_imgs_jax = _render_all_right(all_data)
            right_frames_np = np.array(right_imgs_jax)
            del right_imgs_jax
            gap = np.ones_like(ego_frames_np[:, :, :2, :])
            ego_frames_np = np.concatenate(
                [ego_frames_np, gap, right_frames_np], axis=2
            )
            del right_frames_np, gap

        del all_data
        gc.collect()

        ego_overlay_np = _prepare_ego_overlay(ego_frames_np)
        del ego_frames_np
        gc.collect()

    # -- Render main camera frames + composite overlay ------------------------
    with imageio.get_writer(video_path, fps=fps) as writer:
        termination_dict = {}
        if termination_events:
            termination_dict = {idx: reason for idx, reason in termination_events}
        termination_frame_set = set(termination_dict.keys())

        # HUD accumulators
        cumulative_reward = 0.0
        cumulative_reward_env = 0.0
        gap_crossed_persistent = False
        gap_flash_secs = (
            hud_config.get("gap_flash_duration", 1.5) if hud_config else 1.5
        )
        GAP_FLASH_DURATION = int(fps * gap_flash_secs)
        gap_crossed_display_frames = 0
        episode_step = 0

        # HUD toggle helpers
        def _hud_on(key):
            return hud_enabled and hud_config.get(key, True)

        # BGR color constants
        WHITE = (255, 255, 255)
        YELLOW = (0, 255, 255)
        CYAN = (255, 255, 0)
        GREEN = (0, 255, 0)
        BRIGHT_GREEN = (0, 255, 128)
        GRAY = (180, 180, 180)
        RED = (0, 0, 255)

        for i, state in enumerate(rollout):
            mj_data.qpos = np.array(state.data.qpos)
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data, camera, scene_option=scene_option)
            frame = renderer.render()

            # Overlay egocentric vision in upper-left corner
            if ego_overlay_np is not None:
                ego_scaled = ego_overlay_np[i]
                sh, sw = ego_scaled.shape[:2]
                pad = 2
                y0, x0 = pad + 4, pad + 4
                y1, x1 = y0 + sh, x0 + sw
                if y1 + pad < frame.shape[0] and x1 + pad < frame.shape[1]:
                    frame[y0 - pad : y1 + pad, x0 - pad : x1 + pad] = 255
                    frame[y0:y1, x0:x1] = ego_scaled

            # -- HUD overlay --------------------------------------------------
            if hud_enabled:
                # Reset accumulators on episode boundary
                if i > 0 and (i - 1) in termination_frame_set:
                    cumulative_reward = 0.0
                    cumulative_reward_env = 0.0
                    gap_crossed_persistent = False
                    gap_crossed_display_frames = 0
                    episode_step = 0

                # Extract kinematics from mj_data (already forwarded)
                forward_vel = None
                torso_z = None
                heading_deg = None
                lateral_y = None
                if _torso_id is not None:
                    forward_vel = float(np.asarray(state.data.subtree_linvel[_torso_id, 0]))
                    torso_z = float(mj_data.xpos[_torso_id, 2])
                    lateral_y = float(mj_data.xpos[_torso_id, 1])
                    hx = float(mj_data.xmat[_torso_id].reshape(3, 3)[0, 0])
                    hy = float(mj_data.xmat[_torso_id].reshape(3, 3)[0, 1])
                    heading_deg = math.degrees(math.atan2(hy, hx))

                # Read reward components from state.metrics
                gap_bonus = float(state.metrics.get("rewards/gap_crossing_bonus", 0.0))
                step_reward_env = float(state.reward)
                # The replay buffer stores sparse + lambda*(total - sparse)
                # (rollout.py:200-207), not the env total. Report what the
                # learner is actually paid; keep the env total alongside.
                if reward_remix is not None:
                    _sk = reward_remix.get("sparse_key")
                    _lam = float(reward_remix.get("lambda") or 0.0)
                    _sparse = float(
                        np.nan_to_num(np.asarray(state.metrics.get(_sk, 0.0)))
                    )
                    step_reward = _sparse + _lam * (step_reward_env - _sparse)
                else:
                    _lam = None
                    step_reward = step_reward_env
                cumulative_reward += step_reward
                cumulative_reward_env += step_reward_env
                episode_step += 1

                # Gap crossing persistence
                if gap_bonus > 0:
                    gap_crossed_persistent = True
                    gap_crossed_display_frames = GAP_FLASH_DURATION

                # Action magnitude
                action_rms = None
                if hasattr(state, "info") and "action" in state.info:
                    act = np.asarray(state.info["action"])
                    action_rms = float(np.sqrt(np.mean(act**2)))

                # Distance to next gap (from obs gap features if available)
                dist_to_gap = None
                if hasattr(state, "obs"):
                    obs = state.obs
                    # Navigate OrderedDict: state -> imitation_target
                    if isinstance(obs, dict) and "state" in obs:
                        inner = obs["state"]
                        if isinstance(inner, dict) and "imitation_target" in inner:
                            gap_feats = np.asarray(inner["imitation_target"])
                            if gap_feats.shape[-1] >= 1:
                                dist_to_gap = float(gap_feats[0])

                # Build HUD lines
                hud_lines = []

                if _hud_on("show_speed") and forward_vel is not None:
                    speed_text = f"Speed: {forward_vel:.2f} m/s"
                    if speed_target_type == "single":
                        target = speed_target_info[0]
                        speed_text += f" / {target:.1f} target"
                        pct = min(forward_vel / target, 1.0) if target > 0 else 0
                        speed_color = (
                            GREEN if pct > 0.8 else YELLOW if pct > 0.4 else WHITE
                        )
                    elif speed_target_type == "range":
                        min_s, max_s = speed_target_info
                        speed_text += f" / [{min_s:.1f}-{max_s:.1f}]"
                        if min_s <= forward_vel <= max_s:
                            speed_color = GREEN  # in valid range
                        elif min_s * 0.6 <= forward_vel <= max_s * 1.4:
                            speed_color = YELLOW  # near range
                        else:
                            speed_color = WHITE  # well outside
                    else:
                        speed_color = WHITE
                    hud_lines.append((speed_text, speed_color))

                if _hud_on("show_reward_breakdown"):
                    parts = []
                    for mk, mv in state.metrics.items():
                        if mk.startswith("rewards/"):
                            short = mk.split("/", 1)[1]
                            val = float(mv)
                            if val != 0 or short == "gap_crossing_bonus":
                                parts.append(f"{short}={val:.3f}")
                    _lam_txt = (
                        "" if _lam is None
                        else f" lam={_lam:.2f} env={step_reward_env:.3f}"
                    )
                    hud_lines.append(
                        (f"Reward: {step_reward:.3f}{_lam_txt}  ({', '.join(parts)})", CYAN)
                    )

                if _hud_on("show_cumulative_reward"):
                    _cum_txt = (
                        "" if _lam is None else f" (env {cumulative_reward_env:.1f})"
                    )
                    hud_lines.append(
                        (f"Cumulative: {cumulative_reward:.1f}{_cum_txt}", YELLOW)
                    )

                if _hud_on("show_gap_crossing"):
                    if gap_crossed_display_frames > 0:
                        hud_lines.append(("GAP CROSSED!", BRIGHT_GREEN))
                        gap_crossed_display_frames -= 1
                    elif gap_crossed_persistent:
                        gaps_count = int(state.info.get("gaps_crossed", 0))
                        hud_lines.append((f"Gaps crossed: {gaps_count}", GREEN))

                if _hud_on("show_distance_to_gap") and dist_to_gap is not None:
                    gap_color = (
                        RED
                        if dist_to_gap < 0.1
                        else YELLOW if dist_to_gap < 0.3 else GRAY
                    )
                    hud_lines.append((f"Dist to gap: {dist_to_gap:.3f} m", gap_color))

                if _hud_on("show_lateral_deviation") and lateral_y is not None:
                    lat_color = YELLOW if abs(lateral_y) > 0.3 else GRAY
                    hud_lines.append((f"Lateral: {lateral_y:.3f} m", lat_color))

                if _hud_on("show_height") and torso_z is not None:
                    h_color = RED if torso_z < 0.04 else GRAY
                    hud_lines.append((f"Height: {torso_z:.3f} m", h_color))

                if _hud_on("show_heading") and heading_deg is not None:
                    hd_color = YELLOW if abs(heading_deg) > 15 else GRAY
                    hud_lines.append((f"Heading: {heading_deg:.1f} deg", hd_color))

                if _hud_on("show_action_magnitude") and action_rms is not None:
                    hud_lines.append((f"Action RMS: {action_rms:.3f}", GRAY))

                # Eye angle display for actuable eyes
                if eye_qpos_indices is not None:
                    qpos = np.asarray(state.data.qpos)
                    eye_angles_rad = qpos[eye_qpos_indices]
                    l_deg = math.degrees(float(eye_angles_rad[0]))
                    r_deg = math.degrees(float(eye_angles_rad[1]))
                    # Color: brighter when eyes are moving away from center
                    max_angle = max(abs(l_deg), abs(r_deg))
                    eye_color = YELLOW if max_angle > 10 else GRAY
                    hud_lines.append(
                        (f"Eye L: {l_deg:+.1f} deg  R: {r_deg:+.1f} deg", eye_color)
                    )

                if _hud_on("show_step_counter"):
                    hud_lines.append((f"Step: {episode_step}", GRAY))

                # Draw HUD in bottom-left (avoids ego overlay in upper-left)
                if hud_lines:
                    hud_y_start = frame.shape[0] - len(hud_lines) * 22 - 10
                    _draw_hud(frame, hud_lines, x=10, y_start=hud_y_start)

            # Check if this frame is a termination event
            if termination_dict and i in termination_dict:
                reason = termination_dict[i]
                # Draw termination reason text overlay
                overlay_frame = frame.copy()
                label = f"Terminated: {reason}"
                cv2.putText(
                    overlay_frame,
                    label,
                    (10, frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.append_data(overlay_frame)
                # Fade-out frames (logistic curve, same as imitation.py)
                n_fade = int(fps * termination_fade_seconds)
                for t in range(n_fade):
                    rel_t = t / n_fade
                    fade_factor = 1 / (1 + np.exp(10 * (rel_t - 0.5)))
                    faded = (overlay_frame * fade_factor).astype(np.uint8)
                    writer.append_data(faded)
            else:
                writer.append_data(frame)


# ---------------------------------------------------------------------------
# Eval render config resolution
# ---------------------------------------------------------------------------


def _resolve_eval_render_config(cfg) -> dict:
    """Resolve eval render config from new or legacy config keys.

    Supports four cases:
    1. ``eval_render_config`` exists — use directly.
    2. Both ``render_config`` and ``eval_video`` exist — merge + warn.
    3. Only ``render_config`` exists — merge + warn, defaults for eval fields.
    4. Neither exists — return full defaults.

    Returns a plain dict (not OmegaConf) with structure::

        {
            "video": {"fps": 50, "height": 480, "width": 640, "hud": {...} | None},
            "eye_conditions": ["binocular"],
            "video_naconmax": 512,
        }
    """
    # Case 1: new key exists
    if cfg.get("eval_render_config") is not None:
        erc = OmegaConf.to_container(cfg.eval_render_config, resolve=True)
        erc.setdefault("video", {})
        erc["video"].setdefault("fps", 50)
        erc["video"].setdefault("height", 480)
        erc["video"].setdefault("width", 640)
        erc.setdefault("eye_conditions", ["binocular"])
        erc.setdefault("video_naconmax", 512)
        return erc

    # Cases 2 & 3: legacy keys
    has_render_config = cfg.get("render_config") is not None
    has_eval_video = cfg.get("eval_video") is not None

    if has_render_config or has_eval_video:
        logging.warning(
            "Config uses deprecated 'render_config' / 'eval_video' keys. "
            "Migrate to 'eval_render_config'. See design doc for new structure."
        )

    video = {}
    if has_render_config:
        rc = OmegaConf.to_container(cfg.render_config, resolve=True)
        video["fps"] = rc.get("render_fps", 50)
        video["height"] = rc.get("render_height", 480)
        video["width"] = rc.get("render_width", 640)
        if rc.get("hud") is not None:
            video["hud"] = rc["hud"]
    else:
        video = {"fps": 50, "height": 480, "width": 640}

    eye_conditions = ["binocular"]
    video_naconmax = 512
    if has_eval_video:
        ev = OmegaConf.to_container(cfg.eval_video, resolve=True)
        eye_conditions = ev.get("eye_conditions", ["binocular"])
        video_naconmax = ev.get("video_naconmax", 512)

    return {
        "video": video,
        "eye_conditions": list(eye_conditions),
        "video_naconmax": video_naconmax,
    }


# ---------------------------------------------------------------------------
# MLP mode: Brax PPO with flat obs
# ---------------------------------------------------------------------------


def _train_mlp_highlvl(
    cfg,
    env,
    eval_env,
    decoder_policy_fn,
    mimic_cfg,
    checkpoint_path,
    cfg_dict,
    progress_fn,
    prior_fn=None,
):
    """Train high-level MLP policy using standard Brax PPO.

    The HighLevelWrapper with pass_vision=False produces flat observations
    (state/privileged_state), suitable for a standard MLP policy.
    """
    eval_render_cfg = _resolve_eval_render_config(cfg)

    latent_size = mimic_cfg.network_config.intention_size
    highlvl_obs_key = cfg.transfer.get("highlvl_obs_key", "imitation_target")
    decoder_obs_key = cfg.transfer.get("decoder_obs_key", "proprioception")

    # Read n_eye_actuators from the base (unwrapped) env.
    _unwrapped = env
    while hasattr(_unwrapped, "env"):
        _unwrapped = _unwrapped.env
    n_eye_actuators = getattr(_unwrapped, "n_eye_actuators", 0)
    if n_eye_actuators > 0:
        logging.info(f"Actuable eyes: {n_eye_actuators} eye actuators bypass decoder")

    if prior_fn is not None:
        env = PriorHighLevelWrapper(
            env,
            prior_fn,
            decoder_policy_fn,
            latent_size,
            highlvl_obs_key=highlvl_obs_key,
            decoder_obs_key=decoder_obs_key,
            pass_vision=False,
            deterministic_prior=cfg.transfer.get("deterministic_prior", True),
            noise_logvar=cfg.transfer.get("noise_logvar", -2.0),
            n_eye_actuators=n_eye_actuators,
        )
        eval_env = PriorHighLevelWrapper(
            eval_env,
            prior_fn,
            decoder_policy_fn,
            latent_size,
            highlvl_obs_key=highlvl_obs_key,
            decoder_obs_key=decoder_obs_key,
            pass_vision=False,
            deterministic_prior=cfg.transfer.get("deterministic_prior", True),
            noise_logvar=cfg.transfer.get("noise_logvar", -2.0),
            n_eye_actuators=n_eye_actuators,
        )
    else:
        env = HighLevelWrapper(
            env,
            decoder_policy_fn,
            latent_size,
            highlvl_obs_key=highlvl_obs_key,
            decoder_obs_key=decoder_obs_key,
            pass_vision=False,
            n_eye_actuators=n_eye_actuators,
        )
        eval_env = HighLevelWrapper(
            eval_env,
            decoder_policy_fn,
            latent_size,
            highlvl_obs_key=highlvl_obs_key,
            decoder_obs_key=decoder_obs_key,
            pass_vision=False,
            n_eye_actuators=n_eye_actuators,
        )

    # Brax PPO expects flat array obs and int observation_size.
    # HighLevelWrapper MLP mode returns dict {"state": flat, "privileged_state": flat}.
    # Adapt to flat by extracting obs["state"].
    env = _FlatObsAdapter(env, obs_key="state")
    eval_env = _FlatObsAdapter(eval_env, obs_key="state")

    logging.info(f"MLP HighLevelWrapper: action_size={env.action_size}")
    _log_memory("after MLP HighLevelWrapper")

    # Get observation size from a sample state (now flat arrays)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    rng = jax.random.PRNGKey(0)
    start_state = jit_reset(rng)
    observation_size = start_state.obs.shape[-1]
    logging.info(f"MLP observation_size (from start_state): {observation_size}")

    # PPO training params
    ppo_params = dict(
        OmegaConf.to_container(cfg.train_setup.train_config, resolve=True)
    )
    # Remove ff_ppo-specific keys that Brax PPO does not accept
    for key in [
        "latent_kl_weight",
        "latent_ar1_weight",
        "use_kl_schedule",
        "grad_clip_threshold",
    ]:
        ppo_params.pop(key, None)

    # Network factory: standard MLP
    network_factory = functools.partial(
        brax_ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tuple(cfg.network_config.policy_hidden_layer_sizes),
        value_hidden_layer_sizes=tuple(cfg.network_config.value_hidden_layer_sizes),
    )

    # Normalizer for logging inference fn
    normalize = lambda x, y: x
    if ppo_params.get("normalize_observations", False):
        normalize = running_statistics.normalize

    # Build ppo_network and logging inference fn BEFORE calling train
    # (following the bowl_escape_highlvl.py pattern)
    ppo_network = network_factory(
        observation_size,
        env.action_size,
        preprocess_observations_fn=normalize,
    )

    def make_logging_inference_fn(ppo_networks):
        """Creates params and inference function for the PPO agent."""

        def make_logging_policy(deterministic=False):
            policy_network = ppo_networks.policy_network
            parametric_action_distribution = ppo_networks.parametric_action_distribution

            def logging_policy(params, observations, key_sample):
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
                postprocessed_actions = parametric_action_distribution.postprocess(
                    raw_actions
                )
                return jp.array(postprocessed_actions), {
                    "log_prob": log_prob,
                    "raw_action": raw_actions,
                }

            return logging_policy

        return make_logging_policy

    make_logging_policy = make_logging_inference_fn(ppo_network)
    jit_logging_inference_fn = jax.jit(make_logging_policy(deterministic=True))

    # MuJoCo renderer for eval videos
    mj_model = eval_env.mj_model
    mj_data = mujoco.MjData(mj_model)
    renderer_obj = mujoco.Renderer(
        mj_model,
        height=eval_render_cfg["video"]["height"],
        width=eval_render_cfg["video"]["width"],
    )

    # Save reference before it gets shadowed by the functools.partial closure
    _render_video_fn = render_video

    # Orbax checkpointer for saving inside policy_params_fn
    orbax_checkpointer = ocp.PyTreeCheckpointer()

    episode_length = cfg.train_setup.train_config.episode_length

    def mlp_policy_params_fn(
        current_step, make_policy, params, jit_logging_inference_fn
    ):
        """Callback for Brax PPO: render video and save checkpoint."""
        del make_policy  # Unused; we use our own jit_logging_inference_fn

        _log_memory(f"mlp_policy_params_fn entry step={current_step}")

        # Run an evaluation rollout
        eval_rng = jax.random.PRNGKey(current_step)
        rollout, termination_events = _run_eval_rollout(
            jit_reset,
            jit_step,
            jit_logging_inference_fn,
            params,
            episode_length,
            eval_rng,
        )

        # Log per-step reward metrics
        for metric_name in [
            k for k in rollout[0].metrics.keys() if k.startswith("rewards/")
        ]:
            values = [float(s.metrics[metric_name]) for s in rollout]
            table = wandb.Table(
                data=[[i, v] for i, v in enumerate(values)],
                columns=["frame", metric_name],
            )
            wandb.log(
                {
                    f"eval/rollout_{metric_name}": wandb.plot.line(
                        table, "frame", metric_name, title=metric_name
                    )
                },
                step=current_step,
                commit=False,
            )

        # Render video
        video_path = str(checkpoint_path / f"{current_step}.mp4")
        try:
            _render_video_fn(
                rollout,
                mj_model,
                mj_data,
                renderer_obj,
                video_path,
                fps=eval_render_cfg["video"]["fps"],
                termination_events=termination_events,
                hud_config=eval_render_cfg["video"].get("hud"),
                reward_config=(
                    OmegaConf.to_container(
                        cfg.env_config.env_args.get("reward_terms", {}), resolve=True
                    )
                    if cfg.env_config.get("env_args")
                    and cfg.env_config.env_args.get("reward_terms")
                    else None
                ),
            )
            wandb.log(
                {"videos/rollout": wandb.Video(video_path, format="mp4")},
                step=current_step,
                commit=False,
            )
        except mujoco.FatalError as e:
            logging.warning(f"Video rendering failed: {e}")

        # Save checkpoint via orbax
        try:
            from flax.training import orbax_utils

            save_args = orbax_utils.save_args_from_target(params)
            ckpt_step_path = checkpoint_path / f"{current_step}"
            orbax_checkpointer.save(
                str(ckpt_step_path), params, force=True, save_args=save_args
            )
            logging.info(f"Saved MLP checkpoint at step {current_step}")
        except Exception as e:
            logging.warning(f"Checkpoint save failed: {e}")

        _log_memory(f"mlp_policy_params_fn before cleanup step={current_step}")

        del rollout
        gc.collect()

    # Compute num_evals
    eval_every = cfg.train_setup.get("eval_every", 5_000_000)
    num_evals = max(1, int(ppo_params["num_timesteps"] / eval_every))

    # Build and run Brax PPO train
    train_fn = functools.partial(
        brax_ppo_train.train,
        **ppo_params,
        num_evals=num_evals,
        network_factory=network_factory,
        restore_checkpoint_path=None,
        progress_fn=progress_fn,
        # Honours the top-level `wrappers:` block (full_reset / info reset);
        # with no such block this is exactly mp_wrapper.wrap_for_brax_training.
        wrap_env_fn=functools.partial(_wrap_for_brax_training, cfg=cfg),
        policy_params_fn=functools.partial(
            mlp_policy_params_fn, jit_logging_inference_fn=jit_logging_inference_fn
        ),
    )

    logging.info("Starting MLP high-level PPO training (Brax PPO)...")
    make_inference_fn, params, metrics = train_fn(
        environment=env,
        eval_env=eval_env,
    )
    return make_inference_fn, params, metrics


# ---------------------------------------------------------------------------
# Vision + TaskObs mode: ff_ppo with CNN + body signals
# ---------------------------------------------------------------------------


def _train_vision_task_obs_highlvl(
    cfg,
    env,
    eval_env,
    decoder_policy_fn,
    mimic_cfg,
    checkpoint_path,
    cfg_dict,
    progress_fn,
    prior_fn=None,
    checkpoint_callback=None,
):
    """Train high-level vision+task_obs policy using ff_ppo with CNN encoder.

    The HighLevelWrapper passes both vision and task_obs (body signals),
    and uses a fusion network that combines CNN features with the task
    observation vector.
    """
    eval_render_cfg = _resolve_eval_render_config(cfg)

    latent_size = mimic_cfg.network_config.intention_size
    highlvl_obs_key = cfg.transfer.get("highlvl_obs_key", "imitation_target")
    decoder_obs_key = cfg.transfer.get("decoder_obs_key", "proprioception")

    # Read n_eye_actuators from the base (unwrapped) env.
    _unwrapped = env
    while hasattr(_unwrapped, "env"):
        _unwrapped = _unwrapped.env
    n_eye_actuators = getattr(_unwrapped, "n_eye_actuators", 0)
    if n_eye_actuators > 0:
        logging.info(f"Actuable eyes: {n_eye_actuators} eye actuators bypass decoder")

    if prior_fn is not None:
        env = PriorHighLevelWrapper(
            env,
            prior_fn,
            decoder_policy_fn,
            latent_size,
            highlvl_obs_key=highlvl_obs_key,
            decoder_obs_key=decoder_obs_key,
            pass_vision=True,
            pass_task_obs=True,
            deterministic_prior=cfg.transfer.get("deterministic_prior", True),
            noise_logvar=cfg.transfer.get("noise_logvar", -2.0),
            n_eye_actuators=n_eye_actuators,
        )
        eval_env = PriorHighLevelWrapper(
            eval_env,
            prior_fn,
            decoder_policy_fn,
            latent_size,
            highlvl_obs_key=highlvl_obs_key,
            decoder_obs_key=decoder_obs_key,
            pass_vision=True,
            pass_task_obs=True,
            deterministic_prior=cfg.transfer.get("deterministic_prior", True),
            noise_logvar=cfg.transfer.get("noise_logvar", -2.0),
            n_eye_actuators=n_eye_actuators,
        )
    else:
        env = HighLevelWrapper(
            env,
            decoder_policy_fn,
            latent_size,
            highlvl_obs_key=highlvl_obs_key,
            decoder_obs_key=decoder_obs_key,
            pass_vision=True,
            pass_task_obs=True,
            n_eye_actuators=n_eye_actuators,
        )
        eval_env = HighLevelWrapper(
            eval_env,
            decoder_policy_fn,
            latent_size,
            highlvl_obs_key=highlvl_obs_key,
            decoder_obs_key=decoder_obs_key,
            pass_vision=True,
            pass_task_obs=True,
            n_eye_actuators=n_eye_actuators,
        )

    logging.info(f"Vision+TaskObs HighLevelWrapper: action_size={env.action_size}")
    _log_memory("after Vision+TaskObs HighLevelWrapper")

    # Detect vision shape from environment
    unwrapped = env.env if hasattr(env, "env") else env
    vision_shape = (
        unwrapped.vision_shape
        if hasattr(unwrapped, "vision_shape")
        else (
            cfg.env_config.get("vision_height", 32),
            cfg.env_config.get("vision_width", 32),
            1 if cfg.env_config.get("grayscale", True) else 3,
        )
    )
    logging.info(f"Vision shape: {vision_shape}")

    # PPO training params
    ppo_params = dict(
        OmegaConf.to_container(cfg.train_setup.train_config, resolve=True)
    )
    ppo_params.pop("eval_naconmax", None)

    # Network factory: vision CNN + task_obs fusion + MLP
    network_factory = functools.partial(
        ff_ppo_networks.make_vision_task_obs_highlvl_ppo_networks,
        vision_shape=tuple(vision_shape),
        vision_latent_size=cfg.network_config.vision_latent_size,
        vision_feature_size=cfg.network_config.get("vision_feature_size", 128),
        decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_hidden_layer_sizes),
        value_hidden_layer_sizes=tuple(cfg.network_config.value_hidden_layer_sizes),
        vision_channels=tuple(cfg.network_config.vision_channels),
        fusion_hidden_layer_sizes=tuple(
            cfg.network_config.get("fusion_hidden_layer_sizes", [256])
        ),
    )

    # Create orbax CheckpointManager for ff_ppo
    ckpt_mgr_options = ocp.CheckpointManagerOptions(
        save_interval_steps=1,
        step_prefix="PPONetwork",
        create=True,
    )
    ckpt_mgr = ocp.CheckpointManager(str(checkpoint_path), options=ckpt_mgr_options)

    # Eval rendering setup
    mj_model = eval_env.mj_model
    mj_data = mujoco.MjData(mj_model)
    renderer_obj = mujoco.Renderer(
        mj_model,
        height=eval_render_cfg["video"]["height"],
        width=eval_render_cfg["video"]["width"],
    )
    # Create warp vision renderer (nworld=1) for egocentric overlay in videos
    _video_vision_renderer = None
    from vnl_playground.tasks.rodent.vision_jax import (
        JaxVisionRenderer,
        VisionRenderWrapper,
    )

    _unwrapped = env.env if hasattr(env, "env") else env
    # Walk up the wrapper chain to find the raw env with mj_model
    while hasattr(_unwrapped, "env"):
        _unwrapped = _unwrapped.env

    _video_vision_renderer = JaxVisionRenderer(
        mj_model=_unwrapped.mj_model,
        mjx_model=_unwrapped.mjx_model,
        nworld=1,
        width=cfg.env_config.get("vision_width", 32),
        height=cfg.env_config.get("vision_height", 32),
        grayscale=cfg.env_config.get("grayscale", True),
        camera_name=cfg.env_config.get("vision_camera_name", "egocentric-rodent"),
        render_depth=cfg.env_config.get("render_depth", False),
        use_textures=cfg.env_config.get("use_textures", False),
        use_shadows=cfg.env_config.get("use_shadows", False),
    )
    logging.info("Created warp vision renderer (nworld=1) for video overlay")

    # Eval callback closures with vision rendering (uses _video_vision_renderer)
    _eval_base_reset = eval_env.reset
    _eval_base_step = eval_env.step

    def _eval_reset_with_vision(rng):
        state = _eval_base_reset(rng)
        data_b = _add_batch_dim_for_warp(state.data)
        vision = _video_vision_renderer.render(data_b)[0]
        return state.replace(obs=VisionRenderWrapper._inject_vision(state.obs, vision))

    def _eval_step_with_vision(state, action):
        state = _eval_base_step(state, action)
        data_b = _add_batch_dim_for_warp(state.data)
        vision = _video_vision_renderer.render(data_b)[0]
        return state.replace(obs=VisionRenderWrapper._inject_vision(state.obs, vision))

    jit_reset = jax.jit(_eval_reset_with_vision)
    jit_step = jax.jit(_eval_step_with_vision)

    # Ensure render_config has render_interval for ff_ppo
    if "render_interval" not in cfg_dict.get("render_config", {}):
        cfg_dict.setdefault("render_config", {})["render_interval"] = 1

    # Update config_dict with network_config fields that ff_ppo expects
    cfg_dict["network_config"].update(
        {
            "arch_name": "vision_task_obs",
            "vision_latent_size": cfg.network_config.vision_latent_size,
            "vision_feature_size": cfg.network_config.get("vision_feature_size", 128),
            "decoder_layer_sizes": list(cfg.network_config.decoder_hidden_layer_sizes),
            "critic_layer_sizes": list(cfg.network_config.value_hidden_layer_sizes),
            "fusion_hidden_layer_sizes": list(
                cfg.network_config.get("fusion_hidden_layer_sizes", [256])
            ),
        }
    )

    # Save reference before it gets shadowed by the bool parameter in the callback
    _render_video_fn = render_video

    episode_length = cfg.train_setup.train_config.episode_length

    def vision_task_obs_policy_params_fn(
        current_step,
        jit_logging_inference_fn,
        params,
        policy_params_fn_key,
        render_video,  # noqa: N803 -- bool flag set by ff_ppo caller
        ppo_network,
    ):
        """Callback for ff_ppo: render video with egocentric overlay."""
        if not render_video:
            return

        _log_memory(f"vision_task_obs_policy_params_fn entry step={current_step}")

        # Run an evaluation rollout
        rollout, termination_events = _run_eval_rollout(
            jit_reset,
            jit_step,
            jit_logging_inference_fn,
            params,
            episode_length,
            policy_params_fn_key,
        )

        # Vision sensitivity diagnostic: compare actions with real vs blank vision
        mid = len(rollout) // 2
        obs_with_vision = rollout[mid].obs
        obs_blank_vision = {
            k: (jp.zeros_like(v) if k == "vision" else v)
            for k, v in obs_with_vision.items()
        }
        _, sensitivity_rng = jax.random.split(policy_params_fn_key)
        act_real, _ = jit_logging_inference_fn(params, obs_with_vision, sensitivity_rng)
        act_blank, _ = jit_logging_inference_fn(
            params, obs_blank_vision, sensitivity_rng
        )
        vision_sensitivity = float(jp.linalg.norm(act_real - act_blank))
        wandb.log(
            {"eval/vision_sensitivity": vision_sensitivity},
            step=current_step,
            commit=False,
        )

        # Log per-step reward metrics
        for metric_name in [
            k for k in rollout[0].metrics.keys() if k.startswith("rewards/")
        ]:
            values = [float(s.metrics[metric_name]) for s in rollout]
            table = wandb.Table(
                data=[[i, v] for i, v in enumerate(values)],
                columns=["frame", metric_name],
            )
            wandb.log(
                {
                    f"eval/rollout_{metric_name}": wandb.plot.line(
                        table, "frame", metric_name, title=metric_name
                    )
                },
                step=current_step,
                commit=False,
            )

        # Render video
        video_path = str(checkpoint_path / f"{current_step}.mp4")
        try:
            _render_video_fn(
                rollout,
                mj_model,
                mj_data,
                renderer_obj,
                video_path,
                fps=eval_render_cfg["video"]["fps"],
                vision_renderer=_video_vision_renderer,
                termination_events=termination_events,
                hud_config=eval_render_cfg["video"].get("hud"),
                reward_config=(
                    OmegaConf.to_container(
                        cfg.env_config.env_args.get("reward_terms", {}), resolve=True
                    )
                    if cfg.env_config.get("env_args")
                    and cfg.env_config.env_args.get("reward_terms")
                    else None
                ),
            )
            wandb.log(
                {"videos/rollout": wandb.Video(video_path, format="mp4")},
                step=current_step,
                commit=False,
            )
        except mujoco.FatalError as e:
            logging.warning(f"Video rendering failed: {e}")

        _log_memory(
            f"vision_task_obs_policy_params_fn before cleanup step={current_step}"
        )

        del rollout
        gc.collect()

    # Compute num_evals
    eval_every = cfg.train_setup.get("eval_every", 10_000_000)
    num_evals = max(1, int(ppo_params["num_timesteps"] / eval_every))

    # Checkpoint to restore (if any) — auto-set when resuming
    checkpoint_to_restore = cfg.train_setup.get("checkpoint_to_restore", None)
    if checkpoint_to_restore is None and cfg.train_setup.get("resume_run_id", None):
        checkpoint_to_restore = str(checkpoint_path)
        logging.info(f"Auto-setting checkpoint_to_restore={checkpoint_to_restore}")

    # Vision rendering wrapper for training environments
    unwrapped_env = env.env if hasattr(env, "env") else env
    # Walk up to find the raw env for mj_model
    _raw_env = unwrapped_env
    while hasattr(_raw_env, "env"):
        _raw_env = _raw_env.env

    vision_width = cfg.env_config.get("vision_width", 32)
    vision_height = cfg.env_config.get("vision_height", 32)
    grayscale = cfg.env_config.get("grayscale", True)
    camera_name = cfg.env_config.get("vision_camera_name", "egocentric-rodent")
    render_depth = cfg.env_config.get("render_depth", False)
    use_textures = cfg.env_config.get("use_textures", False)
    use_shadows = cfg.env_config.get("use_shadows", False)

    def wrap_with_vision(
        environment,
        episode_length: int = 1000,
        action_repeat: int = 1,
        randomization_fn=None,
    ):
        """Wrap env for brax training, then add vision rendering."""
        brax_env = _wrap_for_brax_training(
            environment,
            cfg,
            episode_length=episode_length,
            action_repeat=action_repeat,
            randomization_fn=randomization_fn,
        )
        return VisionRenderWrapper(
            brax_env,
            mj_model=_raw_env.mj_model,
            mjx_model=_raw_env.mjx_model,
            width=vision_width,
            height=vision_height,
            grayscale=grayscale,
            camera_name=camera_name,
            render_depth=render_depth,
            use_textures=use_textures,
            use_shadows=use_shadows,
        )

    logging.info(
        f"Vision+TaskObs rendering: {vision_width}x{vision_height}, "
        f"grayscale={grayscale}, camera={camera_name}, "
        f"JAX-callable (inside lax.scan)"
    )

    # Build and run ff_ppo train
    train_fn = functools.partial(
        ff_ppo_train.train,
        **ppo_params,
        num_evals=num_evals,
        ckpt_mgr=ckpt_mgr,
        config_dict=cfg_dict,
        checkpoint_to_restore=checkpoint_to_restore,
        network_factory=network_factory,
        progress_fn=progress_fn,
        policy_params_fn=vision_task_obs_policy_params_fn,
        wrap_for_training=wrap_with_vision,
        checkpoint_callback=checkpoint_callback,
    )

    logging.info("Starting vision+task_obs high-level PPO training (ff_ppo)...")
    make_policy, params, metrics = train_fn(
        environment=env,
        eval_env=eval_env,
    )
    return make_policy, params, metrics


# ---------------------------------------------------------------------------
# Shared-CNN Vision + TaskObs mode: single CNN for both policy and value
# ---------------------------------------------------------------------------


def _train_shared_vision_task_obs_highlvl(
    cfg,
    env,
    eval_env,
    decoder_policy_fn,
    mimic_cfg,
    checkpoint_path,
    cfg_dict,
    progress_fn,
    prior_fn=None,
    checkpoint_callback=None,
):
    """Train high-level vision+task_obs policy with a SHARED CNN.

    Mirrors the vnl-ray architecture: a single VisionEncoder is shared
    between the policy and value heads.  Both policy_loss and v_loss
    gradients flow through the CNN, providing a much stronger learning
    signal for vision features.
    """
    from track_mjx.agent.ff_ppo import losses as ff_ppo_losses

    eval_render_cfg = _resolve_eval_render_config(cfg)

    latent_size = mimic_cfg.network_config.intention_size
    highlvl_obs_key = cfg.transfer.get("highlvl_obs_key", "imitation_target")
    decoder_obs_key = cfg.transfer.get("decoder_obs_key", "proprioception")

    # Read n_eye_actuators from the base (unwrapped) env.
    _unwrapped = env
    while hasattr(_unwrapped, "env"):
        _unwrapped = _unwrapped.env
    n_eye_actuators = getattr(_unwrapped, "n_eye_actuators", 0)
    if n_eye_actuators > 0:
        logging.info(f"Actuable eyes: {n_eye_actuators} eye actuators bypass decoder")

    if prior_fn is not None:
        env = PriorHighLevelWrapper(
            env,
            prior_fn,
            decoder_policy_fn,
            latent_size,
            highlvl_obs_key=highlvl_obs_key,
            decoder_obs_key=decoder_obs_key,
            pass_vision=True,
            pass_task_obs=True,
            deterministic_prior=cfg.transfer.get("deterministic_prior", True),
            noise_logvar=cfg.transfer.get("noise_logvar", -2.0),
            n_eye_actuators=n_eye_actuators,
        )
        eval_env = PriorHighLevelWrapper(
            eval_env,
            prior_fn,
            decoder_policy_fn,
            latent_size,
            highlvl_obs_key=highlvl_obs_key,
            decoder_obs_key=decoder_obs_key,
            pass_vision=True,
            pass_task_obs=True,
            deterministic_prior=cfg.transfer.get("deterministic_prior", True),
            noise_logvar=cfg.transfer.get("noise_logvar", -2.0),
            n_eye_actuators=n_eye_actuators,
        )
    else:
        env = HighLevelWrapper(
            env,
            decoder_policy_fn,
            latent_size,
            highlvl_obs_key=highlvl_obs_key,
            decoder_obs_key=decoder_obs_key,
            pass_vision=True,
            pass_task_obs=True,
            n_eye_actuators=n_eye_actuators,
        )
        eval_env = HighLevelWrapper(
            eval_env,
            decoder_policy_fn,
            latent_size,
            highlvl_obs_key=highlvl_obs_key,
            decoder_obs_key=decoder_obs_key,
            pass_vision=True,
            pass_task_obs=True,
            n_eye_actuators=n_eye_actuators,
        )

    logging.info(
        f"Shared-CNN Vision+TaskObs HighLevelWrapper: action_size={env.action_size}"
    )
    _log_memory("after Shared-CNN HighLevelWrapper")

    # Set Warp's CUDA memory pool release threshold to 512 MB.
    # Memory above this threshold is returned to CUDA on wp.synchronize().
    try:
        cuda_device = wp.get_device("cuda:0")
        wp.set_mempool_release_threshold(cuda_device, 512 * 1024 * 1024)
        logging.info("[MEM] Set Warp mempool release threshold to 512 MB")
    except Exception as e:
        logging.warning(f"Could not set Warp mempool release threshold: {e}")

    # Detect vision shape from environment
    unwrapped = env.env if hasattr(env, "env") else env
    vision_shape = (
        unwrapped.vision_shape
        if hasattr(unwrapped, "vision_shape")
        else (
            cfg.env_config.get("vision_height", 32),
            cfg.env_config.get("vision_width", 32),
            1 if cfg.env_config.get("grayscale", True) else 3,
        )
    )
    logging.info(f"Vision shape: {vision_shape}")

    # PPO training params
    ppo_params = dict(
        OmegaConf.to_container(cfg.train_setup.train_config, resolve=True)
    )
    ppo_params.pop("eval_naconmax", None)

    # Network factory: shared-CNN vision + task_obs
    ppo_network, shared_module = (
        ff_ppo_networks.make_shared_vision_task_obs_highlvl_ppo_networks(
            obs_sizes=env.observation_size,
            action_size=env.action_size,
            vision_shape=tuple(vision_shape),
            vision_latent_size=cfg.network_config.vision_latent_size,
            vision_feature_size=cfg.network_config.get("vision_feature_size", 32),
            decoder_hidden_layer_sizes=tuple(
                cfg.network_config.decoder_hidden_layer_sizes
            ),
            value_hidden_layer_sizes=tuple(cfg.network_config.value_hidden_layer_sizes),
            vision_channels=tuple(cfg.network_config.vision_channels),
            fusion_hidden_layer_sizes=tuple(
                cfg.network_config.get("fusion_hidden_layer_sizes", [256])
            ),
        )
    )

    # Create the shared loss function (pre-baked with shared_module)
    eval_every = cfg.train_setup.get("eval_every", 10_000_000)
    num_evals = max(1, int(ppo_params["num_timesteps"] / eval_every))

    latent_kl_schedule = None
    latent_ar1_schedule = None
    if ppo_params.get("use_kl_schedule", False):
        latent_kl_schedule = ff_ppo_losses.create_ramp_schedule(
            max_value=ppo_params.get("latent_kl_weight", 0.0),
            ramp_steps=int(num_evals * ppo_params.get("kl_ramp_up_frac", 0.25)),
            schedule="linear",
        )
        latent_ar1_schedule = ff_ppo_losses.create_ramp_schedule(
            max_value=ppo_params.get("latent_ar1_weight", 0.0),
            ramp_steps=int(num_evals * ppo_params.get("kl_ramp_up_frac", 0.25)),
            schedule="linear",
        )

    prior_residual_schedule = None
    prior_residual_start = ppo_params.pop("prior_residual_start_weight", 0.0)
    prior_residual_end = ppo_params.pop("prior_residual_end_weight", 0.0)
    prior_residual_decay_frac = ppo_params.pop("prior_residual_decay_frac", 0.5)
    prior_residual_warmup_frac = ppo_params.pop("prior_residual_warmup_frac", 0.0)
    prior_residual_schedule_type = ppo_params.pop("prior_residual_schedule_type", "linear")
    if prior_residual_start > 0.0:
        prior_residual_schedule = ff_ppo_losses.create_ramp_schedule(
            # NOTE: min_value is the START value, max_value is the END value.
            # When min_value > max_value, create_ramp_schedule naturally produces
            # a decay: value = start + progress * (end - start) = start - progress * delta
            min_value=prior_residual_start,
            max_value=prior_residual_end,
            ramp_steps=int(num_evals * prior_residual_decay_frac),
            warmup_steps=int(num_evals * prior_residual_warmup_frac),
            schedule=prior_residual_schedule_type,
        )
        logging.info(
            f"Prior residual penalty: start={prior_residual_start}, "
            f"end={prior_residual_end}, decay over {prior_residual_decay_frac*100:.0f}% of training"
        )

    custom_loss_fn = functools.partial(
        ff_ppo_losses.compute_shared_vision_ppo_loss,
        ppo_network=ppo_network,
        shared_module=shared_module,
        entropy_cost=ppo_params.get("entropy_cost", 1e-3),
        latent_kl_weight=ppo_params.get("latent_kl_weight", 0.0),
        latent_ar1_weight=ppo_params.get("latent_ar1_weight", 0.0),
        discounting=ppo_params.get("discounting", 0.97),
        reward_scaling=ppo_params.get("reward_scaling", 1.0),
        gae_lambda=ppo_params.get("gae_lambda", 0.95),
        clipping_epsilon=ppo_params.get("clipping_epsilon", 0.2),
        normalize_advantage=ppo_params.get("normalize_advantage", True),
        vf_coefficient=ppo_params.get("vf_loss_coefficient", 0.5),
        latent_kl_schedule=latent_kl_schedule,
        latent_ar1_schedule=latent_ar1_schedule,
        prior_residual_weight=prior_residual_start,
        prior_residual_schedule=prior_residual_schedule,
    )

    # Wrap network_factory to return the pre-built ppo_network
    def network_factory(obs_sizes, action_size):
        return ppo_network

    # Create orbax CheckpointManager
    ckpt_mgr_options = ocp.CheckpointManagerOptions(
        save_interval_steps=1,
        max_to_keep=50,
        step_prefix="PPONetwork",
        create=True,
    )
    ckpt_mgr = ocp.CheckpointManager(str(checkpoint_path), options=ckpt_mgr_options)

    # Eval rendering setup
    mj_model = eval_env.mj_model
    mj_data = mujoco.MjData(mj_model)
    renderer_obj = mujoco.Renderer(
        mj_model,
        height=eval_render_cfg["video"]["height"],
        width=eval_render_cfg["video"]["width"],
    )
    # Create warp vision renderer (nworld=1) for egocentric overlay
    _video_vision_renderer = None
    from vnl_playground.tasks.rodent.vision_jax import (
        JaxVisionRenderer,
        VisionRenderWrapper,
    )

    _unwrapped = env.env if hasattr(env, "env") else env
    while hasattr(_unwrapped, "env"):
        _unwrapped = _unwrapped.env

    _video_vision_renderer = JaxVisionRenderer(
        mj_model=_unwrapped.mj_model,
        mjx_model=_unwrapped.mjx_model,
        nworld=1,
        width=cfg.env_config.get("vision_width", 32),
        height=cfg.env_config.get("vision_height", 32),
        grayscale=cfg.env_config.get("grayscale", True),
        camera_name=cfg.env_config.get("vision_camera_name", "egocentric-rodent"),
        render_depth=cfg.env_config.get("render_depth", False),
        use_textures=cfg.env_config.get("use_textures", False),
        use_shadows=cfg.env_config.get("use_shadows", False),
    )
    logging.info("Created warp vision renderer (nworld=1) for video overlay")

    # Eval callback closures with vision rendering (uses _video_vision_renderer)
    _eval_base_reset = eval_env.reset
    _eval_base_step = eval_env.step

    def _eval_reset_with_vision(rng):
        state = _eval_base_reset(rng)
        data_b = _add_batch_dim_for_warp(state.data)
        vision = _video_vision_renderer.render(data_b)[0]
        return state.replace(obs=VisionRenderWrapper._inject_vision(state.obs, vision))

    def _eval_step_with_vision(state, action):
        state = _eval_base_step(state, action)
        data_b = _add_batch_dim_for_warp(state.data)
        vision = _video_vision_renderer.render(data_b)[0]
        return state.replace(obs=VisionRenderWrapper._inject_vision(state.obs, vision))

    jit_reset = jax.jit(_eval_reset_with_vision)
    jit_step = jax.jit(_eval_step_with_vision)

    # Ensure render_config has render_interval
    if "render_interval" not in cfg_dict.get("render_config", {}):
        cfg_dict.setdefault("render_config", {})["render_interval"] = 1

    # Update config_dict
    cfg_dict["network_config"].update(
        {
            "arch_name": "shared_vision_task_obs",
            "vision_latent_size": cfg.network_config.vision_latent_size,
            "vision_feature_size": cfg.network_config.get("vision_feature_size", 32),
            "decoder_layer_sizes": list(cfg.network_config.decoder_hidden_layer_sizes),
            "critic_layer_sizes": list(cfg.network_config.value_hidden_layer_sizes),
            "fusion_hidden_layer_sizes": list(
                cfg.network_config.get("fusion_hidden_layer_sizes", [256])
            ),
        }
    )

    _render_video_fn = render_video

    episode_length = cfg.train_setup.train_config.episode_length

    # Create jit_logging_inference_fn from the shared network
    make_logging_policy = ff_ppo_networks.make_logging_inference_fn(ppo_network)
    jit_logging_inference_fn = jax.jit(make_logging_policy(deterministic=True))

    def shared_vision_policy_params_fn(
        current_step,
        jit_logging_inference_fn,
        params,
        policy_params_fn_key,
        render_video,
        ppo_network,
    ):
        """Callback for shared-CNN ff_ppo: render video with egocentric overlay."""
        if not render_video:
            return

        _log_memory(f"shared_vision_policy_params_fn entry step={current_step}")

        # Run an evaluation rollout
        rollout, termination_events = _run_eval_rollout(
            jit_reset,
            jit_step,
            jit_logging_inference_fn,
            params,
            episode_length,
            policy_params_fn_key,
        )

        # Vision sensitivity diagnostic
        mid = len(rollout) // 2
        obs_with_vision = rollout[mid].obs
        obs_blank_vision = {
            k: (jp.zeros_like(v) if k == "vision" else v)
            for k, v in obs_with_vision.items()
        }
        _, sensitivity_rng = jax.random.split(policy_params_fn_key)
        act_real, _ = jit_logging_inference_fn(params, obs_with_vision, sensitivity_rng)
        act_blank, _ = jit_logging_inference_fn(
            params, obs_blank_vision, sensitivity_rng
        )
        vision_sensitivity = float(jp.linalg.norm(act_real - act_blank))
        wandb.log(
            {"eval/vision_sensitivity": vision_sensitivity},
            step=current_step,
            commit=False,
        )

        # Log per-step reward metrics
        for metric_name in [
            k for k in rollout[0].metrics.keys() if k.startswith("rewards/")
        ]:
            values = [float(s.metrics[metric_name]) for s in rollout]
            table = wandb.Table(
                data=[[i, v] for i, v in enumerate(values)],
                columns=["frame", metric_name],
            )
            wandb.log(
                {
                    f"eval/rollout_{metric_name}": wandb.plot.line(
                        table, "frame", metric_name, title=metric_name
                    )
                },
                step=current_step,
                commit=False,
            )

        # Render video
        video_path = str(checkpoint_path / f"{current_step}.mp4")
        try:
            _render_video_fn(
                rollout,
                mj_model,
                mj_data,
                renderer_obj,
                video_path,
                fps=eval_render_cfg["video"]["fps"],
                vision_renderer=_video_vision_renderer,
                termination_events=termination_events,
                hud_config=eval_render_cfg["video"].get("hud"),
                reward_config=(
                    OmegaConf.to_container(
                        cfg.env_config.env_args.get("reward_terms", {}), resolve=True
                    )
                    if cfg.env_config.get("env_args")
                    and cfg.env_config.env_args.get("reward_terms")
                    else None
                ),
            )
            wandb.log(
                {"videos/rollout": wandb.Video(video_path, format="mp4")},
                step=current_step,
                commit=False,
            )
        except mujoco.FatalError as e:
            logging.warning(f"Video rendering failed: {e}")

        _log_memory(
            f"shared_vision_policy_params_fn before cleanup step={current_step}"
        )
        _log_gpu_memory(f"before cleanup step={current_step}")

        del rollout
        del obs_with_vision, obs_blank_vision
        del act_real, act_blank
        gc.collect()
        jax.clear_caches()

        # Force Warp's CUDA memory pool to release deferred-free allocations.
        # Without this, cudaFreeAsync'd memory stays in Warp's pool indefinitely
        # and eventually exhausts GPU memory during long training runs.
        wp.synchronize()

        _log_memory(f"shared_vision_policy_params_fn after cleanup step={current_step}")
        _log_gpu_memory(f"after cleanup step={current_step}")

    # Checkpoint to restore (if any)
    checkpoint_to_restore = cfg.train_setup.get("checkpoint_to_restore", None)
    if checkpoint_to_restore is None and cfg.train_setup.get("resume_run_id", None):
        checkpoint_to_restore = str(checkpoint_path)
        logging.info(f"Auto-setting checkpoint_to_restore={checkpoint_to_restore}")

    # Vision rendering wrapper for training environments
    unwrapped_env = env.env if hasattr(env, "env") else env
    _raw_env = unwrapped_env
    while hasattr(_raw_env, "env"):
        _raw_env = _raw_env.env

    vision_width = cfg.env_config.get("vision_width", 32)
    vision_height = cfg.env_config.get("vision_height", 32)
    grayscale = cfg.env_config.get("grayscale", True)
    camera_name = cfg.env_config.get("vision_camera_name", "egocentric-rodent")
    render_depth = cfg.env_config.get("render_depth", False)
    use_textures = cfg.env_config.get("use_textures", False)
    use_shadows = cfg.env_config.get("use_shadows", False)

    def wrap_with_vision(
        environment,
        episode_length: int = 1000,
        action_repeat: int = 1,
        randomization_fn=None,
    ):
        """Wrap env for brax training, then add vision rendering."""
        brax_env = _wrap_for_brax_training(
            environment,
            cfg,
            episode_length=episode_length,
            action_repeat=action_repeat,
            randomization_fn=randomization_fn,
        )
        return VisionRenderWrapper(
            brax_env,
            mj_model=_raw_env.mj_model,
            mjx_model=_raw_env.mjx_model,
            width=vision_width,
            height=vision_height,
            grayscale=grayscale,
            camera_name=camera_name,
            render_depth=render_depth,
            use_textures=use_textures,
            use_shadows=use_shadows,
        )

    logging.info(
        f"Shared-CNN Vision+TaskObs rendering: {vision_width}x{vision_height}, "
        f"grayscale={grayscale}, camera={camera_name}"
    )

    # Build and run ff_ppo train with custom shared loss
    train_fn = functools.partial(
        ff_ppo_train.train,
        **ppo_params,
        num_evals=num_evals,
        ckpt_mgr=ckpt_mgr,
        config_dict=cfg_dict,
        checkpoint_to_restore=checkpoint_to_restore,
        network_factory=network_factory,
        progress_fn=progress_fn,
        policy_params_fn=shared_vision_policy_params_fn,
        wrap_for_training=wrap_with_vision,
        custom_loss_fn=custom_loss_fn,
        checkpoint_callback=checkpoint_callback,
    )

    logging.info("Starting shared-CNN vision+task_obs high-level PPO training...")
    make_policy, params, metrics = train_fn(
        environment=env,
        eval_env=eval_env,
    )
    return make_policy, params, metrics


# ---------------------------------------------------------------------------
# Binocular Shared-CNN Vision + TaskObs mode: stereo CNN for policy and value
# ---------------------------------------------------------------------------


def _train_binocular_shared_vision_task_obs_highlvl(
    cfg,
    env,
    eval_env,
    decoder_policy_fn,
    mimic_cfg,
    checkpoint_path,
    cfg_dict,
    progress_fn,
    prior_fn=None,
    checkpoint_callback=None,
):
    """Train high-level vision+task_obs policy with a SHARED binocular CNN.

    Mirrors ``_train_shared_vision_task_obs_highlvl`` but uses binocular
    (stereo) vision: two eye cameras are rendered and concatenated along
    the channel dimension.  A ``BinocularVisionEncoder`` processes the
    stereo input, optionally sharing weights between left and right eyes.
    Both ``policy_loss`` and ``v_loss`` gradients flow through the shared
    binocular CNN.
    """
    from track_mjx.agent.ff_ppo import losses as ff_ppo_losses

    eval_render_cfg = _resolve_eval_render_config(cfg)
    logging.info(f"Eval video eye conditions: {eval_render_cfg['eye_conditions']}")

    # In from_scratch mode (decoder_policy_fn is None) there is no intention
    # bottleneck dimensionality imposed by a frozen decoder — pull the latent
    # size straight from the config instead.
    from_scratch_mode = decoder_policy_fn is None
    if from_scratch_mode:
        latent_size = int(cfg.network_config.get("vision_latent_size", 16))
    else:
        latent_size = mimic_cfg.network_config.intention_size
    highlvl_obs_key = cfg.transfer.get("highlvl_obs_key", "imitation_target")
    decoder_obs_key = cfg.transfer.get("decoder_obs_key", "proprioception")

    # Read n_eye_actuators from the base (unwrapped) env.
    _unwrapped = env
    while hasattr(_unwrapped, "env"):
        _unwrapped = _unwrapped.env
    n_eye_actuators = getattr(_unwrapped, "n_eye_actuators", 0)
    if n_eye_actuators > 0 and not from_scratch_mode:
        logging.info(f"Actuable eyes: {n_eye_actuators} eye actuators bypass decoder")

    if from_scratch_mode:
        env = EndToEndWrapper(
            env,
            highlvl_obs_key=highlvl_obs_key,
            decoder_obs_key=decoder_obs_key,
        )
        eval_env = EndToEndWrapper(
            eval_env,
            highlvl_obs_key=highlvl_obs_key,
            decoder_obs_key=decoder_obs_key,
        )
        logging.info(
            f"EndToEndWrapper (from_scratch): action_size={env.action_size}, "
            f"obs_sizes={env.observation_size}"
        )
        _log_memory("after EndToEndWrapper")
    elif prior_fn is not None:
        env = PriorHighLevelWrapper(
            env,
            prior_fn,
            decoder_policy_fn,
            latent_size,
            highlvl_obs_key=highlvl_obs_key,
            decoder_obs_key=decoder_obs_key,
            pass_vision=True,
            pass_task_obs=True,
            deterministic_prior=cfg.transfer.get("deterministic_prior", True),
            noise_logvar=cfg.transfer.get("noise_logvar", -2.0),
            n_eye_actuators=n_eye_actuators,
        )
        eval_env = PriorHighLevelWrapper(
            eval_env,
            prior_fn,
            decoder_policy_fn,
            latent_size,
            highlvl_obs_key=highlvl_obs_key,
            decoder_obs_key=decoder_obs_key,
            pass_vision=True,
            pass_task_obs=True,
            deterministic_prior=cfg.transfer.get("deterministic_prior", True),
            noise_logvar=cfg.transfer.get("noise_logvar", -2.0),
            n_eye_actuators=n_eye_actuators,
        )
        logging.info(
            f"Binocular Shared-CNN Vision+TaskObs PriorHighLevelWrapper: action_size={env.action_size}"
        )
        _log_memory("after Binocular Shared-CNN PriorHighLevelWrapper")
    else:
        env = HighLevelWrapper(
            env,
            decoder_policy_fn,
            latent_size,
            highlvl_obs_key=highlvl_obs_key,
            decoder_obs_key=decoder_obs_key,
            pass_vision=True,
            pass_task_obs=True,
            n_eye_actuators=n_eye_actuators,
        )
        eval_env = HighLevelWrapper(
            eval_env,
            decoder_policy_fn,
            latent_size,
            highlvl_obs_key=highlvl_obs_key,
            decoder_obs_key=decoder_obs_key,
            pass_vision=True,
            pass_task_obs=True,
            n_eye_actuators=n_eye_actuators,
        )
        logging.info(
            f"Binocular Shared-CNN Vision+TaskObs HighLevelWrapper: action_size={env.action_size}"
        )
        _log_memory("after Binocular Shared-CNN HighLevelWrapper")

    # Set Warp's CUDA memory pool release threshold to 512 MB.
    # Memory above this threshold is returned to CUDA on wp.synchronize().
    try:
        cuda_device = wp.get_device("cuda:0")
        wp.set_mempool_release_threshold(cuda_device, 512 * 1024 * 1024)
        logging.info("[MEM] Set Warp mempool release threshold to 512 MB")
    except Exception as e:
        logging.warning(f"Could not set Warp mempool release threshold: {e}")

    # Read binocular_mode from config to determine shared_weights
    binocular_mode = cfg.network_config.get("binocular_mode", "shared")
    shared_weights = binocular_mode == "shared"
    mono_channels = 1 if cfg.env_config.get("grayscale", True) else 3
    logging.info(f"Binocular mode: {binocular_mode} (shared_weights={shared_weights})")

    # Detect vision shape from environment
    unwrapped = env.env if hasattr(env, "env") else env
    vision_shape = (
        unwrapped.vision_shape
        if hasattr(unwrapped, "vision_shape")
        else (
            cfg.env_config.get("vision_height", 32),
            cfg.env_config.get("vision_width", 32),
            2 * mono_channels,
        )
    )
    logging.info(f"Vision shape: {vision_shape}")

    # PPO training params
    ppo_params = dict(
        OmegaConf.to_container(cfg.train_setup.train_config, resolve=True)
    )
    ppo_params.pop("eval_naconmax", None)

    # Network factory: binocular shared-CNN vision + task_obs
    ppo_network, shared_module = (
        ff_ppo_networks.make_binocular_shared_vision_task_obs_highlvl_ppo_networks(
            obs_sizes=env.observation_size,
            action_size=env.action_size,
            vision_shape=tuple(vision_shape),
            mono_channels=mono_channels,
            shared_weights=shared_weights,
            vision_latent_size=cfg.network_config.vision_latent_size,
            vision_feature_size=cfg.network_config.get("vision_feature_size", 32),
            decoder_hidden_layer_sizes=tuple(
                cfg.network_config.decoder_hidden_layer_sizes
            ),
            value_hidden_layer_sizes=tuple(cfg.network_config.value_hidden_layer_sizes),
            vision_channels=tuple(cfg.network_config.vision_channels),
            fusion_hidden_layer_sizes=tuple(
                cfg.network_config.get("fusion_hidden_layer_sizes", [256])
            ),
        )
    )

    # Create the shared loss function (pre-baked with shared_module)
    eval_every = cfg.train_setup.get("eval_every", 10_000_000)
    num_evals = max(1, int(ppo_params["num_timesteps"] / eval_every))

    latent_kl_schedule = None
    latent_ar1_schedule = None
    if ppo_params.get("use_kl_schedule", False):
        latent_kl_schedule = ff_ppo_losses.create_ramp_schedule(
            max_value=ppo_params.get("latent_kl_weight", 0.0),
            ramp_steps=int(num_evals * ppo_params.get("kl_ramp_up_frac", 0.25)),
            schedule="linear",
        )
        latent_ar1_schedule = ff_ppo_losses.create_ramp_schedule(
            max_value=ppo_params.get("latent_ar1_weight", 0.0),
            ramp_steps=int(num_evals * ppo_params.get("kl_ramp_up_frac", 0.25)),
            schedule="linear",
        )

    prior_residual_schedule = None
    prior_residual_start = ppo_params.pop("prior_residual_start_weight", 0.0)
    prior_residual_end = ppo_params.pop("prior_residual_end_weight", 0.0)
    prior_residual_decay_frac = ppo_params.pop("prior_residual_decay_frac", 0.5)
    prior_residual_warmup_frac = ppo_params.pop("prior_residual_warmup_frac", 0.0)
    prior_residual_schedule_type = ppo_params.pop("prior_residual_schedule_type", "linear")
    if prior_residual_start > 0.0:
        prior_residual_schedule = ff_ppo_losses.create_ramp_schedule(
            # NOTE: min_value is the START value, max_value is the END value.
            # When min_value > max_value, create_ramp_schedule naturally produces
            # a decay: value = start + progress * (end - start) = start - progress * delta
            min_value=prior_residual_start,
            max_value=prior_residual_end,
            ramp_steps=int(num_evals * prior_residual_decay_frac),
            warmup_steps=int(num_evals * prior_residual_warmup_frac),
            schedule=prior_residual_schedule_type,
        )
        logging.info(
            f"Prior residual penalty: start={prior_residual_start}, "
            f"end={prior_residual_end}, decay over {prior_residual_decay_frac*100:.0f}% of training"
        )

    custom_loss_fn = functools.partial(
        ff_ppo_losses.compute_shared_vision_ppo_loss,
        ppo_network=ppo_network,
        shared_module=shared_module,
        entropy_cost=ppo_params.get("entropy_cost", 1e-3),
        latent_kl_weight=ppo_params.get("latent_kl_weight", 0.0),
        latent_ar1_weight=ppo_params.get("latent_ar1_weight", 0.0),
        discounting=ppo_params.get("discounting", 0.97),
        reward_scaling=ppo_params.get("reward_scaling", 1.0),
        gae_lambda=ppo_params.get("gae_lambda", 0.95),
        clipping_epsilon=ppo_params.get("clipping_epsilon", 0.2),
        normalize_advantage=ppo_params.get("normalize_advantage", True),
        vf_coefficient=ppo_params.get("vf_loss_coefficient", 0.5),
        latent_kl_schedule=latent_kl_schedule,
        latent_ar1_schedule=latent_ar1_schedule,
        prior_residual_weight=prior_residual_start,
        prior_residual_schedule=prior_residual_schedule,
    )

    # Wrap network_factory to return the pre-built ppo_network
    def network_factory(obs_sizes, action_size):
        return ppo_network

    # Create orbax CheckpointManager
    ckpt_mgr_options = ocp.CheckpointManagerOptions(
        save_interval_steps=1,
        max_to_keep=50,
        step_prefix="PPONetwork",
        create=True,
    )
    ckpt_mgr = ocp.CheckpointManager(str(checkpoint_path), options=ckpt_mgr_options)

    # Eval rendering setup
    mj_model = eval_env.mj_model
    mj_data = mujoco.MjData(mj_model)
    renderer_obj = mujoco.Renderer(
        mj_model,
        height=eval_render_cfg["video"]["height"],
        width=eval_render_cfg["video"]["width"],
    )
    # Create two warp vision renderers (nworld=1) for binocular eval overlay
    from vnl_playground.tasks.rodent.vision_jax import (
        BinocularVisionRenderWrapper,
        JaxVisionRenderer,
        VisionRenderWrapper,
    )

    _unwrapped = env.env if hasattr(env, "env") else env
    while hasattr(_unwrapped, "env"):
        _unwrapped = _unwrapped.env

    # Extract eye qpos indices for HUD display (None if not actuable)
    _eye_qpos_indices = getattr(_unwrapped, "_eye_qpos_indices", None)
    if _eye_qpos_indices is not None:
        logging.info(f"Actuable eyes detected: eye_qpos_indices={_eye_qpos_indices}")

    vision_width = cfg.env_config.get("vision_width", 32)
    vision_height = cfg.env_config.get("vision_height", 32)
    grayscale = cfg.env_config.get("grayscale", True)
    left_camera = cfg.env_config.get("left_camera_name", "eye_left-rodent")
    right_camera = cfg.env_config.get("right_camera_name", "eye_right-rodent")
    render_depth = cfg.env_config.get("render_depth", False)
    use_textures = cfg.env_config.get("use_textures", False)
    use_shadows = cfg.env_config.get("use_shadows", False)
    eye_dropout_rate = cfg.env_config.get("eye_dropout_rate", 0.0)
    eval_eye_mode = cfg.env_config.get("eval_eye_mode", "binocular")

    _video_left_renderer = JaxVisionRenderer(
        mj_model=_unwrapped.mj_model,
        mjx_model=_unwrapped.mjx_model,
        nworld=1,
        width=vision_width,
        height=vision_height,
        grayscale=grayscale,
        camera_name=left_camera,
        render_depth=render_depth,
        use_textures=use_textures,
        use_shadows=use_shadows,
    )
    _video_right_renderer = JaxVisionRenderer(
        mj_model=_unwrapped.mj_model,
        mjx_model=_unwrapped.mjx_model,
        nworld=1,
        width=vision_width,
        height=vision_height,
        grayscale=grayscale,
        camera_name=right_camera,
        render_depth=render_depth,
        use_textures=use_textures,
        use_shadows=use_shadows,
    )
    logging.info("Created binocular warp vision renderers (nworld=1) for video overlay")

    # Eval callback closures with binocular vision rendering
    _eval_base_reset = eval_env.reset
    _eval_base_step = eval_env.step

    def _mask_vision_in_obs(obs, eye_mode):
        """Apply a deterministic eye mask to the vision key in an obs dict.

        Used for multi-condition evaluation: run the same rollout with
        binocular, left-only, or right-only vision to compare performance.

        Args:
            obs: Observation dict with a top-level "vision" key of shape
                (H, W, 2*C). Channel layout: [left..., right...].
            eye_mode: "binocular" (no mask), "left_only" (zero right
                channels), or "right_only" (zero left channels).

        Returns:
            New obs dict with masked vision. Unchanged if binocular.
        """
        if eye_mode == "binocular":
            return obs
        vision = obs["vision"]
        c = vision.shape[-1] // 2
        if eye_mode == "left_only":
            vision = vision.at[..., c:].set(0.0)
        else:  # right_only
            vision = vision.at[..., :c].set(0.0)
        return type(obs)([(k, vision if k == "vision" else v) for k, v in obs.items()])

    def _make_masked_eval_fns(eye_mode):
        """Create JIT-compiled eval reset/step that mask one eye.

        Each eye_mode gets its own JIT trace (separate compiled function).
        This is intentional — the mask structure differs per mode.
        """

        def _masked_reset(rng):
            state = _eval_reset_with_vision(rng)
            return state.replace(obs=_mask_vision_in_obs(state.obs, eye_mode))

        def _masked_step(state, action):
            state = _eval_step_with_vision(state, action)
            return state.replace(obs=_mask_vision_in_obs(state.obs, eye_mode))

        return jax.jit(_masked_reset), jax.jit(_masked_step)

    def _eval_reset_with_vision(rng):
        state = _eval_base_reset(rng)
        data_b = _add_batch_dim_for_warp(state.data)
        left = _video_left_renderer.render(data_b)[0]
        right = _video_right_renderer.render(data_b)[0]
        vision = jp.concatenate([left, right], axis=-1)
        return state.replace(obs=VisionRenderWrapper._inject_vision(state.obs, vision))

    def _eval_step_with_vision(state, action):
        state = _eval_base_step(state, action)
        data_b = _add_batch_dim_for_warp(state.data)
        left = _video_left_renderer.render(data_b)[0]
        right = _video_right_renderer.render(data_b)[0]
        vision = jp.concatenate([left, right], axis=-1)
        return state.replace(obs=VisionRenderWrapper._inject_vision(state.obs, vision))

    jit_reset = jax.jit(_eval_reset_with_vision)
    jit_step = jax.jit(_eval_step_with_vision)

    # Masked eval functions for multi-condition evaluation
    # Each mode gets its own JIT-compiled reset/step
    _masked_eval_fns = {}
    for mode in eval_render_cfg["eye_conditions"]:
        if mode != "binocular":
            _masked_eval_fns[mode] = _make_masked_eval_fns(mode)

    # Ensure render_config has render_interval
    if "render_interval" not in cfg_dict.get("render_config", {}):
        cfg_dict.setdefault("render_config", {})["render_interval"] = 1

    # Update config_dict
    cfg_dict["network_config"].update(
        {
            "arch_name": "binocular_shared_vision_task_obs",
            "binocular_mode": binocular_mode,
            "vision_latent_size": cfg.network_config.vision_latent_size,
            "vision_feature_size": cfg.network_config.get("vision_feature_size", 32),
            "decoder_layer_sizes": list(cfg.network_config.decoder_hidden_layer_sizes),
            "critic_layer_sizes": list(cfg.network_config.value_hidden_layer_sizes),
            "fusion_hidden_layer_sizes": list(
                cfg.network_config.get("fusion_hidden_layer_sizes", [256])
            ),
        }
    )

    _render_video_fn = render_video

    episode_length = cfg.train_setup.train_config.episode_length

    # Create jit_logging_inference_fn from the shared network
    make_logging_policy = ff_ppo_networks.make_logging_inference_fn(ppo_network)
    jit_logging_inference_fn = jax.jit(make_logging_policy(deterministic=True))

    def _compute_rollout_metrics(rollout):
        """Compute aggregate metrics from a single eval rollout.

        Args:
            rollout: List of brax State objects from _run_eval_rollout.

        Returns:
            Dict of scalar metrics: cumulative_reward, mean_reward_per_step,
            num_episodes, mean_episode_length, total_gap_crossings.
        """
        total_reward = 0.0
        episode_rewards = []
        episode_lengths = []
        current_ep_reward = 0.0
        current_ep_length = 0
        total_gap_crossings = 0

        for state in rollout[1:]:  # skip initial reset state
            r = float(state.reward)
            total_reward += r
            current_ep_reward += r
            current_ep_length += 1

            # Count gap crossings from reward metric
            gap_bonus = float(state.metrics.get("rewards/gap_crossing_bonus", 0.0))
            if gap_bonus > 0:
                total_gap_crossings += 1

            if float(state.done) > 0.5:
                episode_rewards.append(current_ep_reward)
                episode_lengths.append(current_ep_length)
                current_ep_reward = 0.0
                current_ep_length = 0

        # Include the last (possibly incomplete) episode
        if current_ep_length > 0:
            episode_rewards.append(current_ep_reward)
            episode_lengths.append(current_ep_length)

        n_episodes = len(episode_rewards)
        return {
            "cumulative_reward": total_reward,
            "mean_reward_per_step": total_reward / max(len(rollout) - 1, 1),
            "num_episodes": n_episodes,
            "mean_episode_reward": (
                sum(episode_rewards) / n_episodes if n_episodes > 0 else 0.0
            ),
            "mean_episode_length": (
                sum(episode_lengths) / n_episodes if n_episodes > 0 else 0.0
            ),
            "total_gap_crossings": total_gap_crossings,
        }

    def binocular_vision_policy_params_fn(
        current_step,
        jit_logging_inference_fn,
        params,
        policy_params_fn_key,
        render_video,
        ppo_network,
    ):
        """Callback for binocular shared-CNN ff_ppo: multi-condition eval.

        Runs 3 eval rollouts (binocular, left-only, right-only) on the same
        corridor to compare performance fairly. Renders 3 videos and logs
        per-condition metrics to wandb.
        """
        if not render_video:
            return

        _log_memory(f"binocular_vision_policy_params_fn entry step={current_step}")

        eye_modes = eval_render_cfg["eye_conditions"]

        for eye_mode in eye_modes:
            # Select eval functions for this eye mode
            if eye_mode == "binocular":
                _jit_reset, _jit_step = jit_reset, jit_step
            else:
                _jit_reset, _jit_step = _masked_eval_fns[eye_mode]

            # Use same RNG so all conditions start with the same corridor
            rollout, termination_events = _run_eval_rollout(
                _jit_reset,
                _jit_step,
                jit_logging_inference_fn,
                params,
                episode_length,
                policy_params_fn_key,
            )

            # -- Per-condition metrics --
            metrics = _compute_rollout_metrics(rollout)
            for metric_name, value in metrics.items():
                wandb.log(
                    {f"eval/{eye_mode}/{metric_name}": value},
                    step=current_step,
                    commit=False,
                )

            # Vision sensitivity diagnostic (action delta: real vs blank vision)
            mid = len(rollout) // 2
            obs_with_vision = rollout[mid].obs
            obs_blank_vision = type(obs_with_vision)(
                [
                    (k, jp.zeros_like(v) if k == "vision" else v)
                    for k, v in obs_with_vision.items()
                ]
            )
            _, sensitivity_rng = jax.random.split(policy_params_fn_key)
            act_real, _ = jit_logging_inference_fn(
                params, obs_with_vision, sensitivity_rng
            )
            act_blank, _ = jit_logging_inference_fn(
                params, obs_blank_vision, sensitivity_rng
            )
            vision_sensitivity = float(jp.linalg.norm(act_real - act_blank))
            wandb.log(
                {f"eval/{eye_mode}/vision_sensitivity": vision_sensitivity},
                step=current_step,
                commit=False,
            )

            # Per-step reward line plots (only for binocular to avoid clutter)
            if eye_mode == "binocular":
                for metric_name in [
                    k for k in rollout[0].metrics.keys() if k.startswith("rewards/")
                ]:
                    values = [float(s.metrics[metric_name]) for s in rollout]
                    table = wandb.Table(
                        data=[[i, v] for i, v in enumerate(values)],
                        columns=["frame", metric_name],
                    )
                    wandb.log(
                        {
                            f"eval/rollout_{metric_name}": wandb.plot.line(
                                table,
                                "frame",
                                metric_name,
                                title=metric_name,
                            )
                        },
                        step=current_step,
                        commit=False,
                    )

            # Render video for this condition
            video_path = str(checkpoint_path / f"{current_step}_{eye_mode}.mp4")
            try:
                _render_video_fn(
                    rollout,
                    mj_model,
                    mj_data,
                    renderer_obj,
                    video_path,
                    fps=eval_render_cfg["video"]["fps"],
                    vision_renderer=_video_left_renderer,
                    right_vision_renderer=_video_right_renderer,
                    termination_events=termination_events,
                    hud_config=eval_render_cfg["video"].get("hud"),
                    reward_config=(
                        OmegaConf.to_container(
                            cfg.env_config.env_args.get("reward_terms", {}),
                            resolve=True,
                        )
                        if cfg.env_config.get("env_args")
                        and cfg.env_config.env_args.get("reward_terms")
                        else None
                    ),
                    eye_qpos_indices=_eye_qpos_indices,
                )
                wandb.log(
                    {f"videos/{eye_mode}": wandb.Video(video_path, format="mp4")},
                    step=current_step,
                    commit=False,
                )
            except mujoco.FatalError as e:
                logging.warning(f"Video rendering failed for {eye_mode}: {e}")
                jax.clear_caches()
                wp.synchronize()

            # Cleanup between conditions to limit GPU memory
            del rollout, obs_with_vision, obs_blank_vision
            del act_real, act_blank
            gc.collect()

        _log_memory(
            f"binocular_vision_policy_params_fn before final cleanup step={current_step}"
        )
        _log_gpu_memory(f"before final cleanup step={current_step}")

        jax.clear_caches()
        wp.synchronize()

        _log_memory(
            f"binocular_vision_policy_params_fn after cleanup step={current_step}"
        )
        _log_gpu_memory(f"after cleanup step={current_step}")

    # Checkpoint to restore (if any)
    checkpoint_to_restore = cfg.train_setup.get("checkpoint_to_restore", None)
    if checkpoint_to_restore is None and cfg.train_setup.get("resume_run_id", None):
        checkpoint_to_restore = str(checkpoint_path)
        logging.info(f"Auto-setting checkpoint_to_restore={checkpoint_to_restore}")

    # Vision rendering wrapper for training environments (binocular)
    unwrapped_env = env.env if hasattr(env, "env") else env
    _raw_env = unwrapped_env
    while hasattr(_raw_env, "env"):
        _raw_env = _raw_env.env

    def wrap_with_vision(
        environment,
        episode_length: int = 1000,
        action_repeat: int = 1,
        randomization_fn=None,
    ):
        """Wrap env for brax training, then add binocular vision rendering."""
        brax_env = _wrap_for_brax_training(
            environment,
            cfg,
            episode_length=episode_length,
            action_repeat=action_repeat,
            randomization_fn=randomization_fn,
        )
        return BinocularVisionRenderWrapper(
            brax_env,
            mj_model=_raw_env.mj_model,
            mjx_model=_raw_env.mjx_model,
            width=vision_width,
            height=vision_height,
            grayscale=grayscale,
            left_camera_name=left_camera,
            right_camera_name=right_camera,
            render_depth=render_depth,
            use_textures=use_textures,
            use_shadows=use_shadows,
            eye_dropout_rate=eye_dropout_rate,
            eval_eye_mode=eval_eye_mode,
        )

    logging.info(
        f"Binocular Shared-CNN Vision+TaskObs rendering: {vision_width}x{vision_height}, "
        f"grayscale={grayscale}, left_camera={left_camera}, right_camera={right_camera}, "
        f"eye_dropout_rate={eye_dropout_rate}, eval_eye_mode={eval_eye_mode}"
    )

    # Build and run ff_ppo train with custom shared loss
    train_fn = functools.partial(
        ff_ppo_train.train,
        **ppo_params,
        num_evals=num_evals,
        ckpt_mgr=ckpt_mgr,
        config_dict=cfg_dict,
        checkpoint_to_restore=checkpoint_to_restore,
        network_factory=network_factory,
        progress_fn=progress_fn,
        policy_params_fn=binocular_vision_policy_params_fn,
        wrap_for_training=wrap_with_vision,
        custom_loss_fn=custom_loss_fn,
        checkpoint_callback=checkpoint_callback,
    )

    logging.info(
        "Starting binocular shared-CNN vision+task_obs high-level PPO training..."
    )
    make_policy, params, metrics = train_fn(
        environment=env,
        eval_env=eval_env,
    )
    return make_policy, params, metrics


# ---------------------------------------------------------------------------
# Recurrent Vision + TaskObs mode: recurrent PPO with shared CNN+GRU
# ---------------------------------------------------------------------------


def _run_eval_rollout_recurrent(
    jit_reset,
    jit_step,
    inference_fn,
    params,
    episode_length,
    rng,
    init_hidden_fn,
):
    """Run eval rollout with hidden state management for recurrent policies.

    Similar to ``_run_eval_rollout`` but maintains the GRU hidden state
    across timesteps and resets it on episode termination.

    States are moved to CPU after each step to prevent GPU memory
    accumulation across the episode (the Python for-loop prevents XLA
    from reusing output buffers).

    Args:
        jit_reset: JIT-compiled environment reset function.
        jit_step: JIT-compiled environment step function.
        inference_fn: Recurrent policy ``(params, obs, hidden, key) -> (action, extras, new_hidden)``.
        params: Policy parameters.
        episode_length: Number of environment steps to run.
        rng: Random key.
        init_hidden_fn: Callable ``(batch_size) -> hidden`` for zero-init.

    Returns:
        rollout: list of states on CPU (may span multiple episodes).
        termination_events: list of ``(frame_index, reason_string)`` tuples.
    """
    _, reset_rng, act_rng = jax.random.split(rng, 3)
    state = jit_reset(reset_rng)
    # init hidden for batch_size=1, squeeze batch dim
    hidden = init_hidden_fn(1)
    hidden = jax.tree.map(lambda x: x[0], hidden)
    rollout = [jax.device_get(state)]
    termination_events = []

    for _ in range(episode_length):
        _, act_rng = jax.random.split(act_rng)
        action, _, new_hidden = inference_fn(params, state.obs, hidden, act_rng)
        hidden = new_hidden
        state = jit_step(state, action)
        rollout.append(jax.device_get(state))

        if float(state.done) > 0.5:
            reason = _get_termination_reason(state)
            termination_events.append((len(rollout) - 1, reason))
            _, reset_rng = jax.random.split(act_rng)
            state = jit_reset(reset_rng)
            rollout.append(jax.device_get(state))
            hidden = jax.tree.map(lambda x: jp.zeros_like(x), hidden)

    return rollout, termination_events


def _train_recurrent_vision_task_obs_highlvl(
    cfg,
    env,
    eval_env,
    decoder_policy_fn,
    mimic_cfg,
    checkpoint_path,
    cfg_dict,
    progress_fn,
    prior_fn=None,
    checkpoint_callback=None,
):
    """Train high-level vision+task_obs policy with recurrent CNN+GRU backbone.

    Uses a shared ``RecurrentSharedVisionModule`` (CNN encoder + GRU + policy/value
    heads) trained via the recurrent PPO pipeline.  The CNN and GRU weights are
    shared between the policy and value heads, providing a strong vision learning
    signal from both policy gradient and value-function gradients.

    This mirrors ``_train_shared_vision_task_obs_highlvl`` but replaces the
    feed-forward PPO trainer with the recurrent PPO trainer, enabling temporal
    memory through GRU hidden states.
    """
    from track_mjx.agent.recurrent_ppo.recurrent_vision_networks import (
        make_recurrent_vision_highlvl_ppo_networks,
    )
    from track_mjx.agent.recurrent_ppo.recurrent_vision_losses import (
        compute_recurrent_shared_vision_ppo_loss,
    )
    from track_mjx.agent.recurrent_ppo import ppo as recurrent_ppo_train
    from track_mjx.agent.recurrent_ppo import networks as recurrent_ppo_networks

    eval_render_cfg = _resolve_eval_render_config(cfg)

    latent_size = mimic_cfg.network_config.intention_size
    highlvl_obs_key = cfg.transfer.get("highlvl_obs_key", "imitation_target")
    decoder_obs_key = cfg.transfer.get("decoder_obs_key", "proprioception")

    # Read n_eye_actuators from the base (unwrapped) env.
    _unwrapped = env
    while hasattr(_unwrapped, "env"):
        _unwrapped = _unwrapped.env
    n_eye_actuators = getattr(_unwrapped, "n_eye_actuators", 0)
    if n_eye_actuators > 0:
        logging.info(f"Actuable eyes: {n_eye_actuators} eye actuators bypass decoder")

    if prior_fn is not None:
        env = PriorHighLevelWrapper(
            env,
            prior_fn,
            decoder_policy_fn,
            latent_size,
            highlvl_obs_key=highlvl_obs_key,
            decoder_obs_key=decoder_obs_key,
            pass_vision=True,
            pass_task_obs=True,
            deterministic_prior=cfg.transfer.get("deterministic_prior", True),
            noise_logvar=cfg.transfer.get("noise_logvar", -2.0),
            n_eye_actuators=n_eye_actuators,
        )
        eval_env = PriorHighLevelWrapper(
            eval_env,
            prior_fn,
            decoder_policy_fn,
            latent_size,
            highlvl_obs_key=highlvl_obs_key,
            decoder_obs_key=decoder_obs_key,
            pass_vision=True,
            pass_task_obs=True,
            deterministic_prior=cfg.transfer.get("deterministic_prior", True),
            noise_logvar=cfg.transfer.get("noise_logvar", -2.0),
            n_eye_actuators=n_eye_actuators,
        )
    else:
        env = HighLevelWrapper(
            env,
            decoder_policy_fn,
            latent_size,
            highlvl_obs_key=highlvl_obs_key,
            decoder_obs_key=decoder_obs_key,
            pass_vision=True,
            pass_task_obs=True,
            n_eye_actuators=n_eye_actuators,
        )
        eval_env = HighLevelWrapper(
            eval_env,
            decoder_policy_fn,
            latent_size,
            highlvl_obs_key=highlvl_obs_key,
            decoder_obs_key=decoder_obs_key,
            pass_vision=True,
            pass_task_obs=True,
            n_eye_actuators=n_eye_actuators,
        )

    logging.info(
        f"Recurrent Vision+TaskObs HighLevelWrapper: action_size={env.action_size}"
    )
    _log_memory("after Recurrent Vision+TaskObs HighLevelWrapper")

    # Set Warp's CUDA memory pool release threshold to 512 MB.
    try:
        cuda_device = wp.get_device("cuda:0")
        wp.set_mempool_release_threshold(cuda_device, 512 * 1024 * 1024)
        logging.info("[MEM] Set Warp mempool release threshold to 512 MB")
    except Exception as e:
        logging.warning(f"Could not set Warp mempool release threshold: {e}")

    # Detect vision shape from environment
    unwrapped = env.env if hasattr(env, "env") else env
    vision_shape = (
        unwrapped.vision_shape
        if hasattr(unwrapped, "vision_shape")
        else (
            cfg.env_config.get("vision_height", 32),
            cfg.env_config.get("vision_width", 32),
            1 if cfg.env_config.get("grayscale", True) else 3,
        )
    )
    logging.info(f"Vision shape: {vision_shape}")

    # PPO training params
    ppo_params = dict(
        OmegaConf.to_container(cfg.train_setup.train_config, resolve=True)
    )

    # Pop keys consumed elsewhere (not accepted by recurrent_ppo.train)
    vision_lr_multiplier = ppo_params.pop("vision_lr_multiplier", 1.0)
    ppo_params.pop("eval_naconmax", None)
    # Pop prior residual params (consumed by loss fn, not by recurrent_ppo.train)
    _prior_res_start = ppo_params.pop("prior_residual_start_weight", 0.0)
    _prior_res_end = ppo_params.pop("prior_residual_end_weight", 0.0)
    _prior_res_decay = ppo_params.pop("prior_residual_decay_frac", 0.5)
    _prior_res_warmup = ppo_params.pop("prior_residual_warmup_frac", 0.0)
    _prior_res_sched_type = ppo_params.pop("prior_residual_schedule_type", "linear")

    # Network creation: shared CNN+GRU vision module
    recurrent_ppo_network, shared_module = make_recurrent_vision_highlvl_ppo_networks(
        obs_sizes=env.observation_size,
        action_size=env.action_size,
        vision_shape=tuple(vision_shape),
        cnn_feature_size=cfg.network_config.get("vision_feature_size", 32),
        cnn_channels=tuple(cfg.network_config.vision_channels),
        gru_hidden_size=cfg.network_config.get("gru_hidden_size", 256),
        policy_hidden_sizes=tuple(cfg.network_config.get("policy_head_sizes", [256])),
        value_hidden_sizes=tuple(
            cfg.network_config.get("value_head_sizes", [256, 128])
        ),
    )

    # Compute num_evals (needed for prior residual schedule below)
    eval_every = cfg.train_setup.get("eval_every", 10_000_000)
    num_evals = max(1, int(ppo_params["num_timesteps"] / eval_every))

    # Prior residual decay schedule
    from track_mjx.agent.ff_ppo import losses as _ff_ppo_losses

    _prior_res_schedule = None
    if _prior_res_start > 0.0:
        _prior_res_schedule = _ff_ppo_losses.create_ramp_schedule(
            min_value=_prior_res_start,
            max_value=_prior_res_end,
            ramp_steps=int(num_evals * _prior_res_decay),
            warmup_steps=int(num_evals * _prior_res_warmup),
            schedule=_prior_res_sched_type,
        )
        logging.info(
            f"Prior residual penalty: start={_prior_res_start}, "
            f"end={_prior_res_end}, decay over {_prior_res_decay*100:.0f}% of training"
        )

    # Custom loss function for the recurrent shared vision network
    custom_loss_fn = functools.partial(
        compute_recurrent_shared_vision_ppo_loss,
        recurrent_ppo_network=recurrent_ppo_network,
        shared_module=shared_module,
        entropy_cost=ppo_params.get("entropy_cost", 1e-3),
        discounting=ppo_params.get("discounting", 0.97),
        reward_scaling=ppo_params.get("reward_scaling", 1.0),
        gae_lambda=ppo_params.get("gae_lambda", 0.95),
        clipping_epsilon=ppo_params.get("clipping_epsilon", 0.2),
        normalize_advantage=ppo_params.get("normalize_advantage", True),
        vf_coefficient=ppo_params.get("vf_loss_coefficient", 0.5),
        prior_residual_weight=_prior_res_start,
        prior_residual_schedule=_prior_res_schedule,
    )

    # Wrap network_factory to return the pre-built recurrent_ppo_network
    def network_factory(obs_sizes, action_size):
        return recurrent_ppo_network

    # Create orbax CheckpointManager
    ckpt_mgr_options = ocp.CheckpointManagerOptions(
        save_interval_steps=1,
        max_to_keep=50,
        step_prefix="PPONetwork",
        create=True,
    )
    ckpt_mgr = ocp.CheckpointManager(str(checkpoint_path), options=ckpt_mgr_options)

    # Eval rendering setup
    mj_model = eval_env.mj_model
    mj_data = mujoco.MjData(mj_model)
    renderer_obj = mujoco.Renderer(
        mj_model,
        height=eval_render_cfg["video"]["height"],
        width=eval_render_cfg["video"]["width"],
    )
    # Create warp vision renderer (nworld=1) for egocentric overlay
    _video_vision_renderer = None
    from vnl_playground.tasks.rodent.vision_jax import (
        JaxVisionRenderer,
        VisionRenderWrapper,
    )

    _unwrapped = env.env if hasattr(env, "env") else env
    while hasattr(_unwrapped, "env"):
        _unwrapped = _unwrapped.env

    _video_vision_renderer = JaxVisionRenderer(
        mj_model=_unwrapped.mj_model,
        mjx_model=_unwrapped.mjx_model,
        nworld=1,
        width=cfg.env_config.get("vision_width", 32),
        height=cfg.env_config.get("vision_height", 32),
        grayscale=cfg.env_config.get("grayscale", True),
        camera_name=cfg.env_config.get("vision_camera_name", "egocentric-rodent"),
        render_depth=cfg.env_config.get("render_depth", False),
        use_textures=cfg.env_config.get("use_textures", False),
        use_shadows=cfg.env_config.get("use_shadows", False),
    )
    logging.info("Created warp vision renderer (nworld=1) for video overlay")

    # Eval callback closures with vision rendering (uses _video_vision_renderer)
    _eval_base_reset = eval_env.reset
    _eval_base_step = eval_env.step

    def _eval_reset_with_vision(rng):
        state = _eval_base_reset(rng)
        data_b = _add_batch_dim_for_warp(state.data)
        vision = _video_vision_renderer.render(data_b)[0]
        return state.replace(obs=VisionRenderWrapper._inject_vision(state.obs, vision))

    def _eval_step_with_vision(state, action):
        state = _eval_base_step(state, action)
        data_b = _add_batch_dim_for_warp(state.data)
        vision = _video_vision_renderer.render(data_b)[0]
        return state.replace(obs=VisionRenderWrapper._inject_vision(state.obs, vision))

    jit_reset = jax.jit(_eval_reset_with_vision)
    jit_step = jax.jit(_eval_step_with_vision)

    # Ensure render_config has render_interval
    if "render_interval" not in cfg_dict.get("render_config", {}):
        cfg_dict.setdefault("render_config", {})["render_interval"] = 1

    # Update config_dict
    cfg_dict["network_config"].update(
        {
            "arch_name": "recurrent_vision_task_obs",
            "vision_feature_size": cfg.network_config.get("vision_feature_size", 32),
            "gru_hidden_size": cfg.network_config.get("gru_hidden_size", 256),
            "policy_head_sizes": list(
                cfg.network_config.get("policy_head_sizes", [256])
            ),
            "value_head_sizes": list(
                cfg.network_config.get("value_head_sizes", [256, 128])
            ),
        }
    )

    _render_video_fn = render_video

    episode_length = cfg.train_setup.train_config.episode_length

    # Create jit_logging_inference_fn from the recurrent network
    make_logging_policy = recurrent_ppo_networks.make_logging_inference_fn(
        recurrent_ppo_network
    )
    jit_logging_inference_fn = jax.jit(make_logging_policy(deterministic=True))

    # Hidden state initializer for eval rollouts
    init_hidden_fn = recurrent_ppo_network.policy_network.init_hidden

    def recurrent_vision_policy_params_fn(
        current_step,
        jit_logging_inference_fn,
        params,
        policy_params_fn_key,
        render_video,
        ppo_network,
    ):
        """Callback for recurrent PPO: render video with egocentric overlay."""
        if not render_video:
            return

        _log_memory(f"recurrent_vision_policy_params_fn entry step={current_step}")

        # Run a recurrent evaluation rollout
        rollout, termination_events = _run_eval_rollout_recurrent(
            jit_reset,
            jit_step,
            jit_logging_inference_fn,
            params,
            episode_length,
            policy_params_fn_key,
            init_hidden_fn,
        )

        # Vision sensitivity diagnostic
        mid = len(rollout) // 2
        obs_with_vision = rollout[mid].obs
        obs_blank_vision = {
            k: (jp.zeros_like(v) if k == "vision" else v)
            for k, v in obs_with_vision.items()
        }
        # For recurrent sensitivity check, use zero hidden state
        hidden_for_check = init_hidden_fn(1)
        hidden_for_check = jax.tree.map(lambda x: x[0], hidden_for_check)
        _, sensitivity_rng = jax.random.split(policy_params_fn_key)
        act_real, _, _ = jit_logging_inference_fn(
            params, obs_with_vision, hidden_for_check, sensitivity_rng
        )
        act_blank, _, _ = jit_logging_inference_fn(
            params, obs_blank_vision, hidden_for_check, sensitivity_rng
        )
        vision_sensitivity = float(jp.linalg.norm(act_real - act_blank))
        wandb.log(
            {"eval/vision_sensitivity": vision_sensitivity},
            step=current_step,
            commit=False,
        )

        # Log per-step reward metrics
        for metric_name in [
            k for k in rollout[0].metrics.keys() if k.startswith("rewards/")
        ]:
            values = [float(s.metrics[metric_name]) for s in rollout]
            table = wandb.Table(
                data=[[i, v] for i, v in enumerate(values)],
                columns=["frame", metric_name],
            )
            wandb.log(
                {
                    f"eval/rollout_{metric_name}": wandb.plot.line(
                        table, "frame", metric_name, title=metric_name
                    )
                },
                step=current_step,
                commit=False,
            )

        # Render video
        video_path = str(checkpoint_path / f"{current_step}.mp4")
        try:
            _render_video_fn(
                rollout,
                mj_model,
                mj_data,
                renderer_obj,
                video_path,
                fps=eval_render_cfg["video"]["fps"],
                vision_renderer=_video_vision_renderer,
                termination_events=termination_events,
                hud_config=eval_render_cfg["video"].get("hud"),
                reward_config=(
                    OmegaConf.to_container(
                        cfg.env_config.env_args.get("reward_terms", {}), resolve=True
                    )
                    if cfg.env_config.get("env_args")
                    and cfg.env_config.env_args.get("reward_terms")
                    else None
                ),
            )
            wandb.log(
                {"videos/rollout": wandb.Video(video_path, format="mp4")},
                step=current_step,
                commit=False,
            )
        except mujoco.FatalError as e:
            logging.warning(f"Video rendering failed: {e}")

        _log_memory(
            f"recurrent_vision_policy_params_fn before cleanup step={current_step}"
        )
        _log_gpu_memory(f"before cleanup step={current_step}")

        del rollout
        del obs_with_vision, obs_blank_vision
        del act_real, act_blank
        gc.collect()
        jax.clear_caches()

        # Force Warp's CUDA memory pool to release deferred-free allocations.
        wp.synchronize()

        _log_memory(
            f"recurrent_vision_policy_params_fn after cleanup step={current_step}"
        )
        _log_gpu_memory(f"after cleanup step={current_step}")

    # Checkpoint to restore (if any)
    checkpoint_to_restore = cfg.train_setup.get("checkpoint_to_restore", None)
    if checkpoint_to_restore is None and cfg.train_setup.get("resume_run_id", None):
        checkpoint_to_restore = str(checkpoint_path)
        logging.info(f"Auto-setting checkpoint_to_restore={checkpoint_to_restore}")

    # Vision rendering wrapper for training environments
    unwrapped_env = env.env if hasattr(env, "env") else env
    _raw_env = unwrapped_env
    while hasattr(_raw_env, "env"):
        _raw_env = _raw_env.env

    vision_width = cfg.env_config.get("vision_width", 32)
    vision_height = cfg.env_config.get("vision_height", 32)
    grayscale = cfg.env_config.get("grayscale", True)
    camera_name = cfg.env_config.get("vision_camera_name", "egocentric-rodent")
    render_depth = cfg.env_config.get("render_depth", False)
    use_textures = cfg.env_config.get("use_textures", False)
    use_shadows = cfg.env_config.get("use_shadows", False)

    def wrap_with_vision(
        environment,
        episode_length: int = 1000,
        action_repeat: int = 1,
        randomization_fn=None,
    ):
        """Wrap env for brax training, then add vision rendering."""
        brax_env = _wrap_for_brax_training(
            environment,
            cfg,
            episode_length=episode_length,
            action_repeat=action_repeat,
            randomization_fn=randomization_fn,
        )
        return VisionRenderWrapper(
            brax_env,
            mj_model=_raw_env.mj_model,
            mjx_model=_raw_env.mjx_model,
            width=vision_width,
            height=vision_height,
            grayscale=grayscale,
            camera_name=camera_name,
            render_depth=render_depth,
            use_textures=use_textures,
            use_shadows=use_shadows,
        )

    logging.info(
        f"Recurrent Vision+TaskObs rendering: {vision_width}x{vision_height}, "
        f"grayscale={grayscale}, camera={camera_name}"
    )

    # Build and run recurrent PPO train with custom shared loss
    train_fn = functools.partial(
        recurrent_ppo_train.train,
        **ppo_params,
        num_evals=num_evals,
        ckpt_mgr=ckpt_mgr,
        config_dict=cfg_dict,
        checkpoint_to_restore=checkpoint_to_restore,
        network_factory=network_factory,
        progress_fn=progress_fn,
        policy_params_fn=recurrent_vision_policy_params_fn,
        wrap_for_training=wrap_with_vision,
        custom_loss_fn=custom_loss_fn,
        vision_lr_multiplier=vision_lr_multiplier,
        checkpoint_callback=checkpoint_callback,
    )

    logging.info("Starting recurrent vision+task_obs high-level PPO training...")
    make_policy, params, metrics = train_fn(
        environment=env,
        eval_env=eval_env,
    )
    return make_policy, params, metrics


# ---------------------------------------------------------------------------
# Recurrent Binocular Vision + TaskObs mode: recurrent PPO with shared
# binocular CNN+GRU backbone
# ---------------------------------------------------------------------------


def _train_recurrent_binocular_vision_task_obs_highlvl(
    cfg,
    env,
    eval_env,
    decoder_policy_fn,
    mimic_cfg,
    checkpoint_path,
    cfg_dict,
    progress_fn,
    prior_fn=None,
    checkpoint_callback=None,
):
    """Train high-level vision+task_obs policy with recurrent binocular CNN+GRU.

    Combines the recurrent PPO infrastructure from
    ``_train_recurrent_vision_task_obs_highlvl`` with the binocular (stereo)
    rendering and multi-condition evaluation from
    ``_train_binocular_shared_vision_task_obs_highlvl``.

    Uses a ``RecurrentBinocularSharedVisionModule`` (binocular CNN encoder +
    GRU + policy/value heads) trained via the recurrent PPO pipeline.  The CNN
    and GRU weights are shared between the policy and value heads.
    """
    from track_mjx.agent.recurrent_ppo.recurrent_binocular_vision_networks import (
        make_recurrent_binocular_vision_highlvl_ppo_networks,
    )
    from track_mjx.agent.recurrent_ppo.recurrent_vision_losses import (
        compute_recurrent_shared_vision_ppo_loss,
    )
    from track_mjx.agent.recurrent_ppo import ppo as recurrent_ppo_train
    from track_mjx.agent.recurrent_ppo import networks as recurrent_ppo_networks

    eval_render_cfg = _resolve_eval_render_config(cfg)

    latent_size = mimic_cfg.network_config.intention_size
    highlvl_obs_key = cfg.transfer.get("highlvl_obs_key", "imitation_target")
    decoder_obs_key = cfg.transfer.get("decoder_obs_key", "proprioception")

    # Read n_eye_actuators from the base (unwrapped) env.
    _unwrapped = env
    while hasattr(_unwrapped, "env"):
        _unwrapped = _unwrapped.env
    n_eye_actuators = getattr(_unwrapped, "n_eye_actuators", 0)
    if n_eye_actuators > 0:
        logging.info(f"Actuable eyes: {n_eye_actuators} eye actuators bypass decoder")

    if prior_fn is not None:
        env = PriorHighLevelWrapper(
            env,
            prior_fn,
            decoder_policy_fn,
            latent_size,
            highlvl_obs_key=highlvl_obs_key,
            decoder_obs_key=decoder_obs_key,
            pass_vision=True,
            pass_task_obs=True,
            deterministic_prior=cfg.transfer.get("deterministic_prior", True),
            noise_logvar=cfg.transfer.get("noise_logvar", -2.0),
            n_eye_actuators=n_eye_actuators,
        )
        eval_env = PriorHighLevelWrapper(
            eval_env,
            prior_fn,
            decoder_policy_fn,
            latent_size,
            highlvl_obs_key=highlvl_obs_key,
            decoder_obs_key=decoder_obs_key,
            pass_vision=True,
            pass_task_obs=True,
            deterministic_prior=cfg.transfer.get("deterministic_prior", True),
            noise_logvar=cfg.transfer.get("noise_logvar", -2.0),
            n_eye_actuators=n_eye_actuators,
        )
    else:
        env = HighLevelWrapper(
            env,
            decoder_policy_fn,
            latent_size,
            highlvl_obs_key=highlvl_obs_key,
            decoder_obs_key=decoder_obs_key,
            pass_vision=True,
            pass_task_obs=True,
            n_eye_actuators=n_eye_actuators,
        )
        eval_env = HighLevelWrapper(
            eval_env,
            decoder_policy_fn,
            latent_size,
            highlvl_obs_key=highlvl_obs_key,
            decoder_obs_key=decoder_obs_key,
            pass_vision=True,
            pass_task_obs=True,
            n_eye_actuators=n_eye_actuators,
        )

    logging.info(
        f"Recurrent Binocular Vision+TaskObs HighLevelWrapper: action_size={env.action_size}"
    )
    _log_memory("after Recurrent Binocular Vision+TaskObs HighLevelWrapper")

    # -- Lightweight single-world env for video rollout --
    # The main eval_env inherits naconmax=24576 from training config
    # (designed for 2048 parallel envs). For single-world video rollout,
    # this causes Warp to allocate oversized collision buffers (~72 MB EPA).
    # Create a separate env with naconmax appropriate for 1 world.
    video_naconmax = eval_render_cfg["video_naconmax"]
    video_eval_args = dict(
        OmegaConf.to_container(cfg.env_config.get("env_args", {}), resolve=True)
    )
    video_eval_args["naconmax"] = video_naconmax
    # Match ctrl_dt from mimic config (same as main env)
    if hasattr(mimic_cfg, "env_config") and hasattr(mimic_cfg.env_config, "ctrl_dt"):
        video_eval_args["ctrl_dt"] = float(mimic_cfg.env_config.ctrl_dt)
    # Pass vision config (not in env_args, added programmatically in main())
    for vision_key in ("vision_width", "vision_height", "grayscale", "binocular"):
        if vision_key in cfg.env_config:
            video_eval_args[vision_key] = cfg.env_config[vision_key]

    video_eval_env = tasks.load(
        cfg.env_config.env_name, flatten_obs=False, config_overrides=video_eval_args
    )
    if prior_fn is not None:
        video_eval_env = PriorHighLevelWrapper(
            video_eval_env,
            prior_fn,
            decoder_policy_fn,
            latent_size,
            highlvl_obs_key=highlvl_obs_key,
            decoder_obs_key=decoder_obs_key,
            pass_vision=True,
            pass_task_obs=True,
            deterministic_prior=cfg.transfer.get("deterministic_prior", True),
            noise_logvar=cfg.transfer.get("noise_logvar", -2.0),
            n_eye_actuators=n_eye_actuators,
        )
    else:
        video_eval_env = HighLevelWrapper(
            video_eval_env,
            decoder_policy_fn,
            latent_size,
            highlvl_obs_key=highlvl_obs_key,
            decoder_obs_key=decoder_obs_key,
            pass_vision=True,
            pass_task_obs=True,
            n_eye_actuators=n_eye_actuators,
        )
    logging.info(
        f"Created lightweight video eval env: naconmax={video_naconmax}"
    )

    # Set Warp's CUDA memory pool release threshold to 512 MB.
    try:
        cuda_device = wp.get_device("cuda:0")
        wp.set_mempool_release_threshold(cuda_device, 512 * 1024 * 1024)
        logging.info("[MEM] Set Warp mempool release threshold to 512 MB")
    except Exception as e:
        logging.warning(f"Could not set Warp mempool release threshold: {e}")

    # Read binocular_mode from config to determine shared_weights
    binocular_mode = cfg.network_config.get("binocular_mode", "shared")
    shared_weights = binocular_mode == "shared"
    mono_channels = 1 if cfg.env_config.get("grayscale", True) else 3
    logging.info(f"Binocular mode: {binocular_mode} (shared_weights={shared_weights})")

    # Detect vision shape from environment
    unwrapped = env.env if hasattr(env, "env") else env
    vision_shape = (
        unwrapped.vision_shape
        if hasattr(unwrapped, "vision_shape")
        else (
            cfg.env_config.get("vision_height", 32),
            cfg.env_config.get("vision_width", 32),
            2 * mono_channels,
        )
    )
    logging.info(f"Vision shape: {vision_shape}")

    # PPO training params
    ppo_params = dict(
        OmegaConf.to_container(cfg.train_setup.train_config, resolve=True)
    )

    # Pop keys consumed elsewhere (not accepted by recurrent_ppo.train)
    vision_lr_multiplier = ppo_params.pop("vision_lr_multiplier", 1.0)
    ppo_params.pop("eval_naconmax", None)
    # Pop prior residual params (consumed by loss fn, not by recurrent_ppo.train)
    prior_residual_start = ppo_params.pop("prior_residual_start_weight", 0.0)
    prior_residual_end = ppo_params.pop("prior_residual_end_weight", 0.0)
    prior_residual_decay_frac = ppo_params.pop("prior_residual_decay_frac", 0.5)
    prior_residual_warmup_frac = ppo_params.pop("prior_residual_warmup_frac", 0.0)
    prior_residual_schedule_type = ppo_params.pop("prior_residual_schedule_type", "linear")

    # Network creation: shared binocular CNN+GRU vision module
    recurrent_ppo_network, shared_module = (
        make_recurrent_binocular_vision_highlvl_ppo_networks(
            obs_sizes=env.observation_size,
            action_size=env.action_size,
            vision_shape=tuple(vision_shape),
            cnn_feature_size=cfg.network_config.get("vision_feature_size", 32),
            cnn_channels=tuple(cfg.network_config.vision_channels),
            gru_hidden_size=cfg.network_config.get("gru_hidden_size", 256),
            mono_channels=mono_channels,
            shared_weights=shared_weights,
            policy_hidden_sizes=tuple(
                cfg.network_config.get("policy_head_sizes", [256])
            ),
            value_hidden_sizes=tuple(
                cfg.network_config.get("value_head_sizes", [256, 128])
            ),
        )
    )

    # Compute num_evals (needed for prior residual schedule below)
    eval_every = cfg.train_setup.get("eval_every", 10_000_000)
    num_evals = max(1, int(ppo_params["num_timesteps"] / eval_every))

    # Prior residual decay schedule
    from track_mjx.agent.ff_ppo import losses as _ff_ppo_losses

    prior_residual_schedule = None
    if prior_residual_start > 0.0:
        prior_residual_schedule = _ff_ppo_losses.create_ramp_schedule(
            min_value=prior_residual_start,
            max_value=prior_residual_end,
            ramp_steps=int(num_evals * prior_residual_decay_frac),
            warmup_steps=int(num_evals * prior_residual_warmup_frac),
            schedule=prior_residual_schedule_type,
        )
        logging.info(
            f"Prior residual penalty: start={prior_residual_start}, "
            f"end={prior_residual_end}, decay over {prior_residual_decay_frac*100:.0f}% of training"
        )

    # Custom loss function for the recurrent shared vision network
    custom_loss_fn = functools.partial(
        compute_recurrent_shared_vision_ppo_loss,
        recurrent_ppo_network=recurrent_ppo_network,
        shared_module=shared_module,
        entropy_cost=ppo_params.get("entropy_cost", 1e-3),
        discounting=ppo_params.get("discounting", 0.97),
        reward_scaling=ppo_params.get("reward_scaling", 1.0),
        gae_lambda=ppo_params.get("gae_lambda", 0.95),
        clipping_epsilon=ppo_params.get("clipping_epsilon", 0.2),
        normalize_advantage=ppo_params.get("normalize_advantage", True),
        vf_coefficient=ppo_params.get("vf_loss_coefficient", 0.5),
        prior_residual_weight=prior_residual_start,
        prior_residual_schedule=prior_residual_schedule,
    )

    # Wrap network_factory to return the pre-built recurrent_ppo_network
    def network_factory(obs_sizes, action_size):
        return recurrent_ppo_network

    # Create orbax CheckpointManager
    ckpt_mgr_options = ocp.CheckpointManagerOptions(
        save_interval_steps=1,
        max_to_keep=50,
        step_prefix="PPONetwork",
        create=True,
    )
    ckpt_mgr = ocp.CheckpointManager(str(checkpoint_path), options=ckpt_mgr_options)

    # Eval rendering setup
    mj_model = eval_env.mj_model
    mj_data = mujoco.MjData(mj_model)
    renderer_obj = mujoco.Renderer(
        mj_model,
        height=eval_render_cfg["video"]["height"],
        width=eval_render_cfg["video"]["width"],
    )
    # Create two warp vision renderers (nworld=1) for binocular eval overlay
    from vnl_playground.tasks.rodent.vision_jax import (
        BinocularVisionRenderWrapper,
        JaxVisionRenderer,
        VisionRenderWrapper,
    )

    _unwrapped = env.env if hasattr(env, "env") else env
    while hasattr(_unwrapped, "env"):
        _unwrapped = _unwrapped.env

    # Extract eye qpos indices for HUD display (None if not actuable)
    _eye_qpos_indices = getattr(_unwrapped, "_eye_qpos_indices", None)
    if _eye_qpos_indices is not None:
        logging.info(f"Actuable eyes detected: eye_qpos_indices={_eye_qpos_indices}")

    vision_width = cfg.env_config.get("vision_width", 32)
    vision_height = cfg.env_config.get("vision_height", 32)
    grayscale = cfg.env_config.get("grayscale", True)
    left_camera = cfg.env_config.get("left_camera_name", "eye_left-rodent")
    right_camera = cfg.env_config.get("right_camera_name", "eye_right-rodent")
    render_depth = cfg.env_config.get("render_depth", False)
    use_textures = cfg.env_config.get("use_textures", False)
    use_shadows = cfg.env_config.get("use_shadows", False)
    eye_dropout_rate = cfg.env_config.get("eye_dropout_rate", 0.0)
    eval_eye_mode = cfg.env_config.get("eval_eye_mode", "binocular")

    _video_left_renderer = JaxVisionRenderer(
        mj_model=_unwrapped.mj_model,
        mjx_model=_unwrapped.mjx_model,
        nworld=1,
        width=vision_width,
        height=vision_height,
        grayscale=grayscale,
        camera_name=left_camera,
        render_depth=render_depth,
        use_textures=use_textures,
        use_shadows=use_shadows,
    )
    _video_right_renderer = JaxVisionRenderer(
        mj_model=_unwrapped.mj_model,
        mjx_model=_unwrapped.mjx_model,
        nworld=1,
        width=vision_width,
        height=vision_height,
        grayscale=grayscale,
        camera_name=right_camera,
        render_depth=render_depth,
        use_textures=use_textures,
        use_shadows=use_shadows,
    )
    logging.info("Created binocular warp vision renderers (nworld=1) for video overlay")

    # Eval callback closures with binocular vision rendering
    # Use lightweight video_eval_env (small naconmax) for single-world rollout
    _eval_base_reset = video_eval_env.reset
    _eval_base_step = video_eval_env.step

    def _mask_vision_in_obs(obs, eye_mode):
        """Apply a deterministic eye mask to the vision key in an obs dict.

        Used for multi-condition evaluation: run the same rollout with
        binocular, left-only, or right-only vision to compare performance.

        Args:
            obs: Observation dict with a top-level "vision" key of shape
                (H, W, 2*C). Channel layout: [left..., right...].
            eye_mode: "binocular" (no mask), "left_only" (zero right
                channels), or "right_only" (zero left channels).

        Returns:
            New obs dict with masked vision. Unchanged if binocular.
        """
        if eye_mode == "binocular":
            return obs
        vision = obs["vision"]
        c = vision.shape[-1] // 2
        if eye_mode == "left_only":
            vision = vision.at[..., c:].set(0.0)
        else:  # right_only
            vision = vision.at[..., :c].set(0.0)
        return type(obs)([(k, vision if k == "vision" else v) for k, v in obs.items()])

    def _make_masked_eval_fns(eye_mode):
        """Create JIT-compiled eval reset/step that mask one eye.

        Each eye_mode gets its own JIT trace (separate compiled function).
        This is intentional -- the mask structure differs per mode.
        """

        def _masked_reset(rng):
            state = _eval_reset_with_vision(rng)
            return state.replace(obs=_mask_vision_in_obs(state.obs, eye_mode))

        def _masked_step(state, action):
            state = _eval_step_with_vision(state, action)
            return state.replace(obs=_mask_vision_in_obs(state.obs, eye_mode))

        return jax.jit(_masked_reset), jax.jit(_masked_step)

    def _eval_reset_with_vision(rng):
        state = _eval_base_reset(rng)
        data_b = _add_batch_dim_for_warp(state.data)
        left = _video_left_renderer.render(data_b)[0]
        right = _video_right_renderer.render(data_b)[0]
        vision = jp.concatenate([left, right], axis=-1)
        return state.replace(obs=VisionRenderWrapper._inject_vision(state.obs, vision))

    def _eval_step_with_vision(state, action):
        state = _eval_base_step(state, action)
        data_b = _add_batch_dim_for_warp(state.data)
        left = _video_left_renderer.render(data_b)[0]
        right = _video_right_renderer.render(data_b)[0]
        vision = jp.concatenate([left, right], axis=-1)
        return state.replace(obs=VisionRenderWrapper._inject_vision(state.obs, vision))

    jit_reset = jax.jit(_eval_reset_with_vision)
    jit_step = jax.jit(_eval_step_with_vision)

    # Read eval video conditions from config (default: binocular only)
    # Must be defined before _masked_eval_fns which uses it.
    eval_eye_conditions = eval_render_cfg["eye_conditions"]
    logging.info(f"Eval video eye conditions: {eval_eye_conditions}")

    # Only JIT-compile masked eval fns for conditions that are configured
    # (each JIT compilation consumes GPU memory for cached executables)
    _masked_eval_fns = {}
    for mode in eval_eye_conditions:
        if mode != "binocular":
            _masked_eval_fns[mode] = _make_masked_eval_fns(mode)

    # Ensure render_config has render_interval
    if "render_interval" not in cfg_dict.get("render_config", {}):
        cfg_dict.setdefault("render_config", {})["render_interval"] = 1

    # Update config_dict
    cfg_dict["network_config"].update(
        {
            "arch_name": "recurrent_binocular_vision_task_obs",
            "binocular_mode": binocular_mode,
            "vision_feature_size": cfg.network_config.get("vision_feature_size", 32),
            "gru_hidden_size": cfg.network_config.get("gru_hidden_size", 256),
            "policy_head_sizes": list(
                cfg.network_config.get("policy_head_sizes", [256])
            ),
            "value_head_sizes": list(
                cfg.network_config.get("value_head_sizes", [256, 128])
            ),
        }
    )

    _render_video_fn = render_video

    episode_length = cfg.train_setup.train_config.episode_length

    # Create jit_logging_inference_fn from the recurrent network
    make_logging_policy = recurrent_ppo_networks.make_logging_inference_fn(
        recurrent_ppo_network
    )
    jit_logging_inference_fn = jax.jit(make_logging_policy(deterministic=True))

    # Hidden state initializer for eval rollouts
    init_hidden_fn = recurrent_ppo_network.policy_network.init_hidden

    def _compute_rollout_metrics(rollout):
        """Compute aggregate metrics from a single eval rollout.

        Args:
            rollout: List of brax State objects from _run_eval_rollout.

        Returns:
            Dict of scalar metrics: cumulative_reward, mean_reward_per_step,
            num_episodes, mean_episode_length, total_gap_crossings.
        """
        total_reward = 0.0
        episode_rewards = []
        episode_lengths = []
        current_ep_reward = 0.0
        current_ep_length = 0
        total_gap_crossings = 0

        for state in rollout[1:]:  # skip initial reset state
            r = float(state.reward)
            total_reward += r
            current_ep_reward += r
            current_ep_length += 1

            # Count gap crossings from reward metric
            gap_bonus = float(state.metrics.get("rewards/gap_crossing_bonus", 0.0))
            if gap_bonus > 0:
                total_gap_crossings += 1

            if float(state.done) > 0.5:
                episode_rewards.append(current_ep_reward)
                episode_lengths.append(current_ep_length)
                current_ep_reward = 0.0
                current_ep_length = 0

        # Include the last (possibly incomplete) episode
        if current_ep_length > 0:
            episode_rewards.append(current_ep_reward)
            episode_lengths.append(current_ep_length)

        n_episodes = len(episode_rewards)
        return {
            "cumulative_reward": total_reward,
            "mean_reward_per_step": total_reward / max(len(rollout) - 1, 1),
            "num_episodes": n_episodes,
            "mean_episode_reward": (
                sum(episode_rewards) / n_episodes if n_episodes > 0 else 0.0
            ),
            "mean_episode_length": (
                sum(episode_lengths) / n_episodes if n_episodes > 0 else 0.0
            ),
            "total_gap_crossings": total_gap_crossings,
        }

    def recurrent_binocular_vision_policy_params_fn(
        current_step,
        jit_logging_inference_fn,
        params,
        policy_params_fn_key,
        render_video,
        ppo_network,
    ):
        """Callback for recurrent binocular PPO: multi-condition eval.

        Runs 3 eval rollouts (binocular, left-only, right-only) on the same
        corridor using recurrent rollouts with hidden state management.
        Renders 3 videos and logs per-condition metrics to wandb.
        """
        if not render_video:
            return

        _log_memory(
            f"recurrent_binocular_vision_policy_params_fn entry step={current_step}"
        )

        eye_modes = eval_eye_conditions

        for eye_mode in eye_modes:
            # Select eval functions for this eye mode
            if eye_mode == "binocular":
                _jit_reset, _jit_step = jit_reset, jit_step
            else:
                _jit_reset, _jit_step = _masked_eval_fns[eye_mode]

            # Use same RNG so all conditions start with the same corridor
            rollout, termination_events = _run_eval_rollout_recurrent(
                _jit_reset,
                _jit_step,
                jit_logging_inference_fn,
                params,
                episode_length,
                policy_params_fn_key,
                init_hidden_fn,
            )

            # -- Per-condition metrics --
            metrics = _compute_rollout_metrics(rollout)
            for metric_name, value in metrics.items():
                wandb.log(
                    {f"eval/{eye_mode}/{metric_name}": value},
                    step=current_step,
                    commit=False,
                )

            # Vision sensitivity diagnostic (action delta: real vs blank vision)
            mid = len(rollout) // 2
            obs_with_vision = rollout[mid].obs
            obs_blank_vision = type(obs_with_vision)(
                [
                    (k, jp.zeros_like(v) if k == "vision" else v)
                    for k, v in obs_with_vision.items()
                ]
            )
            # For recurrent sensitivity check, use zero hidden state
            hidden_for_check = init_hidden_fn(1)
            hidden_for_check = jax.tree.map(lambda x: x[0], hidden_for_check)
            _, sensitivity_rng = jax.random.split(policy_params_fn_key)
            act_real, _, _ = jit_logging_inference_fn(
                params, obs_with_vision, hidden_for_check, sensitivity_rng
            )
            act_blank, _, _ = jit_logging_inference_fn(
                params, obs_blank_vision, hidden_for_check, sensitivity_rng
            )
            vision_sensitivity = float(jp.linalg.norm(act_real - act_blank))
            wandb.log(
                {f"eval/{eye_mode}/vision_sensitivity": vision_sensitivity},
                step=current_step,
                commit=False,
            )

            # Per-step reward line plots (only for binocular to avoid clutter)
            if eye_mode == "binocular":
                for metric_name in [
                    k for k in rollout[0].metrics.keys() if k.startswith("rewards/")
                ]:
                    values = [float(s.metrics[metric_name]) for s in rollout]
                    table = wandb.Table(
                        data=[[i, v] for i, v in enumerate(values)],
                        columns=["frame", metric_name],
                    )
                    wandb.log(
                        {
                            f"eval/rollout_{metric_name}": wandb.plot.line(
                                table,
                                "frame",
                                metric_name,
                                title=metric_name,
                            )
                        },
                        step=current_step,
                        commit=False,
                    )

            # Render video for this condition
            video_path = str(checkpoint_path / f"{current_step}_{eye_mode}.mp4")
            try:
                _render_video_fn(
                    rollout,
                    mj_model,
                    mj_data,
                    renderer_obj,
                    video_path,
                    fps=eval_render_cfg["video"]["fps"],
                    vision_renderer=_video_left_renderer,
                    right_vision_renderer=_video_right_renderer,
                    termination_events=termination_events,
                    hud_config=eval_render_cfg["video"].get("hud"),
                    reward_config=(
                        OmegaConf.to_container(
                            cfg.env_config.env_args.get("reward_terms", {}),
                            resolve=True,
                        )
                        if cfg.env_config.get("env_args")
                        and cfg.env_config.env_args.get("reward_terms")
                        else None
                    ),
                    use_obs_vision=True,
                    eye_qpos_indices=_eye_qpos_indices,
                )
                wandb.log(
                    {f"videos/{eye_mode}": wandb.Video(video_path, format="mp4")},
                    step=current_step,
                    commit=False,
                )
            except mujoco.FatalError as e:
                logging.warning(f"Video rendering failed for {eye_mode}: {e}")

            # Cleanup between conditions to limit GPU memory
            del rollout, obs_with_vision, obs_blank_vision
            del act_real, act_blank
            gc.collect()

        _log_memory(
            f"recurrent_binocular_vision_policy_params_fn before final cleanup "
            f"step={current_step}"
        )
        _log_gpu_memory(f"before final cleanup step={current_step}")

        jax.clear_caches()
        wp.synchronize()

        _log_memory(
            f"recurrent_binocular_vision_policy_params_fn after cleanup "
            f"step={current_step}"
        )
        _log_gpu_memory(f"after cleanup step={current_step}")

    # Checkpoint to restore (if any)
    checkpoint_to_restore = cfg.train_setup.get("checkpoint_to_restore", None)
    if checkpoint_to_restore is None and cfg.train_setup.get("resume_run_id", None):
        checkpoint_to_restore = str(checkpoint_path)
        logging.info(f"Auto-setting checkpoint_to_restore={checkpoint_to_restore}")

    # Vision rendering wrapper for training environments (binocular)
    unwrapped_env = env.env if hasattr(env, "env") else env
    _raw_env = unwrapped_env
    while hasattr(_raw_env, "env"):
        _raw_env = _raw_env.env

    def wrap_with_vision(
        environment,
        episode_length: int = 1000,
        action_repeat: int = 1,
        randomization_fn=None,
    ):
        """Wrap env for brax training, then add binocular vision rendering."""
        brax_env = _wrap_for_brax_training(
            environment,
            cfg,
            episode_length=episode_length,
            action_repeat=action_repeat,
            randomization_fn=randomization_fn,
        )
        return BinocularVisionRenderWrapper(
            brax_env,
            mj_model=_raw_env.mj_model,
            mjx_model=_raw_env.mjx_model,
            width=vision_width,
            height=vision_height,
            grayscale=grayscale,
            left_camera_name=left_camera,
            right_camera_name=right_camera,
            render_depth=render_depth,
            use_textures=use_textures,
            use_shadows=use_shadows,
            eye_dropout_rate=eye_dropout_rate,
            eval_eye_mode=eval_eye_mode,
        )

    logging.info(
        f"Recurrent Binocular Vision+TaskObs rendering: {vision_width}x{vision_height}, "
        f"grayscale={grayscale}, left_camera={left_camera}, right_camera={right_camera}, "
        f"eye_dropout_rate={eye_dropout_rate}, eval_eye_mode={eval_eye_mode}"
    )

    # Build and run recurrent PPO train with custom shared loss
    train_fn = functools.partial(
        recurrent_ppo_train.train,
        **ppo_params,
        num_evals=num_evals,
        ckpt_mgr=ckpt_mgr,
        config_dict=cfg_dict,
        checkpoint_to_restore=checkpoint_to_restore,
        network_factory=network_factory,
        progress_fn=progress_fn,
        policy_params_fn=recurrent_binocular_vision_policy_params_fn,
        wrap_for_training=wrap_with_vision,
        custom_loss_fn=custom_loss_fn,
        vision_lr_multiplier=vision_lr_multiplier,
        checkpoint_callback=checkpoint_callback,
    )

    logging.info(
        "Starting recurrent binocular vision+task_obs high-level PPO training..."
    )
    make_policy, params, metrics = train_fn(
        environment=env,
        eval_env=eval_env,
    )
    return make_policy, params, metrics


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="rodent_run_gap/vision_task_obs_transfer",
)
def main(cfg: DictConfig):
    """Main training function for high-level transfer learning."""
    try:
        n_devices = jax.device_count(backend="gpu")
        logging.info(f"Using {n_devices} GPUs")
    except Exception:
        n_devices = 1
        logging.info("Not using GPUs")

    logging.info(f"Config: {OmegaConf.to_container(cfg, resolve=True)}")

    # ---- Generate or resume run ID and checkpoint path ----
    # Priority: 1) auto-discover from run_state file, 2) manual resume_run_id, 3) fresh
    existing_run_state = run_state.discover_existing_run_state(cfg)
    resume_run_id = cfg.train_setup.get("resume_run_id", None)

    if existing_run_state:
        run_id = existing_run_state["run_id"]
        checkpoint_path = Path(existing_run_state["checkpoint_path"])
        # Set checkpoint_to_restore so architecture functions auto-load state
        OmegaConf.update(
            cfg, "train_setup.checkpoint_to_restore", str(checkpoint_path), force_add=True
        )
        logging.info(
            f"AUTO-RESUME: run_id={run_id} "
            f"(step {existing_run_state.get('latest_checkpoint_step', '?')})"
        )
    elif resume_run_id:
        # Backward compat: manual resume via hydra override from autoresume.sh
        run_id = str(resume_run_id)
        checkpoint_path = Path(
            hydra.utils.to_absolute_path(f"./{cfg.logging_config.model_path}/{run_id}")
        )
        if not checkpoint_path.exists():
            # Hydra strips underscores from numeric values (Python convention).
            # Re-insert separator for YYMMDD_HHMMSS run IDs (12 digits).
            if run_id.isdigit() and len(run_id) == 12:
                run_id = f"{run_id[:6]}_{run_id[6:]}"
                checkpoint_path = Path(
                    hydra.utils.to_absolute_path(
                        f"./{cfg.logging_config.model_path}/{run_id}"
                    )
                )
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Cannot resume: checkpoint path {checkpoint_path} does not exist"
                )
        logging.info(f"MANUAL RESUME: run_id={run_id}")
    else:
        run_id = datetime.now().strftime("%y%m%d_%H%M%S")
        checkpoint_path = Path(
            hydra.utils.to_absolute_path(f"./{cfg.logging_config.model_path}/{run_id}")
        )
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"NEW run_id: {run_id}")

    # Save config to checkpoint directory
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    with open(checkpoint_path / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2, default=lambda o: str(o))

    logging.info(f"Checkpoint path: {checkpoint_path}")

    # ---- Determine transfer mode ----
    transfer_mode = cfg.transfer.get("mode", "decoder_only")
    logging.info(f"Transfer mode: {transfer_mode}")

    if transfer_mode == "from_scratch":
        # ---- No weight transfer. End-to-end training from random init. ----
        # Skip loading any SCAMPER/mimic checkpoint. The env will be wrapped
        # in EndToEndWrapper (below) which passes full joint actions straight
        # through — no frozen decoder, no prior RNN. The high-level binocular
        # shared-vision network is used as-is, with its internal decoder head
        # trained from random initialization alongside the rest of the policy.
        logging.info("from_scratch mode: no SCAMPER weights will be loaded.")
        prior_fn = None
        decoder_policy_fn = None
        mimic_cfg = None
        _log_memory("from_scratch: skipped prior/decoder load")

    elif transfer_mode == "prior_decoder":
        # ---- Load frozen prior + decoder from SCAMPER prior checkpoint ----
        from vnl_playground.tasks.prior_utils import (
            load_prior_checkpoint,
            make_decoder_inference_fn as make_prior_decoder_fn,
            make_prior_inference_fn,
        )

        prior_ckpt_path = hydra.utils.to_absolute_path(
            cfg.transfer.prior_checkpoint_path
        )
        prior_ckpt_step = cfg.transfer.get("prior_checkpoint_step", None)
        logging.info(f"Loading prior checkpoint from: {prior_ckpt_path}")

        (
            encoder_params,
            prior_params,
            decoder_params,
            normalizer_params,
            prior_cfg,
        ) = load_prior_checkpoint(prior_ckpt_path, prior_ckpt_step)

        latent_size = prior_cfg["network_config"]["intention_size"]
        logging.info(f"Prior checkpoint: intention_size={latent_size}")

        # Create frozen inference functions
        prior_fn = make_prior_inference_fn(prior_params, normalizer_params, prior_cfg)
        decoder_policy_fn = make_prior_decoder_fn(
            decoder_params, normalizer_params, prior_cfg
        )

        # Build a mimic_cfg-like object for dispatch functions
        # NOTE: ctrl_dt enforcement from the prior checkpoint happens below
        # via the shared mimic_cfg.env_config.ctrl_dt path, after env_args is created.
        mimic_cfg = OmegaConf.create(prior_cfg)
        _log_memory("after prior checkpoint load")

    else:
        prior_fn = None

        if cfg.transfer.get("prior_checkpoint_path", None) is not None:
            # ---- Load frozen decoder from SCAMPER prior checkpoint (decoder_only) ----
            from vnl_playground.tasks.prior_utils import (
                load_prior_checkpoint,
                make_decoder_inference_fn as make_scamper_decoder_fn,
            )

            prior_ckpt_path = hydra.utils.to_absolute_path(
                cfg.transfer.prior_checkpoint_path
            )
            prior_ckpt_step = cfg.transfer.get("prior_checkpoint_step", None)
            logging.info(
                f"Loading frozen decoder from SCAMPER prior checkpoint: "
                f"{prior_ckpt_path}"
            )

            (
                _encoder_params,
                _prior_params,
                decoder_params,
                normalizer_params,
                prior_cfg,
            ) = load_prior_checkpoint(prior_ckpt_path, prior_ckpt_step)

            latent_size = prior_cfg["network_config"]["intention_size"]
            logging.info(f"SCAMPER decoder loaded. intention_size={latent_size}")

            decoder_policy_fn = make_scamper_decoder_fn(
                decoder_params, normalizer_params, prior_cfg
            )
            mimic_cfg = OmegaConf.create(prior_cfg)
            _log_memory("after SCAMPER decoder load")

        else:
            # ---- Load frozen decoder from Phase 1 (track-mjx) checkpoint ----
            mimic_ckpt_path = hydra.utils.to_absolute_path(
                f"{cfg.transfer.mimic_checkpoint_dir}/{cfg.transfer.mimic_run_id}"
            )
            logging.info(f"Loading frozen decoder from: {mimic_ckpt_path}")

            mimic_cfg = OmegaConf.create(
                checkpointing.load_config_from_checkpoint(mimic_ckpt_path)
            )
            decoder_policy_fn = ff_ppo_networks.make_decoder_policy_fn(mimic_ckpt_path)
            logging.info(
                f"Decoder loaded. intention_size="
                f"{mimic_cfg.network_config.intention_size}"
            )
            _log_memory("after decoder load")

    # ---- Load environment ----
    env_name = cfg.env_config.env_name
    env_args = OmegaConf.to_container(cfg.env_config.get("env_args", {}), resolve=True)
    if not env_args:
        env_args = {}

    # CRITICAL: Match ctrl_dt from mimic config so the frozen decoder produces
    # correct behavior. The decoder was trained with a specific ctrl_dt.
    # In from_scratch mode there is no frozen decoder, so no ctrl_dt constraint
    # is imposed — the value from cfg.env_config.env_args is used directly.
    if mimic_cfg is not None and hasattr(mimic_cfg, "env_config") and hasattr(
        mimic_cfg.env_config, "ctrl_dt"
    ):
        mimic_ctrl_dt = float(mimic_cfg.env_config.ctrl_dt)
        env_args["ctrl_dt"] = mimic_ctrl_dt
        logging.info(f"Enforcing ctrl_dt={mimic_ctrl_dt} from mimic config")

    # Pass vision config to env so its reported vision_shape matches the
    # VisionRenderWrapper dimensions used for actual rendering.
    for vision_key in ("vision_width", "vision_height", "grayscale", "binocular"):
        if vision_key in cfg.env_config:
            env_args[vision_key] = cfg.env_config[vision_key]

    env = tasks.load(
        env_name, flatten_obs=False, config_overrides=env_args if env_args else None
    )

    # Determine naconmax for eval env: use explicit config if set, otherwise
    # auto-scale from training naconmax proportionally to num_eval_envs.
    eval_env_args = dict(env_args) if env_args else {}
    num_envs = cfg.train_setup.train_config.get("num_envs", 2048)
    num_eval_envs = cfg.train_setup.train_config.get("num_eval_envs", 128)
    configured_eval_naconmax = cfg.train_setup.train_config.get("eval_naconmax", None)
    if configured_eval_naconmax is not None:
        eval_env_args["naconmax"] = configured_eval_naconmax
        logging.info(
            f"Eval env naconmax: {configured_eval_naconmax} (from config)"
        )
    elif "naconmax" in eval_env_args and num_envs > 0:
        per_world_ncon = eval_env_args["naconmax"] / num_envs
        eval_naconmax = max(int(per_world_ncon * num_eval_envs), 256)
        eval_env_args["naconmax"] = eval_naconmax
        logging.info(
            f"Eval env naconmax: {eval_naconmax} "
            f"(scaled from {env_args['naconmax']} for {num_eval_envs} eval envs)"
        )

    eval_env = tasks.load(
        env_name, flatten_obs=False, config_overrides=eval_env_args if eval_env_args else None
    )

    _validate_reset_requirements(cfg, env, env_name)

    logging.info(f"Loaded environment: {env_name}")
    logging.info(f"Action size: {env.action_size}")
    _log_memory("after env load")

    # ---- Initialize wandb ----
    wandb_run_id = f"{cfg.logging_config.exp_name}_{env_name}_{run_id}"
    wandb_resume = "allow"

    if existing_run_state:
        # Auto-resume: use the exact wandb_run_id from run state
        wandb_run_id = existing_run_state["wandb_run_id"]
        wandb_resume = "must"
        logging.info(f"Resuming wandb run (auto): {wandb_run_id}")
    elif resume_run_id:
        # Manual resume: try run_state file in checkpoint dir, then legacy wandb_state
        saved = load_wandb_state(checkpoint_path)
        if saved:
            wandb_run_id = saved["wandb_run_id"]
            wandb_resume = "must"
            logging.info(f"Resuming wandb run (legacy): {wandb_run_id}")

    wandb.init(
        project=cfg.logging_config.project_name,
        config=cfg_dict,
        notes=cfg.logging_config.get("notes", ""),
        id=wandb_run_id,
        resume=wandb_resume,
        group=cfg.logging_config.get("group_name", env_name),
    )

    # Persist run state (atomic, file-locked) for crash recovery
    run_state.save_run_state(cfg, run_id, str(checkpoint_path), wandb_run_id)
    # Also save wandb_run_id inside checkpoint dir (survives --fresh which only
    # deletes run_state_*.json in the model_path root)
    save_wandb_state(checkpoint_path, wandb_run_id)

    # Checkpoint callback updates run state after every checkpoint save
    checkpoint_callback = run_state.create_checkpoint_callback(
        cfg, run_id, str(checkpoint_path), wandb_run_id
    )
    _log_memory("after wandb init")

    def wandb_progress(num_steps, metrics):
        # Convert JAX Arrays to Python floats to prevent wandb holding JAX references
        metrics = {
            k: float(v) if hasattr(v, "dtype") else v for k, v in metrics.items()
        }
        metrics["num_steps_thousands"] = num_steps
        proc = psutil.Process()
        metrics["system/rss_gb"] = proc.memory_info().rss / (1024**3)
        metrics["system/rss_mb"] = proc.memory_info().rss / (1024**2)
        wandb.log(metrics)

    # ---- Dispatch based on architecture (with optional curriculum) ----
    arch_name = cfg.network_config.arch_name

    def _dispatch_train(
        cfg_phase,
        env_phase,
        eval_env_phase,
        checkpoint_path_phase,
        cfg_dict_phase,
        progress_fn_phase,
        checkpoint_callback_phase=None,
    ):
        """Dispatch to the appropriate architecture-specific training function."""
        if arch_name == "mlp":
            return _train_mlp_highlvl(
                cfg_phase,
                env_phase,
                eval_env_phase,
                decoder_policy_fn,
                mimic_cfg,
                checkpoint_path_phase,
                cfg_dict_phase,
                progress_fn=progress_fn_phase,
                prior_fn=prior_fn,
                # MLP uses Brax PPO which doesn't support checkpoint_callback
            )
        elif arch_name == "vision_task_obs":
            return _train_vision_task_obs_highlvl(
                cfg_phase,
                env_phase,
                eval_env_phase,
                decoder_policy_fn,
                mimic_cfg,
                checkpoint_path_phase,
                cfg_dict_phase,
                progress_fn=progress_fn_phase,
                prior_fn=prior_fn,
                checkpoint_callback=checkpoint_callback_phase,
            )
        elif arch_name == "shared_vision_task_obs":
            return _train_shared_vision_task_obs_highlvl(
                cfg_phase,
                env_phase,
                eval_env_phase,
                decoder_policy_fn,
                mimic_cfg,
                checkpoint_path_phase,
                cfg_dict_phase,
                progress_fn=progress_fn_phase,
                prior_fn=prior_fn,
                checkpoint_callback=checkpoint_callback_phase,
            )
        elif arch_name == "recurrent_vision_task_obs":
            return _train_recurrent_vision_task_obs_highlvl(
                cfg_phase,
                env_phase,
                eval_env_phase,
                decoder_policy_fn,
                mimic_cfg,
                checkpoint_path_phase,
                cfg_dict_phase,
                progress_fn=progress_fn_phase,
                prior_fn=prior_fn,
                checkpoint_callback=checkpoint_callback_phase,
            )
        elif arch_name == "binocular_shared_vision_task_obs":
            return _train_binocular_shared_vision_task_obs_highlvl(
                cfg_phase,
                env_phase,
                eval_env_phase,
                decoder_policy_fn,
                mimic_cfg,
                checkpoint_path_phase,
                cfg_dict_phase,
                progress_fn=progress_fn_phase,
                prior_fn=prior_fn,
                checkpoint_callback=checkpoint_callback_phase,
            )
        elif arch_name == "recurrent_binocular_vision_task_obs":
            return _train_recurrent_binocular_vision_task_obs_highlvl(
                cfg_phase,
                env_phase,
                eval_env_phase,
                decoder_policy_fn,
                mimic_cfg,
                checkpoint_path_phase,
                cfg_dict_phase,
                progress_fn=progress_fn_phase,
                prior_fn=prior_fn,
                checkpoint_callback=checkpoint_callback_phase,
            )
        else:
            raise ValueError(
                f"Unknown arch_name: {arch_name}. "
                "Must be 'mlp', 'vision_task_obs', 'shared_vision_task_obs', "
                "'recurrent_vision_task_obs', 'binocular_shared_vision_task_obs', "
                "or 'recurrent_binocular_vision_task_obs'."
            )

    # ---- Check for auto-curriculum mode ----
    has_curriculum = "curriculum" in cfg_dict and "phases" in cfg_dict.get(
        "curriculum", {}
    )

    if has_curriculum:
        from vnl_playground.tasks.rodent.curriculum import (
            GraduationMonitor,
            apply_phase_to_env_config,
            apply_phase_to_train_config,
            build_phases_from_config,
            make_curriculum_progress_fn,
        )

        curriculum_phases = build_phases_from_config(cfg_dict["curriculum"])
        logging.info(
            f"AUTO-CURRICULUM: {len(curriculum_phases)} phases detected. "
            f"Phases: {[p.name for p in curriculum_phases]}"
        )

        base_env_args = dict(env_args)

        # Check for previously completed phases (resume support)
        phase_state_file = checkpoint_path / "curriculum_phase.txt"
        start_phase_idx = 0
        if phase_state_file.exists():
            try:
                saved_phase = int(phase_state_file.read_text().strip())
                start_phase_idx = saved_phase
                logging.info(
                    f"CURRICULUM RESUME: Skipping to phase {start_phase_idx + 1}"
                )
            except ValueError:
                pass

        for phase_idx, phase in enumerate(curriculum_phases):
            if phase_idx < start_phase_idx:
                logging.info(f"Skipping completed phase {phase_idx + 1}: {phase.name}")
                continue

            logging.info(
                f"\n{'='*60}\n"
                f"CURRICULUM PHASE {phase_idx + 1}/{len(curriculum_phases)}: {phase.name}\n"
                f"  Gap distances: {phase.gap_distances}\n"
                f"  Hold duration: {phase.hold_duration}\n"
                f"  Episode length: {phase.episode_length}\n"
                f"  Learning rate: {phase.learning_rate}\n"
                f"  Num timesteps: {phase.num_timesteps:,}\n"
                f"  Graduation threshold: {phase.graduation_threshold}\n"
                f"{'='*60}"
            )

            # Build phase-specific env_args
            phase_env_args = apply_phase_to_env_config(base_env_args, phase)
            # Pass vision config
            for vision_key in ("vision_width", "vision_height", "grayscale"):
                if vision_key in cfg.env_config:
                    phase_env_args[vision_key] = cfg.env_config[vision_key]
            # Enforce ctrl_dt from mimic
            if hasattr(mimic_cfg, "env_config") and hasattr(
                mimic_cfg.env_config, "ctrl_dt"
            ):
                phase_env_args["ctrl_dt"] = float(mimic_cfg.env_config.ctrl_dt)

            # Reload environments with phase-specific config
            phase_env = tasks.load(
                env_name,
                flatten_obs=False,
                config_overrides=phase_env_args if phase_env_args else None,
            )
            phase_eval_env = tasks.load(
                env_name,
                flatten_obs=False,
                config_overrides=phase_env_args if phase_env_args else None,
            )

            # Build phase-specific train config
            phase_train_config = apply_phase_to_train_config(
                OmegaConf.to_container(cfg.train_setup.train_config, resolve=True),
                phase,
            )

            # Create phase-specific OmegaConf config
            phase_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            phase_cfg.train_setup.train_config = OmegaConf.create(phase_train_config)
            phase_cfg.env_config.env_args = OmegaConf.create(phase_env_args)

            # Set checkpoint_to_restore for phases after the first
            if phase_idx > 0:
                phase_cfg.train_setup.checkpoint_to_restore = str(checkpoint_path)
                logging.info(
                    f"Phase {phase_idx + 1}: loading checkpoint from {checkpoint_path}"
                )

            phase_cfg_dict = OmegaConf.to_container(phase_cfg, resolve=True)

            # Phase checkpoint subdirectory
            phase_ckpt_path = checkpoint_path / f"phase_{phase_idx + 1}"
            phase_ckpt_path.mkdir(parents=True, exist_ok=True)

            # Setup graduation monitor
            monitor = GraduationMonitor(
                threshold=phase.graduation_threshold,
                patience=phase.graduation_patience,
                checkpoint_dir=str(phase_ckpt_path),
            )
            phase_progress_fn = make_curriculum_progress_fn(
                monitor,
                wandb_progress,
                phase_idx + 1,
                phase.name,
            )

            # Run training for this phase
            logging.info(f"Architecture: {arch_name}")
            _dispatch_train(
                phase_cfg,
                phase_env,
                phase_eval_env,
                phase_ckpt_path,
                phase_cfg_dict,
                phase_progress_fn,
                checkpoint_callback_phase=checkpoint_callback,
            )

            graduated = monitor.should_graduate
            final_sr = monitor.latest_success_rate
            logging.info(
                f"Phase {phase_idx + 1} complete. "
                f"Graduated: {graduated}, Final success rate: {final_sr:.3f}"
            )

            # Save curriculum progress for resume
            phase_state_file.write_text(str(phase_idx + 1))
            logging.info(f"Saved curriculum state: completed phase {phase_idx + 1}")

            # Clean up phase environments
            del phase_env, phase_eval_env
            gc.collect()

        logging.info("AUTO-CURRICULUM: All phases complete.")

    else:
        # Standard single-phase training
        logging.info(f"Architecture: {arch_name}")
        _dispatch_train(
            cfg, env, eval_env, checkpoint_path, cfg_dict, wandb_progress,
            checkpoint_callback_phase=checkpoint_callback,
        )

    logging.info("Training complete.")
    run_state.cleanup_run_state(cfg)
    wandb.finish()


if __name__ == "__main__":
    main()
