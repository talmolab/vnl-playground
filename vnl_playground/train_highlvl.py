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
from vnl_playground.tasks.wrappers import HighLevelWrapper, PriorHighLevelWrapper


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
    """
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

    # -- Batch ego GPU pre-rendering ------------------------------------------
    ego_overlay_np = None
    if vision_renderer is not None:
        # Stack the kinematic arrays needed for rendering from all rollout states
        all_data = jax.tree.map(
            lambda *xs: jax.numpy.stack(xs), *[s.data for s in rollout]
        )

        @jax.jit
        def _render_all_ego(stacked_data):
            """Render egocentric views for all timesteps in one call."""

            def body(carry, data_slice):
                batched = _add_batch_dim_for_warp(data_slice)
                img = vision_renderer.render(batched)
                return carry, img[0]  # (H, W, C)

            _, all_imgs = jax.lax.scan(body, None, stacked_data)
            return all_imgs  # (T, H, W, C)

        ego_imgs_jax = _render_all_ego(all_data)
        ego_frames_np = np.array(ego_imgs_jax)  # single transfer to host
        del ego_imgs_jax

        # Binocular: render right eye and place side-by-side with left
        if right_vision_renderer is not None:

            @jax.jit
            def _render_all_right(stacked_data):
                def body(carry, data_slice):
                    batched = _add_batch_dim_for_warp(data_slice)
                    img = right_vision_renderer.render(batched)
                    return carry, img[0]

                _, all_imgs = jax.lax.scan(body, None, stacked_data)
                return all_imgs

            right_imgs_jax = _render_all_right(all_data)
            right_frames_np = np.array(right_imgs_jax)
            del right_imgs_jax
            # Side-by-side: concatenate left and right along width (axis=2)
            # Add 2px gap between eyes
            gap = np.ones_like(ego_frames_np[:, :, :2, :])  # (T, H, 2, C) white gap
            ego_frames_np = np.concatenate(
                [ego_frames_np, gap, right_frames_np], axis=2
            )
            del right_frames_np, gap

        del all_data
        gc.collect()

        # Vectorized ego overlay preparation (grayscale→RGB, uint8, 2x scale)
        ego_overlay_np = _prepare_ego_overlay(ego_frames_np)
        del ego_frames_np
        gc.collect()

    # -- Render main camera frames + composite overlay ------------------------
    with imageio.get_writer(video_path, fps=fps) as writer:
        termination_dict = {}
        if termination_events:
            termination_dict = {idx: reason for idx, reason in termination_events}

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
    latent_size = mimic_cfg.network_config.intention_size
    highlvl_obs_key = cfg.transfer.get("highlvl_obs_key", "imitation_target")
    decoder_obs_key = cfg.transfer.get("decoder_obs_key", "proprioception")

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
        )
    else:
        env = HighLevelWrapper(
            env,
            decoder_policy_fn,
            latent_size,
            highlvl_obs_key=highlvl_obs_key,
            decoder_obs_key=decoder_obs_key,
            pass_vision=False,
        )
        eval_env = HighLevelWrapper(
            eval_env,
            decoder_policy_fn,
            latent_size,
            highlvl_obs_key=highlvl_obs_key,
            decoder_obs_key=decoder_obs_key,
            pass_vision=False,
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
        height=cfg.render_config.render_height,
        width=cfg.render_config.render_width,
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
                fps=cfg.render_config.render_fps,
                termination_events=termination_events,
            )
            wandb.log(
                {"videos/rollout": wandb.Video(video_path, format="mp4")},
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
        wrap_env_fn=functools.partial(mp_wrapper.wrap_for_brax_training),
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
):
    """Train high-level vision+task_obs policy using ff_ppo with CNN encoder.

    The HighLevelWrapper passes both vision and task_obs (body signals),
    and uses a fusion network that combines CNN features with the task
    observation vector.
    """
    latent_size = mimic_cfg.network_config.intention_size
    highlvl_obs_key = cfg.transfer.get("highlvl_obs_key", "imitation_target")
    decoder_obs_key = cfg.transfer.get("decoder_obs_key", "proprioception")

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
        )
        eval_env = HighLevelWrapper(
            eval_env,
            decoder_policy_fn,
            latent_size,
            highlvl_obs_key=highlvl_obs_key,
            decoder_obs_key=decoder_obs_key,
            pass_vision=True,
            pass_task_obs=True,
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
        height=cfg.render_config.render_height,
        width=cfg.render_config.render_width,
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
        wandb.log({"eval/vision_sensitivity": vision_sensitivity}, commit=False)

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
                fps=cfg.render_config.render_fps,
                vision_renderer=_video_vision_renderer,
                termination_events=termination_events,
            )
            wandb.log(
                {"videos/rollout": wandb.Video(video_path, format="mp4")},
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
        brax_env = mp_wrapper.wrap_for_brax_training(
            environment,
            episode_length=episode_length,
            action_repeat=action_repeat,
            randomization_fn=randomization_fn,
            full_reset=False,
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
):
    """Train high-level vision+task_obs policy with a SHARED CNN.

    Mirrors the vnl-ray architecture: a single VisionEncoder is shared
    between the policy and value heads.  Both policy_loss and v_loss
    gradients flow through the CNN, providing a much stronger learning
    signal for vision features.
    """
    from track_mjx.agent.ff_ppo import losses as ff_ppo_losses

    latent_size = mimic_cfg.network_config.intention_size
    highlvl_obs_key = cfg.transfer.get("highlvl_obs_key", "imitation_target")
    decoder_obs_key = cfg.transfer.get("decoder_obs_key", "proprioception")

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
        )
        eval_env = HighLevelWrapper(
            eval_env,
            decoder_policy_fn,
            latent_size,
            highlvl_obs_key=highlvl_obs_key,
            decoder_obs_key=decoder_obs_key,
            pass_vision=True,
            pass_task_obs=True,
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
        height=cfg.render_config.render_height,
        width=cfg.render_config.render_width,
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
        wandb.log({"eval/vision_sensitivity": vision_sensitivity}, commit=False)

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
                fps=cfg.render_config.render_fps,
                vision_renderer=_video_vision_renderer,
                termination_events=termination_events,
            )
            wandb.log(
                {"videos/rollout": wandb.Video(video_path, format="mp4")},
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
        brax_env = mp_wrapper.wrap_for_brax_training(
            environment,
            episode_length=episode_length,
            action_repeat=action_repeat,
            randomization_fn=randomization_fn,
            full_reset=False,
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

    latent_size = mimic_cfg.network_config.intention_size
    highlvl_obs_key = cfg.transfer.get("highlvl_obs_key", "imitation_target")
    decoder_obs_key = cfg.transfer.get("decoder_obs_key", "proprioception")

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
        )
        eval_env = HighLevelWrapper(
            eval_env,
            decoder_policy_fn,
            latent_size,
            highlvl_obs_key=highlvl_obs_key,
            decoder_obs_key=decoder_obs_key,
            pass_vision=True,
            pass_task_obs=True,
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
        height=cfg.render_config.render_height,
        width=cfg.render_config.render_width,
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

    vision_width = cfg.env_config.get("vision_width", 32)
    vision_height = cfg.env_config.get("vision_height", 32)
    grayscale = cfg.env_config.get("grayscale", True)
    left_camera = cfg.env_config.get("left_camera_name", "eye_left-rodent")
    right_camera = cfg.env_config.get("right_camera_name", "eye_right-rodent")
    render_depth = cfg.env_config.get("render_depth", False)
    use_textures = cfg.env_config.get("use_textures", False)
    use_shadows = cfg.env_config.get("use_shadows", False)

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

    def binocular_vision_policy_params_fn(
        current_step,
        jit_logging_inference_fn,
        params,
        policy_params_fn_key,
        render_video,
        ppo_network,
    ):
        """Callback for binocular shared-CNN ff_ppo: render video with egocentric overlay."""
        if not render_video:
            return

        _log_memory(f"binocular_vision_policy_params_fn entry step={current_step}")

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
        wandb.log({"eval/vision_sensitivity": vision_sensitivity}, commit=False)

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
                commit=False,
            )

        # Render video with binocular (side-by-side left+right) ego overlay
        video_path = str(checkpoint_path / f"{current_step}.mp4")
        try:
            _render_video_fn(
                rollout,
                mj_model,
                mj_data,
                renderer_obj,
                video_path,
                fps=cfg.render_config.render_fps,
                vision_renderer=_video_left_renderer,
                right_vision_renderer=_video_right_renderer,
                termination_events=termination_events,
            )
            wandb.log(
                {"videos/rollout": wandb.Video(video_path, format="mp4")},
                commit=False,
            )
        except mujoco.FatalError as e:
            logging.warning(f"Video rendering failed: {e}")

        _log_memory(
            f"binocular_vision_policy_params_fn before cleanup step={current_step}"
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
        brax_env = mp_wrapper.wrap_for_brax_training(
            environment,
            episode_length=episode_length,
            action_repeat=action_repeat,
            randomization_fn=randomization_fn,
            full_reset=False,
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
        )

    logging.info(
        f"Binocular Shared-CNN Vision+TaskObs rendering: {vision_width}x{vision_height}, "
        f"grayscale={grayscale}, left_camera={left_camera}, right_camera={right_camera}"
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

    Args:
        jit_reset: JIT-compiled environment reset function.
        jit_step: JIT-compiled environment step function.
        inference_fn: Recurrent policy ``(params, obs, hidden, key) -> (action, extras, new_hidden)``.
        params: Policy parameters.
        episode_length: Number of environment steps to run.
        rng: Random key.
        init_hidden_fn: Callable ``(batch_size) -> hidden`` for zero-init.

    Returns:
        rollout: list of states (may span multiple episodes).
        termination_events: list of ``(frame_index, reason_string)`` tuples.
    """
    _, reset_rng, act_rng = jax.random.split(rng, 3)
    state = jit_reset(reset_rng)
    # init hidden for batch_size=1, squeeze batch dim
    hidden = init_hidden_fn(1)
    hidden = jax.tree.map(lambda x: x[0], hidden)
    rollout = [state]
    termination_events = []

    for _ in range(episode_length):
        _, act_rng = jax.random.split(act_rng)
        action, _, new_hidden = inference_fn(params, state.obs, hidden, act_rng)
        hidden = new_hidden
        state = jit_step(state, action)
        rollout.append(state)

        if float(state.done) > 0.5:
            reason = _get_termination_reason(state)
            termination_events.append((len(rollout) - 1, reason))
            _, reset_rng = jax.random.split(act_rng)
            state = jit_reset(reset_rng)
            rollout.append(state)
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

    latent_size = mimic_cfg.network_config.intention_size
    highlvl_obs_key = cfg.transfer.get("highlvl_obs_key", "imitation_target")
    decoder_obs_key = cfg.transfer.get("decoder_obs_key", "proprioception")

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
        )
        eval_env = HighLevelWrapper(
            eval_env,
            decoder_policy_fn,
            latent_size,
            highlvl_obs_key=highlvl_obs_key,
            decoder_obs_key=decoder_obs_key,
            pass_vision=True,
            pass_task_obs=True,
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

    # Pop vision_lr_multiplier from ppo_params (passed separately to train)
    vision_lr_multiplier = ppo_params.pop("vision_lr_multiplier", 1.0)

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
    )

    # Wrap network_factory to return the pre-built recurrent_ppo_network
    def network_factory(obs_sizes, action_size):
        return recurrent_ppo_network

    # Compute num_evals
    eval_every = cfg.train_setup.get("eval_every", 10_000_000)
    num_evals = max(1, int(ppo_params["num_timesteps"] / eval_every))

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
        height=cfg.render_config.render_height,
        width=cfg.render_config.render_width,
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
        wandb.log({"eval/vision_sensitivity": vision_sensitivity}, commit=False)

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
                fps=cfg.render_config.render_fps,
                vision_renderer=_video_vision_renderer,
                termination_events=termination_events,
            )
            wandb.log(
                {"videos/rollout": wandb.Video(video_path, format="mp4")},
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
        brax_env = mp_wrapper.wrap_for_brax_training(
            environment,
            episode_length=episode_length,
            action_repeat=action_repeat,
            randomization_fn=randomization_fn,
            full_reset=False,
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
    )

    logging.info("Starting recurrent vision+task_obs high-level PPO training...")
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
    resume_run_id = cfg.train_setup.get("resume_run_id", None)
    if resume_run_id:
        run_id = str(resume_run_id)
        checkpoint_path = Path(
            hydra.utils.to_absolute_path(f"./{cfg.logging_config.model_path}/{run_id}")
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Cannot resume: checkpoint path {checkpoint_path} does not exist"
            )
        logging.info(f"RESUMING run_id: {run_id}")
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

    if transfer_mode == "prior_decoder":
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
    if hasattr(mimic_cfg, "env_config") and hasattr(mimic_cfg.env_config, "ctrl_dt"):
        mimic_ctrl_dt = float(mimic_cfg.env_config.ctrl_dt)
        env_args["ctrl_dt"] = mimic_ctrl_dt
        logging.info(f"Enforcing ctrl_dt={mimic_ctrl_dt} from mimic config")

    # Pass vision config to env so its reported vision_shape matches the
    # VisionRenderWrapper dimensions used for actual rendering.
    for vision_key in ("vision_width", "vision_height", "grayscale"):
        if vision_key in cfg.env_config:
            env_args[vision_key] = cfg.env_config[vision_key]

    env = tasks.load(
        env_name, flatten_obs=False, config_overrides=env_args if env_args else None
    )
    eval_env = tasks.load(
        env_name, flatten_obs=False, config_overrides=env_args if env_args else None
    )

    logging.info(f"Loaded environment: {env_name}")
    logging.info(f"Action size: {env.action_size}")
    _log_memory("after env load")

    # ---- Initialize wandb ----
    wandb_run_id = f"{cfg.logging_config.exp_name}_{env_name}_{run_id}"
    wandb.init(
        project=cfg.logging_config.project_name,
        config=cfg_dict,
        notes=cfg.logging_config.get("notes", ""),
        id=wandb_run_id,
        resume="allow",
        group=cfg.logging_config.get("group_name", env_name),
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
            )
        else:
            raise ValueError(
                f"Unknown arch_name: {arch_name}. "
                "Must be 'mlp', 'vision_task_obs', 'shared_vision_task_obs', "
                "'recurrent_vision_task_obs', or 'binocular_shared_vision_task_obs'."
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

        for phase_idx, phase in enumerate(curriculum_phases):
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
            )

            graduated = monitor.should_graduate
            final_sr = monitor.latest_success_rate
            logging.info(
                f"Phase {phase_idx + 1} complete. "
                f"Graduated: {graduated}, Final success rate: {final_sr:.3f}"
            )

            # Clean up phase environments
            del phase_env, phase_eval_env
            gc.collect()

        logging.info("AUTO-CURRICULUM: All phases complete.")

    else:
        # Standard single-phase training
        logging.info(f"Architecture: {arch_name}")
        _dispatch_train(cfg, env, eval_env, checkpoint_path, cfg_dict, wandb_progress)

    logging.info("Training complete.")
    wandb.finish()


if __name__ == "__main__":
    main()
