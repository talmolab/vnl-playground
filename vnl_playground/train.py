"""Generalizable PPO training for VNL tasks using ff_ppo from track-mjx.

Loads any registered task from the vnl_playground registry and trains
using the ff_ppo training loop with support for vision (CNN) and
intention (VAE) network architectures.

This script is task-agnostic: set env_config.env_name in the YAML config
to train any environment registered in vnl_playground.tasks.

The ff_ppo training loop expects dict observations (not flat arrays), so
environments are loaded WITHOUT FlattenObsWrapper.

Usage:
    python -m vnl_playground.train
    python -m vnl_playground.train --config-name=run_gap_vision
    python -m vnl_playground.train train_setup.train_config.num_envs=2048
    python -m vnl_playground.train env_config.env_name=RodentRunGap
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
import json
import logging
from datetime import datetime
from pathlib import Path

import hydra
import imageio
import jax
import jax.numpy as jp
import mujoco
import numpy as np
import wandb
from mujoco_playground import wrapper as mp_wrapper
from omegaconf import DictConfig, OmegaConf
from orbax import checkpoint as ocp

import gc
import psutil

from track_mjx.agent.ff_ppo import ppo as ff_ppo_train
from track_mjx.agent.ff_ppo import ppo_networks as ff_ppo_networks
from track_mjx.agent.ff_ppo.ppo_networks import (
    make_logging_inference_fn as ff_make_logging_inference_fn,
)

from vnl_playground import tasks
from vnl_playground.tasks.wrappers import LegacyObsWrapper

def _log_memory(label: str):
    """Log current process RSS memory usage."""
    proc = psutil.Process()
    rss_gb = proc.memory_info().rss / (1024**3)
    logging.info(f"[MEM] {label}: {rss_gb:.2f} GB RSS")
    return rss_gb


# Enable persistent compilation cache.
# jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update(
#     "jax_persistent_cache_enable_xla_caches",
#     "xla_gpu_per_fusion_autotune_cache_dir",
# )


def render_video(
    rollout,
    mj_model,
    mj_data,
    renderer,
    video_path,
    fps=50,
    vision_renderer=None,
):
    """Render a rollout to an MP4 video file with tracking camera.

    If ``vision_renderer`` (a ``JaxVisionRenderer`` with nworld=1) is
    provided, the agent's egocentric view rendered by the warp GPU
    ray-tracer is overlaid in the upper-left corner of each frame.

    Egocentric renders are batched into a single JAX call via
    ``jax.lax.scan`` to avoid per-call host-memory leaks from the
    Warp FFI ``jax_callable`` bridge.
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

    # Pre-render ALL egocentric frames in one batched JIT call to avoid
    # per-call memory leak from Warp FFI jax_callable.
    ego_frames_np = None
    if vision_renderer is not None:
        # Stack the kinematic arrays needed for rendering from all rollout states
        all_data = jax.tree.map(lambda *xs: jax.numpy.stack(xs), *[s.data for s in rollout])

        @jax.jit
        def _render_all_ego(stacked_data):
            """Render egocentric views for all timesteps in one call."""
            def body(carry, data_slice):
                batched = jax.tree.map(lambda x: x[None, ...], data_slice)
                img = vision_renderer.render(batched)
                return carry, img[0]  # (H, W, C)
            _, all_imgs = jax.lax.scan(body, None, stacked_data)
            return all_imgs  # (T, H, W, C)

        ego_imgs_jax = _render_all_ego(all_data)
        ego_frames_np = np.array(ego_imgs_jax)  # single transfer to host
        del ego_imgs_jax, all_data
        gc.collect()

    with imageio.get_writer(video_path, fps=fps) as writer:
        for i, state in enumerate(rollout):
            mj_data.qpos = np.array(state.data.qpos)
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data, camera, scene_option=scene_option)
            frame = renderer.render()

            # Overlay warp-rendered egocentric vision in upper-left corner
            if ego_frames_np is not None:
                ego_np = ego_frames_np[i]  # (H, W, C) float32 [0,1]

                # Convert to uint8 RGB
                if ego_np.shape[-1] == 1:
                    ego_np = np.repeat(ego_np, 3, axis=-1)
                ego_uint8 = np.clip(ego_np * 255, 0, 255).astype(np.uint8)

                # Scale up for visibility (2x)
                scale = 2
                ego_scaled = np.repeat(
                    np.repeat(ego_uint8, scale, axis=0), scale, axis=1
                )
                sh, sw = ego_scaled.shape[:2]

                # Place with white border
                pad = 2
                y0, x0 = pad + 4, pad + 4
                y1, x1 = y0 + sh, x0 + sw
                if y1 < frame.shape[0] and x1 < frame.shape[1]:
                    frame[y0 - pad : y1 + pad, x0 - pad : x1 + pad] = 255
                    frame[y0:y1, x0:x1] = ego_scaled

            writer.append_data(frame)


@hydra.main(version_base=None, config_path="config", config_name="run_gap_vision")
def main(cfg: DictConfig):
    """Main training function."""
    try:
        n_devices = jax.device_count(backend="gpu")
        logging.info(f"Using {n_devices} GPUs")
    except Exception:
        n_devices = 1
        logging.info("Not using GPUs")

    logging.info(f"Config: {OmegaConf.to_container(cfg, resolve=True)}")

    # Generate run ID and checkpoint path
    run_id = datetime.now().strftime("%y%m%d_%H%M%S")
    checkpoint_path = Path(
        hydra.utils.to_absolute_path(f"./{cfg.logging_config.model_path}/{run_id}")
    )
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Save config to checkpoint directory
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    with open(checkpoint_path / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)

    logging.info(f"run_id: {run_id}")
    logging.info(f"Checkpoint path: {checkpoint_path}")

    # Load environment from registry WITHOUT FlattenObsWrapper
    # ff_ppo expects dict observations
    env_name = cfg.env_config.env_name
    env_args = OmegaConf.to_container(
        cfg.env_config.get("env_args", {}), resolve=True
    )
    if not env_args:
        env_args = None

    env = tasks.load(env_name, flatten_obs=False, config_overrides=env_args)
    eval_env = tasks.load(env_name, flatten_obs=False, config_overrides=env_args)

    # Strip state/privileged_state wrapping for non-vision training.
    # ff_ppo expects dict obs with proprioception/imitation_target at top level.
    arch_name = cfg.network_config.arch_name
    if arch_name != "vision":
        env = LegacyObsWrapper(env)
        eval_env = LegacyObsWrapper(eval_env)

    logging.info(f"Loaded environment: {env_name}")
    logging.info(f"Action size: {env.action_size}")

    # Detect vision shape from environment
    unwrapped = env.env if hasattr(env, "env") else env
    vision_shape = (
        unwrapped.vision_shape
        if hasattr(unwrapped, "vision_shape")
        else (64, 64, 3)
    )
    logging.info(f"Vision shape: {vision_shape}")
    _log_memory("after env load")

    # PPO training params
    ppo_params = dict(
        OmegaConf.to_container(cfg.train_setup.train_config, resolve=True)
    )

    # Build network factory based on architecture

    if arch_name == "vision":
        network_factory = functools.partial(
            ff_ppo_networks.make_vision_ppo_networks,
            vision_shape=tuple(vision_shape),
            vision_latent_size=cfg.network_config.vision_latent_size,
            decoder_hidden_layer_sizes=tuple(
                cfg.network_config.decoder_hidden_layer_sizes
            ),
            value_hidden_layer_sizes=tuple(
                cfg.network_config.value_hidden_layer_sizes
            ),
            vision_channels=tuple(cfg.network_config.vision_channels),
        )
    else:
        # Default: intention-based architecture
        network_factory = functools.partial(
            ff_ppo_networks.make_intention_ppo_networks,
            intention_latent_size=cfg.network_config.get(
                "intention_latent_size", 60
            ),
            encoder_hidden_layer_sizes=tuple(
                cfg.network_config.get("encoder_hidden_layer_sizes", [1024, 1024])
            ),
            decoder_hidden_layer_sizes=tuple(
                cfg.network_config.get("decoder_hidden_layer_sizes", [1024, 1024])
            ),
            value_hidden_layer_sizes=tuple(
                cfg.network_config.get("value_hidden_layer_sizes", [1024, 1024])
            ),
        )

    # Create orbax CheckpointManager for ff_ppo
    ckpt_mgr_options = ocp.CheckpointManagerOptions(
        save_interval_steps=1,
        max_to_keep=5,
        step_prefix="PPONetwork",
        create=True,
    )
    ckpt_mgr = ocp.CheckpointManager(
        str(checkpoint_path), options=ckpt_mgr_options
    )

    # Setup eval rendering
    mj_model = eval_env.mj_model
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(
        mj_model,
        height=cfg.render_config.render_height,
        width=cfg.render_config.render_width,
    )
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)

    # Update config_dict with network_config fields that ff_ppo expects
    cfg_dict["network_config"].update(
        {
            "arch_name": arch_name,
            "vision_latent_size": cfg.network_config.get("vision_latent_size", 32),
            "decoder_layer_sizes": list(
                cfg.network_config.get("decoder_hidden_layer_sizes", [512, 512])
            ),
            "critic_layer_sizes": list(
                cfg.network_config.get("value_hidden_layer_sizes", [512, 512])
            ),
        }
    )

    # Initialize wandb
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
        metrics["num_steps_thousands"] = num_steps
        proc = psutil.Process()
        metrics["system/rss_gb"] = proc.memory_info().rss / (1024**3)
        metrics["system/rss_mb"] = proc.memory_info().rss / (1024**2)
        wandb.log(metrics)

    # Save reference before it gets shadowed by the bool parameter in the callback.
    _render_video_fn = render_video

    # Create a warp vision renderer (nworld=1) for egocentric overlay in videos
    _video_vision_renderer = None
    if arch_name == "vision":
        from vnl_playground.tasks.rodent.vision_jax import JaxVisionRenderer

        _unwrapped = env.env if hasattr(env, "env") else env
        _video_vision_renderer = JaxVisionRenderer(
            mj_model=_unwrapped.mj_model,
            mjx_model=_unwrapped.mjx_model,
            nworld=1,
            width=cfg.env_config.get("vision_width", 32),
            height=cfg.env_config.get("vision_height", 32),
            grayscale=cfg.env_config.get("grayscale", True),
            camera_name=cfg.env_config.get("vision_camera_name", "egocentric-rodent"),
        )
        logging.info("Created warp vision renderer (nworld=1) for video overlay")

    def policy_params_fn(
        current_step,
        jit_logging_inference_fn,
        params,
        policy_params_fn_key,
        render_video,  # noqa: N803 -- bool flag set by ff_ppo caller
        ppo_network,
    ):
        """Callback for policy evaluation, video rendering, and metric logging."""
        if not render_video:
            return

        _log_memory(f"policy_params_fn entry step={current_step}")

        episode_length = cfg.train_setup.train_config.episode_length

        # Run an evaluation rollout using the logging inference function
        _, reset_rng, act_rng = jax.random.split(policy_params_fn_key, 3)
        state = jit_reset(reset_rng)
        rollout = [state]

        for _ in range(episode_length):
            _, act_rng = jax.random.split(act_rng)
            action, _ = jit_logging_inference_fn(params, state.obs, act_rng)
            state = jit_step(state, action)
            rollout.append(state)

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
                renderer,
                video_path,
                fps=cfg.render_config.render_fps,
                vision_renderer=_video_vision_renderer,
            )
            wandb.log(
                {"videos/rollout": wandb.Video(video_path, format="mp4")},
                commit=False,
            )
        except mujoco.FatalError as e:
            logging.warning(f"Video rendering failed: {e}")

        _log_memory(f"policy_params_fn before cleanup step={current_step}")

        # --- Explicit memory cleanup ---
        del rollout
        gc.collect()

    # Compute num_evals for ff_ppo
    num_evals = max(
        1, int(ppo_params["num_timesteps"] / cfg.train_setup.eval_every)
    )

    # Checkpoint to restore (if any)
    checkpoint_to_restore = cfg.train_setup.get("checkpoint_to_restore", None)

    # Build and run train function
    if arch_name == "vision":
        # Vision training: wrap env with VisionRenderWrapper for JAX-native
        # GPU rendering, then use standard ff_ppo.train with lax.scan.
        from vnl_playground.tasks.rodent.vision_jax import VisionRenderWrapper

        # Get the raw env's mj_model and vision config
        unwrapped_env = env.env if hasattr(env, "env") else env
        vision_width = cfg.env_config.get("vision_width", 32)
        vision_height = cfg.env_config.get("vision_height", 32)
        grayscale = cfg.env_config.get("grayscale", True)
        camera_name = cfg.env_config.get("vision_camera_name", "egocentric-rodent")

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
            # nworld=None: renderer is lazily initialized on first reset(),
            # auto-detecting the batch size.  This lets the same wrapper
            # function work for both training (num_envs) and eval
            # (num_eval_envs) environments.
            return VisionRenderWrapper(
                brax_env,
                mj_model=unwrapped_env.mj_model,
                mjx_model=unwrapped_env.mjx_model,
                width=vision_width,
                height=vision_height,
                grayscale=grayscale,
                camera_name=camera_name,
            )

        logging.info(
            f"Vision rendering: {vision_width}x{vision_height}, "
            f"grayscale={grayscale}, camera={camera_name}, "
            f"JAX-callable (inside lax.scan)"
        )

        train_fn = functools.partial(
            ff_ppo_train.train,
            **ppo_params,
            num_evals=num_evals,
            ckpt_mgr=ckpt_mgr,
            config_dict=cfg_dict,
            checkpoint_to_restore=checkpoint_to_restore,
            network_factory=network_factory,
            progress_fn=wandb_progress,
            policy_params_fn=policy_params_fn,
            wrap_for_training=wrap_with_vision,
        )

        logging.info("Starting vision PPO training (JAX-native rendering)...")
        make_policy, params, metrics = train_fn(
            environment=env,
            eval_env=eval_env,
        )
    else:
        # Non-vision training: use standard ff_ppo.train with lax.scan
        train_fn = functools.partial(
            ff_ppo_train.train,
            **ppo_params,
            num_evals=num_evals,
            ckpt_mgr=ckpt_mgr,
            config_dict=cfg_dict,
            checkpoint_to_restore=checkpoint_to_restore,
            network_factory=network_factory,
            progress_fn=wandb_progress,
            policy_params_fn=policy_params_fn,
            wrap_for_training=functools.partial(
                mp_wrapper.wrap_for_brax_training, full_reset=True
            ),
        )

        logging.info("Starting ff_ppo training...")
        make_policy, params, metrics = train_fn(
            environment=env,
            eval_env=eval_env,
        )

    logging.info("Training complete.")

    wandb.finish()


if __name__ == "__main__":
    main()
