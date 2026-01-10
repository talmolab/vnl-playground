"""Entry point for VNL playground transfer learning training."""

import os

# Must set rendering backend before importing MuJoCo
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
# Disable JAX VRAM preallocation (must be set before importing JAX)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import functools
import logging

import hydra
import jax
import orbax.checkpoint as ocp
import wandb
from mujoco_playground import wrapper as playground_wrappers
from omegaconf import DictConfig, OmegaConf

from vnl_playground.tasks.rodent import flat_arena, bowl_escape, maze_forage
from vnl_playground.tasks.rodent import wrappers as vnl_wrappers
from vnl_playground.config.utils import prepare_config

from track_mjx.agent import checkpointing, wandb_logging
from track_mjx.agent.mlp_ppo import ppo, ppo_networks
from track_mjx.agent.domain_randomization import domain_randomization_maker


def _setup_environment() -> None:
    """Configure environment variables for JAX."""
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags


# Task name to environment class mapping
_TASK_ENV_MAP = {
    "maze_forage": maze_forage.MazeForage,
    "bowl_escape": bowl_escape.BowlEscape,
    "flat_arena": flat_arena.FlatWalk,
}


def _create_task_environment(task_name: str, env_config: DictConfig):
    """Create environment based on task_name.

    Args:
        task_name: Name of the task (maze_forage, bowl_escape, flat_arena).
        env_config: Environment configuration to pass as overrides.

    Returns:
        Tuple of (train_env, eval_env) wrapped with FlattenObsWrapper.

    Raises:
        ValueError: If task_name is not recognized.
    """
    if task_name not in _TASK_ENV_MAP:
        raise ValueError(
            f"Unknown task_name: {task_name}. Must be one of: {list(_TASK_ENV_MAP.keys())}"
        )

    env_cls = _TASK_ENV_MAP[task_name]
    env = vnl_wrappers.FlattenObsWrapper(env_cls(config_overrides=env_config))
    eval_env = vnl_wrappers.FlattenObsWrapper(env_cls(config_overrides=env_config))
    return env, eval_env


def _load_checkpoint_config(cfg: DictConfig) -> DictConfig:
    """Load and merge config from checkpoint if restoring.

    Args:
        cfg: Current configuration.

    Returns:
        Updated configuration with checkpoint values merged.
    """
    if cfg.train_setup.checkpoint_to_restore is None:
        return cfg

    checkpoint_to_restore = cfg.train_setup.checkpoint_to_restore
    cfg_loaded = OmegaConf.create(
        checkpointing.load_config_from_checkpoint(checkpoint_to_restore)
    )

    # Overwrite network parameters from checkpoint
    logging.info(
        f"Overwriting decoder layer sizes from checkpoint: "
        f"{cfg.network_config.decoder_layer_sizes} -> {cfg_loaded.network_config.decoder_layer_sizes}"
    )
    cfg.network_config.decoder_layer_sizes = cfg_loaded.network_config.decoder_layer_sizes

    logging.info(
        f"Overwriting intention size from checkpoint: "
        f"{cfg.network_config.intention_size} -> {cfg_loaded.network_config.intention_size}"
    )
    cfg.network_config.intention_size = cfg_loaded.network_config.intention_size

    logging.info(
        f"Overwriting rescale factor from checkpoint: "
        f"{cfg.walker_config.rescale_factor} -> {cfg_loaded.walker_config.rescale_factor}"
    )
    cfg.walker_config.rescale_factor = cfg_loaded.walker_config.rescale_factor
    cfg.env_config.env_args.rescale_factor = cfg_loaded.walker_config.rescale_factor

    return cfg


@hydra.main(version_base=None, config_path="config", config_name="bowl_escape_transfer")
def main(cfg: DictConfig) -> None:
    """Main training entry point using Hydra configs.

    Initializes JAX devices, creates train/eval environments, and runs PPO
    training with wandb logging for transfer learning tasks.

    Args:
        cfg: Hydra configuration containing env_config, network_config,
            train_setup, and logging_config.
    """
    _setup_environment()

    try:
        n_devices = jax.device_count(backend="gpu")
        logging.info(f"Using {n_devices} GPUs")
    except RuntimeError:
        n_devices = 1
        logging.info("Not using GPUs")

    logging.info(f"Configs: {OmegaConf.to_container(cfg, resolve=True)}")

    # Load checkpoint config if restoring
    cfg = _load_checkpoint_config(cfg)

    # Prepare config by resolving walker paths and creating config variants
    cfg, cfg_dict, env_cfg_ml = prepare_config(cfg)

    # Determine how to load from checkpoint
    run_id, checkpoint_path, existing_run_state = checkpointing.load_from_run_state(
        cfg, freeze_decoder=cfg.train_setup.freeze_decoder
    )

    # Initialize checkpoint manager
    mgr_options = ocp.CheckpointManagerOptions(
        create=True,
        max_to_keep=cfg.train_setup.checkpoint_max_to_keep,
        keep_period=cfg.train_setup.checkpoint_keep_period,
        step_prefix="PPONetwork",
    )
    ckpt_mgr = ocp.CheckpointManager(checkpoint_path, options=mgr_options)

    logging.info(f"run_id: {run_id}")
    logging.info(f"Training checkpoint path: {checkpoint_path}")

    # Create environments
    task_name = cfg.env_config.task_name
    env_config = cfg.env_config.env_args
    env, eval_env = _create_task_environment(task_name, env_config)

    logging.info(f"Environment config: {cfg.env_config}")

    # Create network factory
    network_factory = functools.partial(
        ppo_networks.make_intention_ppo_networks,
        intention_latent_size=cfg.network_config.intention_size,
        encoder_hidden_layer_sizes=tuple(cfg.network_config.encoder_layer_sizes),
        decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
        value_hidden_layer_sizes=tuple(cfg.network_config.critic_layer_sizes),
    )

    # Initialize wandb logging
    wandb_logging.initialize_wandb_logging(
        logging_cfg=cfg.logging_config,
        cfg=cfg,
        run_id=run_id,
        existing_run_state=existing_run_state,
    )

    # Save initial run state after wandb initialization
    if existing_run_state is None:
        checkpointing.save_run_state(
            cfg=cfg,
            run_id=run_id,
            checkpoint_path=checkpoint_path,
            wandb_run_id=wandb.run.id,
        )

    # Create the checkpoint callback
    checkpoint_callback = checkpointing.create_checkpoint_callback(
        cfg=cfg,
        run_id=run_id,
        checkpoint_path=checkpoint_path,
        wandb_run_id=wandb.run.id,
    )

    train_fn = functools.partial(
        ppo.train,
        **cfg.train_setup.train_config,
        num_evals=int(
            cfg.train_setup.train_config.num_timesteps / cfg.train_setup.eval_every
        ),
        num_resets_per_eval=cfg.train_setup.eval_every // cfg.train_setup.reset_every,
        latent_kl_weight=cfg.network_config.latent_kl_weight,
        latent_ar1_weight=cfg.network_config.latent_ar1_weight,
        network_factory=network_factory,
        ckpt_mgr=ckpt_mgr,
        checkpoint_to_restore=cfg.train_setup.checkpoint_to_restore,
        config_dict=cfg_dict,
        use_kl_schedule=cfg.network_config.kl_schedule,
        freeze_decoder=cfg.train_setup.freeze_decoder,
        checkpoint_callback=checkpoint_callback,
                wrap_for_training=functools.partial(
            playground_wrappers.wrap_for_brax_training
        ),
        randomization_fn=(
            domain_randomization_maker(
                floor_friction=cfg.env_config.domain_randomization.floor_friction,
                static_friction_scale=cfg.env_config.domain_randomization.static_friction_scale,
                armature_scale=cfg.env_config.domain_randomization.armature_scale,
                com_jitter=cfg.env_config.domain_randomization.com_jitter,
                link_mass_scale=cfg.env_config.domain_randomization.link_mass_scale,
                torso_mass_jitter=cfg.env_config.domain_randomization.torso_mass_jitter,
                qpos0_jitter=cfg.env_config.domain_randomization.qpos0_jitter,
            )
            if cfg.env_config.domain_randomization.use_domain_randomization
            else None
        ),
    )

    # Define the jit reset/step functions for rollout logging
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    policy_params_fn = functools.partial(
        wandb_logging.rollout_logging_fn,
        eval_env,
        jit_reset,
        jit_step,
        cfg,
        checkpoint_path,
    )

    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=wandb_logging.wandb_progress,
        policy_params_fn=policy_params_fn,
    )

    # Clean up run state after successful completion
    try:
        checkpointing.cleanup_run_state(cfg)
        logging.info("Training completed successfully, cleaned up run state")
    except Exception as e:
        logging.warning(f"Failed to cleanup run state: {e}")


if __name__ == "__main__":
    main()
