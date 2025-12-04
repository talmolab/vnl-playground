"""
Entry point for track-mjx. Load the config file, create environments, initialize network, and start training.
"""

import os
import sys

# set default env variable if not set
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = os.environ.get(
#     "XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6"
# )

# # limit to 1 GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # visible GPU masks

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import jax
# Enable persistent compilation cache.
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)

import hydra
from omegaconf import DictConfig, OmegaConf
import functools
import wandb
import orbax.checkpoint as ocp
from track_mjx.agent.mlp_ppo import ppo, ppo_networks
import warnings
from pathlib import Path
from datetime import datetime
import logging
import mujoco

from vnl_mjx.tasks.rodent import flat_arena, bowl_escape, maze_forage

from track_mjx.agent import checkpointing
from track_mjx.agent import wandb_logging
from track_mjx.analysis import render

warnings.filterwarnings("ignore", category=DeprecationWarning)


@hydra.main(version_base=None, config_path="config", config_name="bowl_escape_transfer")
def main(cfg: DictConfig):
    """Main function using Hydra configs"""
    try:
        n_devices = jax.device_count(backend="gpu")
        logging.info(f"Using {n_devices} GPUs")
    except:
        n_devices = 1
        logging.info("Not using GPUs")

    logging.info(f"Configs: {OmegaConf.to_container(cfg, resolve=True)}")

    # Generate a new run_id and associated checkpoint path
    run_id = datetime.now().strftime("%y%m%d_%H%M%S")
    # TODO: Use a base path given by the config
    checkpoint_path = hydra.utils.to_absolute_path(
        f"./{cfg.logging_config.model_path}/{run_id}"
    )

    # Load the checkpoint's config
    if cfg.train_setup["checkpoint_to_restore"] is not None:
        # TODO: We set the restored config's checkpoint_to_restore to itself
        # Because that restored config is used from now on. This is a hack.
        checkpoint_to_restore = cfg.train_setup["checkpoint_to_restore"]
        # Load the checkpoint's config and update the run_id and checkpoint path
        cfg_loaded = OmegaConf.create(
            checkpointing.load_config_from_checkpoint(checkpoint_to_restore)
        )
        print(
            "Overwriting decoder layer sizes from checkpoint from {} to {}".format(
                cfg.network_config.decoder_layer_sizes,
                cfg_loaded.network_config.decoder_layer_sizes,
            )
        )
        cfg.network_config.decoder_layer_sizes = cfg_loaded.network_config.decoder_layer_sizes
        print(
            "Overwriting intention size from checkpoint from {} to {}".format(
                cfg.network_config.intention_size,
                cfg_loaded.network_config.intention_size,
            )
        )
        cfg.network_config.intention_size = cfg_loaded.network_config.intention_size
        print(
            "Overwriting rescale factor from checkpoint from {} to {}".format(
                cfg.walker_config.rescale_factor,
                cfg_loaded.walker_config.rescale_factor,
            )
        )
        cfg.walker_config.rescale_factor = cfg_loaded.walker_config.rescale_factor
        cfg.env_config.env_args.rescale_factor = cfg_loaded.walker_config.rescale_factor

    # Initialize checkpoint manager
    mgr_options = ocp.CheckpointManagerOptions(
        create=True,
        max_to_keep=cfg.train_setup["checkpoint_max_to_keep"],
        keep_period=cfg.train_setup["checkpoint_keep_period"],
        step_prefix="PPONetwork",
    )

    ckpt_mgr = ocp.CheckpointManager(checkpoint_path, options=mgr_options)

    logging.info(f"run_id: {run_id}")
    logging.info(f"Training checkpoint path: {checkpoint_path}")
    print(cfg)
    ppo_params = cfg.train_setup.train_config
    env_config = cfg.env_config.env_args

    # Create environment based on task_name
    task_name = cfg.env_config.task_name
    if task_name == "maze_forage":
        env = maze_forage.MazeForage(config_overrides=env_config)
        evaluator_env = maze_forage.MazeForage(config_overrides=env_config)
    elif task_name == "bowl_escape":
        env = bowl_escape.BowlEscape(config_overrides=env_config)
        evaluator_env = bowl_escape.BowlEscape(config_overrides=env_config)
    elif task_name == "flat_arena":
        env = flat_arena.FlatWalk(config_overrides=env_config)
        evaluator_env = flat_arena.FlatWalk(config_overrides=env_config)
    else:
        raise ValueError(
            f"Unknown task_name: {task_name}. Must be one of: maze_forage, bowl_escape, flat_arena"
        )

    train_fn = functools.partial(
        ppo.train,
        **ppo_params,
        num_evals=int(
            cfg.train_setup.train_config.num_timesteps / cfg.train_setup.eval_every
        ),
        num_resets_per_eval=cfg.train_setup.eval_every // cfg.train_setup.reset_every,
        kl_weight=cfg.network_config.kl_weight,
        network_factory=functools.partial(
            ppo_networks.make_intention_ppo_networks,
            encoder_hidden_layer_sizes=tuple(cfg.network_config.encoder_layer_sizes),
            decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
            value_hidden_layer_sizes=tuple(cfg.network_config.critic_layer_sizes),
            intention_latent_size=cfg.network_config.intention_size,
        ),
        ckpt_mgr=ckpt_mgr,
        checkpoint_to_restore=cfg.train_setup.checkpoint_to_restore,
        freeze_decoder=cfg.train_setup.freeze_decoder,
        config_dict=OmegaConf.to_container(cfg, resolve=True),  # finalize config here
        use_kl_schedule=cfg.network_config.kl_schedule,
    )

    run_id = f"{cfg.logging_config.exp_name}_{cfg.env_config.env_name}_{cfg.env_config.task_name}_{run_id}"
    wandb.init(
        project=cfg.logging_config.project_name,
        config=OmegaConf.to_container(cfg, resolve=True, structured_config_mode=True),
        notes=f"{cfg.logging_config.notes}",
        id=run_id,
        resume="allow",
        group=cfg.logging_config.group_name,
    )

    def wandb_progress(num_steps, metrics):
        metrics["num_steps_thousands"] = num_steps
        wandb.log(metrics, commit=False)

    # # define the jit reset/step functions
    jit_reset = jax.jit(evaluator_env.reset)
    jit_step = jax.jit(evaluator_env.step)
    mj_model = evaluator_env.mj_model
    mj_data = mujoco.MjData(mj_model)
    scene_option = mujoco.MjvOption()
    scene_option.sitegroup[:] = [1, 1, 1, 1, 1, 0]
    renderer = mujoco.Renderer(mj_model, height=512, width=512)
    policy_params_fn = functools.partial(
        wandb_logging.rollout_logging_fn,
        evaluator_env,
        jit_reset,
        jit_step,
        cfg,
        checkpoint_path,
        renderer,
        mj_model,
        mj_data,
        scene_option,
    )

    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=wandb_progress,
        policy_params_fn=policy_params_fn,
    )


if __name__ == "__main__":
    main()