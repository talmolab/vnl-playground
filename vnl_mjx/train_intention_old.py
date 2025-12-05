"""
Entry point for track-mjx. Load the config file, create environments, initialize network, and start training.
"""

import os
import sys

import jax 
import jax.numpy as jp
import numpy as np

from flax.training import orbax_utils

import imageio

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

from brax.training.acme import running_statistics

from vnl_mjx.tasks.rodent import flat_arena, bowl_escape, maze_forage

from track_mjx.agent import checkpointing
from track_mjx.agent import wandb_logging
from track_mjx.analysis import render

warnings.filterwarnings("ignore", category=DeprecationWarning)


@hydra.main(version_base=None, config_path="config", config_name="flat_arena_transfer")
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
    
    network_factory = functools.partial(
            ppo_networks.make_intention_ppo_networks,
            encoder_hidden_layer_sizes=tuple(cfg.network_config.encoder_layer_sizes),
            decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
            value_hidden_layer_sizes=tuple(cfg.network_config.critic_layer_sizes),
            intention_latent_size=cfg.network_config.intention_size,
        )

    train_fn = functools.partial(
        ppo.train,
        **ppo_params,
        num_evals=int(
            cfg.train_setup.train_config.num_timesteps / cfg.train_setup.eval_every
        ),
        num_resets_per_eval=cfg.train_setup.eval_every // cfg.train_setup.reset_every,
        kl_weight=cfg.network_config.kl_weight,
        network_factory=network_factory,
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

    training_params = dict(ppo_params)
    del training_params["network_factory"]
    del training_params["eval_every"]

    normalize = lambda x, y: x
    if training_params["normalize_observations"]:
        normalize = running_statistics.normalize

    def make_logging_inference_fn(ppo_networks):
        """Creates params and inference function for the PPO agent.
        The policy takes the params as an input, so different sets of params can be used.
        """

        def make_logging_policy(deterministic=False):
            policy_network = ppo_networks.policy_network
            # can modify this to provide stochastic action + noise
            parametric_action_distribution = ppo_networks.parametric_action_distribution

            def logging_policy(
                params,
                observations,
                key_sample,
            ):
                param_subset = (params[0], params[1])

                #policy_params = params.policy
                logits = policy_network.apply(*param_subset, observations)
                # logits comes from policy directly, raw predictions that decoder generates (action, intention_mean, intention_logvar)
                if deterministic:
                    actions = parametric_action_distribution.postprocess(parametric_action_distribution.mode(logits)) #post processing to bound actions similar to stochastic
                    return (
                        jp.array(actions),
                        {},
                    )
                # action sampling is happening here, according to distribution parameter logits
                raw_actions = parametric_action_distribution.sample_no_postprocessing(
                    logits, key_sample
                )
                # probability of selection specific action, actions with higher reward should have higher probability
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

    def wandb_progress(num_steps, metrics):
        metrics["num_steps_thousands"] = num_steps
        wandb.log(metrics, commit=False)

    # # define the jit reset/step functions
    jit_reset = jax.jit(evaluator_env.reset)
    jit_step = jax.jit(evaluator_env.step)
    rng = jax.random.PRNGKey(0)
    start_state = jit_reset(rng)
    mj_model = evaluator_env.mj_model
    mj_data = mujoco.MjData(mj_model)
    scene_option = mujoco.MjvOption()
    scene_option.sitegroup[:] = [1, 1, 1, 1, 1, 0]
    renderer = mujoco.Renderer(mj_model, height=512, width=512)

    ppo_network = network_factory(
        start_state.obs.shape[-1],
        env.action_size,
        preprocess_observations_fn=normalize,
    )

    make_logging_policy = make_logging_inference_fn(ppo_network)
    jit_logging_inference_fn = jax.jit(make_logging_policy(deterministic=True))

    #policy_params_fn = functools.partial(
    #    wandb_logging.rollout_logging_fn,
    #    evaluator_env,
    #    jit_reset,
    #    jit_step,
    #    cfg,
    #    checkpoint_path,
    #    renderer,
    #    mj_model,
    #    mj_data,
    #    scene_option,
    #)

    def policy_params_fn(current_step, make_policy, params, jit_logging_inference_fn):
        del make_policy  # Unused.

        # generate a rollout
        rollout = [start_state]
        state = start_state
        rng = jax.random.PRNGKey(0)
        for _ in range(ppo_params.episode_length):
            _, rng = jax.random.split(rng)
            action, _ = jit_logging_inference_fn(params, state.obs, rng)
            state = jit_step(state, action)
            rollout.append(state)

        # render and log
        qposes_rollout = np.array([state.data.qpos for state in rollout])
        video_path = f"{checkpoint_path}/{current_step}.mp4"

        with imageio.get_writer(video_path, fps=int((1.0 / env.dt))) as video:
            for qpos in qposes_rollout:
                mj_data.qpos = qpos
                mujoco.mj_forward(mj_model, mj_data)
                renderer.update_scene(
                    mj_data,
                    camera="close_profile-rodent",
                )
                video.append_data(renderer.render())

        # don't commit because progress_fn is called after
        wandb.log({"eval/rollout": wandb.Video(video_path, format="mp4")}, commit=False)
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        path = checkpoint_path / f"{current_step}"
        orbax_checkpointer.save(path, params, force=True, save_args=save_args)

    make_inference_fn, params, _ = train_fn(
        environment=env,
        eval_env = evaluator_env,
        progress_fn=wandb_progress,
        policy_params_fn=functools.partial(
            policy_params_fn, jit_logging_inference_fn=jit_logging_inference_fn),
    )


if __name__ == "__main__":
    main()