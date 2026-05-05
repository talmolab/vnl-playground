"""DMPO + B-aggressive trainable prior+decoder + action-anchor entry."""
from __future__ import annotations

import logging
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import hydra
import jax
import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf

import vnl_playground.naccdmax_patch  # noqa: F401

from track_mjx.agent.dmpo.checkpoint import (
    make_checkpointer,
    restore as restore_ckpt,
    save as save_ckpt,
)
from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.learner import init_training_state
from track_mjx.agent.dmpo.networks_kl_anchor import make_dmpo_kl_anchor_networks
from track_mjx.agent.dmpo.optim_kl_anchor import make_kl_anchor_optimizers
from track_mjx.agent.dmpo.replay import make_replay
from track_mjx.agent.dmpo.train import (
    _VnlPlaygroundEnvAdapter,
    _filter_dmpo_kwargs,
)
from track_mjx.agent.dmpo.train_dmpo_eval import (
    compute_rollout_metrics,
    render_eval_video,
    run_eval_rollout_envzero,
)
from track_mjx.agent.dmpo.train_dmpo_logging import (
    detect_git_sha,
    load_wandb_state as load_dmpo_wandb_state,
    make_run_id,
    save_wandb_state as save_dmpo_wandb_state,
)
from track_mjx.agent.dmpo.training_loop import run as run_training_loop

from vnl_playground import tasks
from vnl_playground.tasks.wrappers_kl_anchor import KLAnchorPriorDecoderWrapper
from vnl_playground.tasks.prior_utils import (
    load_prior_checkpoint,
    make_decoder_inference_fn as make_prior_decoder_fn,
    make_prior_inference_fn,
)

log = logging.getLogger(__name__)

try:
    import wandb
    _WANDB_IMPORTED = True
except ImportError:
    _WANDB_IMPORTED = False
    wandb = None  # type: ignore


def _build_env(hydra_cfg, prior_fn, decoder_fn, latent_size, action_size):
    """Load registry env, wrap in KLAnchorPriorDecoderWrapper +
    brax wrap_for_brax_training + BinocularVisionRenderWrapper.
    """
    env_name = str(hydra_cfg.env_name)
    env_args = OmegaConf.to_container(hydra_cfg.get("env_config", {}), resolve=True) or {}
    env_args = {k: v for k, v in env_args.items() if k not in ("env_name", "flatten_obs")}
    valid_keys = set(tasks.get_default_config(env_name).keys())
    env_args = {k: v for k, v in env_args.items() if k in valid_keys}
    base_env = tasks.load(env_name, flatten_obs=False, config_overrides=env_args)
    raw_env = base_env.env if hasattr(base_env, "env") else base_env
    mj_model = getattr(raw_env, "mj_model", None)
    mjx_model = getattr(raw_env, "mjx_model", None)
    if mj_model is None or mjx_model is None:
        raise RuntimeError("Could not find mj_model/mjx_model on base env for vision rendering")

    base_env = KLAnchorPriorDecoderWrapper(
        base_env,
        prior_fn=prior_fn,
        decoder_fn=decoder_fn,
        action_size=action_size,
        w_anchor=float(hydra_cfg.kl_anchor.w_anchor),
        alpha_anchor=float(hydra_cfg.kl_anchor.alpha_anchor),
    )

    from mujoco_playground._src import wrapper as mp_wrapper
    from vnl_playground.tasks.rodent.vision_jax import BinocularVisionRenderWrapper

    episode_length = int(
        hydra_cfg.env_config.get(
            "episode_length", hydra_cfg.train_config.get("unroll_length", 1000)
        )
    )
    action_repeat = int(hydra_cfg.env_config.get("action_repeat", 1))
    base_env = mp_wrapper.wrap_for_brax_training(
        base_env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        full_reset=False,
    )
    base_env = BinocularVisionRenderWrapper(
        base_env,
        mj_model=mj_model,
        mjx_model=mjx_model,
        width=int(hydra_cfg.env_config.get("vision_width", 32)),
        height=int(hydra_cfg.env_config.get("vision_height", 32)),
        grayscale=bool(hydra_cfg.env_config.get("grayscale", True)),
        left_camera_name=str(hydra_cfg.env_config.get("left_camera_name", "eye_left-rodent")),
        right_camera_name=str(hydra_cfg.env_config.get("right_camera_name", "eye_right-rodent")),
        render_depth=False,
        use_textures=bool(hydra_cfg.env_config.get("use_textures", False)),
        use_shadows=bool(hydra_cfg.env_config.get("use_shadows", False)),
        eye_dropout_rate=float(hydra_cfg.env_config.get("eye_dropout_rate", 0.0)),
        eval_eye_mode=str(hydra_cfg.env_config.get("eval_eye_mode", "binocular")),
    )
    env_adapter = _VnlPlaygroundEnvAdapter(base_env, pre_batched=True)
    vision_shape = tuple(
        getattr(base_env, "vision_shape",
                getattr(base_env.env, "vision_shape", (32, 32, 2)))
    )
    return env_adapter, base_env, mj_model, mjx_model, vision_shape


@hydra.main(
    config_path="config",
    config_name="rodent_run_gap_dmpo/velocity_only_kl_anchor",
    version_base=None,
)
def main(hydra_cfg: DictConfig):
    raw_train_cfg = OmegaConf.to_container(hydra_cfg.train_config, resolve=True)
    cfg = DMPOConfig(**_filter_dmpo_kwargs(raw_train_cfg))
    iters_per_chunk = int(hydra_cfg.train_config.get("iters_per_chunk", 32))
    cfg_dict = OmegaConf.to_container(hydra_cfg, resolve=True)
    seed = int(hydra_cfg.get("seed", 0))
    rng = jax.random.PRNGKey(seed)

    config_name = str(
        hydra_cfg.get("logging_config", {}).get("exp_name", "dmpo-kl-anchor")
    )
    git_sha = detect_git_sha(Path(__file__).resolve().parents[1])
    run_id = make_run_id(config_name, seed, git_sha)
    log.info("wandb run_id=%s", run_id)

    ckpt_dir = str(hydra_cfg.get("checkpoint_dir", "./checkpoints/dmpo_kl_anchor"))
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    existing = load_dmpo_wandb_state(ckpt_dir)
    if _WANDB_IMPORTED:
        try:
            wandb.init(
                project=str(hydra_cfg.get("logging_config", {}).get("project_name", "dmpo-rodent")),
                config=cfg_dict, mode=os.environ.get("WANDB_MODE", "online"),
                id=existing["wandb_run_id"] if existing else run_id,
                name=existing["wandb_run_id"] if existing else run_id,
                resume="must" if existing else "allow",
                group=str(hydra_cfg.get("logging_config", {}).get("group_name", "kl-anchor")),
                notes=str(hydra_cfg.get("logging_config", {}).get("notes", "")),
                reinit=True,
            )
            save_dmpo_wandb_state(ckpt_dir, run_id if not existing else existing["wandb_run_id"])
        except Exception as e:
            log.warning("wandb.init failed (%s); continuing without wandb.", e)

    # --- 1. Load frozen prior + decoder ---
    prior_ckpt_path = str(hydra_cfg.transfer.prior_checkpoint_path)
    prior_ckpt_step = hydra_cfg.transfer.get("prior_checkpoint_step", None)
    log.info("Loading prior checkpoint from: %s", prior_ckpt_path)
    (
        _enc_params,
        prior_params,
        decoder_params,
        normalizer_params,
        prior_cfg,
    ) = load_prior_checkpoint(prior_ckpt_path, prior_ckpt_step)
    latent_size = int(prior_cfg["network_config"]["intention_size"])
    prior_layer_sizes = tuple(
        prior_cfg["network_config"].get("prior_layer_sizes", [1024, 1024])
    )
    decoder_layer_sizes = tuple(prior_cfg["network_config"]["decoder_layer_sizes"])
    log.info("Prior loaded. intention_size=%d", latent_size)

    prior_fn = make_prior_inference_fn(prior_params, normalizer_params, prior_cfg)
    decoder_fn = make_prior_decoder_fn(decoder_params, normalizer_params, prior_cfg)

    # --- 2. Build env ---
    base_env_for_size = tasks.load(str(hydra_cfg.env_name), flatten_obs=False)
    action_size_base = int(base_env_for_size.action_size)
    env, base_env_wrapped, mj_model, mjx_model, vision_shape = _build_env(
        hydra_cfg, prior_fn, decoder_fn, latent_size, action_size_base,
    )
    obs_size_dict = dict(env.observation_size)
    proprio_size = int(obs_size_dict.get("proprioception", 0))
    task_obs_size = int(obs_size_dict.get("imitation_target", 0))
    action_size = int(env.action_size)
    log.info(
        "env_spec: action_size=%d, proprio=%d, task_obs=%d, vision_shape=%s, latent=%d",
        action_size, proprio_size, task_obs_size, vision_shape, latent_size,
    )

    # --- 3. Build B-aggressive networks (warm-started) ---
    nets = make_dmpo_kl_anchor_networks(
        proprio_size=proprio_size,
        task_obs_size=task_obs_size,
        action_size=action_size,
        latent_size=latent_size,
        vision_shape=tuple(vision_shape),
        cfg=cfg,
        prior_layer_sizes=prior_layer_sizes,
        decoder_layer_sizes=decoder_layer_sizes,
        policy_head_layer_sizes=tuple(
            hydra_cfg.network_config.get("policy_head_layer_sizes", [256, 256, 256])
        ),
        cnn_feature_size=int(hydra_cfg.network_config.get("vision_feature_size", 32)),
        cnn_channels=tuple(hydra_cfg.network_config.get("vision_channels", [4, 8, 16, 32])),
        mono_channels=1 if hydra_cfg.env_config.get("grayscale", True) else 3,
        shared_weights=hydra_cfg.network_config.get("binocular_mode", "shared") == "shared",
        value_hidden_layer_sizes=tuple(
            hydra_cfg.network_config.get("value_hidden_layer_sizes", [512, 512, 512, 512])
        ),
        warm_start_prior_params=prior_params,
        warm_start_decoder_params=decoder_params,
    )

    # --- 4. Asymmetric optimizers ---
    optimizers = make_kl_anchor_optimizers(
        cfg,
        prior_lr_mult=float(hydra_cfg.kl_anchor.get("prior_lr_mult", 0.1)),
        decoder_lr_mult=float(hydra_cfg.kl_anchor.get("decoder_lr_mult", 1.0)),
        policy_head_lr_mult=float(hydra_cfg.kl_anchor.get("policy_head_lr_mult", 1.0)),
    )

    # --- 5. State + replay ---
    env_spec = {
        "obs_template": {
            "vision": jnp.zeros(tuple(vision_shape), dtype=jnp.float32),
            "imitation_target": jnp.zeros((task_obs_size,), dtype=jnp.float32),
            "proprioception": jnp.zeros((proprio_size,), dtype=jnp.float32),
        },
        "action_size": action_size,
    }
    transition_template = {
        "observation": env_spec["obs_template"],
        "action": jnp.zeros((action_size,), dtype=jnp.float32),
        "reward": jnp.zeros((), dtype=jnp.float32),
        "discount": jnp.zeros((), dtype=jnp.float32),
        "next_observation": env_spec["obs_template"],
    }
    rng, k_state = jax.random.split(rng)
    state = init_training_state(k_state, nets, env_spec, cfg)
    # init_training_state builds state with the DEFAULT optimizer; replace
    # the policy_opt_state with one initialized by the kl-anchor multi_transform
    # optimizer so update() can resolve per-block inner_states.
    pol_opt_kl, _, _ = optimizers
    state = state._replace(policy_opt_state=pol_opt_kl.init(state.policy_params))

    ckpt_mgr = make_checkpointer(ckpt_dir)
    restored = restore_ckpt(ckpt_mgr, state_template=state)
    if restored is not None:
        log.info("Restored DMPO checkpoint at step %d", int(restored.steps))
        state = restored

    rb = make_replay(
        max_size=max(cfg.sequence_length + 1, cfg.max_replay_size // cfg.num_envs),
        min_size=max(cfg.sequence_length + 1, cfg.min_replay_size // cfg.num_envs),
        sequence_length=cfg.sequence_length,
        sample_batch_size=cfg.batch_size,
        add_batch_size=cfg.num_envs,
        period=1,
    )
    rb_state = rb.init(transition_template)

    K = max(1, int(cfg.unroll_length * cfg.num_envs / (cfg.batch_size * cfg.samples_per_insert)))
    log.info("DMPO kl-anchor: K=%d SGD updates per rollout", K)

    # --- 6. Callbacks ---
    def wandb_log_cb(payload, env_steps):
        if _WANDB_IMPORTED and wandb is not None and wandb.run is not None:
            wandb.log(payload, step=int(env_steps))

    def ckpt_save_cb(state, env_steps):
        save_ckpt(ckpt_mgr, int(env_steps), state, config=cfg_dict)

    erc_enable = bool(hydra_cfg.get("eval_render_config", {}).get("enable", True))

    def eval_cb(state, env_steps, k_eval):
        if not erc_enable:
            return
        try:
            erc_raw = hydra_cfg.get("eval_render_config", {})
            erc = OmegaConf.to_container(erc_raw, resolve=True) if erc_raw else {}
            eval_episode_length = int(erc.get("episode_length", 1000))
            rollout, term_events = run_eval_rollout_envzero(
                env=base_env_wrapped,
                policy_apply=nets.policy.apply,
                params=state.policy_params,
                rng=k_eval,
                episode_length=eval_episode_length,
                num_envs=cfg.num_envs,
            )
            if rollout and _WANDB_IMPORTED and wandb is not None and wandb.run is not None:
                rm = compute_rollout_metrics(rollout)
                wandb.log({f"eval/{k}": v for k, v in rm.items()} | {"env_steps": int(env_steps)},
                          step=int(env_steps), commit=False)
            if mj_model is not None and rollout:
                video_path = Path(ckpt_dir) / f"eval_{int(env_steps)}.mp4"
                hud_cfg = erc.get("hud") if isinstance(erc, dict) else None
                rew_cfg_raw = hydra_cfg.get("env_config", {}).get("reward_terms", None)
                rew_cfg = (OmegaConf.to_container(rew_cfg_raw, resolve=True)
                           if rew_cfg_raw is not None else None)
                render_eval_video(
                    rollout, mj_model, video_path,
                    fps=int(erc.get("fps", 50)),
                    height=int(erc.get("height", 480)),
                    width=int(erc.get("width", 640)),
                    camera=str(erc.get("camera", "close_profile-rodent")),
                    hud_config=hud_cfg, reward_config=rew_cfg,
                    termination_events=term_events,
                )
                if _WANDB_IMPORTED and wandb is not None and wandb.run is not None:
                    wandb.log({"videos/eval": wandb.Video(str(video_path), format="mp4")},
                              step=int(env_steps))
        except Exception as e:
            log.warning("Eval render failed: %s", e, exc_info=True)
        finally:
            import gc
            gc.collect()

    rng, k_run = jax.random.split(rng)
    state, env_state, rb_state, last_metrics = run_training_loop(
        env=env, nets=nets, optimizers=optimizers, rb=rb, cfg=cfg,
        K=K, iters_per_chunk=iters_per_chunk, rng=k_run,
        state=state, env_state=None, rb_state=rb_state,
        eval_callback=eval_cb,
        wandb_log_callback=wandb_log_cb,
        ckpt_mgr=ckpt_mgr, ckpt_save_callback=ckpt_save_cb,
        cfg_dict=cfg_dict,
    )

    ckpt_mgr.wait_until_finished()
    log.info(
        "Training complete: final policy_loss=%.4g critic_loss=%.4g",
        float(last_metrics.get("policy_loss", 0.0)),
        float(last_metrics.get("critic_loss", 0.0)),
    )
    if _WANDB_IMPORTED and wandb is not None and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
