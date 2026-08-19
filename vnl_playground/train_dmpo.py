"""VNL-playground DMPO training entry — all downstream tasks.

For DMPO on rodent imitation use ``track_mjx.train_dmpo``. This entry
covers downstream tasks: gap-running, vision, vision-scratch, prior-
decoder transfer, etc. Reads ``cfg.transfer.mode`` to pick between:
  - ``""``: flat-obs registry env (run_gap, basic locomotion).
  - ``"prior_decoder"``: load frozen prior + decoder, train high-level
    policy via PriorHighLevelWrapper.
  - ``"from_scratch"``: end-to-end vision policy with EndToEndWrapper.

Usage:
    python -m vnl_playground.train_dmpo --config-name=rodent_run_gap/run_gap_vision_scratch_position
"""
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

from track_mjx.agent.dmpo.checkpoint import (
    make_checkpointer,
    restore as restore_ckpt,
    save as save_ckpt,
)
from track_mjx.agent.dmpo.config import (
    DMPOConfig,
    realized_ratios,
    resolve_sgd_steps_per_rollout,
)
from track_mjx.agent.dmpo.learner import init_training_state, make_optimizers
from track_mjx.agent.dmpo.networks import make_dmpo_networks
from track_mjx.agent.dmpo.networks_vision import make_dmpo_vision_networks
from track_mjx.agent.dmpo.networks_vision_scratch import (
    make_dmpo_vision_scratch_networks,
)
from track_mjx.agent.dmpo.replay import make_replay
from track_mjx.agent.dmpo.train import (
    _VnlPlaygroundEnvAdapter,
    _filter_dmpo_kwargs,
)
from track_mjx.agent.dmpo.train_dmpo_eval import (
    compute_batch_rollout_metrics,
    compute_rollout_metrics,
    compute_vision_sensitivity,
    render_eval_video,
    run_eval_rollout_envzero,
)
from track_mjx.agent.dmpo.schedules import env_steps_estimate, reward_anneal_lambda
from track_mjx.agent.dmpo.train_dmpo_logging import (
    detect_git_sha,
    load_wandb_state as load_dmpo_wandb_state,
    make_run_id,
    save_wandb_state as save_dmpo_wandb_state,
)
from track_mjx.agent.dmpo.training_loop import run as run_training_loop

log = logging.getLogger(__name__)

try:
    import wandb
    _WANDB_IMPORTED = True
except ImportError:
    _WANDB_IMPORTED = False
    wandb = None  # type: ignore


_VALID_TRANSFER_MODES = ("", "prior_decoder", "from_scratch")


def _load_env_and_networks(hydra_cfg: DictConfig, cfg: DMPOConfig):
    """Set up env and networks based on cfg.transfer.mode.

    Returns ``(env, base_env, mj_model, mjx_model, env_spec,
    transition_template, nets, vision_shape_or_none)``.
    """
    transfer_mode = str(hydra_cfg.get("transfer", {}).get("mode", ""))
    if transfer_mode not in _VALID_TRANSFER_MODES:
        raise ValueError(
            f"Unknown transfer.mode={transfer_mode!r}; "
            f"expected one of {_VALID_TRANSFER_MODES}"
        )
    use_prior_decoder = transfer_mode == "prior_decoder"
    use_from_scratch = transfer_mode == "from_scratch"
    use_vision = use_prior_decoder or use_from_scratch
    env_name = str(hydra_cfg.get("env_name", "RodentRunGap"))

    if use_vision:
        from vnl_playground import tasks
        from vnl_playground.tasks.wrappers import (
            PriorHighLevelWrapper,
            EndToEndWrapper,
        )
        from vnl_playground.tasks.prior_utils import (
            load_prior_checkpoint,
            make_decoder_inference_fn as make_prior_decoder_fn,
            make_prior_inference_fn,
        )

        if use_prior_decoder:
            prior_ckpt_path = str(hydra_cfg.transfer.prior_checkpoint_path)
            prior_ckpt_step = hydra_cfg.transfer.get("prior_checkpoint_step", None)
            (
                _enc_params,
                prior_params,
                decoder_params,
                normalizer_params,
                prior_cfg,
            ) = load_prior_checkpoint(prior_ckpt_path, prior_ckpt_step)
            latent_size = int(prior_cfg["network_config"]["intention_size"])
            prior_fn = make_prior_inference_fn(prior_params, normalizer_params, prior_cfg)
            decoder_fn = make_prior_decoder_fn(decoder_params, normalizer_params, prior_cfg)
        else:
            prior_fn = None
            decoder_fn = None
            latent_size = None

        env_args = OmegaConf.to_container(hydra_cfg.get("env_config", {}), resolve=True) or {}
        env_args = {k: v for k, v in env_args.items() if k not in ("env_name", "flatten_obs")}
        base_env = tasks.load(env_name, flatten_obs=False, config_overrides=env_args)
        raw_env = base_env
        mj_model = getattr(raw_env, "mj_model", None) or getattr(
            getattr(raw_env, "env", None), "mj_model", None
        )
        mjx_model = getattr(raw_env, "mjx_model", None) or getattr(
            getattr(raw_env, "env", None), "mjx_model", None
        )
        if mj_model is None or mjx_model is None:
            raise RuntimeError("Could not find mj_model/mjx_model on base env for vision rendering")
        n_eye_actuators = getattr(
            base_env.env if hasattr(base_env, "env") else base_env,
            "n_eye_actuators", 0,
        )

        if use_from_scratch:
            base_env = EndToEndWrapper(
                base_env,
                highlvl_obs_key=str(hydra_cfg.transfer.get("highlvl_obs_key", "task_obs")),
                decoder_obs_key=str(hydra_cfg.transfer.get("decoder_obs_key", "proprioception")),
            )
        elif use_prior_decoder:
            base_env = PriorHighLevelWrapper(
                base_env, prior_fn, decoder_fn, latent_size,
                highlvl_obs_key=str(hydra_cfg.transfer.get("highlvl_obs_key", "task_obs")),
                decoder_obs_key=str(hydra_cfg.transfer.get("decoder_obs_key", "proprioception")),
                pass_vision=True,
                pass_task_obs=True,
                deterministic_prior=bool(hydra_cfg.transfer.get("deterministic_prior", True)),
                noise_logvar=float(hydra_cfg.transfer.get("noise_logvar", -2.0)),
                n_eye_actuators=n_eye_actuators,
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
            base_env, episode_length=episode_length, action_repeat=action_repeat,
            full_reset=False,
        )
        vision_width = int(hydra_cfg.env_config.get("vision_width", 32))
        vision_height = int(hydra_cfg.env_config.get("vision_height", 32))
        grayscale = bool(hydra_cfg.env_config.get("grayscale", True))
        left_camera = str(hydra_cfg.env_config.get("left_camera_name", "eye_left-rodent"))
        right_camera = str(hydra_cfg.env_config.get("right_camera_name", "eye_right-rodent"))
        base_env = BinocularVisionRenderWrapper(
            base_env, mj_model=mj_model, mjx_model=mjx_model,
            width=vision_width, height=vision_height, grayscale=grayscale,
            left_camera_name=left_camera, right_camera_name=right_camera,
            render_depth=False,
            use_textures=bool(hydra_cfg.env_config.get("use_textures", False)),
            use_shadows=bool(hydra_cfg.env_config.get("use_shadows", False)),
            eye_dropout_rate=float(hydra_cfg.env_config.get("eye_dropout_rate", 0.0)),
            eval_eye_mode=str(hydra_cfg.env_config.get("eval_eye_mode", "binocular")),
        )

        env = _VnlPlaygroundEnvAdapter(base_env, pre_batched=True)
        obs_size_dict = dict(env.observation_size)
        action_size = int(env.action_size)
        vision_shape = tuple(getattr(
            base_env, "vision_shape",
            getattr(base_env.env, "vision_shape", (32, 32, 2)),
        ))
        proprio_size = int(obs_size_dict.get("proprioception", 0))
        if use_from_scratch and proprio_size == 0:
            raise RuntimeError(
                "from_scratch requires non-zero proprio (EndToEndWrapper should expose it)."
            )
        proprio_template = jnp.zeros((proprio_size,), dtype=jnp.float32)
        env_spec = {
            "obs_template": {
                "vision": jnp.zeros(vision_shape, dtype=jnp.float32),
                "imitation_target": jnp.zeros(
                    (obs_size_dict.get("imitation_target", 0),), dtype=jnp.float32
                ),
                "proprioception": proprio_template,
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
        if use_from_scratch:
            nets = make_dmpo_vision_scratch_networks(
                task_obs_size=obs_size_dict["imitation_target"],
                proprio_size=proprio_size,
                action_size=action_size,
                vision_shape=vision_shape,
                cfg=cfg,
                cnn_feature_size=int(hydra_cfg.network_config.get("vision_feature_size", 32)),
                cnn_channels=tuple(hydra_cfg.network_config.get("vision_channels", [4, 8, 16, 32])),
                mono_channels=1 if hydra_cfg.env_config.get("grayscale", True) else 3,
                shared_weights=hydra_cfg.network_config.get("binocular_mode", "shared") == "shared",
            )
        else:
            nets = make_dmpo_vision_networks(
                task_obs_size=obs_size_dict["imitation_target"],
                action_size=action_size,
                vision_shape=vision_shape,
                cfg=cfg,
                cnn_feature_size=int(hydra_cfg.network_config.get("vision_feature_size", 32)),
                cnn_channels=tuple(hydra_cfg.network_config.get("vision_channels", [4, 8, 16, 32])),
                mono_channels=1 if hydra_cfg.env_config.get("grayscale", True) else 3,
                shared_weights=hydra_cfg.network_config.get("binocular_mode", "shared") == "shared",
            )
        return env, base_env, mj_model, mjx_model, env_spec, transition_template, nets, vision_shape

    # Flat-obs path (transfer.mode == ""): vnl-playground registry
    from vnl_playground import registry as vp_registry
    raw_env = vp_registry.load(env_name)
    env = _VnlPlaygroundEnvAdapter(raw_env)
    obs_size = int(env.observation_size)
    action_size = int(env.action_size)
    env_spec = {"obs_size": obs_size, "action_size": action_size}
    transition_template = {
        "observation": jnp.zeros((obs_size,), dtype=jnp.float32),
        "action": jnp.zeros((action_size,), dtype=jnp.float32),
        "reward": jnp.zeros((), dtype=jnp.float32),
        "discount": jnp.zeros((), dtype=jnp.float32),
        "next_observation": jnp.zeros((obs_size,), dtype=jnp.float32),
    }
    nets = make_dmpo_networks(obs_size, action_size, cfg)
    return env, raw_env, None, None, env_spec, transition_template, nets, None


@hydra.main(config_path="config", config_name="rodent_run_gap_dmpo/vision_scratch_position",
            version_base=None)
def main(hydra_cfg: DictConfig):
    """DMPO entry for VNL downstream tasks (gap, vision, etc.)."""
    raw_train_cfg = OmegaConf.to_container(hydra_cfg.train_config, resolve=True)
    cfg = DMPOConfig(**_filter_dmpo_kwargs(raw_train_cfg))
    iters_per_chunk = int(hydra_cfg.train_config.get("iters_per_chunk", 32))
    cfg_dict = OmegaConf.to_container(hydra_cfg, resolve=True)
    seed = int(hydra_cfg.get("seed", 0))
    rng = jax.random.PRNGKey(seed)

    config_name = str(
        hydra_cfg.get("logging_config", {}).get("exp_name", hydra_cfg.get("env_name", "dmpo-vnl"))
    )
    git_sha = detect_git_sha(Path(__file__).resolve().parents[1])
    run_id = make_run_id(config_name, seed, git_sha)
    log.info("wandb run_id=%s", run_id)

    ckpt_dir = str(hydra_cfg.get("checkpoint_dir", "./checkpoints/dmpo_vnl"))
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    existing = load_dmpo_wandb_state(ckpt_dir)
    if _WANDB_IMPORTED:
        try:
            wandb.init(
                project=str(hydra_cfg.get("logging_config", {}).get("project_name", "dmpo-vnl")),
                config=cfg_dict, mode=os.environ.get("WANDB_MODE", "online"),
                id=existing["wandb_run_id"] if existing else run_id,
                name=existing["wandb_run_id"] if existing else run_id,
                resume="must" if existing else "allow",
                group=str(hydra_cfg.get("logging_config", {}).get(
                    "group_name", hydra_cfg.get("env_name", "dmpo")
                )),
                notes=str(hydra_cfg.get("logging_config", {}).get("notes", "")),
                reinit=True,
            )
            save_dmpo_wandb_state(ckpt_dir,
                                  run_id if not existing else existing["wandb_run_id"])
        except Exception as e:
            log.warning("wandb.init failed (%s); continuing without wandb.", e)

    env, base_env, mj_model, mjx_model, env_spec, transition_template, nets, vision_shape = (
        _load_env_and_networks(hydra_cfg, cfg)
    )
    rng, k_state = jax.random.split(rng)
    state = init_training_state(k_state, nets, env_spec, cfg)
    optimizers = make_optimizers(cfg)

    ckpt_mgr = make_checkpointer(ckpt_dir)
    restored = restore_ckpt(ckpt_mgr, state_template=state)
    if restored is not None:
        log.info("Restored DMPO checkpoint at training step %d", int(restored.steps))
        state = restored

    rb = make_replay(
        max_size=max(cfg.sequence_length + 1, cfg.max_replay_size // cfg.num_envs),
        min_size=max(cfg.sequence_length + 1, cfg.min_replay_size // cfg.num_envs),
        sequence_length=cfg.sequence_length,
        sample_batch_size=cfg.batch_size, add_batch_size=cfg.num_envs, period=1,
    )
    rb_state = rb.init(transition_template)

    K = resolve_sgd_steps_per_rollout(cfg)
    log.info(
        "DMPO downstream: K=%d SGD updates per rollout | %s",
        K,
        " ".join(f"{k}={v:.4g}" for k, v in realized_ratios(cfg, K).items()),
    )

    def wandb_log_cb(payload: dict, env_steps: int) -> None:
        if _WANDB_IMPORTED and wandb is not None and wandb.run is not None:
            wandb.log(payload, step=int(env_steps))

    def ckpt_save_cb(state: object, env_steps: int) -> None:
        save_ckpt(ckpt_mgr, int(env_steps), state, config=cfg_dict)

    use_vision = mj_model is not None
    erc_enable = bool(hydra_cfg.get("eval_render_config", {}).get("enable", True))

    def eval_cb(state: object, env_steps: int, k_eval: jax.Array) -> None:
        if not erc_enable:
            return
        try:
            erc_raw = hydra_cfg.get("eval_render_config", {})
            erc = OmegaConf.to_container(erc_raw, resolve=True) if erc_raw else {}
            eval_episode_length = int(erc.get("episode_length", 1000))
            # 4-tuple since 2026-08-19: [2] is the UN-remixed all-env batch metric
            # dict, [3] is the raw allenv arrays. We recompute the batch metrics
            # from allenv with the reward-anneal lambda so the reported reward is
            # the one the replay buffer stored (see remix_eval_reward).
            rollout, term_events, _batch_rm_unmixed, allenv = run_eval_rollout_envzero(
                env=base_env if use_vision else env,
                policy_apply=nets.policy.apply,
                params=state.policy_params,
                rng=k_eval,
                episode_length=eval_episode_length,
                num_envs=cfg.num_envs,
            )
            # lambda from state.steps via the SAME expression the fused training
            # step uses (train_dmpo_step.py:98-99) -- the host env_steps int is a
            # different quantity and would drift from the buffer.
            remix_key = getattr(cfg, "reward_anneal_sparse_key", None)
            remix_lambda = (
                float(reward_anneal_lambda(env_steps_estimate(state.steps, cfg, K), cfg))
                if remix_key is not None
                else None
            )
            if rollout and _WANDB_IMPORTED and wandb is not None and wandb.run is not None:
                rollout_metrics = compute_rollout_metrics(rollout, remix_key, remix_lambda)
                rollout_metrics.update(
                    compute_batch_rollout_metrics(allenv, remix_key, remix_lambda)
                )
                wandb.log({f"eval/{k}": v for k, v in rollout_metrics.items()}
                          | {"env_steps": int(env_steps)},
                          step=int(env_steps), commit=False)
            if rollout and "vision" in rollout[0].obs:
                mid = len(rollout) // 2
                sens = compute_vision_sensitivity(
                    nets.policy.apply, state.policy_params, rollout[mid].obs, k_eval,
                )
                if _WANDB_IMPORTED and wandb is not None and wandb.run is not None:
                    wandb.log({"eval/vision_sensitivity": sens, "env_steps": int(env_steps)},
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
                    reward_remix=(
                        {"sparse_key": remix_key, "lambda": remix_lambda}
                        if remix_key is not None
                        else None
                    ),
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
    log.info("Training complete: final policy_loss=%.4g critic_loss=%.4g",
             float(last_metrics.get("policy_loss", 0.0)),
             float(last_metrics.get("critic_loss", 0.0)))
    if _WANDB_IMPORTED and wandb is not None and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
