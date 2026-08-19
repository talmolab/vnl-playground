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
from track_mjx.agent.dmpo.config import (
    DMPOConfig,
    realized_ratios,
    resolve_sgd_steps_per_rollout,
)
from track_mjx.agent.dmpo.learner import init_training_state
from track_mjx.agent.dmpo.networks_kl_anchor import make_dmpo_kl_anchor_networks
from track_mjx.agent.dmpo.normalizer_seeding import seed_proprio_from_imit
from track_mjx.agent.dmpo.optim_kl_anchor import make_kl_anchor_optimizers
from track_mjx.agent.dmpo.replay import make_replay
from track_mjx.agent.dmpo.train import (
    _VnlPlaygroundEnvAdapter,
    _filter_dmpo_kwargs,
)
from track_mjx.agent.dmpo.train_dmpo_eval import (
    compute_batch_rollout_metrics,
    compute_rollout_metrics,
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

from vnl_playground import tasks
from vnl_playground.tasks.wrappers_kl_anchor import KLAnchorPriorDecoderWrapper
from vnl_playground.tasks.prior_utils import (
    load_prior_checkpoint,
    make_decoder_logits_fn,
    make_prior_inference_fn,
)

log = logging.getLogger(__name__)

try:
    import wandb
    _WANDB_IMPORTED = True
except ImportError:
    _WANDB_IMPORTED = False
    wandb = None  # type: ignore


def _build_env(hydra_cfg, prior_fn, decoder_logits_fn, latent_size, action_size):
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
        decoder_logits_fn=decoder_logits_fn,
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
    # Optional per-episode info reset. full_reset=False above leaves
    # state.info untouched across auto-resets, which turns info-derived
    # rewards (gap_crossing_bonus via info["gaps_crossed"]) into cross-episode
    # ratchets during training — see wrappers_info_reset.py for the full
    # mechanism. Opt-in via a top-level `wrappers:` block so that completed
    # arms (m1..m8) stay bit-reproducible.
    wrap_cfg = hydra_cfg.get("wrappers", None)
    if wrap_cfg is not None and bool(wrap_cfg.get("info_reset_on_done", False)):
        from vnl_playground.tasks.wrappers_info_reset import (
            DEFAULT_RUN_GAP_KEYS,
            InfoResetOnDoneWrapper,
        )

        keys = tuple(wrap_cfg.get("info_reset_keys", None) or DEFAULT_RUN_GAP_KEYS)
        base_env = InfoResetOnDoneWrapper(base_env, keys=keys)
        log.info(
            "InfoResetOnDoneWrapper ACTIVE: restoring info keys %s to their "
            "reset-time values on every done (fixes the full_reset=False "
            "gap_crossing_bonus ratchet)", keys,
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
    # Populate the loss-side KL-anchor coefficients on cfg so _policy_loss_fn
    # picks them up. DMPOConfig is a dataclass; mutate in place.
    cfg.kl_anchor_alpha = float(hydra_cfg.kl_anchor.alpha_anchor)
    cfg.kl_anchor_w = float(hydra_cfg.kl_anchor.w_anchor)
    cfg.kl_anchor_w_floor = float(hydra_cfg.kl_anchor.get("w_anchor_floor", 0.0))
    cfg.kl_anchor_beta_linear = float(
        hydra_cfg.kl_anchor.get("beta_linear", 0.0)
    )
    cfg.kl_anchor_decay_sgd_steps = int(
        hydra_cfg.kl_anchor.get("decay_sgd_steps", 0)
    )
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
    decoder_logits_fn = make_decoder_logits_fn(decoder_params, normalizer_params, prior_cfg)

    # --- 2. Build env ---
    base_env_for_size = tasks.load(str(hydra_cfg.env_name), flatten_obs=False)
    action_size_base = int(base_env_for_size.action_size)
    env, base_env_wrapped, mj_model, mjx_model, vision_shape = _build_env(
        hydra_cfg, prior_fn, decoder_logits_fn, latent_size, action_size_base,
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
        residual_mode=str(hydra_cfg.network_config.get("residual_mode", "sigma_tanh")),
        residual_scale=float(hydra_cfg.network_config.get("residual_scale", 2.0)),
        critic_use_proprio=bool(
            hydra_cfg.network_config.get("critic_use_proprio", False)
        ),
    )
    log.info(
        "latent residual: mode=%s scale=%.3g | critic_use_proprio=%s",
        str(hydra_cfg.network_config.get("residual_mode", "sigma_tanh")),
        float(hydra_cfg.network_config.get("residual_scale", 2.0)),
        bool(hydra_cfg.network_config.get("critic_use_proprio", False)),
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
    # Replay schema, shaped by the compression flags (see DMPOConfig).
    obs_template = env_spec["obs_template"]
    if bool(getattr(cfg, "vision_uint8_storage", False)):
        if not (isinstance(obs_template, dict) and "vision" in obs_template):
            raise ValueError(
                "vision_uint8_storage=true but the obs template has no "
                f"'vision' key: {type(obs_template).__name__}"
            )
        obs_template = {
            **obs_template,
            "vision": jnp.zeros(obs_template["vision"].shape, dtype=jnp.uint8),
        }
    transition_template = {
        "observation": obs_template,
        "action": jnp.zeros((action_size,), dtype=jnp.float32),
        "reward": jnp.zeros((), dtype=jnp.float32),
        "discount": jnp.zeros((), dtype=jnp.float32),
        "anchor_mu_imit": jnp.zeros((action_size,), dtype=jnp.float32),
        "anchor_log_std_imit": jnp.zeros((action_size,), dtype=jnp.float32),
    }
    if bool(getattr(cfg, "store_next_observation", True)):
        transition_template["next_observation"] = obs_template
    else:
        eff_n = min(int(cfg.n_step), int(cfg.sequence_length) - 1)
        log.info(
            "Compressed replay schema: next_observation dropped "
            "(bootstrap from observation[:, n]; effective n-step = "
            "min(n_step=%d, sequence_length-1=%d) = %d), vision_uint8=%s",
            int(cfg.n_step), int(cfg.sequence_length) - 1, eff_n,
            bool(getattr(cfg, "vision_uint8_storage", False)),
        )
    # Per-transition storage cost, for buffer sizing against GPU memory.
    _tx_bytes = sum(
        x.size * x.dtype.itemsize for x in jax.tree.leaves(transition_template)
    )
    log.info(
        "Replay transition size: %.2f KB -> max_replay_size=%d is %.2f GB "
        "(%d steps/env deep at num_envs=%d = %.1f rollouts)",
        _tx_bytes / 1024, cfg.max_replay_size,
        _tx_bytes * (cfg.max_replay_size // cfg.num_envs) * cfg.num_envs / 2**30,
        cfg.max_replay_size // cfg.num_envs, cfg.num_envs,
        (cfg.max_replay_size // cfg.num_envs) / cfg.unroll_length,
    )
    rng, k_state = jax.random.split(rng)
    state = init_training_state(k_state, nets, env_spec, cfg)
    # init_training_state builds state with the DEFAULT optimizer; replace
    # the policy_opt_state with one initialized by the kl-anchor multi_transform
    # optimizer so update() can resolve per-block inner_states.
    pol_opt_kl, _, _ = optimizers
    state = state._replace(policy_opt_state=pol_opt_kl.init(state.policy_params))
    # Seed DMPO's running-stats normalizer with the imit checkpoint's proprio
    # stats. Without this, the warm-started prior+decoder receive un-normalized
    # proprio at step 0 and produce garbage — breaking the warm-start invariant
    # that the entire B-aggressive design depends on. Done BEFORE restore_ckpt
    # so a resumed run will overwrite this with its own (already-seeded-and-
    # updated) normalizer from disk.
    state = state._replace(
        normalizer_params=seed_proprio_from_imit(
            state.normalizer_params, normalizer_params,
        )
    )
    log.info(
        "Seeded DMPO normalizer with imit proprio stats: mean=%.3f±%.3f std=%.3f±%.3f",
        float(state.normalizer_params.proprioception.mean.mean()),
        float(state.normalizer_params.proprioception.mean.std()),
        float(state.normalizer_params.proprioception.std.mean()),
        float(state.normalizer_params.proprioception.std.std()),
    )

    # ---- DMPO warm start from ANOTHER run's checkpoint ------------------------
    # transfer.warm_start_dmpo_checkpoint points at a DMPONetwork_<step> dir of a
    # previous DMPO run. Grafts policy_params + target_policy_params +
    # normalizer_params into the fresh state (critic/duals/optimizers stay
    # fresh -- the critic may have a different atom count than the source run,
    # and DMPOConfig.critic_warmup_sgd_steps exists precisely so the fresh
    # critic can fit before the warm policy is allowed to move). The grafted
    # policy also becomes frozen_behavior_params for behavior mixing
    # (cfg.behavior_mix_init). Done BEFORE restore_ckpt so that resuming a
    # warm-started run restores its own state over the graft, and AFTER the
    # normalizer seeding, which it supersedes (the source run's normalizer is
    # the converged version of the seeded one).
    frozen_behavior_params = None
    ws_path = None
    if "transfer" in hydra_cfg:
        ws_path = hydra_cfg.transfer.get("warm_start_dmpo_checkpoint", None)
    if ws_path:
        import flax.serialization as _flax_ser
        from track_mjx.agent.dmpo.checkpoint import load_train_state_items_numpy

        ws = load_train_state_items_numpy(str(ws_path))
        ws_policy = _flax_ser.from_state_dict(state.policy_params, ws["policy_params"])
        ws_target = _flax_ser.from_state_dict(
            state.target_policy_params, ws["target_policy_params"]
        )
        ws_norm = _flax_ser.from_state_dict(
            state.normalizer_params, ws["normalizer_params"]
        )
        state = state._replace(
            policy_params=ws_policy,
            target_policy_params=ws_target,
            normalizer_params=ws_norm,
        )
        # Optimizer state must be re-initialized on the grafted params (Adam
        # moments start at zero for the trainable blocks, as intended).
        state = state._replace(policy_opt_state=pol_opt_kl.init(state.policy_params))
        frozen_behavior_params = ws_policy
        n_params = sum(x.size for x in jax.tree.leaves(ws_policy))
        log.info(
            "DMPO WARM START: grafted policy(+target)+normalizer from %s "
            "(%d policy params). Critic/duals/optimizers are FRESH; "
            "critic_warmup_sgd_steps=%d behavior_mix_init=%.2f",
            ws_path, n_params,
            int(getattr(cfg, "critic_warmup_sgd_steps", 0)),
            float(getattr(cfg, "behavior_mix_init", 0.0)),
        )
    elif float(getattr(cfg, "behavior_mix_init", 0.0)) > 0.0:
        raise ValueError(
            "train_config.behavior_mix_init > 0 requires "
            "transfer.warm_start_dmpo_checkpoint (the frozen behavior policy "
            "is the warm-start policy)."
        )

    ckpt_mgr = make_checkpointer(ckpt_dir)
    restored = restore_ckpt(ckpt_mgr, state_template=state)
    # Checkpoints are saved at `step=total_env_steps`, so the manager's latest
    # step IS the env-step count to resume the training-loop counter from. Without
    # this the loop restarted counting at 0 and a resumed run would train
    # `num_timesteps` MORE steps under colliding checkpoint names.
    start_env_steps = 0
    if restored is not None:
        start_env_steps = int(ckpt_mgr.latest_step() or 0)
        log.info(
            "Restored DMPO checkpoint: sgd_step=%d env_steps=%d (resuming counter)",
            int(restored.steps),
            start_env_steps,
        )
        state = restored

    # ---- Startup invariant probe -------------------------------------------------
    # Run a single env step with the warm-started policy and log the wrapper's
    # anchor metrics. This is a tripwire: a fresh kl-anchor pipeline with
    # working warm-start should observe r_anchor very close to 1.0 (the policy's
    # mode action equals tanh(mu_imit_pretanh) up to numerics). If r_anchor drops,
    # one of the warm-start fixes (Tasks 1-6) regressed.
    try:
        from track_mjx.agent.dmpo.learner import _normalize_obs
        from track_mjx.agent.dmpo.kl_anchor_utils import pretanh_gaussian_kl
        probe_label = (
            "anchor_invariant_probe_resumed"
            if restored is not None
            else "anchor_invariant_probe"
        )
        if ws_path and restored is None:
            # A DMPO-warm-started policy head has learned away from the
            # anchor BY DESIGN -- r_anchor < 1 here is expected, not a
            # regression of the prior/decoder warm-start splice.
            probe_label = "anchor_invariant_probe_dmpo_warmstart(r_anchor<1 expected)"
        rng_probe, k_probe = jax.random.split(rng)
        keys = jax.random.split(k_probe, cfg.num_envs)
        st0 = env.reset(keys)
        norm_obs = _normalize_obs(st0.obs, state.normalizer_params)
        # Online policy distribution at the spawn obs.
        dist0 = jax.vmap(lambda o: nets.policy.apply(state.policy_params, o))(norm_obs)
        mu_theta = dist0.mean()
        log_std_theta = jnp.log(dist0.stddev())
        # Anchor distribution from state.info — populated by the wrapper.
        # Both wrapper and loss compute log_std as log(softplus(raw)+1e-3) so the
        # KL on this side is in the same units as the loss-side anchor signal.
        mu_imit = st0.info["anchor_mu_imit"]
        log_std_imit = st0.info["anchor_log_std_imit"]
        kl = pretanh_gaussian_kl(mu_theta, log_std_theta, mu_imit, log_std_imit)
        # kl is per-sample shape (num_envs,); aggregate via mean(exp(-w*kl))
        # which matches the loss-side r_anchor formula.
        r_anchor = float(jnp.mean(jnp.exp(-cfg.kl_anchor_w * kl)))
        kl_mean = float(jnp.mean(kl))
        # Format keeps `action_mse=` for smoke-test regex compat (post-port
        # the wrapper's MSE diagnostic is no longer relevant for the probe;
        # we report kl_mean here as the meaningful diagnostic).
        log.info(
            "%s r_anchor=%.4f action_mse=%.4f kl_mean=%.4f",
            probe_label, r_anchor, 0.0, kl_mean,
        )
        rng = rng_probe
    except Exception as exc:
        log.warning("Startup invariant probe failed (non-fatal): %s", exc, exc_info=True)
    # ----------------------------------------------------------------------------

    rb = make_replay(
        max_size=max(cfg.sequence_length + 1, cfg.max_replay_size // cfg.num_envs),
        min_size=max(cfg.sequence_length + 1, cfg.min_replay_size // cfg.num_envs),
        sequence_length=cfg.sequence_length,
        sample_batch_size=cfg.batch_size,
        add_batch_size=cfg.num_envs,
        period=1,
    )
    rb_state = rb.init(transition_template)

    # SGD updates per rollout. `sgd_steps_per_rollout`, when set, pins K directly;
    # otherwise fall back to the historical expression so every completed arm stays
    # bit-reproducible. NOTE that historical expression DIVIDES by
    # samples_per_insert, inverting the Acme/Reverb meaning of the knob -- see the
    # long note on DMPOConfig.sgd_steps_per_rollout. Do not "fix" it in place.
    K = resolve_sgd_steps_per_rollout(cfg)
    log.info(
        "K=%d SGD updates/rollout via %s | %s "
        "(Ray reference that SOLVES this task: realized_samples_per_insert=3.236)",
        K,
        "sgd_steps_per_rollout (explicit; samples_per_insert is unread)"
        if cfg.sgd_steps_per_rollout is not None
        else f"legacy inverted formula from samples_per_insert={cfg.samples_per_insert}",
        " ".join(f"{k}={v:.4g}" for k, v in realized_ratios(cfg, K).items()),
    )
    log.info("DMPO kl-anchor: K=%d SGD updates per rollout", K)

    # --- 6. Callbacks ---
    def wandb_log_cb(payload, env_steps):
        if _WANDB_IMPORTED and wandb is not None and wandb.run is not None:
            wandb.log(payload, step=int(env_steps))
        # Also mirror the anchor metric to stdout so non-wandb runs (smoke
        # tests, offline) can verify the warm-start invariant without
        # parsing wandb's binary log.
        if "anchor/r_anchor" in payload:
            log.info(
                "anchor_metrics env_steps=%d r_anchor=%.4f action_mse=%.4f",
                int(env_steps),
                float(payload.get("anchor/r_anchor", 0.0)),
                float(payload.get("anchor/action_mse", 0.0)),
            )

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
            rollout, term_events, _batch_rm_unmixed, allenv = run_eval_rollout_envzero(
                env=base_env_wrapped,
                policy_apply=nets.policy.apply,
                params=state.policy_params,
                rng=k_eval,
                episode_length=eval_episode_length,
                num_envs=cfg.num_envs,
                normalizer_params=state.normalizer_params,
            )
            # Rebuild the reward the replay buffer actually stored. lambda comes
            # from state.steps via the SAME expression the fused training step
            # uses (train_dmpo_step.py:98-99) -- the host env_steps int is a
            # different quantity and would drift from the buffer.
            remix_key = getattr(cfg, "reward_anneal_sparse_key", None)
            remix_lambda = (
                float(reward_anneal_lambda(env_steps_estimate(state.steps, cfg, K), cfg))
                if remix_key is not None
                else None
            )
            if rollout:
                rm = compute_rollout_metrics(rollout, remix_key, remix_lambda)
                # All-env estimator. `rm` is the legacy env-0 statistic, kept so
                # the new arms stay comparable to the 08-11 runs; `batch_rm` is
                # the one to actually judge arms on (env-0 had sd 35.3 over the
                # baseline's own flat window -- see compute_batch_rollout_metrics).
                rm.update(compute_batch_rollout_metrics(allenv, remix_key, remix_lambda))
                # Mirror eval metrics to stdout. Previously these were computed
                # ONLY inside the wandb guard and written ONLY into wandb's
                # binary log -- which is why earlier sessions could not answer
                # "did it ever cross a gap?" from the run logs, and why
                # total_gap_crossings had to be recovered by parsing .wandb
                # protobufs after the fact.
                log.info(
                    "eval env_steps=%d %s",
                    int(env_steps),
                    " ".join(
                        f"{k}={v:.6g}" if isinstance(v, (int, float)) else f"{k}={v}"
                        for k, v in sorted(rm.items())
                    ),
                )
                if _WANDB_IMPORTED and wandb is not None and wandb.run is not None:
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
        extra_state_extras=("anchor_mu_imit", "anchor_log_std_imit"),
        start_env_steps=start_env_steps,
        frozen_behavior_params=frozen_behavior_params,
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
