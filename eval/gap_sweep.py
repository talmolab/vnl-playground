"""Sweep fixed gap width and measure crossing-success rate for a trained
RodentJumpGapTrial policy.

Reconstructs the EXACT training env + network pipeline (shared_vision_task_obs,
prior_decoder transfer, batched warp vision renderer), restores the Orbax
checkpoint (normalizer + policy params), and runs deterministic parallel
rollouts at each fixed gap width.

Success (per episode) = state.info['gaps_crossed'] reaches >= 1 at any point
before the episode terminates (falling AFTER crossing is NOT a failure).

Usage:
    python eval/gap_sweep.py                 # full sweep
    python eval/gap_sweep.py --widths 0.05   # single width (debug)
    python eval/gap_sweep.py --n_envs 256 --explore
"""
import argparse
import json
import os
import functools

import numpy as np

# --- paths -----------------------------------------------------------------
CKPT = "/root/vast/keming/SalkResearch/runs/0706_jgt_basic_3_22/260706_225509"
PRIOR = "/root/vast/keming/gapjump/prior/combined_jump"
RUN_CFG_JSON = os.path.join(CKPT, "config.json")
OUT_DIR = "/root/vast/keming/SalkResearch/eval"
# Single-gap PROBE knobs (for the "did it learn to DECIDE" verification):
PLATFORM_LEN = None   # if set (m), pin the probe landing-platform length; else use trained config's
# STRICT landing criterion (a real landing, not a marginal lunge that then falls in):
Z_STAND = 0.045       # torso z (m) above this = standing on the platform (sinking-into-gap is ~0.02-0.03)
K_HOLD  = 15          # must hold "on platform AND standing" for >= this many consecutive steps
PROBE_SINGLE = True   # force n_platforms=1 so each episode is ONE controlled gap+platform to score


def build_pipeline(gap_w, n_eye_default=0):
    """Build (batched vision env, jit_reset, jit_step, base_env) for gap width gap_w."""
    import jax
    import jax.numpy as jp
    from omegaconf import OmegaConf
    from mujoco_playground import wrapper as mp_wrapper
    from vnl_playground import tasks
    from vnl_playground.tasks.wrappers import PriorHighLevelWrapper
    from vnl_playground.tasks.rodent.vision_jax import VisionRenderWrapper
    from vnl_playground.tasks.prior_utils import (
        load_prior_checkpoint,
        make_decoder_inference_fn,
        make_prior_inference_fn,
    )

    with open(RUN_CFG_JSON) as f:
        cfg = OmegaConf.create(json.load(f))

    # --- prior (frozen) ----------------------------------------------------
    (_enc, prior_params, dec_params, norm_params, prior_cfg) = load_prior_checkpoint(
        PRIOR
    )
    latent_size = prior_cfg["network_config"]["intention_size"]
    prior_fn = make_prior_inference_fn(prior_params, norm_params, prior_cfg)
    decoder_policy_fn = make_decoder_inference_fn(dec_params, norm_params, prior_cfg)

    # --- env args (mirror train_highlvl.main) ------------------------------
    env_name = cfg.env_config.env_name
    env_args = OmegaConf.to_container(cfg.env_config.get("env_args", {}), resolve=True)
    # enforce ctrl_dt from prior checkpoint
    env_args["ctrl_dt"] = float(prior_cfg["env_config"]["ctrl_dt"])
    # pass vision config
    for vk in ("vision_width", "vision_height", "grayscale", "binocular"):
        if vk in cfg.env_config:
            env_args[vk] = cfg.env_config[vk]
    # PIN the gap width.
    # Go/no-go checkpoints (unjumpable_prob>0) sample gaps bimodally from
    # jumpable_gap_range/unjumpable_gap_range at reset and IGNORE gap_length_range
    # (see run_gap.py reset: `elif self._unjumpable_prob > 0.0`). The gap_gradient
    # mode likewise ignores it. To force a true fixed width on EVERY checkpoint we
    # disable both special modes and pin all three ranges to gap_w, so reset falls
    # into the plain fixed-width `else` branch regardless of how it was trained.
    env_args["gap_length_range"] = [float(gap_w), float(gap_w)]
    env_args["jumpable_gap_range"] = [float(gap_w), float(gap_w)]
    env_args["unjumpable_gap_range"] = [float(gap_w), float(gap_w)]
    env_args["unjumpable_prob"] = 0.0
    env_args["gap_gradient"] = False
    env_args["randomize_gaps"] = True
    # Single-gap PROBE: one gap + one (optionally narrow) landing platform, so each
    # episode is ONE scorable decision -- even when the checkpoint was trained on a corridor.
    if PROBE_SINGLE:
        env_args["n_platforms"] = 1
    if PLATFORM_LEN is not None:
        env_args["platform_length_range"] = [float(PLATFORM_LEN), float(PLATFORM_LEN)]

    env = tasks.load(env_name, flatten_obs=False, config_overrides=env_args)

    # n_eye_actuators from base env
    _u = env
    while hasattr(_u, "env"):
        _u = _u.env
    n_eye_actuators = getattr(_u, "n_eye_actuators", n_eye_default)
    base_env = _u

    env = PriorHighLevelWrapper(
        env,
        prior_fn,
        decoder_policy_fn,
        latent_size,
        highlvl_obs_key=cfg.transfer.get("highlvl_obs_key", "task_obs"),
        decoder_obs_key=cfg.transfer.get("decoder_obs_key", "proprioception"),
        pass_vision=True,
        pass_task_obs=True,
        deterministic_prior=cfg.transfer.get("deterministic_prior", True),
        noise_logvar=cfg.transfer.get("noise_logvar", -2.0),
        n_eye_actuators=n_eye_actuators,
    )

    # raw env for vision renderer
    raw = env
    while hasattr(raw, "env"):
        raw = raw.env

    episode_length = int(cfg.train_setup.train_config.episode_length)
    brax_env = mp_wrapper.wrap_for_brax_training(
        env,
        episode_length=episode_length,
        action_repeat=int(cfg.env_config.env_args.get("action_repeat", 1)),
        randomization_fn=None,
        full_reset=False,
    )
    # Match training's vision rendering EXACTLY: binocular checkpoints render two
    # eye cameras (stereo, 2*C channels); monocular ones render a single camera.
    if bool(cfg.env_config.get("binocular", False)):
        from vnl_playground.tasks.rodent.vision_jax import BinocularVisionRenderWrapper
        vision_env = BinocularVisionRenderWrapper(
            brax_env,
            mj_model=raw.mj_model,
            mjx_model=raw.mjx_model,
            width=int(cfg.env_config.get("vision_width", 32)),
            height=int(cfg.env_config.get("vision_height", 32)),
            grayscale=bool(cfg.env_config.get("grayscale", True)),
            left_camera_name=cfg.env_config.get("left_camera_name", "eye_left-rodent"),
            right_camera_name=cfg.env_config.get("right_camera_name", "eye_right-rodent"),
            render_depth=bool(cfg.env_config.get("render_depth", False)),
            use_textures=bool(cfg.env_config.get("use_textures", False)),
            use_shadows=bool(cfg.env_config.get("use_shadows", False)),
            eye_dropout_rate=float(cfg.env_config.get("eye_dropout_rate", 0.0)),
            eval_eye_mode=cfg.env_config.get("eval_eye_mode", "binocular"),
        )
    else:
        vision_env = VisionRenderWrapper(
            brax_env,
            mj_model=raw.mj_model,
            mjx_model=raw.mjx_model,
            width=int(cfg.env_config.get("vision_width", 32)),
            height=int(cfg.env_config.get("vision_height", 32)),
            grayscale=bool(cfg.env_config.get("grayscale", True)),
            camera_name=cfg.env_config.get("vision_camera_name", "egocentric-rodent"),
            render_depth=bool(cfg.env_config.get("render_depth", False)),
            use_textures=bool(cfg.env_config.get("use_textures", False)),
            use_shadows=bool(cfg.env_config.get("use_shadows", False)),
        )

    jit_reset = jax.jit(vision_env.reset)
    jit_step = jax.jit(vision_env.step)

    import mujoco
    torso_bid = mujoco.mj_name2id(raw.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso-rodent")
    # four paw bodies, for the STRICT four-paw landing criterion
    paw_bids = np.array([
        mujoco.mj_name2id(raw.mj_model, mujoco.mjtObj.mjOBJ_BODY, f"{b}-rodent")
        for b in ("hand_L", "hand_R", "foot_L", "foot_R")
    ])
    return vision_env, jit_reset, jit_step, base_env, episode_length, torso_bid, paw_bids


def build_policy():
    """Restore checkpoint (normalizer + policy) and build deterministic policy."""
    import jax
    from track_mjx.agent import checkpointing
    from track_mjx.agent.ff_ppo import ppo_networks as ff_ppo_networks

    from orbax import checkpoint as ocp

    cfg_ck = checkpointing.load_config_from_checkpoint(CKPT)
    ppo_network = checkpointing.make_ppo_network_from_cfg(cfg_ck)

    # NB: load_policy() fails on this config because proprioception has size 0.
    # Zero-sized arrays are stored as scalar sentinels (see checkpointing.save);
    # we must apply the same replace/restore transform used by load_training_state.
    abstract_policy = checkpointing.make_abstract_policy(cfg_ck)
    mgr_opts = ocp.CheckpointManagerOptions(create=False, step_prefix="PPONetwork")
    with ocp.CheckpointManager(CKPT, options=mgr_opts) as mgr:
        step = mgr.latest_step()
        print(f"  restoring policy from step {step}")
        restored = mgr.restore(
            step,
            args=ocp.args.Composite(
                policy=ocp.args.StandardRestore(
                    checkpointing._replace_zero_sized_arrays(abstract_policy)
                )
            ),
        )["policy"]
    params = checkpointing._restore_zero_sized_arrays(restored, abstract_policy)
    make_policy = ff_ppo_networks.make_inference_fn(ppo_network)
    policy = make_policy(params, deterministic=True)
    return jax.jit(policy)


def realized_gap_size(state, base_env):
    """Compute per-env realized gap size (m) from platform geometry."""
    import jax.numpy as jp

    plat_centers = state.data.xpos[:, base_env._platform_body_ids, 0]  # (n, nplat)
    landing_near_edge = plat_centers[:, 0] - base_env._platform_half_length
    gap = landing_near_edge - base_env._start_platform_half_length
    return np.asarray(gap)


def run_width(gap_w, n_envs, seed, jit_policy=None, verbose=False):
    import jax
    import jax.numpy as jp

    vision_env, jit_reset, jit_step, base_env, episode_length, torso_bid, paw_bids = build_pipeline(gap_w)
    if jit_policy is None:
        jit_policy = build_policy()

    key = jax.random.PRNGKey(seed)
    key, rk = jax.random.split(key)
    reset_keys = jax.random.split(rk, n_envs)
    state = jit_reset(reset_keys)

    if verbose:
        print("  obs keys:", list(state.obs.keys()))
        print("  info keys:", list(state.info.keys()))
        for k, v in state.obs.items():
            print("   obs", k, getattr(v, "shape", None))

    gap_realized = realized_gap_size(state, base_env)
    # Landing platform span [near, far] per env (fixed after reset) -- for the DECISION metrics.
    plat_center = np.asarray(state.data.xpos[:, base_env._platform_body_ids, 0][:, 0])
    near_edge = plat_center - float(base_env._platform_half_length)
    far_edge = plat_center + float(base_env._platform_half_length)

    n = n_envs
    ever_crossed = np.zeros(n, dtype=bool)
    active = np.ones(n, dtype=bool)
    max_gaps = np.zeros(n, dtype=np.int32)
    ep_len = np.zeros(n, dtype=np.int32)
    fell_after_cross = np.zeros(n, dtype=bool)
    landed_ok = np.zeros(n, dtype=bool)      # STRICT: stood on platform (z>Z_STAND) held >= K_HOLD steps
    landing_off = np.full(n, np.nan)         # torso_x - near_edge at the moment it counts as landed
    stand_run = np.zeros(n, dtype=np.int32)  # consecutive steps standing on the platform
    fell_term = np.zeros(n, dtype=bool)      # env's OWN terminations/fallen (ground-truth plunge)
    # --- Go/no-go DECISION quadrant flags (read straight from env info) --------
    #   over_unjumpable_void: torso committed out over a gap wider than jump_limit
    #     => a jump it CANNOT make (false-go). Fires off geometry (gap vs jump_limit),
    #     so it is valid even under the fixed-width override where unjumpable_prob=0.
    #   nogo_hold_done: HELD a stop at an un-jumpable edge for nogo_hold_steps
    #     (== the reached_unjumpable_edge success termination) => correct no-go.
    ever_void = np.zeros(n, dtype=bool)   # ever nosed past an un-jumpable edge (includes recovered nose-overs)
    ever_hold = np.zeros(n, dtype=bool)   # ever completed a held no-go stop

    for t in range(episode_length):
        key, ak = jax.random.split(key)
        act, _ = jit_policy(state.obs, ak)
        state = jit_step(state, act)
        gaps = np.asarray(state.info["gaps_crossed"])
        done = np.asarray(state.done).astype(bool)
        fallen_now = (np.asarray(state.metrics["terminations/fallen"]) > 0.5
                      if "terminations/fallen" in state.metrics else np.zeros(n, dtype=bool))
        void_now = np.asarray(state.info.get("over_unjumpable_void", False)).astype(bool)
        hold_now = np.asarray(state.info.get("nogo_hold_done", False)).astype(bool)

        crossed_now = gaps >= 1
        # count crossing only while the episode is still on its first life
        newly = active & crossed_now & (~ever_crossed)
        ever_crossed |= active & crossed_now
        # Ground-truth PLUNGE = the env's fallen termination BEFORE crossing the gap.
        # (A torso-z threshold misses it: the episode auto-resets on the fallen step
        # before z samples below the 0.01 line. Gating on ~ever_crossed excludes the
        # irrelevant walk-off-the-far-end fall AFTER a successful crossing.)
        fell_term |= active & fallen_now & (~ever_crossed)
        ever_void |= active & void_now
        ever_hold |= active & hold_now
        max_gaps = np.maximum(max_gaps, np.where(active, gaps, 0))
        ep_len += active.astype(np.int32)
        # a fall (done) on an env that had already crossed => fell_after_cross
        fell_after_cross |= active & done & ever_crossed

        # --- DECISION metrics: did it land ON the narrow platform, and where? ---
        torso_x = np.asarray(state.data.xpos[:, torso_bid, 0])
        torso_z = np.asarray(state.data.xpos[:, torso_bid, 2])
        # STRICT FOUR-PAW landing: ALL 4 paws past the landing near edge (fully on
        # the platform, not a torso fly-over) AND torso at standing height (not a
        # draped/sinking lunge) -- held for K_HOLD steps.
        paw_x = np.asarray(state.data.xpos[:, paw_bids, 0])          # (n, 4)
        all_paws_on = (paw_x > near_edge[:, None]).all(axis=1)
        stand_now = (active & all_paws_on & (torso_z > Z_STAND))
        stand_run = np.where(stand_now, stand_run + 1, 0)
        newly = (stand_run >= K_HOLD) & (~landed_ok)
        landing_off = np.where(newly & np.isnan(landing_off), torso_x - near_edge, landing_off)
        landed_ok |= (stand_run >= K_HOLD)

        active &= ~done
        if not active.any():
            break

    res = {
        "gap_width_m": float(gap_w),
        "gap_width_cm": round(float(gap_w) * 100, 2),
        "gap_size_realized_m_mean": float(np.mean(gap_realized)),
        "gap_size_realized_m_std": float(np.std(gap_realized)),
        "n_envs": int(n),
        "crossing_rate": float(np.mean(ever_crossed)),
        # Go/no-go decision quadrants (see loop): false-go and correct-no-go rates.
        "over_unjumpable_void_rate": float(np.mean(ever_void)),
        "nogo_hold_rate": float(np.mean(ever_hold)),  # env's own success (WARNING: nogo_hold_steps differs per ckpt)
        # --- FAIR, threshold-independent outcome partition (sums to 1) ------------
        #   fell:      torso dropped below the fallen threshold (0.01) = a real plunge.
        #   landed:    crossed the gap and stood on the far platform.
        #   safe_stop: neither crossed nor fell = stayed up on the near side (refused/stopped).
        # At un-jumpable widths safe_stop = correct no-go (regardless of dwell length);
        # at jumpable widths safe_stop = timid refusal of a doable jump.
        "fell_rate": float(np.mean(fell_term)),
        "safe_stop_rate": float(np.mean((~ever_crossed) & (~fell_term))),
        "fell_after_cross_rate": float(np.mean(fell_after_cross)),
        "mean_gaps_crossed": float(np.mean(max_gaps)),
        "mean_ep_len": float(np.mean(ep_len)),
        "steps_run": int(t + 1),
        # DECISION metrics (the "did it learn to decide" verification):
        #   landed_on_platform_rate: fraction that landed ON the narrow platform.
        #     A FIXED jump succeeds only in a narrow band of gap widths (overshoots small,
        #     undershoots large); an ADAPTIVE (deciding) policy stays high across widths.
        #   mean_landing_offset_cm: where on the platform it lands (torso_x - near_edge).
        #     FIXED jump -> offset slides down with gap width; ADAPTIVE -> stays ~flat.
        "landed_on_platform_rate": float(np.mean(landed_ok)),
        "mean_landing_offset_cm": (
            float(np.nanmean(landing_off) * 100) if np.any(~np.isnan(landing_off)) else None
        ),
        "platform_len_m": (float(PLATFORM_LEN) if PLATFORM_LEN is not None else None),
    }
    if verbose:
        print("  result:", json.dumps(res, indent=2))
    return res, jit_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=float, nargs="*", default=None)
    ap.add_argument("--n_envs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--explore", action="store_true")
    ap.add_argument("--out", type=str, default=os.path.join(OUT_DIR, "gap_sweep_results.json"))
    ap.add_argument("--ckpt", type=str, default=None,
                    help="checkpoint manager dir to evaluate (default: basic run)")
    ap.add_argument("--platform_len", type=float, default=None,
                    help="pin the probe landing-platform length (m); default uses trained config")
    args = ap.parse_args()

    global CKPT, RUN_CFG_JSON, PLATFORM_LEN
    if args.ckpt is not None:
        CKPT = args.ckpt
        RUN_CFG_JSON = os.path.join(CKPT, "config.json")
        print(f"[gap_sweep] evaluating checkpoint dir: {CKPT}")
    if args.platform_len is not None:
        PLATFORM_LEN = args.platform_len
        print(f"[gap_sweep] probe landing-platform length pinned to {PLATFORM_LEN} m")

    if args.widths is not None:
        widths = args.widths
    else:
        widths = [0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.22]

    jit_policy = build_policy()

    results = []
    for i, w in enumerate(widths):
        print(f"[width {w:.3f} m ({w*100:.1f} cm)] running {args.n_envs} envs ...", flush=True)
        res, jit_policy = run_width(
            w, args.n_envs, args.seed, jit_policy=jit_policy,
            verbose=args.explore and i == 0,
        )
        _off = res['mean_landing_offset_cm']
        print(
            f"  -> crossing_rate={res['crossing_rate']:.3f} "
            f"landed_on_platform={res['landed_on_platform_rate']:.3f} "
            f"land_offset={('%.1fcm' % _off) if _off is not None else 'NA'} "
            f"realized_gap={res['gap_size_realized_m_mean']*100:.2f}cm "
            f"safe_stop={res['safe_stop_rate']:.3f} "
            f"fell={res['fell_rate']:.3f}",
            flush=True,
        )
        results.append(res)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    # also csv
    csv_path = args.out.replace(".json", ".csv")
    with open(csv_path, "w") as f:
        cols = ["gap_width_cm", "gap_size_realized_m_mean", "crossing_rate",
                "landed_on_platform_rate", "safe_stop_rate", "fell_rate", "nogo_hold_rate",
                "mean_landing_offset_cm",
                "fell_after_cross_rate", "mean_gaps_crossed", "mean_ep_len", "n_envs"]
        f.write(",".join(cols) + "\n")
        for r in results:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"\nSaved: {args.out}\n        {csv_path}")

    print("\n=== SUMMARY (landed_on_platform + land_offset = the DECISION signal) ===")
    print(f"{'gap_cm':>7} {'cross':>7} {'landed_ok':>10} {'land_off_cm':>12} {'ep_len':>7}")
    for r in results:
        off = r['mean_landing_offset_cm']
        print(f"{r['gap_width_cm']:>7.1f} {r['crossing_rate']:>7.3f} "
              f"{r['landed_on_platform_rate']:>10.3f} "
              f"{('%.1f' % off) if off is not None else 'NA':>12} {r['mean_ep_len']:>7.0f}")
    print("Adaptive (decides): landed_ok stays high across widths, land_off ~flat.")
    print("Fixed jump: landed_ok only in a narrow band; land_off slides down with gap width.")

    print("\n=== GO/NO-GO OUTCOMES (fair, fall-based; jump_limit = should-jump boundary) ===")
    print(f"{'gap_cm':>7} {'landed':>8} {'safe_stop':>10} {'fell':>8}   (landed+safe_stop+fell ~ 1)")
    for r in results:
        print(f"{r['gap_width_cm']:>7.1f} {r['landed_on_platform_rate']:>8.3f} "
              f"{r['safe_stop_rate']:>10.3f} {r['fell_rate']:>8.3f}")
    print("should-jump widths  -> landed = 該跳跳 (good); safe_stop = 該跳不跳 (timid).")
    print("should-NOT widths   -> safe_stop = 不該跳不跳 (correct, good); fell = 不該跳卻跳 (plunge).")


if __name__ == "__main__":
    main()
