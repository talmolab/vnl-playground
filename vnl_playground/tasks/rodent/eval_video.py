"""Rollout → MP4 renderer with tracking cameras, binocular vision strip and HUD.

Extracted verbatim from train_highlvl.render_video (scott_claude/maze-forage-vision
@ 2a4c29f) so that trainers in other packages can render without importing the
PPO trainer. `cameras=None` reproduces the historical single tracking camera
byte-for-byte; a list of {distance, azimuth, elevation, lookat_z, label} dicts
renders one panel each, side by side.
"""

import gc

import cv2
import imageio
import jax
import mujoco
import numpy as np
from mujoco.mjx.warp.types import DATA_NON_VMAP


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
    cameras=None,
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

    ``cameras`` is an optional list of panel specs, rendered left to right and
    concatenated into one frame.  Each spec is a dict of
    ``{distance, azimuth, elevation, lookat_z, label}``, all optional; every
    panel is a ``mjCAMERA_TRACKING`` camera on the walker torso.  ``None``
    (the default) means the single historical camera
    ``distance=1.0, azimuth=90, elevation=-20, lookat_z=0.3`` and produces
    byte-identical video to before, which is what the run-gap and PPO callers
    rely on.

    Multiple panels each get the full ``renderer`` width, so an N-panel video is
    N times as wide as a single-panel one.  The vision strip and the HUD are
    drawn ONCE, on the composite, not per panel.

    Why this exists: the historical camera sits ~0.34 m above the torso and
    ~0.94 m from it horizontally, which is fine in an open arena and useless
    inside a maze -- the walls are 0.30 m tall and the corridors 0.465 m wide,
    so the eye is barely above wall height with a wall between it and the
    walker.  Measured on three random maze resets, the walker was fully
    occluded in two of them.  A top-down panel (``elevation=-90``) cannot be
    occluded at all.

    ``reward_remix`` is an optional ``{"sparse_key": str, "lambda": float}``.
    When given, the HUD's reward and cumulative-reward lines report the reward
    the replay buffer STORED, ``sparse + lambda*(total - sparse)``
    (rollout.py:200-207), with the raw env total shown alongside. The per-term
    breakdown stays raw -- lambda scales the whole dense remainder uniformly,
    so the single displayed lambda covers it.
    """
    import math

    # Try common body names across walkers (rodent, sprout, stick, etc.)
    track_body_names = [
        "torso-rodent",
        "torso_link-sprout",
        "torso",
        "torso_link",
    ]
    for name in track_body_names:
        try:
            track_body_id = mj_model.body(name).id
            break
        except Exception:
            continue
    else:
        track_body_id = 1

    def _make_camera(spec):
        """Tracking camera from a panel spec; unset keys keep the old defaults."""
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = track_body_id
        cam.distance = float(spec.get("distance", 1.0))
        cam.azimuth = float(spec.get("azimuth", 90))
        cam.elevation = float(spec.get("elevation", -20))
        cam.lookat[:] = [0, 0, float(spec.get("lookat_z", 0.3))]
        return cam

    # `None` -> one panel with exactly the historical parameters.
    camera_specs = [dict(spec) for spec in cameras] if cameras else [{}]
    panel_cameras = [_make_camera(spec) for spec in camera_specs]
    panel_labels = [str(spec.get("label", "")) for spec in camera_specs]

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

        left_frames = vision_stack[..., :mono_c]  # (T, H, W, C)
        right_frames = vision_stack[..., mono_c:]  # (T, H, W, C)

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
            # `renderer.render()` hands back its own reusable buffer, so each
            # panel has to be copied out before the next update_scene call.
            panels = []
            for cam, label in zip(panel_cameras, panel_labels):
                renderer.update_scene(mj_data, cam, scene_option=scene_option)
                panel = renderer.render().copy()
                if label:
                    cv2.putText(
                        panel,
                        label,
                        (10, panel.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                panels.append(panel)
            frame = panels[0] if len(panels) == 1 else np.concatenate(panels, axis=1)

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
                    forward_vel = float(
                        np.asarray(state.data.subtree_linvel[_torso_id, 0])
                    )
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
                        ""
                        if _lam is None
                        else f" lam={_lam:.2f} env={step_reward_env:.3f}"
                    )
                    hud_lines.append(
                        (
                            f"Reward: {step_reward:.3f}{_lam_txt}  ({', '.join(parts)})",
                            CYAN,
                        )
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
                        else YELLOW
                        if dist_to_gap < 0.3
                        else GRAY
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
