"""Render RunGap rollouts as video using MuJoCo renderer.

Renders:
1. Zero-action rollout from close_profile-rodent camera (rodent view)
2. Arena overview with a slowly moving camera (shows full corridor)
"""

import os

os.environ["MUJOCO_GL"] = "egl"

import jax
import jax.numpy as jp
import imageio
import mujoco
import numpy as np

from vnl_playground.tasks.rodent import run_gap


def rollout(env, rng, policy_fn, n_steps=500):
    """Run a rollout collecting qpos at each step."""
    rng, reset_rng = jax.random.split(rng)
    state = jax.jit(env.reset)(reset_rng)
    step_fn = jax.jit(env.step)

    qposes = [np.array(state.data.qpos)]

    for _ in range(n_steps):
        rng, act_rng = jax.random.split(rng)
        action = policy_fn(act_rng, env.action_size)
        state = step_fn(state, action)
        qposes.append(np.array(state.data.qpos))
        if state.done > 0.5:
            break

    return np.array(qposes)


def render_video(mj_model, qposes, camera, output_path, fps=50, width=640, height=480):
    """Render qpos trajectory as video with a named camera."""
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=height, width=width)

    with imageio.get_writer(output_path, fps=fps) as video:
        for qpos in qposes:
            mj_data.qpos = qpos
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data, camera=camera)
            video.append_data(renderer.render())

    renderer.close()
    print(f"  Saved: {output_path} ({len(qposes)} frames, {len(qposes)/fps:.1f}s)")


def render_arena_overview(mj_model, qposes, corridor_end_x, output_path,
                          fps=50, width=1280, height=720):
    """Render an arena overview with a camera that pans along the corridor.

    The camera moves slowly from the start to the end of the corridor,
    looking down at an angle to show the full layout of platforms and gaps.
    """
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=height, width=width)

    n_frames = len(qposes)
    # Camera lookat travels from start to end of corridor
    lookat_x_start = -0.5
    lookat_x_end = corridor_end_x + 0.5

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 2.5
    cam.azimuth = 90      # looking from the side
    cam.elevation = -35    # looking down at an angle

    with imageio.get_writer(output_path, fps=fps) as video:
        for i, qpos in enumerate(qposes):
            mj_data.qpos = qpos
            mujoco.mj_forward(mj_model, mj_data)

            # Interpolate camera lookat along corridor
            t = i / max(n_frames - 1, 1)
            cam_x = lookat_x_start + t * (lookat_x_end - lookat_x_start)
            cam.lookat[:] = [cam_x, 0.0, 0.0]

            renderer.update_scene(mj_data, camera=cam)
            video.append_data(renderer.render())

    renderer.close()
    print(f"  Saved: {output_path} ({n_frames} frames, {n_frames/fps:.1f}s)")


def main():
    print("Initializing RunGap environment...")
    env = run_gap.RunGap()
    mj_model = env.mj_model
    fps = int(1.0 / env.dt)

    print(f"  FPS: {fps}")
    print(f"  Corridor end: {env._corridor_end_x:.2f}m")

    rng = jax.random.PRNGKey(0)

    # Zero-action rollout
    print("\nZero-action rollout...")
    rng, sub_rng = jax.random.split(rng)
    zero_qposes = rollout(env, sub_rng, lambda r, s: jp.zeros(s), n_steps=500)
    print(f"  Collected {len(zero_qposes)} frames")

    # Render close_profile view
    print("\nRendering close_profile-rodent view...")
    render_video(
        mj_model, zero_qposes, "close_profile-rodent",
        "scripts/run_gap_zero_action.mp4", fps=fps,
    )

    # Render arena overview
    print("\nRendering arena overview...")
    render_arena_overview(
        mj_model, zero_qposes, env._corridor_end_x,
        "scripts/run_gap_arena_overview.mp4", fps=fps,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
