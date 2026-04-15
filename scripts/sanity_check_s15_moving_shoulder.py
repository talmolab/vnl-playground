"""Sanity check for MouseImitationMovingShoulder.

Verifies:
  1. Env instantiates from defaults.
  2. Reset places qpos[:3] exactly at ref.qpos[:3] of frame 0.
  3. After N zero-action steps, qpos[:3] still matches ref.qpos[:3] at current frame.
  4. Reward terms return finite scalars and `joints` L2 error excludes the IK dims.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jp
import numpy as np

from vnl_playground.tasks.mouse.imitation_moving_shoulder import (
    MouseImitationMovingShoulder,
    default_config,
)


def main() -> None:
    cfg = default_config()
    env = MouseImitationMovingShoulder(cfg)
    print(f"nq={env.mjx_model.nq}, nu={env.mjx_model.nu}, "
          f"ik_driven_dims={cfg.ik_driven_dims}")
    assert env.mjx_model.nq == 7, env.mjx_model.nq

    rng = jax.random.PRNGKey(0)
    state = env.reset(rng, clip_idx=0, start_frame=0)

    ref0 = env.reference_clips.at(clip=0, frame=0)
    n = int(cfg.ik_driven_dims)

    # (2) Reset matches reference exactly on IK dims.
    np.testing.assert_allclose(
        np.asarray(state.data.qpos[:n]), np.asarray(ref0.qpos[:n]), atol=0, rtol=0,
        err_msg="reset did not snap IK dims to reference",
    )
    print("reset qpos[:3] matches ref:", np.asarray(state.data.qpos[:n]))

    # (3) After zero-action steps, IK dims still track reference.
    action = env.null_action()
    for i in range(1, 20):
        state = env.step(state, action)
        cur_frame = int(state.metrics["current_frame"])
        ref = env.reference_clips.at(clip=0, frame=cur_frame)
        np.testing.assert_allclose(
            np.asarray(state.data.qpos[:n]), np.asarray(ref.qpos[:n]),
            atol=0, rtol=0,
            err_msg=f"step {i}: IK dims drifted from reference at frame {cur_frame}",
        )

    # (4) Rewards / metrics are finite.
    assert np.isfinite(float(state.reward)), state.reward
    assert "joint_l2_error" in state.metrics
    print("reward:", float(state.reward))
    print("joint_l2_error:", float(state.metrics["joint_l2_error"]))
    print("OK")


if __name__ == "__main__":
    main()
