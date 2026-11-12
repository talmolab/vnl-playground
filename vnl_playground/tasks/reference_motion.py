"""Shared continuous reference poses and clocks for dense imitation tasks."""

import math
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from jaxtyping import Array, Bool, Float, Integer
from mujoco import mjx

from vnl_playground.tasks.reference_clips import ReferenceClipData, ReferenceClips


@jax.jit
def _slerp(
    a: Float[Array, "... 4"],
    b: Float[Array, "... 4"],
    fraction: float | Float[Array, "..."],
) -> Float[Array, "... 4"]:
    """Shortest-path WXYZ interpolation, including coincident/antipodal inputs."""
    a = a / jp.maximum(jp.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    b = b / jp.maximum(jp.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    dot = jp.sum(a * b, axis=-1, keepdims=True)
    b = jp.where(dot < 0, -b, b)
    distance_sq = jp.sum(jp.square(a - b), axis=-1, keepdims=True)
    small = distance_sq < jp.finfo(a.dtype).eps
    # Guard sqrt at zero;
    distance = jp.sqrt(jp.where(small, 1.0, distance_sq))
    # Quaternion differences preserve tiny angles that acos(dot) rounds away.
    angle = 2 * jp.arctan2(distance, jp.linalg.norm(a + b, axis=-1, keepdims=True))
    u = angle / jp.pi  # JAX sinc(x) is sin(pi*x)/(pi*x).
    denominator = jp.sinc(u)

    def weight(c: float | Float[Array, "..."]) -> Float[Array, "..."]:
        spherical = c * jp.sinc(c * u) / denominator
        # SLERP's series in squared chord distance preserves endpoint gradients.
        limit = c * (1 + (1 - c * c) * distance_sq / 6)
        return jp.where(small, limit, spherical)

    result = weight(1 - fraction) * a + weight(fraction) * b
    return result / jp.maximum(jp.linalg.norm(result, axis=-1, keepdims=True), 1e-12)


class ReferenceMotion:
    """Sample a clip at fractional frame positions using the simulation skeleton.

    Scalar qpos channels are linear and every free/ball quaternion uses SLERP.
    Body transforms come from FK, in the reference file's body order. Stored
    qvel is interpolated as independent supervision.
    """

    def __init__(
        self,
        clips: ReferenceClips,
        model: mujoco.MjModel,
        *,
        mocap_hz: float,
        body_suffix: str = "",
    ) -> None:
        clips._require_collection()
        if clips.n_clips < 1 or clips.n_frames < 1:
            raise ValueError("Reference motion requires nonempty clips.")
        if not math.isfinite(mocap_hz) or mocap_hz <= 0:
            raise ValueError("mocap_hz must be finite and positive.")
        if (clips.qpos.shape[-1], clips.qvel.shape[-1]) != (model.nq, model.nv):
            raise ValueError("Reference qpos/qvel dimensions must match the model.")
        self.clips = clips
        self.mocap_hz = float(mocap_hz)
        self.duration = (clips.n_frames - 1) / self.mocap_hz

        names = [
            model.body(i).name.removesuffix(body_suffix) for i in range(model.nbody)
        ]
        missing = set(clips.metadata.xpos_names) - set(names)
        if missing:
            raise ValueError(
                f"Reference bodies missing from the model: {sorted(missing)}"
            )
        self._body_ids = jp.array(
            [names.index(name) for name in clips.metadata.xpos_names]
        )
        self._quaternion_starts_at_idx = tuple(
            int(address) + (3 if kind == mujoco.mjtJoint.mjJNT_FREE else 0)
            for kind, address in zip(model.jnt_type, model.jnt_qposadr)
            if kind in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL)
        )
        # Pure JAX kinematics avoids allocating a second set of Warp contact worlds.
        self._model = mjx.put_model(model, impl="jax")
        self._data = mjx.make_data(model, impl="jax")

    def sample(
        self, clip: int | Integer[Array, "..."], frame: int | float | Array
    ) -> ReferenceClips:
        """Sample broadcastable clip/frame arrays; clamp each time independently.

        For bracketing frames 0 and 1 and interpolation fraction t in [0, 1]::

            qpos(t)[scalar] = (1 - t) * qpos0[scalar] + t * qpos1[scalar]
            qpos(t)[quat]   = SLERP(qpos0[quat], qpos1[quat], t)
            qvel(t)         = (1 - t) * qvel0 + t * qvel1
            (xpos(t), xquat(t)) = FK(qpos(t)) # Use interpolated qpos(t)

        SLERP applies to each free/ball quaternion block. FK uses the simulation
        model and returns world-space body poses in reference body order.
        Stored qvel is interpolated independently of the pose derivative.
        """
        clip, frame = jp.broadcast_arrays(
            jp.asarray(clip, dtype=jp.int32), jp.asarray(frame)
        )
        frame = jp.clip(frame, 0, self.clips.n_frames - 1)
        lower = jp.floor(frame).astype(jp.int32)
        upper = jp.minimum(lower + 1, self.clips.n_frames - 1)
        fraction = (frame - lower)[..., None]
        q0, q1 = self.clips.qpos[clip, lower], self.clips.qpos[clip, upper]
        qpos = q0 + fraction * (q1 - q0)
        for quat_idx in self._quaternion_starts_at_idx:
            quaternion = _slerp(
                q0[..., quat_idx : quat_idx + 4],
                q1[..., quat_idx : quat_idx + 4],
                fraction,
            )
            qpos = qpos.at[..., quat_idx : quat_idx + 4].set(quaternion)
        v0, v1 = self.clips.qvel[clip, lower], self.clips.qvel[clip, upper]
        qvel = v0 + fraction * (v1 - v0)

        def body_poses(
            q: Array,
        ) -> tuple[Float[Array, "nbody 3"], Float[Array, "nbody 4"]]:
            d = mjx.kinematics(self._model, self._data.replace(qpos=q))
            return d.xpos[self._body_ids], d.xquat[self._body_ids]

        xpos, xquat = jax.vmap(body_poses)(qpos.reshape((-1, qpos.shape[-1])))
        data = ReferenceClipData(
            qpos,
            qvel,
            xpos.reshape((*frame.shape, len(self._body_ids), 3)),
            xquat.reshape((*frame.shape, len(self._body_ids), 4)),
        )
        return replace(
            self.clips,
            data=data,
            source_clip_indices=self.clips.source_clip_indices[clip],
            behaviour_labels=self.clips._label_for_clip(clip)
            if clip.ndim == 0
            else None,
        )


class ReferenceMotionMixin:
    """The shared reference clock, reward target and lookahead for dense tasks."""

    def initialize_reference_motion(self) -> None:
        self._reference_motion = ReferenceMotion(
            self.reference_clips,
            self.mj_model,
            mocap_hz=self._config.mocap_hz,
            body_suffix=self._suffix,
        )
        substeps = int(self._config.ctrl_dt / self._config.sim_dt)
        if substeps < 1:
            raise ValueError("ctrl_dt must advance at least one simulation step.")
        self._reference_control_dt = substeps * self._config.sim_dt
        length = self._config.reference_length
        stride = self._config.get("reference_stride", 1)
        if length < 1 or stride < 1:
            raise ValueError("reference_length and reference_stride must be positive.")
        self._reference_frame_offsets = 1 + jp.arange(length) * stride

        @jax.jit
        def render_targets(info):
            reference = self.get_current_reference_state(None, info)
            # Returning only joint state lets XLA discard reference FK.
            return reference.qpos, reference.qvel

        self._render_targets = render_targets

    def get_render_reference(self, trajectory: Any) -> tuple[np.ndarray, np.ndarray]:
        """Prepare host qpos/qvel for a sequence of states or stacked states."""
        keys = ("reference_clip", "start_frame", "reference_step")
        if isinstance(trajectory, Sequence):
            info = {
                key: jp.asarray([state.info[key] for state in trajectory])
                for key in keys
            }
        else:
            info = {key: trajectory.info[key] for key in keys}
        return jax.device_get(self._render_targets(info))

    def get_reference_frame_idx(
        self, data: mjx.Data | None, info: dict[str, Any]
    ) -> Float[Array, "..."]:
        """Continuous frame position; integer steps avoid accumulated clock drift."""
        elapsed = info["reference_step"] * self._reference_control_dt
        return elapsed * self._reference_motion.mocap_hz + info["start_frame"]

    def get_current_reference_state(
        self, data: mjx.Data | None, info: dict[str, Any]
    ) -> ReferenceClips:
        """The exact reference used by rewards, termination, export and rendering."""
        return self._reference_motion.sample(
            info["reference_clip"], self.get_reference_frame_idx(data, info)
        )

    def get_future_reference_states(
        self, data: mjx.Data, info: dict[str, Any]
    ) -> ReferenceClips:
        frame = self.get_reference_frame_idx(data, info)
        frames = jp.expand_dims(frame, -1) + self._reference_frame_offsets
        clips = jp.expand_dims(info["reference_clip"], -1)
        return self._reference_motion.sample(clips, frames)

    def is_reference_truncated(
        self, data: mjx.Data, info: dict[str, Any]
    ) -> Bool[Array, "..."]:
        last = self.reference_clips.n_frames - 1 - self._reference_frame_offsets[-1]
        # The clock can round an exact clip boundary up by one float32 increment.
        limit = jp.nextafter(jp.float32(last), jp.inf)
        return self.get_reference_frame_idx(data, info) > limit

    @property
    def num_control_steps_per_clip(self) -> int:
        """Control steps spanning the clip, including the initial state separately."""
        return math.floor(
            self._reference_motion.duration / self._reference_control_dt + 1e-9
        )
