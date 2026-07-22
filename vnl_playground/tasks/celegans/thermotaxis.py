"""Thermotaxis task for virtual C. elegans.

A thermal gradient (see :mod:`vnl_playground.tasks.celegans.gradients`) is laid over
the arena floor. The worm senses only the local temperature and must locomote to the
location whose temperature matches a target ``setpoint`` (biologically, its preferred
cultivation temperature).

Reward is a Gaussian on the temperature error ``|T(worm) - setpoint|`` with a large
bonus for being within ``epsilon`` of the setpoint; optional control/energy costs are
available. Episodes terminate when the worm reaches the setpoint, drifts too far (in
temperature), walks off the finite floor and falls, or produces NaNs. The training
harness enforces the fixed step budget via ``episode_length``.

This is the deterministic v1: a fixed left start, a fixed setpoint anchored on the
right, and a linear left->right gradient the worm traverses. The config /
``info`` schema and the :class:`Gradient` factory are architected so per-episode
randomization (setpoint / start / gradient shape) and optional observation noise /
delay can be enabled later without a redesign, including across vmapped parallel
environments.
"""

import collections
import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from vnl_playground.tasks.celegans import base as celegans_base
from vnl_playground.tasks.celegans import consts
from vnl_playground.tasks.celegans.gradients import Gradient
from vnl_playground.tasks.reward_registry import RewardRegistry

_registry = RewardRegistry()


def default_config() -> config_dict.ConfigDict:
    """Returns the default configuration for the Thermotaxis environment."""
    config = config_dict.create(
        episode_length=2000,
        action_repeat=1,
        init_z=0.0,
        # fixed start on the cold/left side (v1 deterministic)
        init_x=-8.0,
        init_y=0.0,
        # kwargs for Gradient.make_gradient; any iterable leaf is [low, high] bounds
        gradient_cfg=config_dict.create(
            gradient_type="linear",  # str | list[str] | "random"; default linear
            setpoint=23.0,  # target temp AT setpoint_loc
            setpoint_loc=(5.0, 0.0),  # (x, y) anchor; each element scalar or [lo, hi]
            min_temp=15.0,
            max_temp=25.0,
            arena_size=(10.0, 10.0),  # (Lx, Ly); must match arena_bounded.xml floor
        ),
        # thresholds (TEMPERATURE space, except fell_off which is physical z)
        epsilon=0.25,
        max_temp_error=8.0,
        min_z=-0.5,
        # observation noise / delay (OFF in v1); buffer size static
        obs_noise_std=0.0,
        obs_delay=0,
        max_obs_delay=4,
        reward_terms={
            "temperature": {"weight": 1.0, "exp_scale": 2.0},
            "setpoint_bonus": {"weight": 10.0, "epsilon": 0.25},
            "control": {"weight": 0.0},
            "energy": {"weight": 0.0, "max_value": 50.0},
        },
        termination_criteria={
            "reached_setpoint": {"epsilon": 0.25},
            "too_far": {"max_temp_error": 8.0},
            "fell_off": {"min_z": -0.5},
            "nan": {},
        },
        **celegans_base.default_config(),
    )
    # Override the base (infinite plane) arena with the bounded box floor so the worm
    # can fall off the edge. Keep it an epath.Path to satisfy ConfigDict type checks.
    config.arena_xml_path = consts.CELEGANS_PATH / "xmls" / "arena_bounded.xml"
    return config


class Thermotaxis(celegans_base.CelegansEnv):
    """Thermotaxis environment: navigate a thermal gradient to a target temperature."""

    _registry = _registry

    def __init__(
        self,
        rng: jax.Array = jax.random.PRNGKey(0),
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, List[Any]]]] = None,
    ) -> None:
        """Initialize the Thermotaxis environment.

        Args:
            rng: Random number generator key (stored for reference; per-episode
                randomness is driven by the key passed to ``reset``).
            config: Configuration dictionary.
            config_overrides: Optional configuration overrides.
        """
        super().__init__(config, config_overrides)
        self._rng = rng

        # ConfigDict annoyingly sorts dictionary by keys
        friction = [
            self.config.friction["tan_floor"],
            self.config.friction["tan_body"],
            self.config.friction["tor"],
            self.config.friction["roll_floor"],
            self.config.friction["roll_body"],
        ]
        solimp = [
            self.config.solimp["d0"],
            self.config.solimp["dwidth"],
            self.config.solimp["width"],
            self.config.solimp["midpoint"],
            self.config.solimp["power"],
        ]
        solref = [
            self.config.solref["timeconst"],
            self.config.solref["dampratio"],
        ]
        solreffriction = [
            self.config.solreffriction["timeconst"],
            self.config.solreffriction["dampratio"],
        ]

        if self.config.contact_geom.lower() == "mesh":
            contact_geom = mujoco.mjtGeom.mjGEOM_MESH
        elif self.config.contact_geom.lower() == "capsule":
            contact_geom = mujoco.mjtGeom.mjGEOM_CAPSULE
        else:
            contact_geom = mujoco.mjtGeom.mjGEOM_SPHERE

        # Spawn at the origin facing +x; the planar (x, y) start is set in reset() by
        # writing the root slide-joint qpos (see _set_root_xy).
        self.add_worm(
            torque_actuators=self._config.torque_actuators,
            rescale_factor=self._config.rescale_factor,
            pos=[0.0, 0.0, self._config.init_z],
            quat=(1, 0, 0, 0),
            friction=friction,
            solimp=solimp,
            solref=solref,
            solreffriction=solreffriction,
            contact_geom=contact_geom,
            muscle_config=self._config.muscle_config,
            joint_config=self._config.joint_config,
        )

        self._spec.worldbody.add_light(pos=[0, 0, 10], dir=[0, 0, -1])
        self.compile()

        # Cache the (static) qpos addresses of the planar root slide joints.
        self._rootx_qadr = int(
            self.mj_model.jnt_qposadr[
                mujoco.mj_name2id(
                    self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, f"rootx{self.suffix}"
                )
            ]
        )
        self._rooty_qadr = int(
            self.mj_model.jnt_qposadr[
                mujoco.mj_name2id(
                    self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, f"rooty{self.suffix}"
                )
            ]
        )

        self.max_reward = sum(
            params["weight"]
            for params in self._config.reward_terms.values()
            if "weight" in params
        )

    # ----------------------------------------------------------------- helpers
    def _set_root_xy(self, data: mjx.Data, x: jp.ndarray, y: jp.ndarray) -> mjx.Data:
        """Place the worm's planar root at world ``(x, y)`` via slide-joint qpos."""
        qpos = data.qpos.at[self._rootx_qadr].set(x).at[self._rooty_qadr].set(y)
        data = data.replace(qpos=qpos)
        return mjx.forward(self.mjx_model, data)

    def _worm_xy(self, data: mjx.Data) -> jp.ndarray:
        """Worm root planar position ``(x, y)``."""
        return self._get_root_pos(data)[: self.config.dim]

    def _temperature_at(self, xy: jp.ndarray, info: Mapping[str, Any]) -> jp.ndarray:
        """Temperature at planar position ``xy`` for this episode's gradient."""
        return Gradient.evaluate(info["shape_id"], info["temp_field"], xy)

    def _temp_error(self, data: mjx.Data, info: Mapping[str, Any]) -> jp.ndarray:
        """Absolute temperature error ``|T(worm) - setpoint|``."""
        temp = self._temperature_at(self._worm_xy(data), info)
        return jp.abs(temp - info["setpoint"])

    def _sense_temperature(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> jp.ndarray:
        """The temperature the agent observes (optionally delayed + noised).

        ``obs_delay`` / ``obs_noise_std`` / ``max_obs_delay`` are static config, so
        both branches resolve at trace time and are zero-cost when disabled.
        """
        sensed = self._temperature_at(self._worm_xy(data), info)
        delay = int(self._config.obs_delay)
        if delay > 0:
            buf_len = int(self._config.max_obs_delay) + 1
            read_idx = (info["hist_index"] - 1 - delay) % buf_len
            sensed = info["temp_history"][read_idx]
        noise_std = float(self._config.obs_noise_std)
        if noise_std > 0.0:
            sensed = sensed + noise_std * jax.random.normal(info["noise_key"], ())
        return sensed

    def _init_info(self, rng: jax.Array) -> Dict[str, Any]:
        """Build the per-episode ``info`` dict (gradient, setpoint, buffers, rng)."""
        grad_key, stream_key = jax.random.split(rng)
        stream_key, noise_key = jax.random.split(stream_key)

        shape_id, params, setpoint = Gradient.make_gradient(
            grad_key, **self._config.gradient_cfg.to_dict()
        )
        start_xy = jp.array(
            [self._config.init_x, self._config.init_y], dtype=jp.float32
        )
        return {
            "start_xy": start_xy,
            "setpoint": setpoint,
            "temp_field": params,
            "shape_id": shape_id,
            "rng": stream_key,  # persistent obs-noise stream
            "noise_key": noise_key,  # refreshed each step
            "temp_history": jp.zeros(
                int(self._config.max_obs_delay) + 1, dtype=jp.float32
            ),
            "hist_index": jp.asarray(0, dtype=jp.int32),
            "prev_action": self.null_action(),
            "action": self.null_action(),
        }

    def null_action(self) -> jp.ndarray:
        """Return zero action."""
        return jp.zeros(self.action_size)

    # ------------------------------------------------------------- env interface
    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Reset the environment: sample a gradient and place the worm at the start.

        Args:
            rng: Random number generator key.

        Returns:
            mjx_env.State: The initial environment state.
        """
        info = self._init_info(rng)

        data = mjx.make_data(
            self.mj_model,
            impl=self._config.mujoco_impl,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        data = self._set_root_xy(data, info["start_xy"][0], info["start_xy"][1])

        # Prefill the obs-delay ring buffer with the initial temperature.
        init_temp = self._temperature_at(self._worm_xy(data), info)
        info["temp_history"] = jp.full_like(info["temp_history"], init_temp)

        metrics: Dict[str, Any] = {}
        obs = self._get_obs(data, info)
        reward = self._get_reward(data, info, metrics)
        done = self._is_done(data, info, metrics)

        return mjx_env.State(data, obs, reward, jp.astype(done, float), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Step the environment forward by one control step.

        Args:
            state: Current environment state.
            action: Action to apply.

        Returns:
            mjx_env.State: The new environment state after stepping.
        """
        n_steps = int(self._config.ctrl_dt / self._config.sim_dt)
        data = mjx_env.step(self.mjx_model, state.data, action, n_steps)

        info = state.info
        info["prev_action"] = info["action"]
        info["action"] = action

        # Advance the obs-noise rng stream.
        rng, noise_key = jax.random.split(info["rng"])
        info["rng"] = rng
        info["noise_key"] = noise_key

        # Push the current true temperature into the obs-delay ring buffer.
        true_temp = self._temperature_at(self._worm_xy(data), info)
        idx = info["hist_index"]
        info["temp_history"] = info["temp_history"].at[idx].set(true_temp)
        info["hist_index"] = (idx + 1) % (int(self._config.max_obs_delay) + 1)

        obs = self._get_obs(data, info)
        done = self._is_done(data, info, state.metrics)
        reward = self._get_reward(data, info, state.metrics)
        reward = jp.nan_to_num(reward)

        return state.replace(
            data=data,
            obs=obs,
            info=info,
            reward=reward,
            done=done.astype(float),
        )

    def _get_obs(self, data: mjx.Data, info: Mapping[str, Any]) -> collections.OrderedDict:
        """Observation: temperature-only task obs plus the standard proprioception.

        Args:
            data: The simulation data.
            info: State info dictionary.

        Returns:
            OrderedDict with a ``state`` key wrapping ``task_obs`` and ``proprioception``.
        """
        task_obs = jp.atleast_1d(self._sense_temperature(data, info))
        obs = collections.OrderedDict(
            task_obs=task_obs,
            proprioception=self._get_proprioception(data, info, flatten=False),
        )
        return collections.OrderedDict(state=obs)

    # ------------------------------------------------------------------ rewards
    @_registry.reward("temperature")
    def _temperature_reward(self, data, info, metrics, weight, exp_scale) -> float:
        """Gaussian reward on the temperature error (1.0 at zero error)."""
        err = self._temp_error(data, info)
        reward = weight * jp.exp(-((err / exp_scale) ** 2) / 2)
        metrics["rewards/temperature"] = metrics["rewards/temperature/per_step"] = (
            reward
        )
        metrics["magnitudes/temp_error"] = metrics[
            "magnitudes/temp_error/per_step"
        ] = err
        return reward

    @_registry.reward("setpoint_bonus")
    def _setpoint_bonus_reward(self, data, info, metrics, weight, epsilon) -> float:
        """Large bonus for being within ``epsilon`` (temperature) of the setpoint."""
        err = self._temp_error(data, info)
        reward = weight * (err < epsilon).astype(float)
        metrics["rewards/setpoint_bonus"] = metrics[
            "rewards/setpoint_bonus/per_step"
        ] = reward
        return reward

    @_registry.reward("control")
    def _control_cost(self, data, info, metrics, weight) -> float:
        """Cost for control effort (action magnitude)."""
        del data
        ctrl_magnitude = jp.sum(jp.square(info["action"]))
        cost = -weight * ctrl_magnitude
        metrics["costs/control"] = metrics["costs/control/per_step"] = cost
        metrics["magnitudes/control"] = metrics["magnitudes/control/per_step"] = (
            ctrl_magnitude
        )
        return cost

    @_registry.reward("energy")
    def _energy_cost(self, data, info, metrics, weight, max_value) -> float:
        """Cost for energy consumption (clipped)."""
        del info
        energy = jp.minimum(
            jp.sum(jp.abs(data.qvel) * jp.abs(data.qfrc_actuator)), max_value
        )
        cost = -weight * energy
        metrics["costs/energy"] = metrics["costs/energy/per_step"] = cost
        metrics["magnitudes/energy"] = metrics["magnitudes/energy/per_step"] = energy
        return cost

    # ------------------------------------------------------------- terminations
    @_registry.termination("reached_setpoint")
    def _reached_setpoint(self, data, info, epsilon) -> bool:
        """Success: within ``epsilon`` (temperature) of the setpoint."""
        return self._temp_error(data, info) < epsilon

    @_registry.termination("too_far")
    def _too_far(self, data, info, max_temp_error) -> bool:
        """Give up: temperature error exceeds ``max_temp_error``."""
        return self._temp_error(data, info) > max_temp_error

    @_registry.termination("fell_off")
    def _fell_off(self, data, info, min_z) -> bool:
        """Walked off the finite floor and fell below ``min_z``."""
        del info
        return self._get_root_pos(data)[2] < min_z

    @_registry.termination("nan")
    def _nan_termination(self, data, info) -> bool:
        """NaNs detected in the simulation state."""
        del info
        return jp.any(jp.isnan(data.qpos))

    # ---------------------------------------------------------------- rendering
    @staticmethod
    def _temperature_colors(temps: np.ndarray, alpha: float) -> np.ndarray:
        """Map temperatures to a blue->grey->red diverging RGBA (in [0, 1]).

        Normalized over the min/max of ``temps`` for maximum contrast. Dependency
        free (no matplotlib) so rendering never fails on a missing colormap.
        """
        t = np.asarray(temps, dtype=np.float64)
        tmin, tmax = float(t.min()), float(t.max())
        tn = (t - tmin) / (tmax - tmin + 1e-8)
        cold = np.array([59.0, 76.0, 192.0])  # coolwarm blue
        mid = np.array([221.0, 221.0, 221.0])  # light grey
        warm = np.array([180.0, 4.0, 38.0])  # coolwarm red
        below = tn < 0.5
        frac = np.where(below, tn / 0.5, (tn - 0.5) / 0.5)[..., None]
        rgb = np.where(below[..., None], cold + (mid - cold) * frac, mid + (warm - mid) * frac)
        rgba = np.concatenate([rgb / 255.0, np.full((rgb.shape[0], 1), alpha)], axis=-1)
        return rgba.astype(np.float32)

    def _overlay_text(self, frame: np.ndarray, lines: Sequence[str]) -> np.ndarray:
        """Draw white-on-black outlined text lines onto ``frame`` (top-left).

        Uses cv2 if available, else Pillow; if neither is installed, warns once and
        returns the frame unchanged so rendering never hard-fails on annotations.
        """
        try:
            import cv2

            for j, line in enumerate(lines):
                org = (10, 24 + j * 22)
                cv2.putText(frame, line, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, line, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 1, cv2.LINE_AA)
            return frame
        except ImportError:
            pass
        try:
            from PIL import Image, ImageDraw

            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)
            for j, line in enumerate(lines):
                x, y = 10, 8 + j * 18
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    draw.text((x + dx, y + dy), line, fill=(0, 0, 0))
                draw.text((x, y), line, fill=(255, 255, 255))
            return np.asarray(img)
        except ImportError:
            if not getattr(self, "_warned_no_text", False):
                warnings.warn("Neither cv2 nor PIL available; skipping annotations.")
                self._warned_no_text = True
            return frame

    def render(
        self,
        trajectory: Sequence[mjx_env.State],
        height: int = 480,
        width: int = 640,
        camera: Optional[Union[str, int]] = None,
        scene_option: Optional["mujoco.MjvOption"] = None,
        annotate: bool = True,
        overlay_gradient: bool = True,
        grid_resolution: int = 48,
        overlay_alpha: float = 0.85,
        show_markers: bool = True,
        fps: Optional[float] = None,
        vid_path: Optional[str] = None,
    ) -> List[np.ndarray]:
        """Render a rollout, overlaying the thermal gradient on the floor.

        The thermal field is drawn as a grid of flat colored tiles (blue = cold,
        red = warm) on the floor, with a green marker at the setpoint location and a
        black marker at the worm (the ~mm worm is otherwise sub-pixel in the ±10 cm
        arena). When ``annotate`` is set, each frame is captioned with the episode
        step, per-step reward, cumulative reward, ``current_temp / setpoint``, and any
        terminations that have fired.

        Args:
            trajectory: Sequence of single-environment states (one per control step),
                e.g. from a rollout. The gradient parameters are read from
                ``trajectory[0].info`` and assumed constant across the episode.
            height, width: Frame size in pixels.
            camera: Camera name/id. ``None`` uses a top-down overhead view framing the
                whole arena (best for seeing the gradient and the worm's trajectory).
            scene_option: Optional MjvOption for the scene.
            annotate: Overlay the text stats described above.
            overlay_gradient: Draw the thermal field on the floor.
            grid_resolution: Tiles per axis for the gradient overlay (higher = smoother
                but more geoms / slower).
            overlay_alpha: Opacity of the gradient tiles.
            show_markers: Draw the worm and setpoint position markers.
            fps: Video frame rate (defaults to ``1 / ctrl_dt``).
            vid_path: If given, also write the frames to this path.

        Returns:
            List of rendered RGB frames (HxWx3 uint8).
        """
        if len(trajectory) == 0:
            return []

        mj_model = self.mj_model
        mj_model.vis.global_.offwidth = max(int(mj_model.vis.global_.offwidth), width)
        mj_model.vis.global_.offheight = max(
            int(mj_model.vis.global_.offheight), height
        )
        mj_data = mujoco.MjData(mj_model)
        root_bid = mj_model.body(f"{self.root_name}{self.suffix}").id
        floor_size = mj_model.geom("floor").size
        arena_lx, arena_ly = float(floor_size[0]), float(floor_size[1])

        max_geom = int(grid_resolution) ** 2 + 2000
        renderer = mujoco.Renderer(
            mj_model, height=height, width=width, max_geom=max_geom
        )

        # Camera: default to a whole-arena top-down view.
        if camera is None:
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [0.0, 0.0, 0.0]
            cam.distance = 2.9 * max(arena_lx, arena_ly)
            cam.azimuth = 90.0
            cam.elevation = -90.0
            render_camera: Any = cam
        else:
            render_camera = camera

        # Episode-constant gradient params (from the first state).
        info0 = trajectory[0].info
        shape_id = int(np.asarray(info0["shape_id"]))
        params = np.asarray(info0["temp_field"], dtype=np.float32)
        setpoint = float(np.asarray(info0["setpoint"]))
        setpoint_loc = (float(params[1]), float(params[2]))  # lx, ly packed in params

        sid_jp = jp.int32(shape_id)
        params_jp = jp.asarray(params)

        def field_temp(xy: np.ndarray) -> float:
            return float(Gradient.evaluate(sid_jp, params_jp, jp.asarray(xy)))

        # Precompute gradient tiles (constant across the episode).
        if overlay_gradient:
            r = int(grid_resolution)
            xs = np.linspace(-arena_lx, arena_lx, r, endpoint=False) + arena_lx / r
            ys = np.linspace(-arena_ly, arena_ly, r, endpoint=False) + arena_ly / r
            gx, gy = np.meshgrid(xs, ys, indexing="xy")
            grid_xy = np.stack([gx.ravel(), gy.ravel()], axis=-1)
            temps = np.asarray(
                jax.vmap(lambda p: Gradient.evaluate(sid_jp, params_jp, p))(
                    jp.asarray(grid_xy)
                )
            )
            tiles_rgba = self._temperature_colors(temps, overlay_alpha)
            tiles_pos = np.column_stack(
                [grid_xy, np.full(len(grid_xy), 0.001)]
            ).astype(np.float64)
            tile_size = np.array([arena_lx / r, arena_ly / r, 0.0005])
            eye = np.eye(3).flatten()

        term_names = list(self.config.termination_criteria.keys())
        frames: List[np.ndarray] = []
        cumulative = 0.0
        for i, state in enumerate(trajectory):
            mj_data.qpos = np.asarray(state.data.qpos)
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(
                mj_data, camera=render_camera, scene_option=scene_option
            )
            scene = renderer.scene

            if overlay_gradient:
                for pos, rgba in zip(tiles_pos, tiles_rgba):
                    if scene.ngeom >= scene.maxgeom:
                        break
                    mujoco.mjv_initGeom(
                        scene.geoms[scene.ngeom],
                        mujoco.mjtGeom.mjGEOM_BOX,
                        tile_size,
                        pos,
                        eye,
                        rgba,
                    )
                    scene.ngeom += 1

            wx = float(mj_data.xpos[root_bid][0])
            wy = float(mj_data.xpos[root_bid][1])
            if show_markers and scene.ngeom + 2 <= scene.maxgeom:
                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([0.4, 0.0, 0.0]),
                    np.array([setpoint_loc[0], setpoint_loc[1], 0.05]),
                    np.eye(3).flatten(),
                    np.array([0.1, 0.9, 0.2, 1.0], dtype=np.float32),
                )
                scene.ngeom += 1
                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([0.25, 0.0, 0.0]),
                    np.array([wx, wy, 0.06]),
                    np.eye(3).flatten(),
                    np.array([0.05, 0.05, 0.05, 1.0], dtype=np.float32),
                )
                scene.ngeom += 1

            frame = np.ascontiguousarray(renderer.render())
            cumulative += float(state.reward)

            if annotate:
                temp = field_temp(np.array([wx, wy]))
                fired = [
                    n
                    for n in term_names
                    if float(np.asarray(state.metrics.get(f"terminations/{n}", 0.0)))
                    > 0
                ]
                lines = [
                    f"step: {i}",
                    f"reward: {float(state.reward):+.3f}",
                    f"cum reward: {cumulative:+.2f}",
                    f"T/set: {temp:.2f} / {setpoint:.2f}",
                    f"term: {', '.join(fired) if fired else '-'}",
                ]
                frame = self._overlay_text(frame, lines)

            frames.append(frame)

        renderer.close()

        if vid_path is not None:
            import imageio

            if fps is None:
                fps = int(round(1.0 / self.ctrl_dt))
            with imageio.get_writer(vid_path, fps=fps) as writer:
                for frame in frames:
                    writer.append_data(frame)

        return frames
