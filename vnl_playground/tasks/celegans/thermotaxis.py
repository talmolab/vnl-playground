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

# Name of the texture + material the renderer bakes the thermal field into.
_FIELD_MATERIAL = "thermal_field"


def default_config() -> config_dict.ConfigDict:
    """Returns the default configuration for the Thermotaxis environment."""
    config = config_dict.create(
        episode_length=2000,
        action_repeat=1,
        init_z=0.0,
        # fixed start on the cold/left side (v1 deterministic). Scale is sized to the
        # worm: ~0.01-0.03 cm/s over a 200 s episode -> a ~1 cm start->setpoint
        # distance is reachable well within the episode.
        init_x=-0.5,
        init_y=0.0,
        # kwargs for Gradient.make_gradient; any iterable leaf is [low, high] bounds
        gradient_cfg=config_dict.create(
            gradient_type="linear",  # str | list[str] | "random"; default linear
            setpoint=23.0,  # target temp AT setpoint_loc
            setpoint_loc=(0.5, 0.0),  # (x, y) anchor; each element scalar or [lo, hi]
            min_temp=15.0,
            max_temp=25.0,
            # (Lx, Ly) half-extents of the *thermal region*, a subregion of the
            # (effectively infinite) arena plane. The gradient is only defined here;
            # outside, the temperature is held at the value on the region boundary.
            arena_size=(2.0, 2.0),
        ),
        # Thermosensory body: C. elegans senses temperature at the head, so the
        # gradient is read at this body's position (not the mid-body root).
        sensor_body="torso1_body",
        # observation realism knobs (all OFF by default, and independent of each
        # other): obs_noise_std = additive white noise; obs_delay = pure transport
        # delay in control steps (<= max_obs_delay); sensor_tau = first-order thermal
        # response lag in seconds (<= 0 disables; larger = more sluggish sensor).
        obs_noise_std=0.0,
        obs_delay=0,
        max_obs_delay=4,
        sensor_tau=0.0,
        reward_terms={
            # Dense guidance as a per-step gaussian on the temperature error:
            # reward = weight * exp(-(err / exp_scale)^2 / 2), i.e. `weight` at the
            # setpoint decaying with a width of `exp_scale` degrees. Being a per-step
            # *level* (not a difference), it is far less sensitive to the undulation
            # noise in err than a step-to-step delta is -- but it is farmable, see the
            # bonus ordering note under `termination_bonus`.
            #
            # exp_scale trades the reward LEVEL far from the goal against the reward
            # SLOPE there, and the slope is what actually teaches. For a fixed error e,
            # d/de of the gaussian is maximized at exp_scale = e / sqrt(2), so tightening
            # the width past that point loses guidance *and* level together -- it only
            # makes the idle reward look smaller.
            #
            # Measured start error on the default field is 2.625 C (the reward is scored
            # at the body centroid, which rests ~0.05 cm behind the root at spawn; the
            # nominal root figure is 2.5 = the 2.5 C/cm slope over the 1 cm
            # init_x -> setpoint_loc traverse). That puts the slope optimum at ~1.86, and
            # 2.0 is within 1% of it:
            #     exp_scale   r(start)   slope(start)   r(plate edge, 6.25)
            #       2.5         0.576       0.242            0.044
            #       2.0         0.423       0.277            0.008    <- default
            #       1.86        0.369       0.280            0.004
            #       1.3         0.130       0.202            ~0
            # So 2.0 idles at 0.42/step instead of 0.58 while pulling ~15% harder on the
            # approach, and still leaves a non-zero signal at the coldest reachable
            # corner for a worm that has wandered off. Note the idle level itself is
            # advantage-invariant (a constant offset adds c/(1-gamma) to V everywhere and
            # cancels in r + gamma*V' - V); what it costs is critic precision, since the
            # per-step signal is ~0.0014 against a mean of 0.42, and it is what a
            # terminating success forfeits.
            #
            # Retune if you change gradient steepness (min_temp/max_temp/arena_size) or
            # the start->setpoint distance: the rule is exp_scale ~= start error / 1.4.
            "temperature": {"weight": 1.0, "exp_scale": 2.0},
            # Alternative dense channel (registered, off by default): potential-based
            # *progress*, the per-step change in a potential of the error. Telescopes
            # over a trajectory, so unlike `temperature` it cannot be farmed by
            # loitering and is worth at most `weight` over an entire episode. Swap it in
            # by replacing the "temperature" entry above with:
            #   "progress": {
            #       "weight": 20.0,
            #       # "linear"   -> phi = 1 - err/max_err: constant-magnitude guidance.
            #       # "gaussian" -> phi = exp(-(err/exp_scale)^2/2): peaks near the goal,
            #       #               weakest where a lost worm needs it most.
            #       "mode": "linear",
            #       "max_err": 0.0,  # 0 = auto (max reachable error, per episode)
            #       "exp_scale": 2.5,  # only used by mode="gaussian"
            #   },
            # If you do, drop reached_setpoint back to ~30 (just above the progress
            # weight) -- see the ordering note below.
            # Graded cost for sitting outside the thermal tolerance band (see
            # `thermal` above). OFF by default.
            "thermal_stress": {"weight": 0.0},
            # Configurable terminal reward/penalty added on the step a termination
            # fires. Keys must be termination names; +ve = bonus, -ve = penalty. This
            # is the single place to tune per-termination bonuses.
            #
            # KEEP reached_setpoint >= temperature weight / (1 - gamma). `temperature`
            # is a positive per-step reward, so hovering just OUTSIDE the success band
            # collects ~weight every remaining step (err 0.6 pays 0.997 of what err 0
            # does) while succeeding ENDS the episode and forfeits the rest. Hovering is
            # worth weight/(1-gamma) in discounted return -- 20 at gamma=0.95, 100 at
            # gamma=0.99 -- so a smaller bonus pays the agent to dither in and out of the
            # band, resetting the hold counter to avoid ever terminating. 100 keeps
            # "succeed" the better option for every discount used in this repo; raise it
            # if you train with gamma > 0.99, and drop it to ~30 if you switch the dense
            # channel back to the (non-farmable) `progress` term.
            "termination_bonus": {
                "bonuses": {
                    "reached_setpoint": 100.0,  # success
                    "out_of_bounds": -10.0,  # left the thermal region
                    "nan": 0.0,
                }
            },
            # Per-step time cost -> reach the setpoint ASAP. Off for now (tune later):
            # the loiter exploit is closed by the reached_setpoint bonus above, not by
            # this term. A nonzero weight here just subtracts a constant from every
            # step, which shrinks the hovering payoff to (weight_temp - weight_time) per
            # step; keep the bonus >= that difference / (1 - gamma) if you enable it.
            "time": {"weight": 0.0},
            "control": {"weight": 0.0},
            "energy": {"weight": 0.0, "max_value": 50.0},
        },
        termination_criteria={
            # Success = the body centroid holds within `epsilon` degrees of the
            # setpoint for `hold_steps` consecutive control steps, i.e. the worm has to
            # *track* the isotherm rather than brush across it once. hold_steps=0 or 1
            # both reduce to the old fire-on-first-contact behavior.
            "reached_setpoint": {"epsilon": 0.5, "hold_steps": 20},
            # Left the thermal region. With require_all_bodies the episode only ends
            # once EVERY body segment is outside |x| > Lx + margin / |y| > Ly + margin
            # (with (Lx, Ly) = gradient_cfg.arena_size), so a worm hanging half off the
            # plate keeps going; margin adds further slack on top of that.
            # NOTE: the field is clamped outside the region, so an escaped worm has no
            # gradient to navigate back with -- keep the margin modest.
            "out_of_bounds": {"margin": 0.0, "require_all_bodies": True},
            "nan": {},
            # "thermal_dose": {"max_dose": 30.0},  # opt-in: see `thermal` above
        },
        **celegans_base.default_config(),
    )
    return config


class Thermotaxis(celegans_base.CelegansEnv):
    """Thermotaxis environment: navigate a thermal gradient to a target temperature.

    Reachability note: the ``time`` cost only yields "reach the setpoint ASAP"
    behavior if the setpoint is actually reachable within ``episode_length``. The
    default scale is sized to the worm (empirical speed ~0.01-0.03 cm/s, max ~0.07):
    the ~1 cm start->setpoint distance is covered in ~1000 steps even at the slow end
    (0.01 cm/s over a 200 s / 2000-step episode), with ample margin at average speed.
    If you change ``init_x`` / ``setpoint_loc`` / ``arena_size`` / ``episode_length``,
    keep a direct traversal comfortably inside the episode -- otherwise every step is
    net-negative and the agent's best move becomes ending the episode early by failing
    (the negative out_of_bounds entry in the ``termination_bonus`` reward discourages
    that; keep its magnitude >= ``time`` * ``episode_length``). With ``time`` at 0,
    discounting already supplies the "sooner is better" pressure.
    """

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

        # NOTE: the floor geom is deliberately left exactly as the shared arena defines
        # it. The thermal region is a task-level construct (a gradient plus an
        # out-of-bounds boundary), not a physical edge, so contact geometry, resting
        # height and every proprioceptive input match the other C. elegans tasks.
        lx, ly = self._region_extent()

        # Visual marker site on the worm so a top-down MuJoCo render still shows the
        # ~mm worm. In a dedicated group (3, hidden by default) so render() can isolate
        # it on the overhead view and hide it on the zoomed gait view, without touching
        # the model's other sites (which are all in group 0).
        worm_site = self._spec.body(f"{self.root_name}{self.suffix}").add_site()
        worm_site.name = "worm_marker"
        worm_site.type = mujoco.mjtGeom.mjGEOM_SPHERE
        worm_site.size = [0.05 * min(lx, ly), 0.0, 0.0]
        worm_site.rgba = [0.1, 0.1, 0.1, 1.0]
        worm_site.group = 3

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

        # Normalizer for the base class's `rewards/normalized` metric. Brax's
        # EvalWrapper SUMS metrics over an episode, so that metric accumulates to
        # (episode return) / max_reward -- meaningful only if max_reward is the best
        # achievable *episode* return.
        #
        # Summing the raw term weights (what the imitation env does) is right when every
        # term is a per-step gaussian in [0, weight], but the two dense channels here
        # differ in kind and cannot share one rule: `temperature` is per-step, so it is
        # worth up to weight * episode_length; `progress` is a potential difference, so
        # it telescopes to weight * (phi_end - phi_start) <= weight over an entire
        # episode, however long. Each is scaled by its own horizon before being added to
        # the best terminal bonus. Both terms are read (rather than just the enabled one)
        # so the normalizer follows whichever dense channel the config selects.
        temperature_weight = float(
            self._config.reward_terms.get("temperature", {}).get("weight", 0.0)
        )
        progress_weight = float(
            self._config.reward_terms.get("progress", {}).get("weight", 0.0)
        )
        bonuses = (
            self._config.reward_terms.get("termination_bonus", {})
            .get("bonuses", {})
            .values()
        )
        self.max_reward = (
            temperature_weight * int(self._config.episode_length)
            + progress_weight
            + max([0.0] + [float(bonus) for bonus in bonuses])
        )

        # The in-band counter is advanced in step() (terminations are pure reads of
        # info), so the success epsilon is cached here.
        self._success_eps = float(
            self._config.termination_criteria.get("reached_setpoint", {}).get(
                "epsilon", 0.0
            )
        )

        # First-order RC sensor-response lag (opt-in; sensor_tau <= 0 disables it).
        # Precompute the forward-Euler blend coefficient, clamped to 1 so a tau below
        # the control step just tracks the true temperature fully (no lag).
        self._sensor_tau = float(self._config.sensor_tau)
        self._rc_lag = self._sensor_tau > 0.0
        self._rc_coeff = (
            min(float(self.ctrl_dt) / self._sensor_tau, 1.0) if self._rc_lag else 1.0
        )

    # ----------------------------------------------------------------- helpers
    def _region_extent(self) -> tuple:
        """Concrete thermal-region half-extents (Lx, Ly) from ``gradient_cfg.arena_size``.

        This is the subregion of the arena plane the gradient is defined over (and,
        plus ``margin``, the out-of-bounds boundary). Uses the upper bound if an axis
        is given as sampling bounds, so the (static) region always contains any
        per-episode sampled gradient extent.
        """
        arena_size = self._config.gradient_cfg.arena_size

        def extent(v):
            return float(max(v)) if isinstance(v, (list, tuple)) else float(v)

        return extent(arena_size[0]), extent(arena_size[1])

    def _set_root_xy(self, data: mjx.Data, x: jp.ndarray, y: jp.ndarray) -> mjx.Data:
        """Place the worm's planar root at world ``(x, y)`` via slide-joint qpos."""
        qpos = data.qpos.at[self._rootx_qadr].set(x).at[self._rooty_qadr].set(y)
        data = data.replace(qpos=qpos)
        return mjx.forward(self.mjx_model, data)

    def _worm_xy(self, data: mjx.Data) -> jp.ndarray:
        """Worm root (mid-body) planar position ``(x, y)``."""
        return self._get_root_pos(data)[: self.config.dim]

    def _sensor_xy(self, data: mjx.Data) -> jp.ndarray:
        """Planar ``(x, y)`` of the thermosensory head body (temperature is read here)."""
        head = data.bind(
            self.mjx_model, self._spec.body(f"{self._config.sensor_body}{self.suffix}")
        )
        return head.xpos[: self.config.dim]

    def _temperature_at(self, xy: jp.ndarray, info: Mapping[str, Any]) -> jp.ndarray:
        """Temperature at the worm's exact (continuous) planar position ``xy``."""
        return Gradient.evaluate(info["shape_id"], info["temp_field"], xy)

    def _temp_error(self, data: mjx.Data, info: Mapping[str, Any]) -> jp.ndarray:
        """Absolute temperature error ``|T(head) - setpoint|`` (sensed at the head)."""
        temp = self._temperature_at(self._sensor_xy(data), info)
        return jp.abs(temp - info["setpoint"])

    def _field_max_error(self, info: Mapping[str, Any]) -> jp.ndarray:
        """Largest ``|T - setpoint|`` reachable on the plate, for this episode's field.

        Evaluated at the region corners, which hold the extremes for every shape
        implemented so far (a plane is extremal at a corner; a radial peak is coldest
        at the farthest point). Because :meth:`_temperature_at` clamps to the region,
        this bounds the error *everywhere in the world*, so the normalized potential
        stays in [0, 1] without the clip ever binding -- no flat dead zone outside.
        """
        lx, ly = self._region_extent()
        corners = jp.array(
            [[-lx, -ly], [-lx, ly], [lx, -ly], [lx, ly]], dtype=jp.float32
        )
        temps = jax.vmap(lambda p: self._temperature_at(p, info))(corners)
        return jp.max(jp.abs(temps - info["setpoint"]))

    def _thermal_stress(self, data: mjx.Data, info: Mapping[str, Any]) -> jp.ndarray:
        """Dimensionless thermal stress at the body: 0 inside the tolerance band.

        A one-sided hinge per side, each normalized by its own tolerance, so 1.0 means
        "one full tolerance beyond the band". Asymmetric by default because heat is the
        noxious side for C. elegans.
        """
        temp = self._temperature_at(self._body_xy(data), info)
        hot = float(self._config.thermal.hot_tol)
        cold = float(self._config.thermal.cold_tol)
        over = jp.maximum(temp - (info["setpoint"] + hot), 0.0) / hot
        under = jp.maximum((info["setpoint"] - cold) - temp, 0.0) / cold
        return over + under

    def _sense_temperature(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> jp.ndarray:
        """The temperature the agent observes: the sensor state (after any RC lag),
        optionally read ``obs_delay`` steps ago, plus optional white noise.

        ``sensor_tau`` / ``obs_delay`` / ``obs_noise_std`` / ``max_obs_delay`` are
        static config, so every branch resolves at trace time and is zero-cost when
        disabled. ``sensor_state`` equals the true temperature when ``sensor_tau<=0``.
        """
        del data
        sensed = info["sensor_state"]
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
            "sensor_state": jp.asarray(0.0, dtype=jp.float32),  # RC-lag filtered temp
            "prev_err": jp.asarray(0.0, dtype=jp.float32),  # progress-reward baseline
            # Per-episode normalizer for the progress potential (see _field_max_error).
            "max_err": jp.asarray(0.0, dtype=jp.float32),
            # Accumulated thermal dose (stress integrated over time) and the run of
            # consecutive control steps spent inside the success band.
            "dose": jp.asarray(0.0, dtype=jp.float32),
            "in_band_steps": jp.asarray(0, dtype=jp.int32),
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

        # Prefill the obs-delay ring buffer and settle the sensor at the initial
        # (head-sensed) temperature.
        init_temp = self._temperature_at(self._sensor_xy(data), info)
        info["temp_history"] = jp.full_like(info["temp_history"], init_temp)
        info["sensor_state"] = init_temp
        # Potential normalizer for this episode's field, then the baseline error so the
        # first-step progress reward is exactly 0. Both read at the reward point (the
        # body), not the head, matching what the term will measure.
        info["max_err"] = self._field_max_error(info)
        info["prev_err"] = self._temp_error(data, info)

        metrics: Dict[str, Any] = {}
        obs = self._get_obs(data, info)
        # Terminations first so the termination_bonus reward can read their flags.
        done = self._is_done(data, info, metrics)
        reward = self._get_reward(data, info, metrics)

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

        # Sensor response: first-order thermal lag (RC low-pass) toward the true
        # head temperature, then push the lagged value into the obs-delay buffer.
        true_temp = self._temperature_at(self._sensor_xy(data), info)
        if self._rc_lag:
            info["sensor_state"] = info["sensor_state"] + self._rc_coeff * (
                true_temp - info["sensor_state"]
            )
        else:
            info["sensor_state"] = true_temp
        idx = info["hist_index"]
        info["temp_history"] = info["temp_history"].at[idx].set(reading)
        info["hist_index"] = (idx + 1) % self._buf_len

        # Interoceptive state, updated before obs/terminations read it: integrate the
        # thermal dose, and advance the run of consecutive in-band steps (reset to 0 the
        # moment the body leaves the band, so `hold_steps` means *consecutive*).
        info["dose"] = info["dose"] + self._thermal_stress(data, info) * self.ctrl_dt
        in_band = self._temp_error(data, info) < self._success_eps
        info["in_band_steps"] = jp.where(in_band, info["in_band_steps"] + 1, 0).astype(
            jp.int32
        )

        obs = self._get_obs(data, info)
        done = self._is_done(data, info, state.metrics)
        reward = self._get_reward(data, info, state.metrics)
        reward = jp.nan_to_num(reward)

        # Advance the progress-reward baseline to this step's error.
        info["prev_err"] = self._temp_error(data, info)

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
    @staticmethod
    def _proximity(err, max_err, mode="linear", exp_scale=2.0):
        """Shaping potential in [0, 1]: 1 at the setpoint, 0 at ``max_err``.

        ``linear`` has a constant slope, so guidance is as strong at 5 degrees of error
        as at 0.5. ``gaussian`` is a bump of width ``exp_scale`` centred on the
        setpoint, whose slope decays away from it (kept for ablation).

        NOTE for non-linear fields: on a planar gradient ``err`` is proportional to
        distance, so the linear potential is linear in space too. On a radial
        (gaussian) field ``1 - err/max_err`` is itself gaussian in space and its slope
        does vanish far from the peak -- when you start using those, normalize by
        ``||grad T||`` here to recover uniform spatial guidance.
        """
        if mode == "linear":
            return 1.0 - jp.clip(err / max_err, 0.0, 1.0)
        if mode == "gaussian":
            return jp.exp(-((err / exp_scale) ** 2) / 2)
        raise ValueError(f"Unknown progress mode '{mode}'; expected linear | gaussian.")

    @_registry.reward("progress")
    def _progress_reward(
        self, data, info, metrics, weight, mode="linear", max_err=0.0, exp_scale=2.0
    ) -> float:
        """Potential-based progress toward the target isotherm.

        ``reward = weight * (phi(err_t) - phi(err_{t-1}))``. Positive when the worm
        closes on the isotherm, negative when it retreats. Being a difference of a
        potential it telescopes over any trajectory to ``phi_final - phi_start``
        regardless of length, so it cannot be farmed by loitering just outside the
        success region, and ``weight`` bounds the dense channel for a whole episode.

        ``max_err <= 0`` uses this episode's auto normalizer (:meth:`_field_max_error`),
        which keeps the potential -- and therefore ``weight`` -- meaningful when the
        gradient steepness changes. Pass a fixed value to pin it.

        Measured at ``config.reward_body`` (the body centroid by default), NOT at the
        thermosensory head: the head's gait sweep swings its temperature by ~0.01-0.03
        degrees per control step against a net progress of ~0.0025-0.0075, so scoring
        there drowns the signal in undulation.
        """
        err = self._temp_error(data, info)
        scale = info["max_err"] if max_err <= 0.0 else jp.asarray(max_err)
        reward = weight * (
            self._proximity(err, scale, mode, exp_scale)
            - self._proximity(info["prev_err"], scale, mode, exp_scale)
        )
        metrics["rewards/progress"] = metrics["rewards/progress/per_step"] = reward
        metrics["magnitudes/temp_error"] = metrics[
            "magnitudes/temp_error/per_step"
        ] = err
        return reward

    @_registry.reward("thermal_stress")
    def _thermal_stress_cost(self, data, info, metrics, weight) -> float:
        """Graded cost for holding the body outside the thermal tolerance band.

        A soft replacement for the old ``too_far`` termination: it pushes back toward
        tolerable temperatures instead of ending the episode at a cliff edge, so the
        down-gradient excursions that real worms make while searching stay affordable.
        The dose it integrates is accumulated (and observed) whatever this weight is.
        """
        stress = self._thermal_stress(data, info)
        cost = -weight * stress
        metrics["costs/thermal_stress"] = metrics["costs/thermal_stress/per_step"] = cost
        metrics["magnitudes/thermal_stress"] = metrics[
            "magnitudes/thermal_stress/per_step"
        ] = stress
        metrics["magnitudes/dose"] = metrics["magnitudes/dose/per_step"] = info["dose"]
        return cost

    @_registry.reward("temperature")
    def _temperature_reward(self, data, info, metrics, weight, exp_scale=2.5) -> float:
        """Gaussian proximity reward on the temperature error (``weight`` at zero error).

        ``reward = weight * exp(-(err / exp_scale)^2 / 2)``, a per-step *level*: the
        default dense channel. ``exp_scale`` is the width in degrees and should be set
        near the typical start error so the gaussian's steepest region covers the
        approach (see ``default_config``).

        NOTE: being a positive per-step reward it is farmable by loitering just outside
        the success region, since terminating forfeits the remaining per-step reward.
        The ``reached_setpoint`` termination bonus has to exceed ``weight / (1 - gamma)``
        to close that -- see the ordering note in ``default_config``. The non-farmable
        alternative is the potential-based ``progress`` term.

        Measured at ``config.reward_body`` (the body centroid by default), NOT at the
        thermosensory head, so the reward tracks where the worm is rather than the
        head's gait sweep.
        """
        err = self._temp_error(data, info)
        reward = weight * jp.exp(-((err / exp_scale) ** 2) / 2)
        metrics["rewards/temperature"] = metrics["rewards/temperature/per_step"] = (
            reward
        )
        metrics["magnitudes/temp_error"] = metrics[
            "magnitudes/temp_error/per_step"
        ] = err
        return reward

    @_registry.reward("termination_bonus")
    def _termination_bonus(self, data, info, metrics, bonuses) -> float:
        """Configurable terminal reward/penalty per termination.

        ``bonuses`` maps termination names to a reward added on the step that
        termination fires (positive = bonus, e.g. reaching the setpoint; negative =
        penalty, e.g. falling off). Because it reads the termination flags written by
        ``_is_done`` (which runs before ``_get_reward`` in both reset and step), the
        bonus is applied exactly once, on the terminating step.
        """
        del data, info
        total = jp.asarray(0.0, dtype=jp.float32)
        for name, bonus in bonuses.items():
            fired = metrics[f"terminations/{name}"]
            contribution = bonus * fired
            total = total + contribution
            metrics[f"rewards/termination_bonus/{name}"] = contribution
        metrics["rewards/termination_bonus"] = metrics[
            "rewards/termination_bonus/per_step"
        ] = total
        return total

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

    @_registry.reward("time")
    def _time_cost(self, data, info, metrics, weight) -> float:
        """Constant per-step cost (temporal pressure to reach the setpoint sooner)."""
        del data, info
        cost = jp.asarray(-weight, dtype=jp.float32)
        metrics["costs/time"] = metrics["costs/time/per_step"] = cost
        return cost

    # ------------------------------------------------------------- terminations
    @_registry.termination("reached_setpoint")
    def _reached_setpoint(self, data, info, epsilon, hold_steps=0) -> bool:
        """Success: the body held within ``epsilon`` of the setpoint for ``hold_steps``.

        Reads the consecutive in-band counter advanced in :meth:`step` (``epsilon`` is
        applied there, to the reward-point error). Requiring a run of steps turns the
        task from "brush across the isotherm once" into isothermal *tracking*, and
        closes the exploit where a single head flick through the band ends the episode
        with the body still outside it. ``hold_steps`` of 0 or 1 both reduce to
        terminating on first contact.
        """
        del data, epsilon
        return info["in_band_steps"] >= max(int(hold_steps), 1)

    @_registry.termination("thermal_dose")
    def _thermal_dose(self, data, info, max_dose) -> bool:
        """Gave up: accumulated thermal dose exceeded ``max_dose``.

        The graded counterpart of the old ``too_far`` cliff -- the worm may go as cold
        or as hot as it likes, but not for arbitrarily long. Not enabled by default;
        add a ``thermal_dose`` entry to ``termination_criteria`` to switch it on (and
        give it a ``termination_bonus`` entry if you want a penalty with it).
        """
        del data
        return info["dose"] > max_dose

    @_registry.termination("out_of_bounds")
    def _out_of_bounds(self, data, info, margin, require_all_bodies=True) -> bool:
        """Left the thermal region: outside ``|x| > Lx + margin`` / ``|y| > Ly + margin``.

        Replaces the old fall-off-the-floor check: the arena is an infinite plane, so
        the region boundary is a task-level rectangle rather than a physical edge. With
        ``require_all_bodies`` the episode ends only once *every* segment is outside, so
        a worm hanging half off the plate keeps going and a head flick across the
        boundary costs nothing; otherwise the reward point alone is tested.
        """
        del info
        lx, ly = self._region_extent()
        bounds = jp.array([lx + margin, ly + margin], dtype=jp.float32)
        if not require_all_bodies:
            return jp.any(jp.abs(self._body_xy(data)) > bounds)
        bodies = self._get_bodies_pos(data, flatten=False)
        xy = jp.stack([p[: self.config.dim] for p in bodies.values()])
        return jp.all(jp.any(jp.abs(xy) > bounds, axis=-1))

    @_registry.termination("nan")
    def _nan_termination(self, data, info) -> bool:
        """NaNs detected in the simulation state."""
        del info
        return jp.any(jp.isnan(data.qpos))

    # ---------------------------------------------------------------- rendering
    def _field_texture(
        self, info, resolution, cells, checker, fine_subdiv, alpha
    ) -> np.ndarray:
        """Bake the thermal field into an ``(N, N, 3)`` uint8 image for the floor.

        Row/column convention (verified empirically, and silently mirror-inverting if
        got wrong): the column index increases with world **+x**, and the row index
        increases with world **-y** -- i.e. row 0 is the ``+y`` edge, the usual image
        convention, matching ``plt.imshow(origin="upper")``.

        The image composites three things: the arena material's own two greys in a
        coarse checker (so ``alpha=0`` reproduces the plain floor), the temperature
        colour blended over them by ``alpha``, and a brightness modulation on the same
        checker plus a fainter sub-checker. The two checker scales exist because one
        static texture has to serve both the whole-plate overhead view and a tracking
        camera zoomed to ~2 mm: the coarse cells read at arena scale, the fine ones
        provide landmarks when zoomed in far enough that a coarse cell fills the frame.
        """
        lx, ly = self._region_extent()
        n = int(resolution)
        xs = np.linspace(-lx, lx, n, endpoint=False) + lx / n
        ys = np.linspace(ly, -ly, n, endpoint=False) - ly / n
        gx, gy = np.meshgrid(xs, ys)  # (n, n): axis 1 -> +x, axis 0 -> -y
        temps = np.asarray(
            jax.vmap(
                lambda p: Gradient.evaluate(
                    info["shape_id"], jp.asarray(info["temp_field"]), p
                )
            )(jp.asarray(np.stack([gx.ravel(), gy.ravel()], axis=-1)))
        )
        temp_rgb = self._temperature_colors(temps, 1.0)[:, :3].reshape(n, n, 3)

        idx = np.arange(n)
        cell = max(1, n // max(int(cells), 1))
        parity = (idx[:, None] // cell + idx[None, :] // cell) % 2
        base = np.where(
            parity[..., None] == 0,
            np.array([0.1, 0.2, 0.3]),  # arena.xml's grid rgb1 / rgb2
            np.array([0.2, 0.3, 0.4]),
        )
        rgb = (1.0 - alpha) * base + alpha * temp_rgb
        rgb = rgb * (1.0 + checker * (2.0 * parity - 1.0))[..., None]
        if int(fine_subdiv) > 1:
            fine = max(1, cell // int(fine_subdiv))
            fine_parity = (idx[:, None] // fine + idx[None, :] // fine) % 2
            rgb = rgb * (1.0 + (checker / 3.0) * (2.0 * fine_parity - 1.0))[..., None]
        return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)

    def _render_model(self, info, tex_resolution, **texture_kwargs) -> mujoco.MjModel:
        """Compile a *render-only* model: floor shrunk to the plate, field baked in.

        Everything happens on a copy of the spec, so the environment's own spec,
        ``mj_model`` and ``mjx_model`` are untouched and rollouts are unaffected --
        this is only ever called from :meth:`render`. (Shrinking a plane is visual-only
        regardless: planes are infinite half-spaces for collision in both MuJoCo and
        MJX, so contacts and resting height do not depend on the rendered size. There
        is still no reason to mutate shared state.)

        Colouring the floor geom itself, rather than covering it with a grid of tile
        geoms, means the texture resolution is decoupled from per-frame cost: a 1024
        texture over a +/-2 cm plate is ~0.04 mm per texel, against ~0.8 mm tiles
        before, and costs zero geoms per frame instead of ``grid_resolution ** 2``.
        """
        spec = self._spec.copy()
        lx, ly = self._region_extent()
        floor = spec.geom("floor")
        floor.size = [lx, ly, float(floor.size[2])]

        image = self._field_texture(info, tex_resolution, **texture_kwargs)
        texture = spec.add_texture()
        texture.name = _FIELD_MATERIAL
        texture.type = mujoco.mjtTexture.mjTEXTURE_2D
        texture.width = texture.height = int(tex_resolution)
        texture.nchannel = 3
        texture.data = image.reshape(-1).tobytes()

        material = spec.add_material()
        material.name = _FIELD_MATERIAL
        material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = _FIELD_MATERIAL
        # texrepeat 1 with texuniform off maps exactly one copy across the plane, so
        # texel (0, 0) lands on the (-x, +y) corner of the plate. Verified, not assumed.
        material.texrepeat = [1, 1]
        material.texuniform = False
        # Kill the specular highlight: on a flat plate it renders as a bright radial
        # hotspot that reads as part of the temperature field.
        material.specular = 0.0
        material.shininess = 0.0
        material.reflectance = 0.0
        floor.material = _FIELD_MATERIAL
        return spec.compile()

    @staticmethod
    def _temperature_colors(temps: np.ndarray, alpha: float) -> np.ndarray:
        """Map temperatures to a saturated blue->red RGBA (in [0, 1]).

        Normalized over the min/max of ``temps``. Uses a steep ramp centered at the
        midpoint so the field reads as mostly blue (cold) on one side and red (warm)
        on the other with only a thin transition band -- no washed-out white middle.
        Dependency free (no matplotlib) so rendering never fails on a missing colormap.
        """
        t = np.asarray(temps, dtype=np.float64)
        tmin, tmax = float(t.min()), float(t.max())
        tn = (t - tmin) / (tmax - tmin + 1e-8)
        cold = np.array([40.0, 90.0, 215.0])  # saturated blue
        warm = np.array([215.0, 45.0, 55.0])  # saturated red
        # steep ramp: pure blue below ~0.44, pure red above ~0.56, thin blend between.
        frac = np.clip((tn - 0.5) / 0.12 + 0.5, 0.0, 1.0)[..., None]
        rgb = (cold + (warm - cold) * frac) / 255.0
        rgba = np.concatenate([rgb, np.full((rgb.shape[0], 1), alpha)], axis=-1)
        return rgba.astype(np.float32)

    def _overlay_text(self, frame: np.ndarray, lines: Sequence[str]) -> np.ndarray:
        """Draw a translucent dark panel with crisp white text (top-left corner).

        Uses cv2 if available, else Pillow; if neither is installed, warns once and
        returns the frame unchanged so rendering never hard-fails on annotations.
        """
        try:
            import cv2

            fs, th, line_h = 0.55, 1, 22
            widths = [
                cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, fs, th)[0][0]
                for l in lines
            ]
            panel_w = min(frame.shape[1], max(widths) + 16)
            panel_h = 12 + line_h * len(lines)
            panel = frame.copy()
            cv2.rectangle(panel, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
            frame = cv2.addWeighted(panel, 0.5, frame, 0.5, 0)
            for j, line in enumerate(lines):
                cv2.putText(frame, line, (8, 24 + j * line_h),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th,
                            cv2.LINE_AA)
            return frame
        except ImportError:
            pass
        try:
            from PIL import Image, ImageDraw

            img = Image.fromarray(frame).convert("RGB")
            draw = ImageDraw.Draw(img, "RGBA")
            line_h = 16
            draw.rectangle([0, 0, 210, 6 + line_h * len(lines)], fill=(0, 0, 0, 140))
            for j, line in enumerate(lines):
                draw.text((6, 3 + j * line_h), line, fill=(255, 255, 255))
            return np.asarray(img)
        except ImportError:
            if not getattr(self, "_warned_no_text", False):
                warnings.warn("Neither cv2 nor PIL available; skipping annotations.")
                self._warned_no_text = True
            return frame

    def _resolve_camera(self, camera, arena_lx: float, arena_ly: float):
        """Resolve a camera spec to ``(camera_arg, is_overhead)``.

        ``None`` / ``"overhead"`` -> a free top-down MjvCamera framing the whole arena
        (the worm is drawn as a marker there). Any other value (a camera name/id such
        as ``"track-worm"``) is passed through and treated as a zoomed/tracking view,
        where the worm marker is omitted so the body/gait is visible.
        """
        if camera is None or camera == "overhead":
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [0.0, 0.0, 0.0]
            cam.distance = 2.9 * max(arena_lx, arena_ly)
            cam.azimuth = 90.0
            cam.elevation = -90.0
            return cam, True
        return camera, False

    def render(
        self,
        trajectory: Sequence[mjx_env.State],
        height: int = 480,
        width: int = 640,
        camera: Optional[Union[str, int, Sequence]] = None,
        scene_option: Optional["mujoco.MjvOption"] = None,
        annotate: bool = True,
        overlay_gradient: bool = True,
        grid_resolution: int = 48,
        overlay_alpha: float = 0.85,
        grid_checker: float = 0.18,
        tex_resolution: int = 1024,
        fine_subdiv: int = 4,
        show_markers: bool = True,
        fps: Optional[float] = None,
        vid_path: Optional[str] = None,
    ) -> List[np.ndarray]:
        """Render a rollout with the thermal gradient painted onto the floor.

        The field is baked into a texture on the floor geom itself (blue = cold, red =
        warm) rather than drawn as an overlay of tile geoms, so the plate is colored
        *and* keeps a real checker, at a resolution independent of per-frame cost.
        A green marker sits at the setpoint and a black one at the worm (the ~mm worm
        is otherwise sub-pixel at arena scale). When ``annotate`` is set, each frame is
        captioned with the episode step, per-step reward, cumulative reward,
        ``current_temp / setpoint``, and any terminations that have fired.

        Rendering uses a private copy of the spec with the floor shrunk to the plate;
        the environment's own model is untouched, so this cannot affect a rollout.
        Beyond the plate nothing is drawn, which makes the out-of-bounds boundary
        visible.

        Args:
            trajectory: Sequence of single-environment states (one per control step),
                e.g. from a rollout. The gradient parameters are read from
                ``trajectory[0].info`` and assumed constant across the episode.
            height, width: Frame size in pixels.
            camera: Camera name/id, or a list of them for side-by-side views tiled
                horizontally. ``None`` / ``"overhead"`` is a top-down whole-arena view
                (worm drawn as a marker); a named camera like ``"track-worm"`` is a
                zoomed/tracking view showing the actual worm body. Example:
                ``["overhead", "track-worm"]`` for arena + gait side by side. Each view
                is ``width`` px wide, so the composed frame is ``width * n_cameras``.
            scene_option: Optional MjvOption for the scene.
            annotate: Overlay the text stats described above.
            overlay_gradient: Paint the thermal field onto the floor. When False the
                environment's own model is used as-is (full-size checkered arena).
            grid_resolution: Coarse checker cells per axis. 48 over a ±2 cm plate is a
                ~0.8 mm cell, i.e. about one worm length -- readable in the overhead
                view. This no longer costs anything per frame.
            overlay_alpha: How far the temperature colour is blended over the arena's
                own checker greys (0 = plain floor, 1 = pure temperature colour).
            grid_checker: Brightness contrast of the checker, as a fraction of the cell
                colour (0 disables).
            tex_resolution: Texels per axis in the baked texture. 1024 over ±2 cm is
                ~0.04 mm/texel, sharp even on a camera zoomed to the worm.
            fine_subdiv: Sub-checker subdivisions inside each coarse cell, drawn at a
                third of the contrast (0 or 1 disables). This is what gives a tracking
                camera visible landmarks once a coarse cell fills the frame.
            show_markers: Draw the worm and setpoint position markers.
            fps: Video frame rate (defaults to ``1 / ctrl_dt``).
            vid_path: If given, also write the frames to this path.

        Returns:
            List of rendered RGB frames (HxWx3 uint8).
        """
        if len(trajectory) == 0:
            return []

        # Episode-constant gradient params (from the first state).
        info0 = trajectory[0].info
        params = np.asarray(info0["temp_field"], dtype=np.float32)
        setpoint = float(np.asarray(info0["setpoint"]))
        setpoint_loc = (float(params[1]), float(params[2]))  # lx, ly packed in params

        # Frame the thermal region, not the (much larger) arena plane.
        arena_lx, arena_ly = self._region_extent()

        if overlay_gradient:
            mj_model = self._render_model(
                info0,
                tex_resolution,
                cells=grid_resolution,
                checker=grid_checker,
                fine_subdiv=fine_subdiv,
                alpha=overlay_alpha,
            )
        else:
            mj_model = self.mj_model
        mj_model.vis.global_.offwidth = max(int(mj_model.vis.global_.offwidth), width)
        mj_model.vis.global_.offheight = max(
            int(mj_model.vis.global_.offheight), height
        )
        mj_data = mujoco.MjData(mj_model)
        # Head body: the temperature annotation reflects what the worm senses there.
        head_bid = mj_model.body(f"{self._config.sensor_body}{self.suffix}").id

        renderer = mujoco.Renderer(mj_model, height=height, width=width, max_geom=2000)

        # Resolve camera(s). Passing a list renders each view and tiles them
        # horizontally (e.g. ["overhead", "track-worm"] for arena + zoomed gait).
        cam_specs = list(camera) if isinstance(camera, (list, tuple)) else [camera]
        resolved_cams = [
            self._resolve_camera(c, arena_lx, arena_ly) for c in cam_specs
        ]

        def field_temp(xy: np.ndarray) -> float:
            """Sensed temperature (region-clamped, as the env reads it)."""
            return float(self._temperature_at(jp.asarray(xy), info0))

        term_names = list(self.config.termination_criteria.keys())
        frames: List[np.ndarray] = []
        cumulative = 0.0
        extent = min(arena_lx, arena_ly)
        for i, state in enumerate(trajectory):
            mj_data.qpos = np.asarray(state.data.qpos)
            mujoco.mj_forward(mj_model, mj_data)
            head_x = float(mj_data.xpos[head_bid][0])
            head_y = float(mj_data.xpos[head_bid][1])

            views = []
            for cam, is_overhead in resolved_cams:
                # Site visibility: on the overhead view show only the worm_marker
                # (group 3) and hide the model's body sites (groups 0-2); on the zoomed
                # gait view hide all sites so the worm body/gait is unobstructed. A
                # user-supplied scene_option is respected as-is.
                if scene_option is not None:
                    view_opt = scene_option
                else:
                    view_opt = mujoco.MjvOption()
                    show_site = show_markers and is_overhead
                    view_opt.sitegroup[:] = [0, 0, 0, 1 if show_site else 0, 0, 0]
                renderer.update_scene(mj_data, camera=cam, scene_option=view_opt)
                scene = renderer.scene

                # setpoint marker (green, semi-transparent) on the overhead view only.
                if show_markers and is_overhead and scene.ngeom < scene.maxgeom:
                    mujoco.mjv_initGeom(
                        scene.geoms[scene.ngeom],
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([0.06 * extent, 0.0, 0.0]),
                        np.array([setpoint_loc[0], setpoint_loc[1], 0.03]),
                        np.eye(3).flatten(),
                        np.array([0.1, 0.9, 0.2, 0.55], dtype=np.float32),
                    )
                    scene.ngeom += 1

                views.append(np.ascontiguousarray(renderer.render()))

            frame = views[0] if len(views) == 1 else np.concatenate(views, axis=1)
            cumulative += float(state.reward)

            if annotate:
                temp = field_temp(np.array([head_x, head_y]))
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

    def render_combined(
        self,
        trajectory: Sequence[mjx_env.State],
        height: int = 480,
        traj_width: int = 560,
        gait_width: int = 480,
        gait_camera: Union[str, int] = "track-worm",
        gait_overlay: bool = True,
        gait_grid_resolution: int = 48,
        fps: Optional[float] = None,
        vid_path: Optional[str] = None,
    ) -> List[np.ndarray]:
        """Side-by-side: matplotlib trajectory (left) + MuJoCo gait view (right).

        Left is the CPU top-down field/trajectory (:meth:`plot_trajectory`); right is a
        MuJoCo camera view of the actual worm body/gait (needs GL/EGL). Frames are the
        same height and concatenated horizontally.

        Args:
            trajectory: Sequence of single-environment states.
            height: Frame height (both panels).
            traj_width, gait_width: Per-panel widths.
            gait_camera: MuJoCo camera for the gait panel (e.g. ``"top-worm"`` for the
                top-down body wave, ``"track-worm"`` for a profile).
            gait_overlay: Draw the checkered gradient tiles on the gait panel's floor
                too -- they are what makes the worm's motion legible on a *tracking*
                camera, where the worm stays centered and only the floor moves.
            gait_grid_resolution: Tiles per axis for that overlay. The gait camera is
                zoomed to ~mm, so the default 48 (~0.8 mm cells over a ±2 cm region)
                shows only ~3 cells across the panel; raise it to 96 or 160 for a finer
                reference grid, at ``resolution**2`` geoms per frame.
            fps: Video frame rate (defaults to ``1 / ctrl_dt``).
            vid_path: If given, also write the composed frames to this path.

        Returns:
            List of composed RGB frames (height x (traj_width + gait_width) x 3).
        """
        traj_frames = self.plot_trajectory(
            trajectory, height=height, width=traj_width, annotate=True
        )
        gait_frames = self.render(
            trajectory, height=height, width=gait_width, camera=gait_camera,
            overlay_gradient=gait_overlay, grid_resolution=gait_grid_resolution,
            annotate=False, show_markers=False,
        )
        combined = [
            np.concatenate([t, g], axis=1)
            for t, g in zip(traj_frames, gait_frames)
        ]

        if vid_path is not None:
            import imageio

            if fps is None:
                fps = int(round(1.0 / self.ctrl_dt))
            with imageio.get_writer(vid_path, fps=fps) as writer:
                for frame in combined:
                    writer.append_data(frame)

        return combined

    def plot_trajectory(
        self,
        trajectory: Sequence[mjx_env.State],
        height: int = 480,
        width: int = 560,
        show_trail: bool = True,
        point_size: float = 5.0,
        annotate: bool = True,
        fps: Optional[float] = None,
        vid_path: Optional[str] = None,
    ) -> List[np.ndarray]:
        """Top-down matplotlib view of the worm's path over the thermal field (CPU only).

        Renders the smooth temperature field as an image (saturated blue->red, minimal
        white), with the worm as a small point plus its trail, the setpoint marker and
        its iso-``setpoint`` contour, cm axes and a colorbar. This is for the
        *trajectory* only -- it needs no MuJoCo/GL and does not show the worm body/gait
        (use :meth:`render` with a camera like ``"top-worm"`` for that).

        Args:
            trajectory: Sequence of single-environment states.
            height, width: Frame size in pixels.
            show_trail: Draw the path travelled so far.
            point_size: Worm marker size (small -> a point).
            annotate: Title each frame with step / T / reward / terminations.
            fps: Video frame rate (defaults to ``1 / ctrl_dt``).
            vid_path: If given, also write the frames to this path.

        Returns:
            List of rendered RGB frames (HxWx3 uint8).
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.colors import LinearSegmentedColormap

        if len(trajectory) == 0:
            return []

        info0 = trajectory[0].info
        shape_id = jp.int32(int(np.asarray(info0["shape_id"])))
        params = jp.asarray(np.asarray(info0["temp_field"], dtype=np.float32))
        setpoint = float(np.asarray(info0["setpoint"]))
        setpoint_loc = (float(params[1]), float(params[2]))
        arena_lx, arena_ly = self._region_extent()

        # Smooth temperature field for imshow / contour (evaluated at the exact
        # continuous grid of sample points -- this is display only).
        n = 200
        cgx, cgy = np.meshgrid(
            np.linspace(-arena_lx, arena_lx, n),
            np.linspace(-arena_ly, arena_ly, n),
        )
        field = np.asarray(
            jax.vmap(lambda p: Gradient.evaluate(shape_id, params, p))(
                jp.asarray(np.stack([cgx.ravel(), cgy.ravel()], axis=-1))
            )
        ).reshape(n, n)
        fvmin, fvmax = float(field.min()), float(field.max())

        cmap = LinearSegmentedColormap.from_list(
            "thermo",
            [(0.0, "#285fd7"), (0.42, "#3a66d2"), (0.5, "#8a4f9e"),
             (0.58, "#d24a52"), (1.0, "#d72d37")],
        )

        # Plot the head (sensing point) trajectory -- that's what reads the gradient,
        # and for a real gait it shows the head's side-to-side sweep.
        head_bid = self.mj_model.body(f"{self._config.sensor_body}{self.suffix}").id
        wpos = np.array(
            [np.asarray(s.data.xpos)[head_bid, :2] for s in trajectory]
        )

        term_names = list(self.config.termination_criteria.keys())
        dpi = 100
        frames: List[np.ndarray] = []
        cumulative = 0.0
        for i, state in enumerate(trajectory):
            cumulative += float(state.reward)
            fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
            ax = fig.add_subplot(111)
            im = ax.imshow(
                field, origin="lower", extent=[-arena_lx, arena_lx, -arena_ly, arena_ly],
                cmap=cmap, vmin=fvmin, vmax=fvmax, aspect="equal",
            )
            ax.contour(cgx, cgy, field, levels=[setpoint], colors="lime",
                       linewidths=1.2, linestyles="--")
            ax.plot(setpoint_loc[0], setpoint_loc[1], marker="*", color="lime",
                    markersize=13, markeredgecolor="black", markeredgewidth=0.6)
            if show_trail and i > 0:
                ax.plot(wpos[: i + 1, 0], wpos[: i + 1, 1], "--", color="black",
                        lw=1.2, alpha=0.7)
            ax.plot(wpos[i, 0], wpos[i, 1], "o", color="black", markersize=point_size)
            ax.set_xlim(-arena_lx, arena_lx)
            ax.set_ylim(-arena_ly, arena_ly)
            ax.set_xlabel("x (cm)")
            ax.set_ylabel("y (cm)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="temperature")
            if annotate:
                temp = float(self._temperature_at(jp.asarray(wpos[i]), state.info))
                fired = [
                    nm for nm in term_names
                    if float(np.asarray(state.metrics.get(f"terminations/{nm}", 0.0))) > 0
                ]
                ax.set_title(
                    f"step {i}   T={temp:.2f}/{setpoint:.2f}   "
                    f"r={float(state.reward):+.3f}  cum={cumulative:+.2f}   "
                    f"[{', '.join(fired) if fired else '-'}]",
                    fontsize=8,
                )
            fig.tight_layout()
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            frame = np.asarray(canvas.buffer_rgba())[..., :3].copy()
            frames.append(frame)
            plt.close(fig)

        if vid_path is not None:
            import imageio

            if fps is None:
                fps = int(round(1.0 / self.ctrl_dt))
            with imageio.get_writer(vid_path, fps=fps) as writer:
                for frame in frames:
                    writer.append_data(frame)

        return frames
