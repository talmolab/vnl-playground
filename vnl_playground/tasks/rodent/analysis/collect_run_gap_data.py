"""Rollout data collector for RunGap corridor kinematics analysis.

Loads a trained binocular vision checkpoint, rolls out the agent in the
RunGap corridor environment, and collects per-timestep head/torso/camera
kinematics for offline motion parallax analysis.

Supports three visual conditions:
  - binocular: baseline stereo vision
  - monocular_left: right eye input zeroed (channel 1)
  - monocular_right: left eye input zeroed (channel 0)

Output is saved as .npz files containing per-timestep kinematic arrays
and per-episode gap geometry.

Usage::

    python -m vnl_playground.tasks.rodent.analysis.collect_run_gap_data \\
        --checkpoint_path /path/to/checkpoint \\
        --condition binocular \\
        --n_episodes 200 \\
        --output_dir ./outputs/motion_parallax
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

# Environment flags must be set before importing JAX/MuJoCo.
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx

# Enable persistent compilation cache.
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

VALID_CONDITIONS = ("binocular", "monocular_left", "monocular_right")


# ---------------------------------------------------------------------------
# Body / Camera ID resolution
# ---------------------------------------------------------------------------


def resolve_body_and_camera_ids(
    mj_model: mujoco.MjModel,
) -> Dict[str, int]:
    """Resolve MuJoCo body and camera IDs for kinematic extraction.

    The rodent XML appends a "-rodent" suffix to all body and camera names
    via ``add_rodent(suffix="-rodent")``.

    Args:
        mj_model: Compiled MuJoCo model.

    Returns:
        Dictionary mapping logical names to integer IDs::

            {
                "skull_body_id": int,
                "torso_body_id": int,
                "eye_left_cam_id": int,
                "eye_right_cam_id": int,
            }

    Raises:
        ValueError: If any required body or camera is not found.
    """
    ids = {}
    name_map = {
        "skull_body_id": ("skull-rodent", mujoco.mjtObj.mjOBJ_BODY),
        "torso_body_id": ("torso-rodent", mujoco.mjtObj.mjOBJ_BODY),
        "eye_left_cam_id": ("eye_left-rodent", mujoco.mjtObj.mjOBJ_CAMERA),
        "eye_right_cam_id": ("eye_right-rodent", mujoco.mjtObj.mjOBJ_CAMERA),
    }

    for key, (name, obj_type) in name_map.items():
        obj_id = mujoco.mj_name2id(mj_model, obj_type, name)
        if obj_id == -1:
            raise ValueError(
                f"Could not find {mujoco.mjtObj(obj_type).name} "
                f"named '{name}' in the model."
            )
        ids[key] = obj_id

    return ids


# ---------------------------------------------------------------------------
# Platform / gap geometry extraction
# ---------------------------------------------------------------------------


def extract_gap_geometry(
    mj_model: mujoco.MjModel,
    data: mjx.Data,
    n_platforms: int,
    platform_half_length: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract gap leading edges and gap lengths from simulation state.

    Reads platform body xpos values and computes the gap geometry between
    consecutive platforms (including the start platform).

    Args:
        mj_model: Compiled MuJoCo model.
        data: Current simulation data (mjx.Data, single-world).
        n_platforms: Number of gap platforms (excludes start platform).
        platform_half_length: Half-length of each gap platform.

    Returns:
        Tuple of:
            - gap_leading_edges: (N,) x-positions where each gap begins
              (trailing edge of the preceding platform).
            - gap_lengths: (N,) width of each gap in meters.
    """
    # Start platform trailing edge
    start_body_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_BODY, "platform_start"
    )
    # Start platform has half-length hardcoded to 1.0 (start_length=2.0)
    start_half_length = 1.0

    xpos = np.asarray(data.xpos)
    start_trailing_edge = float(xpos[start_body_id, 0]) + start_half_length

    # Platform body centers
    platform_centers = []
    for i in range(n_platforms):
        bid = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, f"platform_{i}"
        )
        platform_centers.append(float(xpos[bid, 0]))
    platform_centers = np.array(platform_centers)

    # Leading edges = center - half_length
    platform_leading_edges = platform_centers - platform_half_length
    # Trailing edges = center + half_length
    platform_trailing_edges = platform_centers + platform_half_length

    # All trailing edges including start platform
    all_trailing = np.concatenate(
        [[start_trailing_edge], platform_trailing_edges]
    )

    # Gap leading edge = trailing edge of previous platform
    gap_leading_edges = all_trailing[:-1]
    # Gap length = leading edge of next platform - trailing edge of previous
    gap_lengths = platform_leading_edges - gap_leading_edges

    return gap_leading_edges, gap_lengths


# ---------------------------------------------------------------------------
# Per-timestep kinematics extraction
# ---------------------------------------------------------------------------


def extract_timestep_kinematics(
    data: mjx.Data,
    body_cam_ids: Dict[str, int],
) -> Dict[str, np.ndarray]:
    """Extract kinematic quantities from a single simulation timestep.

    Reads body world position, rotation matrix, subtree linear velocity,
    and camera world positions from the mjx.Data.

    Args:
        data: Simulation data for a single timestep (unbatched mjx.Data).
        body_cam_ids: Dictionary of resolved body/camera IDs from
            :func:`resolve_body_and_camera_ids`.

    Returns:
        Dictionary of numpy arrays::

            {
                "skull_xpos": (3,),
                "skull_xmat": (3, 3),
                "torso_xpos": (3,),
                "torso_xmat": (3, 3),
                "torso_linvel": (3,),
                "cam_eye_left_xpos": (3,),
                "cam_eye_right_xpos": (3,),
            }
    """
    skull_id = body_cam_ids["skull_body_id"]
    torso_id = body_cam_ids["torso_body_id"]
    left_cam_id = body_cam_ids["eye_left_cam_id"]
    right_cam_id = body_cam_ids["eye_right_cam_id"]

    # Body positions and rotation matrices
    # data.xpos shape: (nbody, 3), data.xmat shape: (nbody, 9) in MJX
    # (MJX stores xmat as flat (nbody, 9); reshape to 3x3)
    skull_xpos = np.asarray(data.xpos[skull_id], dtype=np.float32)
    skull_xmat_flat = np.asarray(data.xmat[skull_id], dtype=np.float32)
    skull_xmat = skull_xmat_flat.reshape(3, 3)

    torso_xpos = np.asarray(data.xpos[torso_id], dtype=np.float32)
    torso_xmat_flat = np.asarray(data.xmat[torso_id], dtype=np.float32)
    torso_xmat = torso_xmat_flat.reshape(3, 3)

    # Subtree linear velocity for torso
    torso_linvel = np.asarray(
        data._impl.subtree_linvel[torso_id], dtype=np.float32
    )

    # Camera positions: data.cam_xpos shape (ncam, 3)
    cam_left_xpos = np.asarray(
        data.cam_xpos[left_cam_id], dtype=np.float32
    )
    cam_right_xpos = np.asarray(
        data.cam_xpos[right_cam_id], dtype=np.float32
    )

    return {
        "skull_xpos": skull_xpos,
        "skull_xmat": skull_xmat,
        "torso_xpos": torso_xpos,
        "torso_xmat": torso_xmat,
        "torso_linvel": torso_linvel,
        "cam_eye_left_xpos": cam_left_xpos,
        "cam_eye_right_xpos": cam_right_xpos,
    }


# ---------------------------------------------------------------------------
# Monocular masking
# ---------------------------------------------------------------------------


def apply_monocular_mask(
    vision_obs: jp.ndarray,
    condition: str,
) -> jp.ndarray:
    """Apply monocular masking to binocular vision observations.

    For binocular grayscale, the vision tensor has shape (H, W, 2):
      - channel 0 = left eye
      - channel 1 = right eye

    Args:
        vision_obs: Vision observation array with shape (..., H, W, 2*C).
        condition: One of "binocular", "monocular_left", "monocular_right".

    Returns:
        Vision array with the appropriate eye channel(s) zeroed out.
        For "binocular", the input is returned unchanged.

    Raises:
        ValueError: If condition is not a valid condition string.
    """
    if condition == "binocular":
        return vision_obs
    elif condition == "monocular_left":
        # Keep left eye (channel 0), zero right eye (channel 1)
        n_channels = vision_obs.shape[-1] // 2
        mask = jp.concatenate(
            [jp.ones(n_channels), jp.zeros(n_channels)]
        )
        return vision_obs * mask
    elif condition == "monocular_right":
        # Zero left eye (channel 0), keep right eye (channel 1)
        n_channels = vision_obs.shape[-1] // 2
        mask = jp.concatenate(
            [jp.zeros(n_channels), jp.ones(n_channels)]
        )
        return vision_obs * mask
    else:
        raise ValueError(
            f"Invalid condition: {condition!r}. "
            f"Must be one of {VALID_CONDITIONS}"
        )


# ---------------------------------------------------------------------------
# Environment and policy setup
# ---------------------------------------------------------------------------


def load_config(checkpoint_path: str) -> Dict[str, Any]:
    """Load training configuration from checkpoint directory.

    Args:
        checkpoint_path: Path to the checkpoint directory containing
            ``config.json``.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If config.json is not found.
    """
    config_file = Path(checkpoint_path) / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"No config.json found at {config_file}")
    with open(config_file) as f:
        return json.load(f)


def build_env_config(
    ckpt_config: Dict[str, Any],
) -> config_dict.ConfigDict:
    """Build a RunGapVision environment config from checkpoint config.

    Reconstructs the environment configuration that was used during
    training, including binocular settings, platform count, gap ranges,
    aesthetic, and mesh platforms.

    Args:
        ckpt_config: Parsed checkpoint config dictionary.

    Returns:
        ``config_dict.ConfigDict`` suitable for constructing the environment.
    """
    from vnl_playground.tasks.rodent import run_gap_vision

    cfg = run_gap_vision.default_config()

    # Apply environment args from checkpoint
    env_args = ckpt_config["env_config"].get("env_args", {})
    for key, value in env_args.items():
        if hasattr(cfg, key) or key in cfg:
            cfg[key] = value

    # Vision settings
    env_cfg = ckpt_config["env_config"]
    cfg.vision = env_cfg.get("vision", True)
    cfg.vision_width = env_cfg.get("vision_width", 32)
    cfg.vision_height = env_cfg.get("vision_height", 32)
    cfg.grayscale = env_cfg.get("grayscale", True)
    cfg.binocular = env_cfg.get("binocular", True)
    cfg.left_camera_name = env_cfg.get("left_camera_name", "eye_left-rodent")
    cfg.right_camera_name = env_cfg.get(
        "right_camera_name", "eye_right-rodent"
    )
    cfg.render_depth = env_cfg.get("render_depth", False)
    cfg.use_textures = env_cfg.get("use_textures", True)
    cfg.use_shadows = env_cfg.get("use_shadows", True)

    return cfg


def setup_env_and_policy(
    checkpoint_path: str,
    prior_checkpoint_path: str,
    ckpt_config: Dict[str, Any],
    seed: int = 0,
) -> Tuple[Any, Callable, Any, Any, Any, str, Optional[Callable]]:
    """Set up the environment, wrappers, and policy for rollout.

    Loads the trained binocular vision policy checkpoint and sets up the
    full inference pipeline: base env -> PriorHighLevelWrapper -> policy.
    Supports both MLP (feedforward) and RNN (recurrent) architectures,
    detected automatically from the checkpoint's ``network_config.arch_name``.

    Vision rendering is handled externally (not wrapped here) so the
    caller can control batch size and rendering lifecycle.

    Args:
        checkpoint_path: Path to the trained policy checkpoint directory.
        prior_checkpoint_path: Path to the SCAMPER prior checkpoint.
        ckpt_config: Parsed checkpoint configuration.
        seed: Random seed for environment creation.

    Returns:
        Tuple of ``(wrapped_env, policy_fn, params_tuple, mj_model, base_env, arch, init_hidden_fn)``.

        - ``arch``: ``"mlp"`` or ``"rnn"``.
        - ``init_hidden_fn``: For RNN, ``fn(batch_size) -> hidden_state``. ``None`` for MLP.
        - ``policy_fn``:
            MLP: ``(params, obs, rng) -> (action, extras)``
            RNN: ``(params, obs, hidden, rng) -> (action, extras, new_hidden)``
    """
    from omegaconf import OmegaConf
    from orbax import checkpoint as ocp

    from track_mjx.agent.ff_ppo import ppo_networks as ff_ppo_networks
    from track_mjx.agent.observation_utils import init_dict_normalizer

    from vnl_playground import tasks
    from vnl_playground.tasks.prior_utils import (
        load_prior_checkpoint,
        make_decoder_inference_fn,
        make_prior_inference_fn,
    )
    from vnl_playground.tasks.wrappers import PriorHighLevelWrapper

    # Step 1: Load SCAMPER prior/decoder
    print("  Loading SCAMPER prior/decoder...")
    _enc_params, prior_params, decoder_params, normalizer_params, prior_config = (
        load_prior_checkpoint(prior_checkpoint_path)
    )
    decoder_fn = make_decoder_inference_fn(decoder_params, normalizer_params, prior_config)
    prior_fn = make_prior_inference_fn(prior_params, normalizer_params, prior_config)
    latent_size = prior_config["network_config"]["intention_size"]
    print(f"  Latent size: {latent_size}")

    # Step 2: Create base environment (same pattern as train_highlvl.py)
    print("  Creating base environment...")
    env_name = ckpt_config["env_config"]["env_name"]
    env_args = dict(ckpt_config["env_config"].get("env_args", {}))
    # Pass vision config so env's reported vision_shape matches rendering
    for vk in ("vision_width", "vision_height", "grayscale", "binocular"):
        if vk in ckpt_config["env_config"]:
            env_args[vk] = ckpt_config["env_config"][vk]
    base_env = tasks.load(
        env_name, flatten_obs=False, config_overrides=env_args or None
    )

    # Step 3: Wrap with PriorHighLevelWrapper
    transfer_cfg = ckpt_config["transfer"]
    highlvl_obs_key = transfer_cfg.get("highlvl_obs_key", "task_obs")
    decoder_obs_key = transfer_cfg.get("decoder_obs_key", "proprioception")

    wrapped_env = PriorHighLevelWrapper(
        base_env,
        prior_fn,
        decoder_fn,
        latent_size,
        highlvl_obs_key=highlvl_obs_key,
        decoder_obs_key=decoder_obs_key,
        pass_vision=True,
        pass_task_obs=True,
        deterministic_prior=True,
    )

    # Step 4: Get obs sizes for network construction
    from track_mjx.agent.observation_utils import get_obs_sizes
    _tmp_state = wrapped_env.reset(jax.random.PRNGKey(seed))
    obs_sizes = get_obs_sizes(_tmp_state.obs)
    print(f"  Obs sizes: {obs_sizes}")

    # Step 5: Detect architecture and build network
    net_cfg = ckpt_config["network_config"]
    arch_name = net_cfg.get("arch_name", "binocular_shared_vision_task_obs")
    grayscale = ckpt_config["env_config"].get("grayscale", True)
    mono_channels = 1 if grayscale else 3
    binocular_mode = net_cfg.get("binocular_mode", "shared")
    vision_shape = (
        ckpt_config["env_config"].get("vision_height", 32),
        ckpt_config["env_config"].get("vision_width", 32),
        2 * mono_channels,
    )

    if "recurrent" in arch_name:
        arch = "rnn"
        print(f"  Architecture: RNN ({arch_name})")
        from track_mjx.agent.recurrent_ppo import networks as recurrent_ppo_net
        from track_mjx.agent.recurrent_ppo.recurrent_binocular_vision_networks import (
            make_recurrent_binocular_vision_highlvl_ppo_networks,
        )

        ppo_network, _shared_module = (
            make_recurrent_binocular_vision_highlvl_ppo_networks(
                obs_sizes=obs_sizes,
                action_size=latent_size,
                vision_shape=vision_shape,
                cnn_feature_size=net_cfg.get("vision_feature_size", 32),
                cnn_channels=tuple(net_cfg["vision_channels"]),
                gru_hidden_size=net_cfg.get("gru_hidden_size", 256),
                mono_channels=mono_channels,
                shared_weights=(binocular_mode == "shared"),
                policy_hidden_sizes=tuple(net_cfg.get("policy_head_sizes", [256])),
                value_hidden_sizes=tuple(net_cfg.get("value_head_sizes", [256, 128])),
            )
        )
        init_hidden_fn = ppo_network.policy_network.init_hidden
    else:
        arch = "mlp"
        print(f"  Architecture: MLP ({arch_name})")
        ppo_network, _shared_module = (
            ff_ppo_networks.make_binocular_shared_vision_task_obs_highlvl_ppo_networks(
                obs_sizes=obs_sizes,
                action_size=latent_size,
                vision_shape=vision_shape,
                mono_channels=mono_channels,
                shared_weights=(binocular_mode == "shared"),
                vision_latent_size=net_cfg.get("vision_latent_size", 16),
                vision_feature_size=net_cfg.get("vision_feature_size", 32),
                decoder_hidden_layer_sizes=tuple(net_cfg["decoder_hidden_layer_sizes"]),
                value_hidden_layer_sizes=tuple(net_cfg["value_hidden_layer_sizes"]),
                vision_channels=tuple(net_cfg["vision_channels"]),
                fusion_hidden_layer_sizes=tuple(
                    net_cfg.get("fusion_hidden_layer_sizes", [256])
                ),
            )
        )
        init_hidden_fn = None

    # Step 6: Load checkpoint params
    print("  Loading policy checkpoint...")
    ckpt_mgr = ocp.CheckpointManager(
        checkpoint_path,
        options=ocp.CheckpointManagerOptions(
            create=False, step_prefix="PPONetwork",
        ),
    )
    latest_step = ckpt_mgr.latest_step()

    # Build abstract policy for restore
    key_policy = jax.random.PRNGKey(seed)
    init_policy_params = ppo_network.policy_network.init(key_policy)
    if arch == "rnn":
        dummy_obs = {k: jp.zeros((1, v)) for k, v in obs_sizes.items()}
    else:
        dummy_obs = {
            "imitation_target": jp.zeros((1, obs_sizes.get("imitation_target", 0))),
            "proprioception": jp.zeros((1, obs_sizes.get("proprioception", 0))),
        }
    abstract_normalizer = init_dict_normalizer(dummy_obs)

    # Handle zero-sized arrays for orbax compatibility
    def _replace_zero_sized(pytree):
        def _maybe(x):
            if hasattr(x, "shape") and any(d == 0 for d in x.shape):
                return jp.array(float("nan"))
            return x
        return jax.tree_util.tree_map(_maybe, pytree)

    abstract_policy = _replace_zero_sized((abstract_normalizer, init_policy_params))

    normalizer_params_loaded, policy_params = ckpt_mgr.restore(
        latest_step,
        args=ocp.args.Composite(policy=ocp.args.StandardRestore(abstract_policy)),
    )["policy"]
    params_tuple = (normalizer_params_loaded, policy_params)
    print(f"  Loaded checkpoint step {latest_step}")

    # Step 7: Build inference function
    if arch == "rnn":
        make_logging_policy = recurrent_ppo_net.make_logging_inference_fn(ppo_network)
    else:
        make_logging_policy = ff_ppo_networks.make_logging_inference_fn(ppo_network)
    policy_fn = jax.jit(make_logging_policy(deterministic=True))

    mj_model = base_env.mj_model

    return wrapped_env, policy_fn, params_tuple, mj_model, base_env, arch, init_hidden_fn


# ---------------------------------------------------------------------------
# Episode rollout and data collection
# ---------------------------------------------------------------------------


def _extract_gap_geometry_batch(base_env, state_data):
    """Extract gap geometry for a batch of environments.

    Reads platform body xpos from batched simulation data to determine
    gap boundaries for each environment's (randomized) layout.

    Args:
        base_env: Unwrapped RunGapVision environment instance.
        state_data: mjx.Data with leading batch dimension ``(n_envs, ...)``.

    Returns:
        Tuple of:
            - gap_leading_edges: ``(n_envs, n_platforms)`` x-positions
              where each gap begins.
            - gap_lengths: ``(n_envs, n_platforms)`` width of each gap.
    """
    platform_body_ids = np.asarray(base_env._platform_body_ids)
    platform_half_length = float(base_env._platform_half_length)
    start_body_id = int(base_env._start_platform_body_id)
    start_half_length = float(base_env._start_platform_half_length)

    xpos = np.asarray(state_data.xpos)  # (n_envs, nbody, 3)
    start_trailing = xpos[:, start_body_id, 0] + start_half_length  # (n_envs,)

    plat_centers = xpos[:, platform_body_ids, 0]  # (n_envs, n_platforms)
    plat_leading = plat_centers - platform_half_length
    plat_trailing = plat_centers + platform_half_length

    all_trailing = np.concatenate(
        [start_trailing[:, None], plat_trailing], axis=1
    )  # (n_envs, n_platforms + 1)
    gap_leading_edges = all_trailing[:, :-1]  # (n_envs, n_platforms)
    gap_lengths = plat_leading - gap_leading_edges  # (n_envs, n_platforms)

    return gap_leading_edges, gap_lengths


def collect_episodes_batch(
    wrapped_env: Any,
    policy_fn: Callable,
    params_tuple: Any,
    base_env: Any,
    mj_model: mujoco.MjModel,
    body_cam_ids: Dict[str, int],
    n_episodes: int,
    max_steps: int,
    n_envs: int = 64,
    condition: str = "binocular",
    seed: int = 42,
    arch: str = "mlp",
    init_hidden_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Collect kinematics data using vmapped batch rollouts.

    Runs ``n_envs`` parallel environments simultaneously via ``jax.vmap``,
    with GPU-accelerated binocular vision rendering (``nworld=n_envs``).
    Episodes are collected in batches of ``n_envs`` for maximum throughput.

    Args:
        wrapped_env: PriorHighLevelWrapper (unwrapped, no vision).
        policy_fn: JIT-compiled policy: (params, obs, rng) -> (action, extras).
        params_tuple: Loaded policy parameters (normalizer, policy_params).
        base_env: Unwrapped base RunGapVision env (for gap geometry).
        mj_model: Compiled MuJoCo model.
        body_cam_ids: Resolved body/camera IDs.
        n_episodes: Total number of episodes to collect.
        max_steps: Maximum steps per episode.
        n_envs: Number of parallel environments per batch.
        condition: Visual condition.
        seed: Base random seed.

    Returns:
        Dictionary with per-episode arrays suitable for ``np.savez``.
    """
    from vnl_playground.tasks.rodent.vision_jax import (
        JaxVisionRenderer,
        VisionRenderWrapper,
    )

    # Vision renderer config from env
    vision_width = base_env._config.vision_width
    vision_height = base_env._config.vision_height
    grayscale = base_env._config.grayscale
    left_camera = base_env._config.left_camera_name
    right_camera = base_env._config.right_camera_name
    render_depth = base_env._config.render_depth
    use_textures = base_env._config.use_textures
    use_shadows = base_env._config.use_shadows

    # Create nworld=n_envs renderers for batched GPU rendering
    renderer_kwargs = dict(
        mj_model=base_env.mj_model,
        mjx_model=base_env.mjx_model,
        nworld=n_envs,
        width=vision_width,
        height=vision_height,
        grayscale=grayscale,
        render_depth=render_depth,
        use_textures=use_textures,
        use_shadows=use_shadows,
    )
    left_renderer = JaxVisionRenderer(
        camera_name=left_camera, **renderer_kwargs
    )
    right_renderer = JaxVisionRenderer(
        camera_name=right_camera, **renderer_kwargs
    )

    # Vmapped reset/step: each env gets independent rng → different gap layout
    vmapped_reset = jax.vmap(wrapped_env.reset)
    vmapped_step = jax.vmap(wrapped_env.step)

    def _render_and_inject(state):
        """Render binocular vision and inject into obs."""
        # state.data already has batch dim (n_envs, ...) from vmap
        left = left_renderer.render(state.data)   # (n_envs, H, W, C)
        right = right_renderer.render(state.data)  # (n_envs, H, W, C)
        vision = jp.concatenate([left, right], axis=-1)  # (n_envs, H, W, 2C)
        return state.replace(
            obs=VisionRenderWrapper._inject_vision(state.obs, vision)
        )

    def batched_reset(rngs):
        state = vmapped_reset(rngs)
        return _render_and_inject(state)

    def batched_step(state, actions):
        state = vmapped_step(state, actions)
        return _render_and_inject(state)

    jit_reset = jax.jit(batched_reset)
    jit_step = jax.jit(batched_step)

    # Body/camera indices for kinematics extraction
    skull_id = body_cam_ids["skull_body_id"]
    torso_id = body_cam_ids["torso_body_id"]
    left_cam_id = body_cam_ids["eye_left_cam_id"]
    right_cam_id = body_cam_ids["eye_right_cam_id"]

    # Storage across all batches
    all_skull_xpos = []
    all_skull_xmat = []
    all_torso_xpos = []
    all_torso_xmat = []
    all_torso_linvel = []
    all_cam_left = []
    all_cam_right = []
    all_timesteps = []
    all_episode_ids = []
    all_episode_lengths = []
    all_gap_leading_edges = []
    all_gap_lengths = []
    all_gap_episode_ids = []

    rng = jax.random.PRNGKey(seed)
    n_batches = (n_episodes + n_envs - 1) // n_envs
    episode_count = 0

    for batch_idx in range(n_batches):
        batch_size = min(n_envs, n_episodes - episode_count)
        rng, batch_rng, act_rng = jax.random.split(rng, 3)
        # Always reset n_envs environments (pad if last batch is smaller)
        reset_rngs = jax.random.split(batch_rng, n_envs)

        print(
            f"  Batch {batch_idx + 1}/{n_batches}: "
            f"resetting {n_envs} parallel envs...",
            flush=True,
        )
        state = jit_reset(reset_rngs)

        # Extract gap geometry from initial batched state
        gap_edges, gap_lens = _extract_gap_geometry_batch(
            base_env, state.data
        )  # each (n_envs, n_platforms)

        # Pre-allocate per-timestep arrays: (max_steps+1, n_envs, ...)
        # Note: MJX warp stores xmat as (nbody, 3, 3) not flat (nbody, 9).
        T = max_steps + 1
        batch_skull_xpos = np.zeros((T, n_envs, 3), dtype=np.float32)
        batch_skull_xmat = np.zeros((T, n_envs, 3, 3), dtype=np.float32)
        batch_torso_xpos = np.zeros((T, n_envs, 3), dtype=np.float32)
        batch_torso_xmat = np.zeros((T, n_envs, 3, 3), dtype=np.float32)
        batch_torso_linvel = np.zeros((T, n_envs, 3), dtype=np.float32)
        batch_cam_left = np.zeros((T, n_envs, 3), dtype=np.float32)
        batch_cam_right = np.zeros((T, n_envs, 3), dtype=np.float32)
        batch_done = np.zeros((T, n_envs), dtype=bool)

        def _record_batch(data, t_idx):
            xpos = np.asarray(data.xpos)  # (n_envs, nbody, 3)
            xmat = np.asarray(data.xmat)  # (n_envs, nbody, 3, 3)
            batch_skull_xpos[t_idx] = xpos[:, skull_id]
            batch_skull_xmat[t_idx] = xmat[:, skull_id]
            batch_torso_xpos[t_idx] = xpos[:, torso_id]
            batch_torso_xmat[t_idx] = xmat[:, torso_id]
            batch_torso_linvel[t_idx] = np.asarray(
                data._impl.subtree_linvel
            )[:, torso_id]
            batch_cam_left[t_idx] = np.asarray(
                data.cam_xpos
            )[:, left_cam_id]
            batch_cam_right[t_idx] = np.asarray(
                data.cam_xpos
            )[:, right_cam_id]

        # Record initial state (t=0)
        _record_batch(state.data, 0)

        # Track which envs have finished
        env_done = np.zeros(n_envs, dtype=bool)
        actual_steps = max_steps

        # Initialize RNN hidden state for this batch
        if arch == "rnn" and init_hidden_fn is not None:
            hidden = init_hidden_fn(n_envs)

        for t in range(max_steps):
            obs = state.obs

            # Apply monocular masking (batched, all envs same condition)
            if condition != "binocular" and "vision" in obs:
                obs = {
                    **obs,
                    "vision": apply_monocular_mask(
                        obs["vision"], condition
                    ),
                }

            _, act_rng = jax.random.split(act_rng)
            if arch == "rnn":
                actions, _, hidden = policy_fn(params_tuple, obs, hidden, act_rng)
            else:
                actions, _ = policy_fn(params_tuple, obs, act_rng)
            state = jit_step(state, actions)

            _record_batch(state.data, t + 1)
            step_done = np.asarray(state.done) > 0.5  # (n_envs,)
            batch_done[t + 1] = step_done
            env_done |= step_done

            # Early exit when all environments are done
            if np.all(env_done):
                actual_steps = t + 1
                break

        # Segment per-env episodes and store
        for env_idx in range(batch_size):
            # Find episode length: first timestep where done is True
            done_flags = batch_done[1 : actual_steps + 1, env_idx]
            done_indices = np.where(done_flags)[0]
            if len(done_indices) > 0:
                ep_steps = int(done_indices[0]) + 1
            else:
                ep_steps = actual_steps
            ep_len = ep_steps + 1  # +1 for initial state

            ep_id = episode_count + env_idx

            all_skull_xpos.append(
                batch_skull_xpos[:ep_len, env_idx].copy()
            )
            all_skull_xmat.append(
                batch_skull_xmat[:ep_len, env_idx].copy()
            )
            all_torso_xpos.append(
                batch_torso_xpos[:ep_len, env_idx].copy()
            )
            all_torso_xmat.append(
                batch_torso_xmat[:ep_len, env_idx].copy()
            )
            all_torso_linvel.append(
                batch_torso_linvel[:ep_len, env_idx].copy()
            )
            all_cam_left.append(
                batch_cam_left[:ep_len, env_idx].copy()
            )
            all_cam_right.append(
                batch_cam_right[:ep_len, env_idx].copy()
            )
            all_timesteps.append(np.arange(ep_len, dtype=np.int32))
            all_episode_ids.append(
                np.full(ep_len, ep_id, dtype=np.int32)
            )
            all_episode_lengths.append(ep_len)

            n_gaps = gap_edges.shape[1]
            all_gap_leading_edges.append(gap_edges[env_idx])
            all_gap_lengths.append(gap_lens[env_idx])
            all_gap_episode_ids.append(
                np.full(n_gaps, ep_id, dtype=np.int32)
            )

        episode_count += batch_size

        # Progress report
        mean_len = np.mean(all_episode_lengths[-batch_size:])
        mean_x = np.mean(
            [s[-1, 0] for s in all_torso_xpos[-batch_size:]]
        )
        print(
            f"  Batch {batch_idx + 1}/{n_batches} done: "
            f"{batch_size} episodes, mean_steps={mean_len:.0f}, "
            f"mean_x={mean_x:.2f}",
            flush=True,
        )

    # Concatenate all episodes
    result = {
        "skull_xpos": np.concatenate(all_skull_xpos, axis=0),
        "skull_xmat": np.concatenate(all_skull_xmat, axis=0),
        "torso_xpos": np.concatenate(all_torso_xpos, axis=0),
        "torso_xmat": np.concatenate(all_torso_xmat, axis=0),
        "torso_linvel": np.concatenate(all_torso_linvel, axis=0),
        "cam_eye_left_xpos": np.concatenate(all_cam_left, axis=0),
        "cam_eye_right_xpos": np.concatenate(all_cam_right, axis=0),
        "timesteps": np.concatenate(all_timesteps, axis=0),
        "episode_ids": np.concatenate(all_episode_ids, axis=0),
        "episode_lengths": np.array(all_episode_lengths, dtype=np.int32),
        "condition": condition,
        "n_episodes": n_episodes,
        "max_steps": max_steps,
        "seed": seed,
        "gap_leading_edges_flat": np.concatenate(
            all_gap_leading_edges, axis=0
        ),
        "gap_lengths_flat": np.concatenate(all_gap_lengths, axis=0),
        "gap_episode_ids": np.concatenate(all_gap_episode_ids, axis=0),
    }

    return result


# ---------------------------------------------------------------------------
# Data saving
# ---------------------------------------------------------------------------


def save_kinematics_data(
    data: Dict[str, Any],
    output_dir: str,
    condition: str,
) -> str:
    """Save collected kinematics data to a compressed .npz file.

    Output filename: ``{output_dir}/run_gap_kinematics_{condition}.npz``

    The .npz file contains all arrays from ``collect_episodes`` plus
    scalar metadata stored as 0-d arrays.

    Args:
        data: Dictionary of arrays from :func:`collect_episodes`.
        output_dir: Directory to save the output file.
        condition: Visual condition name (used in filename).

    Returns:
        Absolute path to the saved .npz file.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"run_gap_kinematics_{condition}.npz"
    filepath = os.path.join(output_dir, filename)

    # Separate string/scalar metadata from arrays for np.savez
    save_dict = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            save_dict[key] = value
        elif isinstance(value, (int, float)):
            save_dict[key] = np.array(value)
        elif isinstance(value, str):
            save_dict[key] = np.array(value)
        else:
            # Skip non-serializable values
            print(f"  Warning: skipping non-serializable key '{key}'")

    np.savez_compressed(filepath, **save_dict)

    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  Saved: {filepath} ({file_size_mb:.1f} MB)")

    return filepath


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Collect RunGap rollout kinematics data for motion parallax "
            "analysis. Loads a trained binocular vision checkpoint and "
            "extracts per-timestep skull, torso, and camera positions."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the trained policy checkpoint directory.",
    )
    parser.add_argument(
        "--prior_checkpoint_path",
        type=str,
        default="/home/scott/SalkResearch/data/prior",
        help="Path to the SCAMPER prior checkpoint directory.",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="binocular",
        choices=VALID_CONDITIONS,
        help="Visual condition for rollout.",
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=200,
        help="Number of episodes to collect.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=2000,
        help="Maximum steps per episode (should match episode_length).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/motion_parallax",
        help="Directory for output .npz files.",
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        default=64,
        help="Number of parallel environments per batch (vmap batch size).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main():
    """Main entry point for RunGap kinematics data collection.

    1. Parse CLI arguments.
    2. Load checkpoint configuration.
    3. Build environment config from checkpoint.
    4. Set up environment, wrappers, and policy.
    5. Resolve body/camera IDs from the compiled model.
    6. Run episodes and collect kinematics.
    7. Save output to .npz file.
    """
    args = parse_args()

    print("=" * 60)
    print("RunGap Kinematics Data Collector")
    print("=" * 60)
    print(f"  Checkpoint:  {args.checkpoint_path}")
    print(f"  Prior:       {args.prior_checkpoint_path}")
    print(f"  Condition:   {args.condition}")
    print(f"  Episodes:    {args.n_episodes}")
    print(f"  Max steps:   {args.max_steps}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"  Batch size:  {args.n_envs}")
    print(f"  Seed:        {args.seed}")
    print()

    # --- 1. Load checkpoint config ---
    print("[1/5] Loading checkpoint configuration...")
    ckpt_config = load_config(args.checkpoint_path)
    env_name = ckpt_config["env_config"]["env_name"]
    arch_name = ckpt_config["network_config"]["arch_name"]
    print(f"  Environment: {env_name}")
    print(f"  Architecture: {arch_name}")

    # --- 2. Print environment config ---
    env_args = ckpt_config["env_config"].get("env_args", {})
    env_cfg = ckpt_config["env_config"]
    print(f"[2/5] Environment config from checkpoint:")
    print(f"  Vision: {env_cfg.get('vision_width', 32)}x{env_cfg.get('vision_height', 32)}")
    print(f"  Binocular: {env_cfg.get('binocular', False)}")
    print(f"  Platforms: {env_args.get('n_platforms', '?')}")
    print(f"  Gap range: {env_args.get('gap_length_range', '?')}")
    print(f"  ctrl_dt: {env_args.get('ctrl_dt', '?')}, sim_dt: {env_args.get('sim_dt', '?')}")

    # --- 3. Set up environment and policy ---
    print("[3/5] Setting up environment and policy...")
    wrapped_env, policy_fn, params_tuple, mj_model, base_env, arch, init_hidden_fn = (
        setup_env_and_policy(
            checkpoint_path=args.checkpoint_path,
            prior_checkpoint_path=args.prior_checkpoint_path,
            ckpt_config=ckpt_config,
            seed=args.seed,
        )
    )

    # --- 4. Resolve body/camera IDs ---
    print("[4/5] Resolving body and camera IDs...")
    body_cam_ids = resolve_body_and_camera_ids(mj_model)
    for name, obj_id in body_cam_ids.items():
        print(f"  {name}: {obj_id}")

    # --- 5. Collect episodes ---
    print(f"[5/5] Collecting {args.n_episodes} episodes "
          f"(condition={args.condition})...")
    collected_data = collect_episodes_batch(
        wrapped_env=wrapped_env,
        policy_fn=policy_fn,
        params_tuple=params_tuple,
        base_env=base_env,
        mj_model=mj_model,
        body_cam_ids=body_cam_ids,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        n_envs=args.n_envs,
        condition=args.condition,
        seed=args.seed,
        arch=arch,
        init_hidden_fn=init_hidden_fn,
    )

    # --- Summary ---
    total_timesteps = collected_data["skull_xpos"].shape[0]
    mean_length = np.mean(collected_data["episode_lengths"])
    n_gaps_total = len(collected_data["gap_leading_edges_flat"])
    print()
    print("Collection Summary:")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  Mean episode length: {mean_length:.1f}")
    print(f"  Total gap windows: {n_gaps_total}")

    # --- Save ---
    output_path = save_kinematics_data(
        collected_data, args.output_dir, args.condition
    )

    print()
    print(f"Done. Output saved to: {output_path}")


if __name__ == "__main__":
    main()
