"""Comprehensive rollout data collector with HDF5 output.

Loads a trained binocular vision run-gap agent checkpoint, sets up the
environment with SCAMPER prior/decoder, runs batched GPU-accelerated
rollouts, extracts per-timestep data (observations, rewards, network
activations including CNN spatial maps, kinematics), and writes everything
to HDF5.

Two modes:
  - variable_gap: Random gap layouts (same as training). Filter episodes
    by gap-crossing count (>= min_gaps_crossed).
  - fixed_gap: Sweep uniform gap lengths from gap_min to gap_max in
    gap_step increments. Each env has N identical gaps (n_gaps_per_env).
    Quality filtering on gap crossings.

Usage::

    python -m vnl_playground.tasks.rodent.analysis.collect_rollout_hdf5 \\
        --checkpoint_path /path/to/checkpoint \\
        --output_path ./rollout_data.h5 \\
        --mode variable_gap \\
        --n_episodes 50

    python -m vnl_playground.tasks.rodent.analysis.collect_rollout_hdf5 \\
        --checkpoint_path /path/to/checkpoint \\
        --output_path ./fixed_gap_sweep.h5 \\
        --mode fixed_gap \\
        --gap_min 0.03 --gap_max 0.20 --gap_step 0.01
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Environment flags must be set before importing JAX/MuJoCo.
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.4"
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

import vnl_playground.naccdmax_patch  # noqa: F401 — fix CCD overflow

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

from vnl_playground.tasks.rodent.analysis.rollout_hdf5_config import (
    FixedGapConfig,
    RolloutCollectionConfig,
)


# ---------------------------------------------------------------------------
# Body / Camera ID resolution (reused from collect_run_gap_data.py)
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
# Configuration loading
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


# ---------------------------------------------------------------------------
# Gap geometry extraction (batched)
# ---------------------------------------------------------------------------


def _extract_gap_geometry_batch(
    base_env: Any,
    state_data: mjx.Data,
) -> Tuple[np.ndarray, np.ndarray]:
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


def _count_gaps_crossed(
    torso_x: float,
    gap_leading_edges: np.ndarray,
    gap_lengths: np.ndarray,
) -> int:
    """Count how many gaps the torso has fully crossed.

    A gap is crossed when torso_x exceeds the gap trailing edge
    (gap_leading_edge + gap_length).

    Args:
        torso_x: Current torso x-position.
        gap_leading_edges: (n_platforms,) gap start positions.
        gap_lengths: (n_platforms,) gap widths.

    Returns:
        Number of gaps fully crossed.
    """
    gap_end_positions = gap_leading_edges + gap_lengths
    return int(np.sum(torso_x > gap_end_positions))


# ---------------------------------------------------------------------------
# Environment and policy setup
# ---------------------------------------------------------------------------


def setup_env_and_policy(
    checkpoint_path: str,
    prior_checkpoint_path: str,
    ckpt_config: Dict[str, Any],
    seed: int = 0,
    env_overrides: Optional[Dict[str, Any]] = None,
    get_activation: bool = True,
) -> Tuple[Any, Callable, Any, Any, Any, str, Optional[Callable]]:
    """Set up the environment, wrappers, and policy for rollout.

    Loads the trained binocular vision policy checkpoint and sets up the
    full inference pipeline: base env -> PriorHighLevelWrapper -> policy.
    Supports both MLP (feedforward) and RNN (recurrent) architectures.

    Args:
        checkpoint_path: Path to the trained policy checkpoint directory.
        prior_checkpoint_path: Path to the SCAMPER prior checkpoint.
        ckpt_config: Parsed checkpoint configuration.
        seed: Random seed for environment creation.
        env_overrides: Optional dict of env_args to override (e.g. for
            fixed-gap mode: gap_length_range, n_platforms).
        get_activation: Whether to capture network activations.

    Returns:
        Tuple of ``(wrapped_env, policy_fn, params_tuple, mj_model,
        base_env, arch, init_hidden_fn)``.
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
    decoder_fn = make_decoder_inference_fn(
        decoder_params, normalizer_params, prior_config
    )
    prior_fn = make_prior_inference_fn(
        prior_params, normalizer_params, prior_config
    )
    latent_size = prior_config["network_config"]["intention_size"]
    print(f"  Latent size: {latent_size}")

    # Step 2: Create base environment
    print("  Creating base environment...")
    env_name = ckpt_config["env_config"]["env_name"]
    env_args = dict(ckpt_config["env_config"].get("env_args", {}))
    # Pass vision config so env's reported vision_shape matches rendering
    for vk in ("vision_width", "vision_height", "grayscale", "binocular"):
        if vk in ckpt_config["env_config"]:
            env_args[vk] = ckpt_config["env_config"][vk]
    # Apply env overrides (e.g. fixed-gap settings)
    if env_overrides:
        env_args.update(env_overrides)

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
                policy_hidden_sizes=tuple(
                    net_cfg.get("policy_head_sizes", [256])
                ),
                value_hidden_sizes=tuple(
                    net_cfg.get("value_head_sizes", [256, 128])
                ),
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
                decoder_hidden_layer_sizes=tuple(
                    net_cfg["decoder_hidden_layer_sizes"]
                ),
                value_hidden_layer_sizes=tuple(
                    net_cfg["value_hidden_layer_sizes"]
                ),
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
            create=False,
            step_prefix="PPONetwork",
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
            "imitation_target": jp.zeros(
                (1, obs_sizes.get("imitation_target", 0))
            ),
            "proprioception": jp.zeros(
                (1, obs_sizes.get("proprioception", 0))
            ),
        }
    abstract_normalizer = init_dict_normalizer(dummy_obs)

    # Handle zero-sized arrays for orbax compatibility
    def _replace_zero_sized(pytree):
        def _maybe(x):
            if hasattr(x, "shape") and any(d == 0 for d in x.shape):
                return jp.array(float("nan"))
            return x

        return jax.tree_util.tree_map(_maybe, pytree)

    abstract_policy = _replace_zero_sized(
        (abstract_normalizer, init_policy_params)
    )

    normalizer_params_loaded, policy_params = ckpt_mgr.restore(
        latest_step,
        args=ocp.args.Composite(
            policy=ocp.args.StandardRestore(abstract_policy)
        ),
    )["policy"]
    params_tuple = (normalizer_params_loaded, policy_params)
    print(f"  Loaded checkpoint step {latest_step}")

    # Step 7: Build inference function with activation capture
    if arch == "rnn":
        from track_mjx.agent.recurrent_ppo import networks as recurrent_ppo_net

        make_logging_policy = recurrent_ppo_net.make_logging_inference_fn(
            ppo_network
        )
    else:
        make_logging_policy = ff_ppo_networks.make_logging_inference_fn(
            ppo_network
        )
    policy_fn = jax.jit(
        make_logging_policy(
            deterministic=True, get_activation=get_activation
        )
    )

    mj_model = base_env.mj_model

    return (
        wrapped_env,
        policy_fn,
        params_tuple,
        mj_model,
        base_env,
        arch,
        init_hidden_fn,
    )


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------


def _flatten_activations(
    activations: Dict[str, Any],
    capture_cnn_maps: bool,
) -> Dict[str, Any]:
    """Flatten nested activation dict and optionally strip CNN maps.

    Args:
        activations: Raw activation dict from policy extras.
        capture_cnn_maps: Whether to keep the 'cnn' key.

    Returns:
        Flattened activation dict suitable for numpy storage.
    """
    result = {}
    for key, val in activations.items():
        if key == "cnn" and not capture_cnn_maps:
            continue
        if isinstance(val, dict):
            result[key] = {
                k: np.asarray(v, dtype=np.float32) for k, v in val.items()
            }
        else:
            result[key] = np.asarray(val, dtype=np.float32)
    return result


def _preallocate_activation_storage(
    sample_activations: Dict[str, Any],
    max_steps: int,
    capture_cnn_maps: bool,
) -> Dict[str, Any]:
    """Pre-allocate numpy arrays for activation storage.

    On the first step, inspect the activation shapes and create
    zero-filled arrays for efficient per-timestep writes.

    Args:
        sample_activations: Activation dict from a single step (batched).
        max_steps: Maximum number of timesteps to store.
        capture_cnn_maps: Whether to include CNN spatial maps.

    Returns:
        Dict with same structure, each leaf is a pre-allocated numpy array
        with leading dimension ``max_steps``.
    """
    storage = {}
    for key, val in sample_activations.items():
        if key == "cnn" and not capture_cnn_maps:
            continue
        if isinstance(val, dict):
            storage[key] = {}
            for k, v in val.items():
                arr = np.asarray(v)
                storage[key][k] = np.zeros(
                    (max_steps, *arr.shape), dtype=np.float32
                )
        else:
            arr = np.asarray(val)
            storage[key] = np.zeros(
                (max_steps, *arr.shape), dtype=np.float32
            )
    return storage


def _record_activations(
    storage: Dict[str, Any],
    activations: Dict[str, Any],
    t_idx: int,
    capture_cnn_maps: bool,
) -> None:
    """Copy activations from a single step into pre-allocated storage.

    Args:
        storage: Pre-allocated activation arrays.
        activations: Raw activations from current step.
        t_idx: Timestep index to write into.
        capture_cnn_maps: Whether CNN maps are being stored.
    """
    for key, val in activations.items():
        if key == "cnn" and not capture_cnn_maps:
            continue
        if key not in storage:
            continue
        if isinstance(val, dict):
            for k, v in val.items():
                if k in storage[key]:
                    storage[key][k][t_idx] = np.asarray(v)
        else:
            storage[key][t_idx] = np.asarray(val)


def _slice_activation_storage(
    storage: Dict[str, Any],
    length: int,
    env_idx: int,
) -> Dict[str, Any]:
    """Extract a single episode's activation data from batch storage.

    Args:
        storage: Pre-allocated activation arrays with shape
            ``(max_steps, n_envs, ...)``.
        length: Number of timesteps for this episode.
        env_idx: Index of this environment in the batch.

    Returns:
        Dict with same structure, each leaf sliced to ``(length, ...)``.
    """
    result = {}
    for key, val in storage.items():
        if isinstance(val, dict):
            result[key] = {}
            for k, v in val.items():
                result[key][k] = v[:length, env_idx].copy()
        else:
            result[key] = val[:length, env_idx].copy()
    return result


# ---------------------------------------------------------------------------
# Vision rendering helpers
# ---------------------------------------------------------------------------


def _create_renderers(
    base_env: Any,
    n_envs: int,
    ckpt_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any, str, str]:
    """Create binocular vision renderers for batched rollouts.

    Args:
        base_env: Unwrapped base environment.
        n_envs: Number of parallel environments.

    Returns:
        Tuple of (left_renderer, right_renderer, left_camera, right_camera).
    """
    from vnl_playground.tasks.rodent.vision_jax import JaxVisionRenderer

    vision_width = base_env._config.vision_width
    vision_height = base_env._config.vision_height
    grayscale = base_env._config.grayscale
    left_camera = base_env._config.left_camera_name
    right_camera = base_env._config.right_camera_name
    render_depth = base_env._config.get("render_depth", False)
    # use_textures/use_shadows are in ckpt_config["env_config"] (top-level),
    # NOT in env_args. base_env._config may default to False.
    # Read from ckpt_config when available to match training rendering.
    if ckpt_config is not None:
        use_textures = ckpt_config["env_config"].get("use_textures", True)
        use_shadows = ckpt_config["env_config"].get("use_shadows", True)
    else:
        use_textures = base_env._config.get("use_textures", True)
        use_shadows = base_env._config.get("use_shadows", True)

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

    return left_renderer, right_renderer, left_camera, right_camera


# ---------------------------------------------------------------------------
# Batched rollout collection
# ---------------------------------------------------------------------------


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
    seed: int = 42,
    arch: str = "mlp",
    init_hidden_fn: Optional[Callable] = None,
    capture_vision: bool = True,
    capture_activations: bool = True,
    capture_cnn_maps: bool = False,
    min_gaps_crossed: int = 0,
    gap_length: Optional[float] = None,
    ckpt_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Collect rollout episodes using vmapped batch rollouts.

    Runs ``n_envs`` parallel environments simultaneously via ``jax.vmap``,
    with GPU-accelerated binocular vision rendering. Episodes are collected
    in batches for maximum throughput. Quality filtering by gap crossings.

    Args:
        wrapped_env: PriorHighLevelWrapper (unwrapped, no vision).
        policy_fn: JIT-compiled policy with activation capture.
        params_tuple: Loaded policy parameters (normalizer, policy_params).
        base_env: Unwrapped base RunGapVision env (for gap geometry).
        mj_model: Compiled MuJoCo model.
        body_cam_ids: Resolved body/camera IDs.
        n_episodes: Target number of episodes to collect (after filtering).
        max_steps: Maximum steps per episode.
        n_envs: Number of parallel environments per batch.
        seed: Base random seed.
        arch: Architecture type ("mlp" or "rnn").
        init_hidden_fn: Hidden state initializer for RNN.
        capture_vision: Whether to store vision observations.
        capture_activations: Whether to store network activations.
        capture_cnn_maps: Whether to store CNN spatial maps.
        min_gaps_crossed: Minimum gaps crossed to keep episode.
        gap_length: If set, this is a fixed-gap sweep (for metadata).

    Returns:
        List of episode dicts, each containing timestep data.
    """
    from vnl_playground.tasks.rodent.vision_jax import VisionRenderWrapper

    left_renderer, right_renderer, _, _ = _create_renderers(
        base_env, n_envs, ckpt_config=ckpt_config
    )

    # Vmapped reset/step
    vmapped_reset = jax.vmap(wrapped_env.reset)
    vmapped_step = jax.vmap(wrapped_env.step)

    def _render_and_inject(state):
        """Render binocular vision and inject into obs."""
        left = left_renderer.render(state.data)
        right = right_renderer.render(state.data)
        vision = jp.concatenate([left, right], axis=-1)
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

    # Rodent qpos/qvel start indices
    rodent_qpos_start = int(base_env._rodent_qpos_start)
    rodent_qvel_start = int(base_env._rodent_qvel_start)

    collected_episodes: List[Dict[str, Any]] = []
    rng = jax.random.PRNGKey(seed)
    total_batches_run = 0
    max_collection_batches = max(
        (n_episodes + n_envs - 1) // n_envs * 5, 20
    )  # safety limit

    while len(collected_episodes) < n_episodes:
        if total_batches_run >= max_collection_batches:
            print(
                f"  WARNING: Hit max batch limit ({max_collection_batches}). "
                f"Collected {len(collected_episodes)}/{n_episodes} episodes."
            )
            break

        rng, batch_rng, act_rng = jax.random.split(rng, 3)
        reset_rngs = jax.random.split(batch_rng, n_envs)

        print(
            f"  Batch {total_batches_run + 1}: "
            f"resetting {n_envs} parallel envs "
            f"({len(collected_episodes)}/{n_episodes} collected)...",
            flush=True,
        )
        state = jit_reset(reset_rngs)

        # Extract gap geometry from initial batched state
        gap_edges, gap_lens = _extract_gap_geometry_batch(
            base_env, state.data
        )

        T = max_steps + 1  # +1 for initial state at t=0

        # Pre-allocate per-timestep kinematics: (T, n_envs, ...)
        batch_qpos = np.zeros(
            (T, n_envs, mj_model.nq), dtype=np.float32
        )
        batch_qvel = np.zeros(
            (T, n_envs, mj_model.nv), dtype=np.float32
        )
        batch_actions = np.zeros(
            (T, n_envs, wrapped_env.action_size), dtype=np.float32
        )
        batch_rewards = np.zeros((T, n_envs), dtype=np.float32)
        batch_done = np.zeros((T, n_envs), dtype=bool)

        # World kinematics
        batch_torso_xpos = np.zeros((T, n_envs, 3), dtype=np.float32)
        batch_torso_xmat = np.zeros((T, n_envs, 3, 3), dtype=np.float32)
        batch_torso_linvel = np.zeros((T, n_envs, 3), dtype=np.float32)
        batch_skull_xpos = np.zeros((T, n_envs, 3), dtype=np.float32)
        batch_skull_xmat = np.zeros((T, n_envs, 3, 3), dtype=np.float32)
        batch_cam_left = np.zeros((T, n_envs, 3), dtype=np.float32)
        batch_cam_right = np.zeros((T, n_envs, 3), dtype=np.float32)

        # Hand positions and touch sensors (for psychometric analysis)
        hand_l_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "hand_L-rodent")
        hand_r_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "hand_R-rodent")
        batch_hand_l_xpos = np.zeros((T, n_envs, 3), dtype=np.float32)
        batch_hand_r_xpos = np.zeros((T, n_envs, 3), dtype=np.float32)
        # Touch sensors
        palm_l_sensor_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "palm_L-rodent")
        palm_r_sensor_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "palm_R-rodent")
        has_touch = palm_l_sensor_id >= 0 and palm_r_sensor_id >= 0
        if has_touch:
            palm_l_adr = mj_model.sensor_adr[palm_l_sensor_id]
            palm_r_adr = mj_model.sensor_adr[palm_r_sensor_id]
            batch_palm_l_touch = np.zeros((T, n_envs), dtype=np.float32)
            batch_palm_r_touch = np.zeros((T, n_envs), dtype=np.float32)

        # Vision
        if capture_vision:
            vision_shape = state.obs["vision"].shape  # (n_envs, H, W, C)
            batch_vision = np.zeros(
                (T, *vision_shape), dtype=np.uint8
            )

        # Task obs
        task_obs_key = "imitation_target" if "imitation_target" in state.obs else "task_obs"
        task_obs_shape = state.obs[task_obs_key].shape  # (n_envs, task_obs_dim)
        batch_task_obs = np.zeros((T, *task_obs_shape), dtype=np.float32)

        # Activation storage (initialized on first step)
        activation_storage = None

        def _record_state(state_data, obs, t_idx):
            """Record kinematics and observations at timestep t_idx."""
            xpos = np.asarray(state_data.xpos)
            xmat = np.asarray(state_data.xmat)

            batch_qpos[t_idx] = np.asarray(state_data.qpos)
            batch_qvel[t_idx] = np.asarray(state_data.qvel)

            batch_torso_xpos[t_idx] = xpos[:, torso_id]
            batch_torso_xmat[t_idx] = xmat[:, torso_id]
            batch_torso_linvel[t_idx] = np.asarray(
                state_data._impl.subtree_linvel
            )[:, torso_id]
            batch_skull_xpos[t_idx] = xpos[:, skull_id]
            batch_skull_xmat[t_idx] = xmat[:, skull_id]
            batch_cam_left[t_idx] = np.asarray(
                state_data.cam_xpos
            )[:, left_cam_id]
            batch_cam_right[t_idx] = np.asarray(
                state_data.cam_xpos
            )[:, right_cam_id]
            # Hand positions
            if hand_l_id >= 0:
                batch_hand_l_xpos[t_idx] = xpos[:, hand_l_id]
            if hand_r_id >= 0:
                batch_hand_r_xpos[t_idx] = xpos[:, hand_r_id]
            # Touch sensors
            if has_touch:
                sdata = np.asarray(state_data.sensordata)
                batch_palm_l_touch[t_idx] = sdata[:, palm_l_adr]
                batch_palm_r_touch[t_idx] = sdata[:, palm_r_adr]

            batch_task_obs[t_idx] = np.asarray(obs[task_obs_key])
            if capture_vision and "vision" in obs:
                # Store as uint8 (0-255)
                vision_data = np.asarray(obs["vision"])
                if vision_data.dtype != np.uint8:
                    vision_data = np.clip(
                        vision_data * 255, 0, 255
                    ).astype(np.uint8)
                batch_vision[t_idx] = vision_data

        # Record initial state (t=0)
        _record_state(state.data, state.obs, 0)

        # Track which envs have finished
        env_done = np.zeros(n_envs, dtype=bool)
        actual_steps = max_steps

        # Initialize RNN hidden state
        if arch == "rnn" and init_hidden_fn is not None:
            hidden = init_hidden_fn(n_envs)

        for t in range(max_steps):
            obs = state.obs
            _, act_rng = jax.random.split(act_rng)

            if arch == "rnn":
                actions, extras, hidden = policy_fn(
                    params_tuple, obs, hidden, act_rng
                )
            else:
                actions, extras = policy_fn(params_tuple, obs, act_rng)

            # Record activations
            if capture_activations and "activations" in extras:
                raw_acts = extras["activations"]
                if activation_storage is None:
                    # First step: pre-allocate
                    activation_storage = _preallocate_activation_storage(
                        raw_acts, max_steps, capture_cnn_maps
                    )
                _record_activations(
                    activation_storage, raw_acts, t, capture_cnn_maps
                )

            state = jit_step(state, actions)
            batch_actions[t + 1] = np.asarray(actions)
            batch_rewards[t + 1] = np.asarray(state.reward)

            _record_state(state.data, state.obs, t + 1)

            step_done = np.asarray(state.done) > 0.5
            batch_done[t + 1] = step_done
            env_done |= step_done

            if np.all(env_done):
                actual_steps = t + 1
                break

        # Segment per-env episodes and filter
        batch_kept = 0
        for env_idx in range(n_envs):
            if len(collected_episodes) >= n_episodes:
                break

            # Find episode length
            done_flags = batch_done[1 : actual_steps + 1, env_idx]
            done_indices = np.where(done_flags)[0]
            if len(done_indices) > 0:
                ep_steps = int(done_indices[0]) + 1
            else:
                ep_steps = actual_steps
            ep_len = ep_steps + 1  # +1 for initial state

            # Count gaps crossed
            final_torso_x = float(
                batch_torso_xpos[ep_len - 1, env_idx, 0]
            )
            n_crossed = _count_gaps_crossed(
                final_torso_x,
                gap_edges[env_idx],
                gap_lens[env_idx],
            )

            # Filter
            if n_crossed < min_gaps_crossed:
                continue

            # Build episode dict
            terminated = bool(len(done_indices) > 0)
            total_reward = float(
                np.sum(batch_rewards[1:ep_len, env_idx])
            )

            episode_data: Dict[str, Any] = {
                "n_steps": ep_len,
                "gaps_crossed": n_crossed,
                "total_reward": total_reward,
                "terminated": terminated,
            }

            if gap_length is not None:
                episode_data["gap_length"] = gap_length

            # Timestep data
            timesteps: Dict[str, Any] = {
                "qpos": batch_qpos[:ep_len, env_idx].copy(),
                "qvel": batch_qvel[:ep_len, env_idx].copy(),
                "action": batch_actions[:ep_len, env_idx].copy(),
                "reward": batch_rewards[:ep_len, env_idx].copy(),
                "done": batch_done[:ep_len, env_idx].copy(),
            }

            # Observations
            obs_data: Dict[str, Any] = {
                "task_obs": batch_task_obs[:ep_len, env_idx].copy(),
            }
            if capture_vision:
                obs_data["vision"] = batch_vision[:ep_len, env_idx].copy()
            timesteps["observations"] = obs_data

            # Activations
            if capture_activations and activation_storage is not None:
                # Activations are recorded for steps 0..ep_steps-1
                # (corresponding to policy calls before each step)
                act_len = min(ep_steps, max_steps)
                timesteps["activations"] = _slice_activation_storage(
                    activation_storage, act_len, env_idx
                )

            # World kinematics
            timesteps["world"] = {
                "torso_xpos": batch_torso_xpos[:ep_len, env_idx].copy(),
                "torso_xmat": batch_torso_xmat[:ep_len, env_idx].copy(),
                "torso_linvel": batch_torso_linvel[
                    :ep_len, env_idx
                ].copy(),
                "skull_xpos": batch_skull_xpos[:ep_len, env_idx].copy(),
                "skull_xmat": batch_skull_xmat[:ep_len, env_idx].copy(),
                "cam_left_xpos": batch_cam_left[:ep_len, env_idx].copy(),
                "cam_right_xpos": batch_cam_right[:ep_len, env_idx].copy(),
                "hand_l_xpos": batch_hand_l_xpos[:ep_len, env_idx].copy(),
                "hand_r_xpos": batch_hand_r_xpos[:ep_len, env_idx].copy(),
                **({"palm_l_touch": batch_palm_l_touch[:ep_len, env_idx].copy(),
                    "palm_r_touch": batch_palm_r_touch[:ep_len, env_idx].copy()}
                   if has_touch else {}),
            }

            episode_data["timesteps"] = timesteps

            # Gap geometry
            episode_data["gap_geometry"] = {
                "gap_leading_edges": gap_edges[env_idx].copy(),
                "gap_lengths": gap_lens[env_idx].copy(),
            }

            collected_episodes.append(episode_data)
            batch_kept += 1

        total_batches_run += 1
        print(
            f"  Batch {total_batches_run} done: "
            f"kept {batch_kept}/{n_envs} episodes "
            f"(total {len(collected_episodes)}/{n_episodes})",
            flush=True,
        )

    return collected_episodes


# ---------------------------------------------------------------------------
# HDF5 output building
# ---------------------------------------------------------------------------


def build_output_dict(
    episodes: List[Dict[str, Any]],
    config: RolloutCollectionConfig,
    ckpt_config: Dict[str, Any],
    rodent_qpos_start: int,
    rodent_qvel_start: int,
    fixed_gap_config: Optional[FixedGapConfig] = None,
) -> Dict[str, Any]:
    """Build the nested output dictionary for HDF5 serialization.

    Args:
        episodes: List of collected episode dicts.
        config: Rollout collection configuration.
        ckpt_config: Checkpoint configuration.
        rodent_qpos_start: Start index of rodent qpos in full qpos.
        rodent_qvel_start: Start index of rodent qvel in full qvel.
        fixed_gap_config: Fixed-gap config (if mode is fixed_gap).

    Returns:
        Nested dict ready for ``save_to_h5py``.
    """
    env_args = ckpt_config["env_config"].get("env_args", {})

    # Metadata
    metadata: Dict[str, Any] = {
        "checkpoint_path": config.checkpoint_path,
        "condition_name": "binocular",
        "rollout_mode": config.mode,
        "target_speed": float(env_args.get("target_speed", 0.0)),
        "gap_length_range": list(env_args.get("gap_length_range", [0.0, 0.0])),
        "n_platforms": int(env_args.get("n_platforms", 10)),
        "n_episodes": len(episodes),
        "episode_length": config.episode_length,
        "seed": config.seed,
        "capture_vision": config.capture_vision,
        "capture_activations": config.capture_activations,
        "capture_cnn_maps": config.capture_cnn_maps,
        "network_config": json.dumps(ckpt_config.get("network_config", {})),
        "env_config": json.dumps(ckpt_config.get("env_config", {})),
        "collection_timestamp": datetime.now().isoformat(),
    }

    if config.mode == "variable_gap":
        metadata["min_gaps_crossed"] = config.min_gaps_crossed
    elif config.mode == "fixed_gap" and fixed_gap_config is not None:
        metadata["fixed_gap_min"] = fixed_gap_config.gap_min
        metadata["fixed_gap_max"] = fixed_gap_config.gap_max
        metadata["fixed_gap_step"] = fixed_gap_config.gap_step

    # Episodes
    episodes_dict: Dict[str, Any] = {}
    gaps_crossed_list = []
    episode_lengths_list = []
    total_rewards_list = []

    for i, ep in enumerate(episodes):
        episodes_dict[str(i)] = ep
        gaps_crossed_list.append(ep["gaps_crossed"])
        episode_lengths_list.append(ep["n_steps"])
        total_rewards_list.append(ep["total_reward"])

    # Summary
    summary: Dict[str, Any] = {
        "n_episodes_total": len(episodes),
        "n_episodes_kept": len(episodes),
        "gaps_crossed_distribution": np.array(gaps_crossed_list, dtype=np.int32),
        "episode_lengths": np.array(episode_lengths_list, dtype=np.int32),
        "mean_reward": float(np.mean(total_rewards_list))
        if total_rewards_list
        else 0.0,
        "rodent_qpos_start": rodent_qpos_start,
        "rodent_qvel_start": rodent_qvel_start,
    }

    # qpos layout documentation
    qpos_layout: Dict[str, Any] = {
        "description": (
            "Full qpos including platform slide joints. "
            "Rodent root (freejoint) starts at rodent_qpos_start-7. "
            "Rodent joint angles start at rodent_qpos_start."
        ),
        "rodent_qpos_start": rodent_qpos_start,
        "rodent_qvel_start": rodent_qvel_start,
        "root_qpos_slice": f"qpos[{rodent_qpos_start - 7}:{rodent_qpos_start}]",
        "joint_qpos_slice": f"qpos[{rodent_qpos_start}:]",
    }

    return {
        "metadata": metadata,
        "episodes": episodes_dict,
        "summary": summary,
        "qpos_layout": qpos_layout,
    }


# ---------------------------------------------------------------------------
# Variable-gap mode
# ---------------------------------------------------------------------------


def run_variable_gap_mode(
    config: RolloutCollectionConfig,
    ckpt_config: Dict[str, Any],
) -> None:
    """Run variable-gap rollout collection and save to HDF5.

    Args:
        config: Rollout collection configuration.
        ckpt_config: Parsed checkpoint configuration.
    """
    from track_mjx.analysis.utils import save_to_h5py

    # Resolve prior path
    prior_path = config.prior_checkpoint_path
    if prior_path is None:
        prior_path = ckpt_config.get("transfer", {}).get(
            "prior_checkpoint_path"
        )
        if prior_path is None:
            raise ValueError(
                "No prior_checkpoint_path specified and none found in "
                "checkpoint config.json transfer section."
            )
    print(f"  Prior checkpoint: {prior_path}")

    # Setup env and policy
    print("[2/4] Setting up environment and policy...")
    (
        wrapped_env,
        policy_fn,
        params_tuple,
        mj_model,
        base_env,
        arch,
        init_hidden_fn,
    ) = setup_env_and_policy(
        checkpoint_path=config.checkpoint_path,
        prior_checkpoint_path=prior_path,
        ckpt_config=ckpt_config,
        seed=config.seed,
        get_activation=config.capture_activations,
    )

    # Resolve IDs
    print("[3/4] Resolving body and camera IDs...")
    body_cam_ids = resolve_body_and_camera_ids(mj_model)
    for name, obj_id in body_cam_ids.items():
        print(f"  {name}: {obj_id}")

    rodent_qpos_start = int(base_env._rodent_qpos_start)
    rodent_qvel_start = int(base_env._rodent_qvel_start)
    print(f"  rodent_qpos_start: {rodent_qpos_start}")
    print(f"  rodent_qvel_start: {rodent_qvel_start}")

    # Collect episodes
    print(
        f"[4/4] Collecting {config.n_episodes} episodes "
        f"(min_gaps_crossed={config.min_gaps_crossed})..."
    )
    episodes = collect_episodes_batch(
        wrapped_env=wrapped_env,
        policy_fn=policy_fn,
        params_tuple=params_tuple,
        base_env=base_env,
        mj_model=mj_model,
        body_cam_ids=body_cam_ids,
        n_episodes=config.n_episodes,
        max_steps=config.episode_length,
        n_envs=config.n_envs,
        seed=config.seed,
        arch=arch,
        init_hidden_fn=init_hidden_fn,
        capture_vision=config.capture_vision,
        capture_activations=config.capture_activations,
        capture_cnn_maps=config.capture_cnn_maps,
        min_gaps_crossed=config.min_gaps_crossed,
        ckpt_config=ckpt_config,
    )

    # Build output
    output_dict = build_output_dict(
        episodes=episodes,
        config=config,
        ckpt_config=ckpt_config,
        rodent_qpos_start=rodent_qpos_start,
        rodent_qvel_start=rodent_qvel_start,
    )

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(config.output_path)), exist_ok=True)
    save_to_h5py(config.output_path, output_dict)
    file_size_mb = os.path.getsize(config.output_path) / (1024 * 1024)
    print(f"\n  Saved: {config.output_path} ({file_size_mb:.1f} MB)")

    # Summary
    if episodes:
        gaps = [ep["gaps_crossed"] for ep in episodes]
        print(f"\n  Summary:")
        print(f"    Episodes collected: {len(episodes)}")
        print(f"    Mean gaps crossed: {np.mean(gaps):.1f}")
        print(f"    Mean reward: {output_dict['summary']['mean_reward']:.1f}")


# ---------------------------------------------------------------------------
# Fixed-gap mode
# ---------------------------------------------------------------------------


def run_fixed_gap_mode(
    config: RolloutCollectionConfig,
    ckpt_config: Dict[str, Any],
    fixed_config: FixedGapConfig,
) -> None:
    """Run fixed-gap sweep rollout collection and save to HDF5.

    For each gap length in the sweep, creates a new environment with
    ``n_platforms=n_gaps_per_env`` and ``gap_length_range=[gl, gl]``,
    collects episodes, and aggregates all into a single HDF5 file.

    Args:
        config: Rollout collection configuration.
        ckpt_config: Parsed checkpoint configuration.
        fixed_config: Fixed-gap sweep configuration.
    """
    from track_mjx.analysis.utils import save_to_h5py

    # Resolve prior path
    prior_path = config.prior_checkpoint_path
    if prior_path is None:
        prior_path = ckpt_config.get("transfer", {}).get(
            "prior_checkpoint_path"
        )
        if prior_path is None:
            raise ValueError(
                "No prior_checkpoint_path specified and none found in "
                "checkpoint config.json transfer section."
            )
    print(f"  Prior checkpoint: {prior_path}")

    # Generate gap lengths to sweep
    gap_lengths = np.arange(
        fixed_config.gap_min,
        fixed_config.gap_max + fixed_config.gap_step / 2,
        fixed_config.gap_step,
    )
    gap_lengths = np.round(gap_lengths, 4)  # avoid float precision issues
    print(f"  Sweeping {len(gap_lengths)} gap lengths: "
          f"{gap_lengths[0]:.3f} to {gap_lengths[-1]:.3f}")

    all_episodes: List[Dict[str, Any]] = []
    rodent_qpos_start = None
    rodent_qvel_start = None

    for gi, gl in enumerate(gap_lengths):
        print(f"\n{'='*60}")
        print(f"  Gap length {gi+1}/{len(gap_lengths)}: {gl:.4f} m")
        print(f"{'='*60}")

        # Create env with fixed gap length
        env_overrides = {
            "gap_length_range": [float(gl), float(gl)],
            "n_platforms": fixed_config.n_gaps_per_env,
        }

        (
            wrapped_env,
            policy_fn,
            params_tuple,
            mj_model,
            base_env,
            arch,
            init_hidden_fn,
        ) = setup_env_and_policy(
            checkpoint_path=config.checkpoint_path,
            prior_checkpoint_path=prior_path,
            ckpt_config=ckpt_config,
            seed=config.seed,
            env_overrides=env_overrides,
            get_activation=config.capture_activations,
        )

        if rodent_qpos_start is None:
            body_cam_ids = resolve_body_and_camera_ids(mj_model)
            rodent_qpos_start = int(base_env._rodent_qpos_start)
            rodent_qvel_start = int(base_env._rodent_qvel_start)
            print(f"  rodent_qpos_start: {rodent_qpos_start}")
            print(f"  rodent_qvel_start: {rodent_qvel_start}")
        else:
            body_cam_ids = resolve_body_and_camera_ids(mj_model)

        episodes = collect_episodes_batch(
            wrapped_env=wrapped_env,
            policy_fn=policy_fn,
            params_tuple=params_tuple,
            base_env=base_env,
            mj_model=mj_model,
            body_cam_ids=body_cam_ids,
            n_episodes=fixed_config.episodes_per_gap,
            max_steps=config.episode_length,
            n_envs=config.n_envs,
            seed=config.seed + gi,
            arch=arch,
            init_hidden_fn=init_hidden_fn,
            capture_vision=config.capture_vision,
            capture_activations=config.capture_activations,
            capture_cnn_maps=config.capture_cnn_maps,
            min_gaps_crossed=fixed_config.min_gaps_crossed,
            gap_length=float(gl),
            ckpt_config=ckpt_config,
        )

        all_episodes.extend(episodes)
        print(
            f"  Gap {gl:.4f}: collected {len(episodes)} episodes "
            f"(total: {len(all_episodes)})"
        )

        # Free GPU memory between gap lengths (warp caches accumulate)
        import gc as gc_mod
        del wrapped_env, policy_fn, params_tuple, base_env
        gc_mod.collect()
        jax.clear_caches()

    # Build output
    output_dict = build_output_dict(
        episodes=all_episodes,
        config=config,
        ckpt_config=ckpt_config,
        rodent_qpos_start=rodent_qpos_start or 0,
        rodent_qvel_start=rodent_qvel_start or 0,
        fixed_gap_config=fixed_config,
    )

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(config.output_path)), exist_ok=True)
    save_to_h5py(config.output_path, output_dict)
    file_size_mb = os.path.getsize(config.output_path) / (1024 * 1024)
    print(f"\n  Saved: {config.output_path} ({file_size_mb:.1f} MB)")

    # Summary
    if all_episodes:
        gaps = [ep["gaps_crossed"] for ep in all_episodes]
        print(f"\n  Summary:")
        print(f"    Total episodes: {len(all_episodes)}")
        print(f"    Gap lengths swept: {len(gap_lengths)}")
        print(f"    Mean gaps crossed: {np.mean(gaps):.1f}")
        print(f"    Mean reward: {output_dict['summary']['mean_reward']:.1f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Collect comprehensive rollout data from a trained binocular "
            "vision run-gap agent. Captures observations, network activations "
            "(including CNN spatial maps), kinematics, and full qpos/qvel. "
            "Outputs to HDF5."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the trained policy checkpoint directory.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path for the output HDF5 file.",
    )

    # Optional arguments
    parser.add_argument(
        "--prior_checkpoint_path",
        type=str,
        default=None,
        help=(
            "Path to the SCAMPER prior checkpoint directory. "
            "If not specified, loaded from config.json."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="variable_gap",
        choices=["variable_gap", "fixed_gap"],
        help="Rollout collection mode.",
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=50,
        help="Number of episodes to collect (variable_gap mode).",
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        default=64,
        help="Number of parallel environments per batch.",
    )
    parser.add_argument(
        "--episode_length",
        type=int,
        default=1000,
        help="Maximum steps per episode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--min_gaps_crossed",
        type=int,
        default=8,
        help="Minimum gaps crossed to keep episode (variable_gap mode).",
    )

    # Capture flags
    parser.add_argument(
        "--capture_vision",
        action="store_true",
        default=True,
        help="Store raw binocular vision images (default: True).",
    )
    parser.add_argument(
        "--no_capture_vision",
        action="store_true",
        default=False,
        help="Disable vision capture.",
    )
    parser.add_argument(
        "--capture_activations",
        action="store_true",
        default=True,
        help="Store network activations (default: True).",
    )
    parser.add_argument(
        "--capture_cnn_maps",
        action="store_true",
        default=False,
        help="Store per-layer spatial CNN feature maps (large).",
    )

    # Fixed-gap specific arguments
    fixed_group = parser.add_argument_group("Fixed-gap sweep options")
    fixed_group.add_argument(
        "--gap_min",
        type=float,
        default=0.03,
        help="Minimum gap length in meters.",
    )
    fixed_group.add_argument(
        "--gap_max",
        type=float,
        default=0.20,
        help="Maximum gap length in meters.",
    )
    fixed_group.add_argument(
        "--gap_step",
        type=float,
        default=0.01,
        help="Gap length sweep step size in meters.",
    )
    fixed_group.add_argument(
        "--n_gaps_per_env",
        type=int,
        default=5,
        help="Number of identical gap platforms per environment.",
    )
    fixed_group.add_argument(
        "--episodes_per_gap",
        type=int,
        default=20,
        help="Number of rollout episodes per gap length.",
    )
    fixed_group.add_argument(
        "--fixed_min_gaps_crossed",
        type=int,
        default=0,
        help="Minimum gaps crossed to keep episode (fixed_gap mode).",
    )

    return parser.parse_args()


def main():
    """Main entry point for HDF5 rollout data collection.

    1. Parse CLI arguments and build config dataclasses.
    2. Load checkpoint configuration.
    3. Dispatch to variable_gap or fixed_gap mode.
    4. Save HDF5 output.
    """
    args = parse_args()

    # Handle --no_capture_vision
    capture_vision = args.capture_vision and not args.no_capture_vision

    # Build config
    config = RolloutCollectionConfig(
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        prior_checkpoint_path=args.prior_checkpoint_path,
        mode=args.mode,
        n_episodes=args.n_episodes,
        n_envs=args.n_envs,
        episode_length=args.episode_length,
        seed=args.seed,
        min_gaps_crossed=args.min_gaps_crossed,
        capture_vision=capture_vision,
        capture_activations=args.capture_activations,
        capture_cnn_maps=args.capture_cnn_maps,
    )

    fixed_config = FixedGapConfig(
        gap_min=args.gap_min,
        gap_max=args.gap_max,
        gap_step=args.gap_step,
        n_gaps_per_env=args.n_gaps_per_env,
        episodes_per_gap=args.episodes_per_gap,
        min_gaps_crossed=args.fixed_min_gaps_crossed,
    )

    print("=" * 60)
    print("HDF5 Rollout Data Collector")
    print("=" * 60)
    print(f"  Checkpoint:       {config.checkpoint_path}")
    print(f"  Output:           {config.output_path}")
    print(f"  Prior:            {config.prior_checkpoint_path or '(from config.json)'}")
    print(f"  Mode:             {config.mode}")
    print(f"  N envs:           {config.n_envs}")
    print(f"  Episode length:   {config.episode_length}")
    print(f"  Seed:             {config.seed}")
    print(f"  Capture vision:   {config.capture_vision}")
    print(f"  Capture acts:     {config.capture_activations}")
    print(f"  Capture CNN maps: {config.capture_cnn_maps}")
    if config.mode == "variable_gap":
        print(f"  N episodes:       {config.n_episodes}")
        print(f"  Min gaps crossed: {config.min_gaps_crossed}")
    else:
        print(f"  Gap range:        [{fixed_config.gap_min}, {fixed_config.gap_max}]")
        print(f"  Gap step:         {fixed_config.gap_step}")
        print(f"  Gaps per env:     {fixed_config.n_gaps_per_env}")
        print(f"  Episodes/gap:     {fixed_config.episodes_per_gap}")
        print(f"  Min gaps crossed: {fixed_config.min_gaps_crossed}")
    print()

    # Load checkpoint config
    print("[1/4] Loading checkpoint configuration...")
    ckpt_config = load_config(config.checkpoint_path)
    env_name = ckpt_config["env_config"]["env_name"]
    arch_name = ckpt_config["network_config"]["arch_name"]
    print(f"  Environment: {env_name}")
    print(f"  Architecture: {arch_name}")

    env_args = ckpt_config["env_config"].get("env_args", {})
    env_cfg = ckpt_config["env_config"]
    print(
        f"  Vision: "
        f"{env_cfg.get('vision_width', 32)}x{env_cfg.get('vision_height', 32)}"
    )
    print(f"  Binocular: {env_cfg.get('binocular', False)}")
    print(f"  Platforms: {env_args.get('n_platforms', '?')}")
    print(f"  Gap range: {env_args.get('gap_length_range', '?')}")
    print()

    # Dispatch
    if config.mode == "variable_gap":
        run_variable_gap_mode(config, ckpt_config)
    elif config.mode == "fixed_gap":
        run_fixed_gap_mode(config, ckpt_config, fixed_config)
    else:
        raise ValueError(f"Unknown mode: {config.mode}")

    print("\nDone.")


if __name__ == "__main__":
    main()
