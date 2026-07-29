"""Environment registry for all VNL tasks.

Like mujoco_playground's locomotion/__init__.py - static imports,
dict-based registry, load() function.
"""

from collections.abc import Callable
from typing import Any

from ml_collections import config_dict

from vnl_playground.tasks.celegans import imitation as worm_imitation
from vnl_playground.tasks.fruitfly import imitation as fruitfly_imitation
from vnl_playground.tasks.fruitfly import (
    maintain_velocity as fruitfly_maintain_velocity,
)
from vnl_playground.tasks.mouse import imitation as mouse_imitation
from vnl_playground.tasks.mouse import mouse_reach
from vnl_playground.tasks.reference_clips import (
    load_reference_clips as load_reference_clip_data,
)
from vnl_playground.tasks.rodent import bowl_escape as rodent_bowl_escape
from vnl_playground.tasks.rodent import imitation as rodent_imitation
from vnl_playground.tasks.rodent import joystick as rodent_joystick
from vnl_playground.tasks.rodent import maintain_velocity as rodent_maintain_velocity
from vnl_playground.tasks.rodent import rearing as rodent_rearing
from vnl_playground.tasks.rodent import sparse_imitation as rodent_sparse_imitation
from vnl_playground.tasks.stick import imitation as stick_imitation
from vnl_playground.tasks.stick import maintain_velocity as stick_maintain_velocity

# Unified wrappers and reference clips
from vnl_playground.tasks.wrappers import FlattenObsWrapper

# Registry dicts (like locomotion's _envs, _cfgs)
_envs = {
    "RodentImitation": rodent_imitation.Imitation,
    "RodentSparseImitation": rodent_sparse_imitation.SparseImitation,
    "RodentRearing": rodent_rearing.Rearing,
    "RodentBowlEscape": rodent_bowl_escape.BowlEscape,
    "RodentMaintainVelocity": rodent_maintain_velocity.MaintainVelocity,
    "RodentJoystick": rodent_joystick.Joystick,
    "FruitflyImitation": fruitfly_imitation.Imitation,
    "FruitflyMaintainVelocity": fruitfly_maintain_velocity.MaintainVelocity,
    "MouseReach": mouse_reach.MouseReach,
    "MouseImitation": mouse_imitation.MouseImitation,
    "StickMaintainVelocity": stick_maintain_velocity.MaintainVelocity,
    "StickImitation": stick_imitation.Imitation,
    "WormImitation": worm_imitation.Imitation,
    "CelegansImitation": worm_imitation.Imitation,
}

_cfgs = {
    "RodentImitation": rodent_imitation.default_config,
    "RodentSparseImitation": rodent_sparse_imitation.default_config,
    "RodentRearing": rodent_rearing.default_config,
    "RodentBowlEscape": rodent_bowl_escape.default_config,
    "RodentMaintainVelocity": rodent_maintain_velocity.default_config,
    "RodentJoystick": rodent_joystick.default_config,
    "FruitflyImitation": fruitfly_imitation.default_config,
    "FruitflyMaintainVelocity": fruitfly_maintain_velocity.default_config,
    "MouseReach": mouse_reach.default_config,
    "MouseImitation": mouse_imitation.default_config,
    "StickMaintainVelocity": stick_maintain_velocity.default_config,
    "StickImitation": stick_imitation.default_config,
    "WormImitation": worm_imitation.default_config,
}

# Built-ins use the shared loader; runtime registrations may provide a custom one.
_reference_clip_loaders = {
    "RodentImitation": None,
    "RodentSparseImitation": None,
    "FruitflyImitation": None,
    "MouseImitation": None,
    "StickImitation": None,
    "WormImitation": None,
}


def __getattr__(name):
    """Lazy attribute for ALL_ENVS."""
    if name == "ALL_ENVS":
        return tuple(_envs.keys())
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def register_environment(
    env_name: str,
    env_class: type,
    cfg_class: Callable[[], config_dict.ConfigDict],
    wrapper_class: type | None = None,
    reference_clips_class: type | None = None,
) -> None:
    """Register a new environment at runtime."""
    _envs[env_name] = env_class
    _cfgs[env_name] = cfg_class
    if reference_clips_class is not None:
        _reference_clip_loaders[env_name] = reference_clips_class


def get_default_config(env_name: str) -> config_dict.ConfigDict:
    """Get the default configuration for an environment."""
    if env_name not in _cfgs:
        raise ValueError(
            f"Env '{env_name}' not found in default configs. Available configs:"
            f" {list(_cfgs.keys())}"
        )
    return _cfgs[env_name]()


def load(
    env_name: str,
    config: config_dict.ConfigDict | None = None,
    clips: Any = None,
    flatten_obs: bool = True,
    **kwargs,
):
    """Load an environment instance.

    Args:
        env_name: Environment name (e.g., "RodentImitation").
        config: Configuration dict. Uses default if not provided.
        clips: ReferenceClips for imitation environments.
        flatten_obs: Whether to apply FlattenObsWrapper.
        **kwargs: Additional arguments passed to environment constructor
            (e.g., rng for locomotion environments).

    Returns:
        Instantiated environment.
    """
    if env_name not in _envs:
        raise ValueError(f"Env '{env_name}' not found. Available: {list(_envs.keys())}")
    config = config or get_default_config(env_name)

    # Imitation envs use clips, locomotion envs use rng
    if env_name in _reference_clip_loaders:
        env = _envs[env_name](config=config, clips=clips, **kwargs)
    else:
        env = _envs[env_name](config=config, **kwargs)

    if flatten_obs:
        env = FlattenObsWrapper(env)

    return env


def load_reference_clips(
    env_name: str,
    data_path: str,
    n_frames_per_clip: int,
    clip_indices=None,
    **kwargs,
):
    """Load reference clips for an environment.

    Args:
        env_name: Environment name (e.g., "RodentImitation").
        data_path: Path to HDF5 reference data.
        n_frames_per_clip: Number of frames per clip.
        clip_indices: Optional source clip indices to keep.
        **kwargs: Additional arguments forwarded to the clips class
            (e.g., joint_names, body_names for ReferenceClips).

    Returns:
        Instantiated ReferenceClips object.
    """
    if env_name not in _reference_clip_loaders:
        raise ValueError(
            f"Env '{env_name}' does not support reference clips. "
            f"Available: {sorted(_reference_clip_loaders)}"
        )

    if (custom_loader := _reference_clip_loaders[env_name]) is not None:
        return custom_loader(
            data_path=data_path,
            n_frames_per_clip=n_frames_per_clip,
            keep_clips_idx=clip_indices,
            **kwargs,
        )

    config = get_default_config(env_name)
    kwargs.setdefault("data_format", config.get("reference_data_format", "stac"))
    if "joints" in config:
        kwargs.setdefault("joint_names", config.joints)
    if "bodies" in config:
        kwargs.setdefault("body_names", config.bodies)
    if "root_body" in config:
        kwargs.setdefault("root_body_name", config.root_body)
    return load_reference_clip_data(
        data_path,
        n_frames_per_clip=n_frames_per_clip,
        clip_indices=clip_indices,
        **kwargs,
    )
