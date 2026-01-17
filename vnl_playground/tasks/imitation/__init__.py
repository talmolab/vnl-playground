"""Imitation learning environments for all organisms.

Like mujoco_playground's locomotion/__init__.py - static imports,
dict-based registry, load() function.
"""

from typing import Any, Callable, Optional, Type

from ml_collections import config_dict

from vnl_playground.tasks.rodent import imitation as rodent_imitation
from vnl_playground.tasks.fruitfly import imitation as fruitfly_imitation

# Unified wrappers and reference clips
from vnl_playground.tasks.wrappers import FlattenObsWrapper
from vnl_playground.tasks.reference_clips import ReferenceClips


# Registry dicts (like locomotion's _envs, _cfgs)
_envs = {
    "RodentImitation": rodent_imitation.Imitation,
    "FruitflyImitation": fruitfly_imitation.Imitation,
}

_cfgs = {
    "RodentImitation": rodent_imitation.default_config,
    "FruitflyImitation": fruitfly_imitation.default_config,
}

# Single wrapper class for all environments
_wrappers = {
    "RodentImitation": FlattenObsWrapper,
    "FruitflyImitation": FlattenObsWrapper,
}

# Single ReferenceClips class for all environments
_reference_clips = {
    "RodentImitation": ReferenceClips,
    "FruitflyImitation": ReferenceClips,
}


def __getattr__(name):
    """Lazy attribute for ALL_ENVS."""
    if name == "ALL_ENVS":
        return tuple(_envs.keys())
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def register_environment(
    env_name: str,
    env_class: Type,
    cfg_class: Callable[[], config_dict.ConfigDict],
    wrapper_class: Type = None,
    reference_clips_class: Type = None,
) -> None:
    """Register a new imitation environment at runtime."""
    _envs[env_name] = env_class
    _cfgs[env_name] = cfg_class
    if wrapper_class:
        _wrappers[env_name] = wrapper_class
    if reference_clips_class:
        _reference_clips[env_name] = reference_clips_class


def get_default_config(env_name: str) -> config_dict.ConfigDict:
    """Get the default configuration for an environment."""
    if env_name not in _cfgs:
        raise ValueError(f"Env '{env_name}' not found. Available: {list(_cfgs.keys())}")
    return _cfgs[env_name]()


def load(
    env_name: str,
    config: Optional[config_dict.ConfigDict] = None,
    clips: Any = None,
    flatten_obs: bool = True,
):
    """Load an imitation environment instance.

    Args:
        env_name: Environment name (e.g., "RodentImitation").
        config: Configuration dict. Uses default if not provided.
        clips: ReferenceClips for training/evaluation.
        flatten_obs: Whether to apply FlattenObsWrapper.

    Returns:
        Instantiated environment.
    """
    if env_name not in _envs:
        raise ValueError(f"Env '{env_name}' not found. Available: {list(_envs.keys())}")
    config = config or get_default_config(env_name)
    env = _envs[env_name](config=config, clips=clips)

    if flatten_obs and env_name in _wrappers:
        env = _wrappers[env_name](env)

    return env


def load_reference_clips(
    env_name: str,
    data_path: str,
    n_frames_per_clip: int,
    keep_clips_idx=None,
    joint_names: Optional[list[str]] = None,
    body_names: Optional[list[str]] = None,
):
    """Load reference clips for an imitation environment.

    Args:
        env_name: Environment name (e.g., "RodentImitation").
        data_path: Path to HDF5 reference data.
        n_frames_per_clip: Number of frames per clip.
        keep_clips_idx: Optional indices to keep.
        joint_names: Optional list of joint names. If None, read from H5 or use indices.
        body_names: Optional list of body names. If None, read from H5 or use indices.

    Returns:
        Instantiated ReferenceClips object.
    """
    if env_name not in _reference_clips:
        raise ValueError(
            f"Env '{env_name}' not found. Available: {list(_reference_clips.keys())}"
        )
    return _reference_clips[env_name](
        data_path=data_path,
        n_frames_per_clip=n_frames_per_clip,
        keep_clips_idx=keep_clips_idx,
        joint_names=joint_names,
        body_names=body_names,
    )
