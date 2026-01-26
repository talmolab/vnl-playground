"""Configuration utilities for vnl-playground training pipelines.

This module provides helper functions for preparing and updating OmegaConf
configurations, including walker-specific path resolution and config merging.
"""

import json
import logging
import subprocess
from importlib.metadata import distribution
from pathlib import Path
from typing import Any

from ml_collections import config_dict
from omegaconf import DictConfig, OmegaConf

from vnl_playground.tasks.rodent import consts as rodent_consts


# Project root directory (vnl-playground/)
_PROJECT_ROOT = Path(__file__).parent.parent.parent


def _get_package_commit(package_name: str) -> str:
    """Get the git commit hash for an installed package.

    Works for both editable installs (file://) and VCS installs (git+https://).

    Args:
        package_name: Name of the installed package.

    Returns:
        Git commit hash, or "unknown" if not available.
    """
    try:
        dist = distribution(package_name)
        direct_url = json.loads(dist.read_text("direct_url.json"))
        url = direct_url.get("url", "")

        # Editable install: file:// URL pointing to local repo
        if url.startswith("file://"):
            path = url[7:]
            result = subprocess.run(
                ["git", "-C", path, "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip()

        # VCS install: git+https:// with vcs_info
        if "vcs_info" in direct_url:
            return direct_url["vcs_info"].get("commit_id", "unknown")

    except Exception:
        pass
    return "unknown"


def _resolve_data_path(relative_path: str) -> str:
    """Resolve a relative data path to an absolute path from the project root.

    Args:
        relative_path: Path relative to the project root (e.g., "data/rodent/file.h5").

    Returns:
        Absolute path as a string.
    """
    return str(_PROJECT_ROOT / relative_path)


def prepare_config(
    cfg: DictConfig,
) -> tuple[DictConfig, dict[str, Any], config_dict.ConfigDict]:
    """Prepare configuration by resolving walker-specific paths and creating config variants.

    Takes a Hydra/OmegaConf configuration and resolves walker-specific XML and
    reference data paths based on the walker_name. Updates the env_config section
    with these paths and walker settings, then returns multiple config formats.

    Args:
        cfg: The root OmegaConf configuration containing walker_config and env_config.

    Returns:
        A tuple containing:
            - cfg: The updated OmegaConf DictConfig with resolved paths.
            - cfg_dict: The full config as a plain Python dictionary.
            - env_cfg_ml: The env_config as an ml_collections ConfigDict.

    Raises:
        ValueError: If walker_name is not recognized.
        NotImplementedError: If the specified walker is not yet fully implemented.
    """
    walker_name = cfg.env_config.walker_name

    if walker_name == "rodent":
        logging.info("Using rodent walker")

        # Select walker XML based on walker_xml_type config
        walker_xml_type = getattr(cfg.walker_config, "walker_xml_type", "original")
        if walker_xml_type == "original":
            walker_xml_path = str(rodent_consts.RODENT_XML_PATH)
        elif walker_xml_type == "box_feet":
            walker_xml_path = str(rodent_consts.RODENT_BOX_FEET_PATH)
        elif walker_xml_type == "full_collision":
            walker_xml_path = str(rodent_consts.RODENT_FULL_COLLISION_XML)
        else:
            raise ValueError(
                f"Unknown walker_xml_type: {walker_xml_type}. "
                "Must be one of: 'original', 'box_feet', 'full_collision'"
            )
        logging.info(f"Using walker XML type: {walker_xml_type}")

        arena_xml_path = str(rodent_consts.ARENA_XML_PATH)
        reference_data_path = str(rodent_consts.IMITATION_REFERENCE_PATH)
    else:
        raise ValueError(f"Unknown walker name: {walker_name}")

    # Update env_config with resolved paths and walker settings
    OmegaConf.set_struct(cfg.env_config, False)
    OmegaConf.update(cfg.env_config, "walker_xml_path", walker_xml_path, merge=False)
    OmegaConf.update(cfg.env_config, "arena_xml_path", arena_xml_path, merge=False)
    OmegaConf.update(
        cfg.env_config, "reference_data_path", reference_data_path, merge=False
    )

    # Add walker config values to env_config for easy access
    OmegaConf.update(
        cfg.env_config,
        "torque_actuators",
        cfg.walker_config.torque_actuators,
        merge=False,
    )
    OmegaConf.update(
        cfg.env_config, "rescale_factor", cfg.walker_config.rescale_factor, merge=False
    )

    # Add commit tracking
    OmegaConf.update(
        cfg.env_config,
        "vnl_playground_commit",
        _get_package_commit("vnl-playground"),
        merge=False,
    )

    # Promote timing values from env_args to env_config level for compatibility
    # with wandb_logging.py which expects these at the env_config level
    if hasattr(cfg.env_config, "env_args"):
        env_args = cfg.env_config.env_args
        if hasattr(env_args, "ctrl_dt"):
            OmegaConf.update(cfg.env_config, "ctrl_dt", env_args.ctrl_dt, merge=False)
        if hasattr(env_args, "sim_dt"):
            OmegaConf.update(cfg.env_config, "sim_dt", env_args.sim_dt, merge=False)

    # Set default values for imitation-learning-specific fields if not present
    # These are used by wandb_logging.py for episode length calculation
    if not hasattr(cfg.env_config, "mocap_hz"):
        # Default to 50 Hz (matches typical render_fps)
        OmegaConf.update(cfg.env_config, "mocap_hz", 50, merge=False)
    if not hasattr(cfg.env_config, "clip_length"):
        OmegaConf.update(cfg.env_config, "clip_length", 2000, merge=False)

    OmegaConf.set_struct(cfg.env_config, True)

    # Create ml_collections ConfigDict for env_config
    env_cfg_dict = OmegaConf.to_container(cfg.env_config, resolve=True)
    env_cfg_ml = config_dict.ConfigDict(env_cfg_dict)

    # Convert full config to dict and log
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    logging.info(f"Configs: {cfg_dict}")

    return cfg, cfg_dict, env_cfg_ml


def update_config(cfg: DictConfig, overrides: dict[str, Any]) -> DictConfig:
    """Update an OmegaConf configuration with override values.

    Temporarily disables struct mode to allow adding/modifying keys,
    applies all overrides, then re-enables struct mode.

    Args:
        cfg: The OmegaConf DictConfig to update.
        overrides: Dictionary of key-value pairs to apply. Keys can use
            dot notation for nested updates (e.g., "env_config.num_envs").

    Returns:
        The updated DictConfig (modified in-place).
    """
    OmegaConf.set_struct(cfg, False)
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, merge=False)
    OmegaConf.set_struct(cfg, True)
    return cfg
