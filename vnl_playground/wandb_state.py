"""Wandb run state persistence for checkpoint resume.

Saves/loads wandb_run_id alongside training checkpoints so that
wandb.init(resume="must") can reconnect to the exact same run
after crash recovery.
"""

import json
import logging
from pathlib import Path
from typing import Any


def save_wandb_state(checkpoint_path: Path | str, wandb_run_id: str) -> None:
    """Save wandb run ID to a sidecar JSON file in the checkpoint directory.

    Args:
        checkpoint_path: Path to the checkpoint directory.
        wandb_run_id: The wandb run ID to persist.
    """
    state_file = Path(checkpoint_path) / "wandb_state.json"
    try:
        state_file.write_text(
            json.dumps({"wandb_run_id": wandb_run_id}, indent=2)
        )
        logging.info(f"Saved wandb state to {state_file}")
    except Exception as e:
        logging.warning(f"Failed to save wandb state: {e}")


def load_wandb_state(checkpoint_path: Path | str) -> dict[str, Any] | None:
    """Load wandb run state from a sidecar JSON file.

    Args:
        checkpoint_path: Path to the checkpoint directory.

    Returns:
        Dict with "wandb_run_id" key if found, None otherwise.
    """
    state_file = Path(checkpoint_path) / "wandb_state.json"
    if not state_file.exists():
        logging.info(f"No wandb state file found at {state_file}")
        return None

    try:
        data = json.loads(state_file.read_text())
        if "wandb_run_id" not in data:
            logging.warning("wandb_state.json missing wandb_run_id key")
            return None
        logging.info(f"Loaded wandb state: run_id={data['wandb_run_id']}")
        return data
    except (json.JSONDecodeError, OSError) as e:
        logging.warning(f"Failed to read wandb state: {e}")
        return None
