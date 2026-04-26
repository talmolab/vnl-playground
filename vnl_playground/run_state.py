"""Run state persistence for crash recovery and wandb resume.

Provides atomic, file-locked run state management so that training can
auto-resume after OOM/crash without relying on external bash scripts to
capture run IDs from log output.

Ported from track-mjx's checkpointing module, adapted for vnl-playground:
  - Uses hostname + config_hash for file naming (no PID), enabling
    auto-discovery across OOM restarts where the PID changes.
  - Strips volatile config keys (e.g., resume_run_id) before hashing
    so the hash is stable between initial run and restart.

Typical flow in train_highlvl.py:

    existing = discover_existing_run_state(cfg)
    if existing:
        run_id, checkpoint_path = existing["run_id"], existing["checkpoint_path"]
        wandb_run_id, wandb_resume = existing["wandb_run_id"], "must"
    else:
        run_id = ...  # fresh
        wandb_run_id, wandb_resume = ..., "allow"

    wandb.init(id=wandb_run_id, resume=wandb_resume, ...)

    save_run_state(cfg, run_id, checkpoint_path, wandb_run_id)
    callback = create_checkpoint_callback(cfg, run_id, checkpoint_path, wandb_run_id)
    train_fn(..., checkpoint_callback=callback)

    cleanup_run_state(cfg)  # on success
"""

import fcntl
import hashlib
import json
import logging
import socket
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from omegaconf import DictConfig, OmegaConf


# ---------------------------------------------------------------------------
# Config hashing
# ---------------------------------------------------------------------------

# Keys stripped before hashing so the hash is stable across restarts.
_VOLATILE_KEYS = {
    ("train_setup", "resume_run_id"),
}


def _hash_config(cfg: DictConfig) -> str:
    """Create a short hash of the config for consistency checking.

    Strips volatile keys (like ``resume_run_id``) so the hash is the same
    between a fresh launch and an OOM retry.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    for key_path in _VOLATILE_KEYS:
        d = cfg_dict
        for part in key_path[:-1]:
            if isinstance(d, dict) and part in d:
                d = d[part]
            else:
                d = None
                break
        if isinstance(d, dict) and key_path[-1] in d:
            del d[key_path[-1]]

    cfg_str = json.dumps(cfg_dict, sort_keys=True)
    return hashlib.md5(cfg_str.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def _get_run_state_file_path(cfg: DictConfig) -> Path:
    """Run state file path: ``{model_path}/run_state_{host}_{config_hash}.json``."""
    base_path = Path(cfg.logging_config.model_path)
    if not base_path.is_absolute():
        base_path = Path.cwd() / base_path
    base_path = base_path.resolve()
    hostname = socket.gethostname()
    config_hash = _hash_config(cfg)
    return base_path / f"run_state_{hostname}_{config_hash}.json"


def _atomic_write_json(file_path: Path, data: dict[str, Any]) -> None:
    """Write JSON atomically via temp-file + rename."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=file_path.parent, delete=False, suffix=".tmp"
    ) as tmp:
        json.dump(data, tmp, indent=2)
        tmp_name = tmp.name
    Path(tmp_name).rename(file_path)


def _read_json_with_lock(file_path: Path) -> dict[str, Any] | None:
    """Read JSON with shared file lock."""
    if not file_path.exists():
        return None
    try:
        with open(file_path, "r") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            data = json.load(f)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return data
    except (json.JSONDecodeError, OSError) as e:
        logging.warning(f"Failed to read run state file {file_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def discover_existing_run_state(cfg: DictConfig) -> dict[str, Any] | None:
    """Auto-discover a previous run state for this host + config.

    Validates config hash, checkpoint directory existence, and checkpoint
    contents (via orbax).  Returns the run state dict with
    ``latest_checkpoint_step`` added, or ``None`` if nothing valid is found.
    """
    import orbax.checkpoint as ocp

    state_file = _get_run_state_file_path(cfg)
    logging.info(f"Looking for existing run state at: {state_file}")

    run_state = _read_json_with_lock(state_file)
    if not run_state:
        logging.info("No existing run state found")
        return None

    # Validate required keys
    required = {"run_id", "checkpoint_path", "wandb_run_id", "config_hash"}
    if not required.issubset(run_state):
        logging.warning("Run state file missing required keys, ignoring")
        return None

    # Validate config hash
    current_hash = _hash_config(cfg)
    if run_state["config_hash"] != current_hash:
        logging.warning(
            f"Config hash mismatch (saved={run_state['config_hash']}, "
            f"current={current_hash}), ignoring run state"
        )
        return None

    # Validate checkpoint directory
    ckpt_path = Path(run_state["checkpoint_path"])
    if not ckpt_path.exists():
        logging.warning(f"Checkpoint directory {ckpt_path} not found, ignoring")
        return None

    # Validate checkpoint contents
    try:
        ckpt_mgr = ocp.CheckpointManager(
            str(ckpt_path),
            options=ocp.CheckpointManagerOptions(
                create=False, step_prefix="PPONetwork"
            ),
        )
        latest_step = ckpt_mgr.latest_step()
        if latest_step is None:
            logging.warning("No valid checkpoints found in directory, ignoring")
            return None
        run_state["latest_checkpoint_step"] = latest_step
        logging.info(
            f"Found valid run state: run_id={run_state['run_id']}, "
            f"checkpoint step={latest_step}"
        )
        return run_state
    except Exception as e:
        logging.warning(f"Failed to access checkpoint manager: {e}, ignoring")
        return None


def save_run_state(
    cfg: DictConfig,
    run_id: str,
    checkpoint_path: str | Path,
    wandb_run_id: str,
    latest_step: int | None = None,
) -> None:
    """Persist run state for crash recovery."""
    state_file = _get_run_state_file_path(cfg)
    data = {
        "run_id": run_id,
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "wandb_run_id": wandb_run_id,
        "config_hash": _hash_config(cfg),
        "timestamp": time.time(),
    }
    if latest_step is not None:
        data["latest_checkpoint_step"] = latest_step
    try:
        _atomic_write_json(state_file, data)
        logging.info(f"Saved run state to {state_file}")
    except Exception as e:
        logging.error(f"Failed to save run state: {e}")


def cleanup_run_state(cfg: DictConfig) -> None:
    """Remove run state file after successful training completion."""
    state_file = _get_run_state_file_path(cfg)
    try:
        if state_file.exists():
            state_file.unlink()
            logging.info(f"Cleaned up run state: {state_file}")
    except Exception as e:
        logging.warning(f"Failed to cleanup run state: {e}")


def create_checkpoint_callback(
    cfg: DictConfig,
    run_id: str,
    checkpoint_path: str | Path,
    wandb_run_id: str,
) -> Callable[[int], None]:
    """Return a callback that updates run state after each checkpoint save."""

    def _callback(step: int) -> None:
        try:
            save_run_state(
                cfg=cfg,
                run_id=run_id,
                checkpoint_path=checkpoint_path,
                wandb_run_id=wandb_run_id,
                latest_step=step,
            )
        except Exception as e:
            logging.warning(f"Checkpoint callback failed: {e}")

    return _callback
