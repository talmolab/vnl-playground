"""Unit tests for scripts/bo_optimize.py."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


# Load bo_optimize as a module even though scripts/ is not a package.
_SPEC = importlib.util.spec_from_file_location(
    "bo_optimize",
    Path(__file__).resolve().parents[1] / "scripts" / "bo_optimize.py",
)
bo = importlib.util.module_from_spec(_SPEC)
sys.modules["bo_optimize"] = bo
_SPEC.loader.exec_module(bo)
