"""Unit tests for vnl_playground.eval_metrics.emg."""
from __future__ import annotations

import numpy as np
import pytest


def test_module_imports():
    from vnl_playground.eval_metrics import emg
    assert emg.LAG_RANGE_STEPS_DEFAULT == 20
