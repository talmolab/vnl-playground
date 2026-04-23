"""EMG evaluation metrics shared by the trainer and the eval-replay script.

Pure-numpy implementations; no jax, no mujoco, no brax. All functions accept
(n_trials, T) arrays for sim and bio traces and return plain Python floats /
ints suitable for wandb logging.
"""
from __future__ import annotations

import numpy as np

LAG_RANGE_STEPS_DEFAULT = 20  # ±20 steps × ctrl-dt = ±50 ms at ctrl_dt=2.5 ms
