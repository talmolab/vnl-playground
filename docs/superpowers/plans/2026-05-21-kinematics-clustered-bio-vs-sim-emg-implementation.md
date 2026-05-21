# Kinematics-Clustered Bio vs Sim Muscle-Act Analysis - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible per-animal kinematic-clustering pipeline that compares bio EMG to sim muscle activations, packaged as a Colab notebook plus an HDF5 bundle suitable for handoff to Talmo and Austin.

**Architecture:** Python module `scripts/kin_emg_analysis.py` holds the pure functions (TDD with pytest). A converter `scripts/build_paired_deterministic_h5.py` materializes the source HDF5 from existing npz caches. The Colab notebook `notebooks/bio_vs_sim_kin_clustered.ipynb` is assembled at the end by pasting each tested function into a standalone cell. A bundle script tars the HDF5 outputs, notebook, converter, and figures.

**Tech Stack:** Python 3.12, h5py, numpy, scipy, scikit-learn, pandas, matplotlib, tqdm, pytest. uv-managed venv at `/root/vast/eric/vnl-playground/.venv`. No JAX or MuJoCo needed - the converter reads existing npz caches.

**Spec reference:** `docs/superpowers/specs/2026-05-21-kinematics-clustered-bio-vs-sim-emg-design.md`

**Style constraints (from spec):**
- No long-dash character (U+2014) anywhere in code or docs.
- Avoid the word "delve".
- Every function carries `Args:` / `Returns:` docstring.
- Loops over animals / clusters / trials use `tqdm`.
- Notebook cells are standalone; no `from scripts.kin_emg_analysis import ...` inside the notebook.

---

## File Structure

```
scripts/
├── build_paired_deterministic_h5.py    # converter, npz -> HDF5
├── kin_emg_analysis.py                  # pure analysis functions (clustering, GLM, etc.)
└── make_handoff_bundle.sh               # tar bundle script

tests/
├── test_kin_emg_analysis.py             # pytest tests against synthetic data
└── test_build_paired_deterministic_h5.py # converter schema test on a tiny synthetic input

notebooks/
└── bio_vs_sim_kin_clustered.ipynb       # Colab notebook, cells pasted from kin_emg_analysis.py

docs/superpowers/specs/
└── 2026-05-21-kinematics-clustered-bio-vs-sim-emg-design.md  # the spec (already exists)
```

**Path convention:** Outputs are written to a working directory the user controls. The default in the converter and notebook is `notebooks/kin_emg_bundle/`, created at runtime.

---

## Task 1: Add scipy, tqdm dependencies and install scikit-learn

**Files:**
- Modify: `pyproject.toml` (dependencies section)

- [ ] **Step 1: Confirm the current state of deps**

```bash
cd /root/vast/eric/vnl-playground
grep -E "scipy|tqdm|scikit-learn" pyproject.toml
```

Expected: `scikit-learn` already present; `scipy` and `tqdm` not.

- [ ] **Step 2: Add scipy and tqdm via uv**

```bash
cd /root/vast/eric/vnl-playground
uv add scipy tqdm
```

Expected output: uv updates `pyproject.toml`, regenerates `uv.lock`, installs scipy and tqdm into `.venv`.

- [ ] **Step 3: Install scikit-learn into the venv**

```bash
cd /root/vast/eric/vnl-playground
uv pip install scikit-learn
```

Expected: scikit-learn installs (it's already declared but not present in .venv).

- [ ] **Step 4: Smoke import test**

```bash
/root/vast/eric/vnl-playground/.venv/bin/python -c "
import scipy, sklearn, tqdm, h5py, numpy, pandas, matplotlib
print('scipy', scipy.__version__)
print('sklearn', sklearn.__version__)
print('tqdm', tqdm.__version__)
print('h5py', h5py.__version__)
print('numpy', numpy.__version__)
"
```

Expected: all imports succeed, versions print.

- [ ] **Step 5: Commit**

```bash
cd /root/vast/eric/vnl-playground
git add pyproject.toml uv.lock 2>/dev/null || git add pyproject.toml
git -c user.name="Eric Leonardis" -c user.email="leonardiseric@gmail.com" commit -m "$(cat <<'EOF'
deps: add scipy and tqdm for kin/EMG analysis

scikit-learn was already declared but not installed; scipy and tqdm are
new direct dependencies for clustering, regression, and progress bars in
the kinematics-clustered bio-vs-sim analysis.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Write failing test for paired_deterministic.h5 converter schema

**Files:**
- Create: `tests/test_build_paired_deterministic_h5.py`

- [ ] **Step 1: Create the test file with a synthetic input fixture**

`tests/test_build_paired_deterministic_h5.py`:

```python
"""Schema test for scripts/build_paired_deterministic_h5.py.

Builds a tiny synthetic features.npz and rollout cache, runs the converter,
asserts the HDF5 has the exact groups / shapes / attrs the spec requires.
"""
from pathlib import Path

import h5py
import numpy as np
import pytest


@pytest.fixture
def synthetic_inputs(tmp_path: Path) -> dict:
    """Build a 4-trial features.npz and 5-clip rollout cache for testing.

    Returns:
        Paths to the npz files and a working dir under tmp_path.
    """
    N = 4
    T = 60
    rollout_T = 100
    n_clips_cache = 5
    K, B, J, A, L, H = 9, 6, 7, 12, 4, 512

    # features.npz layout - matches existing
    # notebooks/kinematics_emg_comparison/cache/features.npz schema.
    features = {
        "X_bio_kin":   np.random.rand(N, T * K * 3).astype("float32"),
        "X_bio_emg":   np.random.rand(N, T * 2).astype("float32"),
        "X_sim_kin":   np.random.rand(N, T * K * 3).astype("float32"),
        "X_sim_emg":   np.random.rand(N, T * 2).astype("float32"),
        "X_bio_xpos":  np.random.rand(N, T * B * 3).astype("float32"),
        "X_sim_xpos":  np.random.rand(N, T * B * 3).astype("float32"),
        "X_bio_qpos":  np.random.rand(N, T * J).astype("float32"),
        "X_sim_qpos":  np.random.rand(N, T * J).astype("float32"),
        "meta_animal": np.array(["AT006", "AT006", "AT009", "AT009"], dtype="U5"),
        "meta_trial":  np.array([0, 1, 0, 1], dtype="int32"),
        "meta_rollout": np.array([0, 1, 2, 3], dtype="int32"),
        "bio_emg_per_muscle": np.random.rand(N, T, 3).astype("float32"),
        "sim_emg_per_muscle": np.random.rand(N, T, 3).astype("float32"),
    }
    features_path = tmp_path / "features.npz"
    np.savez(features_path, **features)

    # 278-clip rollout cache layout from talk_figures/figs/rollout_activations/.
    rollout = {
        "ctrl": np.random.rand(n_clips_cache, rollout_T, A).astype("float32") * 2 - 1,
        "act":  np.random.rand(n_clips_cache, rollout_T, A).astype("float32"),
        "qposes_rollout": np.random.rand(n_clips_cache, rollout_T, J).astype("float32"),
        "intention": np.random.rand(n_clips_cache, rollout_T, L).astype("float32"),
        "decoder_layer_0": np.random.rand(n_clips_cache, rollout_T, H).astype("float32"),
        "decoder_layer_1": np.random.rand(n_clips_cache, rollout_T, H).astype("float32"),
        "decoder_layer_2": np.random.rand(n_clips_cache, rollout_T, H).astype("float32"),
    }
    rollout_path = tmp_path / "rollout.npz"
    np.savez(rollout_path, **rollout)

    return {
        "features_npz": features_path,
        "rollout_npz": rollout_path,
        "out_h5": tmp_path / "paired_deterministic.h5",
        "N": N,
        "T": T,
    }


def test_converter_writes_all_top_level_groups(synthetic_inputs):
    """The output HDF5 must contain /meta, /bio, /sim top-level groups."""
    from scripts.build_paired_deterministic_h5 import build_paired_h5

    build_paired_h5(
        features_npz=synthetic_inputs["features_npz"],
        rollout_npz=synthetic_inputs["rollout_npz"],
        out_h5=synthetic_inputs["out_h5"],
        checkpoint="s18-ms-F4-fs1p2-test",
        checkpoint_step=0,
    )

    with h5py.File(synthetic_inputs["out_h5"], "r") as f:
        assert "meta" in f
        assert "bio" in f
        assert "sim" in f


def test_meta_arrays_have_correct_shapes(synthetic_inputs):
    from scripts.build_paired_deterministic_h5 import build_paired_h5

    build_paired_h5(
        features_npz=synthetic_inputs["features_npz"],
        rollout_npz=synthetic_inputs["rollout_npz"],
        out_h5=synthetic_inputs["out_h5"],
        checkpoint="test",
        checkpoint_step=0,
    )
    N = synthetic_inputs["N"]
    with h5py.File(synthetic_inputs["out_h5"], "r") as f:
        assert f["meta/animal"].shape == (N,)
        assert f["meta/trial"].shape == (N,)
        assert f["meta/rollout_row"].shape == (N,)


def test_bio_arrays_have_correct_shapes(synthetic_inputs):
    from scripts.build_paired_deterministic_h5 import build_paired_h5

    build_paired_h5(
        features_npz=synthetic_inputs["features_npz"],
        rollout_npz=synthetic_inputs["rollout_npz"],
        out_h5=synthetic_inputs["out_h5"],
        checkpoint="test",
        checkpoint_step=0,
    )
    N, T = synthetic_inputs["N"], synthetic_inputs["T"]
    with h5py.File(synthetic_inputs["out_h5"], "r") as f:
        assert f["bio/kin"].shape  == (N, T, 9, 3)
        assert f["bio/xpos"].shape == (N, T, 6, 3)
        assert f["bio/qpos"].shape == (N, T, 7)
        assert f["bio/emg"].shape  == (N, T, 2)


def test_sim_arrays_have_correct_shapes(synthetic_inputs):
    from scripts.build_paired_deterministic_h5 import build_paired_h5

    build_paired_h5(
        features_npz=synthetic_inputs["features_npz"],
        rollout_npz=synthetic_inputs["rollout_npz"],
        out_h5=synthetic_inputs["out_h5"],
        checkpoint="test",
        checkpoint_step=0,
    )
    N, T = synthetic_inputs["N"], synthetic_inputs["T"]
    with h5py.File(synthetic_inputs["out_h5"], "r") as f:
        assert f["sim/kin"].shape                 == (N, T, 9, 3)
        assert f["sim/xpos"].shape                == (N, T, 6, 3)
        assert f["sim/qpos"].shape                == (N, T, 7)
        assert f["sim/muscle_act"].shape          == (N, T, 12)
        assert f["sim/muscle_act_AD_biceps"].shape == (N, T, 2)
        assert f["sim/action_raw"].shape          == (N, T, 12)
        assert f["sim/intention"].shape           == (N, T, 4)
        assert f["sim/decoder_layer_0"].shape     == (N, T, 512)
        assert f["sim/decoder_layer_1"].shape     == (N, T, 512)
        assert f["sim/decoder_layer_2"].shape     == (N, T, 512)


def test_required_attrs_present(synthetic_inputs):
    from scripts.build_paired_deterministic_h5 import build_paired_h5

    build_paired_h5(
        features_npz=synthetic_inputs["features_npz"],
        rollout_npz=synthetic_inputs["rollout_npz"],
        out_h5=synthetic_inputs["out_h5"],
        checkpoint="my-net",
        checkpoint_step=99,
    )
    with h5py.File(synthetic_inputs["out_h5"], "r") as f:
        a = f["meta"].attrs
        assert a["checkpoint"] == "my-net"
        assert int(a["checkpoint_step"]) == 99
        assert int(a["target_T"]) == 60
        assert list(a["emg_muscle_names"]) == ["AD", "Biceps"]
        assert "created_utc" in a
        assert "actuator_AD_biceps_idx" in a
```

- [ ] **Step 2: Run the test, confirm it fails**

```bash
cd /root/vast/eric/vnl-playground
/root/vast/eric/vnl-playground/.venv/bin/python -m pytest tests/test_build_paired_deterministic_h5.py -v
```

Expected: tests fail with `ModuleNotFoundError: No module named 'scripts.build_paired_deterministic_h5'` (or `ImportError`).

- [ ] **Step 3: Commit the failing test**

```bash
cd /root/vast/eric/vnl-playground
git add tests/test_build_paired_deterministic_h5.py
git -c user.name="Eric Leonardis" -c user.email="leonardiseric@gmail.com" commit -m "$(cat <<'EOF'
test: schema test for paired_deterministic.h5 converter

TDD red bar. Asserts the converter writes /meta /bio /sim groups with
the exact shapes and attrs from the spec, using a 4-trial synthetic
input.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Implement the converter

**Files:**
- Create: `scripts/build_paired_deterministic_h5.py`

- [ ] **Step 1: Skeleton with `build_paired_h5` signature**

`scripts/build_paired_deterministic_h5.py`:

```python
"""Build paired_deterministic.h5 from existing npz caches.

Inputs:
  features_npz - notebooks/kinematics_emg_comparison/cache/features.npz
                 Per-trial bio + sim features, T=60 resampled grid, N=204
                 trials over 5 animals. Already has the 3-muscle EMG split.
  rollout_npz  - notebooks/talk_figures/figs/rollout_activations/
                 <network>_278clips.npz
                 Full 278-clip rollout cache with ctrl/act (12 actuators),
                 qposes_rollout, intention, and 3 decoder layer activations.

Output:
  paired_deterministic.h5 per the spec in
  docs/superpowers/specs/2026-05-21-kinematics-clustered-bio-vs-sim-emg-design.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import h5py
import numpy as np

# Muscle channel order in features.npz's bio_emg_per_muscle / sim_emg_per_muscle.
MUSCLES_FULL3 = ("AD", "Triceps", "Biceps")
MUSCLES_EMG = ("AD", "Biceps")  # the M=2 comparison set
EMG_MUSCLE_IDX = tuple(MUSCLES_FULL3.index(m) for m in MUSCLES_EMG)  # (0, 2)

# Default actuator names. The full ordering comes from the mouse model XML; the
# AD / Biceps indices are the first and third entries in the actuator block.
# If the model order changes, override via the CLI flag.
DEFAULT_ACTUATOR_NAMES = (
    "AD", "Triceps", "Biceps",
    "PD", "MD",
    "Pectoralis", "Latissimus",
    "Brachialis", "Brachioradialis",
    "PronTeres", "Supinator", "FCR",
)
DEFAULT_ACTUATOR_AD_BICEPS_IDX = (0, 2)

# Joint and body name defaults. Override via CLI if your model differs.
DEFAULT_QPOS_NAMES = (
    "shoulder_flex", "shoulder_abd", "shoulder_rot",
    "elbow_flex", "elbow_rot",
    "wrist_flex", "wrist_dev",
)
DEFAULT_XPOS_NAMES = (
    "shoulder", "upper_arm", "elbow", "forearm", "wrist", "hand",
)
DEFAULT_KIN_NAMES = (
    "snout", "ear_l", "ear_r", "shoulder",
    "elbow", "wrist", "hand", "tail_base", "spine_mid",
)


def build_paired_h5(
    *,
    features_npz: Path,
    rollout_npz: Path,
    out_h5: Path,
    checkpoint: str,
    checkpoint_step: int,
    actuator_names: tuple[str, ...] = DEFAULT_ACTUATOR_NAMES,
    actuator_AD_biceps_idx: tuple[int, int] = DEFAULT_ACTUATOR_AD_BICEPS_IDX,
    qpos_joint_names: tuple[str, ...] = DEFAULT_QPOS_NAMES,
    xpos_body_names: tuple[str, ...] = DEFAULT_XPOS_NAMES,
    kin_marker_names: tuple[str, ...] = DEFAULT_KIN_NAMES,
) -> None:
    """Assemble paired_deterministic.h5 from existing npz caches.

    Args:
        features_npz: Path to the paired-features npz (bio + sim per trial).
        rollout_npz: Path to the 278-clip rollout cache npz.
        out_h5: Output HDF5 path. Overwritten if it exists.
        checkpoint: Source checkpoint name string, stored as an attr.
        checkpoint_step: Source checkpoint step number, stored as an attr.
        actuator_names: 12 actuator names in model order.
        actuator_AD_biceps_idx: (AD_index, Biceps_index) into actuator_names.
        qpos_joint_names: 7 joint names matching qpos column order.
        xpos_body_names: 6 body-part names matching xpos's B axis.
        kin_marker_names: 9 marker names matching kin's K axis.

    Returns:
        None. Writes the HDF5 file at out_h5.
    """
    feats = np.load(features_npz, allow_pickle=False)
    roll  = np.load(rollout_npz, allow_pickle=False)

    N = len(feats["meta_animal"])
    T = 60
    K, B, J, A, L, H = 9, 6, 7, 12, 4, 512

    bio_kin  = feats["X_bio_kin"].reshape(N, T, K, 3)
    bio_xpos = feats["X_bio_xpos"].reshape(N, T, B, 3)
    bio_qpos = feats["X_bio_qpos"].reshape(N, T, J)
    bio_emg_full3 = feats["bio_emg_per_muscle"]              # (N, T, 3)
    bio_emg = bio_emg_full3[..., list(EMG_MUSCLE_IDX)]        # (N, T, 2)

    sim_kin  = feats["X_sim_kin"].reshape(N, T, K, 3)
    sim_xpos = feats["X_sim_xpos"].reshape(N, T, B, 3)
    sim_qpos = feats["X_sim_qpos"].reshape(N, T, J)

    rollout_row = feats["meta_rollout"]
    sim_muscle_act_full = _resample_rollout(roll["act"][rollout_row], T)        # (N, T, 12)
    sim_action_raw      = _resample_rollout(roll["ctrl"][rollout_row], T)       # (N, T, 12)
    sim_intention       = _resample_rollout(roll["intention"][rollout_row], T)  # (N, T, 4)
    sim_dec0 = _resample_rollout(roll["decoder_layer_0"][rollout_row], T)       # (N, T, 512)
    sim_dec1 = _resample_rollout(roll["decoder_layer_1"][rollout_row], T)
    sim_dec2 = _resample_rollout(roll["decoder_layer_2"][rollout_row], T)
    sim_muscle_act_emg = sim_muscle_act_full[..., list(actuator_AD_biceps_idx)]  # (N, T, 2)

    out_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_h5, "w") as f:
        # /meta
        meta = f.create_group("meta")
        animal_bytes = np.array([a.encode("utf-8") for a in feats["meta_animal"]])
        meta.create_dataset("animal", data=animal_bytes)
        meta.create_dataset("trial",  data=feats["meta_trial"].astype("int32"))
        meta.create_dataset("rollout_row", data=feats["meta_rollout"].astype("int32"))

        meta.attrs["checkpoint"] = checkpoint
        meta.attrs["checkpoint_step"] = int(checkpoint_step)
        meta.attrs["trial_duration_s"] = 0.25
        meta.attrs["target_T"] = T
        meta.attrs["ctrl_dt_s"] = 0.25 / T  # nominal
        meta.attrs["emg_muscle_names"] = list(MUSCLES_EMG)
        meta.attrs["actuator_names"]   = list(actuator_names)
        meta.attrs["actuator_AD_biceps_idx"] = list(actuator_AD_biceps_idx)
        meta.attrs["qpos_joint_names"] = list(qpos_joint_names)
        meta.attrs["xpos_body_names"]  = list(xpos_body_names)
        meta.attrs["kin_marker_names"] = list(kin_marker_names)
        meta.attrs["bio_emg_norm"] = "p98, no ceiling clip"
        meta.attrs["kin_detrended"] = True
        meta.attrs["source_features_npz"] = str(features_npz)
        meta.attrs["source_rollout_npz"]  = str(rollout_npz)
        meta.attrs["created_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()

        # /bio
        bio = f.create_group("bio")
        bio.create_dataset("kin",  data=bio_kin.astype("float32"),  compression="gzip")
        bio.create_dataset("xpos", data=bio_xpos.astype("float32"), compression="gzip")
        bio.create_dataset("qpos", data=bio_qpos.astype("float32"), compression="gzip")
        bio.create_dataset("emg",  data=bio_emg.astype("float32"),  compression="gzip")

        # /sim
        sim = f.create_group("sim")
        sim.create_dataset("kin",                  data=sim_kin.astype("float32"),  compression="gzip")
        sim.create_dataset("xpos",                 data=sim_xpos.astype("float32"), compression="gzip")
        sim.create_dataset("qpos",                 data=sim_qpos.astype("float32"), compression="gzip")
        sim.create_dataset("muscle_act",           data=sim_muscle_act_full.astype("float32"), compression="gzip")
        sim.create_dataset("muscle_act_AD_biceps", data=sim_muscle_act_emg.astype("float32"),  compression="gzip")
        sim.create_dataset("action_raw",           data=sim_action_raw.astype("float32"),       compression="gzip")
        sim.create_dataset("intention",            data=sim_intention.astype("float32"),        compression="gzip")
        sim.create_dataset("decoder_layer_0",      data=sim_dec0.astype("float32"),             compression="gzip")
        sim.create_dataset("decoder_layer_1",      data=sim_dec1.astype("float32"),             compression="gzip")
        sim.create_dataset("decoder_layer_2",      data=sim_dec2.astype("float32"),             compression="gzip")


def _resample_rollout(arr: np.ndarray, target_T: int) -> np.ndarray:
    """Resample axis 1 of arr from its current length to target_T via linear interp.

    Args:
        arr: array of shape (N, T_src, D...). T_src is the rollout step count
             (typically 100 in the cached rollout) and may differ from target_T.
        target_T: desired length on axis 1.

    Returns:
        Array of shape (N, target_T, D...) resampled along axis 1.
    """
    N = arr.shape[0]
    T_src = arr.shape[1]
    if T_src == target_T:
        return arr.astype(np.float32, copy=False)
    out_shape = (N, target_T) + arr.shape[2:]
    out = np.empty(out_shape, dtype=np.float32)
    src_idx = np.linspace(0, 1, T_src)
    dst_idx = np.linspace(0, 1, target_T)
    flat = arr.reshape(N, T_src, -1)
    out_flat = out.reshape(N, target_T, -1)
    for i in range(N):
        for d in range(flat.shape[-1]):
            out_flat[i, :, d] = np.interp(dst_idx, src_idx, flat[i, :, d])
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--features-npz", type=Path, required=True)
    p.add_argument("--rollout-npz", type=Path, required=True)
    p.add_argument("--out-h5", type=Path, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--checkpoint-step", type=int, required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_paired_h5(
        features_npz=args.features_npz,
        rollout_npz=args.rollout_npz,
        out_h5=args.out_h5,
        checkpoint=args.checkpoint,
        checkpoint_step=args.checkpoint_step,
    )
    print(f"wrote {args.out_h5}")
```

- [ ] **Step 2: Create scripts/__init__.py if it doesn't exist (for test import path)**

```bash
cd /root/vast/eric/vnl-playground
test -f scripts/__init__.py || touch scripts/__init__.py
ls scripts/__init__.py
```

Expected: file exists (empty).

- [ ] **Step 3: Add a conftest.py so pytest finds the scripts module**

`tests/conftest.py` (only if it doesn't exist):

```python
"""Add the repo root to sys.path so `from scripts.X import Y` works in tests."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
```

```bash
cd /root/vast/eric/vnl-playground
test -f tests/conftest.py && cat tests/conftest.py | head -5
# If it exists and doesn't already add the repo root, append the snippet above.
# If it doesn't exist, write it from the code block.
```

- [ ] **Step 4: Run the converter test, confirm green**

```bash
cd /root/vast/eric/vnl-playground
/root/vast/eric/vnl-playground/.venv/bin/python -m pytest tests/test_build_paired_deterministic_h5.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /root/vast/eric/vnl-playground
git add scripts/build_paired_deterministic_h5.py scripts/__init__.py tests/conftest.py 2>/dev/null
git -c user.name="Eric Leonardis" -c user.email="leonardiseric@gmail.com" commit -m "$(cat <<'EOF'
feat: paired_deterministic.h5 converter

Reads features.npz + 278-clip rollout cache, writes the HDF5 source for
the kin-clustered bio-vs-sim analysis. Preserves all 12 actuators, raw
action, intention latent, and decoder layers per the spec.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Run the converter on the real data and inspect

**Files:**
- Create: `notebooks/kin_emg_bundle/paired_deterministic.h5` (output, gitignored)

- [ ] **Step 1: Run the converter**

```bash
cd /root/vast/eric/vnl-playground
mkdir -p notebooks/kin_emg_bundle
/root/vast/eric/vnl-playground/.venv/bin/python -m scripts.build_paired_deterministic_h5 \
  --features-npz notebooks/kinematics_emg_comparison/cache/features.npz \
  --rollout-npz  notebooks/talk_figures/figs/rollout_activations/s18-ms-F4-fs1p2-20260502-014751_278clips.npz \
  --out-h5       notebooks/kin_emg_bundle/paired_deterministic.h5 \
  --checkpoint   "s18-ms-F4-fs1p2-20260502-014751" \
  --checkpoint-step 0
```

Expected: `wrote notebooks/kin_emg_bundle/paired_deterministic.h5`. File size approximately 80-120 MB.

- [ ] **Step 2: Inspect the file**

```bash
/root/vast/eric/vnl-playground/.venv/bin/python -c "
import h5py
with h5py.File('notebooks/kin_emg_bundle/paired_deterministic.h5', 'r') as f:
    def show(name, obj):
        if hasattr(obj, 'shape'):
            print(f'{name:40s} {str(obj.shape):20s} {obj.dtype}')
    f.visititems(show)
    print('---')
    for k, v in f['meta'].attrs.items():
        print(f'meta.attrs[{k!r}] = {v}')
"
```

Expected output sample: `meta/animal (204,) |S5`, `bio/qpos (204, 60, 7) float32`, `sim/muscle_act (204, 60, 12) float32`, etc. All 14 datasets present.

- [ ] **Step 3: Confirm the file is gitignored**

```bash
cd /root/vast/eric/vnl-playground
echo "notebooks/kin_emg_bundle/" >> .gitignore
grep "kin_emg_bundle" .gitignore
```

Expected: line present.

- [ ] **Step 4: Commit gitignore update**

```bash
cd /root/vast/eric/vnl-playground
git add .gitignore
git -c user.name="Eric Leonardis" -c user.email="leonardiseric@gmail.com" commit -m "$(cat <<'EOF'
gitignore: ignore notebooks/kin_emg_bundle output dir

Local-only output dir for paired_deterministic.h5 and analysis_results.h5.
Bundle handoff is by tarball, not git.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Write failing tests for clustering function

**Files:**
- Create: `tests/test_kin_emg_analysis.py`

- [ ] **Step 1: Create the test file with clustering tests**

`tests/test_kin_emg_analysis.py`:

```python
"""Tests for scripts/kin_emg_analysis.py.

Synthetic data with known structure so each function's behavior is verifiable
without depending on real animal data.
"""
import numpy as np
import pytest


@pytest.fixture
def three_cluster_qpos() -> np.ndarray:
    """Build (n_trials, T, J) qpos with three obvious Gaussian clusters.

    Each cluster has 10 trials around a different mean trajectory.
    T=60, J=7. Centroids well-separated so silhouette should pick k=3 cleanly.
    """
    rng = np.random.default_rng(0)
    T, J = 60, 7
    centroids = np.array([
        np.linspace(0, 1, T)[:, None] * np.array([1, 0, 0, 0, 0, 0, 0]),
        np.linspace(0, 1, T)[:, None] * np.array([0, 1, 0, 0, 0, 0, 0]),
        np.linspace(0, 1, T)[:, None] * np.array([0, 0, 1, 0, 0, 0, 0]),
    ])
    trials = []
    for c in centroids:
        for _ in range(10):
            trials.append(c + rng.normal(0, 0.05, size=c.shape))
    return np.stack(trials).astype("float32"), np.repeat([0, 1, 2], 10)


def test_cluster_animal_recovers_three_clusters(three_cluster_qpos):
    """Silhouette should pick k=3 on well-separated synthetic data."""
    from scripts.kin_emg_analysis import cluster_animal

    qpos, true_labels = three_cluster_qpos
    result = cluster_animal(qpos, k_grid=(2, 3, 4, 5, 6), random_state=0)
    assert result.chosen_k == 3

    # KMeans should produce 3 distinct labels that partition similarly to truth
    # (up to label permutation). Check Adjusted Rand Index would be cleaner,
    # but sklearn's ARI must be near 1 for well-separated clusters.
    from sklearn.metrics import adjusted_rand_score
    assert adjusted_rand_score(true_labels, result.labels_kmeans) > 0.9


def test_cluster_animal_returns_centroids_with_right_shape(three_cluster_qpos):
    from scripts.kin_emg_analysis import cluster_animal

    qpos, _ = three_cluster_qpos
    result = cluster_animal(qpos, k_grid=(2, 3, 4, 5, 6), random_state=0)
    k = result.chosen_k
    T, J = qpos.shape[1], qpos.shape[2]
    assert result.centroids_kmeans.shape == (k, T, J)


def test_cluster_animal_silhouette_curve_length(three_cluster_qpos):
    from scripts.kin_emg_analysis import cluster_animal

    qpos, _ = three_cluster_qpos
    result = cluster_animal(qpos, k_grid=(2, 3, 4, 5, 6), random_state=0)
    assert result.silhouette.shape == (5,)
    assert result.wcss.shape == (5,)
```

- [ ] **Step 2: Run the test, confirm it fails (import error)**

```bash
cd /root/vast/eric/vnl-playground
/root/vast/eric/vnl-playground/.venv/bin/python -m pytest tests/test_kin_emg_analysis.py -v
```

Expected: tests fail with `ModuleNotFoundError: No module named 'scripts.kin_emg_analysis'`.

- [ ] **Step 3: Commit**

```bash
cd /root/vast/eric/vnl-playground
git add tests/test_kin_emg_analysis.py
git -c user.name="Eric Leonardis" -c user.email="leonardiseric@gmail.com" commit -m "$(cat <<'EOF'
test: clustering function red bar for kin_emg_analysis

TDD red bar: three-cluster synthetic qpos, expects silhouette picks k=3
and ARI > 0.9 vs ground truth.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Implement `cluster_animal`

**Files:**
- Create: `scripts/kin_emg_analysis.py`

- [ ] **Step 1: Write the module with `cluster_animal`**

`scripts/kin_emg_analysis.py`:

```python
"""Pure analysis functions for the kinematics-clustered bio-vs-sim pipeline.

Functions in this module are designed to be standalone: each takes numpy arrays
and returns numpy arrays or simple dataclasses. They are tested in
tests/test_kin_emg_analysis.py against synthetic data and then pasted into the
Colab notebook bio_vs_sim_kin_clustered.ipynb so the notebook has no project-
internal imports.

Spec: docs/superpowers/specs/2026-05-21-kinematics-clustered-bio-vs-sim-emg-design.md
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


@dataclass
class ClusterResult:
    """Output of cluster_animal.

    Attributes:
        labels_kmeans: (n_trials,) int32 cluster id per trial from KMeans.
        labels_hier: (n_trials,) int32 cluster id per trial from hierarchical.
        centroids_kmeans: (k, T, J) float32 mean qpos trajectory per cluster.
        wcss: (len(k_grid),) float32 within-cluster sum of squares per k.
        silhouette: (len(k_grid),) float32 silhouette score per k.
        linkage_hier: (n_trials - 1, 4) float32 scipy linkage matrix.
        k_grid: tuple of int k values tested.
        chosen_k: int k selected by the selection rule.
        selection_rule: str describing how chosen_k was picked.
    """
    labels_kmeans: np.ndarray
    labels_hier: np.ndarray
    centroids_kmeans: np.ndarray
    wcss: np.ndarray
    silhouette: np.ndarray
    linkage_hier: np.ndarray
    k_grid: tuple
    chosen_k: int
    selection_rule: str


def cluster_animal(
    qpos: np.ndarray,
    k_grid: tuple = (2, 3, 4, 5, 6),
    random_state: int = 0,
    silhouette_floor: float = 0.1,
    near_tie_margin: float = 0.02,
) -> ClusterResult:
    """Cluster per-animal qpos trajectories with KMeans and hierarchical.

    Args:
        qpos: (n_trials, T, J) per-trial qpos trajectory for one animal.
        k_grid: tuple of k values to evaluate for k selection.
        random_state: seed for KMeans.
        silhouette_floor: if best silhouette < this, fall back to k=3.
        near_tie_margin: if k=2 silhouette is within this of the best, fall back to k=3.

    Returns:
        ClusterResult populated with labels for both methods at chosen_k,
        centroids for KMeans, WCSS / silhouette curves over k_grid, and the
        hierarchical linkage matrix.
    """
    n, T, J = qpos.shape
    X = qpos.reshape(n, T * J).astype("float64")

    wcss = np.empty(len(k_grid), dtype="float32")
    silhouettes = np.empty(len(k_grid), dtype="float32")
    kmeans_per_k = {}

    for i, k in enumerate(k_grid):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        wcss[i] = km.inertia_
        # silhouette undefined for k > n - 1; guard.
        if k < n:
            silhouettes[i] = silhouette_score(X, labels)
        else:
            silhouettes[i] = np.nan
        kmeans_per_k[k] = (km, labels)

    chosen_k, rule = _select_k(
        k_grid, silhouettes,
        silhouette_floor=silhouette_floor,
        near_tie_margin=near_tie_margin,
    )

    km, labels_kmeans = kmeans_per_k[chosen_k]
    centroids = km.cluster_centers_.reshape(chosen_k, T, J).astype("float32")

    Z = linkage(X, method="ward").astype("float32")  # (n-1, 4)
    # Cut the hierarchical tree at chosen_k to get cluster labels.
    from scipy.cluster.hierarchy import fcluster
    labels_hier = fcluster(Z, t=chosen_k, criterion="maxclust").astype("int32") - 1

    return ClusterResult(
        labels_kmeans=labels_kmeans.astype("int32"),
        labels_hier=labels_hier,
        centroids_kmeans=centroids,
        wcss=wcss,
        silhouette=silhouettes,
        linkage_hier=Z,
        k_grid=tuple(k_grid),
        chosen_k=chosen_k,
        selection_rule=rule,
    )


def _select_k(
    k_grid: tuple,
    silhouettes: np.ndarray,
    silhouette_floor: float,
    near_tie_margin: float,
) -> tuple[int, str]:
    """Pick k via argmax silhouette with documented fallbacks.

    Args:
        k_grid: tuple of k values.
        silhouettes: (len(k_grid),) silhouette score per k.
        silhouette_floor: if max silhouette below this, fall back to k=3.
        near_tie_margin: if k=2 within this margin of the max, fall back to k=3.

    Returns:
        (chosen_k, selection_rule_string).
    """
    valid = ~np.isnan(silhouettes)
    if not valid.any():
        return 3, "fallback k=3 (all silhouettes NaN)"
    best_i = int(np.nanargmax(silhouettes))
    best_k = k_grid[best_i]
    best_score = silhouettes[best_i]

    if best_score < silhouette_floor:
        return 3, f"fallback k=3 (best silhouette {best_score:.3f} < floor {silhouette_floor})"

    if k_grid[0] == 2 and best_k == 2:
        # If k=2 won but is essentially tied with a higher k, prefer the higher k.
        rest_max = np.nanmax(silhouettes[1:]) if len(silhouettes) > 1 else -np.inf
        if best_score - rest_max < near_tie_margin:
            for i, k in enumerate(k_grid[1:], start=1):
                if silhouettes[i] >= rest_max:
                    return k, f"argmax silhouette (k=2 within {near_tie_margin} of k={k})"

    return best_k, "argmax silhouette"
```

- [ ] **Step 2: Run the clustering tests, confirm green**

```bash
cd /root/vast/eric/vnl-playground
/root/vast/eric/vnl-playground/.venv/bin/python -m pytest tests/test_kin_emg_analysis.py -v
```

Expected: 3 tests pass.

- [ ] **Step 3: Commit**

```bash
cd /root/vast/eric/vnl-playground
git add scripts/kin_emg_analysis.py
git -c user.name="Eric Leonardis" -c user.email="leonardiseric@gmail.com" commit -m "$(cat <<'EOF'
feat: cluster_animal for per-animal qpos clustering

KMeans + Ward hierarchical with silhouette-based k selection over
k=2..6, fallback to k=3 when silhouette < 0.1 or k=2 is a near-tie.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Add within-cluster self-similarity test, then implement

**Files:**
- Modify: `tests/test_kin_emg_analysis.py` (append)
- Modify: `scripts/kin_emg_analysis.py` (append)

- [ ] **Step 1: Append the self-similarity test**

Append to `tests/test_kin_emg_analysis.py`:

```python
def test_within_cluster_similarity_identical_trials_score_one():
    """Pairwise mean per-muscle Pearson of identical envelopes should be 1.0."""
    from scripts.kin_emg_analysis import within_cluster_similarity

    rng = np.random.default_rng(0)
    T, M = 60, 2
    envelope = rng.random((T, M)).astype("float32")
    trials = np.stack([envelope] * 5)  # 5 identical trials
    labels = np.zeros(5, dtype="int32")

    result = within_cluster_similarity(trials, labels)
    # Single cluster, n_pairs = 10, all pairs identical -> similarity = 1.
    assert result["cluster_summary"].shape == (1, 3)  # (k, mean/std/n_pairs)
    assert np.isclose(result["cluster_summary"][0, 0], 1.0, atol=1e-5)
    assert result["cluster_summary"][0, 2] == 10  # C(5, 2) = 10
    assert result["pairwise"].shape == (5, 5)
    # Diagonal NaN.
    assert np.all(np.isnan(np.diag(result["pairwise"])))


def test_within_cluster_similarity_two_clusters():
    """Two clusters of constant-but-different envelopes -> per-cluster mean = 1, no cross-cluster pairs."""
    from scripts.kin_emg_analysis import within_cluster_similarity

    T, M = 60, 2
    # Make non-constant but identical within each cluster so Pearson is well-defined.
    rng = np.random.default_rng(0)
    env_a = rng.random((T, M)).astype("float32")
    env_b = rng.random((T, M)).astype("float32")
    trials = np.stack([env_a, env_a, env_a, env_b, env_b, env_b])
    labels = np.array([0, 0, 0, 1, 1, 1], dtype="int32")

    result = within_cluster_similarity(trials, labels)
    assert result["cluster_summary"].shape == (2, 3)
    assert np.allclose(result["cluster_summary"][:, 0], 1.0, atol=1e-5)
    assert (result["cluster_summary"][:, 2] == 3).all()  # C(3, 2) = 3 per cluster
```

- [ ] **Step 2: Run the new tests, confirm they fail with AttributeError**

```bash
cd /root/vast/eric/vnl-playground
/root/vast/eric/vnl-playground/.venv/bin/python -m pytest tests/test_kin_emg_analysis.py -v -k "within_cluster"
```

Expected: 2 tests fail with `ImportError` for `within_cluster_similarity`.

- [ ] **Step 3: Implement `within_cluster_similarity` in the module**

Append to `scripts/kin_emg_analysis.py`:

```python
def within_cluster_similarity(
    envelopes: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Pairwise mean per-muscle Pearson r within each cluster.

    Args:
        envelopes: (n_trials, T, M) envelope per trial.
            For bio EMG: M=2 (AD, Biceps).
            For sim muscle_act: M=2 (the AD+Biceps subset).
        labels: (n_trials,) cluster id per trial.

    Returns:
        Dict with:
          'pairwise'         : (n_trials, n_trials) float32, symmetric,
                               NaN on diagonal and on cross-cluster pairs.
          'cluster_summary'  : (k, 3) float32, columns are mean / std / n_pairs.
          'cluster_means'    : (k, T, M) float32, mean envelope per cluster.
    """
    n, T, M = envelopes.shape
    k = int(labels.max()) + 1

    pairwise = np.full((n, n), np.nan, dtype="float32")
    summary = np.full((k, 3), np.nan, dtype="float32")
    means = np.zeros((k, T, M), dtype="float32")

    for c in range(k):
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            summary[c] = (np.nan, np.nan, 0)
            if len(idx) == 1:
                means[c] = envelopes[idx[0]]
            continue

        means[c] = envelopes[idx].mean(axis=0)
        pair_scores = []
        for ii in range(len(idx)):
            for jj in range(ii + 1, len(idx)):
                i, j = idx[ii], idx[jj]
                r = _mean_per_muscle_pearson(envelopes[i], envelopes[j])
                pairwise[i, j] = r
                pairwise[j, i] = r
                pair_scores.append(r)

        pair_scores = np.asarray(pair_scores, dtype="float32")
        summary[c] = (pair_scores.mean(), pair_scores.std(), len(pair_scores))

    return {
        "pairwise": pairwise,
        "cluster_summary": summary,
        "cluster_means": means,
    }


def _mean_per_muscle_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Average Pearson r across muscles between two (T, M) envelopes.

    Args:
        a: (T, M) envelope.
        b: (T, M) envelope.

    Returns:
        Scalar: mean of Pearson r across the M channels.
    """
    T, M = a.shape
    rs = np.empty(M, dtype="float32")
    for m in range(M):
        am = a[:, m] - a[:, m].mean()
        bm = b[:, m] - b[:, m].mean()
        denom = np.sqrt((am * am).sum() * (bm * bm).sum())
        rs[m] = (am * bm).sum() / denom if denom > 0 else np.nan
    return float(np.nanmean(rs))
```

- [ ] **Step 4: Run all kin_emg_analysis tests, confirm green**

```bash
cd /root/vast/eric/vnl-playground
/root/vast/eric/vnl-playground/.venv/bin/python -m pytest tests/test_kin_emg_analysis.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /root/vast/eric/vnl-playground
git add scripts/kin_emg_analysis.py tests/test_kin_emg_analysis.py
git -c user.name="Eric Leonardis" -c user.email="leonardiseric@gmail.com" commit -m "$(cat <<'EOF'
feat: within_cluster_similarity (mean per-muscle Pearson r)

Pairwise self-similarity within a cluster, averaging Pearson r across
the AD and Biceps channels for each trial pair. Returns the full
pairwise matrix plus per-cluster mean / std / n_pairs summary.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: GLM bio-to-sim, per cluster and pooled

**Files:**
- Modify: `tests/test_kin_emg_analysis.py` (append)
- Modify: `scripts/kin_emg_analysis.py` (append)

- [ ] **Step 1: Append GLM tests**

Append to `tests/test_kin_emg_analysis.py`:

```python
def test_fit_glm_per_cluster_recovers_slope_intercept():
    """Synthetic sim = 0.5 * bio + 0.2: per-cluster fit should recover both."""
    from scripts.kin_emg_analysis import fit_glm_per_cluster

    rng = np.random.default_rng(0)
    n, T, M = 30, 60, 2
    bio = rng.random((n, T, M)).astype("float32")
    true_slope = np.array([0.5, 0.7], dtype="float32")
    true_intercept = np.array([0.2, -0.1], dtype="float32")
    sim = bio * true_slope + true_intercept + rng.normal(0, 0.01, size=bio.shape).astype("float32")
    labels = np.repeat([0, 1, 2], 10)

    result = fit_glm_per_cluster(bio, sim, labels)
    assert result["slope"].shape == (3, 2)
    assert result["intercept"].shape == (3, 2)
    assert np.allclose(result["slope"], true_slope, atol=0.02)
    assert np.allclose(result["intercept"], true_intercept, atol=0.02)


def test_fit_glm_pooled_recovers_when_homogeneous():
    """All clusters share the same map -> pooled fit also recovers it."""
    from scripts.kin_emg_analysis import fit_glm_pooled

    rng = np.random.default_rng(0)
    n, T, M = 30, 60, 2
    bio = rng.random((n, T, M)).astype("float32")
    true_slope = np.array([0.5, 0.7], dtype="float32")
    true_intercept = np.array([0.2, -0.1], dtype="float32")
    sim = bio * true_slope + true_intercept + rng.normal(0, 0.01, size=bio.shape).astype("float32")

    result = fit_glm_pooled(bio, sim)
    assert result["slope"].shape == (2,)
    assert np.allclose(result["slope"], true_slope, atol=0.02)
    assert np.allclose(result["intercept"], true_intercept, atol=0.02)
```

- [ ] **Step 2: Run, confirm failure**

```bash
cd /root/vast/eric/vnl-playground
/root/vast/eric/vnl-playground/.venv/bin/python -m pytest tests/test_kin_emg_analysis.py -v -k "fit_glm"
```

Expected: 2 failures, missing `fit_glm_per_cluster` / `fit_glm_pooled`.

- [ ] **Step 3: Implement the GLM fits**

Append to `scripts/kin_emg_analysis.py`:

```python
def fit_glm_per_cluster(
    bio_emg: np.ndarray,
    sim_muscle_act: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Fit sim_muscle_act = slope * bio_emg + intercept per (cluster, muscle).

    Args:
        bio_emg: (n_trials, T, M) bio EMG envelope.
        sim_muscle_act: (n_trials, T, M) sim muscle activation, same shape as bio.
        labels: (n_trials,) cluster id per trial.

    Returns:
        Dict with:
          'slope'     : (k, M) float32
          'intercept' : (k, M) float32
          'r2'        : (k, M) float32
          'n_samples' : (k,)   int32  (trials in cluster) * T
    """
    n, T, M = bio_emg.shape
    k = int(labels.max()) + 1
    slope = np.full((k, M), np.nan, dtype="float32")
    intercept = np.full((k, M), np.nan, dtype="float32")
    r2 = np.full((k, M), np.nan, dtype="float32")
    n_samples = np.zeros(k, dtype="int32")

    for c in range(k):
        idx = np.where(labels == c)[0]
        n_samples[c] = len(idx) * T
        if len(idx) < 1:
            continue
        for m in range(M):
            x = bio_emg[idx, :, m].ravel()
            y = sim_muscle_act[idx, :, m].ravel()
            a, b, r2_val = _fit_affine(x, y)
            slope[c, m] = a
            intercept[c, m] = b
            r2[c, m] = r2_val

    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "n_samples": n_samples,
    }


def fit_glm_pooled(
    bio_emg: np.ndarray,
    sim_muscle_act: np.ndarray,
) -> dict:
    """Fit one affine map per muscle, pooled across all trials and timesteps.

    Args:
        bio_emg: (n_trials, T, M).
        sim_muscle_act: (n_trials, T, M).

    Returns:
        Dict with:
          'slope'     : (M,) float32
          'intercept' : (M,) float32
          'r2'        : (M,) float32
          'n_samples' : int   total points used (n_trials * T).
    """
    n, T, M = bio_emg.shape
    slope = np.empty(M, dtype="float32")
    intercept = np.empty(M, dtype="float32")
    r2 = np.empty(M, dtype="float32")
    for m in range(M):
        x = bio_emg[:, :, m].ravel()
        y = sim_muscle_act[:, :, m].ravel()
        a, b, r2_val = _fit_affine(x, y)
        slope[m] = a
        intercept[m] = b
        r2[m] = r2_val
    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "n_samples": int(n * T),
    }


def _fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Closed-form least-squares fit of y = a*x + b.

    Args:
        x: (N,) predictor.
        y: (N,) response.

    Returns:
        (slope a, intercept b, in-sample R^2).
    """
    x = x.astype("float64")
    y = y.astype("float64")
    n = x.size
    sx, sy = x.sum(), y.sum()
    sxx = (x * x).sum()
    sxy = (x * y).sum()
    denom = n * sxx - sx * sx
    if denom <= 0:
        return float("nan"), float("nan"), float("nan")
    a = (n * sxy - sx * sy) / denom
    b = (sy - a * sx) / n
    yhat = a * x + b
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return float(a), float(b), float(r2)
```

- [ ] **Step 4: Run, confirm green**

```bash
cd /root/vast/eric/vnl-playground
/root/vast/eric/vnl-playground/.venv/bin/python -m pytest tests/test_kin_emg_analysis.py -v
```

Expected: 7 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /root/vast/eric/vnl-playground
git add scripts/kin_emg_analysis.py tests/test_kin_emg_analysis.py
git -c user.name="Eric Leonardis" -c user.email="leonardiseric@gmail.com" commit -m "$(cat <<'EOF'
feat: GLM bio-to-sim, per cluster and pooled

Closed-form affine fits sim_muscle_act = slope * bio_emg + intercept,
one per (cluster, muscle) and one per muscle pooled across clusters.
Returns slope / intercept / R^2 / n_samples.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Plotting helpers (no formal tests, smoke-render only)

**Files:**
- Modify: `scripts/kin_emg_analysis.py` (append)

- [ ] **Step 1: Append four plotting functions**

Append to `scripts/kin_emg_analysis.py`:

```python
def plot_cluster_trajectories(
    qpos: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    joint_names: tuple = None,
    animal: str = "",
):
    """Per-cluster qpos trajectory plot: members thin, centroid bold.

    Args:
        qpos: (n_trials, T, J) qpos.
        labels: (n_trials,) cluster id.
        centroids: (k, T, J) per-cluster mean trajectory.
        joint_names: optional tuple of J joint names for column titles.
        animal: animal id string, used in figure title.

    Returns:
        matplotlib.figure.Figure.
    """
    import matplotlib.pyplot as plt
    k = centroids.shape[0]
    J = qpos.shape[2]
    fig, axes = plt.subplots(k, J, figsize=(1.6 * J, 1.4 * k), sharey="col")
    if k == 1:
        axes = axes[None, :]
    for c in range(k):
        idx = np.where(labels == c)[0]
        for j in range(J):
            ax = axes[c, j]
            for i in idx:
                ax.plot(qpos[i, :, j], color="0.7", lw=0.5)
            ax.plot(centroids[c, :, j], color="C0", lw=1.5)
            if c == 0 and joint_names is not None:
                ax.set_title(joint_names[j], fontsize=8)
            if j == 0:
                ax.set_ylabel(f"cluster {c}\nn={len(idx)}", fontsize=8)
            ax.tick_params(labelsize=6)
    fig.suptitle(f"{animal}: qpos trajectories by cluster (k={k})", fontsize=10)
    fig.tight_layout()
    return fig


def plot_cluster_emg_means(
    cluster_means_bio: np.ndarray,
    cluster_means_sim: np.ndarray,
    muscle_names: tuple = ("AD", "Biceps"),
    animal: str = "",
):
    """Per-cluster mean envelope, bio vs sim, one column per muscle.

    Args:
        cluster_means_bio: (k, T, M) per-cluster mean bio EMG envelope.
        cluster_means_sim: (k, T, M) per-cluster mean sim muscle activation.
        muscle_names: M-tuple of channel names.
        animal: animal id for figure title.

    Returns:
        matplotlib.figure.Figure.
    """
    import matplotlib.pyplot as plt
    k, T, M = cluster_means_bio.shape
    fig, axes = plt.subplots(k, M, figsize=(2.5 * M, 1.4 * k), sharey="col", sharex=True)
    if k == 1:
        axes = axes[None, :]
    for c in range(k):
        for m in range(M):
            ax = axes[c, m]
            ax.plot(cluster_means_bio[c, :, m], color="C3", lw=1.3, label="bio EMG")
            ax.plot(cluster_means_sim[c, :, m], color="C0", lw=1.3, label="sim muscle_act")
            if c == 0:
                ax.set_title(muscle_names[m], fontsize=8)
            if c == 0 and m == M - 1:
                ax.legend(fontsize=6, loc="upper right")
            if m == 0:
                ax.set_ylabel(f"cluster {c}", fontsize=8)
            ax.tick_params(labelsize=6)
    fig.suptitle(f"{animal}: per-cluster mean envelope", fontsize=10)
    fig.tight_layout()
    return fig


def plot_bio_sim_glm_traces(
    cluster_means_bio: np.ndarray,
    cluster_means_sim: np.ndarray,
    glm_per_cluster: dict,
    muscle_names: tuple = ("AD", "Biceps"),
    animal: str = "",
):
    """Per cluster: bio, raw sim, regression-corrected sim = slope * bio + intercept.

    Args:
        cluster_means_bio: (k, T, M) per-cluster mean bio EMG.
        cluster_means_sim: (k, T, M) per-cluster mean sim muscle activation.
        glm_per_cluster: dict from fit_glm_per_cluster.
        muscle_names: M-tuple of channel names.
        animal: animal id for title.

    Returns:
        matplotlib.figure.Figure.
    """
    import matplotlib.pyplot as plt
    k, T, M = cluster_means_bio.shape
    fig, axes = plt.subplots(k, M, figsize=(2.7 * M, 1.5 * k), sharey="col", sharex=True)
    if k == 1:
        axes = axes[None, :]
    slope = glm_per_cluster["slope"]
    intercept = glm_per_cluster["intercept"]

    for c in range(k):
        for m in range(M):
            ax = axes[c, m]
            ax.plot(cluster_means_bio[c, :, m], color="C3", lw=1.3, label="bio")
            ax.plot(cluster_means_sim[c, :, m], color="C0", lw=1.0, alpha=0.7, label="sim raw")
            corrected = slope[c, m] * cluster_means_bio[c, :, m] + intercept[c, m]
            ax.plot(corrected, color="C0", lw=1.6, linestyle="--", label="sim from bio (GLM)")
            if c == 0:
                ax.set_title(muscle_names[m], fontsize=8)
            if c == 0 and m == M - 1:
                ax.legend(fontsize=6, loc="upper right")
            if m == 0:
                ax.set_ylabel(f"c{c}\na={slope[c, m]:.2f}\nb={intercept[c, m]:.2f}", fontsize=7)
            ax.tick_params(labelsize=6)
    fig.suptitle(f"{animal}: bio vs sim vs GLM-corrected", fontsize=10)
    fig.tight_layout()
    return fig


def plot_self_similarity_distributions(
    bio_summary_per_animal: dict,
    sim_summary_per_animal: dict,
):
    """Box / strip plot of per-cluster self-similarity, bio vs sim, all animals.

    Args:
        bio_summary_per_animal: dict[animal_id -> (k, 3) float32 from
            within_cluster_similarity['cluster_summary']]. Column 0 is mean.
        sim_summary_per_animal: same structure, for sim muscle_act.

    Returns:
        matplotlib.figure.Figure.
    """
    import matplotlib.pyplot as plt
    animals = sorted(bio_summary_per_animal.keys())
    fig, ax = plt.subplots(figsize=(1.5 + 0.8 * len(animals), 3))
    xs = np.arange(len(animals))
    bio_vals = [bio_summary_per_animal[a][:, 0] for a in animals]
    sim_vals = [sim_summary_per_animal[a][:, 0] for a in animals]

    for i, a in enumerate(animals):
        ax.scatter(np.full_like(bio_vals[i], xs[i] - 0.12), bio_vals[i],
                   color="C3", s=18, label="bio EMG" if i == 0 else None)
        ax.scatter(np.full_like(sim_vals[i], xs[i] + 0.12), sim_vals[i],
                   color="C0", s=18, label="sim muscle_act" if i == 0 else None)
    ax.set_xticks(xs)
    ax.set_xticklabels(animals, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("per-cluster mean pairwise Pearson r")
    ax.set_title("within-cluster self-similarity")
    ax.legend(fontsize=8)
    ax.axhline(0, color="0.7", lw=0.5)
    fig.tight_layout()
    return fig
```

- [ ] **Step 2: Smoke-render each plot on synthetic data**

```bash
cd /root/vast/eric/vnl-playground
/root/vast/eric/vnl-playground/.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
from scripts.kin_emg_analysis import (
    cluster_animal, within_cluster_similarity, fit_glm_per_cluster,
    plot_cluster_trajectories, plot_cluster_emg_means, plot_bio_sim_glm_traces,
    plot_self_similarity_distributions,
)
rng = np.random.default_rng(0)
qpos = rng.random((30, 60, 7)).astype('float32')
cr = cluster_animal(qpos, k_grid=(2, 3, 4))
print('cluster_animal OK, k=', cr.chosen_k)
bio = rng.random((30, 60, 2)).astype('float32')
sim = bio * 0.6 + 0.1 + rng.normal(0, 0.01, size=bio.shape).astype('float32')
ws = within_cluster_similarity(bio, cr.labels_kmeans)
print('within_cluster_similarity OK')
glm = fit_glm_per_cluster(bio, sim, cr.labels_kmeans)
print('fit_glm_per_cluster OK, slope shape=', glm['slope'].shape)

f1 = plot_cluster_trajectories(qpos, cr.labels_kmeans, cr.centroids_kmeans)
f2 = plot_cluster_emg_means(ws['cluster_means'], ws['cluster_means'])
f3 = plot_bio_sim_glm_traces(ws['cluster_means'], ws['cluster_means'], glm)
f4 = plot_self_similarity_distributions({'A': ws['cluster_summary']}, {'A': ws['cluster_summary']})
print('all four plots rendered')
"
```

Expected: prints "all four plots rendered" with no exceptions.

- [ ] **Step 3: Commit**

```bash
cd /root/vast/eric/vnl-playground
git add scripts/kin_emg_analysis.py
git -c user.name="Eric Leonardis" -c user.email="leonardiseric@gmail.com" commit -m "$(cat <<'EOF'
feat: plotting helpers (cluster traj, emg means, GLM traces, self-sim)

Four matplotlib plotting functions returning figures. Smoke-rendered
against synthetic data. No formal pytest yet; rendering is exercised in
the notebook smoke-run task.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Persistence - write analysis_results.h5

**Files:**
- Modify: `tests/test_kin_emg_analysis.py` (append)
- Modify: `scripts/kin_emg_analysis.py` (append)

- [ ] **Step 1: Append persistence test**

Append to `tests/test_kin_emg_analysis.py`:

```python
def test_write_analysis_results_round_trip(tmp_path):
    """Round-trip a small RESULTS dict through write_analysis_results."""
    from scripts.kin_emg_analysis import (
        cluster_animal, within_cluster_similarity, fit_glm_per_cluster,
        fit_glm_pooled, write_analysis_results,
    )
    rng = np.random.default_rng(0)
    qpos = rng.random((20, 60, 7)).astype("float32")
    bio = rng.random((20, 60, 2)).astype("float32")
    sim = bio * 0.5 + 0.2 + rng.normal(0, 0.01, size=bio.shape).astype("float32")

    cr = cluster_animal(qpos, k_grid=(2, 3, 4))
    ws = within_cluster_similarity(bio, cr.labels_kmeans)
    ws_sim = within_cluster_similarity(sim, cr.labels_kmeans)
    glm_pc = fit_glm_per_cluster(bio, sim, cr.labels_kmeans)
    glm_pool = fit_glm_pooled(bio, sim)

    animals = np.array([b"AT006"] * 20)
    trials = np.arange(20, dtype="int32")
    results = {
        "meta": {"animal": animals, "trial": trials,
                 "chosen_k_per_animal": {"AT006": cr.chosen_k}},
        "per_animal": {
            "AT006": {
                "cluster_result": cr,
                "self_sim_bio": ws,
                "self_sim_sim": ws_sim,
                "glm_per_cluster": glm_pc,
                "glm_pooled": glm_pool,
            }
        },
    }

    out = tmp_path / "analysis_results.h5"
    write_analysis_results(results, out, source_paired_path="paired.h5",
                            source_paired_sha256="abc", notebook_commit="def")

    import h5py
    with h5py.File(out, "r") as f:
        assert "meta" in f
        assert "clustering/AT006" in f
        assert "self_similarity/AT006" in f
        assert "glm_bio_to_sim/AT006/per_cluster" in f
        assert "glm_bio_to_sim/AT006/pooled" in f
        assert f["clustering/AT006/labels_kmeans"].shape == (20,)
        assert f["clustering/AT006"].attrs["chosen_k"] == cr.chosen_k
```

- [ ] **Step 2: Run, confirm fail**

```bash
cd /root/vast/eric/vnl-playground
/root/vast/eric/vnl-playground/.venv/bin/python -m pytest tests/test_kin_emg_analysis.py -v -k "write_analysis"
```

Expected: 1 failure, missing `write_analysis_results`.

- [ ] **Step 3: Implement `write_analysis_results`**

Append to `scripts/kin_emg_analysis.py`:

```python
def write_analysis_results(
    results: dict,
    out_h5,
    *,
    source_paired_path: str,
    source_paired_sha256: str,
    notebook_commit: str,
) -> None:
    """Persist the in-memory RESULTS dict to analysis_results.h5 per the spec.

    Args:
        results: Dict of the form:
            {
              "meta": {"animal": (N,) bytes, "trial": (N,) int32,
                       "chosen_k_per_animal": {animal: k}},
              "per_animal": {animal_id: {
                  "cluster_result": ClusterResult,
                  "self_sim_bio": dict from within_cluster_similarity,
                  "self_sim_sim": dict from within_cluster_similarity,
                  "glm_per_cluster": dict from fit_glm_per_cluster,
                  "glm_pooled": dict from fit_glm_pooled,
              }}
            }
        out_h5: Output path for analysis_results.h5.
        source_paired_path: String path to the source paired_deterministic.h5.
        source_paired_sha256: SHA-256 hex digest of the source file.
        notebook_commit: Git SHA of the notebook commit.

    Returns:
        None. Writes the file.
    """
    import datetime as dt
    import h5py
    import json
    from pathlib import Path

    out_h5 = Path(out_h5)
    out_h5.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_h5, "w") as f:
        meta = f.create_group("meta")
        meta.create_dataset("animal", data=results["meta"]["animal"])
        meta.create_dataset("trial",  data=results["meta"]["trial"])
        meta.attrs["source_paired"] = source_paired_path
        meta.attrs["source_paired_sha256"] = source_paired_sha256
        meta.attrs["notebook_commit"] = notebook_commit
        meta.attrs["created_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        meta.attrs["kin_feature_for_clustering"] = "qpos_flat (T*J=420)"
        meta.attrs["distance_for_clustering"] = "euclidean on qpos_flat"
        meta.attrs["self_similarity_for_emg"] = "mean per-muscle Pearson r"
        meta.attrs["chosen_k_per_animal"] = json.dumps(
            results["meta"]["chosen_k_per_animal"]
        )

        for animal, payload in results["per_animal"].items():
            cr = payload["cluster_result"]
            # /clustering/<animal>
            g = f.create_group(f"clustering/{animal}")
            g.create_dataset("labels_kmeans", data=cr.labels_kmeans)
            g.create_dataset("labels_hier",   data=cr.labels_hier)
            g.create_dataset("centroids_kmeans", data=cr.centroids_kmeans)
            g.create_dataset("wcss",       data=cr.wcss)
            g.create_dataset("silhouette", data=cr.silhouette)
            g.create_dataset("linkage_hier", data=cr.linkage_hier)
            g.attrs["k_grid"] = list(cr.k_grid)
            g.attrs["chosen_k"] = cr.chosen_k
            g.attrs["selection_rule"] = cr.selection_rule

            # /self_similarity/<animal>
            ss_g = f.create_group(f"self_similarity/{animal}")
            ws_bio = payload["self_sim_bio"]
            ws_sim = payload["self_sim_sim"]
            ss_g.create_dataset("bio_emg_pairwise_mean_pearson",
                                data=ws_bio["pairwise"])
            ss_g.create_dataset("sim_muscle_act_pairwise_mean_pearson",
                                data=ws_sim["pairwise"])
            ss_g.create_dataset("mean_bio_emg",       data=ws_bio["cluster_means"])
            ss_g.create_dataset("mean_sim_muscle_act", data=ws_sim["cluster_means"])
            ss_g.create_dataset("within_bio_summary",  data=ws_bio["cluster_summary"])
            ss_g.create_dataset("within_sim_summary",  data=ws_sim["cluster_summary"])

            # /glm_bio_to_sim/<animal>/per_cluster
            pc_g = f.create_group(f"glm_bio_to_sim/{animal}/per_cluster")
            glm_pc = payload["glm_per_cluster"]
            pc_g.create_dataset("slope",     data=glm_pc["slope"])
            pc_g.create_dataset("intercept", data=glm_pc["intercept"])
            pc_g.create_dataset("r2",        data=glm_pc["r2"])
            pc_g.create_dataset("n_samples", data=glm_pc["n_samples"])
            pc_g.attrs["direction"] = "sim_muscle_act = slope * bio_emg + intercept"

            # /glm_bio_to_sim/<animal>/pooled
            pl_g = f.create_group(f"glm_bio_to_sim/{animal}/pooled")
            glm_pl = payload["glm_pooled"]
            pl_g.create_dataset("slope",     data=glm_pl["slope"])
            pl_g.create_dataset("intercept", data=glm_pl["intercept"])
            pl_g.create_dataset("r2",        data=glm_pl["r2"])
            pl_g.create_dataset("n_samples", data=np.int32(glm_pl["n_samples"]))
            pl_g.attrs["direction"] = "sim_muscle_act = slope * bio_emg + intercept"
```

- [ ] **Step 4: Run, confirm green**

```bash
cd /root/vast/eric/vnl-playground
/root/vast/eric/vnl-playground/.venv/bin/python -m pytest tests/test_kin_emg_analysis.py -v
```

Expected: 8 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /root/vast/eric/vnl-playground
git add scripts/kin_emg_analysis.py tests/test_kin_emg_analysis.py
git -c user.name="Eric Leonardis" -c user.email="leonardiseric@gmail.com" commit -m "$(cat <<'EOF'
feat: write_analysis_results for analysis_results.h5

Persists per-animal clustering, self-similarity, and GLM results in the
per-animal-group HDF5 layout from the spec. Round-trip test included.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Assemble the Colab notebook

**Files:**
- Create: `notebooks/bio_vs_sim_kin_clustered.ipynb`

- [ ] **Step 1: Generate the notebook from a Python script**

Run this Python script to materialize the .ipynb. It writes one cell per analysis stage, pasting the functions from `scripts/kin_emg_analysis.py` so the notebook has no project-internal imports (Colab portability).

```bash
cd /root/vast/eric/vnl-playground
/root/vast/eric/vnl-playground/.venv/bin/python - <<'PY'
import json
from pathlib import Path

REPO = Path('.').resolve()
src = (REPO / 'scripts/kin_emg_analysis.py').read_text()

def code(s):  return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": s.splitlines(keepends=True)}
def md(s):    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}

cells = [
    md("# Bio vs Sim Muscle Activation, Clustered by Within-Animal Kinematics\n\nSpec: `docs/superpowers/specs/2026-05-21-kinematics-clustered-bio-vs-sim-emg-design.md`\n\nThis notebook is standalone for Google Colab. It reads `paired_deterministic.h5` and writes `analysis_results.h5`. No project-internal imports.\n"),

    md("## 1. Setup\n\nInstall dependencies and import. In Colab, replace the path with a Drive-mounted location."),
    code("# In Colab: !pip install -q h5py scikit-learn scipy tqdm matplotlib\n"
         "import json, datetime as dt, hashlib\n"
         "from dataclasses import dataclass\n"
         "from pathlib import Path\n"
         "import h5py, numpy as np\n"
         "import matplotlib.pyplot as plt\n"
         "from scipy.cluster.hierarchy import linkage, fcluster\n"
         "from sklearn.cluster import KMeans\n"
         "from sklearn.metrics import silhouette_score\n"
         "from tqdm.auto import tqdm\n"
         "\n"
         "PAIRED_H5 = Path('paired_deterministic.h5')  # adjust for Colab\n"
         "OUT_H5    = Path('analysis_results.h5')\n"),

    md("## 2. Analysis functions (pasted from `scripts/kin_emg_analysis.py`)\n\nKeeping these in one cell means the notebook stays self-contained for Colab."),
    code(src),

    md("## 3. Load and group per animal"),
    code("def load_per_animal(h5_path):\n"
         "    \"\"\"Slice paired_deterministic.h5 into per-animal numpy dicts.\n"
         "\n"
         "    Returns:\n"
         "        Dict[animal_str -> {'qpos_bio', 'emg_bio', 'qpos_sim',\n"
         "                            'muscle_act_AD_biceps', 'trial_idx'}].\n"
         "    \"\"\"\n"
         "    out = {}\n"
         "    with h5py.File(h5_path, 'r') as f:\n"
         "        animals = f['meta/animal'][:].astype(str)\n"
         "        bio_qpos = f['bio/qpos'][:]\n"
         "        bio_emg  = f['bio/emg'][:]\n"
         "        sim_qpos = f['sim/qpos'][:]\n"
         "        sim_mAB  = f['sim/muscle_act_AD_biceps'][:]\n"
         "        trials   = f['meta/trial'][:]\n"
         "    for a in np.unique(animals):\n"
         "        mask = animals == a\n"
         "        out[a] = dict(\n"
         "            qpos_bio=bio_qpos[mask],\n"
         "            emg_bio=bio_emg[mask],\n"
         "            qpos_sim=sim_qpos[mask],\n"
         "            muscle_act_AD_biceps=sim_mAB[mask],\n"
         "            trial_idx=np.where(mask)[0],\n"
         "        )\n"
         "    return out\n"
         "\n"
         "per_animal = load_per_animal(PAIRED_H5)\n"
         "for a, d in per_animal.items():\n"
         "    print(f'{a}: {d[\"qpos_bio\"].shape[0]} trials')\n"),

    md("## 4. Step 1: Per-animal clustering"),
    code("cluster_results = {}\n"
         "for a, d in tqdm(per_animal.items(), desc='cluster'):\n"
         "    cluster_results[a] = cluster_animal(d['qpos_bio'])\n"
         "    print(a, 'chosen_k =', cluster_results[a].chosen_k,\n"
         "          'sil =', cluster_results[a].silhouette[cluster_results[a].k_grid.index(cluster_results[a].chosen_k)])\n"),

    md("## 5. Cluster trajectory plot"),
    code("figs = {}\n"
         "for a, d in per_animal.items():\n"
         "    cr = cluster_results[a]\n"
         "    figs[a] = plot_cluster_trajectories(d['qpos_bio'], cr.labels_kmeans, cr.centroids_kmeans, animal=a)\n"
         "plt.show()\n"),

    md("## 6. Step 2: Within-cluster self-similarity"),
    code("self_sim_bio = {}\n"
         "self_sim_sim = {}\n"
         "for a, d in tqdm(per_animal.items(), desc='self-sim'):\n"
         "    labels = cluster_results[a].labels_kmeans\n"
         "    self_sim_bio[a] = within_cluster_similarity(d['emg_bio'], labels)\n"
         "    self_sim_sim[a] = within_cluster_similarity(d['muscle_act_AD_biceps'], labels)\n"
         "    print(a, 'bio summary:', self_sim_bio[a]['cluster_summary'][:, 0])\n"
         "    print(a, 'sim summary:', self_sim_sim[a]['cluster_summary'][:, 0])\n"),

    md("## 7. Self-similarity distribution plot, bio vs sim"),
    code("fig_ss = plot_self_similarity_distributions(\n"
         "    {a: self_sim_bio[a]['cluster_summary'] for a in per_animal},\n"
         "    {a: self_sim_sim[a]['cluster_summary'] for a in per_animal},\n"
         ")\n"
         "plt.show()\n"),

    md("## 8. Step 3: GLM bio EMG -> sim muscle activation"),
    code("glm_pc = {}\n"
         "glm_pool = {}\n"
         "for a, d in tqdm(per_animal.items(), desc='glm'):\n"
         "    labels = cluster_results[a].labels_kmeans\n"
         "    glm_pc[a]   = fit_glm_per_cluster(d['emg_bio'], d['muscle_act_AD_biceps'], labels)\n"
         "    glm_pool[a] = fit_glm_pooled(d['emg_bio'], d['muscle_act_AD_biceps'])\n"
         "    print(a, 'pooled slope:', glm_pool[a]['slope'], 'intercept:', glm_pool[a]['intercept'])\n"),

    md("## 9. Bio vs Sim vs GLM-corrected traces"),
    code("for a in per_animal:\n"
         "    plot_bio_sim_glm_traces(\n"
         "        self_sim_bio[a]['cluster_means'],\n"
         "        self_sim_sim[a]['cluster_means'],\n"
         "        glm_pc[a], animal=a,\n"
         "    )\n"
         "plt.show()\n"),

    md("## 10. Step 4: Paired delta over time"),
    code("def plot_paired_delta(d, labels, animal=''):\n"
         "    \"\"\"For each cluster and muscle, plot mean(sim - bio) and +/- std band.\"\"\"\n"
         "    bio = d['emg_bio']\n"
         "    sim = d['muscle_act_AD_biceps']\n"
         "    k = int(labels.max()) + 1\n"
         "    M = bio.shape[2]\n"
         "    fig, axes = plt.subplots(k, M, figsize=(2.5 * M, 1.3 * k), sharey=True, sharex=True)\n"
         "    if k == 1:\n"
         "        axes = axes[None, :]\n"
         "    for c in range(k):\n"
         "        idx = np.where(labels == c)[0]\n"
         "        if len(idx) == 0:\n"
         "            continue\n"
         "        delta = sim[idx] - bio[idx]\n"
         "        mu = delta.mean(axis=0)\n"
         "        sd = delta.std(axis=0)\n"
         "        for m in range(M):\n"
         "            ax = axes[c, m]\n"
         "            ax.plot(mu[:, m], color='C0')\n"
         "            ax.fill_between(np.arange(mu.shape[0]), mu[:, m] - sd[:, m], mu[:, m] + sd[:, m], color='C0', alpha=0.2)\n"
         "            ax.axhline(0, color='0.5', lw=0.5)\n"
         "            ax.tick_params(labelsize=6)\n"
         "    fig.suptitle(f'{animal}: sim - bio paired delta, mean +/- std', fontsize=10)\n"
         "    fig.tight_layout()\n"
         "    return fig\n"
         "\n"
         "for a, d in per_animal.items():\n"
         "    plot_paired_delta(d, cluster_results[a].labels_kmeans, animal=a)\n"
         "plt.show()\n"),

    md("## 11. Write `analysis_results.h5`"),
    code("def file_sha256(p):\n"
         "    h = hashlib.sha256()\n"
         "    with open(p, 'rb') as fh:\n"
         "        for chunk in iter(lambda: fh.read(1 << 20), b''):\n"
         "            h.update(chunk)\n"
         "    return h.hexdigest()\n"
         "\n"
         "with h5py.File(PAIRED_H5, 'r') as f:\n"
         "    animals_bytes = f['meta/animal'][:]\n"
         "    trials_int = f['meta/trial'][:]\n"
         "\n"
         "results = {\n"
         "    'meta': {\n"
         "        'animal': animals_bytes,\n"
         "        'trial':  trials_int,\n"
         "        'chosen_k_per_animal': {a: int(cluster_results[a].chosen_k) for a in per_animal},\n"
         "    },\n"
         "    'per_animal': {\n"
         "        a: {\n"
         "            'cluster_result': cluster_results[a],\n"
         "            'self_sim_bio':   self_sim_bio[a],\n"
         "            'self_sim_sim':   self_sim_sim[a],\n"
         "            'glm_per_cluster': glm_pc[a],\n"
         "            'glm_pooled':     glm_pool[a],\n"
         "        }\n"
         "        for a in per_animal\n"
         "    }\n"
         "}\n"
         "\n"
         "write_analysis_results(\n"
         "    results, OUT_H5,\n"
         "    source_paired_path=str(PAIRED_H5),\n"
         "    source_paired_sha256=file_sha256(PAIRED_H5),\n"
         "    notebook_commit='LOCAL',\n"
         ")\n"
         "print('wrote', OUT_H5)\n"),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
Path('notebooks/bio_vs_sim_kin_clustered.ipynb').write_text(json.dumps(nb, indent=1))
print('wrote notebooks/bio_vs_sim_kin_clustered.ipynb')
PY
```

Expected: prints "wrote notebooks/bio_vs_sim_kin_clustered.ipynb".

- [ ] **Step 2: Execute the notebook end to end via nbclient**

```bash
cd /root/vast/eric/vnl-playground
/root/vast/eric/vnl-playground/.venv/bin/python -m pip install --quiet nbclient nbformat
/root/vast/eric/vnl-playground/.venv/bin/python - <<'PY'
import nbformat
from nbclient import NotebookClient
nb = nbformat.read('notebooks/bio_vs_sim_kin_clustered.ipynb', as_version=4)
# Notebook expects paired_deterministic.h5 in cwd; cd to kin_emg_bundle.
import os
os.chdir('notebooks/kin_emg_bundle')
NotebookClient(nb, timeout=600).execute()
nbformat.write(nb, '../bio_vs_sim_kin_clustered.ipynb')
print('notebook executed')
PY
```

Expected: prints "notebook executed". Creates `analysis_results.h5` at `notebooks/kin_emg_bundle/analysis_results.h5`.

- [ ] **Step 3: Inspect the output HDF5**

```bash
/root/vast/eric/vnl-playground/.venv/bin/python -c "
import h5py
with h5py.File('notebooks/kin_emg_bundle/analysis_results.h5', 'r') as f:
    def show(name, obj):
        if hasattr(obj, 'shape'):
            print(f'{name:50s} {str(obj.shape):15s} {obj.dtype}')
    f.visititems(show)
"
```

Expected: per-animal groups under `/clustering/`, `/self_similarity/`, `/glm_bio_to_sim/` for 5 animals.

- [ ] **Step 4: Commit**

```bash
cd /root/vast/eric/vnl-playground
git add notebooks/bio_vs_sim_kin_clustered.ipynb
git -c user.name="Eric Leonardis" -c user.email="leonardiseric@gmail.com" commit -m "$(cat <<'EOF'
feat: Colab notebook bio_vs_sim_kin_clustered

Standalone Jupyter notebook that opens paired_deterministic.h5, runs
the four analysis steps (cluster, self-similarity, GLM, paired delta),
renders the figures, and writes analysis_results.h5. No project-internal
imports - all helper functions pasted from scripts/kin_emg_analysis.py.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Handoff bundle script

**Files:**
- Create: `scripts/make_handoff_bundle.sh`

- [ ] **Step 1: Write the bundle script**

`scripts/make_handoff_bundle.sh`:

```bash
#!/usr/bin/env bash
# Build a tarball for handoff to Talmo and Austin.
#
# Inputs (assumed already produced):
#   notebooks/kin_emg_bundle/paired_deterministic.h5
#   notebooks/kin_emg_bundle/analysis_results.h5
#   notebooks/bio_vs_sim_kin_clustered.ipynb
#   docs/superpowers/specs/2026-05-21-kinematics-clustered-bio-vs-sim-emg-design.md
#
# Output:
#   vnl-playground-kin-emg-bundle-<UTC-date>.tar.gz at repo root.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DATE="$(date -u +%Y-%m-%d)"
STAGE="$(mktemp -d)"
BUNDLE="bundle"
mkdir -p "$STAGE/$BUNDLE/figures"

cp notebooks/kin_emg_bundle/paired_deterministic.h5 "$STAGE/$BUNDLE/"
cp notebooks/kin_emg_bundle/analysis_results.h5     "$STAGE/$BUNDLE/"
cp notebooks/bio_vs_sim_kin_clustered.ipynb         "$STAGE/$BUNDLE/notebook.ipynb"
cp scripts/build_paired_deterministic_h5.py         "$STAGE/$BUNDLE/build_paired_deterministic_h5.py"
cp scripts/kin_emg_analysis.py                      "$STAGE/$BUNDLE/kin_emg_analysis.py"
cp docs/superpowers/specs/2026-05-21-kinematics-clustered-bio-vs-sim-emg-design.md \
   "$STAGE/$BUNDLE/SPEC.md"

cat > "$STAGE/$BUNDLE/README.md" <<MD
# vnl-playground bio vs sim muscle-activation bundle ($DATE)

Two HDF5 files plus the notebook that produced them.

## Files

- \`paired_deterministic.h5\` - the source: per-trial paired bio + sim
  features for 5 mice, 204 trials, plus all 12 sim actuator channels
  and the policy internals (intention latent, decoder activations).
  Index key: \`(meta/animal, meta/trial)\`.

- \`analysis_results.h5\` - per-animal clustering, within-cluster
  self-similarity (mean per-muscle Pearson r), GLM bio->sim fits
  per cluster and pooled.

- \`notebook.ipynb\` - reproduce all results from the two HDF5 files.

- \`build_paired_deterministic_h5.py\` - converter, for reproducibility.

- \`kin_emg_analysis.py\` - pure-function module the notebook pastes from.

- \`SPEC.md\` - the design spec.

## Open from Python

\`\`\`python
import h5py, numpy as np
with h5py.File("paired_deterministic.h5", "r") as f:
    animals = f["meta/animal"][:].astype(str)
    mask = animals == "AT006"
    bio_emg = f["bio/emg"][mask]           # (n_a, 60, 2): AD, Biceps
    sim_mA  = f["sim/muscle_act_AD_biceps"][mask]
print(bio_emg.shape, sim_mA.shape)
\`\`\`

## Open in Colab

Upload the tarball, extract, and run \`notebook.ipynb\`. Cell 1 pip-installs
the deps. Cells 2 onwards use only the two HDF5 files in the same directory.
MD

OUT="vnl-playground-kin-emg-bundle-${DATE}.tar.gz"
tar -C "$STAGE" -czf "$OUT" "$BUNDLE"
echo "wrote $OUT  ($(du -h "$OUT" | cut -f1))"
rm -rf "$STAGE"
```

- [ ] **Step 2: Make it executable and run it**

```bash
cd /root/vast/eric/vnl-playground
chmod +x scripts/make_handoff_bundle.sh
./scripts/make_handoff_bundle.sh
ls -lh vnl-playground-kin-emg-bundle-*.tar.gz
```

Expected: prints "wrote vnl-playground-kin-emg-bundle-YYYY-MM-DD.tar.gz" with size around 100 MB.

- [ ] **Step 3: Verify the tarball contents**

```bash
cd /root/vast/eric/vnl-playground
tar -tzf vnl-playground-kin-emg-bundle-*.tar.gz | head -20
```

Expected: lists `bundle/README.md`, `bundle/notebook.ipynb`, `bundle/paired_deterministic.h5`, `bundle/analysis_results.h5`, `bundle/build_paired_deterministic_h5.py`, `bundle/kin_emg_analysis.py`, `bundle/SPEC.md`, plus `bundle/figures/` (empty).

- [ ] **Step 4: Add the tarball pattern to gitignore**

```bash
cd /root/vast/eric/vnl-playground
echo "vnl-playground-kin-emg-bundle-*.tar.gz" >> .gitignore
```

- [ ] **Step 5: Commit**

```bash
cd /root/vast/eric/vnl-playground
git add scripts/make_handoff_bundle.sh .gitignore
git -c user.name="Eric Leonardis" -c user.email="leonardiseric@gmail.com" commit -m "$(cat <<'EOF'
feat: make_handoff_bundle.sh for Talmo/Austin tarball

Single-shot script that tars paired_deterministic.h5, analysis_results.h5,
the notebook, the converter, the analysis module, the spec, and an
auto-generated README. Output named with UTC date.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Final end-to-end smoke test on a clean checkout

**Files:** (none, validation only)

- [ ] **Step 1: Confirm all tests still pass from scratch**

```bash
cd /root/vast/eric/vnl-playground
/root/vast/eric/vnl-playground/.venv/bin/python -m pytest tests/ -v
```

Expected: 13 tests pass (5 schema + 8 analysis).

- [ ] **Step 2: Re-run the converter**

```bash
cd /root/vast/eric/vnl-playground
rm -f notebooks/kin_emg_bundle/paired_deterministic.h5 \
       notebooks/kin_emg_bundle/analysis_results.h5
/root/vast/eric/vnl-playground/.venv/bin/python -m scripts.build_paired_deterministic_h5 \
  --features-npz notebooks/kinematics_emg_comparison/cache/features.npz \
  --rollout-npz  notebooks/talk_figures/figs/rollout_activations/s18-ms-F4-fs1p2-20260502-014751_278clips.npz \
  --out-h5       notebooks/kin_emg_bundle/paired_deterministic.h5 \
  --checkpoint   "s18-ms-F4-fs1p2-20260502-014751" \
  --checkpoint-step 0
```

Expected: prints "wrote notebooks/kin_emg_bundle/paired_deterministic.h5".

- [ ] **Step 3: Re-execute the notebook**

```bash
cd /root/vast/eric/vnl-playground
/root/vast/eric/vnl-playground/.venv/bin/python - <<'PY'
import os, nbformat
from nbclient import NotebookClient
nb = nbformat.read('notebooks/bio_vs_sim_kin_clustered.ipynb', as_version=4)
os.chdir('notebooks/kin_emg_bundle')
NotebookClient(nb, timeout=900).execute()
nbformat.write(nb, '../bio_vs_sim_kin_clustered.ipynb')
print('notebook executed')
PY
```

Expected: prints "notebook executed". Both HDF5 files exist under `notebooks/kin_emg_bundle/`.

- [ ] **Step 4: Re-bundle and verify size**

```bash
cd /root/vast/eric/vnl-playground
rm -f vnl-playground-kin-emg-bundle-*.tar.gz
./scripts/make_handoff_bundle.sh
ls -lh vnl-playground-kin-emg-bundle-*.tar.gz
```

Expected: bundle ~100 MB.

- [ ] **Step 5: Open the bundle's HDF5 to confirm it's readable**

```bash
cd /root/vast/eric/vnl-playground
mkdir -p /tmp/bundle_check
tar -xzf vnl-playground-kin-emg-bundle-*.tar.gz -C /tmp/bundle_check
/root/vast/eric/vnl-playground/.venv/bin/python -c "
import h5py
with h5py.File('/tmp/bundle_check/bundle/paired_deterministic.h5', 'r') as f:
    print('checkpoint:', f['meta'].attrs['checkpoint'])
    print('animals:', set(f['meta/animal'][:].astype(str)))
with h5py.File('/tmp/bundle_check/bundle/analysis_results.h5', 'r') as f:
    print('source_paired:', f['meta'].attrs['source_paired'])
    print('clustering animals:', list(f['clustering'].keys()))
"
rm -rf /tmp/bundle_check
```

Expected: prints the checkpoint name, the 5-animal set, the source paired path, and the per-animal clustering groups.

- [ ] **Step 6: Final commit if any state changed**

```bash
cd /root/vast/eric/vnl-playground
git status --short
```

If clean: no commit needed. If anything changed: commit it with `chore: smoke test cleanups`.

---

## Self-Review

Spec coverage check:

- 5.1 paired_deterministic.h5 schema: Tasks 2-4.
- 5.2 stochastic_rollouts.h5 schema: deferred (Phase 2), no task needed.
- 5.3 analysis_results.h5 schema: Task 10 (write_analysis_results).
- 5.4 bundle layout: Task 12 (make_handoff_bundle.sh).
- Step 1 clustering: Tasks 5-6, plot in Task 9.
- Step 2 self-similarity: Task 7, plot in Task 9.
- Step 3 GLM per-cluster + pooled: Task 8, plot in Task 9.
- Step 4 paired delta: Task 11 (notebook cell, no separate module function since it's a single matplotlib block).
- Notebook structure (Section 7): Task 11.
- Style constraints (Section 3): enforced inline in each task.

Placeholder scan: no TBD / TODO / "implement later" in the plan body. Every code block has the full content.

Type consistency:
- `ClusterResult` defined in Task 6 with attributes `labels_kmeans`, `labels_hier`, `centroids_kmeans`, `wcss`, `silhouette`, `linkage_hier`, `k_grid`, `chosen_k`, `selection_rule`. Used in Tasks 7, 9, 10, 11 with the same names.
- `within_cluster_similarity` returns dict with keys `pairwise`, `cluster_summary`, `cluster_means`. Used consistently in Tasks 9, 10, 11.
- `fit_glm_per_cluster` / `fit_glm_pooled` return dicts with `slope`, `intercept`, `r2`, `n_samples`. Used consistently.
- `write_analysis_results` signature matches the call site in the notebook (Task 11).

One real risk to flag for the implementer: the rollout cache's `rollout_row` indexing in Task 3 assumes the cache is keyed in the same order as the 278-clip reference glob. If this assumption breaks, the schema test still passes (synthetic data) but the real-data run produces nonsense sim activations. Task 4 step 2 inspects shapes only, not values. **Mitigation:** after Task 4, manually plot one bio EMG and the matched sim muscle_act for a known animal/trial and confirm they share the basic reach envelope timing. If they do not, debug `meta_rollout` alignment before continuing.
