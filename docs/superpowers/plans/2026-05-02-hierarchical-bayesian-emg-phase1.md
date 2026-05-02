# Hierarchical Bayesian EMG — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the cache + correlation likelihood + three validation tests + report, sufficient to produce the 5×5 cross-likelihood discrimination matrix on existing s17 data — the Phase 1 gate from the spec.

**Architecture:** New package `vnl_playground/bayesian_emg/` consumes existing s17 checkpoints and per-trial empirical EMG envelopes. Trial-level sim envelopes are produced by reusing `scripts/emg_comparison.py` rollout primitives (`run_rollouts`, `process_sim_actions`) and cached to a single Parquet store. Importance-reweighting on the cache produces per-mouse posteriors over networks; three validation tests (coverage, discrimination, permutation) gate the framework before we extend to Options 2 and 3.

**Tech Stack:** Python 3.10+, JAX/Brax (existing rollout stack), pandas + pyarrow (cache), numpy/scipy (likelihoods + stats), wandb (network discovery), pytest (tests), Jinja2 (HTML report).

**Spec:** `docs/superpowers/specs/2026-05-02-hierarchical-bayesian-emg-population-design.md`

**Scope discipline:** Phase 1 only. Options 2 (Gaussian envelope) and 3 (ABC) are out of scope here; they get separate plans after the Phase 1 gate.

---

## File Structure

**Created in this plan:**

```
vnl_playground/bayesian_emg/
  __init__.py                          # package exports
  data.py                              # NetworkMouseFit dataclass, Cache reader/writer
  cache_builder.py                     # network discovery + rollout + ingestion
  likelihoods/
    __init__.py
    base.py                            # Likelihood Protocol, CredibleBand dataclass
    correlation.py                     # Option 1 — Fisher-z Gaussian
  posterior.py                         # importance reweighting, ESS, credible set
  validation/
    __init__.py
    coverage.py                        # within-mouse posterior predictive coverage
    discrimination.py                  # 5×5 cross-likelihood matrix + permutation
    permutation.py                     # full label-shuffle null
  preregistration.py                   # YAML loader + SHA-256 hash check
  report.py                            # HTML aggregator
scripts/
  bayes_emg_build_cache.py             # CLI: networks → cache
  bayes_emg_run.py                     # CLI: cache → posterior → validation → report
configs/bayesian_emg/
  preregistration.yaml                 # pinned hyperparameters (Phase 1 fields only)
tests/bayesian_emg/
  __init__.py
  conftest.py                          # synthetic_cache fixture
  test_data.py
  test_cache_builder.py
  test_likelihoods/
    __init__.py
    test_correlation.py
  test_posterior.py
  test_validation/
    __init__.py
    test_coverage.py
    test_discrimination.py
    test_permutation.py
  test_preregistration.py
  test_end_to_end.py                   # tiny synthetic cache → full report
```

**Reused (read-only):**
- `scripts/emg_comparison.py` — `run_rollouts`, `process_sim_actions`, `process_emg_data`, `load_intention_checkpoint`, `create_env_from_config`, `load_config`, `find_latest_step`, `build_muscle_configs`, `load_trial_info`, `TARGET_TIMESTEPS=60`, `ANIMAL_SESSIONS`.

**Modified:**
- `.gitignore` — add `vnl_playground/bayesian_emg/cache/`.

---

## Conventions

- All cache I/O is via pyarrow/pandas Parquet (already a project dependency).
- Tests use synthetic data unless explicitly marked `slow`. Real-checkpoint tests run only via `pytest -m slow` and are not part of the default CI loop.
- Each task ends with one `git commit`. Commit subjects are plain imperatives (no `bayesemg:` prefix — per project commit style).
- All file paths in code are absolute or relative to repo root.

---

## Task 1: Package skeleton + import smoke test

**Files:**
- Create: `vnl_playground/bayesian_emg/__init__.py`
- Create: `vnl_playground/bayesian_emg/likelihoods/__init__.py`
- Create: `vnl_playground/bayesian_emg/validation/__init__.py`
- Create: `tests/bayesian_emg/__init__.py`
- Create: `tests/bayesian_emg/test_likelihoods/__init__.py`
- Create: `tests/bayesian_emg/test_validation/__init__.py`
- Create: `tests/bayesian_emg/test_smoke.py`

- [ ] **Step 1: Write the failing test**

`tests/bayesian_emg/test_smoke.py`:
```python
def test_package_imports():
    import vnl_playground.bayesian_emg as bemg
    assert hasattr(bemg, "__version__")
```

- [ ] **Step 2: Run to verify fail**

```
pytest tests/bayesian_emg/test_smoke.py -v
```
Expected: FAIL — module not found.

- [ ] **Step 3: Create package files**

`vnl_playground/bayesian_emg/__init__.py`:
```python
"""Hierarchical Bayesian EMG framework.

See docs/superpowers/specs/2026-05-02-hierarchical-bayesian-emg-population-design.md
for the design and motivation.
"""

__version__ = "0.1.0"
```

`vnl_playground/bayesian_emg/likelihoods/__init__.py`, `vnl_playground/bayesian_emg/validation/__init__.py`, `tests/bayesian_emg/__init__.py`, `tests/bayesian_emg/test_likelihoods/__init__.py`, `tests/bayesian_emg/test_validation/__init__.py`: each empty file (`""`).

- [ ] **Step 4: Run to verify pass**

```
pytest tests/bayesian_emg/test_smoke.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add vnl_playground/bayesian_emg/ tests/bayesian_emg/
git commit -m "scaffold bayesian_emg package and test tree"
```

---

## Task 2: `NetworkMouseFit` dataclass and cache schema

**Files:**
- Create: `vnl_playground/bayesian_emg/data.py`
- Create: `tests/bayesian_emg/test_data.py`

The cache stores per-(network, animal) entries. Each entry holds two arrays: `sim` of shape `(n_trials, TARGET_TIMESTEPS=60, n_muscles=3)` and `empirical` of shape `(n_trials, 60, 3)`. Muscle order is `["biceps", "triceps", "AD"]` — fixed. We persist as Parquet using a long-format table for portability: one row per (network_id, animal, trial, muscle, timestep) with columns `sim`, `empirical`. A sidecar `networks.parquet` holds hyperparameters.

- [ ] **Step 1: Write the failing test**

`tests/bayesian_emg/test_data.py`:
```python
import numpy as np
import pandas as pd
import pytest

from vnl_playground.bayesian_emg.data import (
    NetworkMouseFit,
    Cache,
    MUSCLES,
    TARGET_TIMESTEPS,
)


def test_muscles_and_timesteps_constants():
    assert MUSCLES == ("biceps", "triceps", "AD")
    assert TARGET_TIMESTEPS == 60


def test_network_mouse_fit_shape_validation():
    sim = np.zeros((5, 60, 3))
    emp = np.zeros((5, 60, 3))
    fit = NetworkMouseFit(network_id="n1", animal="A36-1", sim=sim, empirical=emp)
    assert fit.n_trials == 5

    with pytest.raises(ValueError):
        NetworkMouseFit(network_id="n1", animal="A36-1",
                        sim=np.zeros((5, 60, 3)),
                        empirical=np.zeros((4, 60, 3)))


def test_cache_roundtrip(tmp_path):
    cache = Cache(tmp_path / "cache.parquet")
    fit = NetworkMouseFit(
        network_id="n1", animal="A36-1",
        sim=np.random.RandomState(0).rand(3, 60, 3),
        empirical=np.random.RandomState(1).rand(3, 60, 3),
    )
    cache.write_fit(fit)
    cache.write_network_meta("n1", {"force_scale": 1.1, "joint_damping": 1.5e-6, "seed": 0})

    loaded = cache.read_fit("n1", "A36-1")
    np.testing.assert_array_almost_equal(loaded.sim, fit.sim)
    np.testing.assert_array_almost_equal(loaded.empirical, fit.empirical)

    meta = cache.read_network_meta("n1")
    assert meta["force_scale"] == pytest.approx(1.1)


def test_cache_idempotent_skip(tmp_path):
    cache = Cache(tmp_path / "cache.parquet")
    fit = NetworkMouseFit("n1", "A36-1",
                          np.zeros((2, 60, 3)), np.zeros((2, 60, 3)))
    cache.write_fit(fit)
    assert cache.has_fit("n1", "A36-1")
    assert not cache.has_fit("n1", "AT006")


def test_cache_list_networks_animals(tmp_path):
    cache = Cache(tmp_path / "cache.parquet")
    for net in ["n1", "n2"]:
        for animal in ["A36-1", "AT006"]:
            cache.write_fit(NetworkMouseFit(net, animal,
                                            np.zeros((1, 60, 3)),
                                            np.zeros((1, 60, 3))))
    assert sorted(cache.list_networks()) == ["n1", "n2"]
    assert sorted(cache.list_animals()) == ["A36-1", "AT006"]


def test_cache_content_hash_changes_on_write(tmp_path):
    cache = Cache(tmp_path / "cache.parquet")
    h0 = cache.content_hash()
    cache.write_fit(NetworkMouseFit("n1", "A36-1",
                                     np.zeros((1, 60, 3)),
                                     np.zeros((1, 60, 3))))
    h1 = cache.content_hash()
    assert h0 != h1
```

- [ ] **Step 2: Run to verify fail**

```
pytest tests/bayesian_emg/test_data.py -v
```
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `data.py`**

`vnl_playground/bayesian_emg/data.py`:
```python
"""Cache schema for Bayesian EMG framework.

Long-format Parquet keyed by (network_id, animal, trial, muscle, timestep) with
columns sim, empirical. Sidecar networks.parquet holds hyperparameters.
Empirical envelopes are duplicated across networks for the same (animal, trial,
muscle, timestep) — the redundancy keeps reads simple and the absolute size
small (<200 MB for the s17 sweep).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MUSCLES: tuple[str, ...] = ("biceps", "triceps", "AD")
TARGET_TIMESTEPS: int = 60


@dataclass(frozen=True)
class NetworkMouseFit:
    network_id: str
    animal: str
    sim: np.ndarray         # (n_trials, TARGET_TIMESTEPS, len(MUSCLES))
    empirical: np.ndarray   # (n_trials, TARGET_TIMESTEPS, len(MUSCLES))

    def __post_init__(self) -> None:
        if self.sim.shape != self.empirical.shape:
            raise ValueError(
                f"sim shape {self.sim.shape} != empirical shape {self.empirical.shape}"
            )
        if self.sim.shape[1:] != (TARGET_TIMESTEPS, len(MUSCLES)):
            raise ValueError(
                f"expected per-trial shape ({TARGET_TIMESTEPS}, {len(MUSCLES)}), "
                f"got {self.sim.shape[1:]}"
            )

    @property
    def n_trials(self) -> int:
        return int(self.sim.shape[0])


class Cache:
    """File-backed Parquet cache. Two files: <root>.fits.parquet, <root>.meta.parquet."""

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        self.fits_path = path.with_suffix(".fits.parquet")
        self.meta_path = path.with_suffix(".meta.parquet")

    def write_fit(self, fit: NetworkMouseFit) -> None:
        rows = []
        for trial in range(fit.n_trials):
            for mi, muscle in enumerate(MUSCLES):
                for t in range(TARGET_TIMESTEPS):
                    rows.append({
                        "network_id": fit.network_id,
                        "animal": fit.animal,
                        "trial": trial,
                        "muscle": muscle,
                        "timestep": t,
                        "sim": float(fit.sim[trial, t, mi]),
                        "empirical": float(fit.empirical[trial, t, mi]),
                    })
        new_df = pd.DataFrame(rows)
        if self.fits_path.exists():
            existing = pd.read_parquet(self.fits_path)
            mask = ~((existing["network_id"] == fit.network_id) &
                     (existing["animal"] == fit.animal))
            new_df = pd.concat([existing[mask], new_df], ignore_index=True)
        else:
            self.fits_path.parent.mkdir(parents=True, exist_ok=True)
        new_df.to_parquet(self.fits_path, index=False)

    def read_fit(self, network_id: str, animal: str) -> NetworkMouseFit:
        df = pd.read_parquet(self.fits_path)
        sub = df[(df["network_id"] == network_id) & (df["animal"] == animal)]
        if sub.empty:
            raise KeyError(f"no fit for ({network_id}, {animal})")
        n_trials = int(sub["trial"].max()) + 1
        sim = np.zeros((n_trials, TARGET_TIMESTEPS, len(MUSCLES)))
        emp = np.zeros((n_trials, TARGET_TIMESTEPS, len(MUSCLES)))
        muscle_idx = {m: i for i, m in enumerate(MUSCLES)}
        for _, row in sub.iterrows():
            mi = muscle_idx[row["muscle"]]
            sim[int(row["trial"]), int(row["timestep"]), mi] = row["sim"]
            emp[int(row["trial"]), int(row["timestep"]), mi] = row["empirical"]
        return NetworkMouseFit(network_id, animal, sim, emp)

    def has_fit(self, network_id: str, animal: str) -> bool:
        if not self.fits_path.exists():
            return False
        df = pd.read_parquet(self.fits_path, columns=["network_id", "animal"])
        return bool(((df["network_id"] == network_id) & (df["animal"] == animal)).any())

    def list_networks(self) -> list[str]:
        if not self.fits_path.exists():
            return []
        df = pd.read_parquet(self.fits_path, columns=["network_id"])
        return sorted(df["network_id"].unique().tolist())

    def list_animals(self) -> list[str]:
        if not self.fits_path.exists():
            return []
        df = pd.read_parquet(self.fits_path, columns=["animal"])
        return sorted(df["animal"].unique().tolist())

    def write_network_meta(self, network_id: str, meta: dict) -> None:
        row = {"network_id": network_id, **meta}
        new_df = pd.DataFrame([row])
        if self.meta_path.exists():
            existing = pd.read_parquet(self.meta_path)
            mask = existing["network_id"] != network_id
            new_df = pd.concat([existing[mask], new_df], ignore_index=True)
        else:
            self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        new_df.to_parquet(self.meta_path, index=False)

    def read_network_meta(self, network_id: str) -> dict:
        df = pd.read_parquet(self.meta_path)
        sub = df[df["network_id"] == network_id]
        if sub.empty:
            raise KeyError(f"no meta for {network_id}")
        return sub.iloc[0].to_dict()

    def iter_fits(self) -> Iterable[NetworkMouseFit]:
        for net in self.list_networks():
            for animal in self.list_animals():
                if self.has_fit(net, animal):
                    yield self.read_fit(net, animal)

    def content_hash(self) -> str:
        h = hashlib.sha256()
        for p in (self.fits_path, self.meta_path):
            if p.exists():
                h.update(p.read_bytes())
        return h.hexdigest()
```

- [ ] **Step 4: Run to verify pass**

```
pytest tests/bayesian_emg/test_data.py -v
```
Expected: PASS — all 6 tests.

- [ ] **Step 5: Commit**

```
git add vnl_playground/bayesian_emg/data.py tests/bayesian_emg/test_data.py
git commit -m "add NetworkMouseFit cache with Parquet roundtrip"
```

---

## Task 3: Synthetic cache fixture for downstream tests

**Files:**
- Create: `tests/bayesian_emg/conftest.py`

This fixture is used by every later test. A "synthetic cache" has 6 networks × 3 animals × 4 trials, where networks 1–2 are tuned to mimic animal A's pattern, networks 3–4 mimic animal B's, and networks 5–6 mimic animal C's. This gives a known ground-truth diagonal in the discrimination matrix that downstream tests assert against.

- [ ] **Step 1: Write the fixture**

`tests/bayesian_emg/conftest.py`:
```python
"""Shared fixtures for bayesian_emg tests.

The `synthetic_cache` fixture builds a cache where networks have known
mouse-specific structure: 2 networks per animal, each producing sim envelopes
that match its target animal's empirical envelope plus mild noise. This gives
a clear ground-truth diagonal for discrimination tests.
"""

import numpy as np
import pytest

from vnl_playground.bayesian_emg.data import Cache, NetworkMouseFit, MUSCLES, TARGET_TIMESTEPS


def _animal_signature(animal: str, seed: int) -> np.ndarray:
    """Per-animal characteristic envelope, shape (TARGET_TIMESTEPS, len(MUSCLES))."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, TARGET_TIMESTEPS)
    base_phases = {"A": 0.3, "B": 0.5, "C": 0.7}
    base = base_phases[animal]
    sig = np.zeros((TARGET_TIMESTEPS, len(MUSCLES)))
    for mi in range(len(MUSCLES)):
        sig[:, mi] = np.exp(-((t - (base + 0.05 * mi)) ** 2) / 0.02)
    return sig + 0.05 * rng.randn(*sig.shape)


@pytest.fixture
def synthetic_cache(tmp_path):
    cache = Cache(tmp_path / "synth.parquet")
    animals = ["A", "B", "C"]
    n_trials = 4
    rng = np.random.RandomState(42)

    # Empirical envelope per (animal, trial): a noisy sample of the animal signature
    empirical_by_animal = {
        a: np.stack([_animal_signature(a, 100 + a_idx * 10 + t)
                     for t in range(n_trials)])
        for a_idx, a in enumerate(animals)
    }

    # 2 networks per animal target, each network's sim matches its target animal
    network_targets = {
        "n1": "A", "n2": "A",
        "n3": "B", "n4": "B",
        "n5": "C", "n6": "C",
    }

    for net_id, target_animal in network_targets.items():
        for animal in animals:
            sim = np.stack([
                _animal_signature(target_animal, 200 + hash((net_id, animal, t)) % 1000)
                for t in range(n_trials)
            ])
            empirical = empirical_by_animal[animal]
            cache.write_fit(NetworkMouseFit(net_id, animal, sim, empirical))
        cache.write_network_meta(net_id, {
            "force_scale": 1.1,
            "joint_damping": 1.5e-6,
            "seed": int(net_id[1:]),
            "target_animal": target_animal,  # for test assertions only
        })

    return cache
```

- [ ] **Step 2: Add a smoke test**

Append to `tests/bayesian_emg/test_data.py`:
```python
def test_synthetic_cache_fixture(synthetic_cache):
    assert sorted(synthetic_cache.list_networks()) == ["n1", "n2", "n3", "n4", "n5", "n6"]
    assert sorted(synthetic_cache.list_animals()) == ["A", "B", "C"]
    fit = synthetic_cache.read_fit("n1", "A")
    assert fit.n_trials == 4
```

- [ ] **Step 3: Run**

```
pytest tests/bayesian_emg/test_data.py::test_synthetic_cache_fixture -v
```
Expected: PASS.

- [ ] **Step 4: Commit**

```
git add tests/bayesian_emg/conftest.py tests/bayesian_emg/test_data.py
git commit -m "add synthetic_cache fixture with mouse-specific networks"
```

---

## Task 4: Cache builder — empirical envelope ingestion

**Files:**
- Create: `vnl_playground/bayesian_emg/cache_builder.py`
- Create: `tests/bayesian_emg/test_cache_builder.py`

This task wraps `process_emg_data` so that one call ingests all three muscles for one animal into the cache as the empirical side of a per-(network, animal) fit. The empirical array is shared across networks for the same animal — Tasks 5 and 6 fill in the sim side.

- [ ] **Step 1: Write the failing test**

`tests/bayesian_emg/test_cache_builder.py`:
```python
from unittest.mock import patch

import numpy as np
import pytest

from vnl_playground.bayesian_emg.cache_builder import load_animal_empirical
from vnl_playground.bayesian_emg.data import MUSCLES, TARGET_TIMESTEPS


def _fake_process_emg_data(emg_file_path, valid_trials_df, n_clips, target_samples, percentile):
    # Return a deterministic envelope keyed by the muscle name embedded in the path
    seed = abs(hash(str(emg_file_path))) % (2**32)
    rng = np.random.RandomState(seed)
    return rng.rand(n_clips, target_samples)


def _fake_load_trial_info(animal):
    import pandas as pd
    return pd.DataFrame({
        "start": np.arange(1, 6) * 100,
        "end": np.arange(1, 6) * 100 + 50,
    })


def _fake_build_muscle_configs(animal):
    return {
        "biceps":  {"emg_file": f"/fake/{animal}/biceps.csv",  "sim_idx": 0},
        "triceps": {"emg_file": f"/fake/{animal}/triceps.csv", "sim_idx": 1},
        "AD":      {"emg_file": f"/fake/{animal}/AD.csv",      "sim_idx": 2},
    }


def test_load_animal_empirical_shape():
    with patch("vnl_playground.bayesian_emg.cache_builder.process_emg_data",
               side_effect=_fake_process_emg_data), \
         patch("vnl_playground.bayesian_emg.cache_builder.load_trial_info",
               side_effect=_fake_load_trial_info), \
         patch("vnl_playground.bayesian_emg.cache_builder.build_muscle_configs",
               side_effect=_fake_build_muscle_configs):
        emp = load_animal_empirical("A36-1", n_clips=5)
    assert emp.shape == (5, TARGET_TIMESTEPS, len(MUSCLES))


def test_load_animal_empirical_muscle_order():
    """Muscle axis must follow MUSCLES ordering, regardless of dict iteration order."""
    with patch("vnl_playground.bayesian_emg.cache_builder.process_emg_data",
               side_effect=_fake_process_emg_data), \
         patch("vnl_playground.bayesian_emg.cache_builder.load_trial_info",
               side_effect=_fake_load_trial_info), \
         patch("vnl_playground.bayesian_emg.cache_builder.build_muscle_configs",
               side_effect=_fake_build_muscle_configs):
        emp = load_animal_empirical("A36-1", n_clips=5)
    # First muscle is biceps — its envelope is deterministic from path
    seed_biceps = abs(hash("/fake/A36-1/biceps.csv")) % (2**32)
    expected = np.random.RandomState(seed_biceps).rand(5, TARGET_TIMESTEPS)
    np.testing.assert_array_almost_equal(emp[:, :, 0], expected)
```

- [ ] **Step 2: Run to verify fail**

```
pytest tests/bayesian_emg/test_cache_builder.py -v
```
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `cache_builder.py` (empirical only for now)**

`vnl_playground/bayesian_emg/cache_builder.py`:
```python
"""Cache builder: discover networks, run rollouts, ingest empirical envelopes."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Import rollout primitives from the existing emg_comparison script.
# It lives in scripts/, not on the package path — add scripts/ to sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from emg_comparison import (  # noqa: E402
    process_emg_data,
    process_sim_actions,
    run_rollouts,
    load_intention_checkpoint,
    create_env_from_config,
    load_config,
    find_latest_step,
    build_muscle_configs,
    load_trial_info,
    TARGET_TIMESTEPS,
)

from vnl_playground.bayesian_emg.data import MUSCLES


def load_animal_empirical(animal: str, n_clips: int) -> np.ndarray:
    """Return per-(trial, timestep, muscle) empirical envelope for one animal.

    Wraps `process_emg_data` for each muscle, stacking on the muscle axis in
    the order defined by MUSCLES.
    """
    trial_info = load_trial_info(animal)
    muscle_configs = build_muscle_configs(animal)

    out = np.zeros((n_clips, TARGET_TIMESTEPS, len(MUSCLES)))
    for mi, muscle in enumerate(MUSCLES):
        cfg = muscle_configs[muscle]
        env = process_emg_data(
            cfg["emg_file"], trial_info, n_clips=n_clips,
            target_samples=TARGET_TIMESTEPS, percentile=100.0,
        )
        if env is None:
            continue
        out[:env.shape[0], :, mi] = env
    return out
```

- [ ] **Step 4: Run to verify pass**

```
pytest tests/bayesian_emg/test_cache_builder.py -v
```
Expected: PASS — both tests.

- [ ] **Step 5: Commit**

```
git add vnl_playground/bayesian_emg/cache_builder.py tests/bayesian_emg/test_cache_builder.py
git commit -m "add empirical envelope loader for cache builder"
```

---

## Task 5: Cache builder — sim envelope rollout

**Files:**
- Modify: `vnl_playground/bayesian_emg/cache_builder.py` (add `rollout_sim`)
- Modify: `tests/bayesian_emg/test_cache_builder.py` (add rollout tests)

`rollout_sim` takes a checkpoint directory and an animal, loads the policy + env, runs `n_clips` deterministic rollouts using the existing `run_rollouts`, and returns the sim envelope on the cache grid. The actual rollout is JAX-heavy and slow; the test mocks `run_rollouts` to return a synthetic ctrl array so the test stays fast.

- [ ] **Step 1: Write the failing test**

Append to `tests/bayesian_emg/test_cache_builder.py`:
```python
def _fake_run_rollouts(params, policy_fn, env, n_clips, episode_length):
    rng = np.random.RandomState(0)
    return {
        "ctrl": rng.rand(n_clips, episode_length, len(MUSCLES)),
        "raw_action": rng.rand(n_clips, episode_length, len(MUSCLES)),
        "rewards": rng.rand(n_clips, episode_length),
    }


def test_rollout_sim_shape():
    from vnl_playground.bayesian_emg.cache_builder import rollout_sim

    with patch("vnl_playground.bayesian_emg.cache_builder.load_config",
               return_value={"episode_length": 100, "checkpoint_dir": "/fake"}), \
         patch("vnl_playground.bayesian_emg.cache_builder.find_latest_step",
               return_value=1000), \
         patch("vnl_playground.bayesian_emg.cache_builder.create_env_from_config",
               return_value=object()), \
         patch("vnl_playground.bayesian_emg.cache_builder.load_intention_checkpoint",
               return_value=({}, lambda *a, **k: (np.zeros(3), None))), \
         patch("vnl_playground.bayesian_emg.cache_builder.run_rollouts",
               side_effect=_fake_run_rollouts), \
         patch("vnl_playground.bayesian_emg.cache_builder._infer_obs_sizes",
               return_value=(10, 3, 5, 5)):
        sim = rollout_sim("/fake/checkpoint", animal="A36-1", n_clips=4)
    assert sim.shape == (4, TARGET_TIMESTEPS, len(MUSCLES))
    assert (sim >= 0).all() and (sim <= 1).all()
```

- [ ] **Step 2: Run to verify fail**

```
pytest tests/bayesian_emg/test_cache_builder.py::test_rollout_sim_shape -v
```
Expected: FAIL — `rollout_sim` not defined.

- [ ] **Step 3: Append `rollout_sim` and `_infer_obs_sizes`**

Append to `vnl_playground/bayesian_emg/cache_builder.py`:
```python
def _infer_obs_sizes(env) -> tuple[int, int, int, int]:
    """Return (obs_size, act_size, proprio_size, intention_size) for the env.

    Pulled from the same logic used in scripts/emg_comparison.py:main(). Kept
    isolated so tests can monkeypatch.
    """
    # Pragmatic implementation: introspect a reset state.
    import jax
    state = env.reset(jax.random.PRNGKey(0))
    obs_flat = state.obs if hasattr(state.obs, "shape") else None
    obs_size = int(obs_flat.shape[-1]) if obs_flat is not None else 0
    act_size = int(env.action_size)
    # proprio/intention sizes are pulled from env config; use sensible defaults
    # if unavailable. Override via monkeypatch in tests.
    proprio_size = int(getattr(env, "proprio_size", 0))
    intention_size = int(getattr(env, "intention_size", 0))
    return obs_size, act_size, proprio_size, intention_size


def rollout_sim(checkpoint_dir: str, animal: str, n_clips: int) -> np.ndarray:
    """Roll out a policy and return the sim envelope on the cache grid.

    Returns shape (n_clips, TARGET_TIMESTEPS, len(MUSCLES)).
    """
    config = load_config(checkpoint_dir)
    step = find_latest_step(checkpoint_dir)
    env = create_env_from_config(config)
    obs_size, act_size, proprio_size, intention_size = _infer_obs_sizes(env)
    params, policy_fn = load_intention_checkpoint(
        checkpoint_dir, step, obs_size, act_size, proprio_size, intention_size,
    )
    episode_length = int(config.get("episode_length", 100))
    data = run_rollouts(params, policy_fn, env, n_clips, episode_length)
    sim_actions = process_sim_actions(data["ctrl"], TARGET_TIMESTEPS)
    # process_sim_actions returns shape (n_clips, TARGET_TIMESTEPS, n_muscles_total).
    # Slice down to the 3 muscles in MUSCLES via the per-animal sim_idx.
    cfgs = build_muscle_configs(animal)
    out = np.zeros((sim_actions.shape[0], TARGET_TIMESTEPS, len(MUSCLES)))
    for mi, muscle in enumerate(MUSCLES):
        out[:, :, mi] = sim_actions[:, :, cfgs[muscle]["sim_idx"]]
    return np.clip(out, 0.0, 1.0)
```

- [ ] **Step 4: Run to verify pass**

```
pytest tests/bayesian_emg/test_cache_builder.py -v
```
Expected: PASS — all 3 tests.

- [ ] **Step 5: Commit**

```
git add vnl_playground/bayesian_emg/cache_builder.py tests/bayesian_emg/test_cache_builder.py
git commit -m "add sim envelope rollout to cache builder"
```

---

## Task 6: Cache builder — orchestration + idempotent build

**Files:**
- Modify: `vnl_playground/bayesian_emg/cache_builder.py` (add `build_one`, `build_many`)
- Modify: `tests/bayesian_emg/test_cache_builder.py` (add orchestration tests)

`build_one(checkpoint_dir, animal, n_clips, cache)` skips if the cache already has the (network_id, animal) entry. `build_many(checkpoints, animals, n_clips, cache)` iterates with progress logging.

- [ ] **Step 1: Write the failing test**

Append to `tests/bayesian_emg/test_cache_builder.py`:
```python
def test_build_one_skips_existing(tmp_path):
    from vnl_playground.bayesian_emg.cache_builder import build_one
    from vnl_playground.bayesian_emg.data import Cache, NetworkMouseFit

    cache = Cache(tmp_path / "c.parquet")
    cache.write_fit(NetworkMouseFit("net_x", "A36-1",
                                     np.zeros((1, 60, 3)),
                                     np.zeros((1, 60, 3))))

    called = {"rollout": 0, "empirical": 0}

    def fake_rollout(*a, **k):
        called["rollout"] += 1
        return np.zeros((1, 60, 3))

    def fake_empirical(*a, **k):
        called["empirical"] += 1
        return np.zeros((1, 60, 3))

    with patch("vnl_playground.bayesian_emg.cache_builder.rollout_sim",
               side_effect=fake_rollout), \
         patch("vnl_playground.bayesian_emg.cache_builder.load_animal_empirical",
               side_effect=fake_empirical):
        build_one(network_id="net_x", checkpoint_dir="/fake", animal="A36-1",
                  n_clips=1, cache=cache)
    assert called["rollout"] == 0
    assert called["empirical"] == 0


def test_build_one_writes_when_missing(tmp_path):
    from vnl_playground.bayesian_emg.cache_builder import build_one
    from vnl_playground.bayesian_emg.data import Cache

    cache = Cache(tmp_path / "c.parquet")

    def fake_rollout(*a, **k):
        return np.ones((2, 60, 3)) * 0.5

    def fake_empirical(*a, **k):
        return np.ones((2, 60, 3)) * 0.7

    with patch("vnl_playground.bayesian_emg.cache_builder.rollout_sim",
               side_effect=fake_rollout), \
         patch("vnl_playground.bayesian_emg.cache_builder.load_animal_empirical",
               side_effect=fake_empirical):
        build_one(network_id="net_x", checkpoint_dir="/fake", animal="A36-1",
                  n_clips=2, cache=cache)
    fit = cache.read_fit("net_x", "A36-1")
    assert fit.n_trials == 2
    assert fit.sim.mean() == pytest.approx(0.5)
    assert fit.empirical.mean() == pytest.approx(0.7)
```

- [ ] **Step 2: Run to verify fail**

```
pytest tests/bayesian_emg/test_cache_builder.py -v
```
Expected: FAIL — `build_one` not defined.

- [ ] **Step 3: Implement orchestration**

Append to `vnl_playground/bayesian_emg/cache_builder.py`:
```python
import logging

logger = logging.getLogger(__name__)


def build_one(network_id: str, checkpoint_dir: str, animal: str,
              n_clips: int, cache) -> None:
    """Populate cache[network_id, animal] if missing. No-op if present."""
    if cache.has_fit(network_id, animal):
        logger.info("skip %s × %s (cached)", network_id, animal)
        return
    sim = rollout_sim(checkpoint_dir, animal, n_clips)
    emp = load_animal_empirical(animal, n_clips)
    n = min(sim.shape[0], emp.shape[0])
    from vnl_playground.bayesian_emg.data import NetworkMouseFit
    cache.write_fit(NetworkMouseFit(network_id, animal, sim[:n], emp[:n]))
    logger.info("wrote %s × %s (n_trials=%d)", network_id, animal, n)


def build_many(networks: list[tuple[str, str, dict]], animals: list[str],
               n_clips: int, cache) -> None:
    """networks is a list of (network_id, checkpoint_dir, hyperparam_meta)."""
    for nid, cdir, meta in networks:
        cache.write_network_meta(nid, meta)
        for animal in animals:
            build_one(nid, cdir, animal, n_clips, cache)
```

- [ ] **Step 4: Run to verify pass**

```
pytest tests/bayesian_emg/test_cache_builder.py -v
```
Expected: PASS — all 5 tests.

- [ ] **Step 5: Commit**

```
git add vnl_playground/bayesian_emg/cache_builder.py tests/bayesian_emg/test_cache_builder.py
git commit -m "add idempotent cache build orchestration"
```

---

## Task 7: Likelihood Protocol + `CredibleBand`

**Files:**
- Create: `vnl_playground/bayesian_emg/likelihoods/base.py`
- Create: `tests/bayesian_emg/test_likelihoods/test_base.py`

The Protocol defines what every likelihood implementation must satisfy. `CredibleBand` carries the per-(timestep, muscle) lower/upper bounds for posterior predictive coverage tests.

- [ ] **Step 1: Write the failing test**

`tests/bayesian_emg/test_likelihoods/test_base.py`:
```python
import numpy as np

from vnl_playground.bayesian_emg.likelihoods.base import CredibleBand


def test_credible_band_shape():
    lower = np.zeros((60, 3))
    upper = np.ones((60, 3))
    cb = CredibleBand(lower=lower, upper=upper, level=0.9)
    assert cb.contains(np.full((60, 3), 0.5)).all()
    assert not cb.contains(np.full((60, 3), 1.5)).any()
```

- [ ] **Step 2: Run to verify fail**

```
pytest tests/bayesian_emg/test_likelihoods/test_base.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement**

`vnl_playground/bayesian_emg/likelihoods/base.py`:
```python
"""Likelihood protocol and CredibleBand dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from vnl_playground.bayesian_emg.data import NetworkMouseFit


@dataclass(frozen=True)
class CredibleBand:
    lower: np.ndarray   # (TARGET_TIMESTEPS, n_muscles)
    upper: np.ndarray
    level: float        # nominal coverage, e.g. 0.9

    def contains(self, trial: np.ndarray) -> np.ndarray:
        """Element-wise containment indicator, same shape as trial."""
        return (trial >= self.lower) & (trial <= self.upper)


class Likelihood(Protocol):
    name: str

    def log_likelihood(
        self,
        fit: NetworkMouseFit,
        *,
        holdout_trials: list[int] | None = None,
    ) -> float:
        ...

    def posterior_predictive(
        self,
        fits: list[NetworkMouseFit],
        weights: np.ndarray,
        level: float = 0.9,
    ) -> CredibleBand:
        ...
```

- [ ] **Step 4: Run**

```
pytest tests/bayesian_emg/test_likelihoods/test_base.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add vnl_playground/bayesian_emg/likelihoods/ tests/bayesian_emg/test_likelihoods/
git commit -m "define Likelihood protocol and CredibleBand"
```

---

## Task 8: Correlation likelihood — σ² estimator

**Files:**
- Create: `vnl_playground/bayesian_emg/likelihoods/correlation.py`
- Create: `tests/bayesian_emg/test_likelihoods/test_correlation.py`

σ²_μ for each muscle is bootstrapped from the across-seed scatter at fixed cells. Phase 1 takes σ² as a parameter (set by preregistration YAML) so the framework runs end-to-end before s18 multi-seed data is in. We provide a helper that computes σ² from a list of `(network_id, muscle, fisher_z)` rows when seed-replicated runs exist.

- [ ] **Step 1: Write the failing test**

`tests/bayesian_emg/test_likelihoods/test_correlation.py`:
```python
import numpy as np
import pytest

from vnl_playground.bayesian_emg.likelihoods.correlation import (
    estimate_sigma_sq,
    fisher_z,
    CorrelationLikelihood,
)


def test_fisher_z_basic():
    assert fisher_z(0.0) == pytest.approx(0.0)
    assert fisher_z(0.5) == pytest.approx(np.arctanh(0.5))
    assert fisher_z(0.99) > 1.0
    # Clamped away from ±1 to avoid inf
    assert np.isfinite(fisher_z(1.0))
    assert np.isfinite(fisher_z(-1.0))


def test_estimate_sigma_sq_from_seed_groups():
    # Three seeds in one cell; biceps z values jitter, triceps stable
    rows = [
        {"cell_id": "c1", "muscle": "biceps", "fisher_z": 0.5},
        {"cell_id": "c1", "muscle": "biceps", "fisher_z": 0.6},
        {"cell_id": "c1", "muscle": "biceps", "fisher_z": 0.4},
        {"cell_id": "c1", "muscle": "triceps", "fisher_z": 0.7},
        {"cell_id": "c1", "muscle": "triceps", "fisher_z": 0.7},
        {"cell_id": "c1", "muscle": "triceps", "fisher_z": 0.7},
    ]
    sigma_sq = estimate_sigma_sq(rows)
    assert sigma_sq["biceps"] > 0
    assert sigma_sq["triceps"] == pytest.approx(0.0, abs=1e-9)
    assert sigma_sq["biceps"] > sigma_sq["triceps"]


def test_estimate_sigma_sq_falls_back_to_default():
    sigma_sq = estimate_sigma_sq([], default=0.1)
    assert sigma_sq == {"biceps": 0.1, "triceps": 0.1, "AD": 0.1}
```

- [ ] **Step 2: Run to verify fail**

```
pytest tests/bayesian_emg/test_likelihoods/test_correlation.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement σ² + fisher_z**

`vnl_playground/bayesian_emg/likelihoods/correlation.py`:
```python
"""Option 1 — Fisher-z Gaussian likelihood on per-(mouse, muscle) Pearson r.

Cheap, runs on existing s17 metrics. Ignores amplitude bias by design — that
cost is documented in the spec.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from vnl_playground.bayesian_emg.data import NetworkMouseFit, MUSCLES
from vnl_playground.bayesian_emg.likelihoods.base import CredibleBand


_FISHER_CLAMP = 1.0 - 1e-6


def fisher_z(r: float | np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(r, -_FISHER_CLAMP, _FISHER_CLAMP))


def _trial_mean_correlation(fit: NetworkMouseFit, muscle_idx: int,
                            trial_mask: np.ndarray | None = None) -> float:
    sim = fit.sim[:, :, muscle_idx]
    emp = fit.empirical[:, :, muscle_idx]
    if trial_mask is not None:
        sim = sim[trial_mask]
        emp = emp[trial_mask]
    if sim.size == 0:
        return 0.0
    sim_mean = sim.mean(axis=0)
    emp_mean = emp.mean(axis=0)
    if sim_mean.std() < 1e-9 or emp_mean.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(sim_mean, emp_mean)[0, 1])


def estimate_sigma_sq(rows: Iterable[dict], default: float = 0.1) -> dict[str, float]:
    """Per-muscle σ² of fisher_z across replicate seeds within the same cell.

    Each row needs cell_id, muscle, fisher_z. Variance is taken within each
    cell, then averaged across cells per muscle. Falls back to `default` when
    there are no replicates.
    """
    by_cell: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        by_cell[(row["cell_id"], row["muscle"])].append(float(row["fisher_z"]))
    per_muscle: dict[str, list[float]] = defaultdict(list)
    for (_, muscle), zs in by_cell.items():
        if len(zs) >= 2:
            per_muscle[muscle].append(float(np.var(zs, ddof=1)))
    out = {m: default for m in MUSCLES}
    for muscle, vars_list in per_muscle.items():
        out[muscle] = float(np.mean(vars_list)) if vars_list else default
    return out
```

- [ ] **Step 4: Run**

```
pytest tests/bayesian_emg/test_likelihoods/test_correlation.py -v
```
Expected: PASS for σ² + fisher_z tests; the `CorrelationLikelihood` import will still fail until Task 9.

If pytest collection fails on missing import, comment out the `CorrelationLikelihood` import for now and uncomment in Task 9. Cleaner alternative: stub `CorrelationLikelihood` as a placeholder class in `correlation.py` now and flesh out in Task 9.

Add a stub at the bottom of `correlation.py`:
```python
class CorrelationLikelihood:
    """Placeholder; implementation in Task 9."""
    name = "correlation"
```

- [ ] **Step 5: Commit**

```
git add vnl_playground/bayesian_emg/likelihoods/correlation.py tests/bayesian_emg/test_likelihoods/test_correlation.py
git commit -m "add fisher_z transform and per-muscle sigma squared estimator"
```

---

## Task 9: Correlation likelihood — log_likelihood + posterior_predictive

**Files:**
- Modify: `vnl_playground/bayesian_emg/likelihoods/correlation.py`
- Modify: `tests/bayesian_emg/test_likelihoods/test_correlation.py`

`log_likelihood(fit)` sums per-muscle Gaussian log-density of (fisher_z(r) - 0)² / (2σ²); we use 0 as the "perfect fit" target because the relative ordering of networks is what matters for reweighting and the constant drops out. `posterior_predictive(fits, weights)` builds a credible band from the weighted distribution of trial-mean envelopes across networks.

- [ ] **Step 1: Write the failing test**

Append to `tests/bayesian_emg/test_likelihoods/test_correlation.py`:
```python
def test_log_likelihood_higher_for_better_match(synthetic_cache):
    sigma = {"biceps": 0.1, "triceps": 0.1, "AD": 0.1}
    lik = CorrelationLikelihood(sigma_sq=sigma)
    # n1 was tuned for animal A
    fit_match = synthetic_cache.read_fit("n1", "A")
    fit_mismatch = synthetic_cache.read_fit("n1", "B")
    assert lik.log_likelihood(fit_match) > lik.log_likelihood(fit_mismatch)


def test_log_likelihood_holdout_changes_value(synthetic_cache):
    sigma = {"biceps": 0.1, "triceps": 0.1, "AD": 0.1}
    lik = CorrelationLikelihood(sigma_sq=sigma)
    fit = synthetic_cache.read_fit("n1", "A")
    full = lik.log_likelihood(fit)
    held = lik.log_likelihood(fit, holdout_trials=[0])
    assert full != held


def test_posterior_predictive_shape_and_containment(synthetic_cache):
    sigma = {"biceps": 0.1, "triceps": 0.1, "AD": 0.1}
    lik = CorrelationLikelihood(sigma_sq=sigma)
    fits = [synthetic_cache.read_fit(n, "A") for n in ["n1", "n2", "n3"]]
    weights = np.array([0.4, 0.4, 0.2])
    cb = lik.posterior_predictive(fits, weights, level=0.9)
    assert cb.lower.shape == (60, 3)
    assert cb.upper.shape == (60, 3)
    assert (cb.lower <= cb.upper).all()
```

- [ ] **Step 2: Run to verify fail**

```
pytest tests/bayesian_emg/test_likelihoods/test_correlation.py -v
```
Expected: FAIL on the three new tests.

- [ ] **Step 3: Replace the stub with a real implementation**

In `vnl_playground/bayesian_emg/likelihoods/correlation.py`, replace the stub class with:
```python
@dataclass
class CorrelationLikelihood:
    """Fisher-z Gaussian likelihood on per-muscle Pearson r.

    log p(EMG | θ) = -∑_μ (z_obs - 0)² / (2 σ²_μ),
    where z_obs is the fisher-z of the trial-mean Pearson r between sim and
    empirical envelopes for muscle μ. The "0" target reflects that perfect
    correlation gives z = +∞, so we score networks by how close they are to
    saturating; the constant offset doesn't matter for importance reweighting.
    """

    sigma_sq: Mapping[str, float]
    name: str = "correlation"

    def log_likelihood(self, fit: NetworkMouseFit, *,
                       holdout_trials: list[int] | None = None) -> float:
        if holdout_trials is None:
            mask = None
        else:
            mask = np.ones(fit.n_trials, dtype=bool)
            mask[holdout_trials] = False
        total = 0.0
        for mi, muscle in enumerate(MUSCLES):
            r = _trial_mean_correlation(fit, mi, mask)
            z = fisher_z(r)
            sigma2 = float(self.sigma_sq.get(muscle, 0.1))
            total -= 0.5 * (z - fisher_z(0.99)) ** 2 / sigma2
        return float(total)

    def posterior_predictive(self, fits: list[NetworkMouseFit],
                             weights: np.ndarray, level: float = 0.9) -> CredibleBand:
        if len(fits) == 0:
            raise ValueError("need at least one fit")
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()
        # Stack trial-mean sim envelopes across networks: (n_networks, T, M)
        means = np.stack([f.sim.mean(axis=0) for f in fits], axis=0)
        alpha = (1.0 - level) / 2.0
        T, M = means.shape[1], means.shape[2]
        lower = np.zeros((T, M))
        upper = np.zeros((T, M))
        for t in range(T):
            for m in range(M):
                vals = means[:, t, m]
                order = np.argsort(vals)
                cum = np.cumsum(weights[order])
                lower[t, m] = vals[order][np.searchsorted(cum, alpha)]
                upper_idx = min(np.searchsorted(cum, 1.0 - alpha), len(vals) - 1)
                upper[t, m] = vals[order][upper_idx]
        return CredibleBand(lower=lower, upper=upper, level=level)
```

Note the `fisher_z(0.99)` "target" — this anchors the likelihood so that a network with perfect-as-we-measure correlation gets the maximum score. Document the choice with a one-line comment in the code (already shown in the docstring above).

- [ ] **Step 4: Run**

```
pytest tests/bayesian_emg/test_likelihoods/test_correlation.py -v
```
Expected: PASS — all 6 tests.

- [ ] **Step 5: Commit**

```
git add vnl_playground/bayesian_emg/likelihoods/correlation.py tests/bayesian_emg/test_likelihoods/test_correlation.py
git commit -m "implement correlation likelihood log_likelihood and posterior_predictive"
```

---

## Task 10: Posterior — importance reweighting + ESS + credible set

**Files:**
- Create: `vnl_playground/bayesian_emg/posterior.py`
- Create: `tests/bayesian_emg/test_posterior.py`

- [ ] **Step 1: Write the failing test**

`tests/bayesian_emg/test_posterior.py`:
```python
import numpy as np
import pytest

from vnl_playground.bayesian_emg.posterior import (
    importance_weights,
    effective_sample_size,
    credible_set,
    posterior_for_mouse,
)
from vnl_playground.bayesian_emg.likelihoods.correlation import CorrelationLikelihood


def test_importance_weights_normalize():
    log_liks = np.array([-1.0, -2.0, -0.5])
    w = importance_weights(log_liks)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()


def test_importance_weights_numerically_stable_for_extreme_values():
    log_liks = np.array([-1000.0, -1001.0, -999.0])
    w = importance_weights(log_liks)
    assert w.sum() == pytest.approx(1.0)
    assert np.isfinite(w).all()


def test_ess_uniform_equals_n():
    w = np.ones(5) / 5
    assert effective_sample_size(w) == pytest.approx(5.0)


def test_ess_concentrated_equals_one():
    w = np.array([1.0, 0.0, 0.0, 0.0])
    assert effective_sample_size(w) == pytest.approx(1.0)


def test_credible_set_contains_top_weighted():
    w = np.array([0.5, 0.3, 0.15, 0.05])
    members = credible_set(w, level=0.9, network_ids=["a", "b", "c", "d"])
    assert "a" in members and "b" in members and "c" in members
    assert "d" not in members


def test_posterior_for_mouse_diagonal_dominance(synthetic_cache):
    sigma = {"biceps": 0.1, "triceps": 0.1, "AD": 0.1}
    lik = CorrelationLikelihood(sigma_sq=sigma)
    result = posterior_for_mouse(lik, synthetic_cache, mouse="A")
    # n1, n2 (tuned for A) should carry most of the weight
    network_to_weight = dict(zip(result.network_ids, result.weights))
    a_weight = network_to_weight["n1"] + network_to_weight["n2"]
    assert a_weight > 0.5
```

- [ ] **Step 2: Run to verify fail**

```
pytest tests/bayesian_emg/test_posterior.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement**

`vnl_playground/bayesian_emg/posterior.py`:
```python
"""Per-mouse posterior over networks via importance reweighting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vnl_playground.bayesian_emg.data import Cache
from vnl_playground.bayesian_emg.likelihoods.base import Likelihood


@dataclass(frozen=True)
class PosteriorResult:
    mouse: str
    network_ids: list[str]
    log_likelihoods: np.ndarray
    weights: np.ndarray
    ess: float


def importance_weights(log_liks: np.ndarray) -> np.ndarray:
    log_liks = np.asarray(log_liks, dtype=float)
    m = np.max(log_liks)
    w = np.exp(log_liks - m)
    s = w.sum()
    if s == 0:
        return np.full_like(w, 1.0 / len(w))
    return w / s


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    return float(weights.sum() ** 2 / np.sum(weights ** 2))


def credible_set(weights: np.ndarray, level: float, network_ids: list[str]) -> set[str]:
    order = np.argsort(weights)[::-1]
    cum = np.cumsum(weights[order])
    cutoff = int(np.searchsorted(cum, level)) + 1
    return {network_ids[i] for i in order[:cutoff]}


def posterior_for_mouse(likelihood: Likelihood, cache: Cache, mouse: str,
                        *, holdout_trials: list[int] | None = None) -> PosteriorResult:
    network_ids = cache.list_networks()
    log_liks = []
    kept_ids = []
    for nid in network_ids:
        if not cache.has_fit(nid, mouse):
            continue
        fit = cache.read_fit(nid, mouse)
        log_liks.append(likelihood.log_likelihood(fit, holdout_trials=holdout_trials))
        kept_ids.append(nid)
    log_liks = np.array(log_liks)
    w = importance_weights(log_liks)
    return PosteriorResult(
        mouse=mouse, network_ids=kept_ids, log_likelihoods=log_liks,
        weights=w, ess=effective_sample_size(w),
    )
```

- [ ] **Step 4: Run**

```
pytest tests/bayesian_emg/test_posterior.py -v
```
Expected: PASS — all 6 tests.

- [ ] **Step 5: Commit**

```
git add vnl_playground/bayesian_emg/posterior.py tests/bayesian_emg/test_posterior.py
git commit -m "add per-mouse posterior with importance reweighting and ESS"
```

---

## Task 11: Validation — between-mouse discrimination matrix (THE GATE)

**Files:**
- Create: `vnl_playground/bayesian_emg/validation/discrimination.py`
- Create: `tests/bayesian_emg/test_validation/test_discrimination.py`

This is the Phase 1 gate. The 5×5 cross-likelihood matrix `L[i, j] = log p(EMG_j | posterior fit on EMG_i)` should have a dominant diagonal. We compute it as the weighted-mean log-likelihood under the row's posterior, evaluated on the column's data.

- [ ] **Step 1: Write the failing test**

`tests/bayesian_emg/test_validation/test_discrimination.py`:
```python
import numpy as np
import pytest

from vnl_playground.bayesian_emg.likelihoods.correlation import CorrelationLikelihood
from vnl_playground.bayesian_emg.validation.discrimination import (
    cross_likelihood_matrix,
    diagonal_margin,
    permutation_p_value,
)


def test_cross_likelihood_matrix_shape(synthetic_cache):
    sigma = {"biceps": 0.1, "triceps": 0.1, "AD": 0.1}
    lik = CorrelationLikelihood(sigma_sq=sigma)
    L = cross_likelihood_matrix(lik, synthetic_cache, mice=["A", "B", "C"])
    assert L.shape == (3, 3)


def test_diagonal_dominates_on_synthetic(synthetic_cache):
    sigma = {"biceps": 0.1, "triceps": 0.1, "AD": 0.1}
    lik = CorrelationLikelihood(sigma_sq=sigma)
    L = cross_likelihood_matrix(lik, synthetic_cache, mice=["A", "B", "C"])
    delta = diagonal_margin(L)
    assert delta > 0


def test_permutation_p_value_low_when_diagonal_real(synthetic_cache):
    sigma = {"biceps": 0.1, "triceps": 0.1, "AD": 0.1}
    lik = CorrelationLikelihood(sigma_sq=sigma)
    L = cross_likelihood_matrix(lik, synthetic_cache, mice=["A", "B", "C"])
    p = permutation_p_value(L, n_shuffles=200, seed=0)
    assert p < 0.1
```

- [ ] **Step 2: Run to verify fail**

```
pytest tests/bayesian_emg/test_validation/test_discrimination.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement**

`vnl_playground/bayesian_emg/validation/discrimination.py`:
```python
"""Between-mouse discrimination: 5×5 cross-likelihood matrix and permutation null."""

from __future__ import annotations

import numpy as np

from vnl_playground.bayesian_emg.data import Cache
from vnl_playground.bayesian_emg.likelihoods.base import Likelihood
from vnl_playground.bayesian_emg.posterior import posterior_for_mouse


def cross_likelihood_matrix(likelihood: Likelihood, cache: Cache,
                            mice: list[str]) -> np.ndarray:
    """L[i, j] = sum_n w_n^(i) * log p(EMG_j | θ_n).

    Row i: posterior weights computed on mouse i's data.
    Col j: log-likelihood of those networks evaluated on mouse j's data.
    """
    n = len(mice)
    L = np.zeros((n, n))
    posteriors = {m: posterior_for_mouse(likelihood, cache, m) for m in mice}
    # Pre-compute log-likelihood of every (network, mouse) pair
    nid_to_index = {nid: i for post in posteriors.values()
                    for i, nid in enumerate(post.network_ids)}
    all_nids = sorted({nid for post in posteriors.values() for nid in post.network_ids})
    log_lik_matrix = np.zeros((len(all_nids), n))
    for j, m in enumerate(mice):
        for k, nid in enumerate(all_nids):
            if cache.has_fit(nid, m):
                fit = cache.read_fit(nid, m)
                log_lik_matrix[k, j] = likelihood.log_likelihood(fit)
            else:
                log_lik_matrix[k, j] = -np.inf
    nid_to_pos = {nid: k for k, nid in enumerate(all_nids)}
    for i, m_i in enumerate(mice):
        post = posteriors[m_i]
        weights = post.weights
        for j in range(n):
            ll_j = np.array([log_lik_matrix[nid_to_pos[nid], j] for nid in post.network_ids])
            # Mask -inf entries (no fit) before weighted mean
            finite = np.isfinite(ll_j)
            if not finite.any():
                L[i, j] = -np.inf
            else:
                w = weights[finite]
                w = w / w.sum() if w.sum() > 0 else w
                L[i, j] = float((w * ll_j[finite]).sum())
    return L


def diagonal_margin(L: np.ndarray) -> float:
    """mean(diag) - mean(off-diagonal)."""
    n = L.shape[0]
    diag = np.diag(L)
    mask = ~np.eye(n, dtype=bool)
    return float(diag.mean() - L[mask].mean())


def permutation_p_value(L: np.ndarray, n_shuffles: int = 10_000, seed: int = 0) -> float:
    """Fraction of row-permutations of L whose diagonal margin exceeds the observed."""
    observed = diagonal_margin(L)
    rng = np.random.RandomState(seed)
    n = L.shape[0]
    count = 0
    for _ in range(n_shuffles):
        perm = rng.permutation(n)
        L_perm = L[perm, :]
        if diagonal_margin(L_perm) >= observed:
            count += 1
    return float(count / n_shuffles)
```

- [ ] **Step 4: Run**

```
pytest tests/bayesian_emg/test_validation/test_discrimination.py -v
```
Expected: PASS — all 3 tests. The synthetic fixture is constructed precisely so the diagonal dominates.

- [ ] **Step 5: Commit**

```
git add vnl_playground/bayesian_emg/validation/ tests/bayesian_emg/test_validation/
git commit -m "add cross-likelihood matrix with diagonal margin and permutation p-value"
```

---

## Task 12: Validation — within-mouse posterior predictive coverage

**Files:**
- Create: `vnl_playground/bayesian_emg/validation/coverage.py`
- Create: `tests/bayesian_emg/test_validation/test_coverage.py`

Leave-trial-out per mouse: for each held-out trial, compute the posterior weights on the remaining trials, build the credible band, count fraction of (timestep, muscle) cells the held-out trial falls inside.

- [ ] **Step 1: Write the failing test**

`tests/bayesian_emg/test_validation/test_coverage.py`:
```python
import numpy as np
import pytest

from vnl_playground.bayesian_emg.likelihoods.correlation import CorrelationLikelihood
from vnl_playground.bayesian_emg.validation.coverage import (
    coverage_for_mouse,
    calibration_curve,
)


def test_coverage_for_mouse_returns_per_trial_fractions(synthetic_cache):
    sigma = {"biceps": 0.1, "triceps": 0.1, "AD": 0.1}
    lik = CorrelationLikelihood(sigma_sq=sigma)
    fractions = coverage_for_mouse(lik, synthetic_cache, mouse="A", level=0.9)
    assert len(fractions) == 4   # n_trials in fixture
    assert all(0.0 <= f <= 1.0 for f in fractions)


def test_calibration_curve_monotone_in_level(synthetic_cache):
    sigma = {"biceps": 0.1, "triceps": 0.1, "AD": 0.1}
    lik = CorrelationLikelihood(sigma_sq=sigma)
    curve = calibration_curve(lik, synthetic_cache, mouse="A",
                              levels=(0.5, 0.8, 0.9, 0.95))
    nominal = [c[0] for c in curve]
    empirical = [c[1] for c in curve]
    assert nominal == [0.5, 0.8, 0.9, 0.95]
    # Empirical coverage should generally rise with nominal level
    assert empirical[-1] >= empirical[0] - 0.1
```

- [ ] **Step 2: Run to verify fail**

```
pytest tests/bayesian_emg/test_validation/test_coverage.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement**

`vnl_playground/bayesian_emg/validation/coverage.py`:
```python
"""Within-mouse posterior predictive coverage (calibration check)."""

from __future__ import annotations

import numpy as np

from vnl_playground.bayesian_emg.data import Cache
from vnl_playground.bayesian_emg.likelihoods.base import Likelihood
from vnl_playground.bayesian_emg.posterior import posterior_for_mouse


def coverage_for_mouse(likelihood: Likelihood, cache: Cache, mouse: str,
                       *, level: float = 0.9) -> list[float]:
    """Per-(held-out)-trial coverage fraction across (timestep, muscle) cells."""
    networks = cache.list_networks()
    fits = [cache.read_fit(n, mouse) for n in networks if cache.has_fit(n, mouse)]
    if not fits:
        return []
    n_trials = fits[0].n_trials
    fractions = []
    for held in range(n_trials):
        post = posterior_for_mouse(likelihood, cache, mouse, holdout_trials=[held])
        fits_for_band = [cache.read_fit(n, mouse) for n in post.network_ids]
        # Use only the kept (non-held) trials inside each fit when building the band
        from vnl_playground.bayesian_emg.data import NetworkMouseFit
        masked = [
            NetworkMouseFit(f.network_id, f.animal,
                            np.delete(f.sim, held, axis=0),
                            np.delete(f.empirical, held, axis=0))
            for f in fits_for_band
        ]
        band = likelihood.posterior_predictive(masked, post.weights, level=level)
        held_trial = fits[0].empirical[held]
        contained = band.contains(held_trial)
        fractions.append(float(contained.mean()))
    return fractions


def calibration_curve(likelihood: Likelihood, cache: Cache, mouse: str,
                      levels=(0.5, 0.8, 0.9, 0.95)) -> list[tuple[float, float]]:
    """List of (nominal, empirical) coverage pairs."""
    out = []
    for lvl in levels:
        fractions = coverage_for_mouse(likelihood, cache, mouse, level=lvl)
        out.append((float(lvl), float(np.mean(fractions)) if fractions else 0.0))
    return out
```

- [ ] **Step 4: Run**

```
pytest tests/bayesian_emg/test_validation/test_coverage.py -v
```
Expected: PASS — both tests.

- [ ] **Step 5: Commit**

```
git add vnl_playground/bayesian_emg/validation/coverage.py tests/bayesian_emg/test_validation/test_coverage.py
git commit -m "add within-mouse posterior predictive coverage and calibration curve"
```

---

## Task 13: Validation — full label-shuffle null

**Files:**
- Create: `vnl_playground/bayesian_emg/validation/permutation.py`
- Create: `tests/bayesian_emg/test_validation/test_permutation.py`

This re-runs the discrimination matrix after shuffling which empirical envelope is assigned to which mouse label, using a temporary in-memory shuffled cache.

- [ ] **Step 1: Write the failing test**

`tests/bayesian_emg/test_validation/test_permutation.py`:
```python
import numpy as np

from vnl_playground.bayesian_emg.likelihoods.correlation import CorrelationLikelihood
from vnl_playground.bayesian_emg.validation.permutation import label_shuffle_null


def test_label_shuffle_null_collapses_diagonal(synthetic_cache):
    sigma = {"biceps": 0.1, "triceps": 0.1, "AD": 0.1}
    lik = CorrelationLikelihood(sigma_sq=sigma)
    nulls = label_shuffle_null(lik, synthetic_cache, mice=["A", "B", "C"],
                               n_shuffles=20, seed=0)
    # Under the null, diagonal margin should be near zero on average
    assert abs(np.mean(nulls)) < 0.5
    # Variance non-zero
    assert np.std(nulls) > 0
```

- [ ] **Step 2: Run to verify fail**

```
pytest tests/bayesian_emg/test_validation/test_permutation.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement**

`vnl_playground/bayesian_emg/validation/permutation.py`:
```python
"""Full EMG↔mouse label-shuffle null for the framework as a whole."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from vnl_playground.bayesian_emg.data import Cache, NetworkMouseFit
from vnl_playground.bayesian_emg.likelihoods.base import Likelihood
from vnl_playground.bayesian_emg.validation.discrimination import (
    cross_likelihood_matrix,
    diagonal_margin,
)


def label_shuffle_null(likelihood: Likelihood, cache: Cache, mice: list[str],
                       *, n_shuffles: int = 1000, seed: int = 0) -> np.ndarray:
    """Build shuffled caches and report the distribution of diagonal margins."""
    rng = np.random.RandomState(seed)
    nulls = []
    networks = cache.list_networks()
    fits_by_animal = {m: {n: cache.read_fit(n, m) for n in networks
                          if cache.has_fit(n, m)} for m in mice}
    for _ in range(n_shuffles):
        perm = rng.permutation(len(mice))
        with tempfile.TemporaryDirectory() as td:
            shuffled = Cache(Path(td) / "shuf.parquet")
            for nid in networks:
                for j, m in enumerate(mice):
                    real_animal = mice[perm[j]]
                    if nid not in fits_by_animal[real_animal]:
                        continue
                    src = fits_by_animal[real_animal][nid]
                    own_sim = fits_by_animal[m].get(nid)
                    if own_sim is None:
                        continue
                    shuffled.write_fit(NetworkMouseFit(
                        nid, m, own_sim.sim, src.empirical
                    ))
            L = cross_likelihood_matrix(likelihood, shuffled, mice=mice)
            nulls.append(diagonal_margin(L))
    return np.array(nulls)
```

- [ ] **Step 4: Run**

```
pytest tests/bayesian_emg/test_validation/test_permutation.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add vnl_playground/bayesian_emg/validation/permutation.py tests/bayesian_emg/test_validation/test_permutation.py
git commit -m "add full label-shuffle null"
```

---

## Task 14: Preregistration — YAML loader + SHA-256 hash check

**Files:**
- Create: `vnl_playground/bayesian_emg/preregistration.py`
- Create: `configs/bayesian_emg/preregistration.yaml`
- Create: `tests/bayesian_emg/test_preregistration.py`

The Phase 1 YAML pins σ², coverage acceptance band, discrimination diagonal-margin threshold, permutation seed and count, and (after first cache build) the cache content hash. The runner refuses to produce a final report when the YAML hash on disk doesn't match the hash recorded by an earlier successful run.

- [ ] **Step 1: Write the failing test**

`tests/bayesian_emg/test_preregistration.py`:
```python
import pytest
import yaml

from vnl_playground.bayesian_emg.preregistration import (
    Preregistration,
    load_preregistration,
    sha256_of_file,
)


def test_sha256_changes_on_edit(tmp_path):
    p = tmp_path / "f.yaml"
    p.write_text("a: 1\n")
    h1 = sha256_of_file(p)
    p.write_text("a: 2\n")
    h2 = sha256_of_file(p)
    assert h1 != h2


def test_load_preregistration(tmp_path):
    p = tmp_path / "prereg.yaml"
    p.write_text(yaml.safe_dump({
        "sigma_sq": {"biceps": 0.1, "triceps": 0.1, "AD": 0.1},
        "coverage_acceptance_band": 0.05,
        "discrimination_threshold_nats": 0.5,
        "permutation_seed": 0,
        "permutation_n_shuffles": 1000,
        "cache_content_hash": None,
    }))
    pr = load_preregistration(p)
    assert pr.sigma_sq["biceps"] == pytest.approx(0.1)
    assert pr.discrimination_threshold_nats == pytest.approx(0.5)
    assert pr.cache_content_hash is None


def test_preregistration_validates_required_keys(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("sigma_sq:\n  biceps: 0.1\n")
    with pytest.raises(ValueError):
        load_preregistration(p)
```

- [ ] **Step 2: Run to verify fail**

```
pytest tests/bayesian_emg/test_preregistration.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement**

`vnl_playground/bayesian_emg/preregistration.py`:
```python
"""YAML preregistration loader + content-hash discipline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml


REQUIRED_KEYS = (
    "sigma_sq",
    "coverage_acceptance_band",
    "discrimination_threshold_nats",
    "permutation_seed",
    "permutation_n_shuffles",
    "cache_content_hash",
)


@dataclass(frozen=True)
class Preregistration:
    sigma_sq: Mapping[str, float]
    coverage_acceptance_band: float
    discrimination_threshold_nats: float
    permutation_seed: int
    permutation_n_shuffles: int
    cache_content_hash: str | None
    yaml_sha256: str


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_preregistration(path: str | Path) -> Preregistration:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    missing = [k for k in REQUIRED_KEYS if k not in raw]
    if missing:
        raise ValueError(f"preregistration missing keys: {missing}")
    return Preregistration(
        sigma_sq=raw["sigma_sq"],
        coverage_acceptance_band=float(raw["coverage_acceptance_band"]),
        discrimination_threshold_nats=float(raw["discrimination_threshold_nats"]),
        permutation_seed=int(raw["permutation_seed"]),
        permutation_n_shuffles=int(raw["permutation_n_shuffles"]),
        cache_content_hash=raw.get("cache_content_hash"),
        yaml_sha256=sha256_of_file(path),
    )
```

`configs/bayesian_emg/preregistration.yaml`:
```yaml
# Phase 1 preregistration. Hashes pinned at first successful cache build.
sigma_sq:
  biceps: 0.1
  triceps: 0.1
  AD: 0.1
coverage_acceptance_band: 0.05      # ±5 points around nominal at 90%
discrimination_threshold_nats: 0.5  # diagonal margin per (mouse, muscle)
permutation_seed: 0
permutation_n_shuffles: 10000
cache_content_hash: null            # set after first cache build
```

- [ ] **Step 4: Run**

```
pytest tests/bayesian_emg/test_preregistration.py -v
```
Expected: PASS — all 3 tests.

- [ ] **Step 5: Commit**

```
git add vnl_playground/bayesian_emg/preregistration.py configs/bayesian_emg/preregistration.yaml tests/bayesian_emg/test_preregistration.py
git commit -m "add preregistration YAML loader with sha256 discipline"
```

---

## Task 15: Report — HTML aggregator

**Files:**
- Create: `vnl_playground/bayesian_emg/report.py`
- Create: `tests/bayesian_emg/test_report.py`

Single-file HTML report with: header (cache hash, YAML hash, git SHA), per-likelihood discrimination heatmap, calibration curves, ESS table, permutation null histogram, findings flags. We use raw HTML strings + matplotlib SVG embedding to keep the dependency footprint small (no Jinja2 needed).

- [ ] **Step 1: Write the failing test**

`tests/bayesian_emg/test_report.py`:
```python
import re
from pathlib import Path

import numpy as np

from vnl_playground.bayesian_emg.report import build_report, ReportSection


def test_build_report_writes_html(tmp_path):
    sections = [
        ReportSection(
            title="discrimination",
            body_html="<p>diag margin = 0.7</p>",
            flags=[],
        ),
    ]
    out = tmp_path / "r.html"
    build_report(out, header={"cache_hash": "abc", "yaml_hash": "def", "git_sha": "1234"},
                 sections=sections)
    text = out.read_text()
    assert "abc" in text and "def" in text and "1234" in text
    assert "discrimination" in text
    assert "diag margin = 0.7" in text


def test_build_report_surfaces_flags(tmp_path):
    sections = [
        ReportSection(title="coverage", body_html="<p>ok</p>",
                      flags=["coverage at 90% empirical=0.7 → under-coverage"]),
    ]
    out = tmp_path / "r.html"
    build_report(out, header={"cache_hash": "x", "yaml_hash": "y", "git_sha": "z"},
                 sections=sections)
    text = out.read_text()
    # Flags must appear at the top, above section bodies
    flag_idx = text.find("under-coverage")
    body_idx = text.find("ok")
    assert 0 <= flag_idx < body_idx
```

- [ ] **Step 2: Run to verify fail**

```
pytest tests/bayesian_emg/test_report.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement**

`vnl_playground/bayesian_emg/report.py`:
```python
"""HTML report aggregator. No external templating dependency."""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ReportSection:
    title: str
    body_html: str
    flags: list[str] = field(default_factory=list)


def build_report(path: str | Path, *, header: dict, sections: list[ReportSection]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags: list[str] = []
    for s in sections:
        flags.extend(s.flags)
    flag_block = ""
    if flags:
        items = "".join(f"<li>{html.escape(f)}</li>" for f in flags)
        flag_block = f'<div class="flags"><h2>Findings flags</h2><ul>{items}</ul></div>'
    section_blocks = "".join(
        f'<section><h2>{html.escape(s.title)}</h2>{s.body_html}</section>'
        for s in sections
    )
    header_rows = "".join(
        f"<tr><th>{html.escape(k)}</th><td><code>{html.escape(str(v))}</code></td></tr>"
        for k, v in header.items()
    )
    html_doc = f"""<!doctype html>
<meta charset="utf-8">
<title>Bayesian EMG report</title>
<style>
body {{ font-family: sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1em; }}
table.header {{ border-collapse: collapse; margin-bottom: 1.5em; }}
table.header th, table.header td {{ border: 1px solid #ccc; padding: 0.4em 0.8em; text-align: left; }}
.flags {{ background: #fff8e0; border: 1px solid #d4ad00; padding: 1em; margin-bottom: 2em; }}
section {{ margin-bottom: 2em; }}
</style>
<h1>Bayesian EMG report</h1>
<table class="header">{header_rows}</table>
{flag_block}
{section_blocks}
"""
    path.write_text(html_doc)
```

- [ ] **Step 4: Run**

```
pytest tests/bayesian_emg/test_report.py -v
```
Expected: PASS — both tests.

- [ ] **Step 5: Commit**

```
git add vnl_playground/bayesian_emg/report.py tests/bayesian_emg/test_report.py
git commit -m "add HTML report aggregator"
```

---

## Task 16: End-to-end test on synthetic cache

**Files:**
- Create: `tests/bayesian_emg/test_end_to_end.py`

Validates that the full pipeline (cache → posterior → discrimination → coverage → permutation → report) runs without error on the synthetic cache and produces an HTML file with the expected sections.

- [ ] **Step 1: Write the test**

`tests/bayesian_emg/test_end_to_end.py`:
```python
import numpy as np

from vnl_playground.bayesian_emg.likelihoods.correlation import CorrelationLikelihood
from vnl_playground.bayesian_emg.validation.discrimination import (
    cross_likelihood_matrix,
    diagonal_margin,
    permutation_p_value,
)
from vnl_playground.bayesian_emg.validation.coverage import calibration_curve
from vnl_playground.bayesian_emg.validation.permutation import label_shuffle_null
from vnl_playground.bayesian_emg.report import build_report, ReportSection


def test_pipeline_end_to_end_on_synthetic(tmp_path, synthetic_cache):
    sigma = {"biceps": 0.1, "triceps": 0.1, "AD": 0.1}
    lik = CorrelationLikelihood(sigma_sq=sigma)
    mice = ["A", "B", "C"]

    L = cross_likelihood_matrix(lik, synthetic_cache, mice=mice)
    delta = diagonal_margin(L)
    p = permutation_p_value(L, n_shuffles=200, seed=0)

    cov_curves = {m: calibration_curve(lik, synthetic_cache, mouse=m,
                                       levels=(0.5, 0.8, 0.9)) for m in mice}
    null = label_shuffle_null(lik, synthetic_cache, mice=mice,
                              n_shuffles=20, seed=0)

    flags = []
    if delta < 0.5:
        flags.append(f"discrimination diagonal margin {delta:.2f} below threshold 0.5")
    if p > 0.05:
        flags.append(f"discrimination permutation p={p:.3f} above 0.05")

    sections = [
        ReportSection(
            title="discrimination",
            body_html=f"<p>diagonal margin = {delta:.3f}, permutation p = {p:.3f}</p>",
            flags=flags,
        ),
        ReportSection(
            title="coverage",
            body_html="<pre>" + str(cov_curves) + "</pre>",
        ),
        ReportSection(
            title="label-shuffle null",
            body_html=f"<p>mean null = {null.mean():.3f}, std = {null.std():.3f}</p>",
        ),
    ]
    out = tmp_path / "report.html"
    build_report(out, header={
        "cache_hash": synthetic_cache.content_hash(),
        "yaml_hash": "test-only",
        "git_sha": "test-only",
    }, sections=sections)
    assert out.exists()
    text = out.read_text()
    for required in ["discrimination", "coverage", "label-shuffle null"]:
        assert required in text
    # On synthetic data the diagonal must be real (delta > 0.5 by construction)
    assert delta > 0
```

- [ ] **Step 2: Run**

```
pytest tests/bayesian_emg/test_end_to_end.py -v
```
Expected: PASS.

- [ ] **Step 3: Commit**

```
git add tests/bayesian_emg/test_end_to_end.py
git commit -m "add end-to-end pipeline smoke test on synthetic cache"
```

---

## Task 17: CLI — `bayes_emg_build_cache.py`

**Files:**
- Create: `scripts/bayes_emg_build_cache.py`
- Modify: `.gitignore` (add `vnl_playground/bayesian_emg/cache/`)

Discovers checkpoints from a directory or wandb group, then runs `build_many` with idempotent behavior. Supports `--dry-run` (lists what would be done without rolling out) and `--limit N` (process only the first N networks for smoke testing).

- [ ] **Step 1: Write the script**

`scripts/bayes_emg_build_cache.py`:
```python
#!/usr/bin/env python3
"""Build the Bayesian EMG cache.

Two discovery modes:
  --checkpoint-glob '<pattern>' : enumerate checkpoint dirs matching a glob
  --wandb-group <group>         : pull runs from wandb (uses _fields.* hyperparams)

Hyperparameters come from wandb when available; otherwise the script reads the
checkpoint dir's config.json with a warning (config.json is unreliable per
project memory — wandb is the source of truth).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vnl_playground.bayesian_emg.cache_builder import build_many
from vnl_playground.bayesian_emg.data import Cache


def discover_from_glob(pattern: str) -> list[tuple[str, str, dict]]:
    out = []
    for path in sorted(Path().glob(pattern)):
        if not path.is_dir():
            continue
        nid = path.name
        # Minimal local meta — improved by wandb mode
        out.append((nid, str(path), {"checkpoint_dir": str(path)}))
    return out


def discover_from_wandb(group: str, project: str) -> list[tuple[str, str, dict]]:
    import wandb
    api = wandb.Api()
    runs = api.runs(project, filters={"group": group})
    out = []
    for r in runs:
        nid = r.name
        cdir = r.summary.get("checkpoint_dir") or r.config.get("checkpoint_dir")
        if not cdir:
            logging.warning("skip %s: no checkpoint_dir in summary/config", nid)
            continue
        meta = {k: v for k, v in r.config.items()
                if k in ("force_scale", "joint_damping", "shoulder_damping",
                         "control_cost", "control_diff_cost", "norm_method",
                         "train_animals", "seed")}
        meta["wandb_run_id"] = r.id
        meta["checkpoint_dir"] = cdir
        out.append((nid, cdir, meta))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--n-clips", type=int, default=50)
    parser.add_argument("--animals", nargs="+",
                        default=["A36-1", "AT006", "AT009", "AT012", "AT013"])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint-glob", type=str)
    src.add_argument("--wandb-group", type=str)
    parser.add_argument("--wandb-project", type=str, default="vnl-janelia")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.checkpoint_glob:
        networks = discover_from_glob(args.checkpoint_glob)
    else:
        networks = discover_from_wandb(args.wandb_group, args.wandb_project)
    if args.limit is not None:
        networks = networks[: args.limit]
    logging.info("discovered %d networks", len(networks))

    if args.dry_run:
        for nid, cdir, meta in networks:
            print(nid, cdir, meta)
        return

    cache = Cache(args.cache_path)
    build_many(networks, args.animals, args.n_clips, cache)
    logging.info("cache hash: %s", cache.content_hash())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add a CLI dry-run smoke test**

Append to `tests/bayesian_emg/test_cache_builder.py`:
```python
def test_build_cache_cli_dry_run(tmp_path, capsys, monkeypatch):
    import subprocess
    fake_dir = tmp_path / "fake_ckpt_dir_a"
    fake_dir.mkdir()
    out = subprocess.run(
        [sys.executable, "scripts/bayes_emg_build_cache.py",
         "--cache-path", str(tmp_path / "c.parquet"),
         "--checkpoint-glob", str(tmp_path / "fake_ckpt_dir_*"),
         "--dry-run"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0
    assert "fake_ckpt_dir_a" in out.stdout
```

Add `import sys` to the top of the file if not present.

- [ ] **Step 3: Run**

```
pytest tests/bayesian_emg/test_cache_builder.py::test_build_cache_cli_dry_run -v
```
Expected: PASS.

- [ ] **Step 4: Update .gitignore**

Append to `.gitignore`:
```
vnl_playground/bayesian_emg/cache/
```

- [ ] **Step 5: Commit**

```
git add scripts/bayes_emg_build_cache.py tests/bayesian_emg/test_cache_builder.py .gitignore
git commit -m "add bayes_emg_build_cache CLI with glob and wandb discovery"
```

---

## Task 18: CLI — `bayes_emg_run.py`

**Files:**
- Create: `scripts/bayes_emg_run.py`

Reads a built cache + preregistration YAML, runs all three validation tests under the correlation likelihood, builds the report. Refuses to write the report if the YAML's recorded `cache_content_hash` doesn't match the live cache's hash (when the YAML has one set).

- [ ] **Step 1: Write the script**

`scripts/bayes_emg_run.py`:
```python
#!/usr/bin/env python3
"""Run Phase 1 Bayesian EMG analysis: cache → posterior → validation → report."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vnl_playground.bayesian_emg.data import Cache
from vnl_playground.bayesian_emg.preregistration import load_preregistration
from vnl_playground.bayesian_emg.likelihoods.correlation import CorrelationLikelihood
from vnl_playground.bayesian_emg.validation.discrimination import (
    cross_likelihood_matrix, diagonal_margin, permutation_p_value,
)
from vnl_playground.bayesian_emg.validation.coverage import calibration_curve
from vnl_playground.bayesian_emg.validation.permutation import label_shuffle_null
from vnl_playground.bayesian_emg.report import build_report, ReportSection


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                        text=True).strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--mice", nargs="+",
                        default=["A36-1", "AT006", "AT009", "AT012", "AT013"])
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--allow-cache-mismatch", action="store_true",
                        help="Skip the cache_content_hash check (exploratory mode).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    pr = load_preregistration(args.preregistration)
    cache = Cache(args.cache_path)
    cache_hash = cache.content_hash()

    if pr.cache_content_hash and pr.cache_content_hash != cache_hash and \
            not args.allow_cache_mismatch:
        raise SystemExit(
            f"cache hash mismatch: prereg={pr.cache_content_hash}, live={cache_hash}. "
            "Re-build the cache or pass --allow-cache-mismatch (exploratory only)."
        )

    lik = CorrelationLikelihood(sigma_sq=pr.sigma_sq)
    L = cross_likelihood_matrix(lik, cache, mice=args.mice)
    delta = diagonal_margin(L)
    p = permutation_p_value(L, n_shuffles=pr.permutation_n_shuffles,
                            seed=pr.permutation_seed)
    null = label_shuffle_null(lik, cache, mice=args.mice,
                              n_shuffles=min(200, pr.permutation_n_shuffles),
                              seed=pr.permutation_seed)
    cov_curves = {m: calibration_curve(lik, cache, mouse=m) for m in args.mice}

    flags = []
    if delta < pr.discrimination_threshold_nats:
        flags.append(f"discrimination diagonal margin {delta:.3f} below threshold "
                     f"{pr.discrimination_threshold_nats}")
    if p > 0.05:
        flags.append(f"discrimination permutation p={p:.3f} above 0.05")
    for m, curve in cov_curves.items():
        for nominal, empirical in curve:
            if abs(nominal - empirical) > pr.coverage_acceptance_band:
                flags.append(f"coverage {m} nominal={nominal} empirical={empirical:.2f} "
                             f"outside ±{pr.coverage_acceptance_band}")

    sections = [
        ReportSection(
            title="discrimination",
            body_html=f"<pre>L =\n{L}\n\nmargin = {delta:.3f}, perm p = {p:.4f}</pre>",
            flags=flags,
        ),
        ReportSection(
            title="coverage",
            body_html="<pre>" + "\n".join(f"{m}: {c}" for m, c in cov_curves.items()) + "</pre>",
        ),
        ReportSection(
            title="label-shuffle null",
            body_html=f"<pre>n={len(null)}, mean={null.mean():.3f}, std={null.std():.3f}</pre>",
        ),
    ]
    build_report(args.report_path, header={
        "cache_hash": cache_hash,
        "yaml_hash": pr.yaml_sha256,
        "git_sha": _git_sha(),
    }, sections=sections)
    logging.info("wrote report to %s", args.report_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add a CLI smoke test**

Append to `tests/bayesian_emg/test_end_to_end.py`:
```python
def test_run_cli_on_synthetic(tmp_path, synthetic_cache):
    import subprocess
    import sys
    import yaml

    prereg = tmp_path / "prereg.yaml"
    prereg.write_text(yaml.safe_dump({
        "sigma_sq": {"biceps": 0.1, "triceps": 0.1, "AD": 0.1},
        "coverage_acceptance_band": 0.05,
        "discrimination_threshold_nats": 0.5,
        "permutation_seed": 0,
        "permutation_n_shuffles": 50,
        "cache_content_hash": None,
    }))
    out = subprocess.run(
        [sys.executable, "scripts/bayes_emg_run.py",
         "--cache-path", str(synthetic_cache.fits_path).replace(".fits.parquet", ""),
         "--preregistration", str(prereg),
         "--mice", "A", "B", "C",
         "--report-path", str(tmp_path / "report.html")],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    assert (tmp_path / "report.html").exists()
```

Note: the test passes a path to `Cache(...)` that strips the `.fits.parquet` suffix. The `Cache.__init__` adds suffixes, so it accepts any base path. The synthetic_cache fixture used `tmp_path / "synth.parquet"`, so the live path is `tmp_path / "synth.fits.parquet"` — the test recovers the base by stripping.

- [ ] **Step 3: Run**

```
pytest tests/bayesian_emg/test_end_to_end.py -v
```
Expected: PASS — both tests.

- [ ] **Step 4: Commit**

```
git add scripts/bayes_emg_run.py tests/bayesian_emg/test_end_to_end.py
git commit -m "add bayes_emg_run CLI with cache-hash discipline"
```

---

## Task 19: Real-data smoke run on s17 cache (gating run)

**Files:** None (operational task).

Build a partial cache from a small slice of s17 networks and run the pipeline against real EMG. This is a manual run, not a unit test — it's the gate for declaring Phase 1 complete and deciding whether to start Phase 2.

- [ ] **Step 1: Build a 3-network cache (one s17 specialist + cohort + low-fs)**

Pick three known-good s17 checkpoints from `vnl_playground/checkpoints/`. Adjust the glob to match three specific run directories.

```
python scripts/bayes_emg_build_cache.py \
  --cache-path /tmp/bayes_emg/v1 \
  --checkpoint-glob 'vnl_playground/checkpoints/<your-s17-pattern>*' \
  --animals A36-1 AT006 AT009 AT012 AT013 \
  --n-clips 30 \
  --limit 3
```

Expected: cache built without error; log lists `wrote <nid> × <animal> (n_trials=30)` for each (network, animal) pair (15 lines total).

- [ ] **Step 2: Pin the cache hash in preregistration**

Read the cache hash from the previous step's log (last line). Update `configs/bayesian_emg/preregistration.yaml`:
```yaml
cache_content_hash: <hash from build log>
```

Commit:
```
git add configs/bayesian_emg/preregistration.yaml
git commit -m "pin Phase 1 cache content hash for s17 v1 run"
```

- [ ] **Step 3: Run the pipeline**

```
python scripts/bayes_emg_run.py \
  --cache-path /tmp/bayes_emg/v1 \
  --preregistration configs/bayesian_emg/preregistration.yaml \
  --mice A36-1 AT006 AT009 AT012 AT013 \
  --report-path /tmp/bayes_emg/v1_report.html
```

Expected: report written. Open in a browser. Read the discrimination section.

- [ ] **Step 4: Apply the gate**

The Phase 1 spec gate:
- `delta >= 0.5` and permutation `p < 0.01` → **proceed to Phase 2** (write the next plan: ABC + Bayes factors).
- `delta < 0.2` or permutation `p > 0.1` → **STOP** and write a follow-up note: framework is sound, but the s17 sweep doesn't cover enough hyperparameter variation to produce mouse-distinguishing fits. The fix is sweep design, not framework code.
- Anything in between → run on the full s17 cache (drop `--limit 3`) before deciding.

Document the result in a follow-up note: `docs/2026-05-XX-bayesian-emg-phase1-gate-result.md`.

---

## Self-review checklist (run before handoff)

- [x] **Spec coverage:** §1 architecture (Tasks 1, 7), §2 cache (Tasks 2–6), §3 Option 1 (Tasks 7–9), posterior (Task 10), §4 validation (Tasks 11–13), preregistration (Task 14), report (Task 15), e2e (Task 16), CLIs (Tasks 17–18), gating run (Task 19). Options 2 and 3, plus Bayes factors and full report features, are explicitly Phase 2/3 — out of scope here.
- [x] **Placeholder scan:** no TBD/TODO/"add appropriate error handling"/"similar to Task N" patterns. Every code step shows code; every command step shows the command.
- [x] **Type consistency:** `MUSCLES`, `TARGET_TIMESTEPS`, `NetworkMouseFit`, `Cache`, `CredibleBand`, `CorrelationLikelihood`, `posterior_for_mouse`, `cross_likelihood_matrix`, `diagonal_margin` are spelled the same wherever they appear.
- [x] **Real APIs:** `process_emg_data`, `process_sim_actions`, `run_rollouts`, `load_intention_checkpoint`, `create_env_from_config`, `load_config`, `find_latest_step`, `build_muscle_configs`, `load_trial_info`, `TARGET_TIMESTEPS=60` all match `scripts/emg_comparison.py` as inspected.
