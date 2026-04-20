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


def test_load_warmstart_filters_and_maps():
    fixture = Path(__file__).parent / "fixtures" / "bo_warmstart_sample.csv"
    trials = bo.load_warmstart(fixture)

    names = {t.user_attrs["source_name"] for t in trials}
    assert names == {"keep-ms-a", "keep-ms-b", "keep-out-of-bounds"}


def test_load_warmstart_maps_axis_values():
    fixture = Path(__file__).parent / "fixtures" / "bo_warmstart_sample.csv"
    trials = bo.load_warmstart(fixture)
    by_name = {t.user_attrs["source_name"]: t for t in trials}

    a = by_name["keep-ms-a"]
    assert a.params["fs"] == pytest.approx(1.0)
    assert a.params["damp"] == pytest.approx(5e-7)
    assert a.params["cc"] == pytest.approx(0.025)
    assert a.params["cdc"] == pytest.approx(0.025)
    assert a.params["qvel_init"] == "zeros"
    assert a.values == [pytest.approx(0.57), pytest.approx(0.57),
                        pytest.approx(0.13), pytest.approx(0.14),
                        pytest.approx(0.20), pytest.approx(0.19)]
    assert a.user_attrs["R"] == pytest.approx(396.0)
    assert a.user_attrs["constraint"] == [pytest.approx(380.0 - 396.0)]


def test_load_warmstart_includes_out_of_bounds_fs():
    fixture = Path(__file__).parent / "fixtures" / "bo_warmstart_sample.csv"
    trials = bo.load_warmstart(fixture)
    by_name = {t.user_attrs["source_name"]: t for t in trials}
    oob = by_name["keep-out-of-bounds"]
    assert oob.params["fs"] == pytest.approx(0.3)


def test_make_study_creates_multi_objective_with_constraints(tmp_path):
    journal = tmp_path / "bo_study.log"
    study = bo.make_study("test-study", journal)
    assert len(study.directions) == 6
    assert study.directions[0].name == "MAXIMIZE"
    assert study.directions[1].name == "MAXIMIZE"
    assert study.directions[2].name == "MINIMIZE"
    assert study.directions[3].name == "MINIMIZE"
    assert study.directions[4].name == "MINIMIZE"
    assert study.directions[5].name == "MINIMIZE"
    assert isinstance(study.sampler, bo.NSGAIISampler)


def test_make_study_resumes_from_journal(tmp_path):
    journal = tmp_path / "bo_study.log"
    s1 = bo.make_study("resume-test", journal)
    trial = s1.ask(bo.SEARCH_SPACE_DISTRIBUTIONS)
    trial.set_user_attr("constraint", [-5.0])
    s1.tell(trial, [0.5, 0.5, 0.1, 0.1, 0.2, 0.2])

    s2 = bo.make_study("resume-test", journal)
    assert len(s2.trials) == 1
