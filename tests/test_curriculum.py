"""Tests for auto-curriculum trainer."""

import pytest

from vnl_playground.tasks.rodent.curriculum import (
    CurriculumPhase,
    GraduationMonitor,
    apply_phase_to_env_config,
    apply_phase_to_train_config,
    build_phases_from_config,
    make_curriculum_progress_fn,
)

# ---- GraduationMonitor tests ----


def test_monitor_graduates_after_patience():
    monitor = GraduationMonitor(threshold=0.8, patience=3)
    assert not monitor.should_graduate

    # 3 consecutive evals above threshold -> graduate
    monitor.update({"eval/episode_trial/success": 0.85})
    assert not monitor.should_graduate
    monitor.update({"eval/episode_trial/success": 0.90})
    assert not monitor.should_graduate
    monitor.update({"eval/episode_trial/success": 0.82})
    assert monitor.should_graduate


def test_monitor_resets_on_dip():
    monitor = GraduationMonitor(threshold=0.8, patience=3)

    monitor.update({"eval/episode_trial/success": 0.85})
    monitor.update({"eval/episode_trial/success": 0.90})
    # Dip below threshold resets counter
    monitor.update({"eval/episode_trial/success": 0.70})
    assert not monitor.should_graduate

    # Need 3 fresh above-threshold evals
    monitor.update({"eval/episode_trial/success": 0.85})
    monitor.update({"eval/episode_trial/success": 0.90})
    assert not monitor.should_graduate
    monitor.update({"eval/episode_trial/success": 0.82})
    assert monitor.should_graduate


def test_monitor_ignores_missing_metric():
    monitor = GraduationMonitor(threshold=0.8, patience=1)
    monitor.update({"some_other_metric": 1.0})
    assert not monitor.should_graduate
    assert monitor.latest_success_rate == 0.0


def test_monitor_tracks_latest_success_rate():
    monitor = GraduationMonitor(threshold=0.8, patience=5)
    monitor.update({"eval/episode_trial/success": 0.45})
    assert monitor.latest_success_rate == pytest.approx(0.45)
    monitor.update({"eval/episode_trial/success": 0.67})
    assert monitor.latest_success_rate == pytest.approx(0.67)


def test_monitor_reset():
    monitor = GraduationMonitor(threshold=0.8, patience=1)
    monitor.update({"eval/episode_trial/success": 0.95})
    assert monitor.should_graduate
    monitor.reset()
    assert not monitor.should_graduate
    assert monitor.latest_success_rate == 0.0


def test_monitor_alternative_metric_key():
    """Monitor should also check eval/episode_episode_trial/success."""
    monitor = GraduationMonitor(threshold=0.5, patience=1)
    monitor.update({"eval/episode_episode_trial/success": 0.6})
    assert monitor.should_graduate


# ---- CurriculumPhase + build_phases_from_config tests ----


def test_build_phases_from_config():
    curriculum_cfg = {
        "graduation_threshold": 0.8,
        "graduation_patience": 3,
        "phases": [
            {
                "name": "Phase 1",
                "gap_distances": [0.0, 0.005, 0.01],
                "hold_duration": 0,
                "learning_rate": 3e-4,
                "num_timesteps": 50_000_000,
            },
            {
                "name": "Phase 2",
                "gap_distances": [0.02, 0.04, 0.06],
                "hold_duration": 25,
                "learning_rate": 1e-4,
                "num_timesteps": 100_000_000,
                "extra_reward_terms": {"approach_velocity": {"weight": 0.1}},
            },
        ],
    }
    phases = build_phases_from_config(curriculum_cfg)

    assert len(phases) == 2
    assert phases[0].name == "Phase 1"
    assert phases[0].gap_distances == (0.0, 0.005, 0.01)
    assert phases[0].graduation_threshold == 0.8  # inherited from top-level
    assert phases[0].graduation_patience == 3
    assert phases[0].learning_rate == 3e-4

    assert phases[1].name == "Phase 2"
    assert phases[1].hold_duration == 25
    assert phases[1].extra_reward_terms == {"approach_velocity": {"weight": 0.1}}


def test_build_phases_per_phase_threshold_override():
    curriculum_cfg = {
        "graduation_threshold": 0.8,
        "graduation_patience": 3,
        "phases": [
            {
                "name": "Easy",
                "gap_distances": [0.0],
                "graduation_threshold": 0.5,  # override
                "graduation_patience": 1,
            },
        ],
    }
    phases = build_phases_from_config(curriculum_cfg)
    assert phases[0].graduation_threshold == 0.5
    assert phases[0].graduation_patience == 1


# ---- apply_phase_to_env_config tests ----


def test_apply_phase_to_env_config():
    base = {
        "gap_distances": [0.0],
        "hold_duration": 0,
        "episode_length": 300,
        "max_decision_steps": 200,
        "reward_terms": {
            "jump_success": {"weight": 100.0},
            "fall_penalty": {"weight": -10.0},
        },
        "termination_criteria": {
            "fallen": {"min_torso_z": -0.1},
            "trial_timeout": {"max_steps": 300},
        },
    }
    phase = CurriculumPhase(
        name="Phase 2",
        gap_distances=(0.02, 0.04, 0.06),
        hold_duration=25,
        episode_length=400,
        extra_reward_terms={"approach_velocity": {"weight": 0.1}},
    )
    result = apply_phase_to_env_config(base, phase)

    assert result["gap_distances"] == [0.02, 0.04, 0.06]
    assert result["hold_duration"] == 25
    assert result["episode_length"] == 400
    # Base rewards preserved + extra added
    assert result["reward_terms"]["jump_success"]["weight"] == 100.0
    assert result["reward_terms"]["approach_velocity"]["weight"] == 0.1
    # trial_timeout updated to match episode_length
    assert result["termination_criteria"]["trial_timeout"]["max_steps"] == 400


def test_apply_phase_to_env_config_no_mutate():
    """Applying a phase should not mutate the base config."""
    base = {"gap_distances": [0.0], "hold_duration": 0, "episode_length": 300}
    phase = CurriculumPhase(
        name="Test",
        gap_distances=(0.1,),
        hold_duration=50,
        episode_length=500,
    )
    apply_phase_to_env_config(base, phase)
    assert base["gap_distances"] == [0.0]
    assert base["hold_duration"] == 0


# ---- apply_phase_to_train_config tests ----


def test_apply_phase_to_train_config():
    base = {
        "learning_rate": 3e-4,
        "unroll_length": 10,
        "num_timesteps": 100_000_000,
        "episode_length": 300,
        "batch_size": 1024,
    }
    phase = CurriculumPhase(
        name="Phase 3",
        gap_distances=(0.06, 0.14),
        learning_rate=5e-5,
        unroll_length=50,
        num_timesteps=500_000_000,
        episode_length=500,
    )
    result = apply_phase_to_train_config(base, phase)

    assert result["learning_rate"] == 5e-5
    assert result["unroll_length"] == 50
    assert result["num_timesteps"] == 500_000_000
    assert result["episode_length"] == 500
    assert result["batch_size"] == 1024  # preserved from base


# ---- make_curriculum_progress_fn tests ----


def test_curriculum_progress_fn_adds_metrics():
    monitor = GraduationMonitor(threshold=0.8, patience=1)
    logged = {}

    def base_fn(num_steps, metrics):
        logged.update(metrics)

    progress_fn = make_curriculum_progress_fn(monitor, base_fn, 1, "Phase 1")
    progress_fn(1000, {"eval/episode_trial/success": 0.5})

    assert logged["curriculum/phase"] == 1
    assert logged["curriculum/phase_name"] == "Phase 1"
    assert logged["curriculum/success_rate"] == pytest.approx(0.5)
    assert logged["curriculum/graduated"] == 0.0


def test_curriculum_progress_fn_detects_graduation():
    monitor = GraduationMonitor(threshold=0.5, patience=1)
    calls = []

    def base_fn(num_steps, metrics):
        calls.append(metrics.get("curriculum/graduated"))

    progress_fn = make_curriculum_progress_fn(monitor, base_fn, 1, "Test")
    progress_fn(1000, {"eval/episode_trial/success": 0.6})
    assert calls[-1] == 1.0
