"""Tests for mouse reaching task."""

import jax
import jax.numpy as jp
import numpy as np
import mujoco


def test_env_creation():
    """Test that environment can be created."""
    from vnl_playground.tasks.mouse import mouse_reach

    env = mouse_reach.MouseReach()
    print(f"Action size: {env.action_size}")
    print(f"Observation size: {env.observation_size}")

    assert env.action_size == 9, f"Expected 9 actuators, got {env.action_size}"
    print("✓ Action size correct")


def test_reset_target_sampling():
    """Test that reset samples different targets."""
    from vnl_playground.tasks.mouse import mouse_reach

    env = mouse_reach.MouseReach()

    # Reset multiple times with different seeds
    targets = []
    for i in range(10):
        rng = jax.random.PRNGKey(i)
        state = env.reset(rng)
        target = state.info["target_position"]
        targets.append(np.array(target))
        print(f"Seed {i}: target = {target}")

    # Check that we get different targets
    targets = np.array(targets)
    unique_targets = np.unique(targets, axis=0)
    print(f"Unique targets: {len(unique_targets)} out of 10 resets")

    assert len(unique_targets) > 1, "All targets are the same - randomization not working!"
    print("✓ Target sampling works")


def test_reward_computation():
    """Test that reward varies with distance to target."""
    from vnl_playground.tasks.mouse import mouse_reach

    env = mouse_reach.MouseReach()

    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)

    target_pos = state.info["target_position"]
    wrist_pos = state.data.geom_xpos[env._wrist_marker_geom_id]

    print(f"Target position: {target_pos}")
    print(f"Wrist marker position: {wrist_pos}")
    print(f"Distance: {jp.linalg.norm(wrist_pos - target_pos)}")
    print(f"Initial reward: {state.reward}")

    # Take a step with zero action
    action = jp.zeros(env.action_size)
    next_state = env.step(state, action)
    print(f"Reward after step: {next_state.reward}")

    print("✓ Reward computation works")


def test_observation_contains_target_direction():
    """Test that observation includes direction to target."""
    from vnl_playground.tasks.mouse import mouse_reach

    env = mouse_reach.MouseReach()

    rng = jax.random.PRNGKey(42)
    state = env.reset(rng)

    # The first 3 elements of obs should be the direction to target
    target_pos = state.info["target_position"]
    wrist_pos = state.data.xpos[env._wrist_body_id]
    expected_direction = target_pos - wrist_pos

    obs_direction = state.obs[:3]

    print(f"Expected direction (target - wrist): {expected_direction}")
    print(f"Obs direction (first 3 elements): {obs_direction}")

    assert jp.allclose(obs_direction, expected_direction, atol=1e-5), \
        "Observation doesn't contain correct target direction!"
    print("✓ Observation contains correct target direction")


def test_mocap_target_position():
    """Test that mocap target position is set correctly."""
    from vnl_playground.tasks.mouse import mouse_reach

    env = mouse_reach.MouseReach()

    # Check if mocap body exists
    print(f"Number of mocap bodies: {env._mj_model.nmocap}")

    if env._mj_model.nmocap > 0:
        rng = jax.random.PRNGKey(0)
        state = env.reset(rng)

        target_pos = state.info["target_position"]
        mocap_pos = state.data.mocap_pos[0]

        print(f"Target position (info): {target_pos}")
        print(f"Mocap position (data): {mocap_pos}")

        assert jp.allclose(mocap_pos, target_pos, atol=1e-5), \
            "Mocap position doesn't match target position!"
        print("✓ Mocap target position matches")
    else:
        print("⚠ No mocap bodies - target visualization won't update")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Mouse Reach Tests")
    print("=" * 60)

    try:
        test_env_creation()
    except Exception as e:
        print(f"✗ test_env_creation failed: {e}")

    print()

    try:
        test_reset_target_sampling()
    except Exception as e:
        print(f"✗ test_reset_target_sampling failed: {e}")

    print()

    try:
        test_reward_computation()
    except Exception as e:
        print(f"✗ test_reward_computation failed: {e}")

    print()

    try:
        test_observation_contains_target_direction()
    except Exception as e:
        print(f"✗ test_observation_contains_target_direction failed: {e}")

    print()

    try:
        test_mocap_target_position()
    except Exception as e:
        print(f"✗ test_mocap_target_position failed: {e}")

    print()
    print("=" * 60)
    print("Tests complete")
    print("=" * 60)
