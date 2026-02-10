"""Validate RunGap environment with zero and random action rollouts.

Runs episodes with zero actions (should stand still and eventually fall)
and random actions (should move erratically). Plots reward curves and
rodent trajectory (x position over time).
"""

import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import numpy as np

from vnl_playground.tasks.rodent import run_gap


def rollout(env, rng, policy_fn, n_steps=500):
    """Run a single episode, collecting rewards and positions."""
    rng, reset_rng = jax.random.split(rng)
    state = jax.jit(env.reset)(reset_rng)

    rewards = []
    positions = []
    dones = []

    step_fn = jax.jit(env.step)

    for i in range(n_steps):
        rng, act_rng = jax.random.split(rng)
        action = policy_fn(act_rng, env.action_size)
        state = step_fn(state, action)
        rewards.append(float(state.reward))
        # Get torso position
        torso = state.data.bind(env.mjx_model, env._spec.body("torso-rodent"))
        positions.append(np.array(torso.xpos))
        dones.append(float(state.done))
        if state.done > 0.5:
            break

    return {
        "rewards": np.array(rewards),
        "positions": np.array(positions),
        "dones": np.array(dones),
    }


def zero_policy(rng, action_size):
    return jp.zeros(action_size)


def random_policy(rng, action_size):
    return jax.random.uniform(rng, (action_size,), minval=-1.0, maxval=1.0)


def main():
    print("Initializing RunGap environment...")
    env = run_gap.RunGap()
    print(f"  Action size: {env.action_size}")
    print(f"  Obs size: {env.observation_size}")
    print(f"  Corridor end: {env._corridor_end_x:.2f}m")

    rng = jax.random.PRNGKey(0)

    print("\nRolling out with zero actions...")
    rng, sub_rng = jax.random.split(rng)
    zero_result = rollout(env, sub_rng, zero_policy, n_steps=500)
    print(f"  Steps: {len(zero_result['rewards'])}")
    print(f"  Mean reward: {zero_result['rewards'].mean():.4f}")
    print(f"  Final x-pos: {zero_result['positions'][-1, 0]:.4f}")

    print("\nRolling out with random actions...")
    rng, sub_rng = jax.random.split(rng)
    random_result = rollout(env, sub_rng, random_policy, n_steps=500)
    print(f"  Steps: {len(random_result['rewards'])}")
    print(f"  Mean reward: {random_result['rewards'].mean():.4f}")
    print(f"  Final x-pos: {random_result['positions'][-1, 0]:.4f}")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Reward curves
    axes[0, 0].plot(zero_result["rewards"], label="Zero action")
    axes[0, 0].plot(random_result["rewards"], label="Random action", alpha=0.7)
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].set_title("Reward over time")
    axes[0, 0].legend()

    # X position
    axes[0, 1].plot(zero_result["positions"][:, 0], label="Zero action")
    axes[0, 1].plot(random_result["positions"][:, 0], label="Random action", alpha=0.7)
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("X position (m)")
    axes[0, 1].set_title("Forward progress")
    axes[0, 1].legend()

    # Z position (height)
    axes[1, 0].plot(zero_result["positions"][:, 2], label="Zero action")
    axes[1, 0].plot(random_result["positions"][:, 2], label="Random action", alpha=0.7)
    axes[1, 0].axhline(y=-0.05, color="r", linestyle="--", label="Fall threshold")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Z position (m)")
    axes[1, 0].set_title("Height over time")
    axes[1, 0].legend()

    # XY trajectory
    axes[1, 1].plot(
        zero_result["positions"][:, 0],
        zero_result["positions"][:, 1],
        label="Zero",
    )
    axes[1, 1].plot(
        random_result["positions"][:, 0],
        random_result["positions"][:, 1],
        label="Random",
        alpha=0.7,
    )
    axes[1, 1].set_xlabel("X (m)")
    axes[1, 1].set_ylabel("Y (m)")
    axes[1, 1].set_title("Top-down trajectory")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("scripts/run_gap_validation.png", dpi=150)
    print(f"\nPlot saved to scripts/run_gap_validation.png")


if __name__ == "__main__":
    main()
