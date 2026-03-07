"""Diagnose observation compatibility between SCAMPER checkpoints and vnl-playground envs.

Usage:
    python scripts/diagnose_scamper_obs.py EXPERT_CHECKPOINT_PATH [PRIOR_CHECKPOINT_PATH]

Example:
    python scripts/diagnose_scamper_obs.py /path/to/imitation/checkpoint
    python scripts/diagnose_scamper_obs.py /path/to/imitation/checkpoint /path/to/prior/checkpoint
"""

import sys
import collections

import jax
import jax.numpy as jnp
from jax import flatten_util

from vnl_playground.tasks.rodent import imitation
from scamper.agent.mlp_prior.prior_networks import load_frozen_encoder_decoder
from scamper.agent.observation_utils import flatten_obs_dict


def flatten_nested(obs_dict):
    """Flatten a nested obs dict to get sizes per key."""
    result = {}
    for key, val in obs_dict.items():
        if isinstance(val, (collections.OrderedDict, dict)):
            flat, _ = flatten_util.ravel_pytree(val)
            result[key] = flat.shape[0]
        else:
            flat = val.reshape(-1)
            result[key] = flat.shape[0]
    return result


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/diagnose_scamper_obs.py"
            " EXPERT_CHECKPOINT [PRIOR_CHECKPOINT]"
        )
        sys.exit(1)

    expert_path = sys.argv[1]
    prior_path = sys.argv[2] if len(sys.argv) > 2 else None

    print("=" * 70)
    print("SCAMPER Observation Compatibility Diagnostic")
    print("=" * 70)

    # 1. Load expert (imitation) checkpoint config
    print("\n--- Expert Checkpoint ---")
    print(f"Path: {expert_path}")
    _, _, teacher_norm, teacher_cfg = load_frozen_encoder_decoder(expert_path)
    net_cfg = teacher_cfg["network_config"]

    obs_sizes = net_cfg.get("obs_sizes", None)
    if obs_sizes:
        print(f"  obs_sizes (dict format): {dict(obs_sizes)}")
        expert_imit_size = obs_sizes["imitation_target"]
        expert_proprio_size = obs_sizes["proprioception"]
    else:
        ref_size = net_cfg["reference_obs_size"]
        total_size = net_cfg["observation_size"]
        expert_imit_size = ref_size
        expert_proprio_size = total_size - ref_size
        print(
            f"  obs_sizes (legacy flat): imitation_target={ref_size},"
            f" proprioception={expert_proprio_size}"
        )

    print(f"  intention_size: {net_cfg['intention_size']}")
    print(f"  action_size: {net_cfg['action_size']}")
    print(f"  encoder_layers: {net_cfg['encoder_layer_sizes']}")
    print(f"  decoder_layers: {net_cfg['decoder_layer_sizes']}")

    # Check normalizer stats
    print(
        f"  normalizer.proprioception.mean shape:"
        f" {teacher_norm.proprioception.mean.shape}"
    )
    print(
        f"  normalizer.proprioception.std range:"
        f" [{float(teacher_norm.proprioception.std.min()):.4f},"
        f" {float(teacher_norm.proprioception.std.max()):.4f}]"
    )
    print(
        f"  normalizer.proprioception.count:"
        f" {int(teacher_norm.proprioception.count)}"
    )

    # 2. Create RodentImitation env and check obs
    print("\n--- RodentImitation Environment ---")
    imit_env = imitation.Imitation()
    imit_state = jax.jit(imit_env.reset)(jax.random.PRNGKey(0))
    imit_obs = imit_state.obs

    # Current branch has nested obs: {state: {task_obs, proprioception}, ...}
    if "state" in imit_obs:
        inner = imit_obs["state"]
        adapted = collections.OrderedDict(
            imitation_target=inner["task_obs"],
            proprioception=inner["proprioception"],
        )
    else:
        adapted = imit_obs

    flat_obs = flatten_obs_dict(adapted)
    env_imit_size = flat_obs["imitation_target"].shape[-1]
    env_proprio_size = flat_obs["proprioception"].shape[-1]

    print(f"  imitation_target size: {env_imit_size}")
    print(f"  proprioception size: {env_proprio_size}")
    print(f"  action_size: {imit_env.action_size}")

    # 3. Compare
    print("\n--- Compatibility Check: Expert vs Imitation Env ---")
    imit_ok = True
    if expert_imit_size != env_imit_size:
        print(
            f"  MISMATCH imitation_target:"
            f" expert={expert_imit_size}, env={env_imit_size}"
        )
        imit_ok = False
    else:
        print(f"  imitation_target: MATCH ({expert_imit_size})")

    if expert_proprio_size != env_proprio_size:
        print(
            f"  MISMATCH proprioception:"
            f" expert={expert_proprio_size}, env={env_proprio_size}"
        )
        imit_ok = False
    else:
        print(f"  proprioception: MATCH ({expert_proprio_size})")

    if imit_ok:
        print("  RESULT: Expert checkpoint is compatible with RodentImitation env")
    else:
        print("  RESULT: INCOMPATIBLE -- prior distillation will fail!")

    # 4. Prior checkpoint (optional)
    if prior_path:
        print(f"\n--- Prior Checkpoint ---")
        print(f"Path: {prior_path}")
        try:
            from scamper.agent.task_transfer.walk_rear.checkpoint_utils import (
                load_prior_checkpoint,
            )

            _, prior_params, decoder_params, prior_norm, prior_cfg = (
                load_prior_checkpoint(prior_path)
            )
            prior_net_cfg = prior_cfg["network_config"]
            print(f"  intention_size: {prior_net_cfg['intention_size']}")
            print(f"  action_size: {prior_net_cfg['action_size']}")
            prior_obs_sizes = prior_net_cfg.get("obs_sizes", {})
            print(f"  obs_sizes: {dict(prior_obs_sizes)}")
            print(
                f"  prior_layer_sizes:"
                f" {prior_net_cfg.get('prior_layer_sizes', 'N/A')}"
            )
            print(
                f"  normalizer.proprioception.count:"
                f" {int(prior_norm.proprioception.count)}"
            )

            env_cfg = prior_cfg.get("env_config", {})
            prior_ctrl_dt = env_cfg.get("ctrl_dt", "N/A")
            print(f"  ctrl_dt: {prior_ctrl_dt}")
        except Exception as e:
            print(f"  Error loading prior checkpoint: {e}")

    # 5. ctrl_dt summary
    print("\n--- ctrl_dt Alignment Check ---")
    teacher_env_cfg = teacher_cfg.get("env_config", {})
    teacher_ctrl_dt = teacher_env_cfg.get("ctrl_dt", "N/A")
    print(f"  Expert (imitation) ctrl_dt: {teacher_ctrl_dt}")
    if prior_path:
        try:
            print(f"  Prior checkpoint ctrl_dt: {prior_ctrl_dt}")
        except NameError:
            pass
    print(
        f"  NOTE: All downstream transfer configs MUST use"
        f" ctrl_dt={teacher_ctrl_dt}"
    )

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
