"""Compatibility shims for jax APIs removed out from under pinned dependencies.

jax==0.10.2 (pinned in uv.lock) removed jax.device_put_replicated /
jax.device_put_sharded, which brax==0.14.0's pmap-based PPO trainer
(brax/training/agents/ppo/train.py) still calls directly. Restores
device_put_replicated with the same semantics (stack the pytree along a new
leading axis of size len(devices), sharded one slice per device) using the
modern jax.sharding API, so brax's existing pmap calls keep working unchanged.
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def _device_put_replicated(x, devices):
    devices = list(devices)
    mesh = Mesh(devices, axis_names=("d",))
    sharding = NamedSharding(mesh, P("d"))

    def _rep(leaf):
        leaf = jnp.asarray(leaf)
        return jax.device_put(jnp.stack([leaf] * len(devices), axis=0), sharding)

    return jax.tree_util.tree_map(_rep, x)


def install():
    jax.device_put_replicated = _device_put_replicated
