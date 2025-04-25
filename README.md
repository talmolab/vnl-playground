# vnl-mjx
Virtual Neural Lab (VNL) in MJX. Tasks driven deel-RL learning in JAX

## Features Implemented

### `mujoco_playground`-style Environment Management

We adopt the `mujoco_playground` approach to environment and task management. In our implementation, each **task is tied to a specific walker**, rather than treating tasks and walkers as separate entities.  
In this repository, **a task encapsulates its associated walker**.

### Support for `mj_spec`-based Model Editing

Unlike `mujoco_playground`, which loads models directly from disk via XML files, `vnl-mjx` uses `mj_spec` to enable procedural model creation and dynamic model editing.  
This allows us to generate environments with controlled randomness, such as varying the target location for reaching tasks (@ericleonardis) or randomizing terrain shapes (@scott-yj-yang).

### Out-of-the-Box Training Support

For simple policy architectures (e.g., MLPs and basic vision networks) already implemented by the `brax` team, environments created in this repository can be trained **directly out of the box**, without additional modifications.
