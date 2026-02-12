# vnl-playground
Virtual Neural Lab (VNL) in MJX. Deep reinforcement learning environments for neuroscience following the [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) API

## Quick Start

### Option 1: `pip`

To install `vnl-mjx`, first navigate to the project directory and run:

```bash
pip install -e .[with-cuda]
```

### Option 2: `uv`

#### Prerequisites

- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip
- CUDA 12.x or 13.x (for GPU support, optional)

#### Installing `uv`

If you don't have uv installed:

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

#### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/talmolab/vnl-playground.git
cd vnl-playground
```
2. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
3. Install the package with optional dependencies based on your hardware. CUDA 12, CUDA 13, and CPU-only configurations are supported:

For CUDA 12.x:
```bash
uv pip install -e ".[cuda12]"
```

For CUDA 13.x:
```bash
uv pip install -e ".[cuda13]"
```

For CPU-only:
```bash
uv pip install -e .
```

For development, include the `[dev]` extras in addition to the hardware optional dependencies:
```bash
uv pip install -e ".[cuda13,dev]"
```
4. Verify the installation:
```bash
python -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'Available devices: {jax.devices()}')"
```
5. Register the environment as a Jupyter kernel:
```bash
python -m ipykernel install --user --name=track-mjx --display-name="Python (track-mjx)"
```

## Supported Body Models

| Walker | Description |
|--------|-------------|
| **Rodent** | Biomechanical model of *Rattus norvegicus* (Long Evans) with 74 DoF and 38 torque actuators, derived from [Merel et al., 2018](https://arxiv.org/abs/1811.11711) |
| **Fruitfly** | Anatomically-detailed model of *Drosophila melanogaster* with 102 DoF and 61 torque actuators, reconstructed from confocal microscopy ([Vaxenburg et al., 2025](https://doi.org/10.1038/s41586-025-09029-4)) |
| **Mouse** | Musculoskeletal forelimb model of *Mus musculus* with 4 DoF and 9 Hill-type muscle actuators, built from light sheet microscopy data ([Gilmer et al., 2025](https://doi.org/10.1152/jn.00198.2024)) |
| **Stick Bug** | Model of *Sungaya aeta* with 42 DoF and 42 joints across 43 rigid body segments, constructed from 3D photogrammetry via [scAnt](https://github.com/evo-biomech/scAnt) ([Plum & Labonte, 2021](https://doi.org/10.7717/peerj.11155)) |

## Available Environments

All 12 environments are accessible via the registry: `registry.load("TaskName", config=cfg, ...)`.

### Rodent

| Environment | Description |
|-------------|-------------|
| `RodentImitation` | Multi-clip imitation learning from mocap with dense tracking rewards |
| `RodentSparseImitation` | Sparse-reward imitation via elastic sequence matching (online DP) |
| `RodentRearing` | Raise head above target height relative to torso |
| `RodentBowlEscape` | Escape from a bowl-shaped arena (supports vision) |
| `RodentMaintainVelocity` | Track a target forward velocity in an open arena |
| `RodentJoystick` | Track periodically resampled forward velocity + yaw rate commands |

### Fruitfly

| Environment | Description |
|-------------|-------------|
| `FruitflyImitation` | Multi-clip imitation at 500 Hz control / 5000 Hz physics |
| `FruitflyMaintainVelocity` | Forward velocity tracking at cm-scale |

### Mouse

| Environment | Description |
|-------------|-------------|
| `MouseImitation` | Forelimb imitation learning tracking arm mocap |
| `MouseReach` | Reach toward randomly sampled or fixed 3D targets |

### Stick Bug

| Environment | Description |
|-------------|-------------|
| `StickImitation` | Multi-clip imitation learning from mocap |
| `StickMaintainVelocity` | Forward velocity tracking at mm-scale |

## Usage

```python
from vnl_playground import registry

# List all available environments
print(registry.ALL_ENVS)

# Locomotion tasks (no reference clips needed)
cfg = registry.get_default_config("RodentJoystick")
env = registry.load("RodentJoystick", config=cfg, rng=jax.random.key(0))

# Imitation tasks (require reference clips)
cfg = registry.get_default_config("RodentImitation")
clips = registry.load_reference_clips("RodentImitation", data_path, n_frames_per_clip=250)
env = registry.load("RodentImitation", config=cfg, clips=clips)
```

## Notes

### `mujoco_playground`-style Environment Management

We adopt the `mujoco_playground` approach to environment and task management. Here, each task is tied to a specific walker, rather than treating tasks and walkers as separate entities (as in dm_control Composer). This allows environments to make more assumptions about body model definitions at the cost of repeated environment logic.

### Programmatic Model Editing

`vnl-playground` uses [`mujoco.Mjspec`](https://mujoco.readthedocs.io/en/stable/python.html#model-editing) during model creation and editing. This allows us to generate environments procedurally, such as adding target locations for reaching tasks or randomizing terrain shapes.

### RL Training with Brax (demo scripts wip)

This repo was originally built to house environments for [track-mjx](https://github.com/talmolab/track-mjx), which includes examples of task training using [Brax](https://github.com/google/brax). Demo scripts/notebooks in this repo are wip. Check out the [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground/tree/main/learning) for more examples. 
