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

## Features

### `mujoco_playground`-style Environment Management

We adopt the `mujoco_playground` approach to environment and task management. Here, each task is tied to a specific walker, rather than treating tasks and walkers as separate entities (as in dm_control Composer). This allows environments to make more assumptions about body model definitions at the cost of repeated environment logic.

### Programmatic Model Editing

`vnl-playground` uses [`mujoco.Mjspec`](https://mujoco.readthedocs.io/en/stable/python.html#model-editing) during model creation and editing. This allows us to generate environments procedurally, such as adding target locations for reaching tasks or randomizing terrain shapes.

### RL Training with Brax

RL training can be done out of the box with [Brax](https://github.com/google/brax) and [RSL-RL](https://github.com/leggedrobotics/rsl_rl). Our demo notebooks: WIP. Also, check out the [MuJoCo Playground examples](https://github.com/google-deepmind/mujoco_playground/tree/main/learning). 
