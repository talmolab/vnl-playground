# vnl-mjx
Virtual Neural Lab (VNL) in MJX. Tasks driven deep-RL learning in JAX

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

## Features Implemented

### `mujoco_playground`-style Environment Management

We adopt the `mujoco_playground` approach to environment and task management. In our implementation, each **task is tied to a specific walker**, rather than treating tasks and walkers as separate entities.  
In this repository, **a task encapsulates its associated walker**.

### Support for `mj_spec`-based Model Editing

Unlike `mujoco_playground`, which loads models directly from disk via XML files, `vnl-mjx` uses `mj_spec` to enable procedural model creation and dynamic model editing.  
This allows us to generate environments with controlled randomness, such as varying the target location for reaching tasks (@ericleonardis) or randomizing terrain shapes (@scott-yj-yang).

### Out-of-the-Box Training Support

For simple policy architectures (e.g., MLPs and basic vision networks) already implemented by the `brax` team, environments created in this repository can be trained **directly out of the box**, without additional modifications.
