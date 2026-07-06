# Stick imitation reference data

The reference-clip H5 used by `StickImitation` is **not vendored in this repo**
(it's a large binary, git-ignored here). Fetch it from the **MIMIC-MJX**
HuggingFace dataset.

## Files — [`talmolab/MIMIC-MJX`](https://huggingface.co/datasets/talmolab/MIMIC-MJX/tree/main/data/stick), `data/stick/`

| file | model | layout | notes |
|------|-------|--------|-------|
| `stick_mesh_reference.h5` | **mesh** (default) | 75 clips × 225 frames, **48 qpos** | matches the mesh walker; 39 markers |
| `stick_box_model_reference.h5` | box (legacy) | 16,875 frames, **45 qpos** | older box model (3 thorax joints disabled); 50 markers |
| `stick_insect_mocap.h5` | raw source | 16,875 frames, **88 landmarks** (mm) | pre-STAC source mocap (SLEAP `tracks`); for re-deriving markers / re-fitting STAC — not needed to run imitation |

## Download

Easiest — the baked-in helper (writes into this folder):

```python
from vnl_playground.tasks.stick.base import StickBugEnv

StickBugEnv.download_reference_data()         # mesh -> reference_data/stick_mesh_reference.h5
# StickBugEnv.download_reference_data("box")  # box model
```

Or let the env fetch it on demand:

```python
from vnl_playground import registry

env = registry.load("StickImitation", config_overrides={"auto_download": True})
```

Both use `huggingface_hub` (`pip install huggingface_hub`) under the hood. If the
data is missing and `auto_download` is off, `StickImitation` raises a clear error
pointing here.

## Running imitation

`clip_length` defaults to **225** to match the dataset (75 clips × 225 frames), so
once the file is present no overrides are needed:

```python
from vnl_playground import registry

env = registry.load("StickImitation")   # loads reference_data/stick_mesh_reference.h5
```
