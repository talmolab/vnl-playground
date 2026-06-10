# Stick imitation reference data

The reference-clip H5 used by `StickImitation` is **not vendored in this repo**
(it's a large binary). Download it from the **MIMIC-MJX** HuggingFace dataset and
place it in this folder.

## Files — [`talmolab/MIMIC-MJX`](https://huggingface.co/datasets/talmolab/MIMIC-MJX/tree/main/data/stick), `data/stick/`

| file | model | layout | notes |
|------|-------|--------|-------|
| `stick_mesh_reference.h5` | **mesh** (default) | 75 clips × 225 frames, **48 qpos** | matches the mesh walker (`stick_mesh_fast.xml`); 39 markers |
| `stick_box_model_reference.h5` | box (legacy) | 16,875 frames, **45 qpos** | older box model with the 3 thorax joints disabled; 50 markers |

`consts.IMITATION_REFERENCE_PATH` defaults to `stick_mesh_reference.h5` (this
folder). `StickMaintainVelocity` and the mesh smoke test need **no** reference
data — only `StickImitation` does.

## Download

Python (robust across `huggingface_hub` versions):

```python
import shutil
from huggingface_hub import hf_hub_download

p = hf_hub_download(
    "talmolab/MIMIC-MJX", "data/stick/stick_mesh_reference.h5", repo_type="dataset"
)
shutil.copy(p, "vnl_playground/tasks/stick/reference_data/stick_mesh_reference.h5")
```

or via the CLI:

```bash
huggingface-cli download talmolab/MIMIC-MJX data/stick/stick_mesh_reference.h5 \
  --repo-type dataset --local-dir /tmp/mimic-mjx
cp /tmp/mimic-mjx/data/stick/stick_mesh_reference.h5 \
   vnl_playground/tasks/stick/reference_data/
```

## Running imitation

The mesh dataset is **75 clips of 225 frames each**, so it must be loaded with
`clip_length=225`. The env default `clip_length=100` was set for an old
100-frame sample and will fail to reshape the 16,875-frame file:

```python
from vnl_playground import registry

env = registry.load("StickImitation", config_overrides={"clip_length": 225})
```
