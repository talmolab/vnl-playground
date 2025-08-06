"""Base classes for fruitfly"""

from typing import Any, Dict, Optional, Union

from etils import epath
import logging
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from vnl_mjx.tasks.rodent import consts
from vnl_mjx.tasks.utils import _scale_body_tree, _recolour_tree, dm_scale_spec


def get_assets() -> Dict[str, bytes]:
    assets = {}
    mjx_env.update_assets(assets, consts.RODENT_PATH / "xmls", "*.xml")
    mjx_env.update_assets(assets, consts.RODENT_PATH / "xmls" / "assets")
    return assets
