"""Defines rodents constants."""

from etils import epath

from mujoco_playground._src import mjx_env


RODENT_PATH = epath.Path(__file__)

FEET_ONLY_RODENT = RODENT_PATH / "xmls" / "rodent_only.xml"
