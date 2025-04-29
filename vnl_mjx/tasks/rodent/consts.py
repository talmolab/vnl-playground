"""Defines rodents constants."""

from etils import epath

from mujoco_playground._src import mjx_env


RODENT_PATH = epath.Path(__file__).parent

RODENT_XML_PATH = RODENT_PATH / "xmls" / "rodent.xml"
RODENT_SPHERE_FEET_PATH = RODENT_PATH / "xmls" / "rodent_sphere_feet.xml"
RODENT_BOX_FEET_PATH = RODENT_PATH / "xmls" / "rodent_box_feet.xml"
ARENA_XML_PATH = RODENT_PATH / "xmls" / "arena.xml"
