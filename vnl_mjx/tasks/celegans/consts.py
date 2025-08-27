"""Defines rodents constants."""

from etils import epath

from mujoco_playground._src import mjx_env


CELEGANS_PATH = epath.Path(__file__).parent

CELEGANS_XML_PATH = CELEGANS_PATH / "xmls" / "celegans_fast.xml"
ARENA_XML_PATH = CELEGANS_PATH / "xmls" / "arena.xml"
WHITE_ARENA_XML_PATH = CELEGANS_PATH / "xmls" / "white_arena.xml"