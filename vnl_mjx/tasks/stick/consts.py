"""Defines rodents constants."""

from etils import epath

from mujoco_playground._src import mjx_env


STICK_PATH = epath.Path(__file__).parent

STICK_XML_PATH = STICK_PATH / "xmls" / "sungaya_inexpectata_box.xml"
STICK_FAST_XML_PATH = STICK_PATH / "xmls" / "stick_fast.xml"
ARENA_XML_PATH = STICK_PATH / "xmls" / "arena.xml"
WHITE_ARENA_XML_PATH = STICK_PATH / "xmls" / "white_arena.xml"