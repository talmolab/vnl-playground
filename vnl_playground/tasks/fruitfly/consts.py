"""Defines fruitfly constants."""

from etils import epath

from mujoco_playground._src import mjx_env

FRUITFLY_PATH = epath.Path(__file__).parent

FRUITFLY_XML_PATH = FRUITFLY_PATH / "xmls" / "fruitfly_fast.xml"
ARENA_XML_PATH = FRUITFLY_PATH / "xmls" / "arena.xml"
WHITE_ARENA_XML_PATH = FRUITFLY_PATH / "xmls" / "white_arena.xml"