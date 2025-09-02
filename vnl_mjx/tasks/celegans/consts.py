"""Defines rodents constants."""

from etils import epath

from mujoco_playground._src import mjx_env


CELEGANS_PATH = epath.Path(__file__).parent

CELEGANS_XML_PATH = CELEGANS_PATH / "xmls" / "celegans_fast.xml"
ARENA_XML_PATH = CELEGANS_PATH / "xmls" / "arena.xml"
WHITE_ARENA_XML_PATH = CELEGANS_PATH / "xmls" / "white_arena.xml"

ROOT = "torso1_body"
END_EFFECTORS = ["torso1_body", "torso2_body", "torso3_body", "torso4_body", "torso5_body", "torso21_body", "torso22_body", "torso23_body", "torso24_body", "torso25_body"]
TOUCH_SENSORS = []
BODIES = [
    "torso1_body",
    "torso2_body",
    "torso3_body",
    "torso4_body",
    "torso5_body",
    "torso21_body",
    "torso22_body",
    "torso23_body",
    "torso24_body",
    "torso25_body",
]
JOINTS = [
    "motor1_rot",
    "motor2_rot",
    "rot4",
    "rot5",
    "rot6",
    "rot7",
    "rot8",
    "rot9",
    "rot10",
    "rot11",
    "rot12",
    "rot13",
    "rot14",
    "rot15",
    "rot16",
    "rot17",
    "rot18",
    "rot19",
    "rot20",
    "rot21",
    "rot22",
    "rot23",
    "rot24",
    "rot25",
]