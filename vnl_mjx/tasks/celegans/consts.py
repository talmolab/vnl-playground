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
    "torso6_body",
    "torso7_body",
    "torso8_body",
    "torso9_body",
    "torso10_body",
    "torso11_body",
    "torso12_body",
    "torso13_body",
    "torso14_body",
    "torso15_body",
    "torso16_body",
    "torso17_body",
    "torso18_body",
    "torso19_body",
    "torso20_body",
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
SENSORS = ["accelerometer", "gyro", "velocimeter"]