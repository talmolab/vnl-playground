"""Defines rodents constants."""

from etils import epath

from mujoco_playground._src import mjx_env


RODENT_PATH = epath.Path(__file__).parent

RODENT_XML_PATH = RODENT_PATH / "xmls" / "rodent.xml"
RODENT_BOX_FEET_PATH = RODENT_PATH / "xmls" / "rodent_box_feet.xml"
ARENA_XML_PATH = RODENT_PATH / "xmls" / "arena.xml"
WHITE_ARENA_XML_PATH = RODENT_PATH / "xmls" / "white_arena.xml"
IMITATION_REFERENCE_PATH = RODENT_PATH / "reference_data" / "reference_clips.h5"

END_EFFECTORS = ["lower_arm_R", "lower_arm_L", "foot_R", "foot_L", "skull"]
TOUCH_SENSORS = ["palm_L", "palm_R", "sole_L", "sole_R"]
BODIES = [
    "torso",
    "pelvis",
    "upper_leg_L",
    "lower_leg_L",
    "foot_L",
    "upper_leg_R",
    "lower_leg_R",
    "foot_R",
    "skull",
    "jaw",
    "scapula_L",
    "upper_arm_L",
    "lower_arm_L",
    "finger_L",
    "scapula_R",
    "upper_arm_R",
    "lower_arm_R",
    "finger_R",
]