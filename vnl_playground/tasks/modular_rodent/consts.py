"""Constants for modular rodent environments."""

from etils import epath

MODULAR_RODENT_PATH = epath.Path(__file__).parent
MODULAR_RODENT_WALKER_XML_PATH = MODULAR_RODENT_PATH / "xmls" / "rodent_modular_walker.xml"
MODULAR_RODENT_ARENA_XML_PATH = (
    MODULAR_RODENT_PATH.parent / "rodent" / "xmls" / "arena.xml"
)
MODULAR_RODENT_XML_PATH = MODULAR_RODENT_WALKER_XML_PATH  # backward compat alias
IMITATION_REFERENCE_PATH = (
    MODULAR_RODENT_PATH.parent / "rodent" / "reference_data" / "reference_clips.h5"
)

MODULES = [
    "hand_L",
    "arm_L",
    "hand_R",
    "arm_R",
    "foot_L",
    "leg_L",
    "foot_R",
    "leg_R",
    "torso",
    "head",
]

# Body names in the compiled model
BODY_NAMES = {
    "torso": "torso",
    "pelvis": "pelvis",
    "upper_leg_L": "upper_leg_L",
    "lower_leg_L": "lower_leg_L",
    "foot_L": "foot_L",
    "upper_leg_R": "upper_leg_R",
    "lower_leg_R": "lower_leg_R",
    "foot_R": "foot_R",
    "skull": "skull",
    "jaw": "jaw",
    "scapula_L": "scapula_L",
    "upper_arm_L": "upper_arm_L",
    "lower_arm_L": "lower_arm_L",
    "hand_L": "hand_L",
    "scapula_R": "scapula_R",
    "upper_arm_R": "upper_arm_R",
    "lower_arm_R": "lower_arm_R",
    "hand_R": "hand_R",
}

# Site names in the compiled model
SITE_NAMES = {
    "shoulder_L": "shoulder_L",
    "elbow_L": "elbow_L",
    "shoulder_R": "shoulder_R",
    "elbow_R": "elbow_R",
    "hip_L": "hip_L",
    "hip_R": "hip_R",
    "knee_L": "knee_L",
    "knee_R": "knee_R",
}
