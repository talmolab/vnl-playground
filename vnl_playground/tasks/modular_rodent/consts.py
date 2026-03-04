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

# Ordered actuator names per action module, matching null_action() output dimensions.
# head is 5 elements: [cervical_extend, cervical_bend, cervical_twist, atlas, mandible]
# (cervical maps to ctrl[3:6], atlas+mandible to ctrl[20:22] — non-contiguous in XML order)
MODULE_ACTUATOR_NAMES: dict[str, list[str]] = {
    "torso":  ["lumbar_extend", "lumbar_bend", "lumbar_twist"],
    "head":   ["cervical_extend", "cervical_bend", "cervical_twist", "atlas", "mandible"],
    "leg_L":  ["hip_L_supinate", "hip_L_abduct", "hip_L_extend", "knee_L"],
    "foot_L": ["ankle_L", "toe_L"],
    "leg_R":  ["hip_R_supinate", "hip_R_abduct", "hip_R_extend", "knee_R"],
    "foot_R": ["ankle_R", "toe_R"],
    "arm_L":  ["scapula_L_supinate", "scapula_L_abduct", "scapula_L_extend",
               "shoulder_L", "shoulder_sup_L", "elbow_L"],
    "hand_L": ["wrist_L", "finger_L"],
    "arm_R":  ["scapula_R_supinate", "scapula_R_abduct", "scapula_R_extend",
               "shoulder_R", "shoulder_sup_R", "elbow_R"],
    "hand_R": ["wrist_R", "finger_R"],
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
