"""Constants for modular rodent environments."""

from etils import epath

MODULAR_RODENT_PATH = epath.Path(__file__).parent
MODULAR_RODENT_XML_PATH = MODULAR_RODENT_PATH / "xmls" / "rodent_extra_sensors.xml"
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

# Body names in the compiled model (walker/ prefix from the standalone XML)
BODY_NAMES = {
    "torso": "walker/torso",
    "pelvis": "walker/pelvis",
    "upper_leg_L": "walker/upper_leg_L",
    "lower_leg_L": "walker/lower_leg_L",
    "foot_L": "walker/foot_L",
    "upper_leg_R": "walker/upper_leg_R",
    "lower_leg_R": "walker/lower_leg_R",
    "foot_R": "walker/foot_R",
    "skull": "walker/skull",
    "jaw": "walker/jaw",
    "scapula_L": "walker/scapula_L",
    "upper_arm_L": "walker/upper_arm_L",
    "lower_arm_L": "walker/lower_arm_L",
    "hand_L": "walker/hand_L",
    "scapula_R": "walker/scapula_R",
    "upper_arm_R": "walker/upper_arm_R",
    "lower_arm_R": "walker/lower_arm_R",
    "hand_R": "walker/hand_R",
}

# Site names in the compiled model (walker/ prefix)
SITE_NAMES = {
    "shoulder_L": "walker/shoulder_L",
    "elbow_L": "walker/elbow_L",
    "shoulder_R": "walker/shoulder_R",
    "elbow_R": "walker/elbow_R",
    "hip_L": "walker/hip_L",
    "hip_R": "walker/hip_R",
    "knee_L": "walker/knee_L",
    "knee_R": "walker/knee_R",
}
