"""Defines mouse constants."""

from etils import epath

from mujoco_playground._src import mjx_env

MOUSE_PATH = epath.Path(__file__).parent

MOUSE_XML_PATH = MOUSE_PATH / "xmls" / "akira_muscle.xml"
MOUSE_ARENA_XML_PATH = MOUSE_PATH / "xmls" / "arena.xml"
MOUSE_REFERENCE_DATA_PATH = MOUSE_PATH / "reference_data"

# Body names in the mouse arm model (order matches xpos in reference data)
BODY_NAMES = [
    "world",
    "ground",
    "clavicle",
    "scapula",
    "humerus",
    "ulna",
    "elbow",
    "radius",
    "wrist_body",
]

# Joint names (order matches qpos in reference data)
JOINT_NAMES = ["sh_elv", "sh_extension", "sh_rotation", "elbow_joint"]

# End effector for tracking (wrist)
END_EFFECTOR = "wrist_body"

# Bodies to track for imitation (excluding world/ground)
TRACKED_BODIES = ["scapula", "humerus", "ulna", "radius", "wrist_body"]
