"""Constants for the PlanarWalker environment."""

from mujoco_playground._src import mjx_env

# The walker XML is in mujoco_playground's dm_control_suite
WALKER_XML_PATH = (
    mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "walker.xml"
)

# Body names (excluding worldbody). Order matches walker.xml body tree.
BODIES = [
    "torso",
    "right_thigh",
    "right_leg",
    "right_foot",
    "left_thigh",
    "left_leg",
    "left_foot",
]

END_EFFECTORS = ["right_foot", "left_foot"]

# Joint names (excluding root joints: rootz, rootx, rooty)
JOINT_NAMES = [
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
]

# Root DOFs: [rootz (slide z), rootx (slide x), rooty (hinge y)]
N_ROOT_QPOS = 3  # 3 root qpos: z-pos, x-pos, y-angle
N_ROOT_QVEL = 3  # 3 root qvel: same ordering
N_JOINTS = 6     # 6 leg joints

# Behavior mode indices (for one-hot encoding)
BEHAVIOR_MODES = {
    "stand": 0,
    "walk_slow": 1,
    "run": 2,
    "knee_down": 3,
}
N_BEHAVIOR_MODES = len(BEHAVIOR_MODES)

# Target speeds for locomotion modes (m/s)
WALK_SLOW_SPEED = 0.5
RUN_SPEED = 8.0

# Physical constants for reward computation
STAND_HEIGHT = 1.2          # Minimum torso height for standing
KNEE_DOWN_HEIGHT_RANGE = (0.5, 0.9)  # Torso height range for knee_down
KNEE_DOWN_ANGLE_RANGE = (-2.09, -1.05)  # Knee joint angle range: ~(-120, -60) degrees in radians
