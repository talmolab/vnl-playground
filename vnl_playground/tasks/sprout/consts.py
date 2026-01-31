"""Defines Sprout humanoid robot constants.

Constants derived from the Fauna Robotics Sprout MJCF model (robot.mjcf)
and Fauna simulation documentation.
"""

from etils import epath

from mujoco_playground._src import mjx_env

SPROUT_PATH = epath.Path(__file__).parent

SPROUT_XML_PATH = SPROUT_PATH / "xmls" / "sprout.xml"
ARENA_XML_PATH = SPROUT_PATH / "xmls" / "arena.xml"

# End effectors for egocentric position tracking
END_EFFECTORS = [
    "left_foot_link",
    "right_foot_link",
    "left_gripper_upper_link",
    "right_gripper_upper_link",
    "head_link",
]

# All body links in the kinematic tree
BODIES = [
    "torso_link",
    "pelvis_link",
    # Left leg
    "left_hip_link",
    "left_femur_upper_link",
    "left_femur_lower_link",
    "left_tibia_link",
    "left_foot_link",
    # Right leg
    "right_hip_link",
    "right_femur_upper_link",
    "right_femur_lower_link",
    "right_tibia_link",
    "right_foot_link",
    # Head/Neck
    "neck_link",
    "head_link",
    # Left arm
    "left_shoulder_link",
    "left_humerus_upper_link",
    "left_humerus_lower_link",
    "left_radius_link",
    "left_wrist_link",
    "left_gripper_lower_link",
    "left_gripper_upper_link",
    # Right arm
    "right_shoulder_link",
    "right_humerus_upper_link",
    "right_humerus_lower_link",
    "right_radius_link",
    "right_wrist_link",
    "right_gripper_lower_link",
    "right_gripper_upper_link",
]

# Default standing pose from Fauna simulation documentation (radians).
# Joints not listed default to 0.0.
STANDING_POSE = {
    "left_hip_pitch_joint": -0.06981317,
    "right_hip_pitch_joint": -0.06981317,
    "left_hip_roll_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "left_knee_joint": 0.122173048,
    "right_knee_joint": 0.122173048,
    "left_ankle_joint": -0.034906585,
    "right_ankle_joint": -0.034906585,
    "left_hip_yaw_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "left_shoulder_pitch_joint": 0.073303829,
    "right_shoulder_pitch_joint": 0.073303829,
    "left_shoulder_roll_joint": -1.420698011,
    "right_shoulder_roll_joint": 1.420698011,
    "waist_yaw_joint": 0.0,
    "left_shoulder_yaw_joint": 0.212930169,
    "right_shoulder_yaw_joint": 0.212930169,
    "left_elbow_joint": -0.174532925,
    "right_elbow_joint": 0.174532925,
    "left_wrist_roll_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "left_gripper_joint": 0.0,
    "right_gripper_joint": 0.0,
    "neck_yaw_joint": 0.0,
    "neck_pitch_joint": 0.0,
}

# Motor group parameters from Fauna simulation documentation.
# Maps motor type to (effort_limit, velocity_limit, saturation_effort).
MOTOR_PARAMETERS = {
    "48V-EC-A6408": {
        "effort_limit": 30,
        "velocity_limit": 15.4,
        "saturation_effort": 128.2,
        "joints": [
            "left_hip_pitch_joint",
            "right_hip_pitch_joint",
            "left_hip_roll_joint",
            "right_hip_roll_joint",
            "left_knee_joint",
            "right_knee_joint",
            "left_ankle_joint",
            "right_ankle_joint",
        ],
    },
    "48V-EC-A4301": {
        "effort_limit": 24,
        "velocity_limit": 19.04,
        "saturation_effort": 105.12,
        "joints": [
            "left_shoulder_pitch_joint",
            "right_shoulder_pitch_joint",
            "left_hip_yaw_joint",
            "right_hip_yaw_joint",
            "waist_yaw_joint",
        ],
    },
    "48V-HTDW-5047-36": {
        "effort_limit": 30,
        "velocity_limit": 15.72,
        "saturation_effort": 114.72,
        "joints": [
            "left_shoulder_roll_joint",
            "right_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "right_shoulder_yaw_joint",
            "left_elbow_joint",
            "right_elbow_joint",
        ],
    },
    "48V-HTDW-4438-32": {
        "effort_limit": 18,
        "velocity_limit": 37.48,
        "saturation_effort": 28.50,
        "joints": [
            "neck_yaw_joint",
            "left_wrist_roll_joint",
            "right_wrist_roll_joint",
        ],
    },
    "48V-HTDW-3536-32": {
        "effort_limit": 3.5,
        "velocity_limit": 17.5,
        "saturation_effort": 4.5,
        "joints": [
            "neck_pitch_joint",
            "left_gripper_joint",
            "right_gripper_joint",
            "left_wrist_pitch_joint",
            "right_wrist_pitch_joint",
        ],
    },
}

# Default PD gains from Fauna simulation documentation
DEFAULT_STIFFNESS = 32.5
DEFAULT_DAMPING = 1.0

# Spawn height for standing pose (from Fauna docs: pos=(0, 0, 0.7))
SPAWN_HEIGHT = 0.7
