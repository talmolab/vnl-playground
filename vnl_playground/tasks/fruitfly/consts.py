"""Defines fruitfly constants."""

from etils import epath

FRUITFLY_PATH = epath.Path(__file__).parent

FRUITFLY_XML_PATH = FRUITFLY_PATH / "xmls" / "fruitfly_fast.xml"
ARENA_XML_PATH = FRUITFLY_PATH / "xmls" / "arena.xml"
WHITE_ARENA_XML_PATH = FRUITFLY_PATH / "xmls" / "white_arena.xml"

# Reference data path for imitation learning
IMITATION_REFERENCE_PATH = (
    epath.Path(__file__).parent.parent.parent.parent.parent
    / "track-mjx"
    / "data"
    / "fruitfly"
    / "fly_reference_clip.h5"
)

# 36 joints (6 per leg x 3 legs x 2 sides)
JOINTS = [
    # T1 left
    "coxa_flexion_T1_left",
    "coxa_twist_T1_left",
    "femur_twist_T1_left",
    "femur_T1_left",
    "tibia_T1_left",
    "tarsus_T1_left",
    # T1 right
    "coxa_flexion_T1_right",
    "coxa_twist_T1_right",
    "femur_twist_T1_right",
    "femur_T1_right",
    "tibia_T1_right",
    "tarsus_T1_right",
    # T2 left
    "coxa_flexion_T2_left",
    "coxa_twist_T2_left",
    "femur_twist_T2_left",
    "femur_T2_left",
    "tibia_T2_left",
    "tarsus_T2_left",
    # T2 right
    "coxa_flexion_T2_right",
    "coxa_twist_T2_right",
    "femur_twist_T2_right",
    "femur_T2_right",
    "tibia_T2_right",
    "tarsus_T2_right",
    # T3 left
    "coxa_flexion_T3_left",
    "coxa_twist_T3_left",
    "femur_twist_T3_left",
    "femur_T3_left",
    "tibia_T3_left",
    "tarsus_T3_left",
    # T3 right
    "coxa_flexion_T3_right",
    "coxa_twist_T3_right",
    "femur_twist_T3_right",
    "femur_T3_right",
    "tibia_T3_right",
    "tarsus_T3_right",
]

# 48 bodies (8 segments per leg x 6 legs)
BODIES = [
    # T1 left
    "coxa_T1_left",
    "femur_T1_left",
    "tibia_T1_left",
    "tarsus_T1_left",
    "tarsus2_T1_left",
    "tarsus3_T1_left",
    "tarsus4_T1_left",
    "claw_T1_left",
    # T1 right
    "coxa_T1_right",
    "femur_T1_right",
    "tibia_T1_right",
    "tarsus_T1_right",
    "tarsus2_T1_right",
    "tarsus3_T1_right",
    "tarsus4_T1_right",
    "claw_T1_right",
    # T2 left
    "coxa_T2_left",
    "femur_T2_left",
    "tibia_T2_left",
    "tarsus_T2_left",
    "tarsus2_T2_left",
    "tarsus3_T2_left",
    "tarsus4_T2_left",
    "claw_T2_left",
    # T2 right
    "coxa_T2_right",
    "femur_T2_right",
    "tibia_T2_right",
    "tarsus_T2_right",
    "tarsus2_T2_right",
    "tarsus3_T2_right",
    "tarsus4_T2_right",
    "claw_T2_right",
    # T3 left
    "coxa_T3_left",
    "femur_T3_left",
    "tibia_T3_left",
    "tarsus_T3_left",
    "tarsus2_T3_left",
    "tarsus3_T3_left",
    "tarsus4_T3_left",
    "claw_T3_left",
    # T3 right
    "coxa_T3_right",
    "femur_T3_right",
    "tibia_T3_right",
    "tarsus_T3_right",
    "tarsus2_T3_right",
    "tarsus3_T3_right",
    "tarsus4_T3_right",
    "claw_T3_right",
]

# 6 end effectors (claw tips)
END_EFFECTORS = [
    "claw_T1_left",
    "claw_T1_right",
    "claw_T2_left",
    "claw_T2_right",
    "claw_T3_left",
    "claw_T3_right",
]
