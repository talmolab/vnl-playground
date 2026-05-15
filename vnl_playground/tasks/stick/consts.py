"""Defines stick bug (Sungaya inexpectata) constants."""

from etils import epath

STICK_PATH = epath.Path(__file__).parent

# Default walker is the 41-DoF mesh model; the box variant is kept available
# at STICK_BOX_XML_PATH for users who want to use the older model.
STICK_XML_PATH = STICK_PATH / "xmls" / "stick_mesh_fast.xml"
STICK_BOX_XML_PATH = STICK_PATH / "xmls" / "stick_fast.xml"
ARENA_XML_PATH = STICK_PATH / "xmls" / "arena.xml"

# Reference data path for imitation learning (legacy STAC-fit format)
IMITATION_REFERENCE_PATH = STICK_PATH / "reference_data" / "full_stick.h5"

# 41 joints (3 thorax + 8 abdomen + 30 leg).
# Thorax joints 04-t1-l/05-t2-l/06-t3-l are now active in the mesh model
# (they were commented out in sungaya_inexpectata_box.xml).
JOINTS = [
    # Thorax (newly enabled in the mesh model)
    "04-t1-l",
    "05-t2-l",
    "06-t3-l",
    # Abdomen
    "07-a2-l",
    "08-a3-l",
    "09-a4-l",
    "10-a5-l",
    "11-a6-l",
    "12-a7-l",
    "13-a8-l",
    "14-a9-l",
    # Hind left leg
    "26-h-l-coxa-l",
    "27-h-l-femur-l",
    "28-h-l-tibia-l",
    "29-h-l-tarsus-l",
    "30-h-l-claws-l",
    # Hind right leg
    "42-h-r-coxa-l",
    "43-h-r-femur-l",
    "44-h-r-tibia-l",
    "45-h-r-tarsus-l",
    "46-h-r-claws-l",
    # Middle left leg
    "21-m-l-coxa-l",
    "22-m-l-femur-l",
    "23-m-l-tibia-l",
    "24-m-l-tarsus-l",
    "25-m-l-claws-l",
    # Middle right leg
    "37-m-r-coxa-l",
    "38-m-r-femur-l",
    "39-m-r-tibia-l",
    "40-m-r-tarsus-l",
    "41-m-r-claws-l",
    # Front left leg
    "16-f-l-coxa-l",
    "17-f-l-femur-l",
    "18-f-l-tibia-l",
    "19-f-l-tarsus-l",
    "20-f-l-claws-l",
    # Front right leg
    "31-f-r-coxa-l",
    "32-f-r-femur-l",
    "33-f-r-tibia-l",
    "35-f-r-tarsus-l",
    "36-f-r-claws-l",
]

# Body names (excluding "world" and "floor"). Same tree as the box model.
BODIES = [
    "reference_base",
    "04-t1-l",
    "05-t2-l",
    "06-t3-l",
    "07-a2-l",
    "08-a3-l",
    "09-a4-l",
    "10-a5-l",
    "11-a6-l",
    "12-a7-l",
    "13-a8-l",
    "14-a9-l",
    # Hind left leg
    "26-h-l-coxa-l",
    "27-h-l-femur-l",
    "28-h-l-tibia-l",
    "29-h-l-tarsus-l",
    "30-h-l-claws-l",
    # Hind right leg
    "42-h-r-coxa-l",
    "43-h-r-femur-l",
    "44-h-r-tibia-l",
    "45-h-r-tarsus-l",
    "46-h-r-claws-l",
    # Middle left leg
    "21-m-l-coxa-l",
    "22-m-l-femur-l",
    "23-m-l-tibia-l",
    "24-m-l-tarsus-l",
    "25-m-l-claws-l",
    # Middle right leg
    "37-m-r-coxa-l",
    "38-m-r-femur-l",
    "39-m-r-tibia-l",
    "40-m-r-tarsus-l",
    "41-m-r-claws-l",
    # Front left leg
    "16-f-l-coxa-l",
    "17-f-l-femur-l",
    "18-f-l-tibia-l",
    "19-f-l-tarsus-l",
    "20-f-l-claws-l",
    # Front right leg
    "31-f-r-coxa-l",
    "32-f-r-femur-l",
    "33-f-r-tibia-l",
    "35-f-r-tarsus-l",
    "36-f-r-claws-l",
]

# 6 end effectors (claw bodies, one per leg)
END_EFFECTORS = [
    "20-f-l-claws-l",
    "36-f-r-claws-l",
    "25-m-l-claws-l",
    "41-m-r-claws-l",
    "30-h-l-claws-l",
    "46-h-r-claws-l",
]

# 24 leg-segment bodies (coxa, femur, tibia, tarsus for each of 6 legs).
# These correspond to the 4 non-claw joints per leg — "hip", "knee",
# "ankle", and tarsal joint. Tracking these augments the end-effector
# reward by also requiring the policy to match the full leg pose, not
# just the tip placement.
LEG_JOINTS = [
    # Hind left
    "26-h-l-coxa-l", "27-h-l-femur-l", "28-h-l-tibia-l", "29-h-l-tarsus-l",
    # Hind right
    "42-h-r-coxa-l", "43-h-r-femur-l", "44-h-r-tibia-l", "45-h-r-tarsus-l",
    # Middle left
    "21-m-l-coxa-l", "22-m-l-femur-l", "23-m-l-tibia-l", "24-m-l-tarsus-l",
    # Middle right
    "37-m-r-coxa-l", "38-m-r-femur-l", "39-m-r-tibia-l", "40-m-r-tarsus-l",
    # Front left
    "16-f-l-coxa-l", "17-f-l-femur-l", "18-f-l-tibia-l", "19-f-l-tarsus-l",
    # Front right
    "31-f-r-coxa-l", "32-f-r-femur-l", "33-f-r-tibia-l", "35-f-r-tarsus-l",
]

# Six explicit floor-contact geoms (one sphere primitive per claw body).
# base.py.add_stick() loops over FOOT_GEOMS and adds a pair
# (floor, <geom_name>-stick) for each entry, so the names below must match
# the geom names in sungaya_inexpectata_mesh.xml exactly.
FOOT_GEOMS = [
    "claw_collide_fl",
    "claw_collide_ml",
    "claw_collide_hl",
    "claw_collide_fr",
    "claw_collide_mr",
    "claw_collide_hr",
]
