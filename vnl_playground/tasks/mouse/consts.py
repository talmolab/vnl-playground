"""Defines mouse constants (filesystem paths only).

Model-specific constants (body names, joint names, tracked bodies) should be
specified via YAML config files for flexibility in parameter sweeps.
See vnl_playground/config/mouse_imitation.yaml for an example.
"""

from etils import epath

MOUSE_PATH = epath.Path(__file__).parent

MOUSE_XML_PATH = MOUSE_PATH / "xmls" / "akira_muscle.xml"
JANELIA_MOUSE_XML_PATH = MOUSE_PATH / "xmls" / "mouse_forelimb_right.xml"
MOUSE_ARENA_XML_PATH = MOUSE_PATH / "xmls" / "arena.xml"
MOUSE_REFERENCE_DATA_PATH = MOUSE_PATH / "reference_data"
