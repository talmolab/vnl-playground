"""Defines rodents constants."""

from etils import epath

from mujoco_playground._src import mjx_env

MOUSE_PATH = epath.Path(__file__).parent

MOUSE_XML_PATH = MOUSE_PATH / "xmls" / "akira_torque.xml"
