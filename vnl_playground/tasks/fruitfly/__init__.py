"""Fruitfly task environments."""

from vnl_playground.tasks.fruitfly.base import FruitflyEnv
from vnl_playground.tasks.fruitfly.imitation import Imitation
from vnl_playground.tasks.fruitfly.reference_clips import ReferenceClips
from vnl_playground.tasks.fruitfly import consts

__all__ = ["FruitflyEnv", "Imitation", "ReferenceClips", "consts"]
