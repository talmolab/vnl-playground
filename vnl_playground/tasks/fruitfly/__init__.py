"""Fruitfly task environments."""

from vnl_playground.tasks.fruitfly.base import FruitflyEnv
from vnl_playground.tasks.fruitfly.imitation import Imitation
from vnl_playground.tasks.fruitfly.maintain_velocity import MaintainVelocity
from vnl_playground.tasks.fruitfly import consts

__all__ = ["FruitflyEnv", "Imitation", "MaintainVelocity", "consts"]
