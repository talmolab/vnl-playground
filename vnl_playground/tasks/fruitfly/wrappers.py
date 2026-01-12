"""Wrappers for fruitfly environments.

Re-exports common wrappers from rodent module since they are generic
and work with any MjxEnv-based environment.
"""

from vnl_playground.tasks.rodent.wrappers import FlattenObsWrapper, HighLevelWrapper

__all__ = ["FlattenObsWrapper", "HighLevelWrapper"]
