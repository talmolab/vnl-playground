"""Registry for reward and termination functions."""

from typing import Callable


class RewardRegistry:
    """Registry for reward and termination functions.

    This class provides a decorator-based pattern for registering
    reward, cost, and termination functions that can be looked up by name.
    """

    def __init__(self):
        self.rewards: dict[str, Callable] = {}
        self.costs: dict[str, Callable] = {}
        self.terminations: dict[str, Callable] = {}

    def reward(self, name: str):
        """Decorator to register a reward function.

        Args:
            name: The name to register the reward function under.

        Returns:
            Decorator function that registers and returns the reward function.
        """

        def decorator(fn: Callable) -> Callable:
            self.rewards[name] = fn
            return fn

        return decorator

    def cost(self, name: str):
        """Decorator to register a cost function.

        Args:
            name: The name to register the cost function under.

        Returns:
            Decorator function that registers and returns the cost function.
        """

        def decorator(fn: Callable) -> Callable:
            self.costs[name] = fn
            return fn

        return decorator

    def termination(self, name: str):
        """Decorator to register a termination criterion.

        Args:
            name: The name to register the termination function under.

        Returns:
            Decorator function that registers and returns the termination function.
        """

        def decorator(fn: Callable) -> Callable:
            self.terminations[name] = fn
            return fn

        return decorator
