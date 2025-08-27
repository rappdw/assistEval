"""Base evaluator interface for task-specific scoring.

This module defines the abstract interface that all task evaluators must implement
to ensure consistent scoring across different evaluation tasks.
"""

from abc import ABC, abstractmethod
from typing import Any


class Evaluator(ABC):
    """Abstract base class for task evaluators.

    All evaluator implementations must inherit from this class and implement
    the evaluate method to provide objective scoring for specific tasks.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the evaluator with configuration.

        Args:
            config: Evaluator-specific configuration from test definition
        """
        self.config = config

    @abstractmethod
    def evaluate(
        self, response: dict[str, Any], expected: dict[str, Any]
    ) -> dict[str, Any]:
        """Evaluate response against expected results.

        Args:
            response: Parsed response from AI provider
            expected: Expected results from answer key

        Returns:
            Dictionary with score breakdown and details

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement evaluate method")

    def __repr__(self) -> str:
        """String representation of the evaluator."""
        return f"{self.__class__.__name__}()"
