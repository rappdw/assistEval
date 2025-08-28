"""Task-specific evaluators for benchmark scoring.

This package contains evaluators for different task types, each implementing
objective scoring logic for specific evaluation criteria.
"""

from typing import TYPE_CHECKING, Any

from bench.core.evaluators.base import BaseEvaluator, EvaluationResult
from bench.core.evaluators.exec_summary import ExecSummaryEvaluator
from bench.core.evaluators.metrics_csv import MetricsCSVEvaluator
from bench.core.evaluators.regex_match import RegexMatchEvaluator

if TYPE_CHECKING:
    pass


class EvaluatorRegistry:
    """Registry for task evaluators with dynamic loading support."""

    _evaluators: dict[str, type[BaseEvaluator]] = {}

    @classmethod
    def register(cls, name: str, evaluator_class: type[BaseEvaluator]) -> None:
        """Register an evaluator class.

        Args:
            name: Unique name for the evaluator
            evaluator_class: Evaluator class to register
        """
        if not issubclass(evaluator_class, BaseEvaluator):
            raise ValueError(
                f"Evaluator {evaluator_class} must inherit from BaseEvaluator"
            )

        cls._evaluators[name] = evaluator_class

    @classmethod
    def create_evaluator(cls, name: str, config: dict[str, Any]) -> BaseEvaluator:
        """Create evaluator instance by name.

        Args:
            name: Name of the evaluator to create
            config: Configuration for the evaluator

        Returns:
            Configured evaluator instance

        Raises:
            KeyError: If evaluator name not found
        """
        if name not in cls._evaluators:
            available = list(cls._evaluators.keys())
            raise KeyError(f"Evaluator '{name}' not found. Available: {available}")

        evaluator_class = cls._evaluators[name]
        return evaluator_class(config)

    @classmethod
    def list_evaluators(cls) -> list[str]:
        """List available evaluator names.

        Returns:
            List of registered evaluator names
        """
        return list(cls._evaluators.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if evaluator is registered.

        Args:
            name: Evaluator name to check

        Returns:
            True if evaluator is registered
        """
        return name in cls._evaluators


# Auto-register built-in evaluators
EvaluatorRegistry.register("metrics_csv", MetricsCSVEvaluator)
EvaluatorRegistry.register("regex_match", RegexMatchEvaluator)
EvaluatorRegistry.register("exec_summary", ExecSummaryEvaluator)

__all__ = [
    "BaseEvaluator",
    "EvaluationResult",
    "EvaluatorRegistry",
    "MetricsCSVEvaluator",
    "RegexMatchEvaluator",
    "ExecSummaryEvaluator",
]
