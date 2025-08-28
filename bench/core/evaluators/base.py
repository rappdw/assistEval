"""Base evaluator interface for task-specific scoring.

This module defines the abstract interface that all task evaluators must implement
to ensure consistent scoring across different evaluation tasks.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationResult:
    """Result of task evaluation with detailed scoring."""

    task_id: str
    total_score: float
    max_score: float
    sub_scores: dict[str, float]
    details: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def score_percentage(self) -> float:
        """Calculate score as percentage."""
        if self.max_score == 0:
            return 0.0
        return (self.total_score / self.max_score) * 100.0

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)


class BaseEvaluator(ABC):
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
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def evaluate(
        self,
        response_data: dict[str, Any],
        test_case: dict[str, Any],
        answer_key: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate provider response against test case requirements.

        Args:
            response_data: Validated response data from provider
            test_case: Test case definition with expectations
            answer_key: Optional answer key with expected results

        Returns:
            EvaluationResult with detailed scoring breakdown

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement evaluate method")

    def calculate_weighted_score(
        self, sub_scores: dict[str, float], weights: dict[str, float]
    ) -> float:
        """Calculate weighted total score from sub-component scores.

        Args:
            sub_scores: Dictionary of component scores
            weights: Dictionary of component weights

        Returns:
            Weighted total score
        """
        total = 0.0
        for component, score in sub_scores.items():
            weight = weights.get(component, 0.0)
            total += score * weight
        return total

    def safe_extract_field(
        self, data: dict[str, Any], field_path: str, default: Any = None
    ) -> Any:
        """Safely extract field from nested dictionary.

        Args:
            data: Source dictionary
            field_path: Dot-separated field path (e.g., 'metrics.precision')
            default: Default value if field not found

        Returns:
            Field value or default
        """
        try:
            current = data
            for key in field_path.split("."):
                current = current[key]
            return current
        except (KeyError, TypeError, AttributeError):
            return default

    def __repr__(self) -> str:
        """String representation of the evaluator."""
        return f"{self.__class__.__name__}()"
