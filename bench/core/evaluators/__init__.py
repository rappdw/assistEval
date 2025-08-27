"""Task-specific evaluators for benchmark scoring.

This package contains evaluators for different task types, each implementing
objective scoring logic for specific evaluation criteria.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bench.core.evaluators.base import Evaluator

__all__ = ["Evaluator"]
