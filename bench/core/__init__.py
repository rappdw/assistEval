"""Core evaluation framework components.

This package contains the core logic for running evaluations, validating results,
scoring performance, and generating reports.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bench.core.reporting import Reporter
    from bench.core.runner import TestRunner
    from bench.core.scoring import ScoringEngine

__all__ = ["TestRunner", "ScoringEngine", "Reporter"]
