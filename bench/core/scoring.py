"""Score aggregation and weighting system.

This module provides weighted score aggregation, stability bonus calculation,
and detailed score breakdowns for evaluation results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from bench.core.evaluators.base import EvaluationResult


@dataclass
class TaskScore:
    """Individual task scoring result."""

    task_id: str
    evaluator_name: str
    raw_score: float
    max_score: float
    weighted_score: float
    weight: float
    sub_scores: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def score_percentage(self) -> float:
        """Calculate score as percentage."""
        if self.max_score == 0:
            return 0.0
        return (self.raw_score / self.max_score) * 100.0


@dataclass
class ProviderScore:
    """Complete provider scoring result."""

    provider_name: str
    timestamp: datetime
    task_scores: list[TaskScore] = field(default_factory=list)
    total_score: float = 0.0
    max_score: float = 105.0
    stability_bonus: float = 0.0
    final_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def score_percentage(self) -> float:
        """Calculate final score as percentage."""
        if self.max_score == 0:
            return 0.0
        return (self.final_score / self.max_score) * 100.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "provider_name": self.provider_name,
            "timestamp": self.timestamp.isoformat(),
            "task_scores": [
                {
                    "task_id": ts.task_id,
                    "evaluator_name": ts.evaluator_name,
                    "raw_score": ts.raw_score,
                    "max_score": ts.max_score,
                    "weighted_score": ts.weighted_score,
                    "weight": ts.weight,
                    "sub_scores": ts.sub_scores,
                    "details": ts.details,
                    "errors": ts.errors,
                    "warnings": ts.warnings,
                }
                for ts in self.task_scores
            ],
            "total_score": self.total_score,
            "max_score": self.max_score,
            "stability_bonus": self.stability_bonus,
            "final_score": self.final_score,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProviderScore:
        """Create from dictionary."""
        task_scores = [
            TaskScore(
                task_id=ts["task_id"],
                evaluator_name=ts["evaluator_name"],
                raw_score=ts["raw_score"],
                max_score=ts["max_score"],
                weighted_score=ts["weighted_score"],
                weight=ts["weight"],
                sub_scores=ts["sub_scores"],
                details=ts["details"],
                errors=ts["errors"],
                warnings=ts["warnings"],
            )
            for ts in data["task_scores"]
        ]

        return cls(
            provider_name=data["provider_name"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            task_scores=task_scores,
            total_score=data["total_score"],
            max_score=data["max_score"],
            stability_bonus=data["stability_bonus"],
            final_score=data["final_score"],
            metadata=data["metadata"],
        )


class WeightConfig:
    """Configuration for scoring weights."""

    def __init__(self, weights: dict[str, Any] | None = None):
        """Initialize with weights configuration."""
        self.weights = weights or {}

    @classmethod
    def load_from_file(cls, config_path: Path) -> WeightConfig:
        """Load weights from YAML file."""
        try:
            with open(config_path) as f:
                weights = yaml.safe_load(f)
            return cls(weights)
        except FileNotFoundError:
            return cls()

    def get_task_weights(self, task_id: str) -> dict[str, Any]:
        """Get weights for a specific task."""
        task_mapping = {
            "offline.task1.metrics_csv": "task1_metrics",
            "offline.task2.ssn_regex": "task2_ssn_regex",
            "offline.task3.exec_summary": "task3_exec_summary",
            "online.deep_research": "deep_research",
        }

        task_key = task_mapping.get(task_id, task_id)
        tasks = self.weights.get("tasks", {})
        weights = tasks.get(task_key, {})
        return dict(weights) if isinstance(weights, dict) else {}

    def get_total_possible_score(self) -> float:
        """Calculate total possible score from all tasks."""
        total = 0.0
        tasks = self.weights.get("tasks", {})
        for task_weights in tasks.values():
            if isinstance(task_weights, dict):
                # Sum nested weights
                for value in task_weights.values():
                    if isinstance(value, int | float):
                        total += value
                    elif isinstance(value, dict):
                        total += sum(
                            v for v in value.values() if isinstance(v, int | float)
                        )

        # Add stability bonus
        stability_config = self.weights.get("stability_bonus", {})
        if isinstance(stability_config, dict):
            total += stability_config.get("max_points", 0)

        return total

    def get_stability_config(self) -> dict[str, Any]:
        """Get stability bonus configuration."""
        config = self.weights.get(
            "stability_bonus", {"max_points": 5, "threshold": 0.95, "enabled": True}
        )
        return (
            dict(config)
            if isinstance(config, dict)
            else {"max_points": 5, "threshold": 0.95, "enabled": True}
        )


class ScoringEngine:
    """Main scoring engine for weighted aggregation."""

    def __init__(self, weight_config: WeightConfig):
        """Initialize with weight configuration."""
        self.weight_config = weight_config
        self.stability_analyzer = StabilityAnalyzer(weight_config)

    def calculate_task_score(self, evaluation_result: EvaluationResult) -> TaskScore:
        """Calculate weighted score for a single task."""
        task_weights = self.weight_config.get_task_weights(evaluation_result.task_id)

        if not task_weights:
            # Unknown task - use raw score as weighted score
            parts = evaluation_result.task_id.split(".")
            task_id = parts[-1] if len(parts) > 1 else evaluation_result.task_id
            errors = list(evaluation_result.errors)
            errors.append(f"Unknown task: {evaluation_result.task_id}")
            return TaskScore(
                task_id=task_id,
                evaluator_name=self._extract_evaluator_name(evaluation_result),
                raw_score=evaluation_result.total_score,
                max_score=evaluation_result.max_score,
                weighted_score=evaluation_result.total_score,
                weight=1.0,
                sub_scores=dict(evaluation_result.sub_scores),
                details=evaluation_result.details,
                errors=errors,
                warnings=evaluation_result.warnings,
            )

        # Calculate weighted sub-scores
        weighted_sub_scores = {}
        total_weighted = 0.0
        processed_components = set()

        for component, score in evaluation_result.sub_scores.items():
            if component in task_weights and component not in processed_components:
                weight = task_weights[component]
                processed_components.add(component)

                # Handle nested weights (e.g., confusion_matrix)
                if isinstance(weight, dict):
                    nested_total = 0.0

                    # For confusion matrix, use dict data from sub_scores if available
                    if component == "confusion_matrix":
                        if isinstance(score, dict):  # type: ignore[unreachable]
                            for sub_component, sub_score in score.items():  # type: ignore[unreachable]
                                if sub_component in weight:
                                    sub_weight = weight[sub_component]
                                    weighted_value = float(sub_score) * float(
                                        sub_weight
                                    )
                                    weighted_sub_scores[
                                        f"{component}_{sub_component}"
                                    ] = weighted_value
                                    nested_total += weighted_value
                    # Fallback to details if sub_scores doesn't have dict data
                    elif (
                        component == "confusion_matrix"
                        and "confusion_matrix" in evaluation_result.details
                    ):
                        cm_data = evaluation_result.details["confusion_matrix"]
                        if isinstance(cm_data, dict):
                            for sub_component, sub_score in cm_data.items():
                                if sub_component in weight:
                                    sub_weight = weight[sub_component]
                                    weighted_value = float(sub_score) * float(
                                        sub_weight
                                    )
                                    weighted_sub_scores[
                                        f"{component}_{sub_component}"
                                    ] = weighted_value
                                    nested_total += weighted_value

                    weighted_sub_scores[component] = nested_total
                    total_weighted += nested_total
                else:
                    # Simple weight multiplication
                    weighted_value = float(score) * float(weight)
                    weighted_sub_scores[component] = weighted_value
                    total_weighted += weighted_value

        # Handle confusion_matrix weights if not processed and available in details
        if (
            "confusion_matrix" in task_weights
            and "confusion_matrix" not in processed_components
            and "confusion_matrix" in evaluation_result.details
        ):
            cm_weights = task_weights["confusion_matrix"]
            cm_data = evaluation_result.details["confusion_matrix"]
            if isinstance(cm_weights, dict) and isinstance(cm_data, dict):
                cm_total = 0.0
                for sub_component, sub_score in cm_data.items():
                    if sub_component in cm_weights:
                        sub_weight = cm_weights[sub_component]
                        weighted_value = float(sub_score) * float(sub_weight)
                        weighted_sub_scores[
                            f"confusion_matrix_{sub_component}"
                        ] = weighted_value
                        cm_total += weighted_value

                weighted_sub_scores["confusion_matrix"] = cm_total
                total_weighted += cm_total

        # Calculate total weight (handle nested dictionaries)
        total_weight = 0.0
        for value in task_weights.values():
            if isinstance(value, dict):
                total_weight += sum(float(v) for v in value.values())
            else:
                total_weight += float(value)

        return TaskScore(
            task_id=evaluation_result.task_id,
            evaluator_name=self._extract_evaluator_name(evaluation_result),
            raw_score=evaluation_result.total_score,
            max_score=evaluation_result.max_score,
            weighted_score=total_weighted,
            weight=total_weight,
            sub_scores=weighted_sub_scores,
            details=evaluation_result.details,
            errors=evaluation_result.errors,
            warnings=evaluation_result.warnings,
        )

    def calculate_provider_score(
        self, evaluation_results: list[EvaluationResult], provider_name: str
    ) -> ProviderScore:
        """Calculate complete provider score from evaluation results."""
        task_scores = []
        total_score = 0.0

        # Calculate individual task scores
        for result in evaluation_results:
            task_score = self.calculate_task_score(result)
            task_scores.append(task_score)
            total_score += task_score.weighted_score

        # Calculate stability bonus (placeholder - would need multi-run data)
        stability_bonus = 0.0

        # Calculate final score
        final_score = total_score + stability_bonus

        provider_score = ProviderScore(
            provider_name=provider_name,
            timestamp=datetime.now(),
            task_scores=task_scores,
            total_score=total_score,
            max_score=self.weight_config.get_total_possible_score(),
            stability_bonus=stability_bonus,
            final_score=final_score,
        )

        return provider_score

    def add_stability_bonus(
        self,
        provider_score: ProviderScore,
        multi_run_results: dict[str, list[EvaluationResult]],
    ) -> None:
        """Add stability bonus to provider score based on multi-run consistency."""
        stability_analyzer = StabilityAnalyzer(self.weight_config)
        stability_bonus = stability_analyzer.calculate_stability_bonus(
            multi_run_results
        )

        # Update provider score with stability bonus
        provider_score.stability_bonus = stability_bonus
        provider_score.final_score = provider_score.total_score + stability_bonus

    def validate_scores(self, provider_score: ProviderScore) -> list[str]:
        """Validate provider score for consistency."""
        errors = []

        # Check total score bounds
        if provider_score.total_score < 0:
            errors.append(f"Total score is negative: {provider_score.total_score}")
        if provider_score.total_score > provider_score.max_score:
            errors.append(
                f"Total score exceeds maximum: "
                f"{provider_score.total_score} > {provider_score.max_score}"
            )

        # Check stability bonus bounds
        stability_config = self.weight_config.get_stability_config()
        max_bonus = stability_config.get("max_points", 5)
        if provider_score.stability_bonus < 0:
            errors.append(
                f"Stability bonus is negative: {provider_score.stability_bonus}"
            )
        if provider_score.stability_bonus > max_bonus:
            errors.append(
                f"Stability bonus exceeds maximum: "
                f"{provider_score.stability_bonus} > {max_bonus}"
            )

        return errors

    def _extract_evaluator_name(self, evaluation_result: EvaluationResult) -> str:
        """Extract evaluator name from result metadata."""
        evaluator_name = evaluation_result.metadata.get("evaluator_name", "unknown")
        return str(evaluator_name) if evaluator_name is not None else "unknown"

    def _extract_task_id(self, task_id: str) -> str:
        """Extract task mapping from full task ID."""
        task_mapping = {
            "offline.task1.metrics_csv": "task1_metrics",
            "offline.task2.ssn_regex": "task2_ssn_regex",
            "offline.task3.exec_summary": "task3_exec_summary",
            "online.deep_research": "deep_research",
        }
        return task_mapping.get(task_id, task_id)


class StabilityAnalyzer:
    """Analyzes multi-run consistency for stability bonus."""

    def __init__(self, weight_config: WeightConfig):
        """Initialize with weight configuration."""
        self.weight_config = weight_config
        self.analysis_details: dict[str, Any] = {}

    def calculate_stability_bonus(
        self, multi_run_results: dict[str, list[EvaluationResult]]
    ) -> float:
        """Calculate stability bonus based on multi-run consistency."""
        stability_config = self.weight_config.get_stability_config()
        max_bonus = stability_config.get("max_points", 5)
        threshold = stability_config.get("threshold", 0.95)

        if not stability_config.get("enabled", True):
            return 0.0

        if not multi_run_results:
            return 0.0

        # Analyze consistency across tasks
        task1_consistency = self._analyze_task1_consistency(
            multi_run_results.get("offline.task1.metrics_csv", [])
        )
        structural_consistency = self._analyze_structural_consistency(multi_run_results)

        # Calculate overall consistency
        overall_consistency = (task1_consistency + structural_consistency) / 2

        # Store analysis details
        self.analysis_details = {
            "task1_consistency": task1_consistency,
            "structural_consistency": structural_consistency,
            "overall_consistency": overall_consistency,
            "threshold": threshold,
            "runs_analyzed": sum(
                len(results) for results in multi_run_results.values()
            ),
        }

        # Award bonus based on consistency
        if overall_consistency >= threshold:
            return float(max_bonus)
        else:
            # Linear scaling for partial consistency
            return float(max_bonus) * (float(overall_consistency) / float(threshold))

    def _analyze_task1_consistency(self, results: list[EvaluationResult]) -> float:
        """Check Task 1 numeric consistency (exact matches)."""
        if len(results) < 2:
            return 1.0

        # Compare numeric values across runs
        first_result = results[0]
        consistent_count = 0
        total_comparisons = 0

        for metric in ["precision", "recall", "f1", "accuracy"]:
            if metric in first_result.sub_scores:
                first_value = first_result.sub_scores[metric]
                for other_result in results[1:]:
                    if metric in other_result.sub_scores:
                        other_value = other_result.sub_scores[metric]
                        if abs(first_value - other_value) < 1e-6:
                            consistent_count += 1
                        total_comparisons += 1

        return consistent_count / total_comparisons if total_comparisons > 0 else 1.0

    def _analyze_structural_consistency(
        self, multi_run_results: dict[str, list[EvaluationResult]]
    ) -> float:
        """Check structural consistency (error patterns, warnings)."""
        if not multi_run_results:
            return 1.0

        consistent_runs = 0
        total_runs = 0

        for task_results in multi_run_results.values():
            if len(task_results) < 2:
                continue

            first_result = task_results[0]
            for other_result in task_results[1:]:
                # Check if error patterns are consistent
                if len(first_result.errors) == len(other_result.errors) and len(
                    first_result.warnings
                ) == len(other_result.warnings):
                    consistent_runs += 1
                total_runs += 1

        return consistent_runs / total_runs if total_runs > 0 else 1.0

    def get_analysis_details(self) -> dict[str, Any]:
        """Get detailed analysis results."""
        return getattr(self, "analysis_details", {})


class ScoreManager:
    """Manages score persistence and retrieval."""

    def __init__(self, results_dir: Path):
        """Initialize with results directory."""
        self.results_dir = Path(results_dir)

    def save_provider_score(self, provider_score: ProviderScore, run_dir: Path) -> None:
        """Save provider score to JSON file."""
        scores_dir = run_dir / "scores"
        scores_dir.mkdir(exist_ok=True)

        filename = f"{provider_score.provider_name}_score.json"
        filepath = scores_dir / filename

        with open(filepath, "w") as f:
            json.dump(provider_score.to_dict(), f, indent=2)

    def load_provider_scores(self, run_dir: Path) -> list[ProviderScore]:
        """Load all provider scores from run directory."""
        scores_dir = run_dir / "scores"
        if not scores_dir.exists():
            return []

        provider_scores = []
        for score_file in scores_dir.glob("*_score.json"):
            try:
                with open(score_file) as f:
                    data = json.load(f)
                provider_score = ProviderScore.from_dict(data)
                provider_scores.append(provider_score)
            except (json.JSONDecodeError, KeyError):
                # Skip invalid score files
                continue

        return provider_scores

    def get_score_history(self, provider_name: str) -> list[ProviderScore]:
        """Get historical scores for a provider across all runs."""
        history = []

        # Scan all run directories
        for run_dir in self.results_dir.glob("run_*"):
            if run_dir.is_dir():
                scores = self.load_provider_scores(run_dir)
                for score in scores:
                    if score.provider_name == provider_name:
                        history.append(score)

        # Sort by timestamp
        history.sort(key=lambda x: x.timestamp)
        return history

    def serialize_provider_score(self, provider_score: ProviderScore) -> str:
        """Serialize provider score to JSON string."""
        return json.dumps(provider_score.to_dict(), indent=2)

    def deserialize_provider_score(self, json_str: str) -> ProviderScore:
        """Deserialize provider score from JSON string."""
        data = json.loads(json_str)
        return ProviderScore.from_dict(data)
