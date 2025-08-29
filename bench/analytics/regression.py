"""
Performance regression analysis and detection.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from .statistics import StatisticalAnalyzer


@dataclass
class RegressionAlert:
    """Performance regression detection result."""

    detected: bool
    severity: str  # "minor", "moderate", "severe"
    affected_tasks: list[str]
    performance_drop: float
    confidence: float
    recommended_actions: list[str]
    statistical_evidence: dict[str, Any]
    timestamp: datetime


class RegressionAnalyzer:
    """Automated performance regression detection."""

    def __init__(self, config: dict[str, Any]):
        """Initialize with configuration thresholds."""
        self.thresholds = config.get(
            "thresholds",
            {
                "minor": 0.05,  # 5% drop
                "moderate": 0.10,  # 10% drop
                "severe": 0.20,  # 20% drop
            },
        )
        self.min_samples = config.get("min_samples", 5)
        self.confidence_level = config.get("confidence_level", 0.95)
        self.statistical_analyzer = StatisticalAnalyzer(alpha=1 - self.confidence_level)

    def analyze_regression(
        self,
        baseline_results: list[dict[str, Any]],
        current_results: list[dict[str, Any]],
    ) -> RegressionAlert:
        """Detect and classify performance regressions."""
        if (
            len(baseline_results) < self.min_samples
            or len(current_results) < self.min_samples
        ):
            return RegressionAlert(
                detected=False,
                severity="none",
                affected_tasks=[],
                performance_drop=0.0,
                confidence=0.0,
                recommended_actions=["Insufficient data for regression analysis"],
                statistical_evidence={},
                timestamp=datetime.now(),
            )

        # Extract scores by task
        baseline_by_task: dict[str, list[float]] = self._group_by_provider_task(
            baseline_results
        )
        current_by_task: dict[str, list[float]] = self._group_by_provider_task(
            current_results
        )

        # Analyze each task for regression
        task_regressions = {}
        affected_tasks = []
        max_drop = 0.0

        for task in baseline_by_task:
            if task in current_by_task:
                regression_result = self._analyze_task_regression(
                    baseline_by_task[task], current_by_task[task], task
                )
                task_regressions[task] = regression_result

                if regression_result["regression_detected"]:
                    affected_tasks.append(task)
                    max_drop = max(max_drop, abs(regression_result["performance_drop"]))

        # Classify regression severity
        baseline_mean = np.mean(
            [result["baseline_mean"] for result in task_regressions.values()]
        )
        current_mean = np.mean(
            [result["current_mean"] for result in task_regressions.values()]
        )
        severity_info = self._calculate_regression_severity(baseline_mean, current_mean)
        severity = severity_info["severity"]
        detected = len(affected_tasks) > 0

        # Calculate overall confidence
        confidence = self._calculate_overall_confidence(task_regressions)

        # Generate recommendations
        recommendations = self.generate_recommendations(
            {
                "affected_tasks": affected_tasks,
                "severity": severity,
                "max_drop": max_drop,
                "task_details": task_regressions,
            }
        )

        return RegressionAlert(
            detected=detected,
            severity=severity,
            affected_tasks=affected_tasks,
            performance_drop=max_drop,
            confidence=confidence,
            recommended_actions=recommendations,
            statistical_evidence=task_regressions,
            timestamp=datetime.now(),
        )

    def generate_recommendations(self, regression_data: dict[str, Any]) -> list[str]:
        """Generate actionable recommendations for addressing regressions."""
        recommendations = []

        severity = regression_data["severity"]
        affected_tasks = regression_data["affected_tasks"]
        max_drop = regression_data["max_drop"]

        if severity == "severe":
            recommendations.extend(
                [
                    "🚨 URGENT: Severe performance regression detected",
                    "Consider rolling back recent changes immediately",
                    "Investigate system resource constraints",
                    "Review recent configuration changes",
                ]
            )
        elif severity == "moderate":
            recommendations.extend(
                [
                    "⚠️ Moderate performance regression detected",
                    "Review recent code changes for performance impact",
                    "Check for increased system load or resource contention",
                    "Consider performance profiling of affected components",
                ]
            )
        elif severity == "minor":
            recommendations.extend(
                [
                    "ℹ️ Minor performance regression detected",
                    "Monitor trend to ensure it doesn't worsen",
                    "Review recent changes for potential optimizations",
                ]
            )

        # Task-specific recommendations
        if "task1_metrics" in affected_tasks:
            recommendations.append(
                "• CSV processing performance degraded - check data parsing efficiency"
            )

        if "task2_ssn_regex" in affected_tasks:
            recommendations.append(
                "• Regex performance degraded - review regex complexity "
                "and timeout settings"
            )

        if "task3_exec_summary" in affected_tasks:
            recommendations.append(
                "• Text generation performance degraded - check model response times"
            )

        # Statistical recommendations
        recommendations.extend(
            [
                f"Performance drop: {max_drop:.1%}",
                f"Affected tasks: {len(affected_tasks)} out of "
                f"{len(regression_data.get('task_details', {}))}",
            ]
        )

        return recommendations

    def detect_regressions(
        self, baseline_data: list[dict[str, Any]], current_data: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Detect performance regressions between baseline and current data."""
        # Group data by provider and task
        baseline_grouped = self._group_by_provider_task(baseline_data)
        current_grouped = self._group_by_provider_task(current_data)

        task_regressions = {}
        overall_regression = False

        # Analyze each task for regressions
        for task_id in baseline_grouped:
            if task_id in current_grouped:
                baseline_scores = baseline_grouped[task_id]
                current_scores = current_grouped[task_id]

                if (
                    len(baseline_scores) >= self.min_samples
                    and len(current_scores) >= self.min_samples
                ):
                    regression_result = self._analyze_task_regression(
                        baseline_scores, current_scores, task_id
                    )

                    if regression_result["regression_detected"]:
                        task_regressions[task_id] = regression_result
                        overall_regression = True

        return {
            "overall_regression": overall_regression,
            "task_regressions": task_regressions,
            "confidence": self._calculate_overall_confidence(task_regressions),
            "timestamp": datetime.now(),
        }

    def _group_by_provider_task(
        self, data: list[dict[str, Any]]
    ) -> dict[str, list[float]]:
        """Group evaluation results by task and extract scores."""
        grouped: dict[str, list[float]] = {}

        for result in data:
            task_id = result.get("task_id", "unknown")
            score = result.get("total_score", result.get("score", 0))

            if task_id not in grouped:
                grouped[task_id] = []
            grouped[task_id].append(float(score))

        return grouped

    def _analyze_task_regression(
        self, baseline_scores: list[float], current_scores: list[float], task_name: str
    ) -> dict[str, Any]:
        """Analyze regression for a specific task."""
        # Perform statistical test
        stat_result = self.statistical_analyzer.regression_detection(
            baseline_scores, current_scores
        )

        # Calculate performance metrics
        baseline_mean = np.mean(baseline_scores)
        current_mean = np.mean(current_scores)
        performance_drop = (
            (baseline_mean - current_mean) / baseline_mean if baseline_mean > 0 else 0
        )

        # Determine if regression is detected
        regression_detected = (
            stat_result.significant
            and performance_drop > self.thresholds["minor"]
            and current_mean < baseline_mean
        )

        return {
            "task_name": task_name,
            "regression_detected": regression_detected,
            "performance_drop": performance_drop,
            "baseline_mean": baseline_mean,
            "current_mean": current_mean,
            "baseline_std": np.std(baseline_scores, ddof=1),
            "current_std": np.std(current_scores, ddof=1),
            "statistical_test": {
                "test_name": stat_result.test_name,
                "p_value": stat_result.p_value,
                "effect_size": stat_result.effect_size,
                "confidence_interval": stat_result.confidence_interval,
                "significant": stat_result.significant,
            },
            "sample_sizes": {
                "baseline": len(baseline_scores),
                "current": len(current_scores),
            },
        }

    def _calculate_regression_severity(
        self, baseline_mean: float, current_mean: float
    ) -> dict[str, Any]:
        """Classify regression severity based on performance drop."""
        performance_drop = (
            (baseline_mean - current_mean) / baseline_mean if baseline_mean > 0 else 0
        )

        if performance_drop >= self.thresholds["severe"]:
            return {"severity": "severe", "performance_drop": performance_drop}
        elif performance_drop >= self.thresholds["moderate"]:
            return {"severity": "moderate", "performance_drop": performance_drop}
        elif performance_drop >= self.thresholds["minor"]:
            return {"severity": "minor", "performance_drop": performance_drop}
        else:
            return {"severity": "none", "performance_drop": performance_drop}

    def _calculate_overall_confidence(
        self, task_regressions: dict[str, dict[str, Any]]
    ) -> float:
        """Calculate overall confidence in regression detection."""
        if not task_regressions:
            return 0.0

        # Average confidence across tasks (1 - p_value for significant results)
        confidences = []

        for task_data in task_regressions.values():
            if task_data["regression_detected"]:
                p_value = task_data["statistical_test"]["p_value"]
                confidence = 1 - p_value
                confidences.append(confidence)

        return float(np.mean(confidences)) if confidences else 0.0

    def detect_performance_drift(
        self, historical_scores: list[tuple[datetime, float]], window_size: int = 10
    ) -> dict[str, Any]:
        """Detect gradual performance drift using control charts."""
        if len(historical_scores) < window_size * 2:
            return {"drift_detected": False, "reason": "Insufficient data"}

        # Sort by timestamp
        sorted_scores = sorted(historical_scores, key=lambda x: x[0])
        timestamps, scores = zip(*sorted_scores, strict=False)
        scores = np.array(scores)

        # Calculate control limits using initial window
        initial_window = scores[:window_size]
        center_line = np.mean(initial_window)
        std_dev = np.std(initial_window, ddof=1)

        # 3-sigma control limits
        upper_control_limit = center_line + 3 * std_dev
        lower_control_limit = center_line - 3 * std_dev

        # Check for points outside control limits
        violations = []
        for i, score in enumerate(scores):
            if score > upper_control_limit or score < lower_control_limit:
                violations.append(i)

        # Check for trends (7 consecutive points on same side of center line)
        trend_violations = self._detect_trend_violations(scores, center_line)

        drift_detected = len(violations) > 0 or len(trend_violations) > 0

        return {
            "drift_detected": drift_detected,
            "control_limits": {
                "center": center_line,
                "upper": upper_control_limit,
                "lower": lower_control_limit,
            },
            "violations": {
                "out_of_control": violations,
                "trend_violations": trend_violations,
            },
            "drift_severity": self._assess_drift_severity(
                violations, trend_violations, len(scores)
            ),
        }

    def _detect_trend_violations(
        self, scores: np.ndarray, center_line: float
    ) -> list[int]:
        """Detect trend violations (7+ consecutive points on same side of center)."""
        violations = []
        current_streak = 0
        current_side = None

        for i, score in enumerate(scores):
            side = "above" if score > center_line else "below"

            if side == current_side:
                current_streak += 1
            else:
                current_side = side
                current_streak = 1

            if current_streak >= 7:
                violations.append(i)

        return violations

    def _assess_drift_severity(
        self, violations: list[int], trend_violations: list[int], total_points: int
    ) -> str:
        """Assess severity of performance drift."""
        violation_rate = (len(violations) + len(trend_violations)) / total_points

        if violation_rate > 0.2:  # More than 20% of points violate
            return "severe"
        elif violation_rate > 0.1:  # More than 10% of points violate
            return "moderate"
        elif violation_rate > 0.05:  # More than 5% of points violate
            return "minor"
        else:
            return "none"
