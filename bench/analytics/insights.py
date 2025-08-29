"""AI-powered insights generation from evaluation data."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from .regression import RegressionAnalyzer
from .statistics import StatisticalAnalyzer
from .trends import TrendDetector, TrendType


@dataclass
class Insight:
    """Single analytical insight with metadata."""

    category: str  # "performance", "trend", "comparison", "anomaly"
    severity: str  # "info", "warning", "critical"
    title: str
    description: str
    evidence: dict[str, Any]
    recommendations: list[str]
    confidence: float
    timestamp: datetime
    affected_providers: list[str]
    affected_tasks: list[str]


class InsightsEngine:
    """AI-powered insights generation from evaluation data."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize insights engine with configuration."""
        self.config = config or {}
        self.confidence_threshold = self.config.get("confidence_threshold", 0.5)
        self.statistical_analyzer = StatisticalAnalyzer()
        self.trend_detector = TrendDetector()
        self.regression_analyzer = RegressionAnalyzer(self.config.get("regression", {}))

    def generate_insights(
        self, evaluation_history: list[dict[str, Any]], context: dict[str, Any]
    ) -> list[Insight]:
        """Generate insights from evaluation history."""
        insights: list[Insight] = []

        if not evaluation_history:
            return insights

        # Group data by provider and task
        grouped_data = self._group_by_provider_task(evaluation_history)

        # Analyze performance patterns
        insights.extend(self._analyze_performance_patterns(grouped_data, context))

        # Detect anomalies
        insights.extend(self._detect_anomalies(evaluation_history))

        # Check for regressions
        insights.extend(self._check_regressions(evaluation_history))

        # If no insights generated yet, create at least one basic insight
        if not insights and evaluation_history:
            insights.append(
                Insight(
                    category="general",
                    severity="info",
                    title="Evaluation Data Available",
                    description=(
                        f"Analysis completed on {len(evaluation_history)} "
                        f"evaluation records."
                    ),
                    evidence={"total_records": len(evaluation_history)},
                    recommendations=["Continue monitoring performance trends"],
                    confidence=0.8,
                    timestamp=datetime.now(),
                    affected_providers=list(
                        {e.get("provider", "unknown") for e in evaluation_history}
                    ),
                    affected_tasks=list(
                        {e.get("task", "unknown") for e in evaluation_history}
                    ),
                )
            )

        # Filter by confidence threshold (but ensure we have at least one insight)
        high_confidence_insights = [
            insight
            for insight in insights
            if insight.confidence >= self.confidence_threshold
        ]

        # If filtering removes all insights, return the original list
        if not high_confidence_insights and insights:
            return self._prioritize_insights(insights)

        return self._prioritize_insights(high_confidence_insights)

    def _analyze_performance_patterns(
        self, grouped_data: dict[str, list[dict[str, Any]]], context: dict[str, Any]
    ) -> list[Insight]:
        """Analyze performance patterns and generate insights."""
        insights: list[Insight] = []

        if not grouped_data:
            return insights

        for key, results in grouped_data.items():
            # Extract provider and task from key (format: "provider_task")
            if "_" not in key:
                continue
            provider, task = key.split("_", 1)

            if not isinstance(results, list) or len(results) < 1:
                continue

            scores = [
                r.get("score", 0)
                for r in results
                if isinstance(r.get("score"), int | float)
            ]
            if not scores:
                continue
            mean_score = np.mean(scores)
            std_score = np.std(scores) if len(scores) > 1 else 0.0

            # High variability insight
            if len(scores) > 1 and std_score > 0:
                cv = std_score / mean_score if mean_score > 0 else 0

                if cv > 0.15:  # Lowered threshold to 15%
                    insights.append(
                        Insight(
                            category="performance",
                            severity="warning",
                            title=(
                                f"High Performance Variability - {provider} on {task}"
                            ),
                            description=(
                                f"Performance shows high variability (CV: {cv:.1%}) "
                                f"with mean score {mean_score:.1%} ± {std_score:.1%}"
                            ),
                            evidence={
                                "coefficient_of_variation": cv,
                                "mean_score": mean_score,
                                "std_deviation": std_score,
                                "sample_size": len(scores),
                                "score_range": [min(scores), max(scores)],
                            },
                            recommendations=[
                                "Investigate causes of performance inconsistency",
                                "Consider increasing evaluation repetitions",
                                "Check for environmental factors affecting performance",
                                "Consider using a different evaluation metric",
                            ],
                            confidence=min(
                                0.9, len(scores) / 10
                            ),  # Adjusted confidence calculation
                            timestamp=datetime.now(),
                            affected_providers=[provider],
                            affected_tasks=[task],
                        )
                    )

                # Generate basic performance insight for any data
                insights.append(
                    Insight(
                        category="performance",
                        severity="info",
                        title=f"Performance Analysis - {provider} on {task}",
                        description=(
                            f"{provider} shows average performance of "
                            f"{mean_score:.1%} on {task} based on {len(scores)} "
                            f"evaluations."
                        ),
                        evidence={
                            "mean_score": mean_score,
                            "sample_size": len(scores),
                            "score_range": [min(scores), max(scores)],
                        },
                        recommendations=[
                            "Continue monitoring performance trends",
                            "Consider comparing with other providers",
                            "Analyze task-specific optimization opportunities",
                        ],
                        confidence=0.6,
                        timestamp=datetime.now(),
                        affected_providers=[provider],
                        affected_tasks=[task],
                    )
                )

                # Consistently low performance
                if mean_score < 0.7:  # Lowered threshold to 70%
                    insights.append(
                        Insight(
                            category="performance",
                            severity=("critical" if mean_score < 0.4 else "warning"),
                            title=f"Low Performance - {provider} on {task}",
                            description=(
                                f"{provider} consistently underperforms on {task} "
                                f"with average score of {mean_score:.1%}. "
                                f"This may indicate fundamental capability gaps."
                            ),
                            evidence={
                                "mean_score": mean_score,
                                "performance_percentile": self._calculate_percentile(
                                    mean_score, scores
                                ),
                                "sample_size": len(scores),
                            },
                            recommendations=[
                                "Review task requirements and provider capabilities",
                                "Consider alternative providers for this task type",
                                "Investigate if task complexity exceeds limits",
                            ],
                            confidence=0.85,
                            timestamp=datetime.now(),
                            affected_providers=[provider],
                            affected_tasks=[task],
                        )
                    )

                # Consistently high performance
                elif mean_score > 0.9:  # Above 90%
                    insights.append(
                        Insight(
                            category="performance",
                            severity="info",
                            title=f"Excellent Performance - {provider} on {task}",
                            description=(
                                f"{provider} demonstrates excellent performance on "
                                f"{task} with average score of {mean_score:.1%}. "
                                f"Consider this provider for similar tasks."
                            ),
                            evidence={
                                "mean_score": mean_score,
                                "consistency": 1 - cv,
                                "sample_size": len(scores),
                            },
                            recommendations=[
                                "Use this provider as benchmark for similar tasks",
                                "Investigate what makes this combination successful",
                                "Expand usage of this provider for related tasks",
                            ],
                            confidence=0.8,
                            timestamp=datetime.now(),
                            affected_providers=[provider],
                            affected_tasks=[task],
                        )
                    )

        return insights

    def _compare_providers(self, data: list[dict[str, Any]]) -> list[Insight]:
        """Generate comparative insights between providers."""
        insights = []

        # Group by task to compare providers
        task_performance = self._group_by_provider(data)

        for task, providers in task_performance.items():
            if len(providers) < 2:  # Need at least 2 providers to compare
                continue

            provider_names = list(providers.keys())
            provider_scores = list(providers.values())

            # Find best and worst performers
            mean_scores = [np.mean(scores) for scores in provider_scores]
            best_idx = np.argmax(mean_scores)
            worst_idx = np.argmin(mean_scores)

            best_provider = provider_names[best_idx]
            worst_provider = provider_names[worst_idx]
            best_score = mean_scores[best_idx]
            worst_score = mean_scores[worst_idx]

            # Statistical comparison
            if (
                len(provider_scores[best_idx]) >= 3
                and len(provider_scores[worst_idx]) >= 3
            ):
                stat_result = self.statistical_analyzer.compare_providers(
                    provider_scores[best_idx], provider_scores[worst_idx]
                )

                performance_gap = (
                    (best_score - worst_score) / best_score if best_score > 0 else 0
                )

                if stat_result.significant and performance_gap > 0.1:  # 10% gap
                    severity = "critical" if performance_gap > 0.3 else "warning"

                    insights.append(
                        Insight(
                            category="performance",
                            severity=severity,
                            title=f"Significant Performance Gap - {task}",
                            description=(
                                f"Large performance difference detected on {task}: "
                                f"{best_provider} ({best_score:.1%}) significantly "
                                f"outperforms {worst_provider} ({worst_score:.1%}) "
                                f"with {performance_gap:.1%} gap."
                            ),
                            evidence={
                                "best_provider": best_provider,
                                "worst_provider": worst_provider,
                                "performance_gap": performance_gap,
                                "statistical_test": {
                                    "p_value": stat_result.p_value,
                                    "effect_size": stat_result.effect_size,
                                    "confidence_interval": (
                                        stat_result.confidence_interval
                                    ),
                                },
                            },
                            recommendations=[
                                f"Consider prioritizing {best_provider} for {task}",
                                f"Investigate why {worst_provider} underperforms",
                                "Review task-specific provider configurations",
                            ],
                            confidence=1 - stat_result.p_value,
                            timestamp=datetime.now(),
                            affected_providers=[best_provider, worst_provider],
                            affected_tasks=[task],
                        )
                    )

        return insights

    def _analyze_trends(self, data: list[dict[str, Any]]) -> list[Insight]:
        """Analyze performance trends over time."""
        insights = []

        # Group by provider and task with timestamps
        # time_series_data = self._group_by_time_window(data)  # Unused

        # Simplified trend analysis - group by provider
        provider_data = self._group_by_provider(data)

        for provider, tasks in provider_data.items():
            for task, scores in tasks.items():
                if len(scores) < 7:  # Need at least a week of data
                    continue

                # Create simple time series from scores with datetime objects
                timestamps = [
                    datetime.now() - timedelta(days=i) for i in range(len(scores))
                ]

                # Analyze trend
                trend_analysis = self.trend_detector.analyze_performance_trend(
                    scores, timestamps
                )

                # Generate insights based on trend type
                if (
                    trend_analysis.trend_type == TrendType.DECLINING
                    and trend_analysis.trend_strength > 0.3
                ):
                    insights.append(
                        Insight(
                            category="trend",
                            severity="critical"
                            if trend_analysis.slope < -0.05
                            else "warning",
                            title=f"Declining Performance Trend - {provider} on {task}",
                            description=(
                                f"{provider} shows declining performance on {task} "
                                f"with trend strength "
                                f"{trend_analysis.trend_strength:.2f}. "
                                f"Performance is dropping at "
                                f"{abs(trend_analysis.slope):.3f} points per day."
                            ),
                            evidence={
                                "trend_type": trend_analysis.trend_type.value,
                                "slope": trend_analysis.slope,
                                "r_squared": trend_analysis.r_squared,
                                "trend_strength": trend_analysis.trend_strength,
                                "forecast": trend_analysis.forecast[
                                    :3
                                ],  # Next 3 predictions
                            },
                            recommendations=[
                                "Investigate causes of performance decline",
                                "Monitor trend closely for further degradation",
                                "Consider intervention if trend continues",
                            ],
                            confidence=trend_analysis.trend_strength,
                            timestamp=datetime.now(),
                            affected_providers=[provider],
                            affected_tasks=[task],
                        )
                    )

                elif (
                    trend_analysis.trend_type == TrendType.IMPROVING
                    and trend_analysis.trend_strength > 0.3
                ):
                    insights.append(
                        Insight(
                            category="trend",
                            severity="info",
                            title=f"Improving Performance Trend - {provider} on {task}",
                            description=(
                                f"{provider} shows improving performance on {task} "
                                f"with trend strength "
                                f"{trend_analysis.trend_strength:.2f}. "
                                f"Performance is improving at "
                                f"{trend_analysis.slope:.3f} points per day."
                            ),
                            evidence={
                                "trend_type": trend_analysis.trend_type.value,
                                "slope": trend_analysis.slope,
                                "r_squared": trend_analysis.r_squared,
                                "trend_strength": trend_analysis.trend_strength,
                                "forecast": trend_analysis.forecast[:3],
                            },
                            recommendations=[
                                "Identify factors contributing to improvement",
                                "Apply successful strategies to other tasks",
                                "Monitor to ensure trend sustainability",
                            ],
                            confidence=trend_analysis.trend_strength,
                            timestamp=datetime.now(),
                            affected_providers=[provider],
                            affected_tasks=[task],
                        )
                    )

                elif trend_analysis.trend_type == TrendType.VOLATILE:
                    insights.append(
                        Insight(
                            category="trend",
                            severity="warning",
                            title=f"Volatile Performance - {provider} on {task}",
                            description=(
                                f"{provider} shows volatile performance on {task} "
                                f"with high variability "
                                f"(volatility: {trend_analysis.volatility:.2f}). "
                                f"Performance is unpredictable and inconsistent."
                            ),
                            evidence={
                                "trend_type": trend_analysis.trend_type.value,
                                "volatility": trend_analysis.volatility,
                                "changepoints": trend_analysis.changepoints,
                            },
                            recommendations=[
                                "Investigate sources of performance instability",
                                "Increase evaluation frequency for monitoring",
                                "Look for external factors causing variability",
                            ],
                            confidence=0.8,
                            timestamp=datetime.now(),
                            affected_providers=[provider],
                            affected_tasks=[task],
                        )
                    )

        return insights

    def _generate_regression_insights(
        self, data: list[dict[str, Any]]
    ) -> list[Insight]:
        """Generate insights from regression analysis."""
        insights: list[Insight] = []

        # Split data into baseline (older) and current (recent)
        cutoff_date = datetime.now() - timedelta(days=7)  # Last week as current

        baseline_data = [
            d for d in data if d.get("timestamp", datetime.now()) < cutoff_date
        ]
        current_data = [
            d for d in data if d.get("timestamp", datetime.now()) >= cutoff_date
        ]

        if not baseline_data or not current_data:
            return insights

        # Analyze for regressions
        regression_alert = self.regression_analyzer.analyze_regression(
            baseline_data, current_data
        )

        if regression_alert.detected:
            insights.append(
                Insight(
                    category="regression",
                    severity=regression_alert.severity,
                    title="Performance Regression Detected",
                    description=(
                        f"Performance regression detected with "
                        f"{regression_alert.performance_drop:.1%} drop. "
                        f"Affected tasks: {', '.join(regression_alert.affected_tasks)}"
                    ),
                    evidence=regression_alert.statistical_evidence,
                    recommendations=regression_alert.recommended_actions,
                    confidence=regression_alert.confidence,
                    timestamp=regression_alert.timestamp,
                    affected_providers=list(
                        {
                            provider
                            for task_data in (
                                regression_alert.statistical_evidence.values()
                            )
                            for provider in [task_data.get("provider", "unknown")]
                        }
                    ),
                    affected_tasks=regression_alert.affected_tasks,
                )
            )

        return insights

    def _group_by_provider(
        self, results: list[dict[str, Any]]
    ) -> dict[str, dict[str, list[float]]]:
        """Group evaluation data by provider and task."""
        grouped: dict[str, dict[str, list[float]]] = {}

        for item in results:
            provider = item.get("provider", "unknown")
            task = item.get("task_id", "unknown")
            score = item.get("total_score", item.get("score", 0))

            if provider not in grouped:
                grouped[provider] = {}
            if task not in grouped[provider]:
                grouped[provider][task] = []

            grouped[provider][task].append(float(score))

        return grouped

    def _group_by_time_window(
        self, results: list[dict[str, Any]], window_hours: int = 24
    ) -> dict[datetime, list[dict[str, Any]]]:
        """Group evaluation data by time window."""
        grouped: dict[datetime, list[dict[str, Any]]] = {}

        for item in results:
            timestamp = item.get("timestamp", datetime.now())
            window_start = timestamp - timedelta(
                hours=timestamp.hour % window_hours,
                minutes=timestamp.minute,
                seconds=timestamp.second,
            )

            if window_start not in grouped:
                grouped[window_start] = []

            grouped[window_start].append(item)

        return grouped

    def _calculate_percentile(self, value: float, distribution: list[float]) -> float:
        """Calculate percentile of value in distribution."""
        return (sum(1 for x in distribution if x <= value) / len(distribution)) * 100

    def _assess_anomaly_severity(
        self, anomaly_scores: list[float], normal_scores: list[float]
    ) -> str:
        """Assess severity of anomalies based on deviation from normal."""
        if not normal_scores:
            return "warning"

        normal_mean = np.mean(normal_scores)
        normal_std = np.std(normal_scores, ddof=1)

        if normal_std == 0:
            return "info"

        # Calculate z-scores for anomalies
        z_scores = [abs(score - normal_mean) / normal_std for score in anomaly_scores]
        max_z_score = max(z_scores) if z_scores else 0

        if max_z_score > 3:
            return "critical"
        elif max_z_score > 2:
            return "warning"
        else:
            return "info"

    def _prioritize_insights(self, insights: list[Insight]) -> list[Insight]:
        """Sort insights by priority (severity and confidence)."""
        severity_order = {"critical": 3, "warning": 2, "info": 1}

        return sorted(
            insights,
            key=lambda x: (
                severity_order.get(x.severity, 0),
                x.confidence,
                len(x.affected_providers) + len(x.affected_tasks),
            ),
            reverse=True,
        )

    def _detect_anomalies(
        self, evaluation_history: list[dict[str, Any]]
    ) -> list[Insight]:
        """Detect anomalous performance patterns."""
        insights: list[Insight] = []

        if len(evaluation_history) < 5:
            return insights

        # Group by provider and task
        grouped = self._group_by_provider_task(evaluation_history)

        for key, results in grouped.items():
            if len(results) < 5:
                continue

            provider, task = key.split("_", 1)
            scores = [r.get("score", 0) for r in results]

            # Calculate statistics
            mean_score = np.mean(scores)
            std_score = np.std(scores)

            if std_score == 0:
                continue

            # Detect outliers using z-score
            z_scores = [(score - mean_score) / std_score for score in scores]
            anomalies = [i for i, z in enumerate(z_scores) if abs(z) > 2.5]

            if anomalies:
                insights.append(
                    Insight(
                        category="anomaly",
                        severity="warning",
                        title=(
                            f"Performance anomalies detected for {provider} on {task}"
                        ),
                        description=(
                            f"Found {len(anomalies)} anomalous results "
                            f"out of {len(scores)} evaluations"
                        ),
                        evidence={
                            "anomaly_count": len(anomalies),
                            "total_evaluations": len(scores),
                            "mean_score": mean_score,
                            "std_deviation": std_score,
                            "anomalous_scores": [scores[i] for i in anomalies],
                        },
                        recommendations=[
                            "Investigate conditions during anomalous evaluations",
                            "Check for system resource constraints or network issues",
                            "Consider excluding anomalous results from trend analysis",
                        ],
                        confidence=0.75,
                        timestamp=datetime.now(),
                        affected_providers=[provider],
                        affected_tasks=[task],
                    )
                )

        return insights

    def _group_by_task(
        self, data: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Group evaluation data by task."""
        grouped: dict[str, list[dict[str, Any]]] = {}

        for item in data:
            task = item.get("task", "unknown")

            if task not in grouped:
                grouped[task] = []

            grouped[task].append(item)

        return grouped

    def _group_by_provider_task(
        self, data: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Group evaluation data by provider and task combination."""
        grouped: dict[str, list[dict[str, Any]]] = {}

        for item in data:
            provider = item.get("provider", "unknown")
            task = item.get("task", "unknown")
            key = f"{provider}_{task}"

            if key not in grouped:
                grouped[key] = []

            grouped[key].append(item)

        return grouped

    def _check_regressions(
        self, evaluation_history: list[dict[str, Any]]
    ) -> list[Insight]:
        """Check for performance regressions."""
        insights: list[Insight] = []

        if len(evaluation_history) < 4:
            return insights

        # Group by provider and task
        grouped = self._group_by_provider_task(evaluation_history)

        for key, results in grouped.items():
            if len(results) < 4:
                continue

            provider, task = key.split("_", 1)

            # Sort by timestamp
            sorted_results = sorted(results, key=lambda x: x["timestamp"])
            recent = sorted_results[-5:]  # Last 5 results

            if len(recent) < 4:
                continue

            # Calculate trend
            scores = [r.get("score", 0) for r in recent]

            # Simple regression check - compare recent average to older average
            mid_point = len(scores) // 2
            older_avg = sum(scores[:mid_point]) / mid_point
            recent_avg = sum(scores[mid_point:]) / (len(scores) - mid_point)

            regression_threshold = 0.1  # 10% drop
            if (
                older_avg > 0
                and (older_avg - recent_avg) / older_avg > regression_threshold
            ):
                insights.append(
                    Insight(
                        category="regression",
                        severity="critical",
                        title=(
                            f"Performance regression detected for {provider} on {task}"
                        ),
                        description=(
                            f"Score dropped from {older_avg:.2f} to {recent_avg:.2f}"
                        ),
                        evidence={
                            "provider": provider,
                            "task": task,
                            "older_avg": older_avg,
                            "recent_avg": recent_avg,
                            "drop_percentage": (older_avg - recent_avg)
                            / older_avg
                            * 100,
                        },
                        recommendations=[
                            f"Investigate recent changes affecting {provider}",
                            "Review configuration changes or model updates",
                            "Consider rolling back recent modifications",
                        ],
                        confidence=0.8,
                        timestamp=datetime.now(),
                        affected_providers=[provider],
                        affected_tasks=[task],
                    )
                )

        return insights
