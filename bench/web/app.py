"""Flask web application for analytics dashboard."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any

from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from bench.analytics.insights import InsightsEngine
from bench.analytics.regression import RegressionAnalyzer
from bench.analytics.statistics import StatisticalAnalyzer
from bench.analytics.trends import TrendDetector

# from bench.core.utils import load_evaluation_results


def create_app(config: dict[str, Any] | None = None) -> Flask:
    """Create and configure Flask application."""
    app = Flask(__name__, template_folder="templates", static_folder="static")

    # Configure CORS
    CORS(app)

    # Configure app
    if config:
        app.config.update(config)

    # Initialize analytics components
    app.statistical_analyzer = StatisticalAnalyzer()
    app.trend_detector = TrendDetector()
    app.insights_engine = InsightsEngine()
    app.regression_analyzer = RegressionAnalyzer(
        {"thresholds": {"minor": 0.05, "moderate": 0.10, "severe": 0.20}}
    )

    # Thread pool for async operations
    app.executor = ThreadPoolExecutor(max_workers=4)

    @app.route("/")
    def dashboard() -> str:
        """Main dashboard view."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analytics Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .nav { display: flex; gap: 20px; margin-bottom: 20px; }
        .nav a {
            text-decoration: none;
            color: #007bff;
            padding: 10px 15px;
            background: white;
            border-radius: 5px;
        }
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .chart-placeholder {
            height: 200px;
            background: #e9ecef;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Analytics Dashboard</h1>
            <p>ChatGPT vs Microsoft Copilot Evaluation Harness</p>
        </div>

        <nav class="nav">
            <a href="/" class="nav-link button">Dashboard</a>
            <a href="/api/performance-data" class="nav-link button">Performance Data</a>
            <a href="/api/insights" class="nav-link button">Insights</a>
            <a href="/api/health" class="nav-link button">Health Check</a>
        </nav>

        <script>
            // Dashboard functionality
            console.log('Dashboard loaded');
        </script>

        <div class="dashboard-grid">
            <div class="card">
                <h3>Performance Overview</h3>
                <div class="chart-placeholder">Performance Chart</div>
            </div>

            <div class="card">
                <h3>Provider Comparison</h3>
                <div class="chart-placeholder">Comparison Graph</div>
            </div>

            <div class="card">
                <h3>Trend Analysis</h3>
                <div class="chart-placeholder">Trend Plot</div>
            </div>

            <div class="card">
                <h3>Recent Insights</h3>
                <canvas id="insights-chart" width="200" height="100"></canvas>
                <script src="dashboard.js"></script>
                <p>AI-powered insights and recommendations</p>
            </div>
        </div>
    </div>
</body>
</html>"""

    @app.route("/api/performance-data")
    def performance_data() -> Response:
        """API endpoint for performance data."""
        try:
            # Load recent evaluation results
            results = load_evaluation_results()

            # Calculate metrics
            metrics = calculate_performance_metrics(results)

            return jsonify(
                {
                    "status": "success",
                    "data": metrics,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/trends/<provider>")
    def provider_trends(provider: str) -> Response:
        """API endpoint for provider trend data."""
        try:
            # Validate provider
            valid_providers = ["chatgpt", "copilot", "copilot_manual"]
            if provider not in valid_providers:
                return jsonify(
                    {
                        "status": "error",
                        "message": (
                            f"Invalid provider: {provider}. "
                            f"Valid providers: {valid_providers}"
                        ),
                    }
                ), 404

            baseline_results = load_evaluation_results()
            # Group by task and analyze trends
            trends = {}
            task_groups = group_by_task(baseline_results)

            for task, task_results in task_groups.items():
                if len(task_results) >= 3:  # Need minimum data for trend analysis
                    timestamps = [
                        r.get("timestamp", datetime.now()) for r in task_results
                    ]
                    scores = [r.get("total_score", 0) for r in task_results]

                    trend_analysis = app.trend_detector.analyze_performance_trend(
                        scores, timestamps
                    )

                    trends[task] = {
                        "trend_type": trend_analysis.trend_type.value,
                        "slope": trend_analysis.slope,
                        "r_squared": trend_analysis.r_squared,
                        "volatility": trend_analysis.volatility,
                        "forecast": trend_analysis.forecast,
                        "confidence_bands": {
                            "lower": trend_analysis.confidence_bands[0],
                            "upper": trend_analysis.confidence_bands[1],
                        },
                        "trend_strength": trend_analysis.trend_strength,
                        "data_points": len(scores),
                    }

            return jsonify(
                {
                    "status": "success",
                    "provider": provider,
                    "trends": trends,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/insights")
    def insights() -> Response:
        """API endpoint for generated insights."""
        try:
            min_confidence = request.args.get("min_confidence", 0.0, type=float)
            results = load_evaluation_results()

            # Generate insights
            insights_list = app.insights_engine.generate_insights(results, {})

            # Convert to JSON-serializable format and filter by confidence
            insights_json = []
            for insight in insights_list:
                if insight.confidence >= min_confidence:
                    insights_json.append(
                        {
                            "category": insight.category,
                            "severity": insight.severity,
                            "title": insight.title,
                            "description": insight.description,
                            "evidence": insight.evidence,
                            "recommendations": insight.recommendations,
                            "confidence": insight.confidence,
                            "timestamp": insight.timestamp.isoformat(),
                            "affected_providers": insight.affected_providers,
                            "affected_tasks": insight.affected_tasks,
                        }
                    )

            return jsonify(
                {
                    "status": "success",
                    "insights": insights_json,
                    "count": len(insights_json),
                    "timestamp": datetime.now().isoformat(),
                }
            )

        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/compare")
    def compare_providers() -> Response:
        """API endpoint for provider comparison."""
        try:
            provider_a = request.args.get("provider_a")
            provider_b = request.args.get("provider_b")
            task = request.args.get("task")
            days = request.args.get("days", 30, type=int)

            if not provider_a or not provider_b:
                return jsonify(
                    {
                        "status": "error",
                        "message": "Both provider_a and provider_b are required",
                    }
                ), 400

            # Load data for both providers
            results_a = load_evaluation_results()
            results_b = load_evaluation_results()

            if not results_a or not results_b:
                return jsonify(
                    {
                        "status": "success",
                        "comparison": {
                            "provider_a": provider_a,
                            "provider_b": provider_b,
                            "message": "No data available for comparison",
                        },
                    }
                )

            scores_a = [r.get("score", 0) for r in results_a]
            scores_b = [r.get("score", 0) for r in results_b]

            # Perform statistical comparison
            try:
                stat_result = app.statistical_analyzer.compare_providers(
                    scores_a, scores_b
                )
                # Calculate effect size
                effect_size_result = app.statistical_analyzer.effect_size_analysis(
                    scores_a, scores_b
                )
            except Exception:
                # Handle statistical analysis errors gracefully
                stat_result = {
                    "p_value": None,
                    "statistic": None,
                    "significant": False,
                    "test_type": "failed",
                    "assumptions": {},
                }
                effect_size_result = {
                    "cohens_d": 0.0,
                    "interpretation": "no_effect",
                    "confidence_interval": [0.0, 0.0],
                }

            comparison_result = {
                "provider_a": {
                    "name": provider_a,
                    "mean_score": float(sum(scores_a) / len(scores_a))
                    if scores_a
                    else 0.0,
                    "std_score": float(
                        (
                            sum(
                                (x - sum(scores_a) / len(scores_a)) ** 2
                                for x in scores_a
                            )
                            / max(len(scores_a) - 1, 1)
                        )
                        ** 0.5
                    )
                    if len(scores_a) > 1
                    else 0.0,
                    "sample_size": len(scores_a),
                    "score_distribution": scores_a,
                },
                "provider_b": {
                    "name": provider_b,
                    "mean_score": float(sum(scores_b) / len(scores_b))
                    if scores_b
                    else 0.0,
                    "std_score": float(
                        (
                            sum(
                                (x - sum(scores_b) / len(scores_b)) ** 2
                                for x in scores_b
                            )
                            / max(len(scores_b) - 1, 1)
                        )
                        ** 0.5
                    )
                    if len(scores_b) > 1
                    else 0.0,
                    "sample_size": len(scores_b),
                    "score_distribution": scores_b,
                },
                "statistical_test": {
                    "test_name": getattr(stat_result, "test_name", "unknown"),
                    "p_value": getattr(stat_result, "p_value", None),
                    "statistic": getattr(stat_result, "statistic", None),
                    "significant": getattr(stat_result, "significant", False),
                    "confidence_interval": getattr(
                        stat_result, "confidence_interval", []
                    ),
                    "interpretation": getattr(
                        stat_result, "interpretation", "no_interpretation"
                    ),
                },
                "effect_size": effect_size_result,
                "task": task,
                "comparison_period_days": days,
            }

            return jsonify(
                {
                    "status": "success",
                    "comparison": comparison_result,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/regression-check")
    def regression_check() -> Response:
        """API endpoint for regression detection."""
        try:
            baseline_days = request.args.get("baseline_days", 14, type=int)
            current_days = request.args.get("current_days", 7, type=int)

            # Load baseline and current data
            baseline_results = load_evaluation_results()
            current_results = load_evaluation_results()

            if not baseline_results or not current_results:
                return jsonify(
                    {
                        "status": "error",
                        "message": "Insufficient data for regression analysis",
                    }
                ), 400

            # Analyze regression
            regression_alert = app.regression_analyzer.analyze_regression(
                baseline_results, current_results
            )

            return jsonify(
                {
                    "status": "success",
                    "regression_alert": {
                        "detected": regression_alert.detected,
                        "severity": regression_alert.severity,
                        "affected_tasks": regression_alert.affected_tasks,
                        "performance_drop": regression_alert.performance_drop,
                        "confidence": regression_alert.confidence,
                        "recommendations": regression_alert.recommended_actions,
                        "timestamp": regression_alert.timestamp.isoformat(),
                    },
                    "baseline_period": f"{baseline_days} days",
                    "current_period": f"{current_days} days",
                }
            )

        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/health")
    def health_check() -> Response:
        """Health check endpoint."""
        return jsonify(
            {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": datetime.now().isoformat(),
            }
        )

    @app.errorhandler(404)
    def not_found(error: Exception) -> Response:
        """Handle 404 errors."""
        return jsonify({"status": "error", "message": "Not found"}), 404

    @app.errorhandler(500)
    def internal_error(error: Exception) -> Response:
        """Handle 500 errors."""
        return jsonify({"status": "error", "message": "Internal server error"}), 500

    return app


def calculate_performance_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate performance metrics from evaluation results."""
    if not results:
        return {
            "overall": {"mean_score": 0, "count": 0},
            "by_provider": {},
            "by_task": {},
        }

    # Overall metrics
    scores = [r.get("score", 0) for r in results]
    overall_mean = sum(scores) / len(scores) if scores else 0

    # By provider
    by_provider = {}
    providers = {r.get("provider") for r in results if r.get("provider")}
    for provider in providers:
        provider_results = [r for r in results if r.get("provider") == provider]
        provider_scores = [r.get("score", 0) for r in provider_results]
        by_provider[provider] = {
            "mean_score": sum(provider_scores) / len(provider_scores)
            if provider_scores
            else 0,
            "count": len(provider_results),
        }

    # By task
    by_task = {}
    tasks = {r.get("task") for r in results if r.get("task")}
    for task in tasks:
        task_results = [r for r in results if r.get("task") == task]
        task_scores = [r.get("score", 0) for r in task_results]
        by_task[task] = {
            "mean_score": sum(task_scores) / len(task_scores) if task_scores else 0,
            "count": len(task_results),
        }

    return {
        "overall": {"mean_score": overall_mean, "count": len(results)},
        "by_provider": by_provider,
        "by_task": by_task,
    }


def group_by_task(data: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group results by task."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for result in data:
        task = result.get("task_id", "unknown")
        if task not in groups:
            groups[task] = []
        groups[task].append(result)
    return groups


def group_by_provider(data: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group results by provider."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for result in data:
        provider = result.get("provider", "unknown")
        if provider not in groups:
            groups[provider] = []
        groups[provider].append(result)
    return groups


def group_by_time_window(
    data: list[dict[str, Any]], window_hours: int = 24
) -> dict[str, list[dict[str, Any]]]:
    """Group results by time window."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for result in data:
        timestamp = result.get("timestamp", datetime.now())
        window_start = timestamp - timedelta(
            hours=timestamp.hour % window_hours,
            minutes=timestamp.minute,
            seconds=timestamp.second,
        )
        window_key = window_start.isoformat()
        if window_key not in groups:
            groups[window_key] = []
        groups[window_key].append(result)
    return groups


def calculate_performance_metrics_placeholder(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate overall performance metrics from results."""
    if not results:
        return {}

    # Group by provider
    provider_metrics = {}
    provider_groups = group_by_provider(results)

    for provider, provider_results in provider_groups.items():
        scores = [r.get("total_score", 0) for r in provider_results]

        provider_metrics[provider] = {
            "mean_score": sum(scores) / len(scores),
            "median_score": sorted(scores)[len(scores) // 2],
            "std_score": (
                sum((x - sum(scores) / len(scores)) ** 2 for x in scores)
                / (len(scores) - 1)
            )
            ** 0.5,
            "min_score": min(scores),
            "max_score": max(scores),
            "evaluation_count": len(scores),
            "last_updated": max(
                r.get("timestamp", datetime.now()) for r in provider_results
            ).isoformat(),
        }

    # Overall metrics
    all_scores = [r.get("total_score", 0) for r in results]
    overall_metrics = {
        "total_evaluations": len(results),
        "average_score": sum(all_scores) / len(all_scores),
        "score_range": [min(all_scores), max(all_scores)],
        "providers": list(provider_metrics.keys()),
        "provider_count": len(provider_metrics),
    }

    return {"overall": overall_metrics, "by_provider": provider_metrics}


def group_by_provider_v2(data: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group results by provider."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in data:
        provider = result.get("provider", "unknown")
        if provider not in grouped:
            grouped[provider] = []
        grouped[provider].append(result)
    return grouped


def group_by_task_v2(data: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group results by task."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in data:
        task = result.get("task_id", "unknown")
        if task not in grouped:
            grouped[task] = []
        grouped[task].append(result)
    return grouped


def load_evaluation_results() -> list[dict[str, Any]]:
    """Load evaluation results with filtering."""
    # This is a placeholder implementation
    # In a real system, this would query the database

    # Generate sample data for demonstration
    import secrets

    providers = ["chatgpt", "copilot_manual"]
    tasks = [
        "offline.task1.metrics_csv",
        "offline.task2.ssn_regex",
        "offline.task3.exec_summary",
    ]

    results = []
    for i in range(100):  # Generate 100 sample results
        result = {
            "provider": secrets.choice(providers),
            "task_id": secrets.choice(tasks),
            "total_score": secrets.randbelow(100),
            "timestamp": datetime.now() - timedelta(days=secrets.randbelow(30)),
            "run_id": f"run_{i:03d}",
            "metadata": {
                "duration": secrets.randbelow(10),
                "tokens_used": secrets.randbelow(1000),
            },
        }
        results.append(result)

    return results
