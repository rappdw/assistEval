"""Tests for the reporting system.

This module tests markdown report generation, JSON report generation,
and report aggregation functionality.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from bench.core.reporting import (
    JSONReportGenerator,
    MarkdownReportGenerator,
    ProviderComparison,
    ReportAggregator,
    ReportSummary,
    TaskBreakdown,
)
from bench.core.scoring import ProviderScore, TaskScore


class TestMarkdownReportGenerator:
    """Test markdown report generation."""

    @pytest.fixture
    def sample_provider_scores(self):
        """Sample provider scores for testing."""
        task_scores_chatgpt = [
            TaskScore(
                task_id="offline.task1.metrics_csv",
                evaluator_name="metrics_csv",
                raw_score=38.0,
                max_score=40.0,
                weighted_score=38.0,
                weight=1.0,
                sub_scores={
                    "precision": 9.5,
                    "recall": 9.0,
                    "f1": 9.2,
                    "accuracy": 4.4,
                },
                errors=[],
                warnings=[],
            ),
            TaskScore(
                task_id="offline.task2.ssn_regex",
                evaluator_name="regex_match",
                raw_score=25.0,
                max_score=30.0,
                weighted_score=25.0,
                weight=1.0,
                sub_scores={"validity": 18.0, "line_matches": 7.0},
                errors=[],
                warnings=["Minor regex optimization possible"],
            ),
        ]

        task_scores_copilot = [
            TaskScore(
                task_id="offline.task1.metrics_csv",
                evaluator_name="metrics_csv",
                raw_score=35.0,
                max_score=40.0,
                weighted_score=35.0,
                weight=1.0,
                sub_scores={
                    "precision": 8.5,
                    "recall": 8.0,
                    "f1": 8.5,
                    "accuracy": 4.0,
                },
                errors=["Precision calculation error"],
                warnings=[],
            ),
            TaskScore(
                task_id="offline.task2.ssn_regex",
                evaluator_name="regex_match",
                raw_score=28.0,
                max_score=30.0,
                weighted_score=28.0,
                weight=1.0,
                sub_scores={"validity": 18.0, "line_matches": 10.0},
                errors=[],
                warnings=[],
            ),
        ]

        return {
            "chatgpt": ProviderScore(
                provider_name="chatgpt",
                timestamp=datetime.now(),
                task_scores=task_scores_chatgpt,
                total_score=63.0,
                max_score=105.0,
                stability_bonus=5.0,
                final_score=68.0,
            ),
            "copilot": ProviderScore(
                provider_name="copilot",
                timestamp=datetime.now(),
                task_scores=task_scores_copilot,
                total_score=63.0,
                max_score=105.0,
                stability_bonus=3.0,
                final_score=66.0,
            ),
        }

    @pytest.fixture
    def sample_metadata(self):
        """Sample run metadata."""
        return {
            "timestamp": "2025-08-27T21:19:28-06:00",
            "run_id": "run_20250827_211928",
            "provider_count": 2,
            "task_count": 2,
            "execution_time": 45.2,
        }

    def test_generate_report(self, sample_provider_scores, sample_metadata):
        """Test complete report generation."""
        generator = MarkdownReportGenerator()
        report = generator.generate_report(sample_provider_scores, sample_metadata)

        assert "# AssistEval Benchmark Report" in report
        assert "## Executive Summary" in report
        assert "## Leaderboard" in report
        assert "## Task-by-Task Breakdown" in report
        assert "## Stability Analysis" in report
        assert "## Failure Analysis" in report

    def test_generate_executive_summary(self, sample_provider_scores):
        """Test executive summary generation."""
        generator = MarkdownReportGenerator()
        summary = generator._generate_executive_summary(sample_provider_scores)

        assert "## Executive Summary" in summary
        assert "Overall Winner" in summary
        assert "chatgpt" in summary  # Should be winner with 68.0 vs 66.0
        assert "Score Spread" in summary

    def test_generate_leaderboard_table(self, sample_provider_scores):
        """Test leaderboard formatting."""
        generator = MarkdownReportGenerator()
        leaderboard = generator._generate_leaderboard_table(sample_provider_scores)

        assert "## Leaderboard" in leaderboard
        assert (
            "| Rank | Provider | Total Score | Percentage | Stability Bonus |"
            in leaderboard
        )
        assert "| 1 | chatgpt |" in leaderboard
        assert "| 2 | copilot |" in leaderboard

    def test_generate_task_breakdown(self, sample_provider_scores):
        """Test task analysis section."""
        generator = MarkdownReportGenerator()
        breakdown = generator._generate_task_breakdown(sample_provider_scores)

        assert "## Task-by-Task Breakdown" in breakdown
        assert "### offline.task1.metrics_csv" in breakdown
        assert "### offline.task2.ssn_regex" in breakdown
        assert "**Winner**:" in breakdown

    def test_generate_stability_analysis(self, sample_provider_scores):
        """Test stability analysis section."""
        generator = MarkdownReportGenerator()
        stability = generator._generate_stability_analysis(sample_provider_scores)

        assert "## Stability Analysis" in stability
        assert "Stability Bonus" in stability
        assert "High" in stability  # ChatGPT should have high consistency

    def test_generate_failure_analysis(self, sample_provider_scores):
        """Test failure analysis section."""
        generator = MarkdownReportGenerator()
        failure = generator._generate_failure_analysis(sample_provider_scores)

        assert "## Failure Analysis" in failure
        assert "### Errors" in failure
        assert "### Warnings" in failure
        assert "Precision calculation error" in failure
        assert "Minor regex optimization possible" in failure

    def test_empty_scores_handling(self):
        """Test handling of empty provider scores."""
        generator = MarkdownReportGenerator()
        report = generator.generate_report({}, {})

        assert "No evaluation results available" in report
        assert "No providers to rank" in report
        assert "No task results available" in report


class TestJSONReportGenerator:
    """Test JSON report generation."""

    @pytest.fixture
    def sample_provider_scores(self):
        """Sample provider scores for testing."""
        task_score = TaskScore(
            task_id="offline.task1.metrics_csv",
            evaluator_name="metrics_csv",
            raw_score=38.0,
            max_score=40.0,
            weighted_score=38.0,
            weight=1.0,
            sub_scores={"precision": 9.5},
            errors=[],
            warnings=[],
        )

        return {
            "chatgpt": ProviderScore(
                provider_name="chatgpt",
                timestamp=datetime.now(),
                task_scores=[task_score],
                total_score=38.0,
                max_score=105.0,
                stability_bonus=5.0,
                final_score=43.0,
            )
        }

    @pytest.fixture
    def sample_metadata(self):
        """Sample run metadata."""
        return {
            "timestamp": "2025-08-27T21:19:28-06:00",
            "run_id": "run_20250827_211928",
            "execution_time": 45.2,
        }

    def test_generate_report_structure(self, sample_provider_scores, sample_metadata):
        """Test JSON report structure compliance."""
        generator = JSONReportGenerator()
        report = generator.generate_report(sample_provider_scores, sample_metadata)

        # Check top-level structure
        assert "metadata" in report
        assert "summary" in report
        assert "leaderboard" in report
        assert "tasks" in report
        assert "providers" in report

    def test_metadata_section(self, sample_metadata):
        """Test metadata section generation."""
        generator = JSONReportGenerator()
        metadata = generator._create_metadata_section(sample_metadata)

        assert metadata["timestamp"] == "2025-08-27T21:19:28-06:00"
        assert metadata["version"] == "1.0.0"
        assert metadata["execution_time"] == 45.2
        assert metadata["run_id"] == "run_20250827_211928"

    def test_summary_section(self, sample_provider_scores):
        """Test summary statistics generation."""
        generator = JSONReportGenerator()
        summary = generator._create_summary_section(sample_provider_scores)

        assert summary["total_providers"] == 1
        assert summary["total_tasks"] == 1
        assert summary["overall_winner"] == "chatgpt"
        assert summary["score_spread"] == 0.0  # Only one provider

    def test_leaderboard_section(self, sample_provider_scores):
        """Test leaderboard data generation."""
        generator = JSONReportGenerator()
        leaderboard = generator._create_leaderboard_section(sample_provider_scores)

        assert len(leaderboard) == 1
        assert leaderboard[0]["provider"] == "chatgpt"
        assert leaderboard[0]["rank"] == 1
        assert leaderboard[0]["total_score"] == 43.0

    def test_task_section(self, sample_provider_scores):
        """Test task comparison matrix."""
        generator = JSONReportGenerator()
        tasks = generator._create_task_section(sample_provider_scores)

        assert "offline.task1.metrics_csv" in tasks
        task_data = tasks["offline.task1.metrics_csv"]
        assert task_data["winner"] == "chatgpt"
        assert task_data["scores"]["chatgpt"] == 38.0

    def test_provider_section(self, sample_provider_scores):
        """Test provider detailed results."""
        generator = JSONReportGenerator()
        providers = generator._create_provider_section(sample_provider_scores)

        assert "chatgpt" in providers
        provider_data = providers["chatgpt"]
        assert provider_data["final_score"] == 43.0
        assert provider_data["task_count"] == 1
        assert len(provider_data["task_scores"]) == 1


class TestReportAggregator:
    """Test multi-run aggregation."""

    @pytest.fixture
    def temp_results_dir(self):
        """Create temporary results directory with sample data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir)

            # Create sample run directories
            run1_dir = results_dir / "run_20250827_120000"
            run2_dir = results_dir / "run_20250827_130000"

            for run_dir in [run1_dir, run2_dir]:
                scores_dir = run_dir / "scores"
                scores_dir.mkdir(parents=True)

                # Create sample score files
                score_data = {
                    "provider_name": "chatgpt",
                    "final_score": 85.0 if "120000" in run_dir.name else 87.0,
                    "total_score": 80.0 if "120000" in run_dir.name else 82.0,
                    "stability_bonus": 5.0,
                }

                score_file = scores_dir / "chatgpt_score.json"
                with open(score_file, "w") as f:
                    json.dump(score_data, f)

            yield results_dir

    def test_aggregate_multiple_runs(self, temp_results_dir):
        """Test aggregation across runs."""
        aggregator = ReportAggregator(temp_results_dir)
        result = aggregator.aggregate_runs()

        assert "runs" in result
        assert "trends" in result
        assert "regressions" in result
        assert len(result["runs"]) == 2

    def test_trend_calculation(self, temp_results_dir):
        """Test trend analysis."""
        aggregator = ReportAggregator(temp_results_dir)
        result = aggregator.aggregate_runs()

        trends = result["trends"]
        assert "chatgpt" in trends
        assert trends["chatgpt"]["direction"] == "improving"
        assert trends["chatgpt"]["change"] == 2.0  # 87.0 - 85.0

    def test_regression_detection(self, temp_results_dir):
        """Test regression identification."""
        # Modify second run to have lower score (regression)
        run2_scores = (
            temp_results_dir / "run_20250827_130000" / "scores" / "chatgpt_score.json"
        )
        with open(run2_scores, "w") as f:
            json.dump(
                {
                    "provider_name": "chatgpt",
                    "final_score": 75.0,  # Significant drop
                    "total_score": 70.0,
                    "stability_bonus": 5.0,
                },
                f,
            )

        aggregator = ReportAggregator(temp_results_dir)
        result = aggregator.aggregate_runs()

        regressions = result["regressions"]
        assert len(regressions) == 1
        assert "chatgpt" in regressions
        assert regressions["chatgpt"]["regression_amount"] == 10.0

    def test_load_run_data(self, temp_results_dir):
        """Test loading data from run directory."""
        aggregator = ReportAggregator(temp_results_dir)
        run_dir = temp_results_dir / "run_20250827_120000"

        run_data = aggregator._load_run_data(run_dir)

        assert run_data is not None
        assert run_data["run_id"] == "run_20250827_120000"
        assert "chatgpt" in run_data["provider_scores"]

    def test_empty_results_directory(self):
        """Test handling of empty results directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            aggregator = ReportAggregator(Path(temp_dir))
            result = aggregator.aggregate_runs()

            assert "error" in result
            assert result["error"] == "No valid run data found"


class TestReportDataStructures:
    """Test report data structure classes."""

    def test_report_summary_creation(self):
        """Test ReportSummary data structure."""
        summary = ReportSummary(
            timestamp=datetime.now(),
            total_providers=2,
            total_tasks=3,
            execution_time=45.2,
            stability_runs=3,
            overall_winner="chatgpt",
            score_spread=12.5,
        )

        assert summary.total_providers == 2
        assert summary.overall_winner == "chatgpt"
        assert summary.score_spread == 12.5

    def test_task_breakdown_creation(self):
        """Test TaskBreakdown data structure."""
        breakdown = TaskBreakdown(
            task_id="offline.task1.metrics_csv",
            task_name="Metrics from CSV",
            max_score=40.0,
            provider_scores={"chatgpt": 38.0, "copilot": 35.0},
            winner="chatgpt",
            score_details={},
            failure_reasons={},
        )

        assert breakdown.task_id == "offline.task1.metrics_csv"
        assert breakdown.winner == "chatgpt"
        assert breakdown.provider_scores["chatgpt"] == 38.0

    def test_provider_comparison_creation(self):
        """Test ProviderComparison data structure."""
        comparison = ProviderComparison(
            provider_a="chatgpt",
            provider_b="copilot",
            score_difference=3.0,
            task_wins={"task1": "chatgpt", "task2": "copilot"},
            strengths={"chatgpt": ["accuracy"], "copilot": ["consistency"]},
            weaknesses={"chatgpt": ["speed"], "copilot": ["precision"]},
        )

        assert comparison.provider_a == "chatgpt"
        assert comparison.score_difference == 3.0
        assert comparison.task_wins["task1"] == "chatgpt"


class TestReportIntegration:
    """Test end-to-end reporting workflow."""

    @pytest.fixture
    def sample_run_directory(self):
        """Create a complete sample run directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run_20250827_211928"

            # Create directory structure
            scores_dir = run_dir / "scores"
            scores_dir.mkdir(parents=True)

            # Create sample provider scores
            chatgpt_score = {
                "provider_name": "chatgpt",
                "timestamp": datetime.now().isoformat(),
                "task_scores": [
                    {
                        "task_id": "offline.task1.metrics_csv",
                        "evaluator_name": "metrics_csv",
                        "raw_score": 38.0,
                        "max_score": 40.0,
                        "weighted_score": 38.0,
                        "weight": 1.0,
                        "sub_scores": {"precision": 9.5},
                        "details": {},
                        "errors": [],
                        "warnings": [],
                    }
                ],
                "total_score": 38.0,
                "max_score": 105.0,
                "stability_bonus": 5.0,
                "final_score": 43.0,
                "metadata": {},
            }

            with open(scores_dir / "chatgpt_score.json", "w") as f:
                json.dump(chatgpt_score, f)

            yield run_dir

    def test_full_report_generation_workflow(self, sample_run_directory):
        """Test complete report generation from scores."""
        from bench.core.reporting import JSONReportGenerator, MarkdownReportGenerator
        from bench.core.scoring import ProviderScore, TaskScore

        # Create proper ProviderScore objects for testing
        task_score = TaskScore(
            task_id="offline.task1.metrics_csv",
            evaluator_name="MetricsCSVEvaluator",
            raw_score=38.0,
            max_score=40.0,
            weighted_score=38.0,
            weight=40.0,
            sub_scores={"precision": 0.95, "recall": 0.90},
            details={},
            errors=[],
            warnings=[],
        )

        from datetime import datetime

        provider_score = ProviderScore(
            provider_name="chatgpt",
            timestamp=datetime.now(),
            task_scores=[task_score],
            total_score=38.0,
            max_score=105.0,
            stability_bonus=0.0,
            final_score=38.0,
            metadata={},
        )

        provider_scores = {"chatgpt": provider_score}

        # Test markdown generation
        markdown_gen = MarkdownReportGenerator()
        markdown_content = markdown_gen.generate_report(provider_scores, {})
        assert len(markdown_content) > 0
        assert "chatgpt" in markdown_content

        # Test JSON generation
        json_gen = JSONReportGenerator()
        json_content = json_gen.generate_report(provider_scores, {})
        assert len(json_content) > 0
        assert "chatgpt" in json_content["providers"]

    def test_report_content_accuracy(self, sample_run_directory):
        """Test report content matches input data."""
        from bench.core.reporting import MarkdownReportGenerator
        from bench.core.scoring import ProviderScore, TaskScore

        # Create test data with known values
        task_score = TaskScore(
            task_id="offline.task1.metrics_csv",
            evaluator_name="MetricsCSVEvaluator",
            raw_score=35.0,
            max_score=40.0,
            weighted_score=35.0,
            weight=40.0,
            sub_scores={"precision": 0.88, "recall": 0.92},
            details={},
            errors=[],
            warnings=[],
        )

        from datetime import datetime

        provider_score = ProviderScore(
            provider_name="test_provider",
            timestamp=datetime.now(),
            task_scores=[task_score],
            total_score=35.0,
            max_score=105.0,
            stability_bonus=0.0,
            final_score=35.0,
            metadata={},
        )

        provider_scores = {"test_provider": provider_score}

        # Generate markdown report
        markdown_gen = MarkdownReportGenerator()
        content = markdown_gen.generate_report(provider_scores, {})

        # Verify content contains expected data
        assert "test_provider" in content
        assert "35.0" in content  # Final score
        assert "offline.task1.metrics_csv" in content
