"""Tests for stability bonus calculation."""

from bench.core.evaluators.base import EvaluationResult
from bench.core.scoring import StabilityAnalyzer, WeightConfig


class TestStabilityBonus:
    """Test suite for stability bonus calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        weights = {
            "stability_bonus": {"max_points": 5, "threshold": 0.95, "enabled": True}
        }
        self.weight_config = WeightConfig(weights)
        self.analyzer = StabilityAnalyzer(self.weight_config)

    def test_perfect_task1_consistency(self):
        """Test perfect Task 1 numeric consistency."""
        # Create identical results for Task 1
        results = []
        for _ in range(3):
            result = EvaluationResult(
                task_id="offline.task1.metrics_csv",
                total_score=10.0,
                max_score=10.0,
                sub_scores={
                    "precision": 0.85,
                    "recall": 0.92,
                    "f1": 0.88,
                    "accuracy": 0.90,
                },
            )
            results.append(result)

        multi_run_results = {"offline.task1.metrics_csv": results}
        bonus = self.analyzer.calculate_stability_bonus(multi_run_results)

        assert bonus == 5.0  # Should get full bonus
        details = self.analyzer.get_analysis_details()
        assert details["task1_consistency"] == 1.0
        assert details["overall_consistency"] >= 0.95

    def test_inconsistent_task1_results(self):
        """Test inconsistent Task 1 numeric results."""
        # Create varying results for Task 1
        results = [
            EvaluationResult(
                task_id="offline.task1.metrics_csv",
                total_score=10.0,
                max_score=10.0,
                sub_scores={
                    "precision": 0.85,
                    "recall": 0.92,
                    "f1": 0.88,
                    "accuracy": 0.90,
                },
            ),
            EvaluationResult(
                task_id="offline.task1.metrics_csv",
                total_score=9.5,
                max_score=10.0,
                sub_scores={
                    "precision": 0.83,  # Different value
                    "recall": 0.94,  # Different value
                    "f1": 0.88,
                    "accuracy": 0.90,
                },
            ),
            EvaluationResult(
                task_id="offline.task1.metrics_csv",
                total_score=9.8,
                max_score=10.0,
                sub_scores={
                    "precision": 0.85,
                    "recall": 0.92,
                    "f1": 0.89,  # Different value
                    "accuracy": 0.91,  # Different value
                },
            ),
        ]

        multi_run_results = {"offline.task1.metrics_csv": results}
        bonus = self.analyzer.calculate_stability_bonus(multi_run_results)

        assert bonus < 5.0  # Should get reduced bonus
        details = self.analyzer.get_analysis_details()
        assert details["task1_consistency"] < 1.0

    def test_structural_consistency(self):
        """Test structural consistency across tasks."""
        # Create results with consistent error patterns
        task2_results = [
            EvaluationResult(
                task_id="offline.task2.ssn_regex",
                total_score=8.0,
                max_score=10.0,
                sub_scores={},
                errors=[],
                warnings=["Minor regex issue"],
            ),
            EvaluationResult(
                task_id="offline.task2.ssn_regex",
                total_score=8.0,
                max_score=10.0,
                sub_scores={},
                errors=[],
                warnings=["Minor regex issue"],
            ),
        ]

        task3_results = [
            EvaluationResult(
                task_id="offline.task3.exec_summary",
                total_score=9.0,
                max_score=10.0,
                sub_scores={},
                errors=[],
                warnings=[],
            ),
            EvaluationResult(
                task_id="offline.task3.exec_summary",
                total_score=9.0,
                max_score=10.0,
                sub_scores={},
                errors=[],
                warnings=[],
            ),
        ]

        multi_run_results = {
            "offline.task2.ssn_regex": task2_results,
            "offline.task3.exec_summary": task3_results,
        }

        consistency = self.analyzer._analyze_structural_consistency(multi_run_results)
        assert consistency == 1.0  # Perfect structural consistency

    def test_inconsistent_structural_patterns(self):
        """Test inconsistent structural patterns."""
        # Create results with different error patterns
        task2_results = [
            EvaluationResult(
                task_id="offline.task2.ssn_regex",
                total_score=8.0,
                max_score=10.0,
                sub_scores={},
                errors=["Regex error"],
                warnings=[],
            ),
            EvaluationResult(
                task_id="offline.task2.ssn_regex",
                total_score=8.0,
                max_score=10.0,
                sub_scores={},
                errors=[],  # Different error pattern
                warnings=["Warning message"],  # Different warning pattern
            ),
        ]

        multi_run_results = {"offline.task2.ssn_regex": task2_results}
        consistency = self.analyzer._analyze_structural_consistency(multi_run_results)
        assert consistency < 1.0  # Should detect inconsistency

    def test_partial_consistency_scoring(self):
        """Test prorated scoring for partial consistency."""
        # Create results with 80% consistency
        task1_results = [
            EvaluationResult(
                task_id="offline.task1.metrics_csv",
                total_score=10.0,
                max_score=10.0,
                sub_scores={
                    "precision": 0.85,
                    "recall": 0.92,
                    "f1": 0.88,
                    "accuracy": 0.90,
                },
            ),
            EvaluationResult(
                task_id="offline.task1.metrics_csv",
                total_score=10.0,
                max_score=10.0,
                sub_scores={
                    "precision": 0.85,  # Same
                    "recall": 0.92,  # Same
                    "f1": 0.87,  # Different
                    "accuracy": 0.89,  # Different
                },
            ),
        ]

        multi_run_results = {"offline.task1.metrics_csv": task1_results}
        bonus = self.analyzer.calculate_stability_bonus(multi_run_results)

        # Should get prorated bonus based on consistency level
        assert 0 < bonus < 5.0
        details = self.analyzer.get_analysis_details()
        assert details["task1_consistency"] == 0.5  # 2 out of 4 metrics consistent

    def test_disabled_stability_bonus(self):
        """Test disabled stability bonus."""
        weights = {
            "stability_bonus": {"max_points": 5, "threshold": 0.95, "enabled": False}
        }
        weight_config = WeightConfig(weights)
        analyzer = StabilityAnalyzer(weight_config)

        # Even with perfect consistency, should return 0
        results = [
            EvaluationResult(
                task_id="offline.task1.metrics_csv",
                total_score=10.0,
                max_score=10.0,
                sub_scores={"precision": 0.85, "recall": 0.92},
            ),
            EvaluationResult(
                task_id="offline.task1.metrics_csv",
                total_score=10.0,
                max_score=10.0,
                sub_scores={"precision": 0.85, "recall": 0.92},
            ),
        ]

        multi_run_results = {"offline.task1.metrics_csv": results}
        bonus = analyzer.calculate_stability_bonus(multi_run_results)
        assert bonus == 0.0

    def test_insufficient_runs(self):
        """Test handling of insufficient runs for analysis."""
        # Single run - should still work
        results = [
            EvaluationResult(
                task_id="offline.task1.metrics_csv",
                total_score=10.0,
                max_score=10.0,
                sub_scores={"precision": 0.85},
            )
        ]

        multi_run_results = {"offline.task1.metrics_csv": results}
        bonus = self.analyzer.calculate_stability_bonus(multi_run_results)

        # Should handle gracefully
        assert bonus >= 0.0
        details = self.analyzer.get_analysis_details()
        assert details["task1_consistency"] == 1.0  # Single run gets perfect score

    def test_empty_results(self):
        """Test handling of empty results."""
        bonus = self.analyzer.calculate_stability_bonus({})
        assert bonus == 0.0

    def test_mixed_task_consistency(self):
        """Test mixed consistency across different tasks."""
        # Perfect Task 1 consistency
        task1_results = [
            EvaluationResult(
                task_id="offline.task1.metrics_csv",
                total_score=10.0,
                max_score=10.0,
                sub_scores={"precision": 0.85, "recall": 0.92},
            ),
            EvaluationResult(
                task_id="offline.task1.metrics_csv",
                total_score=10.0,
                max_score=10.0,
                sub_scores={"precision": 0.85, "recall": 0.92},
            ),
        ]

        # Inconsistent structural patterns
        task2_results = [
            EvaluationResult(
                task_id="offline.task2.ssn_regex",
                total_score=8.0,
                max_score=10.0,
                sub_scores={},
                errors=["Error"],
                warnings=[],
            ),
            EvaluationResult(
                task_id="offline.task2.ssn_regex",
                total_score=8.0,
                max_score=10.0,
                sub_scores={},
                errors=[],
                warnings=["Warning"],
            ),
        ]

        multi_run_results = {
            "offline.task1.metrics_csv": task1_results,
            "offline.task2.ssn_regex": task2_results,
        }

        bonus = self.analyzer.calculate_stability_bonus(multi_run_results)
        details = self.analyzer.get_analysis_details()

        # Should average perfect Task 1 consistency with poor structural consistency
        assert details["task1_consistency"] == 1.0
        assert details["structural_consistency"] < 1.0
        assert 0 < details["overall_consistency"] < 1.0
        assert 0 < bonus < 5.0

    def test_custom_threshold(self):
        """Test custom consistency threshold."""
        weights = {
            "stability_bonus": {
                "max_points": 5,
                "threshold": 0.8,  # Lower threshold
                "enabled": True,
            }
        }
        weight_config = WeightConfig(weights)
        analyzer = StabilityAnalyzer(weight_config)

        # Create 80% consistent results
        task1_results = [
            EvaluationResult(
                task_id="offline.task1.metrics_csv",
                total_score=10.0,
                max_score=10.0,
                sub_scores={
                    "precision": 0.85,
                    "recall": 0.92,
                    "f1": 0.88,
                    "accuracy": 0.90,
                },
            ),
            EvaluationResult(
                task_id="offline.task1.metrics_csv",
                total_score=10.0,
                max_score=10.0,
                sub_scores={
                    "precision": 0.85,
                    "recall": 0.92,
                    "f1": 0.87,
                    "accuracy": 0.89,
                },
            ),
        ]

        multi_run_results = {"offline.task1.metrics_csv": task1_results}
        bonus = analyzer.calculate_stability_bonus(multi_run_results)

        # With 80% threshold and 50% consistency, should get partial bonus
        details = analyzer.get_analysis_details()
        assert details["threshold"] == 0.8
        assert bonus > 0  # Should get some bonus with lower threshold

    def test_analysis_details(self):
        """Test detailed analysis information."""
        task1_results = [
            EvaluationResult(
                task_id="offline.task1.metrics_csv",
                total_score=10.0,
                max_score=10.0,
                sub_scores={"precision": 0.85},
            ),
            EvaluationResult(
                task_id="offline.task1.metrics_csv",
                total_score=10.0,
                max_score=10.0,
                sub_scores={"precision": 0.85},
            ),
        ]

        multi_run_results = {"offline.task1.metrics_csv": task1_results}
        bonus = self.analyzer.calculate_stability_bonus(multi_run_results)
        details = self.analyzer.get_analysis_details()

        # Verify bonus is calculated and details are present
        assert bonus >= 0.0
        assert "task1_consistency" in details
        assert "structural_consistency" in details
        assert "overall_consistency" in details
        assert "threshold" in details
        assert "runs_analyzed" in details
        assert details["runs_analyzed"] == 2
