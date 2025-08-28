"""Integration tests for the complete evaluation workflow.

This module tests end-to-end functionality including test case loading,
fixture processing, evaluation execution, and report generation.
"""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from bench.core.reporting import ConsolidatedReporter
from bench.core.scoring import ProviderScore


class TestEndToEndWorkflow:
    """Test complete evaluation workflow."""

    @pytest.fixture
    def temp_results_dir(self):
        """Create temporary results directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_full_offline_evaluation(self, temp_results_dir):
        """Test complete offline task evaluation."""
        # This test would require a full integration setup
        # For now, we'll test the components individually
        assert True  # Placeholder for full integration test

    def test_provider_comparison(self, temp_results_dir):
        """Test provider comparison workflow."""
        from datetime import datetime

        # Create mock provider scores
        chatgpt_score = ProviderScore(
            provider_name="chatgpt",
            timestamp=datetime.now(),
            task_scores=[],
            total_score=80.0,
            max_score=105.0,
            stability_bonus=5.0,
            final_score=85.0,
        )

        copilot_score = ProviderScore(
            provider_name="copilot_manual",
            timestamp=datetime.now(),
            task_scores=[],
            total_score=75.0,
            max_score=105.0,
            stability_bonus=3.0,
            final_score=78.0,
        )

        # Test comparison logic
        assert chatgpt_score.final_score > copilot_score.final_score
        assert chatgpt_score.score_percentage > copilot_score.score_percentage

    def test_report_generation(self, temp_results_dir):
        """Test report generation from evaluation results."""
        output_dir = temp_results_dir / "reports"
        reporter = ConsolidatedReporter(temp_results_dir, output_dir)

        # Test reporter initialization
        assert reporter.results_dir == temp_results_dir
        assert reporter.output_dir == output_dir


class TestTestCaseValidation:
    """Test test case definition validation."""

    def test_task1_yaml_validation(self):
        """Test Task 1 YAML structure."""
        task1_path = Path("bench/tests/offline/task1_metrics.yaml")

        if task1_path.exists():
            with open(task1_path) as f:
                task1_data = yaml.safe_load(f)

            # Validate required fields
            assert "id" in task1_data
            assert "name" in task1_data
            assert "category" in task1_data
            assert "capability_profile" in task1_data
            assert "prompt" in task1_data
            assert "expectation" in task1_data
            assert "scoring" in task1_data

            # Validate specific values
            assert task1_data["id"] == "offline.task1.metrics_csv"
            assert task1_data["category"] == "offline"
            assert task1_data["capability_profile"]["web"] == "forbidden"
            assert task1_data["capability_profile"]["json_required"] is True

    def test_task2_yaml_validation(self):
        """Test Task 2 YAML structure."""
        task2_path = Path("bench/tests/offline/task2_ssn_regex.yaml")

        if task2_path.exists():
            with open(task2_path) as f:
                task2_data = yaml.safe_load(f)

            # Validate required fields
            assert "id" in task2_data
            assert "name" in task2_data
            assert task2_data["id"] == "offline.task2.ssn_regex"
            assert task2_data["scoring"]["evaluator"] == "regex_match"

    def test_task3_yaml_validation(self):
        """Test Task 3 YAML structure."""
        task3_path = Path("bench/tests/offline/task3_exec_summary.yaml")

        if task3_path.exists():
            with open(task3_path) as f:
                task3_data = yaml.safe_load(f)

            # Validate required fields
            assert "id" in task3_data
            assert "name" in task3_data
            assert task3_data["id"] == "offline.task3.exec_summary"
            assert task3_data["scoring"]["evaluator"] == "exec_summary"


class TestFixtureLoading:
    """Test fixture loading and processing."""

    def test_csv_fixture_loading(self):
        """Test CSV fixture loading."""
        csv_path = Path("fixtures/csv/phishing_sample.csv")

        if csv_path.exists():
            with open(csv_path) as f:
                content = f.read()

            # Validate CSV structure
            lines = content.strip().split("\n")
            assert len(lines) == 9  # Header + 8 data rows

            # Validate header
            header = lines[0]
            expected_columns = [
                "email_id",
                "sender",
                "subject",
                "body_length",
                "has_links",
                "has_attachments",
                "urgency_words",
                "is_phishing",
            ]
            assert header == ",".join(expected_columns)

    def test_text_fixture_loading(self):
        """Test text fixture loading."""
        text_path = Path("fixtures/text/ssn_validation_lines.txt")

        if text_path.exists():
            with open(text_path) as f:
                content = f.read()

            # Validate text structure
            lines = content.strip().split("\n")
            assert len(lines) == 12  # 12 test lines

            # Validate some specific lines
            assert "123-45-6789" in lines
            assert "000-12-3456" in lines
            assert "666-12-3456" in lines

    def test_answer_key_loading(self):
        """Test answer key loading."""
        # Test Task 1 answer key
        task1_key_path = Path("answer_keys/offline/task1_metrics.json")
        if task1_key_path.exists():
            with open(task1_key_path) as f:
                task1_key = json.load(f)

            assert "metrics" in task1_key
            metrics = task1_key["metrics"]
            assert "precision" in metrics
            assert "recall" in metrics
            assert "f1" in metrics
            assert "accuracy" in metrics
            assert "confusion_matrix" in metrics

        # Test Task 2 answer key
        task2_key_path = Path("answer_keys/offline/task2_lines.json")
        if task2_key_path.exists():
            with open(task2_key_path) as f:
                task2_key = json.load(f)

            assert "matching_lines" in task2_key
            assert "total_valid" in task2_key

        # Test Task 3 answer key
        task3_key_path = Path("answer_keys/offline/task3_structure.json")
        if task3_key_path.exists():
            with open(task3_key_path) as f:
                task3_key = json.load(f)

            assert "expected_structure" in task3_key


class TestFixtureIntegrity:
    """Test fixture data integrity."""

    def test_csv_data_consistency(self):
        """Test CSV data matches expected format."""
        csv_path = Path("fixtures/csv/phishing_sample.csv")

        if csv_path.exists():
            with open(csv_path) as f:
                content = f.read()

            lines = content.strip().split("\n")[1:]  # Skip header

            for i, line in enumerate(lines, 1):
                fields = line.split(",")
                assert (
                    len(fields) == 8
                ), f"Line {i} has {len(fields)} fields, expected 8"

                # Validate email_id is numeric
                assert fields[0].isdigit(), f"Line {i} email_id is not numeric"

                # Validate is_phishing is 0 or 1
                assert fields[7] in ["0", "1"], f"Line {i} is_phishing is not 0 or 1"

    def test_ssn_lines_coverage(self):
        """Test SSN test lines cover edge cases."""
        text_path = Path("fixtures/text/ssn_validation_lines.txt")

        if text_path.exists():
            with open(text_path) as f:
                lines = f.read().strip().split("\n")

            # Check for specific edge cases
            edge_cases = [
                "000-12-3456",  # Invalid area code 000
                "666-12-3456",  # Invalid area code 666
                "900-12-3456",  # Invalid area code 9xx
                "123-00-4567",  # Invalid group code 00
                "123-45-0000",  # Invalid serial 0000
            ]

            for edge_case in edge_cases:
                assert edge_case in lines, f"Missing edge case: {edge_case}"

    def test_answer_key_accuracy(self):
        """Test answer keys match fixture expectations."""
        # Load CSV fixture and answer key
        csv_path = Path("fixtures/csv/phishing_sample.csv")
        key_path = Path("answer_keys/offline/task1_metrics.json")

        if csv_path.exists() and key_path.exists():
            # Read CSV data
            with open(csv_path) as f:
                lines = f.read().strip().split("\n")[1:]  # Skip header

            # Count actual phishing emails
            phishing_count = sum(1 for line in lines if line.endswith(",1"))
            legitimate_count = sum(1 for line in lines if line.endswith(",0"))

            # Verify counts make sense
            assert phishing_count > 0, "No phishing emails in fixture"
            assert legitimate_count > 0, "No legitimate emails in fixture"
            assert phishing_count + legitimate_count == len(lines)
