"""
Automated QA Pipeline and Reporting Tests for Stage 12 Analytics QA Validation.

This module provides comprehensive automated testing infrastructure for the
Advanced Analytics & Insights Engine, including test orchestration, reporting,
and continuous validation capabilities.
"""

import json
import os
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest


class QATestRunner:
    """Automated QA test runner and orchestrator."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize QA test runner with configuration."""
        self.config = config or {
            "test_timeout": 300,  # 5 minutes
            "parallel_jobs": 4,
            "coverage_threshold": 80.0,
            "performance_threshold": 2.0,  # seconds
            "memory_threshold": 500,  # MB
        }
        self.results = {}
        self.start_time = None
        self.end_time = None

    def run_test_suite(self, test_categories: list[str] = None) -> dict[str, Any]:
        """Run comprehensive QA test suite."""
        self.start_time = datetime.now()

        if test_categories is None:
            test_categories = [
                "statistical_accuracy",
                "performance",
                "dashboard",
                "security",
                "integration",
            ]

        results = {}

        for category in test_categories:
            # Running tests for category
            category_result = self._run_category_tests(category)
            results[category] = category_result

            # Early exit on critical failures
            if category_result.get("critical_failures", 0) > 0:
                # Critical failures detected, stopping execution
                break

        self.end_time = datetime.now()
        self.results = results
        return results

    def _run_category_tests(self, category: str) -> dict[str, Any]:
        """Run tests for a specific category."""
        test_files = {
            "statistical_accuracy": "tests/test_analytics_statistics.py",
            "performance": "tests/test_qa_performance.py",
            "dashboard": "tests/test_qa_dashboard.py",
            "security": "tests/test_qa_security.py",
            "integration": "tests/test_qa_integration.py",
        }

        test_file = test_files.get(category)
        if not test_file or not Path(test_file).exists():
            return {"status": "skipped", "reason": f"Test file {test_file} not found"}

        try:
            # Run pytest with coverage and timing
            cmd = [
                "python",
                "-m",
                "pytest",
                test_file,
                "--cov=bench",
                "--cov-report=json",
                "--tb=short",
                "-v",
                "--durations=10",
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.config["test_timeout"]
            )

            return {
                "status": "passed" if result.returncode == 0 else "failed",
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": self._extract_duration(result.stdout),
                "test_count": self._extract_test_count(result.stdout),
                "failures": self._extract_failures(result.stdout),
                "critical_failures": self._count_critical_failures(result.stdout),
            }

        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "duration": self.config["test_timeout"],
                "critical_failures": 1,
            }
        except Exception as e:
            return {"status": "error", "error": str(e), "critical_failures": 1}

    def _extract_duration(self, output: str) -> float:
        """Extract test duration from pytest output."""
        try:
            lines = output.split("\n")
            for line in lines:
                if " in " in line and ("passed" in line or "failed" in line):
                    # Extract time like "3 passed in 2.34s"
                    parts = line.split(" in ")
                    if len(parts) > 1:
                        time_part = parts[1].split()[0]  # Get first word after "in"
                        time_str = time_part.replace("s", "").strip()
                        return float(time_str)
        except Exception:
            # Expected for malformed output
            return 0.0
        return 0.0

    def _extract_test_count(self, output: str) -> int:
        """Extract number of tests from pytest output."""
        try:
            lines = output.split("\n")
            for line in lines:
                if ("passed" in line or "failed" in line) and " in " in line:
                    # Parse lines like "3 passed in 2.34s"
                    # Split by " in " and take the first part
                    before_in = line.split(" in ")[0]
                    # Remove commas and split
                    before_in = before_in.replace(",", "")
                    parts = before_in.split()
                    total = 0
                    i = 0
                    while i < len(parts):
                        if parts[i].isdigit() and i + 1 < len(parts):
                            if parts[i + 1] in ["passed", "failed", "error", "skipped"]:
                                total += int(parts[i])
                                i += 2  # Skip the status word
                            else:
                                i += 1
                        else:
                            i += 1
                    if total > 0:
                        return total
        except Exception:
            # Expected for malformed output
            return 0
        return 0

    def _extract_failures(self, output: str) -> list[str]:
        """Extract failure information from pytest output."""
        failures = []
        try:
            lines = output.split("\n")
            in_failure = False
            current_failure = []

            for line in lines:
                if line.startswith("FAILED "):
                    if current_failure:
                        failures.append("\n".join(current_failure))
                    current_failure = [line]
                    in_failure = True
                elif in_failure:
                    if line.startswith("=") or line.startswith("_"):
                        if current_failure:
                            failures.append("\n".join(current_failure))
                        current_failure = []
                        in_failure = False
                    else:
                        current_failure.append(line)

            if current_failure:
                failures.append("\n".join(current_failure))

        except Exception:
            # Expected for malformed output
            return []
        return failures

    def _count_critical_failures(self, output: str) -> int:
        """Count critical failures that should stop execution."""
        critical_keywords = [
            "ImportError",
            "ModuleNotFoundError",
            "SyntaxError",
            "AttributeError",
            "TypeError",
            "NameError",
        ]

        count = 0
        for keyword in critical_keywords:
            count += output.count(keyword)
        return count

    def generate_report(self) -> dict[str, Any]:
        """Generate comprehensive QA report."""
        if not self.results:
            return {"error": "No test results available"}

        total_duration = (
            (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        )

        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_duration": total_duration,
            "categories_tested": len(self.results),
            "overall_status": self._determine_overall_status(),
            "test_summary": self._generate_test_summary(),
            "performance_metrics": self._generate_performance_metrics(),
            "recommendations": self._generate_recommendations(),
        }

        return {
            "summary": summary,
            "detailed_results": self.results,
            "configuration": self.config,
        }

    def _determine_overall_status(self) -> str:
        """Determine overall QA validation status."""
        if not self.results:
            return "unknown"

        critical_failures = sum(
            result.get("critical_failures", 0) for result in self.results.values()
        )

        if critical_failures > 0:
            return "critical_failure"

        failed_categories = [
            category
            for category, result in self.results.items()
            if result.get("status") == "failed"
        ]

        if len(failed_categories) > len(self.results) / 2:
            return "failed"
        elif failed_categories:
            return "partial_success"
        else:
            return "passed"

    def _generate_test_summary(self) -> dict[str, Any]:
        """Generate test execution summary."""
        total_tests = sum(
            result.get("test_count", 0) for result in self.results.values()
        )

        total_failures = sum(
            len(result.get("failures", [])) for result in self.results.values()
        )

        return {
            "total_tests": total_tests,
            "total_failures": total_failures,
            "success_rate": (total_tests - total_failures) / max(total_tests, 1) * 100,
            "categories_passed": len(
                [r for r in self.results.values() if r.get("status") == "passed"]
            ),
            "categories_failed": len(
                [r for r in self.results.values() if r.get("status") == "failed"]
            ),
        }

    def _generate_performance_metrics(self) -> dict[str, Any]:
        """Generate performance analysis."""
        durations = [result.get("duration", 0) for result in self.results.values()]

        return {
            "total_execution_time": sum(durations),
            "average_category_time": sum(durations) / max(len(durations), 1),
            "slowest_category": max(
                self.results.items(),
                key=lambda x: x[1].get("duration", 0),
                default=("none", {}),
            )[0],
            "performance_threshold_met": all(
                d <= self.config["performance_threshold"] for d in durations
            ),
        }

    def _generate_recommendations(self) -> list[str]:
        """Generate QA improvement recommendations."""
        recommendations = []

        # Check for failed categories
        failed_categories = [
            category
            for category, result in self.results.items()
            if result.get("status") == "failed"
        ]

        if failed_categories:
            recommendations.append(
                f"Address failures in: {', '.join(failed_categories)}"
            )

        # Check performance
        slow_categories = [
            category
            for category, result in self.results.items()
            if result.get("duration", 0) > self.config["performance_threshold"]
        ]

        if slow_categories:
            recommendations.append(
                f"Optimize performance for: {', '.join(slow_categories)}"
            )

        # Check for timeouts
        timeout_categories = [
            category
            for category, result in self.results.items()
            if result.get("status") == "timeout"
        ]

        if timeout_categories:
            recommendations.append(
                f"Investigate timeouts in: {', '.join(timeout_categories)}"
            )

        if not recommendations:
            recommendations.append("All QA validation tests passed successfully")

        return recommendations


class TestQAAutomation:
    """Test the automated QA pipeline functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = QATestRunner(
            {
                "test_timeout": 60,
                "parallel_jobs": 2,
                "coverage_threshold": 70.0,
                "performance_threshold": 10.0,
                "memory_threshold": 200,
            }
        )

    def test_qa_runner_initialization(self):
        """Test QA runner initialization."""
        assert self.runner.config["test_timeout"] == 60
        assert self.runner.config["parallel_jobs"] == 2
        assert self.runner.results == {}
        assert self.runner.start_time is None

    def test_test_category_execution(self):
        """Test individual test category execution."""
        # Test with a simple analytics test
        result = self.runner._run_category_tests("statistical_accuracy")

        assert "status" in result
        assert result["status"] in ["passed", "failed", "skipped", "timeout", "error"]

        if result["status"] not in ["skipped", "error"]:
            assert "duration" in result
            assert "test_count" in result
            assert isinstance(result.get("failures", []), list)

    def test_duration_extraction(self):
        """Test duration extraction from pytest output."""
        sample_output = "================ 3 passed in 2.34s ================"
        duration = self.runner._extract_duration(sample_output)
        assert duration == 2.34

        # Test with failure output
        sample_output2 = "============ 2 passed, 1 failed in 1.56s ============"
        duration2 = self.runner._extract_duration(sample_output2)
        assert duration2 == 1.56

    def test_test_count_extraction(self):
        """Test test count extraction from pytest output."""
        sample_output = "================ 5 passed in 2.34s ================"
        count = self.runner._extract_test_count(sample_output)
        assert count == 5

        # Test with mixed results
        sample_output2 = "========== 3 passed, 2 failed in 1.56s =========="
        count2 = self.runner._extract_test_count(sample_output2)
        assert count2 == 5

    def test_failure_extraction(self):
        """Test failure information extraction."""
        sample_output = """
FAILED tests/test_example.py::test_function - AssertionError: Expected 5, got 3
    def test_function():
>       assert 5 == 3
E       AssertionError: Expected 5, got 3
======================== short test summary info =========================
"""
        failures = self.runner._extract_failures(sample_output)
        assert len(failures) >= 1
        assert "FAILED tests/test_example.py::test_function" in failures[0]

    def test_critical_failure_detection(self):
        """Test critical failure detection."""
        sample_output_with_import_error = """
ImportError: No module named 'missing_module'
AttributeError: 'NoneType' object has no attribute 'method'
"""
        critical_count = self.runner._count_critical_failures(
            sample_output_with_import_error
        )
        assert critical_count == 2

        # Test with no critical failures
        normal_output = "AssertionError: Expected 5, got 3"
        normal_count = self.runner._count_critical_failures(normal_output)
        assert normal_count == 0

    def test_overall_status_determination(self):
        """Test overall status determination logic."""
        # Test with all passed
        self.runner.results = {
            "category1": {"status": "passed", "critical_failures": 0},
            "category2": {"status": "passed", "critical_failures": 0},
        }
        status = self.runner._determine_overall_status()
        assert status == "passed"

        # Test with critical failure
        self.runner.results = {
            "category1": {"status": "failed", "critical_failures": 1}
        }
        status = self.runner._determine_overall_status()
        assert status == "critical_failure"

        # Test with partial success
        self.runner.results = {
            "category1": {"status": "passed", "critical_failures": 0},
            "category2": {"status": "failed", "critical_failures": 0},
            "category3": {"status": "passed", "critical_failures": 0},
        }
        status = self.runner._determine_overall_status()
        assert status == "partial_success"

    def test_test_summary_generation(self):
        """Test test summary generation."""
        self.runner.results = {
            "category1": {"status": "passed", "test_count": 5, "failures": []},
            "category2": {
                "status": "failed",
                "test_count": 3,
                "failures": ["failure1", "failure2"],
            },
        }

        summary = self.runner._generate_test_summary()
        assert summary["total_tests"] == 8
        assert summary["total_failures"] == 2
        assert summary["success_rate"] == 75.0
        assert summary["categories_passed"] == 1
        assert summary["categories_failed"] == 1

    def test_performance_metrics_generation(self):
        """Test performance metrics generation."""
        self.runner.results = {
            "category1": {"duration": 1.5},
            "category2": {"duration": 2.3},
            "category3": {"duration": 0.8},
        }

        metrics = self.runner._generate_performance_metrics()
        assert metrics["total_execution_time"] == 4.6
        assert metrics["average_category_time"] == pytest.approx(1.53, rel=1e-2)
        assert metrics["slowest_category"] == "category2"
        assert metrics["performance_threshold_met"]  # All under 10.0s threshold

    def test_recommendations_generation(self):
        """Test QA recommendations generation."""
        # Test with failures
        self.runner.results = {
            "category1": {"status": "failed", "duration": 1.0},
            "category2": {"status": "timeout", "duration": 15.0},
        }

        recommendations = self.runner._generate_recommendations()
        assert len(recommendations) >= 2
        assert any("category1" in rec for rec in recommendations)
        assert any("timeout" in rec.lower() for rec in recommendations)

        # Test with all passed
        self.runner.results = {
            "category1": {"status": "passed", "duration": 1.0},
            "category2": {"status": "passed", "duration": 2.0},
        }

        recommendations = self.runner._generate_recommendations()
        assert len(recommendations) == 1
        assert "successfully" in recommendations[0]

    def test_report_generation(self):
        """Test comprehensive report generation."""
        self.runner.start_time = datetime.now() - timedelta(seconds=10)
        self.runner.end_time = datetime.now()
        self.runner.results = {
            "category1": {
                "status": "passed",
                "test_count": 5,
                "duration": 2.0,
                "failures": [],
                "critical_failures": 0,
            }
        }

        report = self.runner.generate_report()

        assert "summary" in report
        assert "detailed_results" in report
        assert "configuration" in report

        summary = report["summary"]
        assert "timestamp" in summary
        assert "total_duration" in summary
        assert "overall_status" in summary
        assert "test_summary" in summary
        assert "performance_metrics" in summary
        assert "recommendations" in summary

        assert summary["categories_tested"] == 1
        assert summary["overall_status"] == "passed"


def test_qa_pipeline_integration():
    """Integration test for the complete QA pipeline."""
    runner = QATestRunner({"test_timeout": 30, "performance_threshold": 5.0})

    # Run a subset of tests for integration validation
    results = runner.run_test_suite(["statistical_accuracy"])

    assert isinstance(results, dict)
    assert len(results) >= 1

    # Generate report
    report = runner.generate_report()
    assert "summary" in report
    assert "detailed_results" in report

    # Verify report structure
    summary = report["summary"]
    assert "overall_status" in summary
    assert "test_summary" in summary
    assert "recommendations" in summary


def test_qa_report_persistence():
    """Test QA report persistence and loading."""
    runner = QATestRunner()
    runner.start_time = datetime.now() - timedelta(seconds=5)
    runner.end_time = datetime.now()
    runner.results = {
        "test_category": {
            "status": "passed",
            "test_count": 3,
            "duration": 1.5,
            "failures": [],
            "critical_failures": 0,
        }
    }

    report = runner.generate_report()

    # Test JSON serialization
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(report, f, indent=2)
        temp_file = f.name

    try:
        # Test loading
        with open(temp_file) as f:
            loaded_report = json.load(f)

        assert (
            loaded_report["summary"]["overall_status"]
            == report["summary"]["overall_status"]
        )
        assert loaded_report["detailed_results"] == report["detailed_results"]

    finally:
        os.unlink(temp_file)


if __name__ == "__main__":
    # Run QA automation tests
    pytest.main([__file__, "-v"])
