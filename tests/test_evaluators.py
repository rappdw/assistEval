"""Comprehensive unit tests for Stage 7 evaluators.

Tests cover all three offline task evaluators with edge cases,
error conditions, and integration scenarios.
"""

from unittest.mock import patch

import pytest

from bench.core.evaluators import (
    EvaluationResult,
    EvaluatorRegistry,
    ExecSummaryEvaluator,
    MetricsCSVEvaluator,
    RegexMatchEvaluator,
)


class TestEvaluationResult:
    """Test EvaluationResult data structure."""

    def test_basic_creation(self):
        """Test basic EvaluationResult creation."""
        result = EvaluationResult(
            task_id="test.task",
            total_score=85.5,
            max_score=100.0,
            sub_scores={"component1": 0.8, "component2": 0.9},
        )

        assert result.task_id == "test.task"
        assert result.total_score == 85.5
        assert result.max_score == 100.0
        assert result.score_percentage == 85.5
        assert result.sub_scores == {"component1": 0.8, "component2": 0.9}
        assert result.errors == []
        assert result.warnings == []

    def test_score_percentage_calculation(self):
        """Test score percentage calculation."""
        result = EvaluationResult(
            task_id="test", total_score=75.0, max_score=100.0, sub_scores={}
        )
        assert result.score_percentage == 75.0

        # Test zero max score
        result.max_score = 0.0
        assert result.score_percentage == 0.0

    def test_error_and_warning_methods(self):
        """Test error and warning addition methods."""
        result = EvaluationResult(
            task_id="test", total_score=0.0, max_score=100.0, sub_scores={}
        )

        result.add_error("Test error")
        result.add_warning("Test warning")

        assert result.errors == ["Test error"]
        assert result.warnings == ["Test warning"]


class TestEvaluatorRegistry:
    """Test EvaluatorRegistry functionality."""

    def test_registry_operations(self):
        """Test basic registry operations."""
        # Test listing evaluators
        evaluators = EvaluatorRegistry.list_evaluators()
        assert "metrics_csv" in evaluators
        assert "regex_match" in evaluators
        assert "exec_summary" in evaluators

    def test_create_evaluator(self):
        """Test evaluator creation."""
        config = {"tolerance": 0.001}
        evaluator = EvaluatorRegistry.create_evaluator("metrics_csv", config)
        assert isinstance(evaluator, MetricsCSVEvaluator)
        assert evaluator.tolerance == 0.001

    def test_unknown_evaluator(self):
        """Test handling of unknown evaluator."""
        with pytest.raises(KeyError, match="Evaluator 'unknown' not found"):
            EvaluatorRegistry.create_evaluator("unknown", {})

    def test_is_registered(self):
        """Test evaluator registration checking."""
        assert EvaluatorRegistry.is_registered("metrics_csv")
        assert not EvaluatorRegistry.is_registered("unknown_evaluator")


class TestMetricsCSVEvaluator:
    """Test MetricsCSVEvaluator functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {"tolerance": 0.0005}
        self.evaluator = MetricsCSVEvaluator(self.config)

        self.test_case = {"id": "offline.task1.metrics_csv"}

        self.sample_response = {
            "metrics": {
                "precision": 0.8500,
                "recall": 0.7800,
                "f1": 0.8135,
                "accuracy": 0.8900,
                "confusion_matrix": {"tp": 156, "fp": 28, "fn": 44, "tn": 372},
            }
        }

        self.sample_answer_key = {
            "metrics": {
                "precision": 0.8500,
                "recall": 0.7800,
                "f1": 0.8135,
                "accuracy": 0.8900,
                "confusion_matrix": {"tp": 156, "fp": 28, "fn": 44, "tn": 372},
            }
        }

    def test_perfect_match(self):
        """Test evaluation with perfect match."""
        result = self.evaluator.evaluate(
            self.sample_response, self.test_case, self.sample_answer_key
        )

        assert result.task_id == "offline.task1.metrics_csv"
        assert result.total_score == result.max_score  # Perfect score
        assert result.score_percentage == 100.0
        assert len(result.errors) == 0
        # Verify all components scored perfectly
        assert all(score == 1.0 for score in result.sub_scores.values())

    def test_tolerance_boundaries(self):
        """Test tolerance boundary conditions."""
        # Test within tolerance
        response_within = {
            "metrics": {
                "precision": 0.8504,  # +0.0004 (within tolerance)
                "recall": 0.7796,  # -0.0004 (within tolerance)
                "f1": 0.8135,
                "accuracy": 0.8900,
                "confusion_matrix": {"tp": 156, "fp": 28, "fn": 44, "tn": 372},
            }
        }

        result = self.evaluator.evaluate(
            response_within, self.test_case, self.sample_answer_key
        )

        # Should still get full points for precision and recall
        assert result.sub_scores["precision"] == 1.0
        assert result.sub_scores["recall"] == 1.0

    def test_outside_tolerance(self):
        """Test values outside tolerance."""
        response_outside = {
            "metrics": {
                "precision": 0.8510,  # +0.001 (outside tolerance)
                "recall": 0.7790,  # -0.001 (outside tolerance)
                "f1": 0.8135,
                "accuracy": 0.8900,
                "confusion_matrix": {"tp": 156, "fp": 28, "fn": 44, "tn": 372},
            }
        }

        result = self.evaluator.evaluate(
            response_outside, self.test_case, self.sample_answer_key
        )

        # Should get 0 points for precision and recall
        assert result.sub_scores["precision"] == 0.0
        assert result.sub_scores["recall"] == 0.0
        assert len(result.warnings) >= 2

    def test_confusion_matrix_mismatch(self):
        """Test confusion matrix with mismatched values."""
        response_cm_wrong = {
            "metrics": {
                "precision": 0.8500,
                "recall": 0.7800,
                "f1": 0.8135,
                "accuracy": 0.8900,
                "confusion_matrix": {
                    "tp": 155,  # Wrong value
                    "fp": 28,
                    "fn": 44,
                    "tn": 372,
                },
            }
        }

        result = self.evaluator.evaluate(
            response_cm_wrong, self.test_case, self.sample_answer_key
        )

        # Should get partial confusion matrix score (3/4 = 0.75)
        assert result.sub_scores["confusion_matrix"] == 0.75

    def test_missing_fields(self):
        """Test handling of missing fields."""
        response_missing = {
            "metrics": {
                "precision": 0.8500,
                # Missing recall, f1, accuracy, confusion_matrix
            }
        }

        result = self.evaluator.evaluate(
            response_missing, self.test_case, self.sample_answer_key
        )

        assert result.sub_scores["precision"] == 1.0
        assert result.sub_scores["recall"] == 0.0
        assert result.sub_scores["f1"] == 0.0
        assert result.sub_scores["accuracy"] == 0.0
        assert result.sub_scores["confusion_matrix"] == 0.0
        assert len(result.errors) >= 4

    def test_no_answer_key(self):
        """Test evaluation without answer key."""
        result = self.evaluator.evaluate(self.sample_response, self.test_case, None)

        assert result.total_score == 0.0
        assert len(result.errors) == 1
        assert "No answer key provided" in result.errors[0]

    def test_invalid_numeric_values(self):
        """Test handling of invalid numeric values."""
        response_invalid = {
            "metrics": {
                "precision": "not_a_number",
                "recall": None,
                "f1": 0.8135,
                "accuracy": 0.8900,
                "confusion_matrix": {"tp": 156, "fp": 28, "fn": 44, "tn": 372},
            }
        }

        result = self.evaluator.evaluate(
            response_invalid, self.test_case, self.sample_answer_key
        )

        assert result.sub_scores["precision"] == 0.0
        assert result.sub_scores["recall"] == 0.0
        assert len(result.errors) >= 2


class TestRegexMatchEvaluator:
    """Test RegexMatchEvaluator functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {"timeout_ms": 100}
        self.evaluator = RegexMatchEvaluator(self.config)

        self.test_case = {
            "id": "offline.task2.ssn_regex",
            "test_lines": [
                "123-45-6789",  # Valid SSN
                "000-45-6789",  # Invalid area code
                "123-00-6789",  # Invalid group code
                "123-45-0000",  # Invalid serial
                "666-45-6789",  # Invalid area code
                "900-45-6789",  # Invalid area code
            ],
        }

        self.answer_key = {
            "expected_matches": [True, False, False, False, False, False]
        }

    def test_valid_regex_pattern(self):
        """Test evaluation with valid regex pattern."""
        response_data = {
            "regex": "^(?!000|666|9\\d{2})\\d{3}-(?!00)\\d{2}-(?!0000)\\d{4}$"
        }

        result = self.evaluator.evaluate(response_data, self.test_case, self.answer_key)

        assert result.task_id == "offline.task2.ssn_regex"
        assert result.total_score > 0
        assert result.sub_scores["regex_validity"] > 0
        assert result.sub_scores["line_matches"] > 0

    def test_invalid_regex_compilation(self):
        """Test handling of invalid regex patterns."""
        response_data = {"regex": "^[invalid regex pattern"}  # Missing closing bracket

        result = self.evaluator.evaluate(response_data, self.test_case, self.answer_key)

        assert result.sub_scores["regex_validity"] == 0.0
        assert len(result.errors) >= 1
        assert "compilation failed" in result.errors[0].lower()

    def test_missing_regex_pattern(self):
        """Test handling of missing regex pattern."""
        response_data = {}  # No regex field

        result = self.evaluator.evaluate(response_data, self.test_case, self.answer_key)

        assert result.sub_scores["regex_validity"] == 0.0
        assert result.sub_scores["line_matches"] == 0.0
        assert len(result.errors) >= 1

    def test_regex_without_anchors(self):
        """Test regex pattern without proper anchoring."""
        response_data = {"regex": "\\d{3}-\\d{2}-\\d{4}"}  # No ^ and $ anchors

        result = self.evaluator.evaluate(response_data, self.test_case, self.answer_key)

        # Should compile but lose points for missing anchors
        assert result.sub_scores["regex_validity"] < 1.0
        assert len(result.warnings) >= 1

    @patch("bench.core.evaluators.regex_match.timeout_guard")
    def test_timeout_protection(self, mock_timeout):
        """Test timeout protection during regex evaluation."""
        mock_timeout.side_effect = Exception("Timeout")

        response_data = {
            "regex": "^(?!000|666|9\\d{2})\\d{3}-(?!00)\\d{2}-(?!0000)\\d{4}$"
        }

        result = self.evaluator.evaluate(response_data, self.test_case, self.answer_key)

        # Should handle timeout gracefully
        assert result.sub_scores["regex_validity"] < 1.0

    def test_constraint_validation(self):
        """Test SSN constraint validation."""
        # Pattern that incorrectly accepts invalid SSNs
        response_data = {"regex": "^\\d{3}-\\d{2}-\\d{4}$"}  # Too permissive

        result = self.evaluator.evaluate(response_data, self.test_case, self.answer_key)

        # Should lose points for not rejecting invalid patterns
        validity_details = result.details.get("validity_checks", {})
        assert not validity_details.get("rejects_000_area", True)
        assert not validity_details.get("rejects_666_area", True)


class TestExecSummaryEvaluator:
    """Test ExecSummaryEvaluator functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {}
        self.evaluator = ExecSummaryEvaluator(self.config)

        self.test_case = {"id": "offline.task3.exec_summary"}

        self.perfect_response = {
            "title": "Q3 Sales Report",  # 3 words (≤6)
            "summary": (
                "The third quarter demonstrated exceptional performance across all "
                "business segments with total revenue increasing by 15% compared to "
                "the previous quarter. Our newly launched product line contributed "
                "significantly to overall growth metrics while operational efficiency "
                "improvements successfully reduced operational costs by 8% through "
                "streamlined processes. Customer satisfaction scores remained "
                "consistently high at 92% indicating strong market positioning and "
                "brand loyalty. The sales team exceeded quarterly targets in both "
                "domestic and international markets with particularly impressive "
                "results in the technology and healthcare sectors. Marketing campaigns "
                "generated substantial lead conversion rates while maintaining "
                "cost-effective customer acquisition strategies. Supply chain "
                "optimization efforts resulted in improved delivery times and reduced "
                "inventory carrying costs. Employee engagement scores increased by 12% "
                "following implementation of new workplace flexibility policies. "
                "Strategic partnerships established during this quarter are expected "
                "to drive additional revenue growth in subsequent periods. Overall "
                "financial performance exceeded board expectations and positions the "
                "company well for continued expansion."
            ),  # 140+ words (120-160)
            "bullets": [
                "Revenue increased 15% quarter-over-quarter",
                "New product line drove significant growth",
                "Customer satisfaction maintained at 92%",
            ],  # Exactly 3 bullets
        }

    def test_perfect_structure(self):
        """Test evaluation with perfect structural compliance."""
        result = self.evaluator.evaluate(self.perfect_response, self.test_case, None)

        assert result.task_id == "offline.task3.exec_summary"
        assert result.sub_scores["structure"] == 1.0
        assert result.total_score > 0
        assert len(result.errors) == 0

    def test_title_too_long(self):
        """Test handling of title that's too long."""
        response_long_title = self.perfect_response.copy()
        response_long_title[
            "title"
        ] = "This is a very long title that exceeds six words"  # 11 words

        result = self.evaluator.evaluate(response_long_title, self.test_case, None)

        assert result.sub_scores["structure"] < 1.0
        assert len(result.warnings) >= 1
        assert "Title too long" in result.warnings[0]

    def test_word_count_out_of_range(self):
        """Test handling of summary word count outside range."""
        response_short = self.perfect_response.copy()
        response_short["summary"] = "Too short."  # Only 2 words (need 120-160)

        result = self.evaluator.evaluate(response_short, self.test_case, None)

        assert result.sub_scores["structure"] < 1.0
        assert len(result.warnings) >= 1
        assert "Summary has 2 words" in result.warnings[0]

    def test_wrong_bullet_count(self):
        """Test handling of wrong number of bullets."""
        response_wrong_bullets = self.perfect_response.copy()
        response_wrong_bullets["bullets"] = [
            "Only one bullet",
            "Only two bullets",
        ]  # 2 bullets instead of 3

        result = self.evaluator.evaluate(response_wrong_bullets, self.test_case, None)

        assert result.sub_scores["structure"] < 1.0
        assert len(result.warnings) >= 1
        # Check that bullet count warning is among the warnings
        bullet_warnings = [w for w in result.warnings if "bullets" in w.lower()]
        assert len(bullet_warnings) >= 1

    def test_missing_required_fields(self):
        """Test handling of missing required fields."""
        response_missing = {
            "title": "Test Title"
            # Missing summary and bullets
        }

        result = self.evaluator.evaluate(response_missing, self.test_case, None)

        assert result.sub_scores["structure"] < 1.0
        assert len(result.errors) >= 2
        assert any("Missing summary text" in error for error in result.errors)

    def test_hype_terms_detection(self):
        """Test detection of hype terms in tone evaluation."""
        response_hype = self.perfect_response.copy()
        response_hype["summary"] = (
            "Our revolutionary and groundbreaking product achieved amazing results "
            "with incredible performance that was absolutely phenomenal and "
            "outstanding in every way possible."
        )  # 130 words with hype terms

        result = self.evaluator.evaluate(response_hype, self.test_case, None)

        assert result.sub_scores["tone"] < 1.0
        assert len(result.warnings) >= 1
        hype_details = result.details.get("hype_terms_found", [])
        assert len(hype_details) > 0

    def test_sentence_length_analysis(self):
        """Test sentence length analysis."""
        # Create summary with very long sentences
        long_sentence = " ".join(["word"] * 30)  # 30-word sentence
        response_long_sentences = self.perfect_response.copy()
        response_long_sentences["summary"] = (
            f"{long_sentence}. {long_sentence}. {long_sentence}. "
            f"{long_sentence}. {long_sentence}."
        )

        result = self.evaluator.evaluate(response_long_sentences, self.test_case, None)

        assert result.sub_scores["tone"] < 1.0
        assert result.details.get("avg_sentence_length", 0) > 24

    def test_professional_tone_checks(self):
        """Test professional tone heuristics."""
        response_unprofessional = self.perfect_response.copy()
        response_unprofessional["summary"] = (
            "WOW!!! This quarter was ABSOLUTELY AMAZING!!! We did SO WELL and "
            "everything was FANTASTIC!!!"
        )  # Multiple exclamations, excessive caps

        result = self.evaluator.evaluate(response_unprofessional, self.test_case, None)

        assert result.sub_scores["tone"] < 1.0
        assert result.details.get("professional_tone_status") == "fail"

    def test_empty_response(self):
        """Test handling of completely empty response."""
        result = self.evaluator.evaluate({}, self.test_case, None)

        assert result.total_score == 0.0
        assert result.sub_scores["structure"] < 1.0
        assert result.sub_scores["tone"] == 0.0
        assert len(result.errors) >= 3


class TestIntegration:
    """Integration tests for evaluator system."""

    def test_evaluator_registry_integration(self):
        """Test full integration through registry."""
        config = {"tolerance": 0.001}
        evaluator = EvaluatorRegistry.create_evaluator("metrics_csv", config)

        response_data = {
            "metrics": {
                "precision": 0.8500,
                "recall": 0.7800,
                "f1": 0.8135,
                "accuracy": 0.8900,
                "confusion_matrix": {"tp": 156, "fp": 28, "fn": 44, "tn": 372},
            }
        }

        test_case = {"id": "integration.test"}
        answer_key = {
            "metrics": {
                "precision": 0.8500,
                "recall": 0.7800,
                "f1": 0.8135,
                "accuracy": 0.8900,
                "confusion_matrix": {"tp": 156, "fp": 28, "fn": 44, "tn": 372},
            }
        }

        result = evaluator.evaluate(response_data, test_case, answer_key)

        assert isinstance(result, EvaluationResult)
        assert result.total_score > 0
        assert result.task_id == "integration.test"

    def test_all_evaluators_registered(self):
        """Test that all expected evaluators are registered."""
        evaluators = EvaluatorRegistry.list_evaluators()

        expected_evaluators = ["metrics_csv", "regex_match", "exec_summary"]
        for expected in expected_evaluators:
            assert expected in evaluators

    def test_evaluator_error_handling(self):
        """Test error handling across all evaluators."""
        test_cases = [
            ("metrics_csv", {}),
            ("regex_match", {}),
            ("exec_summary", {}),
        ]

        for evaluator_name, response_data in test_cases:
            evaluator = EvaluatorRegistry.create_evaluator(evaluator_name, {})
            test_case = {"id": f"error.test.{evaluator_name}"}

            result = evaluator.evaluate(response_data, test_case, None)

            # Should handle errors gracefully
            assert isinstance(result, EvaluationResult)
            assert result.total_score >= 0
            assert result.max_score > 0
