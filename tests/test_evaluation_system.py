#!/usr/bin/env python3
"""Test evaluation system with known inputs and expected outputs.

This test validates that the evaluation pipeline works correctly by:
1. Creating mock responses with known correct/incorrect values
2. Running them through the evaluation system
3. Verifying scores match expected values
"""

import json
import tempfile
from pathlib import Path

from bench.core.evaluators.metrics_csv import MetricsCSVEvaluator
from bench.core.runner import TestRunner


class TestEvaluationSystem:
    """Test the evaluation system with controlled inputs."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_dir = self.temp_dir / "configs"
        self.config_dir.mkdir(parents=True)

        # Create minimal provider config
        provider_config = {
            "version": "1.0",
            "providers": {"test_provider": {"type": "test_manual", "timeout": 60}},
        }

        with open(self.config_dir / "providers.yaml", "w") as f:
            import yaml

            yaml.dump(provider_config, f)

    def test_metrics_evaluator_perfect_score(self):
        """Test metrics evaluator with perfect response."""
        # Perfect response matching answer key exactly
        response_data = {
            "task1_data_metrics": {
                "precision": 0.7500,
                "recall": 0.6000,
                "f1": 0.6667,
                "accuracy": 0.6250,
                "confusion_matrix": {"tp": 3, "fp": 1, "fn": 2, "tn": 2},
            }
        }

        # Answer key (same as response for perfect score)
        answer_key = {
            "metrics": {
                "precision": 0.7500,
                "recall": 0.6000,
                "f1": 0.6667,
                "accuracy": 0.6250,
                "confusion_matrix": {"tp": 3, "fp": 1, "fn": 2, "tn": 2},
            }
        }

        # Test case configuration
        test_case = {
            "id": "test.metrics",
            "scoring": {
                "evaluator": "metrics_csv",
                "config": {
                    "tolerance": 0.0005,
                    "weights": {
                        "precision": 6.0,
                        "recall": 6.0,
                        "f1": 6.0,
                        "accuracy": 6.0,
                        "confusion_matrix": 12.0,
                    },
                },
            },
        }

        # Create evaluator
        evaluator = MetricsCSVEvaluator(test_case["scoring"]["config"])

        # Run evaluation
        result = evaluator.evaluate(response_data, test_case, answer_key)

        # Debug output removed for linting compliance

        # Verify perfect score
        assert result.total_score == 36.0  # Sum of all weights
        assert result.max_score == 36.0
        assert result.score_percentage == 100.0
        assert len(result.errors) == 0

        # Verify individual normalized scores (0.0-1.0)
        assert result.sub_scores["precision"] == 1.0
        assert result.sub_scores["recall"] == 1.0
        assert result.sub_scores["f1"] == 1.0
        assert result.sub_scores["accuracy"] == 1.0
        assert result.sub_scores["confusion_matrix"] == 1.0

    def test_metrics_evaluator_partial_score(self):
        """Test metrics evaluator with partially correct response."""
        # Response with some correct, some incorrect values
        response_data = {
            "task1_data_metrics": {
                "precision": 0.7500,  # Correct
                "recall": 0.5000,  # Incorrect (should be 0.6000)
                "f1": 0.6667,  # Correct
                "accuracy": 0.5000,  # Incorrect (should be 0.6250)
                "confusion_matrix": {
                    "tp": 3,  # Correct
                    "fp": 2,  # Incorrect (should be 1)
                    "fn": 2,  # Correct
                    "tn": 2,  # Correct
                },
            }
        }

        # Answer key
        answer_key = {
            "metrics": {
                "precision": 0.7500,
                "recall": 0.6000,
                "f1": 0.6667,
                "accuracy": 0.6250,
                "confusion_matrix": {"tp": 3, "fp": 1, "fn": 2, "tn": 2},
            }
        }

        # Test case configuration
        test_case = {
            "id": "test.metrics",
            "scoring": {
                "evaluator": "metrics_csv",
                "config": {
                    "tolerance": 0.0005,
                    "weights": {
                        "precision": 6.0,
                        "recall": 6.0,
                        "f1": 6.0,
                        "accuracy": 6.0,
                        "confusion_matrix": 12.0,
                    },
                },
            },
        }

        # Create evaluator
        evaluator = MetricsCSVEvaluator(test_case["scoring"]["config"])

        # Run evaluation
        result = evaluator.evaluate(response_data, test_case, answer_key)

        # Verify partial score
        # Should get: precision(1.0) + f1(1.0) + confusion_matrix_partial(0.75)
        # = weighted total
        # Total weighted: 1.0*6 + 0.0*6 + 1.0*6 + 0.0*6 + 0.75*12 = 21.0
        assert result.total_score == 21.0
        assert result.max_score == 36.0
        assert abs(result.score_percentage - 58.33) < 0.1  # ~58.33%

        # Verify individual normalized scores (0.0-1.0)
        assert result.sub_scores["precision"] == 1.0  # Correct
        assert result.sub_scores["recall"] == 0.0  # Incorrect
        assert result.sub_scores["f1"] == 1.0  # Correct
        assert result.sub_scores["accuracy"] == 0.0  # Incorrect
        assert result.sub_scores["confusion_matrix"] == 0.75  # 3/4 correct

    def test_metrics_evaluator_zero_score(self):
        """Test metrics evaluator with completely incorrect response."""
        # Response with all wrong values
        response_data = {
            "task1_data_metrics": {
                "precision": 0.1000,  # Wrong
                "recall": 0.2000,  # Wrong
                "f1": 0.3000,  # Wrong
                "accuracy": 0.4000,  # Wrong
                "confusion_matrix": {
                    "tp": 10,  # Wrong
                    "fp": 20,  # Wrong
                    "fn": 30,  # Wrong
                    "tn": 40,  # Wrong
                },
            }
        }

        # Answer key
        answer_key = {
            "metrics": {
                "precision": 0.7500,
                "recall": 0.6000,
                "f1": 0.6667,
                "accuracy": 0.6250,
                "confusion_matrix": {"tp": 3, "fp": 1, "fn": 2, "tn": 2},
            }
        }

        # Test case configuration
        test_case = {
            "id": "test.metrics",
            "scoring": {
                "evaluator": "metrics_csv",
                "config": {
                    "tolerance": 0.0005,
                    "weights": {
                        "precision": 6.0,
                        "recall": 6.0,
                        "f1": 6.0,
                        "accuracy": 6.0,
                        "confusion_matrix": 12.0,
                    },
                },
            },
        }

        # Create evaluator
        evaluator = MetricsCSVEvaluator(test_case["scoring"]["config"])

        # Run evaluation
        result = evaluator.evaluate(response_data, test_case, answer_key)

        # Verify zero score
        assert result.total_score == 0.0
        assert result.max_score == 36.0
        assert result.score_percentage == 0.0

        # Verify all individual normalized scores are zero
        assert result.sub_scores["precision"] == 0.0
        assert result.sub_scores["recall"] == 0.0
        assert result.sub_scores["f1"] == 0.0
        assert result.sub_scores["accuracy"] == 0.0
        assert result.sub_scores["confusion_matrix"] == 0.0

    def test_metrics_evaluator_missing_data(self):
        """Test metrics evaluator with missing response data."""
        # Response missing required fields
        response_data = {"wrong_field": {"precision": 0.7500}}

        # Answer key
        answer_key = {
            "metrics": {
                "precision": 0.7500,
                "recall": 0.6000,
                "f1": 0.6667,
                "accuracy": 0.6250,
                "confusion_matrix": {"tp": 3, "fp": 1, "fn": 2, "tn": 2},
            }
        }

        # Test case configuration
        test_case = {
            "id": "test.metrics",
            "scoring": {
                "evaluator": "metrics_csv",
                "config": {
                    "tolerance": 0.0005,
                    "weights": {
                        "precision": 6.0,
                        "recall": 6.0,
                        "f1": 6.0,
                        "accuracy": 6.0,
                        "confusion_matrix": 12.0,
                    },
                },
            },
        }

        # Create evaluator
        evaluator = MetricsCSVEvaluator(test_case["scoring"]["config"])

        # Run evaluation
        result = evaluator.evaluate(response_data, test_case, answer_key)

        # Should get zero score due to missing data
        assert result.total_score == 0.0
        assert result.max_score == 36.0
        assert result.score_percentage == 0.0

    def test_tolerance_handling(self):
        """Test that tolerance is properly applied to numeric comparisons."""
        # Response with values within tolerance
        response_data = {
            "task1_data_metrics": {
                "precision": 0.7504,  # Within tolerance (0.0005)
                "recall": 0.5996,  # Within tolerance
                "f1": 0.6663,  # Within tolerance
                "accuracy": 0.6254,  # Within tolerance
                "confusion_matrix": {"tp": 3, "fp": 1, "fn": 2, "tn": 2},
            }
        }

        # Answer key
        answer_key = {
            "metrics": {
                "precision": 0.7500,
                "recall": 0.6000,
                "f1": 0.6667,
                "accuracy": 0.6250,
                "confusion_matrix": {"tp": 3, "fp": 1, "fn": 2, "tn": 2},
            }
        }

        # Test case configuration with tight tolerance
        test_case = {
            "id": "test.metrics",
            "scoring": {
                "evaluator": "metrics_csv",
                "config": {
                    "tolerance": 0.0005,
                    "weights": {
                        "precision": 6.0,
                        "recall": 6.0,
                        "f1": 6.0,
                        "accuracy": 6.0,
                        "confusion_matrix": 12.0,
                    },
                },
            },
        }

        # Create evaluator
        evaluator = MetricsCSVEvaluator(test_case["scoring"]["config"])

        # Run evaluation
        result = evaluator.evaluate(response_data, test_case, answer_key)

        # Should get full score since all values are within tolerance
        assert result.total_score == 36.0
        assert result.max_score == 36.0
        assert result.score_percentage == 100.0

    def test_answer_key_loading(self):
        """Test that answer keys are loaded correctly."""
        # Create temporary answer key file
        answer_key_data = {
            "metrics": {
                "precision": 0.7500,
                "recall": 0.6000,
                "f1": 0.6667,
                "accuracy": 0.6250,
                "confusion_matrix": {"tp": 3, "fp": 1, "fn": 2, "tn": 2},
            }
        }

        answer_key_file = self.temp_dir / "test_answer_key.json"
        with open(answer_key_file, "w") as f:
            json.dump(answer_key_data, f)

        # Create test runner
        runner = TestRunner(self.config_dir)

        # Create test definition
        # test_def = {"id": "test.metrics", "answer_key": str(answer_key_file)}

        # Load answer key using the test_id
        loaded_key = runner._load_answer_key("test.metrics")

        # The method looks for answer keys in answer_keys/{category}/{test_id}.json
        # Since our test file is in a temp directory, it won't be found
        # Let's test the method with a proper path structure

        # Create proper answer key directory structure
        answer_keys_dir = self.temp_dir / "answer_keys" / "test"
        answer_keys_dir.mkdir(parents=True, exist_ok=True)
        proper_answer_key_file = answer_keys_dir / "test.metrics.json"
        with open(proper_answer_key_file, "w") as f:
            json.dump(answer_key_data, f)

        # Change to temp directory to test relative path loading
        import os

        original_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            loaded_key = runner._load_answer_key("test.metrics")

            # Verify answer key loaded correctly
            assert loaded_key is not None
            assert loaded_key == answer_key_data
        finally:
            os.chdir(original_cwd)

    def test_missing_answer_key(self):
        """Test handling of missing answer key."""
        # Create test runner
        runner = TestRunner(self.config_dir)

        # Create test definition with non-existent answer key
        test_def = {"id": "test.metrics", "answer_key": "nonexistent/path.json"}

        # Try to load answer key
        loaded_key = runner._load_answer_key(test_def)

        # Should return None for missing file
        assert loaded_key is None

    def test_no_answer_key_specified(self):
        """Test handling when no answer key is specified."""
        # Create test runner
        runner = TestRunner(self.config_dir)

        # Create test definition without answer key
        test_def = {
            "id": "test.metrics"
            # No answer_key field
        }

        # Try to load answer key
        loaded_key = runner._load_answer_key(test_def)

        # Should return None when no answer key specified
        assert loaded_key is None


def test_evaluation_integration():
    """Integration test for full evaluation pipeline."""
    # This would test the complete flow from response to final score
    # but requires more setup of the full test infrastructure
    pass


if __name__ == "__main__":
    # Run tests directly
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))

    test_instance = TestEvaluationSystem()
    test_instance.setup_method()

    # Running evaluation system tests...

    try:
        test_instance.test_metrics_evaluator_perfect_score()
        # ✓ Perfect score test passed

        test_instance.test_metrics_evaluator_partial_score()
        # ✓ Partial score test passed

        test_instance.test_metrics_evaluator_zero_score()
        # ✓ Zero score test passed

        test_instance.test_metrics_evaluator_missing_data()
        # ✓ Missing data test passed

        test_instance.test_tolerance_handling()
        # ✓ Tolerance handling test passed

        test_instance.test_answer_key_loading()
        # ✓ Answer key loading test passed

        test_instance.test_missing_answer_key()
        # ✓ Missing answer key test passed

        test_instance.test_no_answer_key_specified()
        # ✓ No answer key specified test passed

        # 🎉 All evaluation system tests passed!

    except Exception as e:
        # Test failed
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
