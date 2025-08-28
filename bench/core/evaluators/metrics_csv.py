"""Task 1 evaluator for metrics from CSV data.

This module evaluates precision, recall, F1, accuracy, and confusion matrix
metrics calculated from phishing detection CSV data.
"""

from typing import Any

from bench.core.evaluators.base import BaseEvaluator, EvaluationResult


class MetricsCSVEvaluator(BaseEvaluator):
    """Evaluates metrics calculated from CSV classification data.

    Validates precision, recall, F1, accuracy with tolerance checking
    and confusion matrix values for exact integer matches.
    """

    # Default scoring weights (40 points total)
    DEFAULT_WEIGHTS = {
        "precision": 6.0,
        "recall": 6.0,
        "f1": 6.0,
        "accuracy": 6.0,
        "confusion_matrix": 12.0,  # 3 points each for TP/FP/FN/TN
    }

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize metrics evaluator.

        Args:
            config: Configuration with tolerance and weights
        """
        super().__init__(config)
        self.tolerance = config.get("tolerance", 0.0005)
        self.weights = config.get("weights", self.DEFAULT_WEIGHTS)

    def _is_numeric(self, value: Any) -> bool:
        """Check if value can be converted to float."""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    def evaluate(
        self,
        response_data: dict[str, Any],
        test_case: dict[str, Any],
        answer_key: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate metrics against expected values.

        Args:
            response_data: Parsed JSON response with calculated metrics
            test_case: Test case definition
            answer_key: Expected metric values from answer key

        Returns:
            EvaluationResult with detailed scoring breakdown
        """
        task_id = test_case.get("id", "unknown")
        result = EvaluationResult(
            task_id=task_id,
            total_score=0.0,
            max_score=sum(self.weights.values()),
            sub_scores={},
        )

        if not answer_key:
            result.add_error("No answer key provided for metrics evaluation")
            return result

        # Extract metrics from response
        metrics = self.safe_extract_field(response_data, "metrics", {})
        expected_metrics = answer_key.get("metrics", {})

        # Evaluate numeric metrics with tolerance
        numeric_fields = ["precision", "recall", "f1", "accuracy"]
        for field in numeric_fields:
            score = self._evaluate_numeric_field(
                metrics, expected_metrics, field, result
            )
            result.sub_scores[field] = score

        # Evaluate confusion matrix with exact matching
        cm_score = self._evaluate_confusion_matrix(metrics, expected_metrics, result)
        result.sub_scores["confusion_matrix"] = cm_score

        # Calculate total weighted score
        result.total_score = self.calculate_weighted_score(
            result.sub_scores, self.weights
        )

        # Add metadata
        result.metadata.update(
            {
                "tolerance": self.tolerance,
                "weights": self.weights,
                "numeric_fields_evaluated": numeric_fields,
            }
        )

        return result

    def _evaluate_numeric_field(
        self,
        metrics: dict[str, Any],
        expected: dict[str, Any],
        field: str,
        result: EvaluationResult,
    ) -> float:
        """Evaluate a single numeric field with tolerance.

        Args:
            metrics: Actual metrics from response
            expected: Expected metrics from answer key
            field: Field name to evaluate
            result: Result object to add details to

        Returns:
            Score for this field (0.0 or weight)
        """
        actual = metrics.get(field)
        expected_val = expected.get(field)

        if actual is None:
            result.add_error(f"Missing {field} in response")
            result.details[f"{field}_status"] = "missing"
            return 0.0

        if expected_val is None:
            result.add_error(f"Missing {field} in answer key")
            result.details[f"{field}_status"] = "no_expected"
            return 0.0

        if not self._is_numeric(actual) or not self._is_numeric(expected_val):
            result.add_error(f"Invalid numeric value for {field}")
            result.details[f"{field}_status"] = "invalid"
            return 0.0

        # Check tolerance
        actual_float = float(actual)
        expected_float = float(expected_val)
        if self._within_tolerance(actual_float, expected_float):
            result.details[f"{field}_status"] = "pass"
            result.details[f"{field}_actual"] = actual_float
            result.details[f"{field}_expected"] = expected_float
            result.details[f"{field}_diff"] = abs(actual_float - expected_float)
            return 1.0  # Will be multiplied by weight
        else:
            result.add_warning(
                f"Expected {expected_float}, got {actual_float} "
                f"(diff: {abs(actual_float - expected_float):.6f})"
            )
            result.details[f"{field}_status"] = "fail"
            result.details[f"{field}_actual"] = actual_float
            result.details[f"{field}_expected"] = expected_float
            result.details[f"{field}_diff"] = abs(actual_float - expected_float)
            return 0.0

    def _evaluate_confusion_matrix(
        self,
        metrics: dict[str, Any],
        expected: dict[str, Any],
        result: EvaluationResult,
    ) -> float:
        """Evaluate confusion matrix with exact integer matching.

        Args:
            metrics: Actual metrics from response
            expected: Expected metrics from answer key
            result: Result object to add details to

        Returns:
            Score for confusion matrix (0.0 to 1.0)
        """
        actual_cm = metrics.get("confusion_matrix", {})
        expected_cm = expected.get("confusion_matrix", {})

        if not actual_cm:
            result.add_error("Missing confusion_matrix in response")
            result.details["confusion_matrix_status"] = "missing"
            return 0.0

        if not expected_cm:
            result.add_error("Missing confusion_matrix in answer key")
            result.details["confusion_matrix_status"] = "no_expected"
            return 0.0

        # Check each component (TP, FP, FN, TN)
        cm_components = ["tp", "fp", "fn", "tn"]
        correct_components = 0
        total_components = len(cm_components)

        for component in cm_components:
            actual_val = actual_cm.get(component)
            expected_val = expected_cm.get(component)

            if actual_val is None:
                result.add_error(f"Missing {component} in confusion matrix")
                result.details[f"cm_{component}_status"] = "missing"
                continue

            if expected_val is None:
                result.add_error(f"Missing {component} in expected confusion matrix")
                result.details[f"cm_{component}_status"] = "no_expected"
                continue

            try:
                actual_int = int(actual_val)
                expected_int = int(expected_val)
            except (ValueError, TypeError):
                result.add_error(f"Invalid integer value for {component}")
                result.details[f"cm_{component}_status"] = "invalid"
                continue

            # Exact match required for confusion matrix
            if actual_int == expected_int:
                correct_components += 1
                result.details[f"cm_{component}_status"] = "pass"
            else:
                result.add_warning(
                    f"Confusion matrix {component} mismatch: "
                    f"{actual_int} vs {expected_int}"
                )
                result.details[f"cm_{component}_status"] = "fail"

            result.details[f"cm_{component}_actual"] = actual_int
            result.details[f"cm_{component}_expected"] = expected_int

        # Calculate proportional score
        score = correct_components / total_components if total_components > 0 else 0.0
        result.details[
            "confusion_matrix_score"
        ] = f"{correct_components}/{total_components}"
        result.details["confusion_matrix_status"] = "evaluated"

        return score

    def _within_tolerance(self, actual: float, expected: float) -> bool:
        """Check if actual value is within tolerance of expected.

        Args:
            actual: Actual value
            expected: Expected value

        Returns:
            True if within tolerance
        """
        diff: float = abs(actual - expected)
        result: bool = diff <= self.tolerance
        return result
