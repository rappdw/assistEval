"""Task 2 evaluator for SSN regex pattern matching.

This module evaluates regex patterns for U.S. Social Security Number validation
with safety guards and line matching verification.
"""

import re
import signal
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from bench.core.evaluators.base import BaseEvaluator, EvaluationResult


class RegexTimeoutError(Exception):
    """Custom timeout exception for regex evaluation."""

    pass


@contextmanager
def timeout_guard(seconds: float) -> Iterator[None]:
    """Context manager for timeout protection.

    Args:
        seconds: Timeout duration in seconds

    Raises:
        RegexTimeoutError: If operation exceeds timeout
    """

    def timeout_handler(signum: int, frame: Any) -> None:
        raise RegexTimeoutError(f"Operation timed out after {seconds} seconds")

    # Set up signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(seconds * 1000) // 1000)  # Convert to whole seconds

    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class RegexMatchEvaluator(BaseEvaluator):
    """Evaluates regex patterns for SSN validation.

    Compiles and tests regex patterns with timeout protection,
    validates against test lines, and checks format compliance.
    """

    # Default scoring weights (30 points total)
    DEFAULT_WEIGHTS = {
        "regex_validity": 18.0,  # Format constraints and anchoring
        "line_matches": 12.0,  # 1 point per correct line (12 lines)
    }

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize regex evaluator.

        Args:
            config: Configuration with timeout and weights
        """
        super().__init__(config)
        self.timeout_ms = config.get("timeout_ms", 100)
        self.weights = config.get("weights", self.DEFAULT_WEIGHTS)

    def evaluate(
        self,
        response_data: dict[str, Any],
        test_case: dict[str, Any],
        answer_key: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate regex pattern against expected line matches.

        Args:
            response_data: Parsed JSON response with regex pattern
            test_case: Test case definition with test lines
            answer_key: Expected line matches from answer key

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

        # Extract regex pattern from response
        pattern = self.safe_extract_field(response_data, "regex", "")
        if not pattern:
            pattern = self.safe_extract_field(response_data, "pattern", "")

        if not pattern:
            result.add_error("No regex pattern found in response")
            result.sub_scores["regex_validity"] = 0.0
            result.sub_scores["line_matches"] = 0.0
            return result

        # Evaluate regex validity constraints
        validity_score = self._evaluate_regex_validity(pattern, result)
        result.sub_scores["regex_validity"] = validity_score

        # Evaluate line matches if we have test data
        test_lines = test_case.get("test_lines", [])
        expected_matches = answer_key.get("expected_matches", []) if answer_key else []

        if test_lines and expected_matches:
            match_score = self._evaluate_line_matches(
                pattern, test_lines, expected_matches, result
            )
            result.sub_scores["line_matches"] = match_score
        else:
            result.add_warning("No test lines or expected matches provided")
            result.sub_scores["line_matches"] = 0.0

        # Calculate total weighted score
        result.total_score = self.calculate_weighted_score(
            result.sub_scores, self.weights
        )

        # Add metadata
        result.metadata.update(
            {
                "pattern": pattern,
                "timeout_ms": self.timeout_ms,
                "weights": self.weights,
            }
        )

        return result

    def _evaluate_regex_validity(self, pattern: str, result: EvaluationResult) -> float:
        """Evaluate regex validity constraints for SSN patterns.

        Args:
            pattern: Regex pattern to evaluate
            result: Result object to add details to

        Returns:
            Validity score (0.0 to 1.0)
        """
        validity_checks = {
            "compiles": False,
            "has_anchors": False,
            "rejects_000_area": False,
            "rejects_666_area": False,
            "rejects_9xx_area": False,
            "rejects_00_group": False,
            "rejects_0000_serial": False,
        }

        # Test if pattern compiles
        try:
            compiled_pattern = re.compile(pattern)
            validity_checks["compiles"] = True
            result.details["pattern_compiles"] = True
        except re.error as e:
            result.add_error(f"Regex compilation failed: {e}")
            result.details["pattern_compiles"] = False
            result.details["compilation_error"] = str(e)
            return 0.0

        # Check for proper anchoring (^ and $)
        if pattern.startswith("^") and pattern.endswith("$"):
            validity_checks["has_anchors"] = True
            result.details["has_anchors"] = True
        else:
            result.add_warning("Pattern should be anchored with ^ and $")
            result.details["has_anchors"] = False

        # Test constraint violations with timeout protection
        constraint_tests = [
            ("000-12-3456", "rejects_000_area"),
            ("666-12-3456", "rejects_666_area"),
            ("900-12-3456", "rejects_9xx_area"),
            ("123-00-3456", "rejects_00_group"),
            ("123-45-0000", "rejects_0000_serial"),
        ]

        for test_ssn, check_name in constraint_tests:
            try:
                with timeout_guard(self.timeout_ms / 1000.0):
                    match = compiled_pattern.fullmatch(test_ssn)
                    # Should NOT match (reject invalid SSNs)
                    if not match:
                        validity_checks[check_name] = True
                        result.details[check_name] = True
                    else:
                        result.add_warning(f"Pattern incorrectly accepts {test_ssn}")
                        result.details[check_name] = False
            except RegexTimeoutError:
                result.add_error(
                    f"Regex compilation timed out after {self.timeout_ms}ms - "
                    "possible catastrophic backtracking"
                )
                result.details[check_name] = False
            except Exception as e:
                result.add_error(f"Error testing {test_ssn}: {e}")
                result.details[check_name] = False

        # Calculate validity score (weighted by importance)
        weights = {
            "compiles": 0.3,
            "has_anchors": 0.2,
            "rejects_000_area": 0.1,
            "rejects_666_area": 0.1,
            "rejects_9xx_area": 0.1,
            "rejects_00_group": 0.1,
            "rejects_0000_serial": 0.1,
        }

        score = sum(
            weights[check] for check, passed in validity_checks.items() if passed
        )
        result.details["validity_checks"] = validity_checks
        result.details["validity_score"] = score

        return score

    def _evaluate_line_matches(
        self,
        pattern: str,
        test_lines: list[str],
        expected_matches: list[bool],
        result: EvaluationResult,
    ) -> float:
        """Evaluate pattern against test lines with timeout protection.

        Args:
            pattern: Compiled regex pattern
            test_lines: Lines to test against
            expected_matches: Expected match results (True/False for each line)
            result: Result object to add details to

        Returns:
            Line match score (0.0 to 1.0)
        """
        try:
            compiled_pattern = re.compile(pattern)
        except re.error:
            result.add_error("Cannot test lines - pattern compilation failed")
            return 0.0

        if len(test_lines) != len(expected_matches):
            result.add_error(
                f"Mismatch: {len(test_lines)} test lines vs "
                f"{len(expected_matches)} expected results"
            )
            return 0.0

        correct_matches = 0
        total_lines = len(test_lines)
        line_results = []

        for i, (line, expected) in enumerate(
            zip(test_lines, expected_matches, strict=False)
        ):
            try:
                with timeout_guard(self.timeout_ms / 1000.0):
                    actual_match = bool(compiled_pattern.fullmatch(line.strip()))
                    is_correct = actual_match == expected

                    if is_correct:
                        correct_matches += 1

                    line_results.append(
                        {
                            "line": line,
                            "expected": expected,
                            "actual": actual_match,
                            "correct": is_correct,
                        }
                    )

            except RegexTimeoutError:
                result.add_error(
                    f"Line matching timed out after {self.timeout_ms}ms - "
                    "possible catastrophic backtracking"
                )
                line_results.append(
                    {
                        "line": line,
                        "expected": expected,
                        "actual": None,
                        "correct": False,
                        "error": "timeout",
                    }
                )
            except Exception as e:
                result.add_error(f"Error testing line {i + 1}: {e}")
                line_results.append(
                    {
                        "line": line,
                        "expected": expected,
                        "actual": None,
                        "correct": False,
                        "error": str(e),
                    }
                )

        # Calculate score
        score = correct_matches / total_lines if total_lines > 0 else 0.0

        result.details["line_matches"] = {
            "correct": correct_matches,
            "total": total_lines,
            "score": score,
            "results": line_results,
        }

        return score
