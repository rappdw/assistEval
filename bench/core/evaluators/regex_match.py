"""Task 2 evaluator for SSN regex pattern matching.

This module evaluates regex patterns for U.S. Social Security Number validation
with safety guards and line matching verification.
"""

from typing import Any

from bench.core.evaluators.base import Evaluator


class RegexMatchEvaluator(Evaluator):
    """Evaluates regex patterns for SSN validation.

    Compiles and tests regex patterns with timeout protection,
    validates against test lines, and checks format compliance.
    """

    def evaluate(
        self, response: dict[str, Any], expected: dict[str, Any]
    ) -> dict[str, Any]:
        """Evaluate regex pattern against expected line matches.

        Args:
            response: Parsed JSON response with regex pattern
            expected: Expected line matches from answer key

        Returns:
            Dictionary with score breakdown and validation results

        Raises:
            NotImplementedError: Implementation pending in Stage 7
        """
        # TODO: Implement regex evaluation in Stage 7
        # - Compile regex with re.fullmatch per token
        # - Use timeout guard (100ms per line) for safety
        # - Evaluate against SSN validation lines
        # - Check format validity (anchors, constraints)
        # - Validate no 000/666/9xx area codes, no 00 group, no 0000 serial
        # - Award points for correct matches and validity rules
        raise NotImplementedError("Implementation pending in Stage 7")
