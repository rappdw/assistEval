"""Task 3 evaluator for executive summary structure and tone.

This module evaluates executive summaries for structural constraints
and tone heuristics including word counts and bullet formatting.
"""

import re
from typing import Any

from bench.core.evaluators.base import BaseEvaluator, EvaluationResult


class ExecSummaryEvaluator(BaseEvaluator):
    """Evaluates executive summaries for structure and tone compliance.

    Validates title length, word count, bullet formatting, and applies
    tone heuristics with configurable denylist and sentence analysis.
    """

    # Default scoring weights (20 points total)
    DEFAULT_WEIGHTS = {
        "structure": 12.0,  # Title, word count, bullets, JSON compliance
        "tone": 8.0,  # Tone heuristics and clarity
    }

    # Default tone heuristics configuration
    DEFAULT_HYPE_TERMS = [
        "revolutionary",
        "groundbreaking",
        "game-changing",
        "unprecedented",
        "amazing",
        "incredible",
        "fantastic",
        "awesome",
        "outstanding",
        "phenomenal",
        "extraordinary",
        "remarkable",
        "stunning",
        "brilliant",
        "world-class",
        "cutting-edge",
        "state-of-the-art",
        "next-generation",
        "paradigm-shifting",
        "disruptive",
        "transformative",
        "innovative",
    ]

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize executive summary evaluator.

        Args:
            config: Configuration with weights and tone settings
        """
        super().__init__(config)
        self.weights = config.get("weights", self.DEFAULT_WEIGHTS)
        self.hype_terms = config.get("hype_terms", self.DEFAULT_HYPE_TERMS)
        self.max_sentence_length = config.get("max_sentence_length", 24)
        self.word_count_range = config.get("word_count_range", (120, 160))
        self.max_title_words = config.get("max_title_words", 6)
        self.required_bullets = config.get("required_bullets", 3)

    def evaluate(
        self,
        response_data: dict[str, Any],
        test_case: dict[str, Any],
        answer_key: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate executive summary structure and tone.

        Args:
            response_data: Parsed JSON response with executive summary
            test_case: Test case definition
            answer_key: Optional expected requirements

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

        # Extract summary data from response
        summary_data = response_data.get("summary", response_data)
        if not isinstance(summary_data, dict):
            summary_data = response_data  # Fallback to root level

        # Evaluate structural requirements
        structure_score = self._evaluate_structure(summary_data, result)
        result.sub_scores["structure"] = structure_score

        # Evaluate tone and clarity
        tone_score = self._evaluate_tone(summary_data, result)
        result.sub_scores["tone"] = tone_score

        # Calculate total weighted score
        result.total_score = self.calculate_weighted_score(
            result.sub_scores, self.weights
        )

        # Add metadata
        result.metadata.update(
            {
                "weights": self.weights,
                "word_count_range": self.word_count_range,
                "max_title_words": self.max_title_words,
                "required_bullets": self.required_bullets,
                "max_sentence_length": self.max_sentence_length,
            }
        )

        return result

    def _evaluate_structure(
        self, summary_data: dict[str, Any], result: EvaluationResult
    ) -> float:
        """Evaluate structural requirements.

        Args:
            summary_data: Summary data from response
            result: Result object to add details to

        Returns:
            Structure score (0.0 to 1.0)
        """
        structure_checks = {
            "title_length": False,
            "word_count": False,
            "bullet_count": False,
            "json_compliance": False,
        }

        # Check title length (≤6 words)
        title = summary_data.get("title", "")
        if title:
            title_words = len(title.split())
            if title_words <= self.max_title_words:
                structure_checks["title_length"] = True
                result.details["title_status"] = "pass"
            else:
                result.add_warning(
                    f"Title too long: {title_words} words (max {self.max_title_words})"
                )
                result.details["title_status"] = "fail"
            result.details["title_word_count"] = title_words
        else:
            result.add_error("Missing title in summary")
            result.details["title_status"] = "missing"

        # Check summary word count (120-160 words, excluding bullets)
        summary_text = summary_data.get("summary", "")
        if summary_text:
            word_count = len(summary_text.split())
            min_words, max_words = self.word_count_range
            if min_words <= word_count <= max_words:
                structure_checks["word_count"] = True
                result.details["word_count_status"] = "pass"
            else:
                result.add_warning(
                    f"Summary has {word_count} words, expected 120-160 words"
                )
                result.details["word_count_status"] = "fail"
            result.details["summary_word_count"] = word_count
        else:
            result.add_error("Missing summary text")
            result.details["word_count_status"] = "missing"

        # Check bullet points (exactly 3)
        bullets = summary_data.get("bullets", [])
        if isinstance(bullets, list):
            bullet_count = len(bullets)
            if bullet_count == self.required_bullets:
                structure_checks["bullet_count"] = True
                result.details["bullet_count_status"] = "pass"
            else:
                result.add_warning(
                    f"Found {bullet_count} bullets, "
                    f"expected exactly {self.required_bullets}"
                )
                result.details["bullet_count_status"] = "fail"
            result.details["bullet_count"] = bullet_count
        else:
            result.add_error("Missing or invalid bullets array")
            result.details["bullet_count_status"] = "missing"

        # Check JSON schema compliance (has required fields)
        required_fields = ["title", "summary", "bullets"]
        missing_fields = [
            field for field in required_fields if not summary_data.get(field)
        ]
        if not missing_fields:
            structure_checks["json_compliance"] = True
            result.details["json_compliance_status"] = "pass"
        else:
            result.add_error(f"Missing required fields: {missing_fields}")
            result.details["json_compliance_status"] = "fail"
            result.details["missing_fields"] = missing_fields

        # Calculate structure score (equal weight for each component)
        score = sum(structure_checks.values()) / len(structure_checks)
        result.details["structure_checks"] = structure_checks
        result.details["structure_score"] = score

        return score

    def _evaluate_tone(
        self, summary_data: dict[str, Any], result: EvaluationResult
    ) -> float:
        """Evaluate tone and clarity heuristics.

        Args:
            summary_data: Summary data from response
            result: Result object to add details to

        Returns:
            Tone score (0.0 to 1.0)
        """
        tone_checks = {
            "no_hype_terms": False,
            "sentence_length": False,
            "professional_tone": False,
        }

        # Combine all text for analysis
        all_text = ""
        title = summary_data.get("title", "")
        summary_text = summary_data.get("summary", "")
        bullets = summary_data.get("bullets", [])

        if title:
            all_text += title + " "
        if summary_text:
            all_text += summary_text + " "
        if isinstance(bullets, list):
            all_text += " ".join(str(bullet) for bullet in bullets)

        if not all_text.strip():
            result.add_error("No text found for tone evaluation")
            result.details["tone_checks"] = tone_checks
            result.details["tone_score"] = 0.0
            return 0.0

        # Check for hype terms
        hype_found = []
        text_lower = all_text.lower()
        for term in self.hype_terms:
            if term.lower() in text_lower:
                hype_found.append(term)

        if not hype_found:
            tone_checks["no_hype_terms"] = True
            result.details["hype_terms_status"] = "pass"
        else:
            result.add_warning(f"Hype terms found: {hype_found}")
            result.details["hype_terms_status"] = "fail"
            result.details["hype_terms_found"] = hype_found

        # Check average sentence length
        sentences = self._split_sentences(all_text)
        if sentences:
            sentence_lengths = [len(sentence.split()) for sentence in sentences]
            avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)

            if avg_sentence_length <= self.max_sentence_length:
                tone_checks["sentence_length"] = True
                result.details["sentence_length_status"] = "pass"
            else:
                result.add_warning(
                    f"Average sentence length is {avg_sentence_length:.1f} words "
                    f"(>{self.max_sentence_length})"
                )
                result.details["sentence_length_status"] = "fail"

            result.details["avg_sentence_length"] = avg_sentence_length
            result.details["sentence_count"] = len(sentences)
        else:
            result.details["sentence_length_status"] = "no_sentences"

        # Basic professional tone check (simple heuristics)
        professional_indicators = [
            not bool(re.search(r"[!]{2,}", all_text)),  # No multiple exclamations
            not bool(re.search(r"[A-Z]{3,}", all_text)),  # No excessive caps
            len(re.findall(r"[.!?]", all_text)) > 0,  # Has proper punctuation
        ]

        if all(professional_indicators):
            tone_checks["professional_tone"] = True
            result.details["professional_tone_status"] = "pass"
        else:
            result.add_warning("Text may lack professional tone")
            result.details["professional_tone_status"] = "fail"

        # Calculate tone score (weighted by importance)
        weights = {
            "no_hype_terms": 0.4,
            "sentence_length": 0.4,
            "professional_tone": 0.2,
        }

        score = sum(weights[check] for check, passed in tone_checks.items() if passed)
        result.details["tone_checks"] = tone_checks
        result.details["tone_score"] = score

        return score

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Simple sentence splitting on common punctuation
        sentences = re.split(r"[.!?]+", text)
        # Filter out empty sentences and strip whitespace
        return [s.strip() for s in sentences if s.strip()]
