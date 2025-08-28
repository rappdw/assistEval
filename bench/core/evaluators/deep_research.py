"""Deep research task evaluator for plan structure and source quality validation.

This module evaluates deep research plans with step structure validation,
risk register analysis, and source recency verification.
"""

import re
from datetime import datetime
from typing import Any

from bench.core.evaluators.base import BaseEvaluator, EvaluationResult


class DeepResearchEvaluator(BaseEvaluator):
    """Evaluates deep research plans for structure and source quality.

    Validates research plan structure (7-10 steps), risk register completeness,
    and source recency with web capability enforcement.
    """

    def evaluate(
        self,
        response_data: dict[str, Any],
        test_case: dict[str, Any],
        answer_key: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate deep research plan structure and sources.

        Args:
            response_data: Parsed JSON response with research plan
            test_case: Test case configuration
            answer_key: Expected structural requirements

        Returns:
            EvaluationResult with score breakdown and validation results
        """
        task_id = test_case.get("id", "deep_research")
        scoring_config = test_case.get("scoring", {}).get("config", {})

        result = EvaluationResult(
            task_id=task_id,
            total_score=0.0,
            max_score=10.0,
            sub_scores={},
            details={},
            metadata={"evaluator_name": "deep_research"},
        )

        try:
            # Evaluate plan quality (5 points)
            plan_score, plan_details = self._evaluate_plan_structure(
                response_data, scoring_config
            )
            result.sub_scores["plan_quality"] = plan_score
            result.details.update(plan_details)

            # Evaluate source quality (5 points)
            source_score, source_details = self._evaluate_source_quality(
                response_data, scoring_config
            )
            result.sub_scores["source_quality"] = source_score
            result.details.update(source_details)

            # Calculate total score
            result.total_score = plan_score + source_score

            # Add web capability validation
            self._validate_web_capability(response_data, test_case, result)

        except Exception as e:
            result.add_error(f"Evaluation failed: {str(e)}")
            result.total_score = 0.0

        return result

    def _evaluate_plan_structure(
        self, response_data: dict[str, Any], scoring_config: dict[str, Any]
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate research plan structure (5 points total)."""
        plan_requirements = scoring_config.get("plan_requirements", {})
        min_steps = plan_requirements.get("min_steps", 7)
        max_steps = plan_requirements.get("max_steps", 10)
        required_fields = plan_requirements.get(
            "required_fields", ["goal", "method", "deliverable"]
        )

        details: dict[str, Any] = {}
        score = 0.0

        # Get research plan from response
        research_plan = response_data.get("research_plan", [])
        if not isinstance(research_plan, list):
            details["plan_error"] = "Research plan must be an array"
            return 0.0, details

        plan_count = len(research_plan)
        details["plan_step_count"] = plan_count

        # Structure validation (2 points): 7-10 steps with required fields
        structure_score = 0.0

        # Check step count
        if min_steps <= plan_count <= max_steps:
            structure_score += 1.0
            details["step_count_valid"] = True
        else:
            details["step_count_valid"] = False
            details["step_count_issue"] = (
                f"Expected {min_steps}-{max_steps} steps, got {plan_count}"
            )

        # Check required fields in each step
        valid_steps = 0
        field_issues = []

        for i, step in enumerate(research_plan):
            if not isinstance(step, dict):
                field_issues.append(f"Step {i + 1} is not an object")
                continue

            missing_fields = [field for field in required_fields if field not in step]
            if not missing_fields:
                valid_steps += 1
            else:
                field_issues.append(
                    f"Step {i + 1} missing: {', '.join(missing_fields)}"
                )

        if plan_count > 0:
            field_completeness = valid_steps / plan_count
            structure_score += (
                field_completeness  # Up to 1 point for field completeness
            )
            details["valid_steps"] = valid_steps
            details["field_completeness"] = field_completeness
            if field_issues:
                details["field_issues"] = field_issues

        # Sequencing validation (2 points): Logical flow and dependencies
        sequencing_score = self._evaluate_sequencing(research_plan)
        details["sequencing_score"] = sequencing_score

        # Completeness validation (1 point): All deliverables specified
        completeness_score = self._evaluate_completeness(research_plan)
        details["completeness_score"] = completeness_score

        score = structure_score + sequencing_score + completeness_score
        details["plan_structure_score"] = structure_score

        return min(score, 5.0), details

    def _evaluate_sequencing(self, research_plan: list[dict[str, Any]]) -> float:
        """Evaluate logical sequencing of research steps (2 points)."""
        if len(research_plan) == 0:
            return 0.0  # No steps, no score
        elif len(research_plan) == 1:
            return 1.0  # Single step gets partial credit

        # Look for logical progression indicators
        sequence_indicators = 0.0
        total_checks = len(research_plan) - 1

        for i in range(len(research_plan) - 1):
            current_step = research_plan[i]
            next_step = research_plan[i + 1]

            if isinstance(current_step, dict) and isinstance(next_step, dict):
                # Check for logical progression in goals/methods
                current_goal = str(current_step.get("goal", "")).lower()
                next_goal = str(next_step.get("goal", "")).lower()
                next_method = str(next_step.get("method", "")).lower()

                # Look for dependency keywords
                dependency_words = [
                    "based on",
                    "using",
                    "following",
                    "after",
                    "then",
                    "next",
                    "from",
                ]
                if any(
                    word in next_goal or word in next_method
                    for word in dependency_words
                ):
                    sequence_indicators += 1
                # Look for progressive complexity
                elif (
                    ("analyze" in current_goal and "synthesize" in next_goal)
                    or ("collect" in current_goal and "analyze" in next_goal)
                    or ("research" in current_goal and "evaluate" in next_goal)
                    or ("define" in current_goal and "analyze" in next_goal)
                    or ("survey" in current_goal and "evaluate" in next_goal)
                ):
                    sequence_indicators += 1
                # Look for typical research flow patterns
                elif i < len(research_plan) - 2:  # Not the last comparison
                    # Give partial credit for reasonable research progression
                    research_words = [
                        "research",
                        "analyze",
                        "evaluate",
                        "assess",
                        "synthesize",
                        "review",
                    ]
                    if any(word in current_goal for word in research_words) and any(
                        word in next_goal for word in research_words
                    ):
                        sequence_indicators += 0.5

        if total_checks == 0:
            return 1.0

        sequencing_ratio = sequence_indicators / total_checks
        return min(2.0 * sequencing_ratio, 2.0)

    def _evaluate_completeness(self, research_plan: list[dict[str, Any]]) -> float:
        """Evaluate completeness of deliverables (1 point)."""
        if not research_plan:
            return 0.0

        deliverable_count = 0
        for step in research_plan:
            if isinstance(step, dict) and step.get("deliverable"):
                deliverable_text = str(step["deliverable"]).strip()
                if (
                    deliverable_text and len(deliverable_text) > 10
                ):  # Meaningful deliverable
                    deliverable_count += 1

        if deliverable_count == 0:
            return 0.0
        elif deliverable_count == len(research_plan):
            return 1.0
        else:
            return deliverable_count / len(research_plan)

    def _evaluate_source_quality(
        self, response_data: dict[str, Any], config: dict[str, Any]
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate source quality (5 points total)."""
        source_requirements = config.get("source_requirements", {})
        min_sources = source_requirements.get("min_sources", 5)
        max_sources = source_requirements.get("max_sources", 8)
        min_recent_sources = source_requirements.get("min_recent_sources", 3)
        recency_years = source_requirements.get("recency_years", 3)

        details: dict[str, Any] = {}
        score = 0.0

        # Get sources from response
        sources = response_data.get("sources", [])
        if not isinstance(sources, list):
            details["source_error"] = "Sources must be an array"
            return 0.0, details

        source_count = len(sources)
        details["source_count"] = source_count

        # Count validation (2 points): 5-8 sources present
        count_score = 0.0
        if min_sources <= source_count <= max_sources:
            count_score = 2.0
            details["source_count_valid"] = True
        elif source_count > 0:
            # Partial credit for having some sources
            if source_count < min_sources:
                count_score = 2.0 * (source_count / min_sources)
            else:  # source_count > max_sources
                count_score = 2.0 * (max_sources / source_count)
            details["source_count_valid"] = False
        else:
            details["source_count_valid"] = False

        # Recency validation (2 points): ≥3 sources within 3 years
        recency_score = 0.0
        recent_sources = 0
        current_year = datetime.now().year
        cutoff_year = current_year - recency_years

        year_issues = []
        for i, source in enumerate(sources):
            if not isinstance(source, dict) and not isinstance(source, str):
                continue

            source_text = (
                str(source)
                if isinstance(source, str)
                else str(
                    source.get("citation", source.get("title", source.get("url", "")))
                )
            )

            # Extract year from source text using regex
            year_matches = re.findall(r"\b(19\d{2}|20\d{2})\b", source_text)
            if year_matches:
                try:
                    year = int(year_matches[-1])  # Take the last year found
                    if year >= cutoff_year:
                        recent_sources += 1
                    details[f"source_{i + 1}_year"] = year
                except ValueError:
                    yr_info = year_matches[-1]
                    year_issues.append(
                        f"Source {i + 1}: Could not parse year from '{yr_info}'"
                    )
            else:
                year_issues.append(
                    f"Source {i + 1}: No year found in '{source_text[:100]}...'"
                )

        details["recent_sources"] = recent_sources
        details["required_recent_sources"] = min_recent_sources
        if year_issues:
            details["year_parsing_issues"] = year_issues

        if recent_sources >= min_recent_sources:
            recency_score = 2.0
        elif recent_sources > 0:
            recency_score = 2.0 * (recent_sources / min_recent_sources)

        # Quality validation (1 point): Credible sources and proper citations
        quality_score = self._evaluate_source_credibility(sources)
        details["source_quality_score"] = quality_score

        score = count_score + recency_score + quality_score
        details["source_count_score"] = count_score
        details["source_recency_score"] = recency_score

        return min(score, 5.0), details

    def _evaluate_source_credibility(self, sources: list[Any]) -> float:
        """Evaluate source credibility and citation quality (1 point)."""
        if not sources:
            return 0.0

        credible_indicators = 0
        total_sources = len(sources)

        # Credible source indicators
        credible_domains = [
            ".edu",
            ".gov",
            ".org",
            "ieee",
            "acm",
            "springer",
            "elsevier",
            "nature",
            "science",
            "arxiv",
            "pubmed",
            "scholar.google",
        ]

        for source in sources:
            source_text = str(source).lower()

            # Check for credible domains
            if any(domain in source_text for domain in credible_domains):
                credible_indicators += 1
            # Check for proper citation format (author, year, title patterns)
            elif re.search(r"\w+,?\s+\w+.*\d{4}", source_text):
                credible_indicators += 1
            # Check for DOI or URL patterns
            elif "doi:" in source_text or "http" in source_text:
                credible_indicators += 1

        return credible_indicators / total_sources if total_sources > 0 else 0.0

    def _validate_web_capability(
        self,
        response_data: dict[str, Any],
        test_case: dict[str, Any],
        result: EvaluationResult,
    ) -> None:
        """Validate web capability usage and add penalties if needed."""
        capability_profile = test_case.get("capability_profile", {})
        web_required = capability_profile.get("web") == "required"

        # Check if response indicates web access was used
        assumptions = response_data.get("assumptions", [])
        limitations = response_data.get("limitations", [])

        web_used = True  # Assume web was used unless indicated otherwise

        # Look for indicators that web was not used
        limitation_text = " ".join(str(item) for item in limitations).lower()
        assumption_text = " ".join(str(item) for item in assumptions).lower()

        no_web_indicators = [
            "no web access",
            "browsing disabled",
            "no internet",
            "offline",
            "placeholder",
            "simulated",
            "hypothetical",
            "assumed",
        ]

        if any(
            indicator in limitation_text or indicator in assumption_text
            for indicator in no_web_indicators
        ):
            web_used = False

        result.details["web_capability_used"] = web_used
        result.details["web_capability_required"] = web_required

        if web_required and not web_used:
            # Penalize for not using required web capability
            penalty = min(result.total_score * 0.2, 2.0)  # 20% penalty, max 2 points
            result.total_score = max(0.0, result.total_score - penalty)
            result.add_warning(
                f"Web access required but not used. "
                f"Applied {penalty:.1f} point penalty."
            )
            result.details["web_penalty"] = penalty
