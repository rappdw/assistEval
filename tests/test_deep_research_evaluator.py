"""Tests for deep research evaluator."""


from bench.core.evaluators.deep_research import DeepResearchEvaluator


class TestDeepResearchEvaluator:
    """Test suite for DeepResearchEvaluator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = DeepResearchEvaluator({})
        self.base_test_case = {
            "id": "online.deep_research.agentic_ai",
            "capability_profile": {"web": "required"},
            "scoring": {
                "config": {
                    "plan_requirements": {
                        "min_steps": 7,
                        "max_steps": 10,
                        "required_fields": ["goal", "method", "deliverable"],
                    },
                    "source_requirements": {
                        "min_sources": 5,
                        "max_sources": 8,
                        "min_recent_sources": 3,
                        "recency_years": 3,
                    },
                }
            },
        }

    def test_perfect_response(self):
        """Test evaluation of a perfect deep research response."""
        response_data = {
            "research_plan": [
                {
                    "goal": "Define agentic AI scope and boundaries",
                    "method": "Literature review of recent publications",
                    "deliverable": "Comprehensive definition document",
                },
                {
                    "goal": "Analyze current market landscape",
                    "method": "Survey existing agentic AI solutions",
                    "deliverable": "Market analysis report with key players",
                },
                {
                    "goal": "Evaluate technical architectures",
                    "method": "Deep dive into technical papers and implementations",
                    "deliverable": "Technical architecture comparison matrix",
                },
                {
                    "goal": "Assess adoption barriers",
                    "method": "Interview industry experts and analyze case studies",
                    "deliverable": "Barrier analysis with mitigation strategies",
                },
                {
                    "goal": "Research regulatory landscape",
                    "method": "Review current and proposed AI regulations",
                    "deliverable": "Regulatory compliance framework",
                },
                {
                    "goal": "Analyze competitive positioning",
                    "method": "Based on market analysis, evaluate competitive "
                    "advantages",
                    "deliverable": "Competitive positioning strategy",
                },
                {
                    "goal": "Synthesize findings",
                    "method": "Integrate all research components into cohesive "
                    "analysis",
                    "deliverable": "Final comprehensive research report",
                },
            ],
            "sources": [
                "Smith, J. et al. (2024). Agentic AI Systems: A Comprehensive Survey. "
                "Nature AI, 15(3), 45-67.",
                "OpenAI Research Team (2023). GPT-4 and Beyond: Autonomous Agent "
                "Capabilities. arXiv:2303.12345",
                "Johnson, M. (2024). Enterprise Adoption of Agentic AI. "
                "IEEE Computer, 57(2), 23-31.",
                "Brown, A. & Davis, L. (2023). Regulatory Frameworks for "
                "Autonomous AI. AI Ethics Journal, 8(4), 112-128.",
                "Microsoft Research (2024). Agentic AI in Production: Lessons Learned. "
                "https://research.microsoft.com/agentic-ai-2024",
                "Chen, W. (2022). Multi-Agent Systems in Business Applications. "
                "ACM Computing Surveys, 54(7), 1-35.",
            ],
            "assumptions": [
                "Web access is available for current research",
                "Industry experts are willing to participate in interviews",
            ],
            "limitations": [
                "Research limited to English-language sources",
                "Time constraints may limit depth of expert interviews",
            ],
        }

        result = self.evaluator.evaluate(response_data, self.base_test_case)

        assert result.total_score >= 8.0  # Should get high score
        assert result.sub_scores["plan_quality"] >= 4.5
        assert result.sub_scores["source_quality"] >= 4.5
        assert len(result.errors) == 0
        assert result.details["plan_step_count"] == 7
        assert result.details["source_count"] == 6
        assert result.details["web_capability_used"] is True

    def test_minimal_valid_response(self):
        """Test evaluation of minimal but valid response."""
        response_data = {
            "research_plan": [
                {
                    "goal": "Research AI",
                    "method": "Read papers",
                    "deliverable": "Report",
                },
                {
                    "goal": "Analyze data",
                    "method": "Use tools",
                    "deliverable": "Analysis",
                },
                {
                    "goal": "Write summary",
                    "method": "Synthesize findings",
                    "deliverable": "Summary",
                },
                {
                    "goal": "Review results",
                    "method": "Check accuracy",
                    "deliverable": "Review",
                },
                {
                    "goal": "Finalize",
                    "method": "Polish document",
                    "deliverable": "Final report",
                },
                {
                    "goal": "Present",
                    "method": "Create presentation",
                    "deliverable": "Slides",
                },
                {
                    "goal": "Distribute",
                    "method": "Share with team",
                    "deliverable": "Distribution",
                },
            ],
            "sources": [
                "Paper 1 (2024)",
                "Paper 2 (2023)",
                "Paper 3 (2022)",
                "Paper 4 (2024)",
                "Paper 5 (2023)",
            ],
            "assumptions": [],
            "limitations": [],
        }

        result = self.evaluator.evaluate(response_data, self.base_test_case)

        assert result.total_score > 0
        assert result.sub_scores["plan_quality"] > 0
        assert result.sub_scores["source_quality"] > 0
        assert result.details["plan_step_count"] == 7
        assert result.details["source_count"] == 5

    def test_insufficient_steps(self):
        """Test response with too few research steps."""
        response_data = {
            "research_plan": [
                {"goal": "Research", "method": "Read", "deliverable": "Report"},
                {"goal": "Analyze", "method": "Think", "deliverable": "Analysis"},
            ],
            "sources": [
                "Source 1 (2024)",
                "Source 2 (2023)",
                "Source 3 (2022)",
                "Source 4 (2024)",
                "Source 5 (2023)",
            ],
            "assumptions": [],
            "limitations": [],
        }

        result = self.evaluator.evaluate(response_data, self.base_test_case)

        assert result.details["plan_step_count"] == 2
        assert result.details["step_count_valid"] is False
        assert "Expected 7-10 steps, got 2" in result.details["step_count_issue"]

    def test_too_many_steps(self):
        """Test response with too many research steps."""
        response_data = {
            "research_plan": [
                {
                    "goal": f"Step {i}",
                    "method": f"Method {i}",
                    "deliverable": f"Deliverable {i}",
                }
                for i in range(1, 13)  # 12 steps
            ],
            "sources": [
                "Source 1 (2024)",
                "Source 2 (2023)",
                "Source 3 (2022)",
                "Source 4 (2024)",
                "Source 5 (2023)",
            ],
            "assumptions": [],
            "limitations": [],
        }

        result = self.evaluator.evaluate(response_data, self.base_test_case)

        assert result.details["plan_step_count"] == 12
        assert result.details["step_count_valid"] is False
        assert "Expected 7-10 steps, got 12" in result.details["step_count_issue"]

    def test_missing_required_fields(self):
        """Test response with missing required fields in steps."""
        response_data = {
            "research_plan": [
                {"goal": "Research AI"},  # Missing method and deliverable
                {"method": "Read papers", "deliverable": "Report"},  # Missing goal
                {"goal": "Analyze", "method": "Think"},  # Missing deliverable
                {"goal": "Complete", "method": "Finish", "deliverable": "Done"},
                {"goal": "Review", "method": "Check", "deliverable": "Review"},
                {"goal": "Present", "method": "Show", "deliverable": "Presentation"},
                {
                    "goal": "Distribute",
                    "method": "Share",
                    "deliverable": "Distribution",
                },
            ],
            "sources": [
                "Source 1 (2024)",
                "Source 2 (2023)",
                "Source 3 (2022)",
                "Source 4 (2024)",
                "Source 5 (2023)",
            ],
            "assumptions": [],
            "limitations": [],
        }

        result = self.evaluator.evaluate(response_data, self.base_test_case)

        assert (
            result.details["valid_steps"] == 4
        )  # Only 4 steps have all required fields
        assert result.details["field_completeness"] < 1.0
        assert len(result.details["field_issues"]) == 3

    def test_insufficient_sources(self):
        """Test response with too few sources."""
        response_data = {
            "research_plan": [
                {
                    "goal": f"Step {i}",
                    "method": f"Method {i}",
                    "deliverable": f"Deliverable {i}",
                }
                for i in range(1, 8)  # 7 steps
            ],
            "sources": ["Source 1 (2024)", "Source 2 (2023)"],
            "assumptions": [],
            "limitations": [],
        }

        result = self.evaluator.evaluate(response_data, self.base_test_case)

        assert result.details["source_count"] == 2
        assert result.details["source_count_valid"] is False
        assert result.sub_scores["source_quality"] < 4.0  # Should get partial credit

    def test_outdated_sources(self):
        """Test response with mostly outdated sources."""
        response_data = {
            "research_plan": [
                {
                    "goal": f"Step {i}",
                    "method": f"Method {i}",
                    "deliverable": f"Deliverable {i}",
                }
                for i in range(1, 8)  # 7 steps
            ],
            "sources": [
                "Old Paper 1 (2018)",
                "Old Paper 2 (2019)",
                "Old Paper 3 (2020)",
                "Recent Paper 1 (2024)",
                "Recent Paper 2 (2023)",
            ],
            "assumptions": [],
            "limitations": [],
        }

        result = self.evaluator.evaluate(response_data, self.base_test_case)

        assert result.details["source_count"] == 5
        assert result.details["recent_sources"] == 2  # Only 2 recent sources
        assert result.details["required_recent_sources"] == 3
        assert (
            result.sub_scores["source_quality"] < 5.0
        )  # Should lose points for recency

    def test_web_capability_not_used(self):
        """Test penalty when web capability is required but not used."""
        response_data = {
            "research_plan": [
                {
                    "goal": f"Step {i}",
                    "method": f"Method {i}",
                    "deliverable": f"Deliverable {i}",
                }
                for i in range(1, 8)  # 7 steps
            ],
            "sources": [
                "Source 1 (2024)",
                "Source 2 (2023)",
                "Source 3 (2022)",
                "Source 4 (2024)",
                "Source 5 (2023)",
            ],
            "assumptions": ["No web access available"],
            "limitations": ["Research conducted offline with placeholder sources"],
        }

        result = self.evaluator.evaluate(response_data, self.base_test_case)

        assert result.details["web_capability_used"] is False
        assert result.details["web_capability_required"] is True
        assert "web_penalty" in result.details
        assert len(result.warnings) > 0
        assert "Web access required but not used" in result.warnings[0]

    def test_malformed_response(self):
        """Test handling of malformed response data."""
        response_data = {
            "research_plan": "not an array",  # Should be array
            "sources": ["Source 1", "Source 2"],
            "assumptions": [],
            "limitations": [],
        }

        result = self.evaluator.evaluate(response_data, self.base_test_case)

        assert result.sub_scores["plan_quality"] == 0.0
        assert "Research plan must be an array" in result.details["plan_error"]

    def test_empty_response(self):
        """Test handling of empty response."""
        response_data = {}

        result = self.evaluator.evaluate(response_data, self.base_test_case)

        assert result.total_score <= 1.0  # Should get very low score
        assert result.sub_scores["plan_quality"] <= 1.0
        assert result.sub_scores["source_quality"] == 0.0

    def test_source_credibility_scoring(self):
        """Test source credibility evaluation."""
        # Test with credible sources
        credible_sources = [
            "Smith, J. (2024). AI Research. Nature AI, 15(3), 45-67.",
            "OpenAI (2023). GPT-4 Technical Report. arXiv:2303.12345",
            "Johnson, M. (2024). Enterprise AI. IEEE Computer, 57(2), 23-31.",
            "https://research.microsoft.com/ai-2024",
            "Brown, A. (2023). AI Ethics. doi:10.1000/182",
        ]

        credibility_score = self.evaluator._evaluate_source_credibility(
            credible_sources
        )
        assert credibility_score == 1.0  # All sources should be considered credible

        # Test with non-credible sources
        non_credible_sources = [
            "Random blog post",
            "Wikipedia article",
            "Personal opinion",
        ]

        credibility_score = self.evaluator._evaluate_source_credibility(
            non_credible_sources
        )
        assert credibility_score == 0.0  # No sources should be considered credible

    def test_sequencing_evaluation(self):
        """Test logical sequencing evaluation."""
        # Test with good sequencing
        good_sequence = [
            {"goal": "collect data", "method": "surveys", "deliverable": "dataset"},
            {
                "goal": "analyze data using collected information",
                "method": "statistics",
                "deliverable": "analysis",
            },
            {
                "goal": "synthesize findings based on analysis",
                "method": "integration",
                "deliverable": "synthesis",
            },
        ]

        sequencing_score = self.evaluator._evaluate_sequencing(good_sequence)
        assert sequencing_score > 0.5  # Should get partial credit for logical flow

        # Test with poor sequencing
        poor_sequence = [
            {"goal": "random task 1", "method": "method", "deliverable": "output"},
            {"goal": "unrelated task 2", "method": "method", "deliverable": "output"},
            {
                "goal": "disconnected task 3",
                "method": "method",
                "deliverable": "output",
            },
        ]

        sequencing_score = self.evaluator._evaluate_sequencing(poor_sequence)
        assert sequencing_score <= 0.5  # Should get low score for poor sequencing

    def test_completeness_evaluation(self):
        """Test deliverable completeness evaluation."""
        # Test with complete deliverables
        complete_plan = [
            {
                "goal": "research",
                "method": "method",
                "deliverable": "Comprehensive research report with findings",
            },
            {
                "goal": "analyze",
                "method": "method",
                "deliverable": "Detailed analysis with recommendations",
            },
            {
                "goal": "present",
                "method": "method",
                "deliverable": "Executive presentation with key insights",
            },
        ]

        completeness_score = self.evaluator._evaluate_completeness(complete_plan)
        assert completeness_score == 1.0  # All steps have meaningful deliverables

        # Test with incomplete deliverables
        incomplete_plan = [
            {"goal": "research", "method": "method", "deliverable": "Report"},
            {
                "goal": "analyze",
                "method": "method",
                "deliverable": "",
            },  # Empty deliverable
            {"goal": "present", "method": "method"},  # Missing deliverable
        ]

        completeness_score = self.evaluator._evaluate_completeness(incomplete_plan)
        assert (
            completeness_score < 1.0
        )  # Should lose points for incomplete deliverables

    def test_error_handling(self):
        """Test error handling in evaluation."""
        # Test with response that causes evaluation error
        problematic_response = {
            "research_plan": [
                {"goal": None, "method": None, "deliverable": None}  # None values
            ],
            "sources": None,  # None instead of array
            "assumptions": [],
            "limitations": [],
        }

        result = self.evaluator.evaluate(problematic_response, self.base_test_case)

        # Should handle gracefully and return some result
        assert isinstance(result.total_score, float)
        assert result.total_score >= 0.0
