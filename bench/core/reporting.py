"""Report generation system for evaluation results.

This module provides Markdown and JSON report generation, leaderboards,
and detailed failure analysis for evaluation runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from bench.core.scoring import ProviderScore


@dataclass
class ReportSummary:
    """High-level report summary."""

    timestamp: datetime
    total_providers: int
    total_tasks: int
    execution_time: float
    stability_runs: int
    overall_winner: str
    score_spread: float


@dataclass
class TaskBreakdown:
    """Detailed task performance breakdown."""

    task_id: str
    task_name: str
    max_score: float
    provider_scores: dict[str, float]
    winner: str
    score_details: dict[str, dict[str, Any]]
    failure_reasons: dict[str, list[str]]


@dataclass
class ProviderComparison:
    """Provider-to-provider comparison."""

    provider_a: str
    provider_b: str
    score_difference: float
    task_wins: dict[str, str]
    strengths: dict[str, list[str]]
    weaknesses: dict[str, list[str]]


class MarkdownReportGenerator:
    """Generate human-readable markdown reports."""

    def __init__(self, template_dir: Path | None = None):
        self.template_dir = template_dir or Path(__file__).parent / "templates"

    def generate_report(
        self, provider_scores: dict[str, ProviderScore], run_metadata: dict[str, Any]
    ) -> str:
        """Generate complete markdown report."""
        sections = []

        # Header
        sections.append(self._generate_header(run_metadata))

        # Executive Summary
        sections.append(self._generate_executive_summary(provider_scores))

        # Leaderboard
        sections.append(self._generate_leaderboard_table(provider_scores))

        # Task Breakdown
        sections.append(self._generate_task_breakdown(provider_scores))

        # Stability Analysis
        sections.append(self._generate_stability_analysis(provider_scores))

        # Failure Analysis
        sections.append(self._generate_failure_analysis(provider_scores))

        return "\n\n".join(sections)

    def _generate_header(self, metadata: dict[str, Any]) -> str:
        """Generate report header."""
        timestamp = metadata.get("timestamp", datetime.now().isoformat())
        execution_time = metadata.get("execution_time", 0.0)
        provider_count = metadata.get("provider_count", 0)
        task_count = metadata.get("task_count", 0)

        return f"""# AssistEval Benchmark Report

**Generated**: {timestamp}
**Execution Time**: {execution_time:.1f}s
**Providers Tested**: {provider_count}
**Tasks Evaluated**: {task_count}"""

    def _generate_executive_summary(
        self, provider_scores: dict[str, ProviderScore]
    ) -> str:
        """Create high-level summary section."""
        if not provider_scores:
            return "## Executive Summary\n\nNo evaluation results available."

        # Find winner
        winner = max(provider_scores.items(), key=lambda x: x[1].final_score)
        winner_name, winner_score = winner

        # Calculate score spread
        all_scores = [score.final_score for score in provider_scores.values()]
        score_spread = max(all_scores) - min(all_scores) if len(all_scores) > 1 else 0.0

        summary = f"""## Executive Summary

**Overall Winner**: {winner_name} ({winner_score.final_score:.1f}/\
{winner_score.max_score:.0f} points, {winner_score.score_percentage:.1f}%)
**Score Spread**: {score_spread:.1f} points
**Stability Bonus**: {winner_score.stability_bonus:.1f} points

"""

        # Add key highlights
        highlights = []
        for provider_name, provider_score in provider_scores.items():
            if provider_score.final_score == winner_score.final_score:
                highlights.append(
                    f"• {provider_name} achieved the highest score with "
                    f"strong performance across tasks"
                )
            elif len(provider_score.task_scores) > 0:
                best_task = max(
                    provider_score.task_scores, key=lambda x: x.score_percentage
                )
                highlights.append(
                    f"• {provider_name} excelled in {best_task.task_id} "
                    f"({best_task.score_percentage:.1f}%)"
                )

        if highlights:
            summary += "**Key Highlights**:\n" + "\n".join(highlights)

        return summary

    def _generate_leaderboard_table(
        self, provider_scores: dict[str, ProviderScore]
    ) -> str:
        """Create provider ranking table."""
        if not provider_scores:
            return "## Leaderboard\n\nNo providers to rank."

        # Sort by final score descending
        sorted_providers = sorted(
            provider_scores.items(), key=lambda x: x[1].final_score, reverse=True
        )

        leaderboard = "## Leaderboard\n\n"
        leaderboard += (
            "| Rank | Provider | Total Score | Percentage | Stability Bonus |\n"
        )
        leaderboard += (
            "|------|----------|-------------|------------|-----------------|\n"
        )

        for rank, (provider_name, provider_score) in enumerate(sorted_providers, 1):
            leaderboard += (
                f"| {rank} | {provider_name} | "
                f"{provider_score.final_score:.1f}/{provider_score.max_score:.0f} | "
                f"{provider_score.score_percentage:.1f}% | "
                f"{provider_score.stability_bonus:.1f} |\n"
            )

        return leaderboard

    def _generate_task_breakdown(
        self, provider_scores: dict[str, ProviderScore]
    ) -> str:
        """Create detailed per-task analysis."""
        if not provider_scores:
            return "## Task-by-Task Breakdown\n\nNo task results available."

        # Collect all unique tasks
        all_tasks = set()
        for provider_score in provider_scores.values():
            for task_score in provider_score.task_scores:
                all_tasks.add(task_score.task_id)

        breakdown = "## Task-by-Task Breakdown\n\n"

        for task_id in sorted(all_tasks):
            breakdown += f"### {task_id}\n\n"

            # Task scores table
            breakdown += "| Provider | Score | Percentage | Errors | Warnings |\n"
            breakdown += "|----------|-------|------------|--------|----------|\n"

            task_scores = []
            for provider_name, provider_score in provider_scores.items():
                task_score_iter = (
                    ts for ts in provider_score.task_scores if ts.task_id == task_id
                )
                task_score_result = next(task_score_iter, None)
                if task_score_result:
                    task_score = task_score_result
                    task_scores.append((provider_name, task_score))
                    error_count = len(task_score.errors)
                    warning_count = len(task_score.warnings)
                    breakdown += (
                        f"| {provider_name} | {task_score.weighted_score:.1f}/"
                        f"{task_score.max_score:.0f} | "
                        f"{task_score.score_percentage:.1f}% | "
                        f"{error_count} | {warning_count} |\n"
                    )

            # Winner for this task
            if task_scores:
                winner = max(task_scores, key=lambda x: x[1].weighted_score)
                breakdown += (
                    f"\n**Winner**: {winner[0]} "
                    f"({winner[1].weighted_score:.1f} points)\n\n"
                )

        return breakdown

    def _generate_stability_analysis(
        self, provider_scores: dict[str, ProviderScore]
    ) -> str:
        """Create multi-run consistency analysis."""
        stability_section = "## Stability Analysis\n\n"

        has_stability_data = any(
            score.stability_bonus > 0 for score in provider_scores.values()
        )

        if not has_stability_data:
            stability_section += (
                "No stability bonus data available "
                "(single run or no multi-run analysis).\n"
            )
            return stability_section

        stability_section += "| Provider | Stability Bonus | Consistency |\n"
        stability_section += "|----------|-----------------|-------------|\n"

        for provider_name, provider_score in provider_scores.items():
            consistency = (
                "High"
                if provider_score.stability_bonus >= 4.0
                else ("Medium" if provider_score.stability_bonus >= 2.0 else "Low")
            )
            stability_section += (
                f"| {provider_name} | {provider_score.stability_bonus:.1f}/5.0 | "
                f"{consistency} |\n"
            )

        return stability_section

    def _generate_failure_analysis(
        self, provider_scores: dict[str, ProviderScore]
    ) -> str:
        """Create detailed error and warning breakdown."""
        failure_section = "## Failure Analysis\n\n"

        # Collect all errors and warnings
        all_errors = {}
        all_warnings = {}

        for provider_name, provider_score in provider_scores.items():
            provider_errors = []
            provider_warnings = []

            for task_score in provider_score.task_scores:
                for error in task_score.errors:
                    provider_errors.append(f"{task_score.task_id}: {error}")
                for warning in task_score.warnings:
                    provider_warnings.append(f"{task_score.task_id}: {warning}")

            if provider_errors:
                all_errors[provider_name] = provider_errors
            if provider_warnings:
                all_warnings[provider_name] = provider_warnings

        if not all_errors and not all_warnings:
            failure_section += "No errors or warnings reported.\n"
            return failure_section

        # Errors section
        if all_errors:
            failure_section += "### Errors\n\n"
            for provider_name, errors in all_errors.items():
                failure_section += f"**{provider_name}**:\n"
                for error in errors:
                    failure_section += f"- {error}\n"
                failure_section += "\n"

        # Warnings section
        if all_warnings:
            failure_section += "### Warnings\n\n"
            for provider_name, warnings in all_warnings.items():
                failure_section += f"**{provider_name}**:\n"
                for warning in warnings:
                    failure_section += f"- {warning}\n"
                failure_section += "\n"

        return failure_section


class JSONReportGenerator:
    """Generate machine-processable JSON reports."""

    def generate_report(
        self, provider_scores: dict[str, ProviderScore], run_metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate structured JSON report."""
        return {
            "metadata": self._create_metadata_section(run_metadata),
            "summary": self._create_summary_section(provider_scores),
            "leaderboard": self._create_leaderboard_section(provider_scores),
            "tasks": self._create_task_section(provider_scores),
            "providers": self._create_provider_section(provider_scores),
        }

    def _create_metadata_section(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Create execution metadata."""
        return {
            "timestamp": metadata.get("timestamp", datetime.now().isoformat()),
            "version": "1.0.0",
            "execution_time": metadata.get("execution_time", 0.0),
            "run_id": metadata.get("run_id", "unknown"),
        }

    def _create_summary_section(
        self, scores: dict[str, ProviderScore]
    ) -> dict[str, Any]:
        """Create summary statistics."""
        if not scores:
            return {
                "total_providers": 0,
                "total_tasks": 0,
                "overall_winner": None,
                "score_spread": 0.0,
                "stability_runs": 0,
            }

        winner = max(scores.items(), key=lambda x: x[1].final_score)
        all_scores = [score.final_score for score in scores.values()]
        score_spread = max(all_scores) - min(all_scores) if len(all_scores) > 1 else 0.0

        # Count unique tasks
        all_tasks = set()
        for provider_score in scores.values():
            for task_score in provider_score.task_scores:
                all_tasks.add(task_score.task_id)

        return {
            "total_providers": len(scores),
            "total_tasks": len(all_tasks),
            "overall_winner": winner[0],
            "score_spread": score_spread,
            "stability_runs": 3,  # Default assumption
        }

    def _create_leaderboard_section(
        self, scores: dict[str, ProviderScore]
    ) -> list[dict[str, Any]]:
        """Create leaderboard data."""
        sorted_providers = sorted(
            scores.items(), key=lambda x: x[1].final_score, reverse=True
        )

        leaderboard: list[dict[str, Any]] = []
        for rank, (provider_name, provider_score) in enumerate(sorted_providers, 1):
            leaderboard.append(
                {
                    "provider": provider_name,
                    "total_score": provider_score.final_score,
                    "max_score": provider_score.max_score,
                    "percentage": provider_score.score_percentage,
                    "rank": rank,
                    "stability_bonus": provider_score.stability_bonus,
                }
            )

        return leaderboard

    def _create_task_section(self, scores: dict[str, ProviderScore]) -> dict[str, Any]:
        """Create per-task comparison matrix."""
        # Collect all unique tasks
        all_tasks = set()
        for provider_score in scores.values():
            for task_score in provider_score.task_scores:
                all_tasks.add(task_score.task_id)

        tasks = {}
        for task_id in all_tasks:
            task_scores = {}
            max_score = 0.0
            winner = None
            winner_score = -1.0

            for provider_name, provider_score in scores.items():
                task_score_iter = (
                    ts for ts in provider_score.task_scores if ts.task_id == task_id
                )
                task_score_result = next(task_score_iter, None)
                if task_score_result:
                    task_score = task_score_result
                    task_scores[provider_name] = task_score.weighted_score
                    max_score = task_score.max_score
                    if task_score.weighted_score > winner_score:
                        winner = provider_name
                        winner_score = task_score.weighted_score

            tasks[task_id] = {
                "max_score": max_score,
                "winner": winner,
                "scores": task_scores,
            }

        return tasks

    def _create_provider_section(
        self, scores: dict[str, ProviderScore]
    ) -> dict[str, Any]:
        """Create per-provider detailed results."""
        providers = {}

        for provider_name, provider_score in scores.items():
            # Collect errors and warnings
            all_errors = []
            all_warnings = []

            for task_score in provider_score.task_scores:
                all_errors.extend(task_score.errors)
                all_warnings.extend(task_score.warnings)

            providers[provider_name] = {
                "total_score": provider_score.total_score,
                "final_score": provider_score.final_score,
                "max_score": provider_score.max_score,
                "percentage": provider_score.score_percentage,
                "stability_bonus": provider_score.stability_bonus,
                "task_count": len(provider_score.task_scores),
                "errors": all_errors,
                "warnings": all_warnings,
                "task_scores": [
                    {
                        "task_id": ts.task_id,
                        "evaluator_name": ts.evaluator_name,
                        "raw_score": ts.raw_score,
                        "weighted_score": ts.weighted_score,
                        "max_score": ts.max_score,
                        "percentage": ts.score_percentage,
                        "sub_scores": ts.sub_scores,
                        "errors": ts.errors,
                        "warnings": ts.warnings,
                    }
                    for ts in provider_score.task_scores
                ],
            }

        return providers


class ConsolidatedReporter:
    """Main reporting orchestration class."""

    def __init__(self, results_dir: Path, output_dir: Path):
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.markdown_generator = MarkdownReportGenerator()
        self.json_generator = JSONReportGenerator()
        self.aggregator = ReportAggregator(results_dir)

    def generate_reports(
        self,
        run_dirs: list[Path] | None = None,
        formats: list[str] | None = None,
        aggregate: bool = False,
    ) -> dict[str, Path]:
        """Generate all requested report formats."""
        if formats is None:
            formats = ["markdown", "json"]
        if run_dirs is None:
            run_dirs = self._find_latest_runs(1)

        if not run_dirs:
            raise ValueError("No valid run directories found")

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        generated_files = {}

        if aggregate and len(run_dirs) > 1:
            # Generate aggregated report
            aggregated_data = self.aggregator.aggregate_runs([str(d) for d in run_dirs])
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            for format_type in formats:
                if format_type == "markdown":
                    filename = f"aggregated_report_{timestamp}.md"
                    filepath = self.output_dir / filename
                    content = self.markdown_generator.generate_report(
                        aggregated_data["provider_scores"],
                        aggregated_data.get("metadata", {}),
                    )
                    filepath.write_text(content)
                    generated_files[f"aggregated_{format_type}"] = filepath
                elif format_type == "json":
                    filename = f"aggregated_report_{timestamp}.json"
                    filepath = self.output_dir / filename
                    report_content = self.json_generator.generate_report(
                        aggregated_data["provider_scores"],
                        aggregated_data.get("metadata", {}),
                    )
                    import json

                    json_content = json.dumps(report_content, indent=2)
                    filepath.write_text(json_content)
                    generated_files[f"aggregated_{format_type}"] = filepath
        else:
            # Generate individual run reports
            for run_dir in run_dirs:
                run_data = self.aggregator._load_run_data(run_dir)
                if not run_data:
                    continue

                run_name = run_dir.name
                for format_type in formats:
                    if format_type == "markdown":
                        filename = f"{run_name}_report.md"
                        filepath = self.output_dir / filename
                        content = self.markdown_generator.generate_report(
                            run_data["provider_scores"], run_data.get("metadata", {})
                        )
                        filepath.write_text(content)
                        generated_files[f"{run_name}_{format_type}"] = filepath
                    elif format_type == "json":
                        filename = f"{run_name}_report.json"
                        filepath = self.output_dir / filename
                        report_content = self.json_generator.generate_report(
                            run_data["provider_scores"], run_data.get("metadata", {})
                        )
                        import json

                        json_content = json.dumps(report_content, indent=2)
                        filepath.write_text(json_content)
                        generated_files[f"{run_name}_{format_type}"] = filepath

        return generated_files

    def _find_latest_runs(self, count: int = 1) -> list[Path]:
        """Find the latest run directories."""
        if not self.results_dir.exists():
            return []

        run_dirs = [
            d
            for d in self.results_dir.iterdir()
            if d.is_dir() and d.name.startswith("run_")
        ]

        # Sort by modification time (newest first)
        run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        return run_dirs[:count]


class ReportAggregator:
    """Aggregate multiple runs into consolidated reports."""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir

    def aggregate_runs(self, run_patterns: list[str] | None = None) -> dict[str, Any]:
        """Aggregate multiple run results."""
        if run_patterns is None:
            # Find all run directories
            run_dirs = [
                d
                for d in self.results_dir.iterdir()
                if d.is_dir() and d.name.startswith("run_")
            ]
        else:
            # Use specific patterns
            run_dirs_list: list[Path] = []
            for pattern in run_patterns:
                run_dirs_list.extend(self.results_dir.glob(pattern))
            run_dirs = run_dirs_list

        # Load data from each run
        historical_data = []
        for run_dir in sorted(run_dirs):
            try:
                run_data = self._load_run_data(run_dir)
                if run_data:
                    historical_data.append(run_data)
            except Exception as e:
                # Skip invalid run directories
                import logging

                logging.warning(f"Skipping invalid run directory {run_dir}: {e}")
                continue

        if not historical_data:
            return {"error": "No valid run data found"}

        return {
            "runs": historical_data,
            "trends": self._calculate_trends(historical_data),
            "regressions": self._identify_regressions(historical_data),
        }

    def _load_run_data(self, run_dir: Path) -> dict[str, Any] | None:
        """Load data from a single run directory."""
        scores_dir = run_dir / "scores"
        if not scores_dir.exists():
            return None

        # Load all provider scores
        provider_scores = {}
        for score_file in scores_dir.glob("*_score.json"):
            provider_name = score_file.stem.replace("_score", "")
            try:
                with open(score_file) as f:
                    score_data = json.load(f)
                    provider_scores[provider_name] = score_data
            except Exception as e:
                import logging

                logging.warning(f"Failed to load score file {score_file}: {e}")
                continue

        if not provider_scores:
            return None

        return {
            "run_id": run_dir.name,
            "timestamp": run_dir.stat().st_mtime,
            "provider_scores": provider_scores,
        }

    def _calculate_trends(
        self, historical_data: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Calculate performance trends over time."""
        if len(historical_data) < 2:
            return {"insufficient_data": True}

        # Extract scores over time for each provider
        provider_trends: dict[str, Any] = {}

        for run_data in historical_data:
            for provider_name, score_data in run_data["provider_scores"].items():
                if provider_name not in provider_trends:
                    provider_trends[provider_name] = []

                provider_trends[provider_name].append(
                    {
                        "timestamp": run_data["timestamp"],
                        "final_score": score_data.get("final_score", 0.0),
                    }
                )

        # Calculate trend direction for each provider
        trends = {}
        for provider_name, scores in provider_trends.items():
            if len(scores) >= 2:
                first_score = scores[0]["final_score"]
                last_score = scores[-1]["final_score"]
                trend_direction = (
                    "improving"
                    if last_score > first_score
                    else ("declining" if last_score < first_score else "stable")
                )

                trends[provider_name] = {
                    "direction": trend_direction,
                    "change": last_score - first_score,
                    "data_points": len(scores),
                }

        return trends

    def _identify_regressions(
        self, historical_data: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Identify performance regressions."""
        if len(historical_data) < 2:
            return {}

        regressions: dict[str, Any] = {}

        # Compare last two runs
        if len(historical_data) >= 2:
            prev_run = historical_data[-2]
            curr_run = historical_data[-1]

            for provider_name in curr_run["provider_scores"]:
                if provider_name in prev_run["provider_scores"]:
                    prev_score = prev_run["provider_scores"][provider_name].get(
                        "final_score", 0.0
                    )
                    curr_score = curr_run["provider_scores"][provider_name].get(
                        "final_score", 0.0
                    )

                    # Consider it a regression if score dropped by more than 5%
                    if curr_score < prev_score * 0.95:
                        regressions[provider_name] = {
                            "previous_score": prev_score,
                            "current_score": curr_score,
                            "regression_amount": prev_score - curr_score,
                            "regression_percentage": (
                                (prev_score - curr_score) / prev_score
                            )
                            * 100,
                        }

        return regressions
