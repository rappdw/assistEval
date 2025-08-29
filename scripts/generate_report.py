#!/usr/bin/env python3
"""Generate summary reports from evaluation results.

This script generates comparative summary reports from existing evaluation results,
allowing users to create reports after test execution has completed.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from bench.core.reporting import JSONReportGenerator, MarkdownReportGenerator
from bench.core.scoring import ProviderScore


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=Console(stderr=True))],
    )


def load_provider_scores(results_dir: Path) -> dict[str, ProviderScore]:
    """Load provider scores from results directory.

    Args:
        results_dir: Directory containing provider score files

    Returns:
        Dictionary mapping provider names to ProviderScore objects
    """
    provider_scores: dict[str, ProviderScore] = {}
    scores_dir = results_dir / "scores"
    score_files = list(scores_dir.glob("*_score.json")) if scores_dir.exists() else []

    if not score_files:
        logging.warning(f"No provider score files found in {results_dir}")
        return provider_scores

    for score_file in score_files:
        try:
            with open(score_file) as f:
                score_data = json.load(f)

            # Convert to ProviderScore object
            provider_score = ProviderScore.from_dict(score_data)
            provider_scores[provider_score.provider_name] = provider_score
            logging.info(f"Loaded scores for provider: {provider_score.provider_name}")

        except Exception as e:
            logging.error(f"Failed to load score file {score_file}: {e}")
            continue

    return provider_scores


def generate_reports(
    provider_scores: dict[str, ProviderScore], output_dir: Path
) -> None:
    """Generate markdown and JSON reports.

    Args:
        provider_scores: Dictionary of provider scores
        output_dir: Directory to save reports
    """
    console = Console()

    if not provider_scores:
        console.print(
            "[red]No valid provider scores found. Cannot generate reports.[/red]"
        )
        return

    # Create run metadata for reports
    run_metadata = {
        "run_directory": str(output_dir),
        "timestamp": datetime.now().isoformat(),
        "total_providers": len(provider_scores),
        "total_tests": len(
            [f for f in output_dir.rglob("*.json") if "metadata" in str(f)]
        ),
        "execution_time": 0.0,
        "stability_runs": 1,
    }

    # Generate markdown report
    try:
        markdown_generator = MarkdownReportGenerator()
        markdown_report = markdown_generator.generate_report(
            provider_scores, run_metadata
        )

        markdown_path = output_dir / "summary_report.md"
        with open(markdown_path, "w") as f:
            f.write(markdown_report)

        console.print(
            f"[green]✓ Markdown summary report saved: {markdown_path}[/green]"
        )

    except Exception as e:
        logging.error(f"Failed to generate markdown report: {e}")

    # Generate JSON report
    try:
        json_generator = JSONReportGenerator()
        json_report = json_generator.generate_report(provider_scores, run_metadata)

        json_path = output_dir / "summary_report.json"
        with open(json_path, "w") as f:
            json.dump(json_report, f, indent=2)

        console.print(f"[green]✓ JSON summary report saved: {json_path}[/green]")

    except Exception as e:
        logging.error(f"Failed to generate JSON report: {e}")


def display_executive_summary(provider_scores: dict[str, ProviderScore]) -> None:
    """Display executive summary in console.

    Args:
        provider_scores: Dictionary of provider scores
    """
    console = Console()

    if not provider_scores:
        return

    # Find winner
    winner = max(provider_scores.items(), key=lambda x: x[1].final_score)
    winner_name, winner_score = winner

    # Calculate score spread
    scores = [score.final_score for score in provider_scores.values()]
    score_spread = max(scores) - min(scores)

    console.print("\n" + "=" * 60)
    console.print("[bold blue]EVALUATION SUMMARY[/bold blue]")
    console.print("=" * 60)

    console.print(f"[bold green]🏆 Overall Winner: {winner_name}[/bold green]")
    console.print(
        f"   Score: {winner_score.final_score:.1f}/"
        f"{winner_score.max_score:.0f} points "
        f"({winner_score.score_percentage:.1f}%)"
    )
    console.print(f"   Stability Bonus: {winner_score.stability_bonus:.1f} points")

    console.print(f"\n[bold]📊 Score Spread: {score_spread:.1f} points[/bold]")

    console.print("\n[bold]🔍 Provider Rankings:[/bold]")
    sorted_providers = sorted(
        provider_scores.items(), key=lambda x: x[1].final_score, reverse=True
    )

    for i, (name, score) in enumerate(sorted_providers, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        console.print(
            f"   {medal} {name}: {score.final_score:.1f} points "
            f"({score.score_percentage:.1f}%)"
        )

    console.print("=" * 60 + "\n")


def find_latest_results_dir() -> Path | None:
    """Find the most recent results directory.

    Returns:
        Path to the latest results directory, or None if not found
    """
    results_base = Path("results")
    if not results_base.exists():
        return None

    # Find all run directories (format: run_YYYYMMDD_HHMMSS)
    run_dirs = [
        d for d in results_base.iterdir() if d.is_dir() and d.name.startswith("run_")
    ]

    if not run_dirs:
        return None

    # Sort by name (which includes timestamp) to get the latest
    latest_dir = sorted(run_dirs)[-1]
    return latest_dir


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate summary reports from evaluation results"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Directory containing evaluation results (defaults to latest run)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save reports (defaults to results directory)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only display executive summary, don't generate report files",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Determine results directory
    if args.results_dir:
        results_dir = args.results_dir
    else:
        results_dir = find_latest_results_dir()
        if not results_dir:
            logging.error("No results directory found. Please specify --results-dir")
            sys.exit(1)

    if not results_dir.exists():
        logging.error(f"Results directory does not exist: {results_dir}")
        sys.exit(1)

    logging.info(f"Loading results from: {results_dir}")

    # Load provider scores
    provider_scores = load_provider_scores(results_dir)

    if not provider_scores:
        logging.error("No provider scores found")
        sys.exit(1)

    # Display executive summary
    display_executive_summary(provider_scores)

    # Generate report files unless summary-only mode
    if not args.summary_only:
        output_dir = args.output_dir or results_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        generate_reports(provider_scores, output_dir)


if __name__ == "__main__":
    main()
