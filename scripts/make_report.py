#!/usr/bin/env python3
"""Generate consolidated reports from evaluation results.

This script creates comprehensive reports from prior evaluation runs,
combining results across providers and generating leaderboards.
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

# Add bench package to path for development
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import after path modification
from bench.core.reporting import ConsolidatedReporter  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate consolidated reports from evaluation results"
    )

    parser.add_argument(
        "results_dir", type=Path, help="Directory containing evaluation results"
    )

    parser.add_argument(
        "output_dir", type=Path, help="Directory to save generated reports"
    )

    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["markdown", "json"],
        default=["markdown", "json"],
        help="Report formats to generate (default: both)",
    )

    parser.add_argument(
        "--runs", nargs="+", type=str, help="Specific run directories to include"
    )

    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Generate aggregated report across multiple runs",
    )

    parser.add_argument(
        "--latest",
        type=int,
        default=1,
        help="Number of latest runs to include (default: 1)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for report generation."""
    console = Console()

    try:
        args = parse_arguments()

        # Validate input directory
        if not args.results_dir.exists():
            console.print(
                f"[red]Error: Results directory {args.results_dir} does not exist[/red]"
            )
            sys.exit(1)

        # Create reporter
        reporter = ConsolidatedReporter(args.results_dir, args.output_dir)

        # Determine run directories
        run_dirs: list[Path] = []
        if args.runs:
            # Use specific runs
            for run_name in args.runs:
                run_path = args.results_dir / run_name
                if run_path.exists():
                    run_dirs.append(run_path)
                else:
                    console.print(
                        f"[yellow]Warning: Run directory {run_path} not found[/yellow]"
                    )
        else:
            # Use latest runs
            run_dirs = reporter._find_latest_runs(args.latest)

        if not run_dirs:
            console.print("[red]Error: No valid run directories found[/red]")
            sys.exit(1)

        # Generate reports
        console.print(
            Panel.fit(
                f"Generating reports for {len(run_dirs)} run(s)",
                title="Report Generation",
            )
        )

        generated_files = reporter.generate_reports(
            run_dirs=run_dirs, formats=args.formats, aggregate=args.aggregate
        )

        # Display results
        console.print("\n[green]✓ Reports generated successfully![/green]")
        for report_type, file_path in generated_files.items():
            console.print(f"  {report_type}: {file_path}")

    except Exception as e:
        console.print(f"[red]Error generating reports: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
