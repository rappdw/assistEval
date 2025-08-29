#!/usr/bin/env python3
"""Command-line interface for the evaluation harness.

Provides CLI commands for running evaluations, generating reports,
and managing test configurations with comprehensive argument parsing.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table


def handle_prepare_manual_command(args: argparse.Namespace, console: Console) -> None:
    """Handle prepare-manual command."""
    from bench.core.runner import TestRunner
    from bench.core.utils import load_yaml_config

    console.print("[bold blue]Preparing manual evaluation prompts...[/bold blue]")

    try:
        # Load configuration
        config_dir = Path("configs")
        runmatrix_config = load_yaml_config(config_dir / "runmatrix.yaml")

        # Initialize runner
        runner = TestRunner(config_dir)

        # Get test cases based on test set
        test_cases = []
        if args.test_set in ["offline", "all"]:
            test_cases.extend(runmatrix_config.get("test_sets", {}).get("offline", []))
        if args.test_set in ["online", "all"]:
            test_cases.extend(runmatrix_config.get("test_sets", {}).get("online", []))

        if not test_cases:
            console.print("[red]No test cases found for the specified test set[/red]")
            return

        # Generate prompts
        prompts = []
        for i in range(args.repetitions):
            for test_id in test_cases:
                try:
                    test_path = Path("bench") / test_id
                    test_case = runner.context.load_test_definition(test_path)
                    prompt_text = _format_manual_prompt(test_case, args.format, i + 1)
                    prompts.append(prompt_text)
                except Exception as e:
                    console.print(
                        f"[yellow]Warning: Failed to load test {test_id}: {e}[/yellow]"
                    )

        # Write prompts to file
        with open(args.output, "w", encoding="utf-8") as f:
            if args.format == "template":
                f.write("# Manual Evaluation Prompts\n\n")
                f.write("Copy each prompt below and paste into Microsoft Copilot.\n")
                f.write("Save responses in the same order for processing.\n\n")
                f.write("=" * 80 + "\n\n")

            for i, prompt in enumerate(prompts, 1):
                f.write(
                    f"## Prompt {i}\n\n"
                    if args.format == "template"
                    else f"PROMPT {i}:\n"
                )
                f.write(prompt)
                f.write(
                    "\n\n" + "=" * 80 + "\n\n"
                    if args.format == "template"
                    else "\n\n---\n\n"
                )

        console.print(
            f"[green]Generated {len(prompts)} prompts in {args.output}[/green]"
        )

    except Exception as e:
        console.print(f"[red]Error preparing manual prompts: {e}[/red]")
        raise


def handle_process_manual_command(args: argparse.Namespace, console: Console) -> None:
    """Handle process-manual command."""
    from datetime import datetime

    from bench.core.runner import TestRunner
    from bench.core.utils import load_yaml_config

    console.print("[bold blue]Processing manual evaluation results...[/bold blue]")

    try:
        # Load responses file
        if not args.responses.exists():
            console.print(f"[red]Responses file not found: {args.responses}[/red]")
            return

        with open(args.responses, encoding="utf-8") as f:
            responses_text = f.read()

        # Parse responses (simple delimiter-based parsing)
        responses = _parse_manual_responses(responses_text)

        if not responses:
            console.print("[red]No responses found in the file[/red]")
            return

        # Load configuration
        config_dir = args.config_dir
        runmatrix_config = load_yaml_config(config_dir / "runmatrix.yaml")

        # Initialize runner and results collector
        TestRunner(config_dir)

        # Determine output directory
        output_dir = (
            args.output
            or Path("results") / f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Note: ResultsCollector would be used for actual result storage
        # results_collector = ResultsCollector(output_dir)

        # Process each response
        test_cases = runmatrix_config.get("test_sets", {}).get("offline", [])

        for i, _response in enumerate(responses):
            if i >= len(test_cases):
                console.print(
                    "[yellow]Warning: More responses than test cases, "
                    "ignoring extra responses[/yellow]"
                )
                break

            test_id = test_cases[i]

            try:
                # Create result entry (placeholder for actual implementation)
                # result = {
                #     "test_id": test_id,
                #     "provider": args.provider,
                #     "timestamp": datetime.now().isoformat(),
                #     "response": response,
                #     "metadata": {
                #         "manual_evaluation": True,
                #         "processed_at": datetime.now().isoformat(),
                #     },
                # }

                # Save raw result (placeholder - would need actual implementation)
                # results_collector.save_raw_result(test_id, args.provider, result)
                pass

                console.print(f"[green]Processed response for {test_id}[/green]")

            except Exception as e:
                console.print(
                    f"[yellow]Warning: Failed to process response for "
                    f"{test_id}: {e}[/yellow]"
                )

        console.print(
            f"[green]Processed {len(responses)} manual responses in "
            f"{output_dir}[/green]"
        )
        console.print(
            f"[blue]Run evaluation with: python scripts/bench.py run "
            f"--results-dir {output_dir}[/blue]"
        )

    except Exception as e:
        console.print(f"[red]Error processing manual responses: {e}[/red]")
        raise


def _format_manual_prompt(
    test_case: dict[str, Any], format_type: str, repetition: int
) -> str:
    """Format a test case as a manual prompt."""
    prompt_parts = []

    if format_type == "template":
        prompt_parts.append(f"**Test ID:** {test_case.get('id', 'Unknown')}")
        prompt_parts.append(f"**Repetition:** {repetition}")
        prompt_parts.append("")

    # Add system prompt if present
    if "prompt" in test_case and "system" in test_case["prompt"]:
        if format_type == "template":
            prompt_parts.append("**System Instructions:**")
        prompt_parts.append(test_case["prompt"]["system"])
        prompt_parts.append("")

    # Add user prompt
    if "prompt" in test_case and "user" in test_case["prompt"]:
        if format_type == "template":
            prompt_parts.append("**User Prompt:**")
        prompt_parts.append(test_case["prompt"]["user"])

    return "\n".join(prompt_parts)


def _parse_manual_responses(text: str) -> list[str]:
    """Parse manual responses from text file."""
    responses = []
    current_response: list[str] = []

    lines = text.split("\n")
    in_response = False

    for line in lines:
        line = line.strip()

        # Check if this is a delimiter line
        is_delimiter = any(
            delim in line.upper() for delim in ["---", "===", "RESPONSE"]
        )

        if is_delimiter and current_response:
            # End of current response
            response_text = "\n".join(current_response).strip()
            if response_text:
                responses.append(response_text)
            current_response = []
            in_response = False
        elif is_delimiter:
            # Start of new response
            in_response = True
        elif in_response or (not responses and line):
            # Part of response content
            current_response.append(line)

    # Add final response if exists
    if current_response:
        response_text = "\n".join(current_response).strip()
        if response_text:
            responses.append(response_text)

    return responses


def main() -> None:
    """Main entry point for the CLI."""
    # Add the project root to Python path for imports
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    # Import after path setup
    from bench.core.runner import ConfigurationError, TestExecutionError

    console = Console()

    # Create argument parser
    parser = argparse.ArgumentParser(
        description="ChatGPT vs Microsoft Copilot Evaluation Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.epilog = """Examples:
  # Run full test matrix
  python scripts/bench.py run --matrix

  # Run single test with specific provider
  python scripts/bench.py run --provider chatgpt \
    --test bench/tests/offline/task1_metrics.yaml

  # Validate test definition
  python scripts/bench.py validate \
    --test bench/tests/offline/task1_metrics.yaml

  # Check execution status
  python scripts/bench.py status --run results/run_20250827_160000

  # Generate reports from latest run
  python scripts/bench.py report

  # Generate aggregated report from multiple runs
  python scripts/bench.py report --aggregate --latest 3
        """

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Execute tests")
    run_group = run_parser.add_mutually_exclusive_group(required=True)
    run_group.add_argument(
        "--matrix",
        action="store_true",
        help="Execute full test matrix from runmatrix.yaml",
    )
    run_group.add_argument(
        "--provider",
        help="Run single test with specific provider",
    )

    run_parser.add_argument(
        "--test",
        type=Path,
        help="Path to specific test file (required with --provider)",
    )
    run_parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of repetitions for single test (default: 1)",
    )
    run_parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
        help="Configuration directory (default: configs)",
    )
    run_parser.add_argument(
        "--matrix-config",
        type=Path,
        help="Path to runmatrix.yaml file (default: configs/runmatrix.yaml)",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without executing tests",
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate test definitions"
    )
    validate_parser.add_argument(
        "--test",
        type=Path,
        required=True,
        help="Path to test definition file",
    )
    validate_parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
        help="Configuration directory (default: configs)",
    )

    # Status command
    status_parser = subparsers.add_parser("status", help="Check execution status")
    status_parser.add_argument(
        "--run",
        type=Path,
        help="Path to specific run directory (default: latest)",
    )
    status_parser.add_argument(
        "--list",
        action="store_true",
        help="List all available runs",
    )

    # Providers command
    providers_parser = subparsers.add_parser(
        "providers", help="List available providers"
    )
    providers_parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
        help="Configuration directory (default: configs)",
    )

    # Report command
    report_parser = subparsers.add_parser(
        "report", help="Generate reports from evaluation results"
    )
    report_parser.add_argument(
        "--results",
        type=Path,
        default=Path("results"),
        help="Results directory (default: results)",
    )
    report_parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for reports (default: results/reports)",
    )
    report_parser.add_argument(
        "--format",
        choices=["markdown", "json", "both"],
        default="both",
        help="Report format to generate (default: both)",
    )
    report_parser.add_argument(
        "--runs",
        nargs="+",
        help="Specific run directories to include",
    )
    report_parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Generate aggregated report across multiple runs",
    )
    report_parser.add_argument(
        "--latest",
        type=int,
        default=1,
        help="Number of latest runs to include (default: 1)",
    )

    # Prepare manual command
    prepare_manual_parser = subparsers.add_parser(
        "prepare-manual", help="Prepare manual evaluation prompts"
    )
    prepare_manual_parser.add_argument(
        "--provider",
        default="copilot_manual",
        help="Provider for manual evaluation (default: copilot_manual)",
    )
    prepare_manual_parser.add_argument(
        "--test-set",
        choices=["offline", "online", "all"],
        default="offline",
        help="Test set to prepare (default: offline)",
    )
    prepare_manual_parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="""Benchmarking CLI for the evaluation harness.

This script provides command-line interface for running benchmarks,
validating configurations, and managing evaluation results.""",
    )
    prepare_manual_parser.add_argument(
        "--output",
        type=Path,
        default=Path("manual_prompts.txt"),
        help="Output file for prompts (default: manual_prompts.txt)",
    )
    prepare_manual_parser.add_argument(
        "--format",
        choices=["plain", "markdown", "template"],
        default="template",
        help="Output format (default: template)",
    )

    # Process manual command
    process_manual_parser = subparsers.add_parser(
        "process-manual", help="Process manual evaluation results"
    )
    process_manual_parser.add_argument(
        "--responses",
        type=Path,
        required=True,
        help="File containing manual evaluation responses",
    )
    process_manual_parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for processed results",
    )
    process_manual_parser.add_argument(
        "--provider",
        default="copilot_manual",
        help="Provider name for results (default: copilot_manual)",
    )
    process_manual_parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
        help="Configuration directory (default: configs)",
    )

    # Analytics command
    analytics_parser = subparsers.add_parser(
        "analytics", help="Analytics and insights commands"
    )
    analytics_subparsers = analytics_parser.add_subparsers(
        dest="analytics_command", help="Analytics subcommands"
    )

    # Dashboard subcommand
    dashboard_parser = analytics_subparsers.add_parser(
        "dashboard", help="Start analytics dashboard"
    )
    dashboard_parser.add_argument(
        "--host", default="localhost", help="Dashboard host (default: localhost)"
    )
    dashboard_parser.add_argument(
        "--port", type=int, default=5000, help="Dashboard port (default: 5000)"
    )
    dashboard_parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode"
    )

    # Insights subcommand
    insights_parser = analytics_subparsers.add_parser(
        "insights", help="Generate AI insights"
    )
    insights_parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Results directory to analyze (default: results)",
    )
    insights_parser.add_argument(
        "--output", type=Path, help="Output file for insights (JSON format)"
    )
    insights_parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="Minimum confidence threshold for insights (default: 0.7)",
    )

    # Trends subcommand
    trends_parser = analytics_subparsers.add_parser(
        "trends", help="Analyze performance trends"
    )
    trends_parser.add_argument(
        "--provider", required=True, help="Provider to analyze trends for"
    )
    trends_parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Results directory to analyze (default: results)",
    )
    trends_parser.add_argument(
        "--output", type=Path, help="Output file for trend analysis (JSON format)"
    )

    # Compare subcommand
    compare_parser = analytics_subparsers.add_parser(
        "compare", help="Compare provider performance"
    )
    compare_parser.add_argument(
        "--providers", nargs="+", required=True, help="Providers to compare"
    )
    compare_parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Results directory to analyze (default: results)",
    )
    compare_parser.add_argument(
        "--statistical-test",
        choices=["auto", "ttest", "mannwhitney"],
        default="auto",
        help="Statistical test to use for comparison (default: auto)",
    )

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "run":
            handle_run_command(args, console)
        elif args.command == "validate":
            handle_validate_command(args, console)
        elif args.command == "status":
            handle_status_command(args, console)
        elif args.command == "providers":
            handle_providers_command(args, console)
        elif args.command == "report":
            handle_report_command(args, console)
        elif args.command == "prepare-manual":
            handle_prepare_manual_command(args, console)
        elif args.command == "process-manual":
            handle_process_manual_command(args, console)
        elif args.command == "analytics":
            handle_analytics_command(args, console)

    except (ConfigurationError, TestExecutionError) as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Execution cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)


def handle_run_command(args: argparse.Namespace, console: Console) -> None:
    """Handle the run command."""
    from bench.core.runner import TestRunner

    runner = TestRunner(config_dir=args.config_dir)

    if args.dry_run:
        console.print("[yellow]Dry run mode - validating configuration only[/yellow]")
        try:
            runner.context.load_configurations()
            console.print("[green]✓ Configuration validation passed[/green]")
            return
        except Exception as e:
            console.print(f"[red]✗ Configuration validation failed: {e}[/red]")
            return

    if args.matrix:
        console.print("[blue]Executing full test matrix...[/blue]")
        result = runner.run_matrix(args.matrix_config)

        console.print("\n[green]Matrix execution completed![/green]")
        console.print(f"Run directory: {result['run_directory']}")
        console.print(f"Total tests executed: {result['total_tests']}")

    else:
        # Single test execution
        if not args.test:
            console.print("[red]Error: --test is required when using --provider[/red]")
            sys.exit(1)

        console.print(f"[blue]Running single test with {args.provider}...[/blue]")
        result = runner.run_single(args.provider, args.test, args.repetitions)

        console.print("\n[green]Test execution completed![/green]")
        console.print(f"Run directory: {result['run_directory']}")
        console.print(f"Test: {result['test_id']}")
        console.print(f"Repetitions: {result['repetitions']}")


def handle_validate_command(args: argparse.Namespace, console: Console) -> None:
    """Handle the validate command."""
    from bench.core.runner import TestRunner

    runner = TestRunner(config_dir=args.config_dir)

    console.print(f"[blue]Validating test definition: {args.test}[/blue]")
    result = runner.validate_test(args.test)

    if result["valid"]:
        console.print("[green]✓ Test definition is valid[/green]")
        console.print(f"Test ID: {result['test_id']}")

        if result["warnings"]:
            console.print("\n[yellow]Warnings:[/yellow]")
            for warning in result["warnings"]:
                console.print(f"  • {warning}")
    else:
        console.print("[red]✗ Test definition is invalid[/red]")
        console.print(f"Test ID: {result['test_id']}")

        if result["errors"]:
            console.print("\n[red]Errors:[/red]")
            for error in result["errors"]:
                console.print(f"  • {error}")

        if result["warnings"]:
            console.print("\n[yellow]Warnings:[/yellow]")
            for warning in result["warnings"]:
                console.print(f"  • {warning}")


def handle_status_command(args: argparse.Namespace, console: Console) -> None:
    """Handle the status command."""
    results_dir = Path("results")

    if args.list:
        # List all runs
        if not results_dir.exists():
            console.print("[yellow]No results directory found[/yellow]")
            return

        runs = [
            d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run_")
        ]
        runs.sort(key=lambda x: x.name, reverse=True)

        if not runs:
            console.print("[yellow]No runs found[/yellow]")
            return

        table = Table(title="Available Runs")
        table.add_column("Run Directory", style="cyan")
        table.add_column("Timestamp", style="green")
        table.add_column("Status", style="yellow")

        for run_dir in runs[:10]:  # Show last 10 runs
            timestamp = run_dir.name.replace("run_", "").replace("_", " ")

            # Check if run completed successfully
            metadata_file = run_dir / "metadata" / "execution.json"
            status = "Completed" if metadata_file.exists() else "Incomplete"

            table.add_row(run_dir.name, timestamp, status)

        console.print(table)
        return

    # Show specific run status
    if args.run:
        run_dir = args.run
    else:
        # Use latest run
        latest_link = results_dir / "latest"
        if not latest_link.exists():
            console.print("[yellow]No runs found[/yellow]")
            return
        run_dir = latest_link.resolve()

    if not run_dir.exists():
        console.print(f"[red]Run directory not found: {run_dir}[/red]")
        return

    # Load and display run metadata
    metadata_file = run_dir / "metadata" / "execution.json"
    if not metadata_file.exists():
        console.print(f"[yellow]Run metadata not found in {run_dir}[/yellow]")
        return

    with open(metadata_file, encoding="utf-8") as f:
        metadata = json.load(f)

    console.print(f"[blue]Run Status: {run_dir.name}[/blue]")
    console.print(f"Timestamp: {metadata.get('timestamp', 'Unknown')}")

    if "provider" in metadata:
        # Single test run
        console.print(f"Provider: {metadata['provider']}")
        console.print(f"Test: {metadata['test_id']}")
        console.print(f"Repetitions: {metadata['repetitions']}")

        results = metadata.get("results", [])
        successful = sum(1 for r in results if r.get("success", False))
        console.print(f"Success rate: {successful}/{len(results)}")

    elif "matrix_config" in metadata:
        # Matrix run
        console.print(f"Total tests: {metadata['total_tests']}")
        console.print("Matrix execution completed")

    console.print(f"Run directory: {run_dir}")


def handle_providers_command(args: argparse.Namespace, console: Console) -> None:
    """Handle the providers command."""
    from bench.adapters import list_available_providers

    try:
        providers = list_available_providers()

        table = Table(title="Available Providers")
        table.add_column("Provider Name", style="cyan")
        table.add_column("Status", style="green")

        for provider in providers:
            table.add_row(provider, "Available")

        console.print(table)

    except Exception as e:
        console.print(f"[red]Failed to list providers: {e}[/red]")


def handle_report_command(args: argparse.Namespace, console: Console) -> None:
    """Handle the report command."""
    import sys
    from pathlib import Path

    # Add project root to path for imports
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    try:
        from bench.core.reporting import ConsolidatedReporter

        # Validate results directory
        if not args.results.exists():
            console.print(
                f"[red]Error: Results directory {args.results} does not exist[/red]"
            )
            return

        # Set default output directory
        if args.output is None:
            args.output = args.results / "reports"

        # Determine formats
        formats = ["markdown", "json"] if args.format == "both" else [args.format]

        # Create reporter
        reporter = ConsolidatedReporter(args.results, args.output)

        # Determine run directories
        if args.runs:
            run_dirs = [args.results / run_name for run_name in args.runs]
            # Validate specified runs exist
            for run_dir in run_dirs:
                if not run_dir.exists():
                    console.print(
                        f"[yellow]Warning: Run directory {run_dir} "
                        f"does not exist[/yellow]"
                    )
            run_dirs = [d for d in run_dirs if d.exists()]
        else:
            run_dirs = reporter._find_latest_runs(args.latest)

        if not run_dirs:
            console.print("[red]Error: No valid run directories found[/red]")
            return

        console.print(
            f"[green]Generating reports for {len(run_dirs)} run(s)...[/green]"
        )
        for run_dir in run_dirs:
            console.print(f"  - {run_dir.name}")

        # Generate reports
        generated_files = reporter.generate_reports(
            run_dirs=run_dirs, formats=formats, aggregate=args.aggregate
        )

        # Report success
        console.print(
            f"\n[green]Generated {len(generated_files)} report file(s):[/green]"
        )
        for report_type, file_path in generated_files.items():
            console.print(f"  - [cyan]{report_type}[/cyan]: {file_path}")

    except ImportError as e:
        console.print(f"[red]Error: Missing dependencies for reporting: {e}[/red]")
        console.print(
            "[yellow]Make sure the reporting system is properly installed[/yellow]"
        )
    except Exception as e:
        console.print(f"[red]Error generating reports: {e}[/red]")


def handle_analytics_command(args: argparse.Namespace, console: Console) -> None:
    """Handle analytics subcommands."""
    if not args.analytics_command:
        console.print("[red]Error: Analytics subcommand required[/red]")
        console.print("Available subcommands: dashboard, insights, trends, compare")
        return

    if args.analytics_command == "dashboard":
        try:
            from bench.web.app import create_app

            console.print(
                f"[green]Starting analytics dashboard at http://{args.host}:{args.port}[/green]"
            )
            console.print("[yellow]Press Ctrl+C to stop the server[/yellow]")

            app = create_app()
            app.run(host=args.host, port=args.port, debug=args.debug)
        except ImportError as e:
            console.print(
                f"[red]Error: Dashboard dependencies not available: {e}[/red]"
            )
        except Exception as e:
            console.print(f"[red]Error starting dashboard: {e}[/red]")

    elif args.analytics_command == "insights":
        try:
            from bench.analytics.insights import InsightsEngine

            console.print("[blue]Generating AI insights...[/blue]")

            engine = InsightsEngine()

            # Load evaluation history (placeholder - would load from results directory)
            evaluation_history: list[dict[str, Any]] = []
            provider_configs: dict[str, Any] = {}

            insights = engine.generate_insights(evaluation_history, provider_configs)

            # Filter by confidence
            filtered_insights = [
                insight
                for insight in insights
                if insight.confidence >= args.min_confidence
            ]

            if args.output:
                # Convert insights to dict for JSON serialization
                insights_data = [
                    {
                        "title": insight.title,
                        "description": insight.description,
                        "category": insight.category,
                        "severity": insight.severity,
                        "confidence": insight.confidence,
                        "recommendations": insight.recommendations,
                        "evidence": insight.evidence,
                        "timestamp": insight.timestamp.isoformat(),
                    }
                    for insight in filtered_insights
                ]

                with open(args.output, "w") as f:
                    json.dump(insights_data, f, indent=2)
                console.print(f"[green]Insights saved to {args.output}[/green]")
            else:
                # Print to console
                if not filtered_insights:
                    console.print(
                        "[yellow]No insights generated with the specified "
                        "confidence threshold[/yellow]"
                    )
                    return

                for insight in filtered_insights:
                    severity_color = {
                        "critical": "red",
                        "warning": "yellow",
                        "info": "blue",
                    }.get(insight.severity, "white")

                    console.print(
                        f"\n[{severity_color}][{insight.severity.upper()}] "
                        f"{insight.title}[/{severity_color}]"
                    )
                    console.print(f"Confidence: {insight.confidence:.2f}")
                    console.print(f"Description: {insight.description}")
                    if insight.recommendations:
                        console.print("Recommendations:")
                        for rec in insight.recommendations[:3]:
                            console.print(f"  • {rec}")

        except ImportError as e:
            console.print(
                f"[red]Error: Analytics dependencies not available: {e}[/red]"
            )
        except Exception as e:
            console.print(f"[red]Error generating insights: {e}[/red]")

    elif args.analytics_command == "trends":
        try:
            console.print(
                f"[blue]Analyzing trends for provider: {args.provider}[/blue]"
            )
            console.print(f"Results directory: {args.results_dir}")

            # TODO: Load evaluation data for the specified provider
            # Placeholder for trend analysis
            trends = {
                "provider": args.provider,
                "trend_analysis": "Not implemented yet - requires integration "
                "with evaluation data",
                "message": "Trend analysis will be implemented when integrated "
                "with evaluation results storage",
            }

            if args.output:
                with open(args.output, "w") as f:
                    json.dump(trends, f, indent=2)
                console.print(f"[green]Trend analysis saved to {args.output}[/green]")
            else:
                console.print(json.dumps(trends, indent=2))

        except ImportError as e:
            console.print(
                f"[red]Error: Analytics dependencies not available: {e}[/red]"
            )
        except Exception as e:
            console.print(f"[red]Error analyzing trends: {e}[/red]")

    elif args.analytics_command == "compare":
        try:
            console.print(
                f"[blue]Comparing providers: {', '.join(args.providers)}[/blue]"
            )
            console.print(f"Results directory: {args.results_dir}")
            console.print(f"Statistical test: {args.statistical_test}")

            # TODO: Load evaluation data for comparison
            # Placeholder for comparison analysis
            comparison = {
                "providers": args.providers,
                "statistical_test": args.statistical_test,
                "comparison_results": "Not implemented yet - requires integration "
                "with evaluation data",
                "message": "Provider comparison will be implemented when integrated "
                "with evaluation results storage",
            }

            console.print(json.dumps(comparison, indent=2))

        except ImportError as e:
            console.print(
                f"[red]Error: Analytics dependencies not available: {e}[/red]"
            )
        except Exception as e:
            console.print(f"[red]Error comparing providers: {e}[/red]")


if __name__ == "__main__":
    main()
