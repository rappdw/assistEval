#!/usr/bin/env python3
"""Command-line interface for the evaluation harness.

Provides CLI commands for running evaluations, generating reports,
and managing test configurations with comprehensive argument parsing.
"""

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table


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
  python scripts/run_bench.py run --matrix

  # Run single test with specific provider
  python scripts/run_bench.py run --provider chatgpt \
    --test bench/tests/offline/task1_metrics.yaml

  # Validate test definition
  python scripts/run_bench.py validate \
    --test bench/tests/offline/task1_metrics.yaml

  # Check execution status
  python scripts/run_bench.py status --run results/run_20250827_160000
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


if __name__ == "__main__":
    main()
