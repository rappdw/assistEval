#!/usr/bin/env python3
"""Command-line interface for the evaluation harness.

Provides CLI commands for running evaluations, generating reports,
and managing test configurations with comprehensive argument parsing.
"""

import sys
from pathlib import Path


def main() -> None:
    """Main entry point for the CLI."""
    # Add the project root to Python path for imports
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    # Import after path setup
    from bench.cli import main as cli_main

    # TODO: Implement CLI in Stage 5 - Core Runner & CLI
    # - Add argparse configuration for all commands
    # - Implement run, report, validate subcommands
    # - Add configuration file loading
    # - Support matrix execution and single test runs
    # - Provide progress indicators and logging
    # - Add validate command for test definitions
    # - Add configuration loading and validation
    # - Add progress reporting and logging

    cli_main()


if __name__ == "__main__":
    main()
