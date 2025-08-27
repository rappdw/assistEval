"""Command line interface for the evaluation harness."""

from typing import Any

import click


@click.group()
@click.version_option(version="0.1.0")
def main() -> Any:
    """ChatGPT vs Microsoft Copilot Evaluation Harness."""
    pass


@main.command()
def run() -> None:
    """Run evaluation benchmarks."""
    click.echo("Benchmark runner not yet implemented.")


@main.command()
def validate() -> None:
    """Validate test definitions."""
    click.echo("Test validator not yet implemented.")


if __name__ == "__main__":
    main()
