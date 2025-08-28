"""Command line interface entry point for the evaluation harness."""

import sys
from pathlib import Path


def main() -> None:
    """Entry point that delegates to the actual CLI implementation."""
    # Add the scripts directory to the path so we can import the CLI
    scripts_dir = Path(__file__).parent.parent / "scripts"
    sys.path.insert(0, str(scripts_dir))

    # Import and run the actual CLI
    try:
        import subprocess

        bench_script = scripts_dir / "bench.py"
        result = subprocess.run([sys.executable, str(bench_script)] + sys.argv[1:])
        sys.exit(result.returncode)
    except Exception as e:
        import logging

        logging.error(f"Error running CLI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
