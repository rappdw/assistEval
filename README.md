# ChatGPT vs Microsoft Copilot Evaluation Harness

A fair, extensible, and repeatable evaluation framework for comparing AI assistants on offline and online tasks with objective scoring.

[![CI](https://github.com/your-org/assistEval/workflows/CI/badge.svg)](https://github.com/your-org/assistEval/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

This repository provides an open, reproducible harness to compare general-purpose AI assistants (initially Microsoft Copilot vs ChatGPT) on well-specified tasks. The framework enforces strict constraints, produces objective numeric scores, and generates detailed reports.

### Key Features

- **Declarative Test Definitions**: Add/modify test cases via YAML/JSON without code changes
- **Multiple Provider Support**: ChatGPT API integration + manual paste mode for other assistants
- **Objective Scoring**: Numeric metrics with configurable weights and tolerances
- **Comprehensive Validation**: Schema validation, structural checks, and content normalization
- **Modular Architecture**: Pluggable evaluators and validators for easy extension

## Quick Start

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd assistEval

# Install dependencies with uv
uv sync --all-extras

# Install pre-commit hooks (optional but recommended)
uv run pre-commit install
```

### Basic Usage

```bash
# Run full evaluation matrix
uv run python scripts/bench.py run --config configs/runmatrix.yaml --weights configs/weights.default.yaml

# Run a single test for ChatGPT
uv run python scripts/bench.py run --provider chatgpt --test tests/offline/task1_metrics.yaml

# Generate consolidated report
uv run python scripts/make_report.py --results results/
```

## Project Structure

```
bench/                          # Main package
├── adapters/                   # Provider implementations
│   ├── base.py                # Provider interface
│   ├── chatgpt.py             # ChatGPT API adapter
│   └── copilot_manual.py      # Manual paste adapter
├── core/                      # Core evaluation logic
│   ├── runner.py              # Orchestrates evaluation runs
│   ├── validators.py          # JSON schema and structure validation
│   ├── evaluators/            # Task-specific evaluators
│   ├── scoring.py             # Score aggregation and weighting
│   ├── reporting.py           # Report generation
│   └── utils.py               # Utilities and helpers
└── tests/                     # Test definitions
    ├── offline/               # Offline task definitions
    └── online/                # Online task definitions (optional)

configs/                       # Configuration files
├── providers.yaml             # Provider settings
├── weights.default.yaml       # Default scoring weights
└── runmatrix.yaml            # Test execution matrix

fixtures/                      # Test data and fixtures
├── csv/                      # CSV data files
└── text/                     # Text data files

answer_keys/                   # Expected outputs for validation
├── offline/                  # Offline task answer keys
└── online/                   # Online task answer keys

results/                       # Evaluation results
└── run_<timestamp>/          # Individual run results
```

## Development Setup

### Environment Setup

```bash
# Create and activate virtual environment (handled by uv)
uv sync --all-extras

# Install pre-commit hooks
uv run pre-commit install

# Verify installation
uv run pytest --version
uv run black --version
uv run ruff --version
uv run mypy --version
```

### Development Workflow

```bash
# Run tests
uv run pytest

# Format code
uv run black .
uv run isort .

# Lint code
uv run ruff check .
uv run flake8 .

# Type checking
uv run mypy .

# Run all quality checks
uv run pre-commit run --all-files
```

### Code Quality Standards

This project enforces strict code quality standards:

- **Formatting**: Black with 88-character line length
- **Import Sorting**: isort with Black profile
- **Linting**: Ruff with comprehensive rule set
- **Type Checking**: mypy in strict mode
- **Testing**: pytest with coverage reporting
- **Pre-commit Hooks**: Automated quality checks on commit

## Test Categories

### Offline Tasks

1. **Metrics from CSV** (40 pts): Calculate precision, recall, F1, accuracy from phishing detection data
2. **DLP Regex for SSN** (30 pts): Create regex pattern for U.S. Social Security Numbers
3. **Executive Summary** (20 pts): Generate structured summary with word count and tone constraints

### Online Tasks (Optional)

4. **Deep Research** (10 pts): Multi-step research plan with source validation and risk assessment

## Configuration

### Provider Configuration

Edit `configs/providers.yaml` to configure AI providers:

```yaml
providers:
  - name: chatgpt
    adapter: chatgpt
    options:
      temperature: 0
      max_tokens: 1200
      seed: 42
      response_format: json
  - name: copilot
    adapter: copilot_manual
    options: {}
```

### Scoring Weights

Customize scoring in `configs/weights.default.yaml`:

```yaml
weights:
  task1_metrics: 40
  task2_ssn_regex: 30
  task3_exec_summary: 20
  deep_research: 10
stability_bonus: 5
```

## Adding New Tests

1. Copy a template from `tests/_templates/`
2. Add fixtures to `fixtures/` directory
3. Create answer key in `answer_keys/`
4. Validate: `uv run python scripts/bench.py validate --test tests/your_test.yaml`

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes following the code quality standards
4. Run tests and quality checks: `uv run pre-commit run --all-files`
5. Commit changes: `git commit -m "Description"`
6. Push to branch: `git push origin feature-name`
7. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Roadmap

- **Phase 1**: Offline tasks with ChatGPT and manual Copilot support
- **Phase 2**: Online research tasks and stability analysis
- **Phase 3**: Additional providers and expanded task library

## Support

For questions, issues, or contributions, please:

1. Check existing [Issues](https://github.com/your-org/assistEval/issues)
2. Create a new issue with detailed description
3. Follow the contributing guidelines above
