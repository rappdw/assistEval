# Stage 9 Implementation Plan: Reporting System

**Project**: ChatGPT vs Microsoft Copilot Evaluation Harness
**Stage**: 9 - Reporting System
**Priority**: Medium
**Estimated Effort**: 3-4 hours
**Dependencies**: Stage 8 (Scoring & Aggregation)

## Overview

Stage 9 implements the comprehensive reporting system that transforms scoring results into human-readable and machine-processable outputs. This stage creates markdown reports, JSON summaries, leaderboards, and detailed failure analysis to provide clear insights into provider performance comparisons.

## Objectives

- **Markdown Report Generation**: Human-readable reports with formatting and structure
- **JSON Report Structure**: Machine-processable data for automation and integration
- **Leaderboard Creation**: Provider rankings with score comparisons
- **Detailed Failure Analysis**: Per-task breakdowns with specific error explanations
- **Multi-Run Aggregation**: Consolidated reporting across multiple test executions
- **Artifact Management**: Report storage and retrieval system

## Architecture Position

Stage 9 sits at the end of the evaluation pipeline:
- **Input**: `ProviderScore` objects from Stage 8 scoring system
- **Processing**: Report formatting, aggregation, and analysis
- **Output**: Markdown files, JSON reports, and consolidated summaries

## Implementation Tasks

### Core Reporting System (`bench/core/reporting.py`)

#### 1. Report Data Structures
```python
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
```

#### 2. Markdown Report Generator
```python
class MarkdownReportGenerator:
    """Generate human-readable markdown reports."""

    def __init__(self, template_dir: Optional[Path] = None):
        self.template_dir = template_dir or Path(__file__).parent / "templates"

    def generate_report(
        self,
        provider_scores: dict[str, ProviderScore],
        run_metadata: dict[str, Any]
    ) -> str:
        """Generate complete markdown report."""

    def _generate_executive_summary(self, scores: dict[str, ProviderScore]) -> str:
        """Create high-level summary section."""

    def _generate_leaderboard(self, scores: dict[str, ProviderScore]) -> str:
        """Create provider ranking table."""

    def _generate_task_breakdown(self, scores: dict[str, ProviderScore]) -> str:
        """Create detailed per-task analysis."""

    def _generate_stability_analysis(self, scores: dict[str, ProviderScore]) -> str:
        """Create multi-run consistency analysis."""

    def _generate_failure_analysis(self, scores: dict[str, ProviderScore]) -> str:
        """Create detailed error and warning breakdown."""
```

#### 3. JSON Report Generator
```python
class JSONReportGenerator:
    """Generate machine-processable JSON reports."""

    def generate_report(
        self,
        provider_scores: dict[str, ProviderScore],
        run_metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate structured JSON report."""

    def _create_summary_section(self, scores: dict[str, ProviderScore]) -> dict:
        """Create summary statistics."""

    def _create_provider_section(self, scores: dict[str, ProviderScore]) -> dict:
        """Create per-provider detailed results."""

    def _create_task_section(self, scores: dict[str, ProviderScore]) -> dict:
        """Create per-task comparison matrix."""

    def _create_metadata_section(self, metadata: dict[str, Any]) -> dict:
        """Create execution metadata."""
```

#### 4. Report Aggregator
```python
class ReportAggregator:
    """Aggregate multiple runs into consolidated reports."""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir

    def aggregate_runs(
        self,
        run_patterns: list[str] = None
    ) -> dict[str, Any]:
        """Aggregate multiple run results."""

    def _load_run_data(self, run_dir: Path) -> dict[str, Any]:
        """Load data from a single run directory."""

    def _calculate_trends(self, historical_data: list[dict]) -> dict[str, Any]:
        """Calculate performance trends over time."""

    def _identify_regressions(self, historical_data: list[dict]) -> list[dict]:
        """Identify performance regressions."""
```

### Report Templates (`bench/core/templates/`)

#### 1. Markdown Report Template
```markdown
# AssistEval Benchmark Report

**Generated**: {timestamp}
**Execution Time**: {execution_time}s
**Providers Tested**: {provider_count}
**Tasks Evaluated**: {task_count}

## Executive Summary

{executive_summary}

## Leaderboard

{leaderboard_table}

## Task-by-Task Breakdown

{task_breakdown}

## Stability Analysis

{stability_analysis}

## Detailed Results

{detailed_results}

## Failure Analysis

{failure_analysis}
```

#### 2. HTML Report Template (Optional Enhancement)
```html
<!DOCTYPE html>
<html>
<head>
    <title>AssistEval Benchmark Report</title>
    <style>
        /* Modern, responsive CSS styling */
    </style>
</head>
<body>
    <!-- Interactive charts and tables -->
</body>
</html>
```

### Consolidated Reporting Script (`scripts/make_report.py`)

#### 1. CLI Interface
```python
def main():
    parser = argparse.ArgumentParser(description="Generate consolidated benchmark reports")
    parser.add_argument("--results", type=Path, required=True, help="Results directory")
    parser.add_argument("--output", type=Path, help="Output directory for reports")
    parser.add_argument("--format", choices=["markdown", "json", "both"], default="both")
    parser.add_argument("--runs", nargs="+", help="Specific run directories to include")
    parser.add_argument("--template", type=Path, help="Custom template directory")
    parser.add_argument("--aggregate", action="store_true", help="Generate aggregated report")

    args = parser.parse_args()

    # Implementation
```

#### 2. Report Generation Logic
```python
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
        run_dirs: list[Path] = None,
        formats: list[str] = ["markdown", "json"],
        aggregate: bool = False
    ) -> dict[str, Path]:
        """Generate all requested report formats."""

    def _find_latest_runs(self, count: int = 1) -> list[Path]:
        """Find most recent run directories."""

    def _validate_run_directory(self, run_dir: Path) -> bool:
        """Validate run directory contains required files."""
```

### Report Utilities (`bench/core/report_utils.py`)

#### 1. Formatting Helpers
```python
def format_score_table(scores: dict[str, float], max_width: int = 80) -> str:
    """Format scores into aligned table."""

def format_percentage(value: float, decimals: int = 1) -> str:
    """Format percentage with proper rounding."""

def format_duration(seconds: float) -> str:
    """Format execution duration in human-readable form."""

def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text with ellipsis."""
```

#### 2. Chart Generation (Optional)
```python
def generate_score_chart(
    scores: dict[str, float],
    output_path: Path,
    chart_type: str = "bar"
) -> Path:
    """Generate score comparison chart."""

def generate_trend_chart(
    historical_data: list[dict],
    output_path: Path
) -> Path:
    """Generate performance trend chart."""
```

## Report Structure Specification

### Markdown Report Sections

1. **Executive Summary**
   - Overall winner and score
   - Key performance highlights
   - Notable failures or issues
   - Execution metadata

2. **Leaderboard**
   - Provider rankings by total score
   - Score percentages and raw scores
   - Stability bonus inclusion
   - Score differences between providers

3. **Task-by-Task Breakdown**
   - Per-task winner and scores
   - Sub-component score details
   - Weight application explanation
   - Task-specific failure reasons

4. **Stability Analysis**
   - Multi-run consistency metrics
   - Variance analysis across runs
   - Stability bonus calculation
   - Reliability assessment

5. **Detailed Results**
   - Complete score matrices
   - Raw evaluation results
   - Configuration snapshots
   - Execution timing

6. **Failure Analysis**
   - Categorized error types
   - Provider-specific issues
   - Task-specific problems
   - Recommendations for improvement

### JSON Report Structure

```json
{
  "metadata": {
    "timestamp": "2025-08-27T21:19:28-06:00",
    "version": "1.0.0",
    "execution_time": 45.2,
    "run_id": "run_20250827_211928"
  },
  "summary": {
    "total_providers": 2,
    "total_tasks": 3,
    "overall_winner": "chatgpt",
    "score_spread": 12.5,
    "stability_runs": 3
  },
  "leaderboard": [
    {
      "provider": "chatgpt",
      "total_score": 87.5,
      "max_score": 105.0,
      "percentage": 83.3,
      "rank": 1
    }
  ],
  "tasks": {
    "offline.task1.metrics_csv": {
      "max_score": 40.0,
      "winner": "chatgpt",
      "scores": {
        "chatgpt": 38.0,
        "copilot": 32.5
      },
      "details": {}
    }
  },
  "providers": {
    "chatgpt": {
      "total_score": 87.5,
      "task_scores": [],
      "stability_bonus": 5.0,
      "errors": [],
      "warnings": []
    }
  }
}
```

## Integration Points

### Stage 8 Integration
- Load `ProviderScore` objects from scoring system
- Access detailed task breakdowns and error information
- Retrieve stability analysis results

### Results Directory Structure
```
results/
  run_20250827_211928/
    config/
      providers.yaml
      weights.yaml
      runmatrix.yaml
    raw/
      chatgpt/
        task1.txt
        task2.txt
    parsed/
      chatgpt/
        task1.json
        task2.json
    scores/
      chatgpt_score.json
      copilot_score.json
    reports/
      summary.md
      detailed.json
      leaderboard.html
```

### CLI Integration
- Extend `scripts/run_bench.py` with `--report` flag
- Automatic report generation after successful runs
- Integration with existing artifact management

## Testing Strategy

### Unit Tests (`tests/test_reporting.py`)
```python
class TestMarkdownReportGenerator:
    """Test markdown report generation."""

    def test_generate_executive_summary(self):
        """Test executive summary generation."""

    def test_generate_leaderboard_table(self):
        """Test leaderboard formatting."""

    def test_generate_task_breakdown(self):
        """Test task analysis section."""

class TestJSONReportGenerator:
    """Test JSON report generation."""

    def test_report_structure_compliance(self):
        """Test JSON structure matches schema."""

    def test_score_aggregation_accuracy(self):
        """Test score calculations in reports."""

class TestReportAggregator:
    """Test multi-run aggregation."""

    def test_aggregate_multiple_runs(self):
        """Test aggregation across runs."""

    def test_trend_calculation(self):
        """Test trend analysis."""
```

### Integration Tests (`tests/test_reporting_integration.py`)
```python
class TestReportingIntegration:
    """Test end-to-end reporting workflow."""

    def test_full_report_generation(self):
        """Test complete report generation from scores."""

    def test_consolidated_reporting(self):
        """Test make_report.py script functionality."""

    def test_report_artifact_management(self):
        """Test report storage and retrieval."""
```

## Configuration

### Report Configuration (`configs/reporting.yaml`)
```yaml
reporting:
  formats:
    - markdown
    - json

  markdown:
    template: "default"
    include_charts: false
    max_line_length: 80

  json:
    pretty_print: true
    include_raw_data: false

  aggregation:
    max_runs: 10
    trend_analysis: true
    regression_detection: true

  output:
    directory: "reports"
    filename_pattern: "{timestamp}_{format}"
    archive_old_reports: true
```

## Error Handling

### Report Generation Errors
- Missing score data handling
- Template rendering failures
- File system permission issues
- Invalid run directory structures

### Data Validation
- Score data consistency checks
- Metadata validation
- Template variable validation
- Output format compliance

## Performance Considerations

### Large Dataset Handling
- Streaming JSON generation for large reports
- Pagination for extensive task lists
- Memory-efficient aggregation algorithms
- Lazy loading of historical data

### Report Optimization
- Template caching for repeated generation
- Incremental report updates
- Compressed report storage
- Fast lookup indices for aggregation

## Success Criteria

### Functional Requirements
- [ ] Generate readable markdown reports with all sections
- [ ] Produce valid JSON reports matching schema
- [ ] Create accurate leaderboards and comparisons
- [ ] Provide detailed failure analysis
- [ ] Support multi-run aggregation
- [ ] Integrate seamlessly with scoring system

### Quality Requirements
- [ ] Reports are human-readable and well-formatted
- [ ] JSON structure is consistent and machine-processable
- [ ] All calculations are accurate and verifiable
- [ ] Error handling is robust and informative
- [ ] Performance is acceptable for typical dataset sizes
- [ ] Templates are customizable and extensible

### Integration Requirements
- [ ] Works with existing results directory structure
- [ ] Integrates with CLI and runner system
- [ ] Supports both single-run and consolidated reporting
- [ ] Maintains compatibility with Stage 8 scoring output
- [ ] Provides clear API for programmatic access

## Future Enhancements

### Phase 2 Features
- Interactive HTML reports with charts
- Real-time report updates during execution
- Email/Slack notification integration
- Custom report templates and themes

### Phase 3 Features
- Web dashboard for historical analysis
- API endpoints for report data access
- Advanced analytics and insights
- Automated report scheduling and distribution

## Dependencies

### External Libraries
```toml
[tool.poetry.dependencies]
jinja2 = "^3.1.0"  # Template rendering
markdown = "^3.5.0"  # Markdown processing
plotly = "^5.17.0"  # Optional chart generation
```

### Internal Dependencies
- Stage 8: `ProviderScore` objects and scoring results
- Stage 5: Results directory structure and metadata
- Stage 3: Configuration system integration

## Deliverables

1. **Core Reporting System** (`bench/core/reporting.py`)
   - MarkdownReportGenerator class
   - JSONReportGenerator class
   - ReportAggregator class
   - Report data structures

2. **Consolidated Reporting Script** (`scripts/make_report.py`)
   - CLI interface for report generation
   - Multi-run aggregation support
   - Flexible output format options

3. **Report Templates** (`bench/core/templates/`)
   - Markdown report template
   - JSON schema definitions
   - Customizable formatting options

4. **Comprehensive Test Suite**
   - Unit tests for all report generators
   - Integration tests for end-to-end workflow
   - Template rendering validation tests

5. **Documentation**
   - Report format specifications
   - Template customization guide
   - CLI usage examples
   - Integration instructions

This implementation provides a complete reporting system that transforms scoring results into actionable insights while maintaining flexibility for future enhancements and customizations.
