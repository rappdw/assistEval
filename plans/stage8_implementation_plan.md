# Stage 8 Implementation Plan: Scoring & Aggregation

**Project**: ChatGPT vs Microsoft Copilot Evaluation Harness
**Stage**: 8 - Scoring & Aggregation
**Priority**: Medium
**Estimated Effort**: 2-3 hours
**Dependencies**: Stage 7 (Offline Task Evaluators)

## Overview

Stage 8 implements the weighted score aggregation system that combines individual evaluator results into final provider scores. This stage transforms the detailed evaluation results from Stage 7 into the standardized 105-point scoring rubric defined in the specification.

## Objectives

- **Weighted Score Aggregation**: Combine individual task scores using configurable weights
- **Stability Bonus Framework**: Multi-run consistency analysis for offline tasks
- **Score Validation**: Bounds checking and normalization
- **Detailed Breakdowns**: Per-task and sub-component score reporting
- **Score Persistence**: Save and load scoring results for reporting

## Architecture Position

Stage 8 sits between Stage 7 (Offline Task Evaluators) and Stage 9 (Reporting System):
- **Input**: `EvaluationResult` objects from individual task evaluators
- **Processing**: Weight application, aggregation, stability analysis
- **Output**: `ProviderScore` objects with detailed breakdowns for reporting

## Implementation Tasks

### Core Scoring System (`bench/core/scoring.py`)

#### 1. Score Data Structures
```python
@dataclass
class TaskScore:
    """Individual task scoring result."""
    task_id: str
    evaluator_name: str
    raw_score: float
    max_score: float
    weighted_score: float
    weight: float
    sub_scores: dict[str, float]
    details: dict[str, Any]
    errors: list[str]
    warnings: list[str]

@dataclass
class ProviderScore:
    """Complete provider scoring result."""
    provider_name: str
    total_score: float
    max_score: float
    score_percentage: float
    task_scores: dict[str, TaskScore]
    stability_bonus: float
    metadata: dict[str, Any]
    timestamp: datetime
```

#### 2. Scoring Engine
```python
class ScoringEngine:
    """Aggregates evaluation results into weighted provider scores."""

    def __init__(self, weights_config: dict[str, Any]):
        """Initialize with weight configuration from weights.default.yaml."""

    def calculate_provider_score(
        self,
        evaluation_results: list[EvaluationResult],
        provider_name: str
    ) -> ProviderScore:
        """Calculate weighted total score for a provider."""

    def calculate_task_score(
        self,
        evaluation_result: EvaluationResult,
        task_weights: dict[str, float]
    ) -> TaskScore:
        """Calculate weighted score for individual task."""

    def validate_scores(self, provider_score: ProviderScore) -> list[str]:
        """Validate score bounds and consistency."""
```

#### 3. Weight Configuration Loading
```python
class WeightConfig:
    """Manages scoring weight configuration."""

    @classmethod
    def load_from_file(cls, weights_path: Path) -> "WeightConfig":
        """Load weights from YAML configuration."""

    def get_task_weights(self, task_id: str) -> dict[str, float]:
        """Get weight configuration for specific task."""

    def get_total_possible_score(self) -> float:
        """Calculate maximum possible score (105 points)."""
```

### Stability Bonus System

#### 4. Multi-Run Analysis
```python
class StabilityAnalyzer:
    """Analyzes consistency across multiple runs for stability bonus."""

    def calculate_stability_bonus(
        self,
        multi_run_results: dict[str, list[EvaluationResult]]
    ) -> float:
        """Calculate 0-5 point stability bonus based on consistency."""

    def analyze_task1_consistency(
        self,
        results: list[EvaluationResult]
    ) -> float:
        """Check Task 1 numeric consistency (exact matches)."""

    def analyze_structural_consistency(
        self,
        results: list[EvaluationResult]
    ) -> float:
        """Check Task 2/3 structural consistency (no schema failures)."""
```

### Score Persistence

#### 5. Score Storage and Loading
```python
class ScoreManager:
    """Manages score persistence and retrieval."""

    def save_provider_score(
        self,
        provider_score: ProviderScore,
        results_dir: Path
    ) -> None:
        """Save provider score to JSON file."""

    def load_provider_scores(
        self,
        results_dir: Path
    ) -> dict[str, ProviderScore]:
        """Load all provider scores from results directory."""

    def get_score_history(
        self,
        provider_name: str,
        results_base_dir: Path
    ) -> list[ProviderScore]:
        """Get historical scores for trending analysis."""
```

## Scoring Rubric Implementation

### Weight Configuration (105 Total Points)

Based on `configs/weights.default.yaml`:

```yaml
# Task 1: Metrics from CSV (40 points)
task1_metrics:
  precision: 6      # ±0.0005 tolerance
  recall: 6         # ±0.0005 tolerance
  f1: 6             # ±0.0005 tolerance
  accuracy: 6       # ±0.0005 tolerance
  confusion_matrix: # Exact integer match required
    tp: 3
    fp: 3
    fn: 3
    tn: 3

# Task 2: SSN Regex (30 points)
task2_ssn_regex:
  regex_validity: 18  # Anchors, constraints, no catastrophic backtracking
  line_matches: 12    # 1 point per correct line match

# Task 3: Executive Summary (20 points)
task3_exec_summary:
  structure: 12       # Title ≤6 words, 120-160 word count, 3 bullets, JSON compliance
  tone: 8            # Hype term detection, sentence length, professional tone

# Deep Research (10 points) - Optional online task
deep_research:
  plan_quality: 5     # 7-10 steps with goal/method/deliverable
  source_quality: 5   # 5-8 sources with ≥3 recent (within 3 years)

# Stability Bonus (5 points)
stability_bonus:
  max_points: 5
  consistency_threshold: 0.95  # 95% consistency required for full bonus
```

### Scoring Logic

#### Task Score Calculation
1. **Load evaluation result** from individual evaluator
2. **Apply task-specific weights** to sub-scores
3. **Calculate weighted total** for the task
4. **Validate bounds** (0 ≤ score ≤ max_score)
5. **Record detailed breakdown** for reporting

#### Provider Score Aggregation
1. **Sum all weighted task scores**
2. **Add stability bonus** (if multi-run data available)
3. **Calculate percentage** (score / 105 * 100)
4. **Validate total bounds** (0 ≤ total ≤ 105)
5. **Generate metadata** (timestamp, configuration hash)

#### Stability Bonus Calculation
- **Task 1**: Exact numeric consistency across 3 runs
- **Task 2/3**: No schema validation failures across runs
- **Scoring**: Linear scale from 0-5 points based on consistency percentage
- **Threshold**: 95% consistency required for full 5-point bonus

## Integration Points

### With Stage 7 (Evaluators)
- **Input**: `EvaluationResult` objects from task evaluators
- **Processing**: Extract sub-scores and apply weights
- **Validation**: Ensure all expected fields are present

### With Stage 5 (Core Runner)
- **Integration**: Runner calls scoring engine after evaluation
- **Multi-run**: Collect results across repetitions for stability analysis
- **Storage**: Save scores alongside raw results and parsed data

### With Stage 9 (Reporting)
- **Output**: `ProviderScore` objects for report generation
- **Breakdown**: Detailed task and sub-component scores
- **Comparison**: Provider-vs-provider score analysis

## Error Handling

### Score Validation
- **Bounds checking**: Ensure 0 ≤ score ≤ max_score for all components
- **Weight validation**: Verify weights sum to expected totals
- **Missing data**: Handle incomplete evaluation results gracefully

### Stability Analysis
- **Insufficient runs**: Partial bonus for < 3 runs
- **Mixed results**: Handle cases with some failed evaluations
- **Statistical significance**: Basic variance analysis

## Testing Strategy

### Unit Tests
- **Weight loading**: Test YAML configuration parsing
- **Score calculation**: Verify arithmetic with known inputs
- **Bounds validation**: Test edge cases and invalid inputs
- **Stability bonus**: Test consistency analysis logic

### Integration Tests
- **End-to-end scoring**: Full pipeline from evaluation to final score
- **Multi-provider**: Verify comparative scoring
- **Multi-run stability**: Test bonus calculation with real data

### Test Data
- **Mock evaluation results**: Controlled inputs for deterministic testing
- **Edge cases**: Missing fields, invalid scores, boundary conditions
- **Real scenarios**: Integration with actual evaluator outputs

## File Structure

```
bench/core/scoring.py           # Main scoring engine
bench/core/stability.py         # Multi-run consistency analysis
tests/test_scoring.py          # Comprehensive unit tests
tests/test_stability.py        # Stability bonus testing
tests/fixtures/scoring/        # Test data and mock results
```

## Success Criteria

- [ ] **Weight Configuration**: Load and validate weights from YAML
- [ ] **Task Score Calculation**: Apply weights to evaluation results correctly
- [ ] **Provider Score Aggregation**: Sum weighted scores with bounds checking
- [ ] **Stability Bonus**: Calculate consistency bonus for multi-run scenarios
- [ ] **Score Persistence**: Save and load provider scores reliably
- [ ] **Integration**: Seamless integration with Stage 7 evaluators and Stage 5 runner
- [ ] **Validation**: Comprehensive error handling and bounds checking
- [ ] **Testing**: 100% test coverage with unit and integration tests

## Dependencies

### Required Stages
- **Stage 7**: Offline Task Evaluators (provides `EvaluationResult` objects)
- **Stage 5**: Core Runner (orchestrates scoring integration)
- **Stage 3**: Schema & Configuration (provides weight configuration)

### External Dependencies
- **PyYAML**: Configuration file parsing
- **Pydantic**: Data validation and serialization
- **datetime**: Timestamp management
- **pathlib**: File system operations

## Deliverables

1. **Scoring Engine** (`bench/core/scoring.py`)
   - `ScoringEngine` class with weight-based aggregation
   - `TaskScore` and `ProviderScore` data structures
   - Score validation and bounds checking

2. **Stability Analysis** (`bench/core/stability.py`)
   - `StabilityAnalyzer` for multi-run consistency
   - Bonus calculation based on 95% consistency threshold
   - Statistical analysis utilities

3. **Score Management**
   - `ScoreManager` for persistence and retrieval
   - JSON serialization with metadata
   - Historical score tracking

4. **Integration Layer**
   - Runner integration for automatic scoring
   - Evaluator result processing
   - Configuration loading and validation

5. **Comprehensive Testing**
   - Unit tests for all scoring components
   - Integration tests with mock evaluation data
   - Edge case and error condition testing

## Next Steps

After Stage 8 completion:
- **Stage 9**: Reporting System will consume `ProviderScore` objects
- **Stage 10**: Sample Tests will provide real evaluation data for scoring
- **Stage 11**: CI/CD will include scoring validation in automated tests

## Notes

- **Deterministic Scoring**: All calculations must be reproducible
- **Extensible Design**: Easy to add new tasks or modify weights
- **Performance**: Efficient aggregation for large result sets
- **Audit Trail**: Complete score breakdown for transparency
- **Configuration-Driven**: No hardcoded weights or scoring logic
