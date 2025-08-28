# Stage 7: Offline Task Evaluators Implementation Plan

**Project**: ChatGPT vs Microsoft Copilot Evaluation Harness
**Stage**: 7 - Offline Task Evaluators
**Created**: 2025-08-27
**Status**: Planning Phase
**Priority**: High
**Estimated Effort**: 5-6 hours

## Overview

Stage 7 implements the core task-specific evaluators for the three offline tasks defined in the specification. These evaluators consume validated provider outputs from Stage 6 and compute objective scores using the established rubric. This stage is critical as it transforms raw provider responses into quantified performance metrics.

## Objectives

- **Task-Specific Evaluation**: Implement evaluators for metrics CSV, SSN regex, and executive summary tasks
- **Objective Scoring**: Ensure deterministic, reproducible scoring based on specification rubric
- **Safety & Performance**: Add timeout guards and error handling for robust evaluation
- **Extensible Framework**: Create evaluator base class for easy addition of new task types

## Architecture Position

Stage 7 sits between the validation framework (Stage 6) and scoring aggregation (Stage 8):

```
Provider Outputs → Stage 6 Validators → Stage 7 Evaluators → Stage 8 Scorer → Reports
```

The evaluators receive clean, validated JSON data and return structured evaluation results with detailed scoring breakdowns.

## Dependencies

- **Stage 6**: Validation Framework (SchemaValidator, FieldExtractor, StructuralValidator, ContentNormalizer)
- **Stage 5**: Core Runner & CLI (TestRunner, configuration loading)
- **Stage 3**: Schema & Configuration (test case definitions, scoring weights)

## Implementation Tasks

### Task 1: Base Evaluator Framework

**File**: `bench/core/evaluators/base.py`

**Objectives**:
- Define abstract base class for all evaluators
- Establish common evaluation result data structures
- Provide shared utilities for score calculation

**Implementation Details**:

```python
@dataclass
class EvaluationResult:
    """Result of task evaluation with detailed scoring."""
    task_id: str
    total_score: float
    max_score: float
    sub_scores: dict[str, float]
    details: dict[str, Any]
    errors: list[str]
    warnings: list[str]
    metadata: dict[str, Any]

class BaseEvaluator(ABC):
    """Abstract base class for task evaluators."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def evaluate(
        self,
        response_data: dict[str, Any],
        test_case: dict[str, Any],
        answer_key: dict[str, Any] | None = None
    ) -> EvaluationResult:
        """Evaluate provider response against test case requirements."""
        pass

    def calculate_weighted_score(
        self,
        sub_scores: dict[str, float],
        weights: dict[str, float]
    ) -> float:
        """Calculate weighted total score from sub-component scores."""
        pass
```

**Key Features**:
- Structured evaluation results with sub-score breakdowns
- Common scoring utilities and error handling
- Logging and debugging support
- Configuration-driven evaluation parameters

### Task 2: Metrics CSV Evaluator

**File**: `bench/core/evaluators/metrics_csv.py`

**Objectives**:
- Evaluate Task 1 (Metrics from CSV) responses
- Validate numeric precision/recall/F1/accuracy values
- Check confusion matrix integer values
- Apply tolerance-based scoring (±0.0005)

**Scoring Rubric** (40 points total):
- Precision: 6 points
- Recall: 6 points
- F1: 6 points
- Accuracy: 6 points
- Confusion Matrix: 12 points (3 each for TP/FP/FN/TN)
- Binary scoring: exact match within tolerance = full points, else 0

**Implementation Details**:

```python
class MetricsCSVEvaluator(BaseEvaluator):
    """Evaluator for Task 1: Metrics from CSV analysis."""

    def evaluate(self, response_data, test_case, answer_key) -> EvaluationResult:
        # Extract metrics from validated response
        # Compare against answer key with tolerance
        # Calculate sub-scores for each metric
        # Return detailed evaluation result

    def validate_numeric_field(
        self,
        actual: float,
        expected: float,
        tolerance: float = 0.0005
    ) -> bool:
        """Check if numeric value is within tolerance."""

    def validate_confusion_matrix(
        self,
        actual: dict[str, int],
        expected: dict[str, int]
    ) -> dict[str, bool]:
        """Validate confusion matrix integer values (exact match)."""
```

**Key Features**:
- Tolerance-based numeric validation (configurable, default ±0.0005)
- Exact integer matching for confusion matrix
- Detailed error reporting for failed validations
- Support for partial scoring if some metrics are correct

### Task 3: SSN Regex Evaluator

**File**: `bench/core/evaluators/regex_match.py`

**Objectives**:
- Evaluate Task 2 (SSN Regex) responses
- Compile and test provided regex patterns
- Validate against test lines with timeout protection
- Check regex validity constraints

**Scoring Rubric** (30 points total):
- Regex Validity: 18 points (anchors, format constraints, no forbidden patterns)
- Line Matches: 12 points (1 point per correct line, 12 test lines)

**Implementation Details**:

```python
class RegexMatchEvaluator(BaseEvaluator):
    """Evaluator for Task 2: SSN Regex validation."""

    def evaluate(self, response_data, test_case, answer_key) -> EvaluationResult:
        # Extract regex pattern from response
        # Validate regex syntax and constraints
        # Test against validation lines with timeout
        # Score validity and line matches

    def validate_regex_constraints(self, pattern: str) -> dict[str, bool]:
        """Check SSN regex validity constraints."""
        # Must anchor start/end (^ and $)
        # Must handle area/group/serial number constraints
        # Must reject 000, 666, 9xx area codes
        # Must reject 00 group codes
        # Must reject 0000 serial numbers

    def test_regex_lines(
        self,
        pattern: str,
        test_lines: list[str],
        timeout_ms: int = 100
    ) -> dict[int, bool]:
        """Test regex against validation lines with timeout protection."""
```

**Key Features**:
- Regex compilation with error handling
- Timeout guards to prevent catastrophic backtracking
- SSN format constraint validation
- Line-by-line match testing with detailed results

### Task 4: Executive Summary Evaluator

**File**: `bench/core/evaluators/exec_summary.py`

**Objectives**:
- Evaluate Task 3 (Executive Summary) responses
- Validate structural requirements (title, word count, bullets)
- Apply tone and clarity heuristics
- Support optional LLM-as-judge for tone scoring

**Scoring Rubric** (20 points total):
- Structure: 12 points
  - Title length ≤6 words: 3 points
  - Word count 120-160: 3 points
  - Exactly 3 bullets: 3 points
  - JSON schema compliance: 3 points
- Tone/Clarity: 8 points (heuristics + optional LLM judge)

**Implementation Details**:

```python
class ExecSummaryEvaluator(BaseEvaluator):
    """Evaluator for Task 3: Executive Summary assessment."""

    def evaluate(self, response_data, test_case, answer_key) -> EvaluationResult:
        # Validate structural requirements
        # Apply tone heuristics
        # Optional LLM-as-judge scoring
        # Calculate weighted sub-scores

    def validate_structure(self, summary_data: dict) -> dict[str, Any]:
        """Validate title, word count, bullet requirements."""

    def evaluate_tone_heuristics(self, text: str) -> dict[str, float]:
        """Apply tone and clarity heuristics."""
        # Check for hype terms (denylist)
        # Calculate average sentence length
        # Assess conciseness and clarity

    def llm_judge_tone(self, text: str) -> float:
        """Optional LLM-as-judge for tone assessment."""
        # Use configured provider for tone evaluation
        # Map to 0-8 point scale
```

**Key Features**:
- Structural validation using Stage 6 validators
- Configurable tone heuristics with denylist
- Optional LLM-as-judge integration
- Detailed breakdown of structural vs. tone scoring

### Task 5: Evaluator Registry & Loading

**File**: `bench/core/evaluators/__init__.py`

**Objectives**:
- Create evaluator registry for dynamic loading
- Provide factory pattern for evaluator instantiation
- Support configuration-driven evaluator selection

**Implementation Details**:

```python
class EvaluatorRegistry:
    """Registry for task evaluators."""

    _evaluators: dict[str, type[BaseEvaluator]] = {}

    @classmethod
    def register(cls, name: str, evaluator_class: type[BaseEvaluator]):
        """Register an evaluator class."""

    @classmethod
    def create_evaluator(cls, name: str, config: dict) -> BaseEvaluator:
        """Create evaluator instance by name."""

    @classmethod
    def list_evaluators(cls) -> list[str]:
        """List available evaluator names."""

# Auto-register built-in evaluators
EvaluatorRegistry.register("metrics_csv", MetricsCSVEvaluator)
EvaluatorRegistry.register("regex_match", RegexMatchEvaluator)
EvaluatorRegistry.register("exec_summary", ExecSummaryEvaluator)
```

### Task 6: Integration with Core Runner

**File**: `bench/core/runner.py` (modifications)

**Objectives**:
- Integrate evaluators into test execution pipeline
- Handle evaluator errors gracefully
- Store evaluation results with artifacts

**Implementation Details**:
- Add evaluator loading to TestRunner initialization
- Call appropriate evaluator based on test case configuration
- Store evaluation results in run artifacts
- Handle evaluator exceptions with detailed error reporting

### Task 7: Comprehensive Testing

**Files**: `tests/test_evaluators.py`, `tests/test_integration.py`

**Objectives**:
- Unit tests for each evaluator
- Integration tests with validation framework
- Edge case and error condition testing

**Test Coverage**:
- Metrics CSV: tolerance boundaries, missing fields, invalid numbers
- SSN Regex: valid/invalid patterns, timeout scenarios, constraint violations
- Executive Summary: word count boundaries, bullet variations, tone edge cases
- Integration: end-to-end evaluation pipeline testing

## Implementation Checklist

### Phase 1: Core Framework
- [ ] Implement BaseEvaluator abstract class
- [ ] Define EvaluationResult data structure
- [ ] Create common scoring utilities
- [ ] Add logging and error handling framework

### Phase 2: Task Evaluators
- [ ] Implement MetricsCSVEvaluator
  - [ ] Numeric tolerance validation
  - [ ] Confusion matrix checking
  - [ ] Score calculation with weights
- [ ] Implement RegexMatchEvaluator
  - [ ] Regex compilation and validation
  - [ ] SSN constraint checking
  - [ ] Line testing with timeout guards
- [ ] Implement ExecSummaryEvaluator
  - [ ] Structural validation integration
  - [ ] Tone heuristics implementation
  - [ ] Optional LLM-as-judge support

### Phase 3: Integration & Testing
- [ ] Create EvaluatorRegistry system
- [ ] Integrate with core runner
- [ ] Write comprehensive unit tests
- [ ] Add integration tests
- [ ] Performance and timeout testing

### Phase 4: Documentation & Polish
- [ ] Add docstrings and type annotations
- [ ] Create evaluator configuration documentation
- [ ] Add example usage and troubleshooting guides
- [ ] Ensure all code passes quality gates

## Success Criteria

### Functional Requirements
- [ ] All three offline task evaluators implemented and working
- [ ] Evaluators produce deterministic, reproducible scores
- [ ] Proper error handling for malformed inputs
- [ ] Integration with validation framework complete

### Quality Requirements
- [ ] 100% test coverage for evaluator logic
- [ ] All code passes linting and type checking
- [ ] Comprehensive error messages and logging
- [ ] Performance meets timeout requirements

### Integration Requirements
- [ ] Seamless integration with Stage 6 validators
- [ ] Proper artifact storage and retrieval
- [ ] Configuration-driven evaluator selection
- [ ] Ready for Stage 8 scoring aggregation

## Risk Mitigation

### Technical Risks
- **Regex Timeout**: Implement robust timeout guards with fallback error handling
- **Numeric Precision**: Use decimal arithmetic for tolerance comparisons
- **LLM Judge Reliability**: Make LLM-as-judge optional with heuristic fallback

### Integration Risks
- **Validation Dependencies**: Ensure proper error handling when validation fails
- **Configuration Complexity**: Provide clear examples and validation for evaluator configs
- **Performance**: Profile evaluator performance and optimize bottlenecks

## Next Steps

After Stage 7 completion:
1. **Stage 8**: Scoring & Aggregation - Weight-based score combination
2. **Stage 9**: Reporting System - Generate detailed evaluation reports
3. **Stage 10**: Sample Tests & Fixtures - Complete test definitions and answer keys

## Notes

- Evaluators are designed to be stateless and thread-safe
- All scoring follows the exact rubric specified in the main specification
- Configuration-driven approach allows easy tuning without code changes
- Comprehensive error handling ensures robust evaluation even with malformed inputs
- Integration with Stage 6 validators provides clean, validated input data

---

**Estimated Timeline**: 5-6 hours for complete implementation
**Dependencies**: Stage 6 (Validation Framework) must be completed
**Next Stage**: Stage 8 (Scoring & Aggregation)
