# Stage 10 Implementation Plan: Sample Tests & Fixtures

**Project**: ChatGPT vs Microsoft Copilot Evaluation Harness
**Stage**: 10 - Sample Tests & Fixtures
**Priority**: High
**Estimated Effort**: 4-5 hours
**Dependencies**: Stage 9 (Reporting System)

## Overview

Stage 10 completes the evaluation harness by implementing the actual test definitions, fixtures, and answer keys that define the three core offline tasks. This stage transforms the framework built in Stages 1-9 into a working benchmark system with concrete, executable test cases that can objectively compare ChatGPT and Microsoft Copilot performance.

## Objectives

- **Complete Test Definitions**: Create YAML test case files for all three offline tasks
- **Sample Fixtures**: Add realistic CSV and text fixtures for evaluation
- **Answer Keys**: Provide ground truth data for objective scoring
- **Test Suite Enhancement**: Expand pytest coverage for end-to-end workflows
- **Documentation**: Complete usage examples and test case documentation

## Architecture Position

Stage 10 sits at the application layer, utilizing all the framework components built in previous stages:
- **Input**: Framework from Stages 1-9 (providers, evaluators, scoring, reporting)
- **Processing**: Test case definitions, fixtures, and validation
- **Output**: Complete working benchmark system ready for evaluation runs

## Implementation Tasks

### Test Case Definitions (`bench/tests/offline/`)

#### 1. Task 1: Metrics from CSV (`task1_metrics.yaml`)
```yaml
id: "offline.task1.metrics_csv"
name: "Task 1 — Metrics from CSV"
category: "offline"
capability_profile:
  web: "forbidden"
  json_required: true
  retries: 1
prompt:
  system: |
    You are an enterprise assistant. Follow instructions exactly. Do not browse the web.
    Do not fabricate sources. Respond only in valid JSON format.
  user: |
    Analyze the following CSV data and calculate classification metrics.

    CSV Data:
    {csv_content}

    Calculate the following metrics and return them in JSON format:
    - precision (4 decimal places)
    - recall (4 decimal places)
    - f1 (4 decimal places)
    - accuracy (4 decimal places)
    - confusion_matrix with tp, fp, fn, tn (integers)

    Required JSON format:
    {
      "task1_data_metrics": {
        "precision": 0.0000,
        "recall": 0.0000,
        "f1": 0.0000,
        "accuracy": 0.0000,
        "confusion_matrix": {
          "tp": 0,
          "fp": 0,
          "fn": 0,
          "tn": 0
        }
      }
    }

expectation:
  schema_name: "task1_metrics"
  fields:
    - path: $.task1_data_metrics.precision
      type: number
    - path: $.task1_data_metrics.recall
      type: number
    - path: $.task1_data_metrics.f1
      type: number
    - path: $.task1_data_metrics.accuracy
      type: number
    - path: $.task1_data_metrics.confusion_matrix.tp
      type: integer
    - path: $.task1_data_metrics.confusion_matrix.fp
      type: integer
    - path: $.task1_data_metrics.confusion_matrix.fn
      type: integer
    - path: $.task1_data_metrics.confusion_matrix.tn
      type: integer

scoring:
  evaluator: metrics_csv
  config:
    tolerance: 0.0005
    round_to: 4
    weights:
      precision: 6
      recall: 6
      f1: 6
      accuracy: 6
      confusion_matrix:
        tp: 3
        fp: 3
        fn: 3
        tn: 3

fixtures:
  - path: "fixtures/csv/phishing_sample.csv"
    variable: "csv_content"

answer_key: "answer_keys/offline/task1_metrics.json"
```

#### 2. Task 2: SSN Regex (`task2_ssn_regex.yaml`)
```yaml
id: "offline.task2.ssn_regex"
name: "Task 2 — DLP Regex for U.S. SSN"
category: "offline"
capability_profile:
  web: "forbidden"
  json_required: true
  retries: 1
prompt:
  system: |
    You are an enterprise assistant. Follow instructions exactly. Do not browse the web.
    Do not fabricate sources. Respond only in valid JSON format.
  user: |
    Create a regular expression to identify valid U.S. Social Security Numbers (SSNs).

    Requirements:
    - Match format: XXX-XX-XXXX (with hyphens)
    - Reject invalid area codes: 000, 666, 900-999
    - Reject invalid group codes: 00
    - Reject invalid serial numbers: 0000
    - Use anchors to match complete strings only
    - No catastrophic backtracking

    Test your regex against these lines:
    {test_lines}

    Return in JSON format:
    {
      "task2_ssn_regex": {
        "regex_pattern": "your_regex_here",
        "matching_lines": [1, 2, 3],
        "explanation": "Brief explanation of regex components"
      }
    }

expectation:
  schema_name: "task2_ssn_regex"
  fields:
    - path: $.task2_ssn_regex.regex_pattern
      type: string
    - path: $.task2_ssn_regex.matching_lines
      type: array
    - path: $.task2_ssn_regex.explanation
      type: string

scoring:
  evaluator: regex_match
  config:
    timeout_ms: 100
    weights:
      regex_validity: 18
      line_matches: 12
    validity_checks:
      - anchors_required: true
      - area_code_restrictions: [000, 666, "9xx"]
      - group_code_restrictions: ["00"]
      - serial_restrictions: ["0000"]

fixtures:
  - path: "fixtures/text/ssn_validation_lines.txt"
    variable: "test_lines"

answer_key: "answer_keys/offline/task2_lines.json"
```

#### 3. Task 3: Executive Summary (`task3_exec_summary.yaml`)
```yaml
id: "offline.task3.exec_summary"
name: "Task 3 — Executive Summary"
category: "offline"
capability_profile:
  web: "forbidden"
  json_required: true
  retries: 1
prompt:
  system: |
    You are an enterprise assistant. Follow instructions exactly. Do not browse the web.
    Do not fabricate sources. Respond only in valid JSON format.
  user: |
    Write an executive summary about the current state of artificial intelligence in enterprise applications.

    Requirements:
    - Title: Maximum 6 words
    - Summary: 120-160 words (excluding title and bullets)
    - Exactly 3 bullet points highlighting key insights
    - Professional tone, avoid hype terms
    - Concise sentences (average ≤24 words)

    Return in JSON format:
    {
      "task3_exec_summary": {
        "title": "Your Title Here",
        "summary": "Your summary paragraph here...",
        "key_points": [
          "First key insight",
          "Second key insight",
          "Third key insight"
        ]
      }
    }

expectation:
  schema_name: "task3_exec_summary"
  fields:
    - path: $.task3_exec_summary.title
      type: string
    - path: $.task3_exec_summary.summary
      type: string
    - path: $.task3_exec_summary.key_points
      type: array

scoring:
  evaluator: exec_summary
  config:
    weights:
      structure: 12
      tone: 8
    structure_checks:
      title_max_words: 6
      summary_word_range: [120, 160]
      bullet_count: 3
      json_compliance: true
    tone_checks:
      hype_terms_denylist: ["revolutionary", "game-changing", "world-class", "cutting-edge", "breakthrough"]
      max_avg_sentence_length: 24
      professional_tone: true

answer_key: "answer_keys/offline/task3_structure.json"
```

### Sample Fixtures (`fixtures/`)

#### 1. CSV Data (`fixtures/csv/phishing_sample.csv`)
```csv
email_id,sender,subject,body_length,has_links,has_attachments,urgency_words,is_phishing
1,support@bank.com,Account Verification Required,245,1,0,3,1
2,newsletter@company.com,Weekly Updates,156,2,0,0,0
3,urgent@security.com,IMMEDIATE ACTION REQUIRED,89,1,1,5,1
4,info@vendor.com,Invoice #12345,134,0,1,0,0
5,admin@internal.com,Policy Update,198,1,0,1,0
6,noreply@service.com,Password Reset,67,1,0,2,1
7,team@project.com,Meeting Notes,289,0,0,0,0
8,alert@system.com,System Maintenance,145,1,0,1,0
```

#### 2. SSN Test Lines (`fixtures/text/ssn_validation_lines.txt`)
```
123-45-6789
000-12-3456
123-00-4567
123-45-0000
666-12-3456
987-65-4321
555-55-5555
123-456-7890
12-345-6789
1234-56-789
900-12-3456
123-45-67890
```

### Answer Keys (`answer_keys/offline/`)

#### 1. Task 1 Metrics (`task1_metrics.json`)
```json
{
  "precision": 0.7500,
  "recall": 0.6000,
  "f1": 0.6667,
  "accuracy": 0.6250,
  "confusion_matrix": {
    "tp": 3,
    "fp": 1,
    "fn": 2,
    "tn": 2
  }
}
```

#### 2. Task 2 Lines (`task2_lines.json`)
```json
{
  "matching_lines": [1, 6, 7],
  "total_valid": 3,
  "explanation": "Lines 1, 6, 7 contain valid SSN formats that pass all validation rules"
}
```

#### 3. Task 3 Structure (`task3_structure.json`)
```json
{
  "expected_structure": {
    "title_max_words": 6,
    "summary_word_range": [120, 160],
    "bullet_count": 3,
    "tone_requirements": {
      "avoid_hype": true,
      "professional": true,
      "concise_sentences": true
    }
  }
}
```

### Enhanced Test Suite (`tests/`)

#### 1. End-to-End Integration Tests (`tests/test_integration.py`)
```python
class TestEndToEndWorkflow:
    """Test complete evaluation workflow."""

    def test_full_offline_evaluation(self):
        """Test complete offline task evaluation."""

    def test_provider_comparison(self):
        """Test provider comparison workflow."""

    def test_report_generation(self):
        """Test report generation from evaluation results."""

class TestTestCaseValidation:
    """Test test case definition validation."""

    def test_task1_yaml_validation(self):
        """Test Task 1 YAML structure."""

    def test_task2_yaml_validation(self):
        """Test Task 2 YAML structure."""

    def test_task3_yaml_validation(self):
        """Test Task 3 YAML structure."""

class TestFixtureLoading:
    """Test fixture loading and processing."""

    def test_csv_fixture_loading(self):
        """Test CSV fixture loading."""

    def test_text_fixture_loading(self):
        """Test text fixture loading."""

    def test_answer_key_loading(self):
        """Test answer key loading."""
```

#### 2. Fixture Validation Tests (`tests/test_fixtures.py`)
```python
class TestFixtureIntegrity:
    """Test fixture data integrity."""

    def test_csv_data_consistency(self):
        """Test CSV data matches expected format."""

    def test_ssn_lines_coverage(self):
        """Test SSN test lines cover edge cases."""

    def test_answer_key_accuracy(self):
        """Test answer keys match fixture expectations."""
```

### Documentation Updates

#### 1. Test Case Documentation (`docs/test_cases.md`)
- Detailed explanation of each test case
- Scoring rubric breakdown
- Expected provider behavior
- Common failure modes

#### 2. Usage Examples (`docs/examples.md`)
- Running individual tests
- Comparing providers
- Interpreting results
- Adding new test cases

#### 3. Fixture Documentation (`docs/fixtures.md`)
- Fixture format specifications
- Answer key structure
- Data generation guidelines

## Configuration Integration

### Test Case Schema Updates (`schemas/test_case.schema.json`)
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["id", "name", "category", "capability_profile", "prompt", "expectation", "scoring"],
  "properties": {
    "fixtures": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "path": {"type": "string"},
          "variable": {"type": "string"}
        }
      }
    },
    "answer_key": {"type": "string"}
  }
}
```

### Runner Integration Updates

#### 1. Fixture Loading (`bench/core/runner.py`)
- Add fixture loading and variable substitution
- Template rendering for test prompts
- Answer key loading and validation

#### 2. Test Discovery (`bench/core/utils.py`)
- Automatic test case discovery
- Fixture path resolution
- Answer key validation

## Testing Strategy

### Unit Tests
- Test case YAML validation
- Fixture loading and processing
- Answer key integrity checks
- Template rendering accuracy

### Integration Tests
- End-to-end evaluation workflows
- Provider comparison accuracy
- Report generation completeness
- Error handling robustness

### Validation Tests
- Schema compliance verification
- Scoring accuracy validation
- Fixture-answer key consistency
- Performance benchmarking

## Error Handling

### Test Case Validation
- YAML syntax validation
- Schema compliance checking
- Fixture path verification
- Answer key format validation

### Runtime Error Handling
- Missing fixture graceful degradation
- Invalid answer key handling
- Template rendering failures
- Provider response validation

## Performance Considerations

### Fixture Loading Optimization
- Lazy loading for large fixtures
- Caching for repeated test runs
- Memory-efficient processing
- Parallel fixture loading

### Test Execution Efficiency
- Concurrent test execution where safe
- Provider response caching
- Incremental result updates
- Resource cleanup

## Success Criteria

### Functional Requirements
- [ ] All three offline test cases execute successfully
- [ ] Fixtures load correctly and integrate with test prompts
- [ ] Answer keys provide accurate ground truth for scoring
- [ ] Test cases produce deterministic, objective scores
- [ ] End-to-end workflow completes without errors
- [ ] Reports generate with meaningful comparisons

### Quality Requirements
- [ ] Test cases follow specification exactly
- [ ] Fixtures represent realistic evaluation scenarios
- [ ] Answer keys are mathematically accurate
- [ ] All tests pass with good coverage
- [ ] Documentation is complete and accurate
- [ ] Error messages are clear and actionable

### Integration Requirements
- [ ] Test cases integrate seamlessly with existing framework
- [ ] Fixtures work with all provider adapters
- [ ] Scoring produces expected point distributions
- [ ] Reports show meaningful provider comparisons
- [ ] CLI commands work end-to-end
- [ ] CI/CD pipeline executes successfully

## Future Enhancements

### Phase 2 Features
- Online deep research test case
- Additional fixture variations
- Multi-language test cases
- Performance benchmarking tests

### Phase 3 Features
- Dynamic fixture generation
- Adaptive difficulty scaling
- Custom test case templates
- Automated answer key generation

## Dependencies

### External Libraries
```toml
[tool.poetry.dependencies]
pyyaml = "^6.0"      # YAML test case loading
jinja2 = "^3.1.0"    # Template rendering
jsonpath-ng = "^1.6.0"  # Field extraction
```

### Internal Dependencies
- Stage 9: Reporting system for result processing
- Stage 8: Scoring system for point calculation
- Stage 7: Evaluators for task-specific logic
- Stage 6: Validators for response processing
- Stage 5: Runner for test execution
- Stage 4: Providers for model interaction

## Deliverables

1. **Complete Test Case Definitions**
   - `task1_metrics.yaml` with CSV analysis requirements
   - `task2_ssn_regex.yaml` with regex validation requirements
   - `task3_exec_summary.yaml` with writing requirements

2. **Sample Fixtures and Data**
   - `phishing_sample.csv` with realistic classification data
   - `ssn_validation_lines.txt` with edge case coverage
   - Answer keys with ground truth values

3. **Enhanced Test Suite**
   - End-to-end integration tests
   - Fixture validation tests
   - Test case schema validation
   - Performance benchmarking tests

4. **Complete Documentation**
   - Test case specifications
   - Usage examples and tutorials
   - Fixture format documentation
   - Troubleshooting guides

5. **Working Benchmark System**
   - Fully functional evaluation harness
   - Provider comparison capabilities
   - Objective scoring and reporting
   - CI/CD integration ready

This implementation provides a complete, working benchmark system that can objectively compare ChatGPT and Microsoft Copilot performance across three well-defined offline tasks, with all the infrastructure needed for reliable, reproducible evaluations.
