# Stage 3 Implementation Plan: Schema & Configuration Foundation

**Project**: ChatGPT vs Microsoft Copilot Evaluation Harness
**Stage**: 3 of 11
**Created**: 2025-08-27
**Status**: Ready for Implementation

## Overview
**Objective**: Define JSON schemas for test case validation and create configuration templates for providers and scoring
**Priority**: High
**Estimated Effort**: 2-3 hours
**Dependencies**: Stage 2 (Completed)

## Detailed Task Breakdown

### 1. JSON Schema Definitions

#### 1.1 Test Case Schema (`schemas/test_case.schema.json`)
**Purpose**: Validate test case YAML structure and enforce data contracts

**Schema Requirements**:
- **Root Properties**:
  - `id`: Unique test identifier (e.g., "offline.task1.metrics")
  - `name`: Human-readable test name
  - `category`: Test category ("offline" | "online")
  - `capability_profile`: Capability constraints object
  - `prompt`: System and user prompt definitions
  - `expectation`: Expected response structure
  - `scoring`: Evaluator configuration

- **Capability Profile Schema**:
  - `web`: Web access control ("forbidden" | "allowed" | "required")
  - `json_required`: Boolean for JSON response requirement
  - `retries`: Number of retry attempts (integer)
  - `tools`: Array of allowed tools (optional)

- **Prompt Schema**:
  - `system`: System prompt string with constraints
  - `user`: User prompt string with task specification

- **Expectation Schema**:
  - `schema_name`: Reference to response validation schema
  - `fields`: Array of field extraction definitions
    - `path`: JSONPath selector string
    - `type`: Expected data type ("string" | "number" | "integer" | "boolean" | "array" | "object")
    - `required`: Boolean (default: true)

- **Scoring Schema**:
  - `evaluator`: Evaluator name ("metrics_csv" | "regex_match" | "exec_summary" | "deep_research")
  - `config`: Evaluator-specific configuration object

#### 1.2 Scoring Rubric Schema (`schemas/rubric.schema.json`)
**Purpose**: Validate scoring weight configurations and ensure consistency

**Schema Requirements**:
- **Root Properties**:
  - `version`: Schema version string
  - `total_points`: Maximum possible score (integer)
  - `tasks`: Task-specific scoring configurations

- **Task Scoring Schema**:
  - Task name as key (e.g., "task1_metrics")
  - `total`: Total points for this task
  - Task-specific weight breakdowns:
    - **Task 1**: `precision`, `recall`, `f1`, `accuracy`, `confusion_matrix` sub-weights
    - **Task 2**: `regex_validity`, `line_matches` weights
    - **Task 3**: `structure`, `tone` weights
    - **Deep Research**: `plan_quality`, `source_quality` weights

- **Stability Bonus Schema**:
  - `enabled`: Boolean flag
  - `max_points`: Maximum bonus points (5)
  - `consistency_threshold`: Threshold for consistency scoring

### 2. Provider Configuration

#### 2.1 Provider Settings (`configs/providers.yaml`)
**Purpose**: Define provider-specific settings and capability enforcement

**Configuration Structure**:
```yaml
providers:
  chatgpt:
    adapter: "chatgpt"
    options:
      model: "gpt-4"
      temperature: 0
      max_tokens: 1200
      seed: 42
      response_format: "json_object"
      timeout_seconds: 60
    capabilities:
      web: "forbidden"        # Default for offline tasks
      tools: []              # No additional tools
      json_required: true    # Require JSON response format

  copilot_manual:
    adapter: "copilot_manual"
    options:
      display_format: "markdown"  # How to display prompts
      timeout_seconds: 300        # Manual response timeout
    capabilities:
      web: "forbidden"          # No web access for offline tasks
      tools: []                # No additional tools
      json_required: true      # Require JSON response format
```

**Key Features**:
- Deterministic settings for reproducibility (temperature=0, seed=42)
- Capability enforcement at provider level
- Timeout configurations for both API and manual modes
- Extensible options for future provider additions

#### 2.2 Default Scoring Weights (`configs/weights.default.yaml`)
**Purpose**: Define point allocation per specification requirements

**Weight Structure** (105 total points):
```yaml
version: "1.0"
total_points: 105

tasks:
  task1_metrics:
    total: 40
    precision: 6      # Precision metric accuracy
    recall: 6         # Recall metric accuracy
    f1: 6            # F1 score accuracy
    accuracy: 6       # Overall accuracy metric
    confusion_matrix: # Confusion matrix components (12 points total)
      tp: 3          # True positives
      fp: 3          # False positives
      fn: 3          # False negatives
      tn: 3          # True negatives

  task2_ssn_regex:
    total: 30
    regex_validity: 18  # Regex format and constraint validation
    line_matches: 12    # Correct line matching (1 point per line)

  task3_exec_summary:
    total: 20
    structure: 12       # Title, word count, bullets, schema (3 each)
    tone: 8            # Heuristics and clarity assessment

  deep_research:        # Optional online task
    total: 10
    plan_quality: 5     # Structure and sequencing
    source_quality: 5   # Recency and verification

stability_bonus:
  enabled: true
  max_points: 5
  consistency_threshold: 0.95  # 95% consistency for full bonus
```

**Scoring Rules**:
- **Task 1**: Numeric tolerance ±0.0005, integers exact match
- **Task 2**: Regex validity checks + line matching accuracy
- **Task 3**: Structural compliance + tone heuristics
- **Stability**: Multi-run consistency bonus (0-5 points)

#### 2.3 Run Matrix Configuration (`configs/runmatrix.yaml`)
**Purpose**: Define test execution combinations and repetitions

**Matrix Structure**:
```yaml
version: "1.0"
description: "Default evaluation matrix for ChatGPT vs Copilot comparison"

matrix:
  - provider: "chatgpt"
    test_set: "offline"
    repetitions: 3          # For stability analysis
    parallel: false         # Sequential execution

  - provider: "copilot_manual"
    test_set: "offline"
    repetitions: 3          # For stability analysis
    parallel: false         # Manual input required

  - provider: "chatgpt"
    test_set: "online"      # Optional deep research task
    repetitions: 1          # Single run for online tasks
    parallel: false

  - provider: "copilot_manual"
    test_set: "online"      # Optional deep research task
    repetitions: 1          # Single run for online tasks
    parallel: false

test_sets:
  offline:
    - "tests/offline/task1_metrics.yaml"
    - "tests/offline/task2_ssn_regex.yaml"
    - "tests/offline/task3_exec_summary.yaml"

  online:
    - "tests/online/deep_research_agentic_ai.yaml"

global_settings:
  output_dir: "results"
  save_raw_responses: true
  save_parsed_responses: true
  generate_report: true
  fail_fast: false         # Continue on individual test failures
```

### 3. Schema Validation Integration

#### 3.1 Schema Loading and Validation
**Implementation Requirements**:
- JSON Schema validation using `jsonschema` library
- Schema file loading with error handling
- Validation error reporting with clear messages
- Schema version compatibility checking

#### 3.2 Configuration Validation
**Validation Checks**:
- Provider configuration completeness
- Weight total validation (should sum to expected totals)
- Test set file existence verification
- Capability constraint consistency
- Required field presence validation

### 4. Documentation and Examples

#### 4.1 Configuration Documentation
**Create**: `docs/configuration.md`
- Provider configuration options
- Scoring weight customization
- Run matrix setup
- Schema validation rules

#### 4.2 Schema Examples
**Create**: Example files demonstrating:
- Valid test case definitions
- Custom scoring configurations
- Provider setup variations
- Error scenarios and fixes

## Implementation Steps

### Step 1: Create JSON Schemas (45 minutes)
1. **Create `schemas/test_case.schema.json`**:
   - Define complete test case structure
   - Add validation rules for all fields
   - Include pattern matching for IDs and categories
   - Add description fields for documentation

2. **Create `schemas/rubric.schema.json`**:
   - Define scoring weight structure
   - Add validation for point totals
   - Include task-specific weight schemas
   - Add stability bonus configuration

### Step 2: Create Configuration Files (45 minutes)
1. **Create `configs/providers.yaml`**:
   - ChatGPT provider with deterministic settings
   - Copilot manual provider configuration
   - Capability enforcement definitions
   - Timeout and retry settings

2. **Create `configs/weights.default.yaml`**:
   - Implement 40+30+20+10+5 point structure
   - Define sub-metric weights per specification
   - Add stability bonus configuration
   - Include validation metadata

3. **Create `configs/runmatrix.yaml`**:
   - Define provider × test set combinations
   - Set repetition counts for stability analysis
   - Configure output and execution settings
   - Add test set definitions

### Step 3: Schema Integration (30 minutes)
1. **Update existing placeholder files**:
   - Add schema references to test YAML templates
   - Update configuration loading in core modules
   - Add validation hooks to runner framework

2. **Create validation utilities**:
   - Schema loading functions
   - Configuration validation helpers
   - Error reporting utilities

### Step 4: Testing and Validation (30 minutes)
1. **Validate all schemas**:
   - Test schema files with JSON Schema validators
   - Verify configuration file compliance
   - Check cross-references and dependencies

2. **Create test configurations**:
   - Minimal valid configurations for testing
   - Invalid configurations for error testing
   - Edge cases and boundary conditions

## Success Criteria

- [ ] **JSON Schemas Created**: Both `test_case.schema.json` and `rubric.schema.json` validate correctly
- [ ] **Configuration Files**: All three YAML files (`providers.yaml`, `weights.default.yaml`, `runmatrix.yaml`) are valid and complete
- [ ] **Schema Compliance**: All configuration files validate against their respective schemas
- [ ] **Point Allocation**: Scoring weights sum to specification requirements (40+30+20+10+5 = 105 points)
- [ ] **Provider Settings**: ChatGPT configured with deterministic settings (temp=0, seed=42)
- [ ] **Capability Enforcement**: Web access and tool restrictions properly defined
- [ ] **Documentation**: Configuration options and schema rules documented
- [ ] **Validation Ready**: Schema validation can be integrated into core components

## Deliverables

### Core Files
- `schemas/test_case.schema.json` - Test case validation schema
- `schemas/rubric.schema.json` - Scoring rubric validation schema
- `configs/providers.yaml` - Provider configuration with ChatGPT and Copilot settings
- `configs/weights.default.yaml` - Default scoring weights (105 points total)
- `configs/runmatrix.yaml` - Test execution matrix configuration

### Documentation
- Configuration documentation with examples
- Schema validation rules and error handling
- Provider setup instructions
- Scoring customization guide

### Validation
- Schema validation utilities
- Configuration compliance checking
- Error reporting and debugging aids

## Dependencies

**Required**: Stage 2 (Repository Structure & Scaffolding) - Completed
**Enables**: Stage 4 (Provider Abstraction & Adapters)

## Notes

- All configurations follow YAML format for human readability
- JSON Schemas provide strict validation and documentation
- Point allocation exactly matches specification (40+30+20+10+5)
- Provider settings ensure deterministic and reproducible results
- Configuration is extensible for future providers and test types
- Schema validation enables early error detection and debugging

---

**Next Action**: Begin implementation of JSON schemas and configuration files
**Following Stage**: Stage 4 - Provider Abstraction & Adapters
