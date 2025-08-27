# Objective Benchmark Implementation Plan

**Project**: ChatGPT vs Microsoft Copilot Evaluation Harness  
**Created**: 2025-08-27  
**Status**: Planning Phase

## Overview

This document outlines the staged implementation plan for building a fair, extensible, and repeatable evaluation harness that compares general-purpose assistants (ChatGPT vs Microsoft Copilot) on offline and online tasks with objective scoring.

## Project Goals

- **Open & Reproducible**: Fair comparison harness with well-specified tasks
- **Declarative Configuration**: Add/modify test cases via YAML/JSON without code changes
- **Strict Constraints**: Enforce capabilities (no browsing for offline tests, JSON-only outputs)
- **Objective Scoring**: Numeric scores with detailed breakdown and failure reasons
- **Multi-Provider Support**: ChatGPT API + manual paste mode for Copilot

## Architecture Components

1. **Test Definitions** - YAML files with tasks, constraints, expected outputs, scoring rules
2. **Runner** - Executes prompts, collects outputs, stores artifacts
3. **Validators** - Schema validation, structural checks, content normalization
4. **Evaluators** - Task-specific correctness computation
5. **Scorer** - Aggregates task scores with configurable weights
6. **Report Generator** - Markdown/JSON summaries with leaderboards

## Implementation Stages

### ✅ Stage 0: Planning & Analysis
**Status**: Completed  
**Deliverables**: 
- Analyzed specification requirements
- Created comprehensive implementation roadmap
- Established staged development approach

---

### 🔄 Stage 1: Project Bootstrap & Tooling Setup
**Status**: Pending  
**Priority**: High  
**Estimated Effort**: 1-2 hours

**Objectives**:
- Initialize modern Python project structure
- Configure development toolchain per user preferences
- Set up CI/CD foundation

**Tasks**:
- [ ] Initialize project with `uv` package manager
- [ ] Create `pyproject.toml` with dependencies and tool configurations
- [ ] Configure development tools:
  - `pytest` for testing framework
  - `black` for code formatting
  - `isort` for import sorting
  - `ruff` for code quality checks
  - `mypy` for type checking
  - `flake8` for additional style checks
- [ ] Create basic GitHub Actions CI skeleton
- [ ] Write initial `README.md` with setup instructions

**Deliverables**:
- `pyproject.toml` with complete tool configuration
- `.github/workflows/ci.yml` skeleton
- `README.md` with project overview and setup

**Dependencies**: None

---

### 📁 Stage 2: Repository Structure & Scaffolding
**Status**: Pending  
**Priority**: High  
**Estimated Effort**: 1 hour

**Objectives**:
- Create complete directory structure per specification
- Establish Python package hierarchy
- Prepare placeholder files for development

**Tasks**:
- [ ] Create directory structure:
  ```
  bench/
    adapters/
    core/
      evaluators/
  tests/
    offline/
    online/
  fixtures/
    csv/
    text/
  answer_keys/
    offline/
  schemas/
  configs/
  scripts/
  results/
  ```
- [ ] Add `__init__.py` files for Python packages
- [ ] Add `.gitkeep` files for empty directories
- [ ] Create `.gitignore` for Python projects

**Deliverables**:
- Complete directory structure
- Python package initialization
- Git configuration files

**Dependencies**: Stage 1

---

### 📋 Stage 3: Schema & Configuration Foundation
**Status**: Pending  
**Priority**: High  
**Estimated Effort**: 2-3 hours

**Objectives**:
- Define JSON schemas for test case validation
- Create configuration templates for providers and scoring
- Establish data contracts for the system

**Tasks**:
- [ ] Create `schemas/test_case.schema.json`:
  - Test case structure validation
  - Prompt format requirements
  - Expectation field definitions
  - Scoring configuration schema
- [ ] Create `schemas/rubric.schema.json`:
  - Scoring rubric validation
  - Weight configuration schema
- [ ] Create `configs/providers.yaml`:
  - ChatGPT provider settings (temperature=0, seed=42, etc.)
  - Copilot manual provider configuration
  - Capability enforcement settings
- [ ] Create `configs/weights.default.yaml`:
  - Task 1: 40 points (precision, recall, F1, accuracy, confusion matrix)
  - Task 2: 30 points (regex validity, line matches)
  - Task 3: 20 points (structure, tone)
  - Deep Research: 10 points (optional)
  - Stability bonus: 5 points
- [ ] Create `configs/runmatrix.yaml`:
  - Provider × test set combinations
  - Repetition settings for stability analysis

**Deliverables**:
- JSON Schema files for validation
- YAML configuration templates
- Documentation of configuration options

**Dependencies**: Stage 2

---

### 🔌 Stage 4: Provider Abstraction & Adapters
**Status**: Pending  
**Priority**: High  
**Estimated Effort**: 3-4 hours

**Objectives**:
- Implement provider interface abstraction
- Build ChatGPT API integration
- Create manual paste mode for Copilot
- Enforce capability constraints

**Tasks**:
- [ ] Implement `bench/adapters/base.py`:
  - `Provider` abstract base class
  - `invoke()` method signature
  - Capability enforcement interface
- [ ] Implement `bench/adapters/chatgpt.py`:
  - OpenAI API integration
  - Deterministic settings (temperature=0, seed)
  - JSON response format enforcement
  - Error handling and retries
- [ ] Implement `bench/adapters/copilot_manual.py`:
  - Interactive prompt display
  - Manual response collection
  - Raw text storage and processing
- [ ] Add capability enforcement:
  - Web access controls (`forbidden|required|allowed`)
  - Tool restrictions
  - JSON output requirements

**Deliverables**:
- Provider abstraction layer
- ChatGPT API adapter with deterministic settings
- Manual Copilot adapter for paste mode
- Capability constraint system

**Dependencies**: Stage 3

---

### ⚙️ Stage 5: Core Runner & CLI
**Status**: Pending  
**Priority**: High  
**Estimated Effort**: 4-5 hours

**Objectives**:
- Build orchestration engine for test execution
- Create command-line interface
- Implement results artifact management

**Tasks**:
- [ ] Implement `bench/core/runner.py`:
  - Test execution orchestration
  - Provider invocation management
  - Results collection and storage
  - Error handling and recovery
- [ ] Build `scripts/bench.py` CLI:
  - `run` command (full matrix or single test)
  - `validate` command (test case validation)
  - Configuration loading and validation
  - Progress reporting and logging
- [ ] Add results management:
  - Timestamped run directories
  - Raw output storage
  - Parsed JSON artifacts
  - Configuration snapshots

**Deliverables**:
- Core runner orchestration engine
- Command-line interface with subcommands
- Results artifact management system

**Dependencies**: Stage 4

---

### ✅ Stage 6: Validation Framework
**Status**: Pending  
**Priority**: Medium  
**Estimated Effort**: 3-4 hours

**Objectives**:
- Implement JSON schema validation
- Build field extraction system
- Add structural validation checks

**Tasks**:
- [ ] Implement `bench/core/validators.py`:
  - JSON schema validation against test case schemas
  - JSONPath-like field extraction
  - Structural checks (word counts, bullet counting)
  - Content normalization utilities
- [ ] Implement `bench/core/utils.py`:
  - Tokenization and word counting
  - Seed management utilities
  - Text processing helpers
  - Timeout and safety guards

**Deliverables**:
- JSON validation framework
- Field extraction system
- Structural validation utilities
- Common utility functions

**Dependencies**: Stage 5

---

### 🎯 Stage 7: Offline Task Evaluators
**Status**: Pending  
**Priority**: High  
**Estimated Effort**: 5-6 hours

**Objectives**:
- Implement evaluators for the three offline tasks
- Add safety guards and timeout protection
- Ensure objective scoring compliance

**Tasks**:
- [ ] Implement `bench/core/evaluators/metrics_csv.py`:
  - Load expected metrics from answer keys
  - Numeric field validation (tolerance ±0.0005)
  - Confusion matrix integer validation
  - Score calculation with weights
- [ ] Implement `bench/core/evaluators/regex_match.py`:
  - Regex compilation with `re.fullmatch`
  - Timeout guards (100ms per line)
  - Line matching validation
  - SSN format validity checks (anchors, constraints)
- [ ] Implement `bench/core/evaluators/exec_summary.py`:
  - Title word count validation (≤6 words)
  - Summary word count (120-160 words)
  - Bullet count validation (exactly 3)
  - Tone heuristics (denylist, sentence length)
- [ ] Add evaluator registry and loading system

**Deliverables**:
- Task 1: Metrics from CSV evaluator
- Task 2: SSN Regex evaluator  
- Task 3: Executive Summary evaluator
- Evaluator framework with safety guards

**Dependencies**: Stage 6

---

### 📊 Stage 8: Scoring & Aggregation
**Status**: Pending  
**Priority**: Medium  
**Estimated Effort**: 2-3 hours

**Objectives**:
- Implement weighted score aggregation
- Add stability bonus calculation framework
- Provide detailed score breakdowns

**Tasks**:
- [ ] Implement `bench/core/scoring.py`:
  - Weight-based score aggregation
  - Per-task score calculation
  - Stability bonus framework (multi-run consistency)
  - Score normalization and reporting
- [ ] Add score validation and bounds checking
- [ ] Implement score persistence and loading

**Deliverables**:
- Scoring aggregation system
- Stability bonus calculation
- Score validation and persistence

**Dependencies**: Stage 7

---

### 📈 Stage 9: Reporting System
**Status**: Pending  
**Priority**: Medium  
**Estimated Effort**: 3-4 hours

**Objectives**:
- Generate Markdown and JSON reports
- Create leaderboards and comparisons
- Provide detailed failure analysis

**Tasks**:
- [ ] Implement `bench/core/reporting.py`:
  - Markdown report generation
  - JSON report structure
  - Leaderboard formatting
  - Per-task breakdown with failure reasons
- [ ] Build `scripts/make_report.py`:
  - Consolidated report generation
  - Multi-run comparison
  - Artifact aggregation
- [ ] Add report templates and styling

**Deliverables**:
- Report generation system
- Consolidated reporting script
- Markdown and JSON output formats

**Dependencies**: Stage 8

---

### 🧪 Stage 10: Sample Tests & Fixtures
**Status**: Pending  
**Priority**: High  
**Estimated Effort**: 4-5 hours

**Objectives**:
- Create complete offline test definitions
- Add fixtures and answer keys
- Write comprehensive test suite

**Tasks**:
- [ ] Create test definitions:
  - `tests/offline/task1_metrics.yaml`
  - `tests/offline/task2_ssn_regex.yaml`
  - `tests/offline/task3_exec_summary.yaml`
- [ ] Add fixtures:
  - `fixtures/csv/phishing_sample.csv`
  - `fixtures/text/ssn_validation_lines.txt`
- [ ] Create answer keys:
  - `answer_keys/offline/task1_metrics.json`
  - `answer_keys/offline/task2_lines.json`
- [ ] Write pytest test suite:
  - Unit tests for validators
  - Unit tests for evaluators
  - Integration tests for runner
  - End-to-end workflow tests

**Deliverables**:
- Complete offline test definitions
- Sample fixtures and answer keys
- Comprehensive pytest test suite
- Test documentation

**Dependencies**: Stage 9

---

### 🚀 Stage 11: CI/CD Finalization
**Status**: Pending  
**Priority**: Medium  
**Estimated Effort**: 2-3 hours

**Objectives**:
- Complete GitHub Actions workflow
- Add automated quality checks
- Set up artifact publishing

**Tasks**:
- [ ] Complete `.github/workflows/benchmark.yml`:
  - Automated linting (black, isort, ruff, flake8)
  - Type checking (mypy)
  - Test execution (pytest)
  - Report artifact publishing
- [ ] Add manual Copilot workflow documentation
- [ ] Set up automated dependency updates
- [ ] Add security scanning

**Deliverables**:
- Complete CI/CD pipeline
- Automated quality assurance
- Documentation for manual workflows

**Dependencies**: Stage 10

---

## Phase 2 Features (Optional)

### 🌐 Stage 12: Online Features & Enhancements
**Status**: Pending  
**Priority**: Low  
**Estimated Effort**: 6-8 hours

**Objectives**:
- Add online deep research task
- Implement stability bonus logic
- Create HTML reports with visualizations

**Tasks**:
- [ ] Implement `bench/core/evaluators/deep_research.py`:
  - Plan structure validation (7-10 steps)
  - Risk register validation (≥5 items with likelihood/impact)
  - Source recency checks (≥3 sources within 3 years)
  - Web capability enforcement
- [ ] Add stability bonus implementation:
  - Multi-run variance analysis
  - Consistency scoring across repetitions
  - Statistical significance testing
- [ ] Create HTML report generation:
  - Interactive charts and visualizations
  - Responsive design
  - Export capabilities

**Dependencies**: Stage 11

---

## Phase 3 Extensions (Future)

### 🔮 Stage 13: Advanced Features
**Status**: Pending  
**Priority**: Low  
**Estimated Effort**: 10+ hours

**Objectives**:
- Add Copilot API integration when available
- Expand task library
- Build adjudication interface

**Tasks**:
- [ ] Copilot API adapter (when API becomes available)
- [ ] UI automation with Playwright (fallback option)
- [ ] Expanded task library:
  - Code comprehension tasks
  - Table synthesis tasks
  - Policy extraction tasks
- [ ] Human adjudication interface for borderline cases
- [ ] Advanced analytics and trending

**Dependencies**: Stage 12

---

## Success Criteria

### MVP Acceptance (Stages 1-11)
- [ ] New engineer can clone repo and run `bench.py run` successfully
- [ ] Produces `report.md` with per-task scores for ChatGPT and Manual Copilot
- [ ] Adding new offline test requires **no Python changes** (YAML + answer key only)
- [ ] Scores are deterministic for ChatGPT and repeatable for manual Copilot
- [ ] All code passes linting, type checking, and test suite

### Quality Gates
- [ ] 100% test coverage for core components
- [ ] All code formatted with black and isort
- [ ] No ruff or flake8 violations
- [ ] All functions have type annotations
- [ ] mypy passes with no errors
- [ ] Documentation covers all public APIs

## Risk Mitigation

### Technical Risks
- **OpenAI API Changes**: Use versioned API endpoints, implement adapter pattern
- **Rate Limiting**: Add exponential backoff and retry logic
- **JSON Parsing Failures**: Robust error handling with fallback extraction

### Process Risks
- **Scope Creep**: Strict adherence to staged approach, defer Phase 2/3 features
- **Integration Issues**: Comprehensive integration tests, early end-to-end validation
- **Performance**: Timeout guards, async processing where beneficial

## Notes

- Each stage builds incrementally toward working benchmark system
- Stages 1-11 deliver complete MVP with all core functionality
- Modular design allows easy extension and new test case addition
- Configuration-driven approach minimizes code changes for new tests
- Comprehensive testing ensures reliability and maintainability

---

**Next Action**: Begin Stage 1 - Project Bootstrap & Tooling Setup
