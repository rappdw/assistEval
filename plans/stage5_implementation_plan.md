# Stage 5 Implementation Plan: Core Runner & CLI

**Project**: ChatGPT vs Microsoft Copilot Evaluation Harness
**Stage**: 5 - Core Runner & CLI
**Created**: 2025-08-27
**Dependencies**: Stage 4 (Provider Abstraction & Adapters)
**Estimated Effort**: 4-5 hours

## Overview

Stage 5 implements the core orchestration engine and command-line interface that ties together the provider system with test execution, results management, and configuration handling. This stage delivers the primary user interface for running benchmarks and managing evaluation workflows.

## Objectives

- **Test Execution Orchestration**: Build the core runner that manages test execution across providers
- **Command-Line Interface**: Create intuitive CLI with subcommands for different operations
- **Results Artifact Management**: Implement comprehensive results storage and organization
- **Configuration Integration**: Connect all configuration systems (providers, weights, run matrix)
- **Error Handling & Recovery**: Robust error handling with graceful degradation
- **Progress Reporting**: Real-time feedback during long-running evaluations

## Architecture Components

### Core Runner (`bench/core/runner.py`)
- **TestRunner**: Main orchestration class
- **ExecutionContext**: Manages test execution state and configuration
- **ResultsCollector**: Handles output collection and artifact storage
- **ProgressReporter**: Provides real-time execution feedback

### CLI Interface (`scripts/run_bench.py`)
- **Main CLI**: Entry point with subcommand routing
- **Run Command**: Execute full matrix or individual tests
- **Validate Command**: Test case and configuration validation
- **Status Command**: Check execution status and results

### Results Management
- **Timestamped Directories**: Organized result storage by execution time
- **Artifact Storage**: Raw outputs, parsed JSON, configuration snapshots
- **Symlink Management**: Latest results pointer for automation

## Detailed Implementation Tasks

### 1. Core Runner Implementation

#### 1.1 Base Runner Architecture
```python
# bench/core/runner.py
class TestRunner:
    """Core orchestration engine for benchmark execution."""

    def __init__(self, config_path: str | None = None):
        """Initialize runner with configuration."""

    def run_matrix(self, matrix_config: str) -> RunResults:
        """Execute full test matrix across providers."""

    def run_single(self, provider: str, test_path: str) -> TestResult:
        """Execute single test with specific provider."""

    def validate_test(self, test_path: str) -> ValidationResult:
        """Validate test case definition."""
```

#### 1.2 Execution Context Management
- Load and validate all configuration files
- Initialize providers from configuration
- Manage test case loading and validation
- Handle execution state and progress tracking
- Coordinate between providers, validators, and evaluators

#### 1.3 Results Collection System
- Create timestamped execution directories
- Store raw provider outputs
- Save parsed and validated JSON
- Capture configuration snapshots
- Generate execution metadata and logs

#### 1.4 Error Handling & Recovery
- Graceful handling of provider failures
- Test case validation errors
- Configuration loading issues
- Partial execution recovery
- Detailed error reporting and logging

### 2. CLI Implementation

#### 2.1 Main CLI Structure
```bash
# Primary commands
python scripts/run_bench.py run --config configs/runmatrix.yaml
python scripts/run_bench.py run --provider chatgpt --test tests/offline/task1_metrics.yaml
python scripts/run_bench.py validate --test tests/offline/task1_metrics.yaml
python scripts/run_bench.py status --run results/run_20250827_160000/
```

#### 2.2 Run Command Implementation
- **Full Matrix Mode**: Execute complete test matrix from configuration
- **Single Test Mode**: Run specific test with specific provider
- **Provider Selection**: Filter execution by provider names
- **Test Set Filtering**: Execute only offline or online tests
- **Repetition Control**: Override default repetition counts
- **Dry Run Mode**: Validate without execution

#### 2.3 Validate Command Implementation
- **Test Case Validation**: Schema compliance and structure checks
- **Configuration Validation**: Provider and matrix configuration checks
- **Fixture Validation**: Ensure required files exist and are accessible
- **Answer Key Validation**: Verify answer keys match test expectations

#### 2.4 Status Command Implementation
- **Execution Status**: Check running or completed executions
- **Results Summary**: Quick overview of latest results
- **Error Reporting**: Display any execution failures or issues
- **Progress Tracking**: Show completion status for long-running executions

### 3. Configuration Integration

#### 3.1 Configuration Loading System
- **Hierarchical Loading**: Support for config overrides and defaults
- **Validation Pipeline**: Ensure all configurations are valid before execution
- **Environment Integration**: Support for environment variable overrides
- **Path Resolution**: Relative path handling for portability

#### 3.2 Provider Integration
- **Dynamic Provider Loading**: Use provider factory from Stage 4
- **Capability Enforcement**: Apply test-specific capability constraints
- **Error Handling**: Graceful degradation when providers fail
- **Retry Logic**: Configurable retry behavior for transient failures

#### 3.3 Test Case Management
- **Discovery System**: Automatic test case discovery from directories
- **Filtering**: Support for test selection by category, tags, or patterns
- **Dependency Checking**: Ensure fixtures and answer keys are available
- **Validation Pipeline**: Schema validation before execution

### 4. Results Management System

#### 4.1 Directory Structure
```
results/
  run_20250827_160000/
    config/
      providers.yaml          # Provider configuration snapshot
      weights.yaml           # Scoring weights snapshot
      runmatrix.yaml         # Execution matrix snapshot
    raw/
      chatgpt/
        task1_metrics.txt     # Raw provider output
        task2_ssn_regex.txt
        task3_exec_summary.txt
      copilot_manual/
        task1_metrics.txt
        task2_ssn_regex.txt
        task3_exec_summary.txt
    parsed/
      chatgpt/
        task1_metrics.json    # Parsed and validated JSON
        task2_ssn_regex.json
        task3_exec_summary.json
      copilot_manual/
        task1_metrics.json
        task2_ssn_regex.json
        task3_exec_summary.json
    metadata/
      execution.json          # Execution metadata and timing
      errors.json            # Error log and failure details
    latest -> ../run_20250827_160000/  # Symlink to latest run
```

#### 4.2 Artifact Management
- **Atomic Writes**: Ensure result integrity during execution
- **Compression**: Optional compression for large result sets
- **Retention Policy**: Configurable cleanup of old results
- **Export Utilities**: Support for result export and sharing

### 5. Progress Reporting & Logging

#### 5.1 Progress Reporting
- **Real-time Updates**: Live progress during execution
- **ETA Calculation**: Estimated time to completion
- **Provider Status**: Current provider and test being executed
- **Error Notifications**: Immediate notification of failures

#### 5.2 Logging System
- **Structured Logging**: JSON-formatted logs for automation
- **Log Levels**: Configurable verbosity (DEBUG, INFO, WARN, ERROR)
- **File Logging**: Persistent logs in results directories
- **Console Output**: User-friendly console progress display

## Implementation Checklist

### Core Runner (`bench/core/runner.py`)
- [ ] Implement `TestRunner` class with configuration loading
- [ ] Add `ExecutionContext` for state management
- [ ] Build `run_matrix()` method for full execution
- [ ] Build `run_single()` method for individual tests
- [ ] Add `validate_test()` method for test validation
- [ ] Implement `ResultsCollector` for artifact management
- [ ] Add comprehensive error handling and recovery
- [ ] Implement progress reporting and logging

### CLI Interface (`scripts/run_bench.py`)
- [ ] Create main CLI entry point with argument parsing
- [ ] Implement `run` subcommand with matrix and single modes
- [ ] Implement `validate` subcommand for test validation
- [ ] Implement `status` subcommand for execution monitoring
- [ ] Add configuration file discovery and loading
- [ ] Add help text and usage examples
- [ ] Implement dry-run mode for validation
- [ ] Add verbose and quiet output modes

### Results Management
- [ ] Implement timestamped directory creation
- [ ] Add raw output storage system
- [ ] Add parsed JSON storage system
- [ ] Implement configuration snapshot system
- [ ] Add metadata and execution log storage
- [ ] Create symlink management for latest results
- [ ] Add result validation and integrity checks
- [ ] Implement cleanup and retention policies

### Configuration Integration
- [ ] Integrate provider factory from Stage 4
- [ ] Add configuration validation pipeline
- [ ] Implement hierarchical configuration loading
- [ ] Add environment variable support
- [ ] Implement path resolution utilities
- [ ] Add configuration override mechanisms
- [ ] Create configuration documentation

### Testing & Quality
- [ ] Unit tests for `TestRunner` class
- [ ] Unit tests for CLI command parsing
- [ ] Integration tests for full execution workflow
- [ ] Tests for error handling and recovery
- [ ] Tests for results artifact management
- [ ] Performance tests for large test matrices
- [ ] Documentation for CLI usage and examples

## Success Criteria

### Functional Requirements
- [ ] **Full Matrix Execution**: Can execute complete test matrix from `runmatrix.yaml`
- [ ] **Single Test Execution**: Can run individual tests with specific providers
- [ ] **Configuration Validation**: Validates all configuration files before execution
- [ ] **Results Storage**: Properly stores all execution artifacts in organized structure
- [ ] **Error Handling**: Gracefully handles provider failures and configuration errors
- [ ] **Progress Reporting**: Provides clear feedback during execution

### Quality Requirements
- [ ] **CLI Usability**: Intuitive command-line interface with helpful error messages
- [ ] **Documentation**: Complete CLI help text and usage examples
- [ ] **Performance**: Efficient execution with minimal overhead
- [ ] **Reliability**: Robust error handling with graceful degradation
- [ ] **Maintainability**: Clean, well-documented code following project standards

### Integration Requirements
- [ ] **Provider Integration**: Seamlessly uses provider factory from Stage 4
- [ ] **Configuration Integration**: Loads and validates all configuration files
- [ ] **Schema Integration**: Uses JSON schemas from Stage 3 for validation
- [ ] **Results Integration**: Prepares artifacts for Stage 6 validation framework

## Dependencies

### Required from Previous Stages
- **Stage 3**: JSON schemas and configuration files
- **Stage 4**: Provider abstraction and adapter implementations
- **pyproject.toml**: Dependencies for CLI libraries (click, rich, etc.)

### External Dependencies
- **click**: Command-line interface framework
- **rich**: Enhanced terminal output and progress bars
- **pydantic**: Configuration validation and parsing
- **pathlib**: Path manipulation utilities

## Risk Mitigation

### Technical Risks
- **Configuration Complexity**: Use pydantic for robust validation and clear error messages
- **Execution Failures**: Implement comprehensive error handling with partial recovery
- **Results Corruption**: Use atomic writes and validation for result integrity
- **Performance Issues**: Profile execution and optimize bottlenecks

### Usability Risks
- **CLI Complexity**: Design intuitive commands with clear help text and examples
- **Error Messages**: Provide actionable error messages with suggested fixes
- **Progress Visibility**: Implement clear progress reporting for long-running executions

## Next Steps

After Stage 5 completion:
1. **Stage 6**: Validation Framework - JSON validation and field extraction
2. **Stage 7**: Offline Task Evaluators - Task-specific scoring logic
3. **Integration Testing**: End-to-end workflow validation with real providers

## Notes

- CLI design should prioritize ease of use for new engineers
- Results structure should support future reporting and analysis tools
- Error handling should provide actionable feedback for troubleshooting
- Progress reporting should work well in both interactive and CI environments
- Configuration system should be extensible for future provider types

---

**Ready for Implementation**: This plan provides comprehensive guidance for implementing the core orchestration engine and CLI interface that will serve as the primary user interaction point for the benchmark system.
