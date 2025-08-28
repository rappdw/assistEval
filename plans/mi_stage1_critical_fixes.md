# Missing Implementation Stage 1: Critical Fixes

**Project**: ChatGPT vs Microsoft Copilot Evaluation Harness
**Stage**: MI-1 (Missing Implementation - Critical Fixes)
**Priority**: High
**Estimated Effort**: 2-3 hours
**Dependencies**: None

## Overview

Stage 1 addresses critical gaps that prevent users from successfully following documentation and using the system as specified. These are primarily naming inconsistencies, missing convenience files, and documentation updates that can be completed quickly without complex logic changes.

## Objectives

- Fix CLI script naming inconsistency between specification and implementation
- Create missing test template directory and files for user convenience
- Update all documentation to reflect actual implementation
- Verify and fix answer key naming consistency
- Ensure package entry point configuration works correctly

## Tasks

### Task 1.1: CLI Script Naming Resolution
**Priority**: Critical
**Effort**: 30 minutes

**Issue**: Specification requires `scripts/bench.py` but implementation uses `scripts/run_bench.py`

**Decision**: Rename to match specification for consistency

**Actions**:
- [ ] Rename `scripts/run_bench.py` to `scripts/bench.py`
- [ ] Update any internal imports or references
- [ ] Verify CLI functionality after rename
- [ ] Test entry point: `python scripts/bench.py --help`

**Acceptance Criteria**:
- CLI script accessible as `scripts/bench.py`
- All functionality preserved after rename
- Help system displays correctly

### Task 1.2: Create Test Templates Directory
**Priority**: High
**Effort**: 45 minutes

**Issue**: Missing `tests/_templates/` directory referenced in documentation

**Actions**:
- [ ] Create `tests/_templates/` directory
- [ ] Create template files:
  - `tests/_templates/offline_metrics.yaml` (Task 1 template)
  - `tests/_templates/offline_regex.yaml` (Task 2 template)
  - `tests/_templates/offline_summary.yaml` (Task 3 template)
  - `tests/_templates/online_research.yaml` (Deep research template)
- [ ] Add comprehensive comments and placeholder values in templates
- [ ] Create `tests/_templates/README.md` with usage instructions

**Template Structure**:
```yaml
# Template for [TASK_TYPE] - [DESCRIPTION]
# Copy this file and customize for your specific test case

id: "offline.task_name.description"
name: "Task Name - Description"
category: "offline"  # or "online"

capability_profile:
  web: "forbidden"    # forbidden|allowed|required
  json_required: true
  retries: 1

prompt:
  system: |
    # System prompt template
  user: |
    # User prompt template with placeholders

expectation:
  schema_name: "task_schema_name"
  fields:
    # Field extraction definitions

scoring:
  evaluator: "evaluator_name"
  config:
    # Evaluator-specific configuration
```

**Acceptance Criteria**:
- Templates directory exists with all 4 template files
- Templates contain comprehensive examples and documentation
- Users can copy and modify templates to create new tests

### Task 1.3: Documentation Updates
**Priority**: High
**Effort**: 45 minutes

**Issue**: Multiple documentation files reference incorrect filenames and placeholders

**Actions**:
- [ ] Update `README.md`:
  - Fix all `scripts/bench.py` command examples
  - Replace `<repository-url>` placeholder with actual instructions
  - Verify all code examples are accurate
- [ ] Update `Specification.md` if needed for consistency
- [ ] Update any other documentation files with incorrect references
- [ ] Verify installation instructions work end-to-end

**Files to Update**:
- `README.md` (primary focus)
- `docs/MANUAL_EVALUATION.md` (if it contains CLI references)
- Any other markdown files with CLI examples

**Acceptance Criteria**:
- All command examples use correct script name
- Installation instructions are complete and accurate
- Documentation matches actual implementation

### Task 1.4: Answer Key Verification
**Priority**: Medium
**Effort**: 30 minutes

**Issue**: Task 3 answer key naming may be inconsistent

**Actions**:
- [ ] Examine `bench/tests/offline/task3_exec_summary.yaml`
- [ ] Check what answer key filename it expects
- [ ] Verify `answer_keys/offline/task3_structure.json` content matches expectations
- [ ] Rename or restructure answer key if needed
- [ ] Test Task 3 evaluation with corrected answer key

**Acceptance Criteria**:
- Task 3 answer key filename matches test expectations
- Answer key content structure is correct for evaluator
- Task 3 evaluation runs without errors

### Task 1.5: Package Entry Point Verification
**Priority**: Low
**Effort**: 15 minutes

**Issue**: `pyproject.toml` entry point may not match implementation

**Actions**:
- [ ] Examine `bench/cli.py` to verify `main()` function exists
- [ ] Test package installation: `uv install -e .`
- [ ] Test entry point: `bench --help`
- [ ] Fix entry point configuration if needed

**Acceptance Criteria**:
- Package installs correctly with `uv install -e .`
- `bench` command works from command line
- Entry point matches actual implementation

## Quality Assurance

### Testing Requirements
- [ ] All CLI commands work with new script name
- [ ] Template files are valid YAML and follow schema
- [ ] Documentation examples can be copy-pasted and executed
- [ ] Answer keys work with their respective evaluators
- [ ] Package installation and entry point function correctly

### Code Quality
- [ ] All changes pass pre-commit hooks
- [ ] No linting errors introduced
- [ ] All files follow project formatting standards

## Risk Mitigation

**Risk**: Breaking existing functionality during rename
**Mitigation**: Test all CLI commands after each change

**Risk**: Template files don't match current schema
**Mitigation**: Validate templates against JSON schemas

**Risk**: Documentation updates miss some references
**Mitigation**: Search entire codebase for old script name references

## Success Criteria

- [ ] Users can follow README instructions end-to-end successfully
- [ ] CLI script name matches specification (`scripts/bench.py`)
- [ ] Test templates directory exists with working examples
- [ ] All documentation examples are accurate and executable
- [ ] Answer keys work correctly with their evaluators
- [ ] Package installation works via entry point

## Dependencies for Stage 2

Stage 1 completion enables:
- Users to successfully run the system following documentation
- Developers to create new tests using templates
- Stage 2 focus on complex evaluator implementation

## Deliverables

1. Renamed CLI script (`scripts/bench.py`)
2. Complete test templates directory (`tests/_templates/`)
3. Updated documentation with accurate examples
4. Verified answer key consistency
5. Working package entry point configuration
6. Updated verification checklist confirming all fixes work

**Next Stage**: MI-2 (Deep Research Evaluator & Advanced Features)
