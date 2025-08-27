# Stage 2 Implementation Plan: Repository Structure & Scaffolding

**Project**: ChatGPT vs Microsoft Copilot Evaluation Harness
**Stage**: 2 of 11
**Created**: 2025-08-27
**Status**: Ready for Implementation

## Overview
**Objective**: Create complete directory structure and Python package hierarchy per specification
**Priority**: High
**Estimated Effort**: 1-2 hours
**Dependencies**: Stage 1 (Completed)

## Detailed Task Breakdown

### 1. Create Core Package Structure
**Directory**: `/bench/`

**Core Modules**:
- `bench/adapters/` - Provider implementations
- `bench/core/` - Core evaluation logic
- `bench/core/evaluators/` - Task-specific evaluators

**Files to Create**:
```
bench/
├── __init__.py (already exists)
├── adapters/
│   ├── __init__.py
│   ├── base.py (placeholder)
│   ├── chatgpt.py (placeholder)
│   └── copilot_manual.py (placeholder)
└── core/
    ├── __init__.py
    ├── runner.py (placeholder)
    ├── validators.py (placeholder)
    ├── scoring.py (placeholder)
    ├── reporting.py (placeholder)
    ├── utils.py (placeholder)
    └── evaluators/
        ├── __init__.py
        ├── metrics_csv.py (placeholder)
        ├── regex_match.py (placeholder)
        ├── exec_summary.py (placeholder)
        └── deep_research.py (placeholder)
```

### 2. Create Test Structure
**Directory**: `/bench/tests/`

**Test Categories**:
- `offline/` - Offline task definitions
- `online/` - Online task definitions (optional)

**Files to Create**:
```
bench/tests/
├── __init__.py
├── offline/
│   ├── __init__.py
│   ├── task1_metrics.yaml (placeholder)
│   ├── task2_ssn_regex.yaml (placeholder)
│   └── task3_exec_summary.yaml (placeholder)
└── online/
    ├── __init__.py
    └── deep_research_agentic_ai.yaml (placeholder)
```

### 3. Create Data Directories
**Directories**: Test fixtures and answer keys

**Structure**:
```
fixtures/
├── csv/
│   ├── .gitkeep
│   └── phishing_sample.csv (placeholder)
└── text/
    ├── .gitkeep
    └── ssn_validation_lines.txt (placeholder)

answer_keys/
├── offline/
│   ├── .gitkeep
│   ├── task1_metrics.json (placeholder)
│   └── task2_lines.json (placeholder)
└── online/
    └── .gitkeep
```

### 4. Create Configuration Structure
**Directory**: `/configs/`

**Configuration Files**:
```
configs/
├── providers.yaml (placeholder)
├── weights.default.yaml (placeholder)
└── runmatrix.yaml (placeholder)
```

### 5. Create Schema Structure
**Directory**: `/schemas/`

**Schema Files**:
```
schemas/
├── test_case.schema.json (placeholder)
└── rubric.schema.json (placeholder)
```

### 6. Create Scripts Structure
**Directory**: `/scripts/`

**Script Files**:
```
scripts/
├── bench.py (placeholder - will extend existing CLI)
└── make_report.py (placeholder)
```

### 7. Create Results Structure
**Directory**: `/results/`

**Results Management**:
```
results/
└── .gitkeep
```

### 8. Create Project Configuration Files
**Files**: Git and project configuration

**Files to Create**:
- `.gitignore` - Python project gitignore
- Update existing files as needed

## Implementation Steps

### Step 1: Create Directory Structure
```bash
# Core package directories
mkdir -p bench/adapters
mkdir -p bench/core/evaluators

# Test directories
mkdir -p bench/tests/offline
mkdir -p bench/tests/online

# Data directories
mkdir -p fixtures/csv
mkdir -p fixtures/text
mkdir -p answer_keys/offline
mkdir -p answer_keys/online

# Configuration directories
mkdir -p configs
mkdir -p schemas
mkdir -p scripts
mkdir -p results
```

### Step 2: Create Python Package Files
- Add `__init__.py` files to all Python packages
- Include proper docstrings and module-level documentation
- Set up package imports and exports

### Step 3: Create Placeholder Files
- Add placeholder implementations for core modules
- Include proper type hints and docstrings
- Add TODO comments for Stage 3+ implementation

### Step 4: Create Configuration Placeholders
- Add YAML configuration templates
- Include JSON schema placeholders
- Add documentation comments

### Step 5: Create Git Configuration
- Add comprehensive `.gitignore` for Python projects
- Add `.gitkeep` files for empty directories
- Ensure proper version control setup

## Success Criteria

- [ ] All directories created per specification layout
- [ ] All Python packages have `__init__.py` files
- [ ] Placeholder files created for all core modules
- [ ] Configuration structure established
- [ ] Git configuration properly set up
- [ ] `uv run python -c "import bench.core.runner"` succeeds (imports work)
- [ ] Directory structure matches specification exactly
- [ ] All placeholder files have proper docstrings and type hints

## Deliverables

1. **Complete Directory Structure** - All directories per specification
2. **Python Package Hierarchy** - Proper `__init__.py` files and imports
3. **Placeholder Implementations** - Skeleton files for all core modules
4. **Configuration Templates** - YAML and JSON placeholder files
5. **Git Configuration** - `.gitignore` and `.gitkeep` files
6. **Documentation** - Module docstrings and structure documentation

## Quality Gates

- All Python files have proper docstrings
- All modules can be imported without errors
- Directory structure exactly matches specification
- Git ignores appropriate files and tracks structure
- No linting errors in placeholder files
- Type hints present in all function signatures

## File Templates

### Python Module Template
```python
"""Module description.

This module provides [functionality description].
"""

from typing import Any

# TODO: Implement in Stage [X]


def placeholder_function() -> None:
    """Placeholder function for future implementation."""
    raise NotImplementedError("Implementation pending in Stage [X]")
```

### YAML Configuration Template
```yaml
# Configuration file description
# TODO: Implement in Stage [X]

# Placeholder structure
example_key: "placeholder_value"
```

### JSON Schema Template
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "placeholder-schema",
  "title": "Placeholder Schema",
  "description": "TODO: Implement in Stage [X]",
  "type": "object"
}
```

## Risk Mitigation

### Technical Risks
- **Import Conflicts**: Use absolute imports and proper package structure
- **Directory Permissions**: Ensure proper file system permissions
- **Git Tracking**: Use `.gitkeep` for empty directories that need tracking

### Process Risks
- **Structure Mismatch**: Carefully follow specification layout exactly
- **Missing Files**: Use checklist to verify all required files created
- **Package Issues**: Test imports after each package creation

## Notes

- This stage creates the foundation for all subsequent development
- Placeholder files include proper structure for easy Stage 3+ implementation
- All directories match the specification layout exactly
- Python package structure enables proper imports and modularity
- Configuration structure supports declarative test definitions

---

**Next Action**: Begin implementation of directory structure and package hierarchy
**Following Stage**: Stage 3 - Schema & Configuration Foundation
