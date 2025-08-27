# Stage 1 Implementation Plan: Project Bootstrap & Tooling Setup

**Project**: ChatGPT vs Microsoft Copilot Evaluation Harness
**Stage**: 1 of 11
**Created**: 2025-08-27
**Status**: Ready for Implementation

## Overview
**Objective**: Initialize modern Python project structure with proper development toolchain per user preferences
**Priority**: High
**Estimated Effort**: 1-2 hours
**Dependencies**: None

## Detailed Task Breakdown

### 1. Initialize Project with UV Package Manager
**File**: Root directory initialization
- Run `uv init` to create basic project structure
- Verify UV is properly configured for dependency management
- Set Python version requirement (3.11+ recommended for modern features)

### 2. Create pyproject.toml Configuration
**File**: `/pyproject.toml`

**Core Dependencies**:
- `openai` - ChatGPT API integration
- `pydantic` - Data validation and settings
- `jsonschema` - JSON schema validation
- `pyyaml` - YAML configuration parsing
- `jsonpath-ng` - JSONPath field extraction
- `click` - CLI framework
- `rich` - Enhanced terminal output

**Development Dependencies**:
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `black` - Code formatting
- `isort` - Import sorting
- `ruff` - Fast Python linter
- `mypy` - Type checking
- `flake8` - Style checking
- `pre-commit` - Git hooks

**Tool Configurations**:
```toml
[tool.black]
line-length = 88
target-version = ['py311']

[tool.isort]
profile = "black"
multi_line_output = 3

[tool.ruff]
line-length = 88
target-version = "py311"
select = ["E", "F", "W", "C90", "I", "N", "UP", "S", "B", "A", "C4", "T20"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
addopts = "--cov=bench --cov-report=term-missing --cov-report=html"
```

### 3. Configure Development Tools
**Files**: Tool-specific configuration files
- `.pre-commit-config.yaml` - Git hooks for automated quality checks
- `.flake8` - Additional style configuration
- `pytest.ini` or pyproject.toml pytest section

**Pre-commit hooks**:
- black (code formatting)
- isort (import sorting)
- ruff (linting)
- mypy (type checking)
- trailing-whitespace removal
- end-of-file-fixer

### 4. Create GitHub Actions CI Skeleton
**File**: `.github/workflows/ci.yml`

**Workflow triggers**:
- Push to main branch
- Pull requests
- Manual workflow dispatch

**Jobs**:
- **Lint & Format**: black, isort, ruff, flake8
- **Type Check**: mypy
- **Test**: pytest with coverage
- **Security**: Basic security scanning

**Matrix strategy**:
- Python versions: 3.11, 3.12
- OS: ubuntu-latest (expand later if needed)

### 5. Write Initial README.md
**File**: `/README.md`

**Sections**:
- Project overview and objectives
- Quick start guide
- Development setup instructions
- Project structure overview
- Contributing guidelines
- License information

**Setup instructions**:
```bash
# Clone repository
git clone <repo-url>
cd assistEval

# Install dependencies with uv
uv sync

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
ruff check .
black --check .
isort --check-only .
mypy .
```

## Implementation Steps

1. **Initialize UV project structure**
   ```bash
   uv init --name bench
   ```

2. **Create comprehensive pyproject.toml**
   - Add all dependencies and tool configurations
   - Set up project metadata and entry points

3. **Configure development toolchain**
   - Set up pre-commit hooks
   - Configure all linting and formatting tools
   - Ensure consistent code quality standards

4. **Create GitHub Actions workflow**
   - Basic CI pipeline with quality gates
   - Automated testing and linting
   - Artifact publishing setup (for future use)

5. **Write project documentation**
   - Clear setup instructions
   - Development workflow documentation
   - Project architecture overview

## Success Criteria

- [ ] `uv sync` successfully installs all dependencies
- [ ] All development tools run without errors on empty project
- [ ] Pre-commit hooks execute successfully
- [ ] GitHub Actions workflow passes on initial commit
- [ ] README provides clear setup instructions for new developers
- [ ] Project follows user's preferred toolchain (uv, pytest, black, isort, ruff, mypy, flake8)

## Deliverables

1. **pyproject.toml** - Complete dependency and tool configuration
2. **.github/workflows/ci.yml** - CI/CD pipeline skeleton
3. **README.md** - Project overview and setup instructions
4. **Development configuration files** - Pre-commit, tool configs
5. **Working development environment** - Ready for Stage 2 implementation

## Quality Gates

- All code formatted with black and isort
- No ruff or flake8 violations
- All functions have type annotations where applicable
- mypy passes with no errors
- Documentation covers setup and development workflow

## Risk Mitigation

### Technical Risks
- **UV compatibility**: Verify UV works correctly on target systems
- **Tool conflicts**: Test all development tools work together
- **CI failures**: Validate GitHub Actions workflow locally first

### Process Risks
- **Setup complexity**: Provide clear, step-by-step instructions
- **Tool learning curve**: Document common commands and workflows
- **Version conflicts**: Pin dependency versions for reproducibility

## Notes

- This foundation enables rapid development while maintaining high code quality
- All tool configurations align with user preferences
- Modular setup allows easy extension and modification
- Comprehensive testing ensures reliability from the start

---

**Next Action**: Begin implementation of Stage 1 tasks
**Following Stage**: Stage 2 - Repository Structure & Scaffolding
