# Stage 11 Implementation Plan: CI/CD Finalization & Production Readiness

**Project**: ChatGPT vs Microsoft Copilot Evaluation Harness
**Stage**: 11 - CI/CD Finalization
**Status**: Planning
**Priority**: Medium
**Estimated Effort**: 2-3 hours

## Overview

Stage 11 completes the CI/CD pipeline and prepares the evaluation harness for production use. This stage enhances the existing GitHub Actions workflow, adds comprehensive automation, and establishes deployment processes for the benchmark system.

## Current State Analysis

**Completed Components:**
- ✅ Basic CI workflow with lint, type-check, test, and security jobs
- ✅ UV-based dependency management
- ✅ Multi-Python version testing (3.11, 3.12)
- ✅ Code quality gates (black, isort, ruff, flake8, mypy)
- ✅ Test suite with pytest and coverage reporting
- ✅ Security scanning with bandit

**Missing Components:**
- 🔄 Benchmark execution workflow
- 🔄 Automated report generation and publishing
- 🔄 Manual Copilot workflow documentation
- 🔄 Dependency update automation
- 🔄 Release management and versioning
- 🔄 Performance monitoring and regression detection

## Objectives

1. **Complete Benchmark Workflow**: Add automated benchmark execution with artifact publishing
2. **Manual Integration**: Document and streamline manual Copilot evaluation process
3. **Automation Enhancement**: Add dependency updates, release management, and monitoring
4. **Production Readiness**: Ensure system is ready for continuous benchmarking

## Implementation Tasks

### Task 1: Benchmark Execution Workflow
**Priority**: High
**Estimated Time**: 1 hour

Create `.github/workflows/benchmark.yml` for automated benchmark execution:

```yaml
name: Benchmark Execution

on:
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Monday at 6 AM UTC
  workflow_dispatch:
    inputs:
      providers:
        description: 'Providers to test (comma-separated)'
        required: false
        default: 'chatgpt'
      test_set:
        description: 'Test set to run'
        required: false
        default: 'offline'
        type: choice
        options:
        - offline
        - online
        - all

jobs:
  benchmark:
    runs-on: ubuntu-latest
    environment: benchmark

    steps:
    - uses: actions/checkout@v4

    - name: Install uv
      uses: astral-sh/setup-uv@v2
      with:
        version: "latest"

    - name: Set up Python 3.11
      run: uv python install 3.11

    - name: Install dependencies
      run: uv sync --all-extras

    - name: Run ChatGPT benchmark
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      run: |
        uv run python scripts/run_bench.py run \
          --provider chatgpt \
          --test-set ${{ github.event.inputs.test_set || 'offline' }} \
          --output results/run_$(date +%Y%m%d_%H%M%S)

    - name: Generate report
      run: |
        uv run python scripts/make_report.py \
          --results results/ \
          --output benchmark_report.md

    - name: Upload benchmark results
      uses: actions/upload-artifact@v4
      with:
        name: benchmark-results-${{ github.run_number }}
        path: |
          results/
          benchmark_report.md
        retention-days: 90

    - name: Comment PR with results
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v7
      with:
        script: |
          const fs = require('fs');
          const report = fs.readFileSync('benchmark_report.md', 'utf8');
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: `## Benchmark Results\n\n${report}`
          });
```

### Task 2: Manual Copilot Workflow
**Priority**: High
**Estimated Time**: 30 minutes

Create comprehensive documentation for manual Copilot evaluation:

**File**: `.github/workflows/manual-copilot.yml`
```yaml
name: Manual Copilot Evaluation

on:
  workflow_dispatch:
    inputs:
      test_set:
        description: 'Test set to evaluate'
        required: true
        default: 'offline'
        type: choice
        options:
        - offline
        - online

jobs:
  prepare-manual-evaluation:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Install uv
      uses: astral-sh/setup-uv@v2
      with:
        version: "latest"

    - name: Set up Python 3.11
      run: uv python install 3.11

    - name: Install dependencies
      run: uv sync --all-extras

    - name: Generate manual evaluation prompts
      run: |
        uv run python scripts/run_bench.py prepare-manual \
          --provider copilot_manual \
          --test-set ${{ github.event.inputs.test_set }} \
          --output manual_prompts.txt

    - name: Upload manual prompts
      uses: actions/upload-artifact@v4
      with:
        name: manual-copilot-prompts
        path: manual_prompts.txt

    - name: Create evaluation issue
      uses: actions/github-script@v7
      with:
        script: |
          const fs = require('fs');
          const prompts = fs.readFileSync('manual_prompts.txt', 'utf8');
          github.rest.issues.create({
            owner: context.repo.owner,
            repo: context.repo.repo,
            title: `Manual Copilot Evaluation - ${new Date().toISOString().split('T')[0]}`,
            body: `## Manual Copilot Evaluation Required\n\n` +
                  `Please evaluate the following prompts with Microsoft Copilot and paste responses:\n\n` +
                  `${prompts}\n\n` +
                  `Upload completed responses as artifact when done.`,
            labels: ['manual-evaluation', 'copilot']
          });
```

### Task 3: Dependency Management Automation
**Priority**: Medium
**Estimated Time**: 30 minutes

Create `.github/workflows/dependencies.yml`:

```yaml
name: Dependency Updates

on:
  schedule:
    - cron: '0 9 * * 1'  # Weekly on Monday at 9 AM UTC
  workflow_dispatch:

jobs:
  update-dependencies:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4
      with:
        token: ${{ secrets.GITHUB_TOKEN }}

    - name: Install uv
      uses: astral-sh/setup-uv@v2
      with:
        version: "latest"

    - name: Set up Python 3.11
      run: uv python install 3.11

    - name: Update dependencies
      run: |
        uv sync --upgrade
        uv export --format requirements-txt --output-file requirements-updated.txt

    - name: Check for changes
      id: changes
      run: |
        if ! cmp -s uv.lock uv.lock.bak 2>/dev/null; then
          echo "changes=true" >> $GITHUB_OUTPUT
        else
          echo "changes=false" >> $GITHUB_OUTPUT
        fi

    - name: Create Pull Request
      if: steps.changes.outputs.changes == 'true'
      uses: peter-evans/create-pull-request@v5
      with:
        token: ${{ secrets.GITHUB_TOKEN }}
        commit-message: 'chore: update dependencies'
        title: 'chore: update dependencies'
        body: |
          ## Dependency Updates

          This PR updates project dependencies to their latest versions.

          ### Changes
          - Updated uv.lock with latest dependency versions

          ### Testing
          - [ ] All tests pass
          - [ ] Benchmark execution works correctly
          - [ ] No breaking changes detected
        branch: chore/update-dependencies
        delete-branch: true
```

### Task 4: Release Management
**Priority**: Medium
**Estimated Time**: 45 minutes

Create `.github/workflows/release.yml`:

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'
  workflow_dispatch:
    inputs:
      version:
        description: 'Release version (e.g., v1.0.0)'
        required: true

jobs:
  release:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0

    - name: Install uv
      uses: astral-sh/setup-uv@v2
      with:
        version: "latest"

    - name: Set up Python 3.11
      run: uv python install 3.11

    - name: Install dependencies
      run: uv sync --all-extras

    - name: Run full test suite
      run: uv run pytest --cov=bench --cov-report=xml

    - name: Build distribution
      run: uv build

    - name: Generate changelog
      run: |
        echo "# Changelog" > CHANGELOG.md
        git log --pretty=format:"- %s (%h)" $(git describe --tags --abbrev=0 HEAD^)..HEAD >> CHANGELOG.md

    - name: Create GitHub Release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: ${{ github.ref }}
        release_name: Release ${{ github.ref }}
        body_path: CHANGELOG.md
        draft: false
        prerelease: false

    - name: Upload release assets
      uses: actions/upload-release-asset@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        upload_url: ${{ steps.create_release.outputs.upload_url }}
        asset_path: ./dist/
        asset_name: benchmark-harness-dist
        asset_content_type: application/zip
```

### Task 5: Performance Monitoring
**Priority**: Low
**Estimated Time**: 30 minutes

Add performance regression detection to existing CI:

**Enhancement to `.github/workflows/ci.yml`**:
```yaml
  performance:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'

    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0

    - name: Install uv
      uses: astral-sh/setup-uv@v2
      with:
        version: "latest"

    - name: Set up Python 3.11
      run: uv python install 3.11

    - name: Install dependencies
      run: uv sync --all-extras

    - name: Run performance benchmarks
      run: |
        uv run python -m pytest tests/test_performance.py \
          --benchmark-json=benchmark.json

    - name: Compare with baseline
      run: |
        uv run python scripts/compare_performance.py \
          --current benchmark.json \
          --baseline .github/baseline-performance.json \
          --threshold 10
```

## Documentation Updates

### Task 6: Manual Workflow Documentation
**Priority**: High
**Estimated Time**: 15 minutes

Create `docs/MANUAL_EVALUATION.md`:

```markdown
# Manual Copilot Evaluation Guide

## Overview
This guide explains how to conduct manual evaluations with Microsoft Copilot when API access is not available.

## Process

### 1. Trigger Manual Evaluation
- Go to Actions → Manual Copilot Evaluation
- Select test set (offline/online)
- Click "Run workflow"

### 2. Download Prompts
- Wait for workflow completion
- Download "manual-copilot-prompts" artifact
- Extract prompts.txt file

### 3. Evaluate with Copilot
For each prompt in the file:
1. Copy the system prompt to Copilot
2. Copy the user prompt to Copilot
3. Copy Copilot's response to a response file
4. Label response with task ID

### 4. Upload Results
- Create responses.txt with format:
  ```
  TASK_ID: offline.task1.metrics_csv
  RESPONSE: {copilot response here}
  ---
  TASK_ID: offline.task2.ssn_regex
  RESPONSE: {copilot response here}
  ---
  ```
- Upload as artifact to the evaluation issue

### 5. Process Results
- Run: `python scripts/run_bench.py process-manual --responses responses.txt`
- Results will be generated in standard format
```

## Success Criteria

### Functional Requirements
- [ ] Automated benchmark execution workflow operational
- [ ] Manual Copilot evaluation process documented and tested
- [ ] Dependency updates automated with PR creation
- [ ] Release management workflow functional
- [ ] Performance regression detection active

### Quality Requirements
- [ ] All workflows pass security review
- [ ] Documentation complete and accessible
- [ ] Error handling robust across all workflows
- [ ] Artifact retention policies appropriate
- [ ] Secrets management secure

### Integration Requirements
- [ ] Workflows integrate with existing CI/CD
- [ ] Manual processes clearly documented
- [ ] Automated processes require minimal intervention
- [ ] Results consistently formatted and accessible

## Dependencies

**Prerequisites:**
- Stage 10 completion (Sample Tests & Fixtures)
- OpenAI API key configured in repository secrets
- GitHub repository permissions for workflow execution

**External Dependencies:**
- GitHub Actions runner availability
- OpenAI API service availability
- UV package manager stability

## Risk Mitigation

### Technical Risks
- **API Rate Limits**: Implement exponential backoff and retry logic
- **Workflow Failures**: Add comprehensive error handling and notifications
- **Secret Exposure**: Use GitHub secrets and environment protection

### Process Risks
- **Manual Process Complexity**: Detailed documentation and validation steps
- **Result Consistency**: Standardized formats and validation checks
- **Maintenance Overhead**: Automated dependency updates and monitoring

## Testing Strategy

### Workflow Testing
1. Test each workflow in isolation
2. Verify artifact generation and upload
3. Validate secret handling and security
4. Test manual process end-to-end

### Integration Testing
1. Full benchmark execution with ChatGPT
2. Manual Copilot evaluation simulation
3. Release process validation
4. Performance regression detection

## Deliverables

1. **GitHub Workflows**:
   - `benchmark.yml` - Automated benchmark execution
   - `manual-copilot.yml` - Manual evaluation orchestration
   - `dependencies.yml` - Dependency management
   - `release.yml` - Release automation

2. **Documentation**:
   - Manual evaluation guide
   - Workflow troubleshooting guide
   - Release process documentation

3. **Scripts**:
   - Performance comparison utilities
   - Manual result processing tools
   - Workflow helper scripts

4. **Configuration**:
   - Environment protection rules
   - Secret management setup
   - Artifact retention policies

## Post-Implementation

### Monitoring
- Weekly automated benchmark runs
- Dependency update notifications
- Performance regression alerts
- Manual evaluation completion tracking

### Maintenance
- Quarterly workflow review and updates
- Annual security audit of CI/CD processes
- Regular documentation updates
- Performance baseline updates

---

**Next Steps**: Begin implementation with Task 1 (Benchmark Execution Workflow)
