# Missing Implementation Stage 2: Advanced Features

**Project**: ChatGPT vs Microsoft Copilot Evaluation Harness
**Stage**: MI-2 (Missing Implementation - Advanced Features)
**Priority**: Medium
**Estimated Effort**: 4-6 hours
**Dependencies**: MI-1 Complete

## Overview

Stage 2 addresses complex implementation gaps requiring significant development work. This includes completing the deep research evaluator, implementing stability bonus calculations, and verifying advanced system features work correctly.

## Objectives

- Complete deep research evaluator implementation per specification
- Implement and verify stability bonus calculation system
- Validate CI/CD workflows match specification requirements
- Ensure all advanced features work end-to-end

## Tasks

### Task 2.1: Deep Research Evaluator Implementation
**Priority**: High
**Effort**: 3-4 hours

**Issue**: `bench/core/evaluators/deep_research.py` is placeholder code, needs full implementation

**Specification Requirements**:
- Plan structure: 7-10 ordered steps with {goal, method, deliverable}
- Risk register: ≥5 items with {risk, likelihood 1-5, impact 1-5, mitigation}
- Sources: 5-8 entries with ≥3 sources within 3 years
- Web capability enforcement and penalties

**Actions**:
- [ ] **Analyze Current Implementation**:
  - Review existing `deep_research.py` placeholder
  - Examine `tests/online/deep_research_agentic_ai.yaml` test definition
  - Check expected JSON schema structure

- [ ] **Implement Plan Structure Validation**:
  - Validate 7-10 steps in response
  - Check each step has required fields: {goal, method, deliverable}
  - Verify logical sequencing and completeness
  - Award points based on structure quality (0-5 points)

- [ ] **Implement Risk Register Validation**:
  - Validate ≥5 risk items in response
  - Check each risk has: {risk, likelihood 1-5, impact 1-5, mitigation}
  - Validate likelihood and impact are integers 1-5
  - Award points for completeness and quality (0-5 points)

- [ ] **Implement Source Quality Assessment**:
  - Extract 5-8 sources from response
  - Parse publication years from citations
  - Verify ≥3 sources are within last 3 years (current_year - 3)
  - Handle missing dates gracefully
  - Award points for recency and quality

- [ ] **Add Web Capability Enforcement**:
  - Check if web capability was actually used
  - Penalize if web declared but not used
  - Award partial credit for structure when web unavailable
  - Implement explicit assumptions/limitations field validation

- [ ] **Scoring Implementation**:
  ```python
  # Scoring breakdown (10 points total)
  plan_quality: 5 points
    - Structure (2 pts): 7-10 steps with required fields
    - Sequencing (2 pts): Logical flow and dependencies
    - Completeness (1 pt): All deliverables specified

  source_quality: 5 points
    - Count (2 pts): 5-8 sources present
    - Recency (2 pts): ≥3 sources within 3 years
    - Quality (1 pt): Credible sources and proper citations
  ```

- [ ] **Error Handling and Edge Cases**:
  - Handle malformed JSON responses
  - Graceful degradation for missing fields
  - Timeout protection for complex parsing
  - Clear error messages for validation failures

**Acceptance Criteria**:
- Evaluator handles all specification requirements
- Proper scoring breakdown with detailed feedback
- Robust error handling for edge cases
- Integration tests pass with sample responses

### Task 2.2: Stability Bonus Implementation Verification
**Priority**: Medium
**Effort**: 1-2 hours

**Issue**: Stability bonus framework exists but implementation completeness unclear

**Specification Requirements**:
- Run each offline task 3× per provider
- Award 5 points for exact Task 1 numbers across runs
- Award points for Task 2/3 structural consistency
- Partial consistency should be prorated

**Actions**:
- [ ] **Examine Current Implementation**:
  - Review `bench/core/scoring.py` stability bonus code
  - Check if multi-run variance analysis exists
  - Verify consistency calculation logic

- [ ] **Implement Missing Components** (if needed):
  - Task 1 exact numeric consistency checking
  - Task 2/3 structural consistency validation
  - Prorated scoring for partial consistency
  - Statistical significance testing

- [ ] **Add Consistency Metrics**:
  ```python
  # Task 1: Exact numeric consistency
  - precision, recall, f1, accuracy must be identical across runs
  - confusion_matrix values must be identical
  - Award 5 points for perfect consistency, 0 for any variance

  # Task 2: Structural consistency
  - Regex pattern should be identical or functionally equivalent
  - Line matches should be consistent
  - Award partial points for minor variations

  # Task 3: Structural consistency
  - Title word count should be consistent
  - Summary word count should be in same range
  - Bullet count should be identical (exactly 3)
  - Award partial points based on consistency level
  ```

- [ ] **Testing and Validation**:
  - Create test cases with consistent and inconsistent runs
  - Verify bonus calculation accuracy
  - Test edge cases (partial consistency scenarios)

**Acceptance Criteria**:
- Stability bonus calculates correctly for all task types
- Prorated scoring works for partial consistency
- Integration with main scoring system functions properly
- Comprehensive test coverage for stability scenarios

### Task 2.3: CI/CD Workflow Verification
**Priority**: Low
**Effort**: 1 hour

**Issue**: Multiple workflow files exist, need to verify specification compliance

**Current Workflows**:
- `benchmark.yml` (specified in documentation)
- `ci.yml` (standard CI pipeline)
- `dependencies.yml` (dependency management)
- `manual-copilot.yml` (manual evaluation workflow)
- `release.yml` (release automation)

**Actions**:
- [ ] **Review Specification Requirements**:
  - Check what `benchmark.yml` should contain
  - Verify automated linting, type checking, testing
  - Confirm artifact publishing requirements

- [ ] **Audit Existing Workflows**:
  - Verify `benchmark.yml` matches specification
  - Check if `ci.yml` duplicates or complements benchmark workflow
  - Ensure manual Copilot workflow is documented properly
  - Validate dependency and release workflows

- [ ] **Consolidate or Document Differences**:
  - Merge redundant workflows if needed
  - Document purpose of each workflow file
  - Ensure specification requirements are met
  - Update documentation if workflow names differ

**Acceptance Criteria**:
- All workflows serve clear, documented purposes
- Specification requirements for CI/CD are met
- No redundant or conflicting workflow logic
- Documentation accurately reflects workflow behavior

### Task 2.4: End-to-End System Validation
**Priority**: High
**Effort**: 1 hour

**Issue**: Need comprehensive testing to ensure all components work together

**Actions**:
- [ ] **MVP Acceptance Testing**:
  - Clone fresh repository and follow setup instructions
  - Run full evaluation matrix with both providers
  - Verify report generation and artifact storage
  - Test single test execution scenarios

- [ ] **Deep Research Task Testing**:
  - Execute online deep research task (if web access available)
  - Verify evaluator handles response correctly
  - Check scoring breakdown and feedback
  - Test web capability enforcement

- [ ] **Stability Bonus Testing**:
  - Run offline tasks with 3 repetitions
  - Verify stability bonus calculation
  - Test with both consistent and inconsistent results
  - Validate scoring aggregation

- [ ] **Error Handling Testing**:
  - Test with malformed provider responses
  - Verify graceful degradation scenarios
  - Check error messages are helpful and clear
  - Test timeout and safety guard functionality

**Acceptance Criteria**:
- All MVP acceptance criteria pass
- Deep research evaluator works correctly
- Stability bonus calculates properly
- System handles errors gracefully
- Performance meets specification requirements

## Quality Assurance

### Testing Requirements
- [ ] Unit tests for deep research evaluator (≥90% coverage)
- [ ] Integration tests for stability bonus calculation
- [ ] End-to-end tests for complete evaluation workflows
- [ ] Error handling tests for edge cases

### Performance Requirements
- [ ] Deep research evaluation completes within reasonable time
- [ ] Stability analysis doesn't significantly impact runtime
- [ ] Memory usage remains within acceptable bounds
- [ ] Timeout guards prevent hanging operations

### Code Quality
- [ ] All new code passes linting and type checking
- [ ] Documentation updated for new features
- [ ] Error messages are clear and actionable
- [ ] Logging provides adequate debugging information

## Risk Mitigation

**Risk**: Deep research evaluator too complex to implement correctly
**Mitigation**: Break into smaller components, extensive testing with sample data

**Risk**: Stability bonus calculation affects performance
**Mitigation**: Implement efficient algorithms, add performance monitoring

**Risk**: Online task testing limited by web access
**Mitigation**: Create comprehensive unit tests, mock web responses for testing

## Success Criteria

- [ ] Deep research evaluator fully implements specification requirements
- [ ] Stability bonus calculation works correctly for all scenarios
- [ ] CI/CD workflows meet specification requirements
- [ ] All MVP acceptance criteria pass end-to-end testing
- [ ] System performance meets specification benchmarks
- [ ] Error handling is robust and user-friendly

## Deliverables

1. **Complete Deep Research Evaluator**:
   - Full implementation in `bench/core/evaluators/deep_research.py`
   - Comprehensive test suite
   - Integration with scoring system

2. **Verified Stability Bonus System**:
   - Working multi-run consistency analysis
   - Prorated scoring implementation
   - Performance optimization

3. **Validated CI/CD Workflows**:
   - Documented workflow purposes
   - Specification compliance verification
   - Updated documentation if needed

4. **End-to-End Validation Report**:
   - MVP acceptance criteria results
   - Performance benchmarks
   - Error handling verification
   - Recommendations for production deployment

**Project Completion**: After MI-2, the system will be 100% compliant with specification requirements and ready for production use.

## Integration with Main Project

Upon completion of MI-2:
- All specification requirements will be met
- System will achieve A+ grade (100% complete)
- Repository will be ready for external users
- Documentation will be fully accurate
- All acceptance criteria will pass
