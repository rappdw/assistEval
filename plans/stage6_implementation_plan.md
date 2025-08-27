# Stage 6: Validation Framework Implementation Plan

**Project**: ChatGPT vs Microsoft Copilot Evaluation Harness
**Stage**: 6 - Validation Framework
**Created**: 2025-08-27
**Status**: Ready for Implementation
**Priority**: Medium
**Estimated Effort**: 3-4 hours

## Overview

Stage 6 implements the validation framework that bridges the gap between raw provider outputs and structured evaluation. This stage builds the foundation for all subsequent evaluators by providing robust JSON validation, field extraction, and structural checks.

## Objectives

- **JSON Schema Validation**: Validate provider outputs against test case schemas
- **Field Extraction System**: Extract specific fields using JSONPath-like selectors
- **Structural Validation**: Word counts, bullet counting, format validation
- **Content Normalization**: Text processing and cleanup utilities
- **Safety Guards**: Timeout protection and error handling

## Architecture Context

```
Provider Output (raw text)
    ↓
Validators (Stage 6) ← JSON Schemas (Stage 3)
    ↓
Structured Data
    ↓
Evaluators (Stage 7) ← Answer Keys
    ↓
Scores (Stage 8)
```

Stage 6 sits between the core runner (Stage 5) and the task evaluators (Stage 7), providing the critical data validation and extraction layer.

## Implementation Tasks

### 1. Core Validation Framework (`bench/core/validators.py`)

**Priority**: High
**Estimated Time**: 2 hours

#### 1.1 JSON Schema Validation Engine
```python
class SchemaValidator:
    """Validates JSON against test case schemas."""

    def __init__(self, schema_dir: Path)
    def validate_response(self, response: dict, schema_name: str) -> ValidationResult
    def load_schema(self, schema_name: str) -> dict
```

**Features**:
- Load JSON schemas from `schemas/` directory
- Validate provider responses against expected schemas
- Generate detailed error messages with field-level feedback
- Support for custom validation rules beyond basic JSON Schema

#### 1.2 Field Extraction System
```python
class FieldExtractor:
    """Extracts fields using JSONPath-like selectors."""

    def extract_fields(self, data: dict, field_specs: list) -> dict
    def extract_single_field(self, data: dict, path: str, field_type: str) -> Any
    def validate_field_type(self, value: Any, expected_type: str) -> bool
```

**Features**:
- JSONPath-style field extraction (`$.task1_data_metrics.precision`)
- Type validation (string, number, integer, boolean, array)
- Nested object navigation
- Array indexing and filtering
- Error handling for missing or malformed fields

#### 1.3 Structural Validation
```python
class StructuralValidator:
    """Validates structural requirements like word counts."""

    def validate_word_count(self, text: str, min_words: int, max_words: int) -> ValidationResult
    def validate_bullet_count(self, bullets: list, expected_count: int) -> ValidationResult
    def validate_title_length(self, title: str, max_words: int) -> ValidationResult
    def count_sentences(self, text: str) -> int
    def calculate_avg_sentence_length(self, text: str) -> float
```

**Features**:
- Word counting with configurable tokenization
- Bullet point validation (top-level only, no nesting)
- Title length constraints
- Sentence analysis for tone evaluation
- Text structure validation

#### 1.4 Content Normalization
```python
class ContentNormalizer:
    """Normalizes and cleans text content."""

    def normalize_whitespace(self, text: str) -> str
    def extract_json_from_text(self, text: str) -> dict
    def clean_markdown_formatting(self, text: str) -> str
    def normalize_numbers(self, value: str) -> float
```

**Features**:
- Whitespace normalization
- JSON extraction from mixed text responses
- Markdown formatting cleanup
- Number parsing and normalization
- Unicode handling

### 2. Utility Functions (`bench/core/utils.py`)

**Priority**: Medium
**Estimated Time**: 1 hour

#### 2.1 Text Processing Utilities
```python
def tokenize_text(text: str, method: str = "whitespace") -> list[str]
def count_words(text: str, exclude_stopwords: bool = False) -> int
def extract_bullets(text: str) -> list[str]
def parse_title(text: str) -> str
def clean_text(text: str) -> str
```

#### 2.2 Safety and Timeout Guards
```python
class TimeoutGuard:
    """Provides timeout protection for operations."""

    def __init__(self, timeout_seconds: float)
    def __enter__(self)
    def __exit__(self, exc_type, exc_val, exc_tb)

def safe_regex_match(pattern: str, text: str, timeout_ms: int = 100) -> bool
def safe_json_parse(text: str) -> dict | None
```

#### 2.3 Seed Management and Reproducibility
```python
def set_random_seed(seed: int) -> None
def generate_deterministic_hash(data: str) -> str
def ensure_reproducible_ordering(items: list) -> list
```

### 3. Validation Result Types

**Priority**: High
**Estimated Time**: 30 minutes

#### 3.1 Result Data Classes
```python
@dataclass
class ValidationResult:
    """Result of validation operation."""
    valid: bool
    errors: list[str]
    warnings: list[str]
    extracted_data: dict | None = None
    metadata: dict = field(default_factory=dict)

@dataclass
class FieldExtractionResult:
    """Result of field extraction."""
    success: bool
    value: Any
    field_path: str
    error_message: str | None = None

@dataclass
class StructuralValidationResult:
    """Result of structural validation."""
    passed: bool
    actual_value: Any
    expected_range: tuple | Any
    score: float = 0.0
    details: str = ""
```

### 4. Integration Points

**Priority**: High
**Estimated Time**: 30 minutes

#### 4.1 Runner Integration
- Integrate validators into `TestRunner.run_single()` and `TestRunner.run_matrix()`
- Add validation step between provider invocation and evaluation
- Store validation results in run artifacts

#### 4.2 Configuration Integration
- Load validation settings from test case YAML files
- Support for custom validation rules per test
- Schema name mapping from test definitions

## Implementation Checklist

### Core Validation Framework
- [ ] Implement `SchemaValidator` class with JSON Schema support
- [ ] Build `FieldExtractor` with JSONPath-like functionality
- [ ] Create `StructuralValidator` for word counts and formatting
- [ ] Add `ContentNormalizer` for text cleanup
- [ ] Write comprehensive error handling and logging

### Utility Functions
- [ ] Implement text processing utilities (tokenization, word counting)
- [ ] Add timeout guards for regex operations
- [ ] Create seed management functions
- [ ] Build JSON extraction helpers
- [ ] Add Unicode and encoding support

### Data Types and Results
- [ ] Define validation result data classes
- [ ] Create field extraction result types
- [ ] Add structural validation result types
- [ ] Implement result serialization for artifacts

### Integration and Testing
- [ ] Integrate validators into core runner
- [ ] Add configuration loading for validation settings
- [ ] Write unit tests for all validator classes
- [ ] Create integration tests with sample data
- [ ] Add performance benchmarks for timeout guards

### Documentation and Examples
- [ ] Document JSONPath syntax supported
- [ ] Create validation configuration examples
- [ ] Write troubleshooting guide for common validation errors
- [ ] Add performance tuning guidelines

## Success Criteria

### Functional Requirements
- [ ] **JSON Schema Validation**: All test case responses validate against schemas
- [ ] **Field Extraction**: JSONPath selectors extract expected fields correctly
- [ ] **Structural Validation**: Word counts, bullet counts work accurately
- [ ] **Error Handling**: Clear error messages for validation failures
- [ ] **Performance**: Validation completes within reasonable time limits

### Quality Requirements
- [ ] **Type Safety**: Full mypy compliance with proper type annotations
- [ ] **Test Coverage**: 95%+ coverage for validation components
- [ ] **Documentation**: All public APIs documented with examples
- [ ] **Error Messages**: User-friendly validation error descriptions

### Integration Requirements
- [ ] **Runner Integration**: Seamless integration with Stage 5 runner
- [ ] **Configuration**: Loads validation settings from test definitions
- [ ] **Artifact Storage**: Validation results stored in run directories
- [ ] **Evaluator Preparation**: Provides clean data for Stage 7 evaluators

## Dependencies

### Required (Stage 5)
- ✅ Core runner implementation
- ✅ Test execution orchestration
- ✅ Results artifact management
- ✅ Configuration loading system

### Required (Stage 3)
- ✅ JSON schemas for test cases
- ✅ Test case YAML structure
- ✅ Configuration file formats

### External Dependencies
- `jsonschema` - JSON Schema validation
- `jsonpath-ng` - JSONPath field extraction
- `pydantic` - Data validation and parsing
- `regex` - Advanced regex with timeout support

## Risk Mitigation

### Technical Risks
- **JSON Parsing Failures**: Implement robust fallback extraction methods
- **Regex Timeout Issues**: Use timeout guards and safe regex compilation
- **Schema Evolution**: Version schemas and provide migration utilities
- **Performance Bottlenecks**: Profile validation operations and optimize hot paths

### Integration Risks
- **Data Format Mismatches**: Comprehensive testing with real provider outputs
- **Configuration Complexity**: Clear documentation and validation examples
- **Error Propagation**: Proper error handling without breaking evaluation pipeline

## Testing Strategy

### Unit Tests
- Schema validation with valid/invalid inputs
- Field extraction with various JSONPath expressions
- Structural validation with edge cases
- Content normalization with different text formats

### Integration Tests
- End-to-end validation with sample test cases
- Integration with core runner pipeline
- Configuration loading and error handling
- Performance testing with large responses

### Test Data
- Sample provider responses for each task type
- Invalid JSON responses for error handling
- Edge cases for structural validation
- Performance test data for timeout scenarios

## Future Extensions

### Phase 2 Enhancements
- **Custom Validators**: Plugin system for task-specific validation
- **Validation Caching**: Cache validation results for repeated runs
- **Advanced JSONPath**: Support for more complex path expressions
- **Validation Analytics**: Metrics on validation success rates

### Phase 3 Features
- **Schema Evolution**: Automatic schema migration and versioning
- **Validation UI**: Web interface for validation rule configuration
- **ML-Based Validation**: Use ML models for semantic validation
- **Real-time Validation**: Stream validation for long responses

## Deliverables

1. **`bench/core/validators.py`** - Complete validation framework
2. **`bench/core/utils.py`** - Text processing and utility functions
3. **Validation result types** - Data classes for validation results
4. **Integration code** - Runner integration points
5. **Unit test suite** - Comprehensive tests for all validators
6. **Documentation** - API docs and usage examples
7. **Performance benchmarks** - Validation timing and optimization data

## Next Steps

After Stage 6 completion:
1. **Stage 7**: Implement offline task evaluators using validation framework
2. **Stage 8**: Build scoring and aggregation system
3. **Stage 9**: Create reporting system with validation insights

---

**Ready for Implementation**: All dependencies satisfied, clear requirements defined, comprehensive testing strategy in place.
