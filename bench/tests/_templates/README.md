# Test Templates

This directory contains template files for creating new test cases in the evaluation harness. Each template provides a complete structure with placeholders and instructions for customization.

## Available Templates

### Offline Tasks

#### `offline_metrics.yaml`
**Purpose**: Classification metrics calculation from CSV data
**Use Case**: Tasks requiring precision, recall, F1, accuracy, and confusion matrix calculation
**Example**: Phishing detection, fraud classification, sentiment analysis

#### `offline_regex.yaml`
**Purpose**: Regular expression pattern validation
**Use Case**: Tasks requiring regex creation with specific constraints
**Example**: SSN validation, email pattern matching, data format validation

#### `offline_summary.yaml`
**Purpose**: Executive summary writing with structure and tone validation
**Use Case**: Tasks requiring structured writing with word count and style constraints
**Example**: Business summaries, technical reports, policy briefs

### Online Tasks

#### `online_research.yaml`
**Purpose**: Multi-step research planning with source validation
**Use Case**: Tasks requiring web research, risk assessment, and source quality evaluation
**Example**: Market research, competitive analysis, technology assessment

## How to Use Templates

1. **Choose the appropriate template** based on your task type
2. **Copy the template file** to your desired location (e.g., `bench/tests/offline/my_new_task.yaml`)
3. **Replace all placeholders** marked with `[BRACKETS]` with your specific values
4. **Create required fixtures** in the `fixtures/` directory
5. **Create answer keys** in the `answer_keys/` directory
6. **Validate your test** using: `python scripts/bench.py validate --test path/to/your/test.yaml`

## Template Customization Guide

### Common Placeholders

- `[TASK_NAME]`: Short identifier for your task (e.g., "fraud_detection")
- `[DESCRIPTION]`: Human-readable task description
- `[YOUR_TASK]`: Task identifier used in filenames and paths

### Metrics Template Specific

- `[CLASSIFICATION_TASK]`: What you're classifying (e.g., "fraud detection")
- `[INSERT_CSV_DATA_HERE]`: Your actual CSV data with headers
- `[TARGET_COLUMN]`: Ground truth column name
- `[YOUR_CSV_FILE]`: Fixture filename without path

### Regex Template Specific

- `[PATTERN_DESCRIPTION]`: What pattern you're matching
- `[FORMAT_SPECIFICATION]`: Expected format (e.g., "XXX-XX-XXXX")
- `[CONSTRAINT_1]`, `[CONSTRAINT_2]`, etc.: Validation rules
- `[TEST_LINE_1]` through `[TEST_LINE_12]`: Test cases
- `[PATTERN_TYPE]`: Short pattern identifier (e.g., "ssn", "email")

### Summary Template Specific

- `[TOPIC_DESCRIPTION]`: Summary topic
- `[MAX_TITLE_WORDS]`: Maximum title length (typically 6)
- `[MIN_WORDS]`, `[MAX_WORDS]`: Summary word count range
- `[BULLET_COUNT]`: Required bullet points (typically 3)
- `[MAX_SENTENCE_LENGTH]`: Maximum average sentence length

### Research Template Specific

- `[RESEARCH_TOPIC]`: Research subject area
- Customize example research steps, risks, and sources for your domain

## File Structure Requirements

When creating a new test, ensure you also create:

### Required Files
- **Test definition**: `bench/tests/[category]/[task_name].yaml`
- **Answer key**: `answer_keys/[category]/[task_name]_[type].json`

### Optional Files
- **Fixtures**: `fixtures/csv/[data_file].csv` or `fixtures/text/[data_file].txt`

## Validation

Always validate your test definition before use:

```bash
# Validate test structure and schema compliance
python scripts/bench.py validate --test bench/tests/offline/my_task.yaml

# Test with a specific provider (dry run)
python scripts/bench.py run --provider chatgpt --test bench/tests/offline/my_task.yaml --dry-run
```

## Best Practices

1. **Use descriptive IDs**: Make test IDs self-explanatory (e.g., "offline.fraud.credit_card_detection")
2. **Provide clear instructions**: Ensure prompts are unambiguous and vendor-neutral
3. **Include examples**: Show expected JSON format in prompts
4. **Set appropriate weights**: Ensure scoring weights align with task importance
5. **Test thoroughly**: Validate with multiple providers before deployment
6. **Document constraints**: Clearly specify any validation rules or limitations

## Troubleshooting

### Common Issues

- **Schema validation errors**: Check JSON path expressions in `expectation.fields`
- **Missing fixtures**: Ensure fixture files exist and paths are correct
- **Answer key mismatches**: Verify answer key structure matches evaluator expectations
- **Scoring errors**: Check that weight totals align with your scoring rubric

### Getting Help

- Review existing test definitions in `bench/tests/offline/` for working examples
- Check evaluator documentation in `bench/core/evaluators/`
- Validate against JSON schemas in `schemas/`
- Run tests in verbose mode for detailed error messages
