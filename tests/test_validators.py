"""Unit tests for validation framework."""

import json
import tempfile
from pathlib import Path

import pytest

from bench.core.validators import (
    ContentNormalizer,
    FieldExtractor,
    SchemaValidator,
    StructuralValidator,
    ValidationError,
)


class TestSchemaValidator:
    """Test JSON schema validation."""

    def test_schema_validator_init(self):
        """Test schema validator initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            schema_dir = Path(tmpdir)
            validator = SchemaValidator(schema_dir)
            assert validator.schema_dir == schema_dir
            assert validator.schemas == {}

    def test_load_schema_success(self):
        """Test successful schema loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            schema_dir = Path(tmpdir)
            schema_file = schema_dir / "test.schema.json"

            schema = {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
                "required": ["name"],
            }

            with open(schema_file, "w") as f:
                json.dump(schema, f)

            validator = SchemaValidator(schema_dir)
            loaded_schema = validator.load_schema("test")
            assert loaded_schema == schema

    def test_load_schema_not_found(self):
        """Test schema loading with missing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            schema_dir = Path(tmpdir)
            validator = SchemaValidator(schema_dir)

            with pytest.raises(ValidationError, match="Schema file not found"):
                validator.load_schema("nonexistent")

    def test_validate_response_success(self):
        """Test successful response validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            schema_dir = Path(tmpdir)
            schema_file = schema_dir / "test.schema.json"

            schema = {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
                "required": ["name"],
            }

            with open(schema_file, "w") as f:
                json.dump(schema, f)

            validator = SchemaValidator(schema_dir)
            response = {"name": "John", "age": 30}
            result = validator.validate_response(response, "test")

            assert result.valid is True
            assert result.errors == []
            assert result.metadata["schema_name"] == "test"

    def test_validate_response_failure(self):
        """Test response validation failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            schema_dir = Path(tmpdir)
            schema_file = schema_dir / "test.schema.json"

            schema = {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
                "required": ["name"],
            }

            with open(schema_file, "w") as f:
                json.dump(schema, f)

            validator = SchemaValidator(schema_dir)
            response = {"age": 30}  # Missing required "name"
            result = validator.validate_response(response, "test")

            assert result.valid is False
            assert len(result.errors) > 0
            assert "name" in result.errors[0]


class TestFieldExtractor:
    """Test JSONPath field extraction."""

    def test_field_extractor_init(self):
        """Test field extractor initialization."""
        extractor = FieldExtractor()
        assert extractor._compiled_paths == {}

    def test_extract_single_field_success(self):
        """Test successful single field extraction."""
        extractor = FieldExtractor()
        data = {"task1_data_metrics": {"precision": 0.75, "recall": 0.6}}

        result = extractor.extract_single_field(
            data, "$.task1_data_metrics.precision", "number"
        )

        assert result.success is True
        assert result.value == 0.75
        assert result.field_path == "$.task1_data_metrics.precision"

    def test_extract_single_field_not_found(self):
        """Test field extraction with missing field."""
        extractor = FieldExtractor()
        data = {"other": "value"}

        result = extractor.extract_single_field(data, "$.missing.field", "string")

        assert result.success is False
        assert result.value is None
        assert "Field not found" in result.error_message

    def test_extract_single_field_type_mismatch(self):
        """Test that field extraction fails when field type doesn't match expected."""
        extractor = FieldExtractor()
        data = {"value": "string_value"}

        result = extractor.extract_single_field(data, "$.value", "number")

        assert result.success is False
        assert "type mismatch" in result.error_message

    def test_extract_fields_multiple(self):
        """Test extracting multiple fields."""
        extractor = FieldExtractor()
        data = {"metrics": {"precision": 0.75, "recall": 0.6, "f1": 0.67}}

        field_specs = [
            {"path": "$.metrics.precision", "type": "number"},
            {"path": "$.metrics.recall", "type": "number"},
            {"path": "$.metrics.missing", "type": "number"},
        ]

        results = extractor.extract_fields(data, field_specs)

        assert len(results) == 3
        assert results["$.metrics.precision"].success is True
        assert results["$.metrics.recall"].success is True
        assert results["$.metrics.missing"].success is False

    def test_validate_field_type(self):
        """Test field type validation."""
        extractor = FieldExtractor()

        assert extractor.validate_field_type("string", "string") is True
        assert extractor.validate_field_type(42, "number") is True
        assert extractor.validate_field_type(3.14, "number") is True
        assert extractor.validate_field_type(True, "boolean") is True
        assert extractor.validate_field_type([], "array") is True
        assert extractor.validate_field_type({}, "object") is True
        assert extractor.validate_field_type("anything", "any") is True

        assert extractor.validate_field_type("string", "number") is False
        assert extractor.validate_field_type(42, "string") is False


class TestStructuralValidator:
    """Test structural validation."""

    def test_structural_validator_init(self):
        """Test structural validator initialization."""
        validator = StructuralValidator()
        assert validator is not None

    def test_validate_word_count_success(self):
        """Test successful word count validation."""
        validator = StructuralValidator()
        text = "This is a test sentence with seven words."

        result = validator.validate_word_count(text, min_words=5, max_words=10)

        assert result.passed is True
        assert result.actual_value == 8
        assert result.score == 1.0

    def test_validate_word_count_too_few(self):
        """Test word count validation with too few words."""
        validator = StructuralValidator()
        text = "Short text"

        result = validator.validate_word_count(text, min_words=5, max_words=10)

        assert result.passed is False
        assert result.actual_value == 2
        assert "below minimum" in result.details

    def test_validate_word_count_too_many(self):
        """Test word count validation with too many words."""
        validator = StructuralValidator()
        text = (
            "This is a very long sentence with many words that exceeds "
            "the maximum limit"
        )

        result = validator.validate_word_count(text, min_words=5, max_words=10)

        assert result.passed is False
        assert result.actual_value == 14
        assert "above maximum" in result.details

    def test_validate_bullet_count_success(self):
        """Test successful bullet count validation."""
        validator = StructuralValidator()
        bullets = ["First bullet", "Second bullet", "Third bullet"]

        result = validator.validate_bullet_count(bullets, 3)

        assert result.passed is True
        assert result.actual_value == 3
        assert result.score == 1.0

    def test_validate_bullet_count_mismatch(self):
        """Test bullet count validation with mismatch."""
        validator = StructuralValidator()
        bullets = ["First bullet", "Second bullet"]

        result = validator.validate_bullet_count(bullets, 3)

        assert result.passed is False
        assert result.actual_value == 2
        assert "Expected 3 bullets, got 2" in result.details

    def test_validate_title_length_success(self):
        """Test successful title length validation."""
        validator = StructuralValidator()
        title = "Short Title"

        result = validator.validate_title_length(title, max_words=6)

        assert result.passed is True
        assert result.actual_value == 2
        assert result.score == 1.0

    def test_validate_title_length_too_long(self):
        """Test title length validation with too many words."""
        validator = StructuralValidator()
        title = "This is a very long title that exceeds the limit"

        result = validator.validate_title_length(title, max_words=6)

        assert result.passed is False
        assert result.actual_value == 10
        assert "maximum is 6" in result.details

    def test_count_sentences(self):
        """Test sentence counting."""
        validator = StructuralValidator()

        text = "First sentence. Second sentence! Third sentence?"
        assert validator.count_sentences(text) == 3

        text = "Single sentence."
        assert validator.count_sentences(text) == 1

        text = ""
        assert validator.count_sentences(text) == 0

    def test_calculate_avg_sentence_length(self):
        """Test average sentence length calculation."""
        validator = StructuralValidator()

        text = "Short. This is longer sentence."
        avg_length = validator.calculate_avg_sentence_length(text)
        assert avg_length == 2.5  # (1 + 4) / 2


class TestContentNormalizer:
    """Test content normalization."""

    def test_content_normalizer_init(self):
        """Test content normalizer initialization."""
        normalizer = ContentNormalizer()
        assert normalizer is not None

    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        normalizer = ContentNormalizer()

        text = "  Multiple   spaces\n\nand   newlines  "
        result = normalizer.normalize_whitespace(text)
        assert result == "Multiple spaces and newlines"

    def test_extract_json_from_text_code_block(self):
        """Test JSON extraction from code block."""
        normalizer = ContentNormalizer()

        text = """
        Here is the result:
        ```json
        {"name": "test", "value": 42}
        ```
        """

        result = normalizer.extract_json_from_text(text)
        assert result == {"name": "test", "value": 42}

    def test_extract_json_from_text_plain(self):
        """Test JSON extraction from plain text."""
        normalizer = ContentNormalizer()

        text = '{"name": "test", "value": 42}'
        result = normalizer.extract_json_from_text(text)
        assert result == {"name": "test", "value": 42}

    def test_extract_json_from_text_none(self):
        """Test JSON extraction when no JSON present."""
        normalizer = ContentNormalizer()

        text = "This is just plain text with no JSON"
        result = normalizer.extract_json_from_text(text)
        assert result is None

    def test_clean_markdown_formatting(self):
        """Test markdown formatting removal."""
        normalizer = ContentNormalizer()

        text = """
        # Header

        This is **bold** and *italic* text.

        Here's `code` and a [link](http://example.com).

        ```
        code block
        ```
        """

        result = normalizer.clean_markdown_formatting(text)
        assert "Header" in result
        assert "bold" in result
        assert "italic" in result
        assert "code" in result
        assert "link" in result
        assert "#" not in result
        assert "**" not in result
        assert "*" not in result
        assert "`" not in result
        assert "[" not in result
        assert "]" not in result

    def test_normalize_numbers(self):
        """Test number normalization."""
        normalizer = ContentNormalizer()

        assert normalizer.normalize_numbers(42) == 42.0
        assert normalizer.normalize_numbers(3.14) == 3.14
        assert normalizer.normalize_numbers("42") == 42.0
        assert normalizer.normalize_numbers("3.14") == 3.14
        assert normalizer.normalize_numbers("1,234.56") == 1234.56
        assert normalizer.normalize_numbers("75%") == 75.0

        with pytest.raises(ValueError):
            normalizer.normalize_numbers("not a number")

        with pytest.raises(ValueError):
            normalizer.normalize_numbers([1, 2, 3])  # type: ignore[arg-type]
