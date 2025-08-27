"""Validation framework for test responses.

This module provides comprehensive validation utilities including JSON schema
validation, field extraction, structural checks, and content normalization.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema
from jsonpath_ng import parse as jsonpath_parse
from jsonpath_ng.exceptions import JSONPathError


@dataclass
class ValidationResult:
    """Result of validation operation."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    extracted_data: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


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
    expected_range: tuple[Any, Any] | Any
    score: float = 0.0
    details: str = ""


class ValidationError(Exception):
    """Raised when validation fails."""

    pass


class SchemaValidator:
    """Validates JSON against test case schemas."""

    def __init__(self, schema_dir: Path):
        """Initialize schema validator.

        Args:
            schema_dir: Directory containing JSON schema files
        """
        self.schema_dir = schema_dir
        self.schemas: dict[str, dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)

    def load_schema(self, schema_name: str) -> dict[str, Any]:
        """Load JSON schema from file.

        Args:
            schema_name: Name of schema file (without .json extension)

        Returns:
            Loaded JSON schema

        Raises:
            ValidationError: If schema file cannot be loaded
        """
        if schema_name in self.schemas:
            return self.schemas[schema_name]

        schema_path = self.schema_dir / f"{schema_name}.schema.json"
        if not schema_path.exists():
            raise ValidationError(f"Schema file not found: {schema_path}")

        try:
            with open(schema_path, encoding="utf-8") as f:
                schema = json.load(f)
            self.schemas[schema_name] = schema
            return schema  # type: ignore[no-any-return]
        except (json.JSONDecodeError, OSError) as e:
            raise ValidationError(f"Failed to load schema {schema_name}: {e}") from e

    def validate_response(
        self, response: dict[str, Any], schema_name: str
    ) -> ValidationResult:
        """Validate response against JSON schema.

        Args:
            response: JSON response to validate
            schema_name: Name of schema to validate against

        Returns:
            Validation result with errors and warnings
        """
        try:
            schema = self.load_schema(schema_name)
            jsonschema.validate(response, schema)

            return ValidationResult(valid=True, metadata={"schema_name": schema_name})

        except jsonschema.ValidationError as e:
            error_msg = f"Schema validation failed: {e.message}"
            if e.absolute_path:
                error_msg += f" at path: {'.'.join(str(p) for p in e.absolute_path)}"

            return ValidationResult(
                valid=False,
                errors=[error_msg],
                metadata={"schema_name": schema_name, "validation_error": str(e)},
            )

        except ValidationError as e:
            return ValidationResult(
                valid=False, errors=[str(e)], metadata={"schema_name": schema_name}
            )


class FieldExtractor:
    """Extract field using JSONPath with error handling and type validation."""

    def __init__(self) -> None:
        """Initialize field extractor."""
        pass
        self.logger = logging.getLogger(__name__)
        self._compiled_paths: dict[str, Any] = {}

    def extract_fields(
        self, data: dict[str, Any], field_specs: list[dict[str, Any]]
    ) -> dict[str, FieldExtractionResult]:
        """Extract multiple fields from data.

        Args:
            data: JSON data to extract from
            field_specs: List of field specifications with 'path' and 'type'

        Returns:
            Dictionary mapping field paths to extraction results
        """
        results = {}

        for spec in field_specs:
            path = spec["path"]
            expected_type = spec.get("type", "any")

            result = self.extract_single_field(data, path, expected_type)
            results[path] = result

        return results

    def extract_single_field(
        self, data: dict[str, Any], path: str, field_type: str = "any"
    ) -> FieldExtractionResult:
        """Extract single field from data.

        Args:
            data: JSON data to extract from
            path: JSONPath expression
            field_type: Expected field type

        Returns:
            Field extraction result
        """
        try:
            # Compile and cache JSONPath expression
            if path not in self._compiled_paths:
                self._compiled_paths[path] = jsonpath_parse(path)

            jsonpath_expr = self._compiled_paths[path]
            matches = jsonpath_expr.find(data)

            if not matches:
                return FieldExtractionResult(
                    success=False,
                    value=None,
                    field_path=path,
                    error_message=f"Field not found at path: {path}",
                )

            # Get first match value
            value = matches[0].value

            # Validate field type
            if not self.validate_field_type(value, field_type):
                return FieldExtractionResult(
                    success=False,
                    value=value,
                    field_path=path,
                    error_message=f"Field type mismatch: expected {field_type}, "
                    f"got {type(value).__name__}",
                )

            return FieldExtractionResult(success=True, value=value, field_path=path)

        except JSONPathError as e:
            return FieldExtractionResult(
                success=False,
                value=None,
                field_path=path,
                error_message=f"Invalid JSONPath expression: {e}",
            )
        except Exception as e:
            return FieldExtractionResult(
                success=False,
                value=None,
                field_path=path,
                error_message=f"Extraction error: {e}",
            )

    def validate_field_type(self, value: Any, expected_type: str) -> bool:
        """Validate field type.

        Args:
            value: Value to validate
            expected_type: Expected type name

        Returns:
            True if type matches expectation
        """
        if expected_type == "any":
            return True

        type_mapping = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type is None:
            self.logger.warning(f"Unknown field type: {expected_type}")
            return True

        return isinstance(value, expected_python_type)  # type: ignore[arg-type]


class StructuralValidator:
    """Validates structural requirements like word counts."""

    def __init__(self) -> None:
        """Initialize structural validator."""
        pass
        self.logger = logging.getLogger(__name__)

    def validate_word_count(
        self, text: str, min_words: int | None = None, max_words: int | None = None
    ) -> StructuralValidationResult:
        """Validate word count in text.

        Args:
            text: Text to validate
            min_words: Minimum word count (optional)
            max_words: Maximum word count (optional)

        Returns:
            Structural validation result
        """
        word_count = len(text.split())

        if min_words is not None and word_count < min_words:
            return StructuralValidationResult(
                passed=False,
                actual_value=word_count,
                expected_range=(min_words, max_words),
                details=f"Word count {word_count} below minimum {min_words}",
            )

        if max_words is not None and word_count > max_words:
            return StructuralValidationResult(
                passed=False,
                actual_value=word_count,
                expected_range=(min_words, max_words),
                details=f"Word count {word_count} above maximum {max_words}",
            )

        return StructuralValidationResult(
            passed=True,
            actual_value=word_count,
            expected_range=(min_words, max_words),
            score=1.0,
            details=f"Word count {word_count} within range",
        )

    def validate_bullet_count(
        self, bullets: list[str], expected_count: int
    ) -> StructuralValidationResult:
        """Validate bullet point count.

        Args:
            bullets: List of bullet points
            expected_count: Expected number of bullets

        Returns:
            Structural validation result
        """
        actual_count = len(bullets)

        if actual_count != expected_count:
            return StructuralValidationResult(
                passed=False,
                actual_value=actual_count,
                expected_range=expected_count,
                details=f"Expected {expected_count} bullets, got {actual_count}",
            )

        return StructuralValidationResult(
            passed=True,
            actual_value=actual_count,
            expected_range=expected_count,
            score=1.0,
            details=f"Bullet count matches expectation: {actual_count}",
        )

    def validate_title_length(
        self, title: str, max_words: int
    ) -> StructuralValidationResult:
        """Validate title word count.

        Args:
            title: Title text
            max_words: Maximum allowed words

        Returns:
            Structural validation result
        """
        word_count = len(title.split())

        if word_count > max_words:
            return StructuralValidationResult(
                passed=False,
                actual_value=word_count,
                expected_range=(0, max_words),
                details=f"Title has {word_count} words, maximum is {max_words}",
            )

        return StructuralValidationResult(
            passed=True,
            actual_value=word_count,
            expected_range=(0, max_words),
            score=1.0,
            details=f"Title length within limit: {word_count} words",
        )

    def count_sentences(self, text: str) -> int:
        """Count sentences in text.

        Args:
            text: Text to analyze

        Returns:
            Number of sentences
        """
        # Simple sentence splitting on common sentence endings
        sentences = re.split(r"[.!?]+", text.strip())
        # Filter out empty strings
        sentences = [s.strip() for s in sentences if s.strip()]
        return len(sentences)

    def calculate_avg_sentence_length(self, text: str) -> float:
        """Calculate average sentence length in words.

        Args:
            text: Text to analyze

        Returns:
            Average words per sentence
        """
        sentences = re.split(r"[.!?]+", text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        total_words = sum(len(sentence.split()) for sentence in sentences)
        return total_words / len(sentences)


class ContentNormalizer:
    """Normalizes and cleans text content."""

    def __init__(self) -> None:
        """Initialize content normalizer."""
        pass
        self.logger = logging.getLogger(__name__)

    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text.

        Args:
            text: Text to normalize

        Returns:
            Text with normalized whitespace
        """
        # Replace multiple whitespace with single space
        text = re.sub(r"\s+", " ", text)
        # Strip leading/trailing whitespace
        return text.strip()

    def extract_json_from_text(self, text: str) -> dict[str, Any] | None:
        """Extract JSON from mixed text response.

        Args:
            text: Text that may contain JSON

        Returns:
            Extracted JSON object or None if not found
        """
        # Try to find JSON in text using regex
        json_patterns = [
            r"```json\s*({.*?})\s*```",  # JSON in code blocks
            r"```\s*({.*?})\s*```",  # JSON in generic code blocks
            r"({.*?})",  # Any JSON-like object
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)  # type: ignore[no-any-return]
                except json.JSONDecodeError:
                    continue

        # Try parsing the entire text as JSON
        try:
            return json.loads(text)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            return None

    def clean_markdown_formatting(self, text: str) -> str:
        """Remove markdown formatting from text.

        Args:
            text: Text with markdown formatting

        Returns:
            Plain text without markdown
        """
        # Remove markdown headers
        text = re.sub(r"^(\s*)#+\s*(.*)$", r"\1\2", text, flags=re.MULTILINE)
        # Remove bold/italic
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        text = re.sub(r"\*(.*?)\*", r"\1", text)
        # Remove code blocks
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        text = re.sub(r"`(.*?)`", r"\1", text)
        # Remove links
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

        return self.normalize_whitespace(text)

    def normalize_numbers(self, value: str | float | int) -> float:
        """Normalize number values.

        Args:
            value: Number value to normalize

        Returns:
            Normalized float value

        Raises:
            ValueError: If value cannot be converted to float
        """
        if isinstance(value, int | float):
            return float(value)

        if isinstance(value, str):
            # Remove common number formatting
            cleaned = value.strip().replace(",", "").replace("%", "")
            try:
                return float(cleaned)
            except ValueError as e:
                raise ValueError(f"Cannot convert '{value}' to number") from e

        raise ValueError(f"Unsupported number type: {type(value)}")
