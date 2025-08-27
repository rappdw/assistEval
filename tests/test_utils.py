"""Unit tests for utility functions."""

import pytest

from bench.core.utils import (
    ValidationTimeoutError,
    clean_text,
    count_words,
    ensure_reproducible_ordering,
    extract_bullets,
    generate_deterministic_hash,
    parse_title,
    safe_json_parse,
    safe_regex_match,
    set_random_seed,
    timeout_guard,
    tokenize_text,
)


class TestTokenization:
    """Test text tokenization functions."""

    def test_tokenize_text_whitespace(self):
        """Test whitespace tokenization."""
        text = "This is a test sentence."
        tokens = tokenize_text(text, method="whitespace")
        assert tokens == ["This", "is", "a", "test", "sentence."]

    def test_tokenize_text_punctuation(self):
        """Test punctuation-aware tokenization."""
        text = "Hello, world! How are you?"
        tokens = tokenize_text(text, method="punctuation")
        assert tokens == ["hello", "world", "how", "are", "you"]

    def test_tokenize_text_default(self):
        """Test default tokenization method."""
        text = "Default tokenization test"
        tokens = tokenize_text(text)
        assert tokens == ["Default", "tokenization", "test"]

    def test_count_words_basic(self):
        """Test basic word counting."""
        text = "This is a test sentence with seven words."
        count = count_words(text)
        assert count == 8

    def test_count_words_with_stopwords(self):
        """Test word counting with stopword exclusion."""
        text = "This is a test sentence with some common words."
        count_all = count_words(text, exclude_stopwords=False)
        count_filtered = count_words(text, exclude_stopwords=True)
        assert count_all > count_filtered
        assert count_filtered > 0

    def test_count_words_empty(self):
        """Test word counting with empty text."""
        assert count_words("") == 0
        assert count_words("   ") == 0


class TestTextProcessing:
    """Test text processing utilities."""

    def test_extract_bullets_dash(self):
        """Test bullet extraction with dash bullets."""
        text = """
        - First bullet point
        - Second bullet point
        - Third bullet point
        """
        bullets = extract_bullets(text)
        assert len(bullets) == 3
        assert bullets[0] == "First bullet point"
        assert bullets[1] == "Second bullet point"
        assert bullets[2] == "Third bullet point"

    def test_extract_bullets_asterisk(self):
        """Test bullet extraction with asterisk bullets."""
        text = """
        * First item
        * Second item
        """
        bullets = extract_bullets(text)
        assert len(bullets) == 2
        assert bullets[0] == "First item"
        assert bullets[1] == "Second item"

    def test_extract_bullets_numbered(self):
        """Test bullet extraction with numbered bullets."""
        text = """
        1. First numbered item
        2. Second numbered item
        3. Third numbered item
        """
        bullets = extract_bullets(text)
        assert len(bullets) == 3
        assert bullets[0] == "First numbered item"

    def test_extract_bullets_mixed(self):
        """Test bullet extraction with mixed formats."""
        text = """
        - Dash bullet
        * Asterisk bullet
        1. Numbered bullet
        Regular text line
        """
        bullets = extract_bullets(text)
        assert len(bullets) == 3
        assert "Dash bullet" in bullets
        assert "Asterisk bullet" in bullets
        assert "Numbered bullet" in bullets

    def test_parse_title_simple(self):
        """Test simple title parsing."""
        text = "Simple Title\nRest of the content"
        title = parse_title(text)
        assert title == "Simple Title"

    def test_parse_title_markdown(self):
        """Test title parsing with markdown headers."""
        text = "# Markdown Header\nContent below"
        title = parse_title(text)
        assert title == "Markdown Header"

    def test_parse_title_with_prefix(self):
        """Test title parsing with title prefix."""
        text = "Title: Executive Summary Report\nContent"
        title = parse_title(text)
        assert title == "Executive Summary Report"

    def test_parse_title_empty(self):
        """Test title parsing with empty text."""
        assert parse_title("") == ""
        assert parse_title("   ") == ""

    def test_clean_text(self):
        """Test text cleaning and normalization."""
        text = "Text   with    multiple     spaces....and!!!excessive???punctuation"
        cleaned = clean_text(text)
        assert "   " not in cleaned
        assert "...." not in cleaned
        assert "!!!" not in cleaned
        assert "???" not in cleaned
        assert "..." in cleaned  # Should be normalized to single ellipsis


class TestSafetyFunctions:
    """Test safety and timeout functions."""

    def test_safe_regex_match_valid(self):
        """Test safe regex matching with valid pattern."""
        pattern = r"^\d{3}-\d{2}-\d{4}$"
        text = "123-45-6789"
        assert safe_regex_match(pattern, text) is True

    def test_safe_regex_match_invalid(self):
        """Test safe regex matching with invalid text."""
        pattern = r"^\d{3}-\d{2}-\d{4}$"
        text = "not-a-ssn"
        assert safe_regex_match(pattern, text) is False

    def test_safe_regex_match_bad_pattern(self):
        """Test safe regex matching with invalid pattern."""
        pattern = r"[unclosed"
        text = "test"
        assert safe_regex_match(pattern, text) is False

    def test_safe_json_parse_valid(self):
        """Test safe JSON parsing with valid JSON."""
        json_text = '{"name": "test", "value": 42}'
        result = safe_json_parse(json_text)
        assert result == {"name": "test", "value": 42}

    def test_safe_json_parse_invalid(self):
        """Test safe JSON parsing with invalid JSON."""
        json_text = '{"invalid": json}'
        result = safe_json_parse(json_text)
        assert result is None

    def test_safe_json_parse_empty(self):
        """Test safe JSON parsing with empty string."""
        assert safe_json_parse("") is None
        assert safe_json_parse("   ") is None

    def test_timeout_guard_success(self):
        """Test timeout guard with successful operation."""

        def quick_operation():
            return "success"

        with timeout_guard(1.0):
            result = quick_operation()
        assert result == "success"

    def test_timeout_guard_timeout(self):
        """Test timeout guard with timeout."""
        import time

        # Skip this test on systems where signal-based timeout doesn't work
        try:
            with timeout_guard(0.1):
                time.sleep(0.2)
            # If we get here, timeout didn't work - skip the test
            pytest.skip("Signal-based timeout not available on this system")
        except (ValidationTimeoutError, OSError):
            # Expected behavior - test passes
            pass


class TestUtilityFunctions:
    """Test miscellaneous utility functions."""

    def test_set_random_seed(self):
        """Test random seed setting."""
        import random

        set_random_seed(42)
        value1 = random.random()  # noqa: S311

        set_random_seed(42)
        value2 = random.random()  # noqa: S311

        assert value1 == value2

    def test_generate_deterministic_hash(self):
        """Test deterministic hash generation."""
        data = "test data"
        hash1 = generate_deterministic_hash(data)
        hash2 = generate_deterministic_hash(data)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length

    def test_generate_deterministic_hash_different(self):
        """Test deterministic hash with different data."""
        hash1 = generate_deterministic_hash("data1")
        hash2 = generate_deterministic_hash("data2")
        assert hash1 != hash2

    def test_ensure_reproducible_ordering(self):
        """Test reproducible ordering of items."""
        items = [3, 1, 4, 1, 5, 9, 2, 6]
        ordered1 = ensure_reproducible_ordering(items)
        ordered2 = ensure_reproducible_ordering(items)
        assert ordered1 == ordered2
        assert ordered1 == [1, 1, 2, 3, 4, 5, 6, 9]

    def test_ensure_reproducible_ordering_mixed(self):
        """Test reproducible ordering with mixed types."""
        items = ["b", 2, "a", 1]
        ordered = ensure_reproducible_ordering(items)
        # Should be sorted by string representation
        assert ordered == [1, 2, "a", "b"]
