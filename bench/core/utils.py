"""Utility functions for evaluation framework.

This module provides common utilities for tokenization, word counting,
seed management, timeout guards, and other helper functions.
"""

import hashlib
import json
import random
import re
import signal
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

# Common English stopwords for optional filtering
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
    "this",
    "but",
    "they",
    "have",
    "had",
    "what",
    "said",
    "each",
    "which",
    "their",
    "time",
    "if",
    "up",
    "out",
    "many",
    "then",
    "them",
    "these",
    "so",
    "some",
}


def tokenize_text(text: str, method: str = "whitespace") -> list[str]:
    """Tokenize text for word counting and analysis.

    Args:
        text: Input text to tokenize
        method: Tokenization method ('whitespace', 'punctuation')

    Returns:
        List of tokens
    """
    if method == "whitespace":
        # Simple whitespace tokenization
        return text.split()
    elif method == "punctuation":
        # Split on whitespace and punctuation
        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens
    else:
        # Default to whitespace
        return text.split()


def count_words(text: str, exclude_stopwords: bool = False) -> int:
    """Count words in text with optional stopword exclusion.

    Args:
        text: Input text to count
        exclude_stopwords: Whether to exclude common stopwords

    Returns:
        Word count
    """
    tokens = tokenize_text(
        text, method="punctuation" if exclude_stopwords else "whitespace"
    )

    if exclude_stopwords:
        tokens = [token for token in tokens if token.lower() not in STOPWORDS]

    return len(tokens)


def extract_bullets(text: str) -> list[str]:
    """Extract bullet points from text.

    Args:
        text: Text containing bullet points

    Returns:
        List of bullet point text (without bullet markers)
    """
    # Match various bullet formats
    bullet_patterns = [
        r"^\s*[-*•]\s+(.+)$",  # - * • bullets
        r"^\s*\d+\.\s+(.+)$",  # numbered bullets
        r"^\s*[a-zA-Z]\.\s+(.+)$",  # lettered bullets
    ]

    bullets = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        for pattern in bullet_patterns:
            match = re.match(pattern, line, re.MULTILINE)
            if match:
                bullets.append(match.group(1).strip())
                break

    return bullets


def parse_title(text: str) -> str:
    """Extract title from text (first line or heading).

    Args:
        text: Text to extract title from

    Returns:
        Extracted title text
    """
    lines = text.strip().split("\n")
    if not lines:
        return ""

    first_line = lines[0].strip()

    # Remove markdown heading markers
    first_line = re.sub(r"^#+\s*", "", first_line)

    # Remove common title formatting
    first_line = re.sub(r"^(Title:|TITLE:)\s*", "", first_line, flags=re.IGNORECASE)

    return first_line.strip()


def clean_text(text: str) -> str:
    """Clean and normalize text for analysis.

    Args:
        text: Text to clean

    Returns:
        Cleaned text
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove excessive punctuation
    text = re.sub(r"[.]{3,}", "...", text)
    text = re.sub(r"[!]{2,}", "!", text)
    text = re.sub(r"[?]{2,}", "?", text)

    return text.strip()


class ValidationTimeoutError(Exception):
    """Raised when operation times out."""

    pass


@contextmanager
def timeout_guard(timeout_seconds: float) -> Generator[None, None, None]:
    """Context manager for timeout protection.

    Args:
        timeout_seconds: Timeout in seconds

    Yields:
        None

    Raises:
        TimeoutError: If operation exceeds timeout
    """

    def timeout_handler(signum: int, frame: Any) -> None:
        raise ValidationTimeoutError(
            f"Operation timed out after {timeout_seconds} seconds"
        )

    # Set up signal handler (Unix only)
    try:
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout_seconds))

        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    except (AttributeError, OSError) as e:
        # Fallback for Windows or systems without signal support
        start_time = time.time()
        yield
        if time.time() - start_time > timeout_seconds:
            raise ValidationTimeoutError(
                f"Operation timed out after {timeout_seconds} seconds"
            ) from e


def safe_regex_match(pattern: str, text: str, timeout_ms: int = 100) -> bool:
    """Safely match regex with timeout protection.

    Args:
        pattern: Regex pattern to match
        text: Text to match against
        timeout_ms: Timeout in milliseconds

    Returns:
        True if pattern matches, False otherwise

    Raises:
        TimeoutError: If regex matching times out
    """
    timeout_seconds = timeout_ms / 1000.0

    try:
        with timeout_guard(timeout_seconds):
            compiled_pattern = re.compile(pattern)
            return bool(compiled_pattern.fullmatch(text))
    except TimeoutError:
        # Log timeout and return False
        return False
    except re.error:
        # Invalid regex pattern
        return False


def safe_json_parse(text: str) -> dict[str, Any] | None:
    """Safely parse JSON with error handling.

    Args:
        text: Text to parse as JSON

    Returns:
        Parsed JSON object or None if parsing fails
    """
    try:
        return json.loads(text)  # type: ignore[no-any-return]
    except (json.JSONDecodeError, TypeError):
        return None


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducible results.

    Args:
        seed: Seed value to use
    """
    random.seed(seed)


def generate_deterministic_hash(data: str) -> str:
    """Generate deterministic hash for data.

    Args:
        data: Data to hash

    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def ensure_reproducible_ordering(items: list[Any]) -> list[Any]:
    """Ensure reproducible ordering of items.

    Args:
        items: List of items to order

    Returns:
        Sorted list with deterministic ordering
    """
    # Sort by string representation for deterministic ordering
    return sorted(items, key=lambda x: str(x))
