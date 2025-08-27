"""Provider adapters for AI assistant evaluation.

This package contains adapters for different AI providers, implementing a common
interface for consistent evaluation across platforms.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bench.adapters.base import Provider

__all__ = ["Provider"]
