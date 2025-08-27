"""Base provider interface for AI assistant evaluation.

This module defines the abstract interface that all provider adapters must implement
to ensure consistent evaluation across different AI platforms.
"""

from abc import ABC, abstractmethod
from typing import Any


class Provider(ABC):
    """Abstract base class for AI provider adapters.

    All provider implementations must inherit from this class and implement
    the invoke method to handle AI assistant interactions.
    """

    def __init__(self, name: str, **kwargs: Any) -> None:
        """Initialize the provider.

        Args:
            name: Human-readable name for this provider
            **kwargs: Provider-specific configuration options
        """
        self.name = name
        self.config = kwargs

    @abstractmethod
    def invoke(
        self,
        system: str,
        user: str,
        *,
        options: dict[str, Any],
        capabilities: dict[str, Any],
    ) -> dict[str, str]:
        """Invoke the AI provider with the given prompts.

        Args:
            system: System prompt to set context and constraints
            user: User prompt with the actual task
            options: Provider-specific options (temperature, max_tokens, etc.)
            capabilities: Capability constraints (web access, tools, etc.)

        Returns:
            Dictionary with 'raw_text' key containing the provider's response

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement invoke method")

    def __repr__(self) -> str:
        """String representation of the provider."""
        return f"{self.__class__.__name__}(name='{self.name}')"
