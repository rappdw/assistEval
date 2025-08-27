"""ChatGPT provider adapter using OpenAI API.

This module implements the ChatGPT provider for deterministic evaluation
with controlled settings and capability enforcement.
"""

from typing import Any

from bench.adapters.base import Provider


class ChatGPTProvider(Provider):
    """ChatGPT provider implementation using OpenAI API.

    Provides deterministic responses with controlled temperature, seed,
    and capability enforcement for fair evaluation.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize ChatGPT provider.

        Args:
            **kwargs: Configuration options including API key, model, etc.
        """
        super().__init__(name="ChatGPT", **kwargs)
        # TODO: Implement in Stage 4 - Provider Abstraction & Adapters

    def invoke(
        self,
        system: str,
        user: str,
        *,
        options: dict[str, Any],
        capabilities: dict[str, Any],
    ) -> dict[str, str]:
        """Invoke ChatGPT with deterministic settings.

        Args:
            system: System prompt with constraints and context
            user: User prompt with the evaluation task
            options: OpenAI API options (temperature=0, seed=42, etc.)
            capabilities: Capability constraints to enforce

        Returns:
            Dictionary with 'raw_text' key containing ChatGPT's response

        Raises:
            NotImplementedError: Implementation pending in Stage 4
        """
        # TODO: Implement OpenAI API integration in Stage 4
        # - Set temperature=0 for deterministic responses
        # - Use seed for reproducibility when available
        # - Enforce JSON response format when required
        # - Handle capability constraints (no web access for offline tasks)
        # - Add retry logic with exponential backoff
        raise NotImplementedError("Implementation pending in Stage 4")
