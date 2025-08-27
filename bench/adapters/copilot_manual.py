"""Manual Copilot provider adapter for paste-based evaluation.

This module implements a manual provider that displays prompts and collects
responses via copy-paste, useful when no API is available.
"""

from typing import Any

from bench.adapters.base import Provider


class CopilotManualProvider(Provider):
    """Manual Copilot provider for paste-based interaction.

    Displays prompts to the user and collects responses via manual paste,
    storing raw text for evaluation when API access is not available.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize manual Copilot provider.

        Args:
            **kwargs: Configuration options for manual interaction
        """
        super().__init__(name="Copilot (Manual)", **kwargs)
        # TODO: Implement in Stage 4 - Provider Abstraction & Adapters

    def invoke(
        self,
        system: str,
        user: str,
        *,
        options: dict[str, Any],
        capabilities: dict[str, Any],
    ) -> dict[str, str]:
        """Display prompt and collect manual response.

        Args:
            system: System prompt with constraints and context
            user: User prompt with the evaluation task
            options: Manual provider options (display format, etc.)
            capabilities: Capability constraints to display

        Returns:
            Dictionary with 'raw_text' key containing manually pasted response

        Raises:
            NotImplementedError: Implementation pending in Stage 4
        """
        # TODO: Implement manual interaction in Stage 4
        # - Display formatted prompt to user
        # - Show capability constraints clearly
        # - Prompt for manual response paste
        # - Store raw text response
        # - Add validation for non-empty responses
        raise NotImplementedError("Implementation pending in Stage 4")
