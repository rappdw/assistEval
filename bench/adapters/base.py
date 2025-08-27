"""Base provider interface for AI assistant adapters.

This module defines the abstract Provider class that all provider implementations
must inherit from to ensure consistent interfaces across different AI services.
"""

from abc import ABC, abstractmethod
from typing import Any


class ProviderError(Exception):
    """Base exception for provider-related errors."""

    pass


class ProviderConfigurationError(ProviderError):
    """Raised when provider configuration is invalid."""

    pass


class ProviderCapabilityError(ProviderError):
    """Raised when capability constraints are violated."""

    pass


class ProviderTimeoutError(ProviderError):
    """Raised when provider request times out."""

    pass


class ProviderRateLimitError(ProviderError):
    """Raised when provider rate limits are exceeded."""

    pass


class Provider(ABC):
    """Abstract base class for AI provider implementations.

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
            ProviderError: Base class for all provider-related errors
            ProviderConfigurationError: Invalid configuration
            ProviderCapabilityError: Capability constraint violation
            ProviderTimeoutError: Request timeout
            ProviderRateLimitError: Rate limit exceeded
        """
        pass

    def validate_capabilities(self, capabilities: dict[str, Any]) -> None:
        """Validate and enforce capability constraints.

        Args:
            capabilities: Capability constraints to validate

        Raises:
            ProviderCapabilityError: If constraints cannot be satisfied
        """
        # Validate web access constraint
        web_access = capabilities.get("web", "allowed")
        if web_access not in ["forbidden", "allowed", "required"]:
            raise ProviderCapabilityError(
                f"Invalid web access setting: {web_access}. "
                "Must be 'forbidden', 'allowed', or 'required'"
            )

        # Validate JSON requirement
        json_required = capabilities.get("json_required", False)
        if not isinstance(json_required, bool):
            raise ProviderCapabilityError(
                f"json_required must be boolean, got {type(json_required)}"
            )

        # Validate tools list
        tools = capabilities.get("tools", [])
        if not isinstance(tools, list):
            raise ProviderCapabilityError(f"tools must be a list, got {type(tools)}")

    def _modify_system_prompt(
        self, system_prompt: str, capabilities: dict[str, Any]
    ) -> str:
        """Modify system prompt to enforce capability constraints.

        Args:
            system_prompt: Original system prompt
            capabilities: Capability constraints

        Returns:
            Modified system prompt with constraints
        """
        constraints = []

        # Add web access constraints
        web_access = capabilities.get("web", "allowed")
        if web_access == "forbidden":
            constraints.append("Do not browse the web or access external URLs.")
        elif web_access == "required":
            constraints.append(
                "Browsing required; provide sources with dates. "
                "If browsing is disabled, mark sources as placeholders "
                "and state limitations."
            )

        # Add JSON format requirement
        if capabilities.get("json_required", False):
            constraints.append("Respond only in valid JSON format.")

        # Add tool restrictions
        tools = capabilities.get("tools", [])
        if not tools:
            constraints.append("Do not use any external tools or functions.")

        if constraints:
            constraint_text = " ".join(constraints)
            return f"{system_prompt}\n\nIMPORTANT CONSTRAINTS: {constraint_text}"

        return system_prompt

    def __repr__(self) -> str:
        """String representation of the provider."""
        return f"{self.__class__.__name__}(name='{self.name}')"
