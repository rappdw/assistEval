"""ChatGPT provider implementation using OpenAI API.

Provides deterministic ChatGPT responses for evaluation benchmarks
with controlled settings and capability enforcement.
"""

import json
import os
import time
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletion

from bench.adapters.base import Provider, ProviderConfigurationError


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

        # Get API key from environment or config
        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ProviderConfigurationError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or provide api_key in configuration."
            )

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=api_key,
            organization=kwargs.get("organization") or os.getenv("OPENAI_ORG_ID"),
        )

        # Set default options
        self.model = kwargs.get("model", "gpt-4")
        self.temperature = kwargs.get("temperature")  # None if not specified
        self.max_completion_tokens = kwargs.get("max_completion_tokens", 1200)
        self.seed = kwargs.get("seed", 42)
        self.timeout_seconds = kwargs.get("timeout_seconds", 60)

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
            options: ChatGPT-specific options (temperature, seed, etc.)
            capabilities: Capability constraints to enforce

        Returns:
            Dictionary with 'raw_text' key containing ChatGPT's response

        Raises:
            ProviderConfigurationError: Invalid configuration
            ProviderRateLimitError: Rate limit exceeded
            ProviderTimeoutError: Request timeout
        """
        # Validate capabilities
        self.validate_capabilities(capabilities)

        # Modify system prompt with constraints
        modified_system = self._modify_system_prompt(system, capabilities)

        # Prepare request parameters
        request_params = {
            "model": options.get("model", self.model),
            "messages": [
                {"role": "system", "content": modified_system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": options.get(
                "max_completion_tokens", self.max_completion_tokens
            ),
            "seed": options.get("seed", self.seed),
            "timeout": options.get("timeout_seconds", self.timeout_seconds),
        }

        # Only add temperature if specified (GPT-5 doesn't support temperature=0)
        temperature = options.get("temperature", self.temperature)
        if temperature is not None:
            request_params["temperature"] = temperature

        # Add JSON response format if required
        if capabilities.get("json_required", False):
            request_params["response_format"] = {"type": "json_object"}

        # Make API call with retry logic
        return self._make_request_with_retry(request_params)

    def _make_request_with_retry(
        self, request_params: dict[str, Any]
    ) -> dict[str, str]:
        """Make OpenAI API request with exponential backoff retry logic.

        Args:
            request_params: Parameters for the OpenAI API request

        Returns:
            Dictionary with 'raw_text' key containing the response

        Raises:
            ProviderRateLimitError: Rate limit exceeded after retries
            ProviderTimeoutError: Request timeout
        """
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries + 1):
            try:
                # Make the API call
                response: ChatCompletion = self.client.chat.completions.create(
                    **request_params
                )

                # Extract response text
                if response.choices and len(response.choices) > 0:
                    message = response.choices[0].message
                    if message.content:
                        raw_text = message.content
                        return {"raw_text": raw_text}
                    else:
                        # Debug empty content
                        raise ProviderConfigurationError(
                            f"Empty content from OpenAI API. Message: {message}, "
                            f"Finish reason: {response.choices[0].finish_reason}"
                        )
                else:
                    raise ProviderConfigurationError(
                        f"No choices in OpenAI API response. Response: {response}"
                    )

            except Exception as e:
                error_message = str(e).lower()

                # Handle rate limiting
                if "rate limit" in error_message and attempt < max_retries:
                    delay = base_delay * (2**attempt)
                    time.sleep(delay)
                    continue

                # Handle connection errors
                if attempt < max_retries and (
                    "connection" in error_message or "network" in error_message
                ):
                    delay = base_delay * (2**attempt)
                    time.sleep(delay)
                    continue

                # Re-raise other exceptions
                raise ProviderConfigurationError(f"OpenAI API error: {e}") from e

        raise ProviderConfigurationError("Failed to get response after all retries")

    def _validate_json_response(self, response_text: str) -> bool:
        """Validate that response is valid JSON.

        Args:
            response_text: Response text to validate

        Returns:
            True if valid JSON, False otherwise
        """
        try:
            json.loads(response_text)
            return True
        except json.JSONDecodeError:
            return False
