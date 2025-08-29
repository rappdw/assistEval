"""Provider adapters for AI assistant evaluation.

This package contains adapters for different AI providers, implementing a common
interface for consistent evaluation across platforms.
"""

from pathlib import Path
from typing import Any

import yaml

from bench.adapters.base import Provider, ProviderError


def load_provider_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load provider configuration from YAML file.

    Args:
        config_path: Path to providers.yaml config file. If None, uses default.

    Returns:
        Dictionary containing provider configurations

    Raises:
        ProviderError: If config file cannot be loaded or parsed
    """
    if config_path is None:
        # Default to configs/providers.yaml relative to project root
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "configs" / "providers.yaml"

    config_path = Path(config_path)

    if not config_path.exists():
        raise ProviderError(f"Provider config file not found: {config_path}")

    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict) or "providers" not in config:
            raise ProviderError("Invalid provider config: missing 'providers' section")

        return config

    except yaml.YAMLError as e:
        raise ProviderError(f"Failed to parse provider config: {e}") from e
    except Exception as e:
        raise ProviderError(f"Failed to load provider config: {e}") from e


def create_provider(
    provider_name: str, config_path: str | Path | None = None
) -> Provider:
    """Create a provider instance from configuration.

    Args:
        provider_name: Name of the provider to create
        config_path: Path to providers.yaml config file. If None, uses default.

    Returns:
        Configured provider instance

    Raises:
        ProviderError: If provider cannot be created or configured
    """
    # Load configuration
    config = load_provider_config(config_path)
    providers_config = config["providers"]

    if provider_name not in providers_config:
        available = list(providers_config.keys())
        raise ProviderError(
            f"Provider '{provider_name}' not found in config. "
            f"Available providers: {available}"
        )

    provider_config = providers_config[provider_name]
    provider_type = provider_config.get("type")

    if not provider_type:
        raise ProviderError(f"Provider '{provider_name}' missing 'type' in config")

    # Import and instantiate the appropriate provider class
    provider_class = _get_provider_class(provider_type)

    try:
        # Pass all config parameters to the provider constructor
        return provider_class(**provider_config)
    except Exception as e:
        raise ProviderError(f"Failed to create provider '{provider_name}': {e}") from e


def _get_provider_class(provider_type: str) -> type[Provider]:
    """Get provider class by type name.

    Args:
        provider_type: Type of provider ('chatgpt', 'chatgpt_manual', 'copilot_manual')

    Returns:
        Provider class

    Raises:
        ProviderError: If provider type is not supported
    """
    if provider_type == "chatgpt":
        from bench.adapters.chatgpt import ChatGPTProvider

        return ChatGPTProvider
    elif provider_type == "chatgpt_manual":
        from bench.adapters.chatgpt_manual import ChatGPTManualProvider

        return ChatGPTManualProvider
    elif provider_type == "copilot_manual":
        from bench.adapters.copilot_manual import CopilotManualProvider

        return CopilotManualProvider
    else:
        raise ProviderError(f"Unsupported provider type: {provider_type}")


def list_available_providers(config_path: str | Path | None = None) -> list[str]:
    """List all available providers from configuration.

    Args:
        config_path: Path to providers.yaml config file. If None, uses default.

    Returns:
        List of available provider names

    Raises:
        ProviderError: If config cannot be loaded
    """
    config = load_provider_config(config_path)
    return list(config["providers"].keys())


__all__ = [
    "Provider",
    "load_provider_config",
    "create_provider",
    "list_available_providers",
]
