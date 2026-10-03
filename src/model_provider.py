from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderConfig:
    """Provider-neutral configuration shared by both agents."""

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Return the canonical provider name, accepting a few common aliases."""

    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "anthorpic": "anthropic",
        "claude": "anthropic",
        "google": "gemini",
        "google-genai": "gemini",
        "open-ai": "openai",
        "openai-compatible": "custom",
        "open-router": "openrouter",
    }
    normalized = aliases.get(normalized, normalized)
    supported = {"openai", "custom", "gemini", "anthropic", "ollama", "openrouter"}
    if normalized not in supported:
        choices = ", ".join(sorted(supported))
        raise ValueError(f"Unsupported provider {value!r}. Choose one of: {choices}")
    return normalized


def build_chat_model(config: ProviderConfig) -> Any:
    """Instantiate the LangChain chat model for the selected provider."""

    provider = normalize_provider(config.provider)

    try:
        if provider in {"openai", "custom"}:
            from langchain_openai import ChatOpenAI

            kwargs: dict[str, Any] = {
                "model": config.model_name,
                "temperature": config.temperature,
            }
            if config.api_key:
                kwargs["api_key"] = config.api_key
            if provider == "custom":
                if not config.base_url:
                    raise ValueError("CUSTOM_BASE_URL is required for the custom provider")
                kwargs["base_url"] = config.base_url
            return ChatOpenAI(**kwargs)

        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            kwargs = {"model": config.model_name, "temperature": config.temperature}
            if config.api_key:
                kwargs["google_api_key"] = config.api_key
            return ChatGoogleGenerativeAI(**kwargs)

        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            kwargs = {"model": config.model_name, "temperature": config.temperature}
            if config.api_key:
                kwargs["api_key"] = config.api_key
            return ChatAnthropic(**kwargs)

        if provider == "ollama":
            from langchain_ollama import ChatOllama

            kwargs = {"model": config.model_name, "temperature": config.temperature}
            if config.base_url:
                kwargs["base_url"] = config.base_url
            return ChatOllama(**kwargs)

        from langchain_openrouter import ChatOpenRouter

        kwargs = {"model": config.model_name, "temperature": config.temperature}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        return ChatOpenRouter(**kwargs)
    except ImportError as exc:
        package_hint = {
            "openai": "langchain-openai",
            "custom": "langchain-openai",
            "gemini": "langchain-google-genai",
            "anthropic": "langchain-anthropic",
            "ollama": "langchain-ollama",
            "openrouter": "langchain-openrouter",
        }[provider]
        raise RuntimeError(
            f"Provider {provider!r} requires the optional package {package_hint!r}."
        ) from exc
