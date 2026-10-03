from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Shared paths, compact-memory settings, and model configuration."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load optional environment values and return a ready-to-use config."""

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env", override=False)
    except ImportError:
        # Offline mode deliberately has no dependency on python-dotenv.
        pass

    provider = normalize_provider(os.getenv("LLM_PROVIDER", "openai"))
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    judge_provider = normalize_provider(os.getenv("JUDGE_PROVIDER", provider))
    judge_model_name = os.getenv("JUDGE_MODEL", model_name)

    def provider_values(name: str) -> tuple[str | None, str | None]:
        key_map = {
            "openai": "OPENAI_API_KEY",
            "custom": "CUSTOM_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "ollama": "OLLAMA_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }
        base_map = {
            "custom": "CUSTOM_BASE_URL",
            "ollama": "OLLAMA_BASE_URL",
            "openrouter": "OPENROUTER_BASE_URL",
        }
        canonical = normalize_provider(name)
        return os.getenv(key_map.get(canonical, "")) or None, os.getenv(base_map.get(canonical, "")) or None

    def env_int(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero")
        return value

    api_key, base_url = provider_values(provider)
    judge_api_key, judge_base_url = provider_values(judge_provider)
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=env_int("COMPACT_THRESHOLD_TOKENS", 1_200),
        compact_keep_messages=env_int("COMPACT_KEEP_MESSAGES", 6),
        model=ProviderConfig(
            provider=provider,
            model_name=model_name,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
            api_key=api_key,
            base_url=base_url,
        ),
        judge_model=ProviderConfig(
            provider=judge_provider,
            model_name=judge_model_name,
            temperature=float(os.getenv("JUDGE_TEMPERATURE", "0")),
            api_key=judge_api_key,
            base_url=judge_base_url,
        ),
    )
