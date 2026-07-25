"""Model-agnostic LLM providers.

Supported: mock | openai | anthropic | xai | ollama | openai_compatible

Any OpenAI-compatible API (Groq, Together, Fireworks, Azure, local vLLM, etc.)
works via provider=openai_compatible with base_url + api_key.
"""

from __future__ import annotations

from fund.agent.providers.base import LLMProvider, LLMResponse, Message
from fund.agent.providers.mock import MockProvider
from fund.config import AgentConfig


def create_provider(config: AgentConfig, role: str = "analyst") -> LLMProvider:
    """Factory: role is 'analyst' or 'pm' (selects model name)."""
    provider = (config.provider or "mock").lower()
    model = config.analyst_model if role == "analyst" else config.pm_model
    temperature = config.temperature
    api_key = config.api_key
    base_url = config.base_url

    if provider == "mock":
        return MockProvider(model=model, temperature=temperature)

    if provider in ("openai", "xai", "ollama", "openai_compatible"):
        from fund.agent.providers.openai_compatible import OpenAICompatibleProvider

        defaults = {
            "openai": "https://api.openai.com/v1",
            "xai": "https://api.x.ai/v1",
            "ollama": "http://localhost:11434/v1",
            "openai_compatible": base_url or "http://localhost:8000/v1",
        }
        return OpenAICompatibleProvider(
            model=model,
            api_key=api_key or "no-key",
            base_url=base_url or defaults[provider],
            temperature=temperature,
        )

    if provider == "anthropic":
        from fund.agent.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            model=model,
            api_key=api_key or "",
            temperature=temperature,
        )

    raise ValueError(
        f"Unknown LLM provider '{provider}'. "
        "Use: mock | openai | anthropic | xai | ollama | openai_compatible"
    )


__all__ = [
    "LLMProvider",
    "LLMResponse",
    "Message",
    "MockProvider",
    "create_provider",
]
