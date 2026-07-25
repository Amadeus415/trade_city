"""Provider-agnostic LLM interface.

No decision-time internet tools. Structured JSON via response_schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class Message:
    role: str  # system | user | assistant
    content: str


@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def model_version(self) -> str:
        return f"{self.provider}:{self.model}"


@runtime_checkable
class LLMProvider(Protocol):
    """Swap implementations without touching agent logic."""

    name: str

    def complete(
        self,
        messages: list[Message],
        *,
        response_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> LLMResponse: ...
