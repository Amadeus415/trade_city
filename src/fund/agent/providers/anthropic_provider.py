"""Anthropic Messages API adapter."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from fund.agent.providers.base import LLMResponse, Message


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        model: str,
        api_key: str,
        temperature: float = 0.1,
        timeout: float = 120.0,
        max_tokens: int = 4096,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens

    def complete(
        self,
        messages: list[Message],
        *,
        response_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        t0 = time.perf_counter()
        system = "\n".join(m.content for m in messages if m.role == "system")
        user_msgs = [m for m in messages if m.role != "system"]
        if response_schema is not None and user_msgs:
            last = user_msgs[-1]
            user_msgs = user_msgs[:-1] + [
                Message(
                    role=last.role,
                    content=last.content
                    + "\n\nRespond with a single JSON object matching this schema:\n"
                    + json.dumps(response_schema),
                )
            ]
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "system": system or "You are a careful financial analyst. Output JSON only.",
            "messages": [{"role": m.role, "content": m.content} for m in user_msgs],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        parts = data.get("content", [])
        content = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
        usage = data.get("usage", {})
        ms = int((time.perf_counter() - t0) * 1000)
        return LLMResponse(
            content=content,
            model=data.get("model", self.model),
            provider=self.name,
            latency_ms=ms,
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            raw=data,
        )
