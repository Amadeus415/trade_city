"""OpenAI-compatible chat completions (OpenAI, xAI, Ollama, vLLM, Groq, …)."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from fund.agent.providers.base import LLMResponse, Message


class OpenAICompatibleProvider:
    name = "openai_compatible"

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.1,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        # Refine provider label from URL
        if "api.openai.com" in base_url:
            self.name = "openai"
        elif "api.x.ai" in base_url:
            self.name = "xai"
        elif "11434" in base_url:
            self.name = "ollama"

    def complete(
        self,
        messages: list[Message],
        *,
        response_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        t0 = time.perf_counter()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature if temperature is not None else self.temperature,
        }
        # Prefer JSON object mode; schema enforcement where supported
        if response_schema is not None:
            payload["response_format"] = {"type": "json_object"}
            # Append schema hint for providers without strict schema
            schema_hint = (
                "\n\nRespond with a single JSON object matching this schema:\n"
                + json.dumps(response_schema)
            )
            payload["messages"] = list(payload["messages"])
            payload["messages"][-1] = {
                **payload["messages"][-1],
                "content": payload["messages"][-1]["content"] + schema_hint,
            }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        ms = int((time.perf_counter() - t0) * 1000)
        return LLMResponse(
            content=content,
            model=data.get("model", self.model),
            provider=self.name,
            latency_ms=ms,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            raw=data,
        )
