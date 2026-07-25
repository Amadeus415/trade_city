"""Per-lens analyst calls — structured output only."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from fund.agent.cache import LLMCache
from fund.agent.providers.base import LLMProvider, Message
from fund.agent.schema import ANALYST_JSON_SCHEMA, AnalystOutput
from fund.logging_setup import get_logger

log = get_logger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"

PROMPT_FILES = {
    "technical": "technical.md",
    "fundamental": "fundamental.md",
    "catalyst": "catalyst.md",
}


def prompt_version(name: str) -> str:
    path = PROMPTS_DIR / PROMPT_FILES[name]
    text = path.read_text()
    # version from first heading line e.g. "# Technical Analyst v1"
    first = text.splitlines()[0] if text else name
    return f"{name}:{first.strip()}"


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / PROMPT_FILES[name]).read_text()


class Analyst:
    def __init__(
        self,
        name: str,
        provider: LLMProvider,
        cache: LLMCache | None = None,
        max_retries: int = 1,
    ) -> None:
        self.name = name
        self.provider = provider
        self.cache = cache
        self.max_retries = max_retries
        self.system = load_prompt(name)
        self.version = prompt_version(name)

    def run(self, feature_packet: str, model_name: str) -> AnalystOutput:
        key = None
        if self.cache:
            key = LLMCache.make_key(self.version, model_name, feature_packet, self.system)
            cached = self.cache.get(key)
            if cached is not None:
                return AnalystOutput.model_validate_json(cached)

        messages = [
            Message(role="system", content=self.system),
            Message(
                role="user",
                content=f"Feature packet follows. Produce structured views.\n\n{feature_packet}",
            ),
        ]
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.provider.complete(
                    messages, response_schema=ANALYST_JSON_SCHEMA
                )
                content = resp.content.strip()
                # Strip markdown fences if present
                if content.startswith("```"):
                    content = content.split("\n", 1)[-1]
                    content = content.rsplit("```", 1)[0].strip()
                out = AnalystOutput.model_validate_json(content)
                if self.cache and key:
                    self.cache.put(
                        key,
                        provider=resp.provider,
                        model=resp.model,
                        prompt_version=self.version,
                        response=out.model_dump_json(),
                    )
                return out
            except (ValidationError, json.JSONDecodeError) as e:
                last_err = e
                log.warning("analyst_parse_fail", analyst=self.name, attempt=attempt, error=str(e))
                messages.append(
                    Message(
                        role="user",
                        content=f"Validation error: {e}. Return corrected JSON only.",
                    )
                )
        # Abstain on failure
        log.error("analyst_abstain", analyst=self.name, error=str(last_err))
        return AnalystOutput(
            analyst=self.name,
            views=[],
            notes=f"parse_failure: {last_err}",
        )
