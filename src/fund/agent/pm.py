"""Portfolio manager reconciliation — three analysts → proposals."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from fund.agent.analysts import Analyst
from fund.agent.cache import LLMCache
from fund.agent.masking import MaskingContext
from fund.agent.providers import create_provider
from fund.agent.providers.base import LLMProvider, Message
from fund.agent.schema import PM_JSON_SCHEMA, PMOutput
from fund.config import AgentConfig
from fund.logging_setup import get_logger
from fund.types import Action, MaskingMode, Proposal

log = get_logger(__name__)
PROMPTS_DIR = Path(__file__).parent / "prompts"


class PortfolioManager:
    def __init__(
        self,
        config: AgentConfig,
        cache: LLMCache | None = None,
    ) -> None:
        self.config = config
        self.cache = cache
        self.analyst_provider = create_provider(config, role="analyst")
        self.pm_provider = create_provider(config, role="pm")
        self.analysts = [
            Analyst("technical", self.analyst_provider, cache, config.max_retries),
            Analyst("fundamental", self.analyst_provider, cache, config.max_retries),
            Analyst("catalyst", self.analyst_provider, cache, config.max_retries),
        ]
        self.system = (PROMPTS_DIR / "pm.md").read_text()
        first = self.system.splitlines()[0] if self.system else "pm"
        self.version = f"pm:{first.strip()}"
        self.masking = MaskingContext(mode=MaskingMode(config.masking_mode))

    def run(self, feature_packet: str) -> tuple[list[Proposal], dict]:
        """Returns proposals and a journal metadata dict."""
        masked = self.masking.mask_packet(feature_packet)

        analyst_outputs = []
        for a in self.analysts:
            out = a.run(masked, self.config.analyst_model)
            # Unmask symbols in views
            data = self.masking.unmask_symbol_fields(out.model_dump(mode="json"))
            from fund.agent.schema import AnalystOutput

            analyst_outputs.append(AnalystOutput.model_validate(data))

        combined = {
            "analysts": [o.model_dump(mode="json") for o in analyst_outputs],
            "packet": masked,
        }
        user_blob = (
            "Analyst outputs (JSON) and feature packet follow. "
            "Reconcile into proposals. Abstaining is valid.\n\n"
            + json.dumps(combined, indent=2)
        )

        key = None
        if self.cache and self.config.cache_enabled:
            key = LLMCache.make_key(
                self.version, self.config.pm_model, user_blob, self.system
            )
            cached = self.cache.get(key)
            if cached is not None:
                pm_out = PMOutput.model_validate_json(cached)
                return pm_out.to_proposals(), {
                    "prompt_version": self.version,
                    "model": f"{self.config.provider}:{self.config.pm_model}",
                    "raw": cached,
                    "cached": True,
                    "masking_mode": self.config.masking_mode,
                    "analyst_versions": [a.version for a in self.analysts],
                }

        messages = [
            Message(role="system", content=self.system),
            Message(role="user", content=user_blob),
        ]
        last_err: Exception | None = None
        raw = ""
        for attempt in range(self.config.max_retries + 1):
            try:
                resp = self.pm_provider.complete(messages, response_schema=PM_JSON_SCHEMA)
                raw = resp.content.strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                data = json.loads(raw)
                data = self.masking.unmask_symbol_fields(data)
                pm_out = PMOutput.model_validate(data)
                # Filter pure abstains into hold/empty
                proposals = [
                    p
                    for p in pm_out.to_proposals()
                    if p.action != Action.ABSTAIN
                    or p.target_weight > 0  # keep explicit holds separately
                ]
                # Keep HOLD/ABSTAIN as valid but ABSTAIN with 0 weight can be dropped
                proposals = [p for p in pm_out.to_proposals() if p.action != Action.ABSTAIN]
                if self.cache and key and self.config.cache_enabled:
                    self.cache.put(
                        key,
                        provider=resp.provider,
                        model=resp.model,
                        prompt_version=self.version,
                        response=pm_out.model_dump_json(),
                    )
                meta = {
                    "prompt_version": self.version,
                    "model": resp.model_version,
                    "raw": pm_out.model_dump_json(),
                    "cached": False,
                    "latency_ms": resp.latency_ms,
                    "cost_usd": resp.cost_usd,
                    "masking_mode": self.config.masking_mode,
                    "analyst_versions": [a.version for a in self.analysts],
                }
                return proposals, meta
            except (ValidationError, json.JSONDecodeError, KeyError) as e:
                last_err = e
                log.warning("pm_parse_fail", attempt=attempt, error=str(e))
                messages.append(
                    Message(
                        role="user",
                        content=f"Validation error: {e}. Return corrected JSON only.",
                    )
                )

        log.error("pm_abstain", error=str(last_err))
        return [], {
            "prompt_version": self.version,
            "model": f"{self.config.provider}:{self.config.pm_model}",
            "raw": raw or json.dumps({"error": str(last_err)}),
            "cached": False,
            "masking_mode": self.config.masking_mode,
            "abstain": True,
        }
