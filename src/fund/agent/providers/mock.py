"""Deterministic mock LLM for backtests/tests — no network."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from fund.agent.providers.base import LLMResponse, Message


class MockProvider:
    name = "mock"

    def __init__(self, model: str = "mock-v1", temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature

    def complete(
        self,
        messages: list[Message],
        *,
        response_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        t0 = time.perf_counter()
        blob = "\n".join(m.content for m in messages)
        # Detect PM vs analyst from schema / prompt
        is_pm = response_schema is not None and "proposals" in str(response_schema.get("properties", {}))
        if is_pm or "Portfolio Manager" in blob or "reconcile" in blob.lower():
            content = self._pm_response(blob)
        else:
            content = self._analyst_response(blob)
        ms = int((time.perf_counter() - t0) * 1000)
        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.name,
            latency_ms=ms,
            input_tokens=len(blob) // 4,
            output_tokens=len(content) // 4,
            cost_usd=0.0,
        )

    def _extract_symbols(self, blob: str) -> list[str]:
        """Pull tickers from free-text packets and JSON analyst outputs."""
        import re

        skip = {
            "UNIVERSE",
            "SNAPSHOT",
            "PORTFOLIO",
            "CONSTRAINTS",
            "RISK",
            "STATE",
            "NORMAL",
            "REDUCED",
            "HALTED",
            "HOLD",
            "OPEN",
            "ADD",
            "TRIM",
            "CLOSE",
            "SYMS",
            "ETF",
            "JSON",
            "NULL",
            "TRUE",
            "FALSE",
            "BULLISH",
            "BEARISH",
            "NEUTRAL",
            "ABSTAIN",
            "EDT",
            "EST",
            "UTC",
            "ANALYSTS",
            "ANALYST",
            "PACKET",
            "VIEWS",
            "NOTES",
            "MOCK",
            "THESIS",
            "SCORE",
            "STANCE",
            "SYMBOL",
            "ACTION",
            "CONFIDENCE",
            "SOURCE",
            "FEATURES",
            "KEY",
            "HORIZON",
            "INVALIDATION",
            "PROPOSALS",
            "REASON",
            "DAYS",
            "WEIGHT",
            "CLUSTER",
            "CASH",
            "GROSS",
            "EQUITY",
        }
        # JSON "symbol": "AAPL" and free-text tickers
        candidates = re.findall(
            r'(?:"symbol"\s*:\s*"([A-Z]{1,5})")|(?:\b([A-Z]{1,5})\b)',
            blob,
        )
        symbols: list[str] = []
        seen: set[str] = set()
        for a, b in candidates:
            tok = a or b
            if not tok or tok in skip or tok in seen:
                continue
            if not tok.isalpha():
                continue
            seen.add(tok)
            symbols.append(tok)
        return symbols[:20]

    def _analyst_response(self, blob: str) -> str:
        symbols = self._extract_symbols(blob)
        views = []
        for sym in symbols:
            h = int(hashlib.sha256(f"{self.model}|{sym}|{blob[:200]}".encode()).hexdigest()[:8], 16)
            score = ((h % 200) - 100) / 100.0
            stance = "bullish" if score > 0.15 else ("bearish" if score < -0.15 else "neutral")
            views.append(
                {
                    "symbol": sym,
                    "stance": stance,
                    "score": f"{score:.2f}",
                    "confidence": f"{0.3 + (h % 50) / 100:.2f}",
                    "thesis": f"Mock {stance} view on {sym} based on momentum/vol features.",
                    "key_features": ["mom_63d", "realized_vol_21d"],
                }
            )
        if not views:
            views.append(
                {
                    "symbol": "SPY",
                    "stance": "neutral",
                    "score": "0.00",
                    "confidence": "0.20",
                    "thesis": "No symbols in packet; abstaining.",
                    "key_features": [],
                }
            )
        return json.dumps({"analyst": self.model, "views": views, "notes": "mock"})

    def _pm_response(self, blob: str) -> str:
        symbols = self._extract_symbols(blob)
        proposals = []
        # Simple rule: open top-2 by hash if not already in portfolio section
        portfolio_syms: set[str] = set()
        in_port = False
        for line in blob.splitlines():
            if line.startswith("PORTFOLIO"):
                in_port = True
                continue
            if in_port and line.startswith("CONSTRAINTS"):
                break
            if in_port:
                parts = line.split()
                if parts and parts[0].isupper() and parts[0].isalpha():
                    portfolio_syms.add(parts[0])

        ranked = sorted(
            symbols,
            key=lambda s: hashlib.sha256(f"pm|{s}|{blob[:100]}".encode()).hexdigest(),
        )
        for i, sym in enumerate(ranked[:3]):
            if sym in portfolio_syms:
                proposals.append(
                    {
                        "symbol": sym,
                        "action": "hold",
                        "target_weight": "0.05",
                        "confidence": "0.40",
                        "thesis": f"Maintain position in {sym}.",
                        "invalidation": f"{sym} breaks below 63d momentum floor",
                        "horizon_days": 21,
                        "source_features": ["mom_63d"],
                    }
                )
            else:
                proposals.append(
                    {
                        "symbol": sym,
                        "action": "open",
                        "target_weight": "0.05",
                        "confidence": f"{0.45 + i * 0.05:.2f}",
                        "thesis": f"Initiate small position in {sym} on relative strength.",
                        "invalidation": f"mom_63d turns negative for {sym}",
                        "horizon_days": 42,
                        "source_features": ["mom_63d", "mom_63d_rank"],
                    }
                )
        if not proposals:
            return json.dumps(
                {
                    "proposals": [],
                    "abstain_reason": "No actionable symbols; abstaining is correct.",
                }
            )
        return json.dumps({"proposals": proposals, "abstain_reason": None})
