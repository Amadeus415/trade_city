"""Ticker/date anonymisation (KTD-Fin leakage control).

Modes: bright | stock_blind | date_blind | blinded
Consistent aliases within a run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from fund.types import MaskingMode


@dataclass
class MaskingContext:
    mode: MaskingMode
    _sym_to_alias: dict[str, str] = field(default_factory=dict)
    _alias_to_sym: dict[str, str] = field(default_factory=dict)
    _counter: int = 0

    def alias_symbol(self, symbol: str) -> str:
        if self.mode in (MaskingMode.BRIGHT, MaskingMode.DATE_BLIND):
            return symbol
        if symbol not in self._sym_to_alias:
            self._counter += 1
            alias = f"SYM_{self._counter:02d}"
            self._sym_to_alias[symbol] = alias
            self._alias_to_sym[alias] = symbol
        return self._sym_to_alias[symbol]

    def unmask_symbol(self, token: str) -> str:
        return self._alias_to_sym.get(token, token)

    def mask_packet(self, packet: str, as_of_label: str | None = None) -> str:
        text = packet
        # Symbol masking — longest first to avoid partial overlaps
        if self.mode in (MaskingMode.STOCK_BLIND, MaskingMode.BLINDED):
            # Collect candidates: uppercase tokens 1-5 letters
            tokens = set(re.findall(r"\b[A-Z]{1,5}\b", text))
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
                "ET",
                "USD",
            }
            for tok in sorted(tokens, key=len, reverse=True):
                if tok in skip:
                    continue
                alias = self.alias_symbol(tok)
                text = re.sub(rf"\b{re.escape(tok)}\b", alias, text)

        if self.mode in (MaskingMode.DATE_BLIND, MaskingMode.BLINDED):
            # Replace ISO dates with relative anchors
            text = re.sub(
                r"\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\b",
                "t-0",
                text,
            )
            text = re.sub(r"as_of\s+\S+", "as_of t-0", text)
        return text

    def unmask_text(self, text: str) -> str:
        if self.mode in (MaskingMode.BRIGHT, MaskingMode.DATE_BLIND):
            return text
        for alias, sym in sorted(
            self._alias_to_sym.items(), key=lambda x: len(x[0]), reverse=True
        ):
            text = re.sub(rf"\b{re.escape(alias)}\b", sym, text)
        return text

    def unmask_symbol_fields(self, payload: dict) -> dict:
        """Walk common LLM output shapes and restore tickers."""
        import copy

        data = copy.deepcopy(payload)

        def walk(obj):
            if isinstance(obj, dict):
                if "symbol" in obj and isinstance(obj["symbol"], str):
                    obj["symbol"] = self.unmask_symbol(obj["symbol"])
                for v in obj.values():
                    walk(v)
            elif isinstance(obj, list):
                for item in obj:
                    walk(item)

        walk(data)
        return data
