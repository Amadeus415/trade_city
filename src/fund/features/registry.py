"""Versioned feature registry. LLM never sees raw OHLCV."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

FeatureFn = Callable[..., float | None]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    version: int
    requires: list[str]
    lookback: int
    fn: FeatureFn

    @property
    def key(self) -> str:
        return f"{self.name}@v{self.version}"


_REGISTRY: dict[str, FeatureSpec] = {}


def feature(
    name: str,
    version: int = 1,
    requires: list[str] | None = None,
    lookback: int = 1,
) -> Callable[[FeatureFn], FeatureFn]:
    def deco(fn: FeatureFn) -> FeatureFn:
        spec = FeatureSpec(
            name=name,
            version=version,
            requires=requires or ["close"],
            lookback=lookback,
            fn=fn,
        )
        _REGISTRY[name] = spec
        return fn

    return deco


def get_registry() -> dict[str, FeatureSpec]:
    return dict(_REGISTRY)


class FeatureRegistry:
    def __init__(self) -> None:
        self._specs = get_registry()

    def max_lookback(self) -> int:
        if not self._specs:
            return 252
        return max(s.lookback for s in self._specs.values())

    def compute_symbol(
        self,
        bars: pd.DataFrame,
        names: list[str] | None = None,
    ) -> dict[str, float | None]:
        """Compute features for a single-symbol bar frame. NaN → None."""
        specs = self._specs
        if names:
            specs = {k: v for k, v in specs.items() if k in names}
        out: dict[str, float | None] = {}
        for name, spec in specs.items():
            if len(bars) < spec.lookback:
                out[name] = None
                continue
            try:
                val = spec.fn(bars)
            except Exception:
                val = None
            if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                out[name] = None
            else:
                out[name] = float(val)
        return out

    def compute_universe(
        self,
        bars: pd.DataFrame,
        symbols: list[str],
    ) -> dict[str, dict[str, float | None]]:
        result: dict[str, dict[str, float | None]] = {}
        for sym in symbols:
            sb = bars[bars["symbol"] == sym].sort_values("session")
            result[sym] = self.compute_symbol(sb)
        return result
