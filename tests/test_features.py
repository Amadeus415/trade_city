"""Feature registry basics."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from fund.features.registry import FeatureRegistry
import fund.features.price  # noqa: F401


def _bars(n: int = 300, start: float = 100.0) -> pd.DataFrame:
    sessions = [date(2020, 1, 2) + timedelta(days=i) for i in range(n)]
    # filter to weekdays roughly
    rng = np.random.default_rng(0)
    prices = start * np.cumprod(1 + rng.normal(0.0005, 0.01, size=n))
    return pd.DataFrame(
        {
            "symbol": "AAA",
            "session": sessions,
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "adj_close": prices,
            "volume": rng.integers(1e6, 2e6, size=n),
        }
    )


def test_momentum_and_nan_propagation():
    reg = FeatureRegistry()
    short = _bars(10)
    out = reg.compute_symbol(short, names=["mom_63d"])
    assert out["mom_63d"] is None  # insufficient history

    long = _bars(100)
    out2 = reg.compute_symbol(long, names=["mom_21d", "mom_63d", "rsi_14"])
    assert out2["mom_21d"] is not None
    assert out2["mom_63d"] is not None
    assert out2["rsi_14"] is not None
