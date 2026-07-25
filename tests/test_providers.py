from __future__ import annotations

from datetime import date

from fund.ingest.providers import SyntheticProvider, bars_to_frame, get_provider


def test_synthetic_provider():
    bars = SyntheticProvider().fetch(
        ["AAA", "SPY"], date(2024, 1, 2), date(2024, 1, 31)
    )
    assert len(bars) > 0
    assert {b.symbol for b in bars} == {"AAA", "SPY"}
    df = bars_to_frame(bars)
    assert "close" in df.columns


def test_get_provider_synthetic():
    p = get_provider("synthetic")
    assert p.name == "synthetic"
