"""Text-derived features from news/filings (counts; LLM-scored tone is cached separately)."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from fund.features.registry import feature


def news_count_7d(news_df: pd.DataFrame, symbol: str) -> int:
    if news_df.empty:
        return 0
    mask = news_df["symbols"].astype(str).str.contains(symbol, regex=False)
    return int(mask.sum())


def filing_events_30d(filings_df: pd.DataFrame, symbol: str) -> list[str]:
    if filings_df is None or filings_df.empty:
        return []
    sub = filings_df[filings_df["symbol"] == symbol]
    if "item_codes" in sub.columns:
        codes: list[str] = []
        for v in sub["item_codes"]:
            if isinstance(v, list):
                codes.extend(v)
            elif isinstance(v, str):
                codes.extend(v.split(","))
        return codes
    return []


# Registered features that operate on bar frames are limited; text features
# are typically computed in the packet builder with news store access.
@feature(name="news_count_7d", version=1, requires=[], lookback=1)
def news_count_7d_feature(bars: pd.DataFrame) -> float | None:
    """Placeholder — real count injected by packet builder via override."""
    if "news_count_7d" in bars.columns:
        return float(bars.iloc[-1]["news_count_7d"])
    return 0.0
