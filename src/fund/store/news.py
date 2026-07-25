"""News storage with PIT-enforcing readers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from fund.store.pit import pit_filter


class NewsStore:
    def __init__(self, data_dir: str | Path) -> None:
        self.root = Path(data_dir) / "news"
        self.root.mkdir(parents=True, exist_ok=True)
        self._index = self.root / "index.parquet"
        self.raw_dir = self.root / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def write_articles(self, rows: list[dict] | pd.DataFrame) -> int:
        if isinstance(rows, list):
            if not rows:
                return 0
            df = pd.DataFrame(rows)
        else:
            df = rows.copy()
        if df.empty:
            return 0
        df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True)
        df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
        if self._index.exists():
            existing = pd.read_parquet(self._index)
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["article_id"], keep="last")
        else:
            combined = df
        combined.to_parquet(self._index, index=False)
        return len(df)

    def get_news(
        self,
        symbols: list[str] | None,
        as_of: datetime,
        lookback_days: int = 7,
    ) -> pd.DataFrame:
        """News with observed_at <= as_of only."""
        if not self._index.exists():
            return pd.DataFrame()
        df = pd.read_parquet(self._index)
        df = pit_filter(df, as_of)
        if df.empty:
            return df
        cutoff = pd.Timestamp(as_of)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        else:
            cutoff = cutoff.tz_convert("UTC")
        start = cutoff - pd.Timedelta(days=lookback_days)
        df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
        df = df[df["event_time"] >= start]
        if symbols is not None:
            # symbol may be comma-joined multi-ticker
            mask = df["symbols"].apply(
                lambda s: any(sym in str(s).split(",") for sym in symbols)
            )
            df = df[mask]
        return df.sort_values("event_time").reset_index(drop=True)
