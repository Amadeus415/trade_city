"""Vintaged fundamentals — (symbol, period, vintage) storage."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd

from fund.store.pit import pit_filter


class FundamentalsStore:
    """One row per (symbol, period, vintage). PIT via observed_at = vintage publication."""

    def __init__(self, data_dir: str | Path) -> None:
        self.root = Path(data_dir) / "fundamentals"
        self.root.mkdir(parents=True, exist_ok=True)
        self._path = self.root / "facts.parquet"

    def write_facts(self, rows: list[dict] | pd.DataFrame) -> int:
        if isinstance(rows, list):
            if not rows:
                return 0
            df = pd.DataFrame(rows)
        else:
            df = rows.copy()
        if df.empty:
            return 0
        df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True)
        df["period_end"] = pd.to_datetime(df["period_end"]).dt.date
        if self._path.exists():
            existing = pd.read_parquet(self._path)
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["symbol", "metric", "period_end", "vintage"],
                keep="last",
            )
        else:
            combined = df
        combined.to_parquet(self._path, index=False)
        return len(df)

    def get_fundamentals(
        self,
        symbols: list[str],
        as_of: datetime,
        metrics: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return latest vintage of each fact with observed_at <= as_of.

        Restated figures published after as_of are invisible.
        """
        if not self._path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(self._path)
        df = df[df["symbol"].isin(symbols)]
        df = pit_filter(df, as_of)
        if df.empty:
            return df
        if metrics:
            df = df[df["metric"].isin(metrics)]
        # For each (symbol, metric, period_end) keep the latest vintage still visible
        df = df.sort_values("observed_at")
        df = df.drop_duplicates(
            subset=["symbol", "metric", "period_end"],
            keep="last",
        )
        return df.reset_index(drop=True)

    def get_metric_as_of(
        self,
        symbol: str,
        metric: str,
        as_of: datetime,
        period_end: date | None = None,
    ) -> float | None:
        df = self.get_fundamentals([symbol], as_of=as_of, metrics=[metric])
        if df.empty:
            return None
        if period_end is not None:
            df = df[df["period_end"] == period_end]
        if df.empty:
            return None
        # Most recent period
        df = df.sort_values("period_end")
        return float(df.iloc[-1]["value"])
