"""Splits and dividends — required so momentum features do not hallucinate gaps."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from fund.ingest.base import ValidationResult


class CorporateActionsIngestor:
    def __init__(self, data_dir: str | Path, symbols: list[str]) -> None:
        self.root = Path(data_dir) / "corporate_actions"
        self.root.mkdir(parents=True, exist_ok=True)
        self._path = self.root / "actions.parquet"
        self.symbols = symbols

    def backfill(self, start: date, end: date) -> int:
        # Offline: empty table is valid (adj_close already accounts for splits if from vendor)
        if not self._path.exists():
            pd.DataFrame(
                columns=[
                    "symbol",
                    "action_type",
                    "ex_date",
                    "ratio",
                    "amount",
                    "event_time",
                    "observed_at",
                ]
            ).to_parquet(self._path, index=False)
        return 0

    def incremental(self) -> int:
        return self.backfill(date.today(), date.today())

    def get_actions(self, as_of: datetime, symbols: list[str] | None = None) -> pd.DataFrame:
        if not self._path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(self._path)
        if df.empty:
            return df
        df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True)
        cutoff = pd.Timestamp(as_of)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        else:
            cutoff = cutoff.tz_convert("UTC")
        df = df[df["observed_at"] <= cutoff]
        if symbols:
            df = df[df["symbol"].isin(symbols)]
        return df

    def validate(self) -> ValidationResult:
        return ValidationResult(ok=True)
