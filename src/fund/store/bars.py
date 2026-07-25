"""Bar storage and PIT-enforcing readers."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import duckdb
import pandas as pd

from fund.store.pit import pit_filter
from fund.types import Bar


class BarStore:
    """Parquet bars partitioned by symbol/year with strict as_of filtering."""

    def __init__(self, data_dir: str | Path) -> None:
        self.root = Path(data_dir) / "bars"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str, year: int) -> Path:
        d = self.root / f"symbol={symbol}" / f"year={year}"
        d.mkdir(parents=True, exist_ok=True)
        return d / "part.parquet"

    def write_bars(self, bars: list[Bar] | pd.DataFrame) -> int:
        if isinstance(bars, list):
            if not bars:
                return 0
            df = pd.DataFrame([b.model_dump(mode="python") for b in bars])
        else:
            df = bars.copy()
        if df.empty:
            return 0
        df["session"] = pd.to_datetime(df["session"]).dt.date
        df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True)
        df["year"] = pd.to_datetime(df["session"]).dt.year
        n = 0
        for (symbol, year), group in df.groupby(["symbol", "year"]):
            path = self._path(str(symbol), int(year))
            g = group.drop(columns=["year"], errors="ignore")
            if path.exists():
                existing = pd.read_parquet(path)
                combined = pd.concat([existing, g], ignore_index=True)
                combined = combined.drop_duplicates(
                    subset=["symbol", "session"], keep="last"
                )
                combined = combined.sort_values("session")
                combined.to_parquet(path, index=False)
                n += len(g)
            else:
                g = g.sort_values("session")
                g.to_parquet(path, index=False)
                n += len(g)
        return n

    def get_bars(
        self,
        symbols: list[str],
        as_of: datetime,
        lookback: int | None = None,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        """Returns bars with observed_at <= as_of. No kwargs to disable PIT."""
        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            pattern = str(self.root / f"symbol={symbol}" / "year=*" / "part.parquet")
            files = list((self.root / f"symbol={symbol}").glob("year=*/part.parquet"))
            if not files:
                continue
            # Use duckdb for efficient multi-file read
            con = duckdb.connect()
            try:
                rel = con.execute(
                    f"SELECT * FROM read_parquet([{', '.join(repr(str(f)) for f in files)}])"
                )
                frame = rel.fetchdf()
            finally:
                con.close()
            if frame.empty:
                continue
            frames.append(frame)
        if not frames:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "session",
                    "open",
                    "high",
                    "low",
                    "close",
                    "adj_close",
                    "volume",
                    "observed_at",
                ]
            )
        df = pd.concat(frames, ignore_index=True)
        df = pit_filter(df, as_of)
        df["session"] = pd.to_datetime(df["session"]).dt.date
        if start is not None:
            df = df[df["session"] >= start]
        if end is not None:
            df = df[df["session"] <= end]
        df = df.sort_values(["symbol", "session"])
        if lookback is not None and lookback > 0:
            parts = [g.tail(lookback) for _, g in df.groupby("symbol", sort=False)]
            df = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]
        return df.reset_index(drop=True)

    def get_last_bar(self, symbol: str, as_of: datetime) -> Bar | None:
        df = self.get_bars([symbol], as_of=as_of, lookback=1)
        if df.empty:
            return None
        row = df.iloc[0]
        return Bar(
            symbol=str(row["symbol"]),
            session=row["session"] if isinstance(row["session"], date) else pd.Timestamp(row["session"]).date(),
            open=Decimal(str(row["open"])),
            high=Decimal(str(row["high"])),
            low=Decimal(str(row["low"])),
            close=Decimal(str(row["close"])),
            adj_close=Decimal(str(row["adj_close"])),
            volume=int(row["volume"]),
            observed_at=pd.Timestamp(row["observed_at"]).to_pydatetime(),
        )

    def available_symbols(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(
            p.name.split("=", 1)[1]
            for p in self.root.glob("symbol=*")
            if p.is_dir()
        )
