"""Bar storage and PIT-enforcing readers (with optional in-memory cache)."""

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
        self._cache: pd.DataFrame | None = None
        self._cache_symbols: frozenset[str] | None = None

    def _path(self, symbol: str, year: int) -> Path:
        d = self.root / f"symbol={symbol}" / f"year={year}"
        d.mkdir(parents=True, exist_ok=True)
        return d / "part.parquet"

    def clear_cache(self) -> None:
        self._cache = None
        self._cache_symbols = None

    def preload(self, symbols: list[str] | None = None) -> int:
        """Load bars into memory for fast repeated PIT queries (backtests)."""
        symbols = symbols or self.available_symbols()
        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            files = list((self.root / f"symbol={symbol}").glob("year=*/part.parquet"))
            if not files:
                continue
            con = duckdb.connect()
            try:
                paths = ", ".join(repr(str(f)) for f in files)
                frame = con.execute(f"SELECT * FROM read_parquet([{paths}])").fetchdf()
            finally:
                con.close()
            if not frame.empty:
                frames.append(frame)
        if not frames:
            self._cache = pd.DataFrame(
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
        else:
            self._cache = pd.concat(frames, ignore_index=True)
            self._cache["session"] = pd.to_datetime(self._cache["session"]).dt.date
            self._cache["observed_at"] = pd.to_datetime(self._cache["observed_at"], utc=True)
        self._cache_symbols = frozenset(symbols)
        return 0 if self._cache is None else len(self._cache)

    def write_bars(self, bars: list[Bar] | pd.DataFrame) -> int:
        self.clear_cache()
        if isinstance(bars, list):
            if not bars:
                return 0
            df = pd.DataFrame([b.model_dump(mode="python") for b in bars])
        else:
            df = bars.copy()
        if df.empty:
            return 0
        df = self._normalize_frame(df)
        df["year"] = pd.to_datetime(df["session"]).dt.year
        n = 0
        for (symbol, year), group in df.groupby(["symbol", "year"]):
            path = self._path(str(symbol), int(year))
            g = group.drop(columns=["year"], errors="ignore")
            if path.exists():
                existing = self._normalize_frame(pd.read_parquet(path))
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

    @staticmethod
    def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
        """Stable dtypes so multi-year Parquet reads don't hit DECIMAL precision clashes."""
        out = df.copy()
        out["symbol"] = out["symbol"].astype(str)
        out["session"] = pd.to_datetime(out["session"]).dt.date
        out["observed_at"] = pd.to_datetime(out["observed_at"], utc=True)
        for col in ("open", "high", "low", "close", "adj_close"):
            if col in out.columns:
                out[col] = out[col].astype("float64")
        if "volume" in out.columns:
            out["volume"] = out["volume"].astype("int64")
        return out

    def get_bars(
        self,
        symbols: list[str],
        as_of: datetime,
        lookback: int | None = None,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        """Returns bars with observed_at <= as_of. No kwargs to disable PIT."""
        if self._cache is not None and self._cache_symbols is not None:
            if set(symbols).issubset(self._cache_symbols) or not self._cache_symbols:
                df = self._cache[self._cache["symbol"].isin(symbols)].copy()
            else:
                df = self._load_from_disk(symbols)
        else:
            df = self._load_from_disk(symbols)

        if df.empty:
            return df.reset_index(drop=True)

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

    def _load_from_disk(self, symbols: list[str]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            files = sorted((self.root / f"symbol={symbol}").glob("year=*/part.parquet"))
            if not files:
                continue
            con = duckdb.connect()
            try:
                paths = ", ".join(repr(str(f)) for f in files)
                # union_by_name + cast avoids DECIMAL(p,s) mismatches across years
                frame = con.execute(
                    f"""
                    SELECT
                      symbol::VARCHAR AS symbol,
                      session,
                      open::DOUBLE AS open,
                      high::DOUBLE AS high,
                      low::DOUBLE AS low,
                      close::DOUBLE AS close,
                      adj_close::DOUBLE AS adj_close,
                      volume::BIGINT AS volume,
                      observed_at
                    FROM read_parquet([{paths}], union_by_name=true)
                    """
                ).fetchdf()
            finally:
                con.close()
            if not frame.empty:
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
        return pd.concat(frames, ignore_index=True)

    def rewrite_all_float(self) -> int:
        """One-shot normalize of on-disk parquet dtypes (DECIMAL → float64)."""
        n = 0
        for symbol in self.available_symbols():
            files = list((self.root / f"symbol={symbol}").glob("year=*/part.parquet"))
            for path in files:
                df = self._normalize_frame(pd.read_parquet(path))
                df.to_parquet(path, index=False)
                n += len(df)
        self.clear_cache()
        return n

    def get_last_bar(self, symbol: str, as_of: datetime) -> Bar | None:
        df = self.get_bars([symbol], as_of=as_of, lookback=1)
        if df.empty:
            return None
        row = df.iloc[0]
        return Bar(
            symbol=str(row["symbol"]),
            session=row["session"]
            if isinstance(row["session"], date)
            else pd.Timestamp(row["session"]).date(),
            open=Decimal(str(row["open"])),
            high=Decimal(str(row["high"])),
            low=Decimal(str(row["low"])),
            close=Decimal(str(row["close"])),
            adj_close=Decimal(str(row["adj_close"])),
            volume=int(row["volume"]),
            observed_at=pd.Timestamp(row["observed_at"]).to_pydatetime(),
        )

    def price_matrix(
        self,
        symbols: list[str],
        as_of: datetime,
        *,
        start: date | None = None,
        end: date | None = None,
        use_adj: bool = True,
    ) -> pd.DataFrame:
        """Wide close matrix for baselines (index=session, columns=symbol)."""
        df = self.get_bars(symbols, as_of=as_of, start=start, end=end)
        if df.empty:
            return pd.DataFrame()
        col = "adj_close" if use_adj and "adj_close" in df.columns else "close"
        pivot = df.pivot_table(index="session", columns="symbol", values=col, aggfunc="last")
        pivot.index = pd.to_datetime(pivot.index)
        return pivot.sort_index()

    def available_symbols(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(
            p.name.split("=", 1)[1] for p in self.root.glob("symbol=*") if p.is_dir()
        )
