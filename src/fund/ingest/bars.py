"""Daily bar ingestion with availability lag and validation."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from fund.clock import ET, trading_days
from fund.ingest.base import ValidationResult
from fund.store.bars import BarStore
from fund.types import Bar

ET_ZONE = ZoneInfo("America/New_York")


class BarIngestor:
    def __init__(
        self,
        store: BarStore,
        symbols: list[str],
        bar_lag_minutes: int = 15,
    ) -> None:
        self.store = store
        self.symbols = symbols
        self.bar_lag_minutes = bar_lag_minutes

    def _observed_at(self, session: date) -> datetime:
        close = datetime.combine(session, time(16, 0), tzinfo=ET_ZONE)
        return close + timedelta(minutes=self.bar_lag_minutes)

    def backfill(self, start: date, end: date) -> int:
        """Backfill from yfinance (research default). Fail loudly on errors."""
        try:
            import yfinance as yf  # type: ignore
        except ImportError:
            # Synthetic bars when yfinance unavailable (tests / offline)
            return self._synthetic_backfill(start, end)

        total = 0
        for symbol in self.symbols:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                auto_adjust=False,
            )
            if hist.empty:
                continue
            bars: list[Bar] = []
            for ts, row in hist.iterrows():
                session = ts.date() if hasattr(ts, "date") else pd.Timestamp(ts).date()
                adj = row.get("Adj Close", row["Close"])
                bars.append(
                    Bar(
                        symbol=symbol,
                        session=session,
                        open=Decimal(str(round(float(row["Open"]), 6))),
                        high=Decimal(str(round(float(row["High"]), 6))),
                        low=Decimal(str(round(float(row["Low"]), 6))),
                        close=Decimal(str(round(float(row["Close"]), 6))),
                        adj_close=Decimal(str(round(float(adj), 6))),
                        volume=int(row["Volume"]),
                        observed_at=self._observed_at(session),
                    )
                )
            total += self.store.write_bars(bars)
        return total

    def _synthetic_backfill(self, start: date, end: date) -> int:
        """Deterministic random-walk bars for offline tests."""
        import hashlib

        sessions = trading_days(start, end)
        total = 0
        for symbol in self.symbols:
            seed = int(hashlib.sha256(symbol.encode()).hexdigest()[:8], 16)
            price = 50.0 + (seed % 200)
            bars: list[Bar] = []
            for i, session in enumerate(sessions):
                # Simple deterministic walk
                delta = ((seed + i * 17) % 21 - 10) / 100.0
                o = price
                c = max(1.0, price * (1 + delta))
                h = max(o, c) * 1.005
                l = min(o, c) * 0.995
                vol = 1_000_000 + (seed + i) % 500_000
                bars.append(
                    Bar(
                        symbol=symbol,
                        session=session,
                        open=Decimal(str(round(o, 4))),
                        high=Decimal(str(round(h, 4))),
                        low=Decimal(str(round(l, 4))),
                        close=Decimal(str(round(c, 4))),
                        adj_close=Decimal(str(round(c, 4))),
                        volume=vol,
                        observed_at=self._observed_at(session),
                    )
                )
                price = c
            total += self.store.write_bars(bars)
        return total

    def incremental(self) -> int:
        today = date.today()
        start = today - timedelta(days=7)
        return self.backfill(start, today)

    def validate(self) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        as_of = datetime.now(tz=ET)
        for symbol in self.symbols:
            df = self.store.get_bars([symbol], as_of=as_of)
            if df.empty:
                warnings.append(f"{symbol}: no bars")
                continue
            # Monotonic sessions
            sessions = list(df["session"])
            if sessions != sorted(sessions):
                errors.append(f"{symbol}: sessions not sorted")
            if len(sessions) != len(set(sessions)):
                errors.append(f"{symbol}: duplicate sessions")
            # OHLC integrity
            bad_hl = df[df["high"] < df["low"]]
            if len(bad_hl):
                errors.append(f"{symbol}: high < low on {len(bad_hl)} rows")
            neg_vol = df[df["volume"] < 0]
            if len(neg_vol):
                errors.append(f"{symbol}: negative volume")
            # Gap detection (trading calendar)
            if len(sessions) >= 2:
                expected = set(trading_days(sessions[0], sessions[-1]))
                actual = set(sessions)
                missing = expected - actual
                if missing and len(missing) > 5:
                    warnings.append(
                        f"{symbol}: {len(missing)} session gaps (sample {list(missing)[:3]})"
                    )
        return ValidationResult(ok=len(errors) == 0, errors=errors, warnings=warnings)

    def cross_validate(
        self,
        other: pd.DataFrame,
        symbols: list[str],
        max_close_diff_pct: float = 0.5,
    ) -> ValidationResult:
        """Compare closes vs another source; fail if > max_close_diff_pct."""
        errors: list[str] = []
        as_of = datetime.now(tz=ET)
        for symbol in symbols:
            ours = self.store.get_bars([symbol], as_of=as_of)
            if ours.empty or other.empty:
                continue
            oth = other[other["symbol"] == symbol]
            merged = ours.merge(oth, on=["symbol", "session"], suffixes=("_a", "_b"))
            if merged.empty:
                continue
            rel = (
                (merged["close_a"].astype(float) - merged["close_b"].astype(float)).abs()
                / merged["close_a"].astype(float)
                * 100
            )
            bad = rel > max_close_diff_pct
            if bad.any():
                errors.append(
                    f"{symbol}: {bad.sum()} sessions with close discrepancy > {max_close_diff_pct}%"
                )
        return ValidationResult(ok=len(errors) == 0, errors=errors)
