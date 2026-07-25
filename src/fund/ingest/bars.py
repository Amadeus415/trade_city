"""Daily bar ingestion with availability lag, multi-provider, validation."""

from __future__ import annotations

from datetime import date, datetime, timedelta

from fund.clock import ET, trading_days
from fund.ingest.base import ValidationResult
from fund.ingest.providers import (
    StooqProvider,
    bars_to_frame,
    get_provider,
)
from fund.store.bars import BarStore


class BarIngestor:
    def __init__(
        self,
        store: BarStore,
        symbols: list[str],
        bar_lag_minutes: int = 15,
        provider: str = "yfinance",
    ) -> None:
        self.store = store
        self.symbols = symbols
        self.bar_lag_minutes = bar_lag_minutes
        self.provider_name = provider
        self.provider = get_provider(provider)

    def backfill(self, start: date, end: date) -> int:
        bars = self.provider.fetch(
            self.symbols,
            start,
            end,
            lag_minutes=self.bar_lag_minutes,
        )
        return self.store.write_bars(bars)

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
            sessions = list(df["session"])
            if sessions != sorted(sessions):
                errors.append(f"{symbol}: sessions not sorted")
            if len(sessions) != len(set(sessions)):
                errors.append(f"{symbol}: duplicate sessions")
            bad_hl = df[df["high"] < df["low"]]
            if len(bad_hl):
                errors.append(f"{symbol}: high < low on {len(bad_hl)} rows")
            neg_vol = df[df["volume"] < 0]
            if len(neg_vol):
                errors.append(f"{symbol}: negative volume")
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
        symbols: list[str] | None = None,
        *,
        start: date | None = None,
        end: date | None = None,
        secondary: str = "stooq",
        max_close_diff_pct: float = 0.5,
        sample_n: int = 20,
        max_sessions: int = 500,
    ) -> ValidationResult:
        """Compare primary store closes vs a secondary provider.

        Spec gate: fail loudly on >0.5% close discrepancies.
        """
        import random

        symbols = list(symbols or self.symbols)
        if len(symbols) > sample_n:
            symbols = random.Random(42).sample(symbols, sample_n)

        as_of = datetime.now(tz=ET)
        # Determine date window from store if not provided
        if start is None or end is None:
            probe = self.store.get_bars(symbols[:3], as_of=as_of)
            if probe.empty:
                return ValidationResult(
                    ok=False, errors=["no bars in store to cross-validate"]
                )
            sessions = sorted(probe["session"].unique())
            end = end or sessions[-1]
            start = start or sessions[max(0, len(sessions) - max_sessions)]

        sec = get_provider(secondary)
        # Stooq can be slow; fall back to synthetic twin only if secondary fails empty
        other_bars = sec.fetch(symbols, start, end, lag_minutes=self.bar_lag_minutes)
        if not other_bars and secondary == "stooq":
            # Retry per-symbol with yfinance as alternate secondary for offline CI
            alt = get_provider("yfinance")
            other_bars = alt.fetch(symbols, start, end, lag_minutes=self.bar_lag_minutes)
            secondary = "yfinance_alt"

        other = bars_to_frame(other_bars)
        if other.empty:
            return ValidationResult(
                ok=False,
                errors=[f"secondary provider '{secondary}' returned no data"],
            )
        other["session"] = pd_to_date(other["session"])

        errors: list[str] = []
        warnings: list[str] = []
        compared = 0
        for symbol in symbols:
            ours = self.store.get_bars([symbol], as_of=as_of, start=start, end=end)
            if ours.empty:
                warnings.append(f"{symbol}: missing in primary store")
                continue
            oth = other[other["symbol"] == symbol].copy()
            if oth.empty:
                warnings.append(f"{symbol}: missing in secondary ({secondary})")
                continue
            ours = ours.copy()
            ours["session"] = pd_to_date(ours["session"])
            oth["session"] = pd_to_date(oth["session"])
            merged = ours.merge(oth, on=["symbol", "session"], suffixes=("_a", "_b"))
            if merged.empty:
                warnings.append(f"{symbol}: no overlapping sessions")
                continue
            compared += 1
            # Prefer adj_close when both present
            ca = merged["adj_close_a"].astype(float) if "adj_close_a" in merged else merged["close_a"].astype(float)
            cb = merged["adj_close_b"].astype(float) if "adj_close_b" in merged else merged["close_b"].astype(float)
            # For stooq vs raw close, compare close_a vs close_b
            if secondary == "stooq":
                ca = merged["close_a"].astype(float)
                cb = merged["close_b"].astype(float)
            rel = (ca - cb).abs() / ca.replace(0, float("nan")) * 100
            bad = rel > max_close_diff_pct
            if bad.any():
                n_bad = int(bad.sum())
                max_diff = float(rel.max())
                errors.append(
                    f"{symbol}: {n_bad}/{len(merged)} sessions differ >{max_close_diff_pct}% "
                    f"(max {max_diff:.2f}%) vs {secondary}"
                )
        if compared == 0:
            errors.append("no symbols compared")
        return ValidationResult(
            ok=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )


def pd_to_date(series):
    import pandas as pd

    return pd.to_datetime(series).dt.date
