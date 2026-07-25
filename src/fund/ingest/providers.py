"""Market data providers. Primary + secondary for cross-validation."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Protocol
from zoneinfo import ZoneInfo

import pandas as pd

from fund.types import Bar

ET = ZoneInfo("America/New_York")


def observed_at_for_session(session: date, lag_minutes: int = 15) -> datetime:
    close = datetime.combine(session, time(16, 0), tzinfo=ET)
    return close + timedelta(minutes=lag_minutes)


class BarProvider(Protocol):
    name: str

    def fetch(
        self,
        symbols: list[str],
        start: date,
        end: date,
        *,
        lag_minutes: int = 15,
    ) -> list[Bar]: ...


class YFinanceProvider:
    name = "yfinance"

    def fetch(
        self,
        symbols: list[str],
        start: date,
        end: date,
        *,
        lag_minutes: int = 15,
    ) -> list[Bar]:
        import yfinance as yf

        out: list[Bar] = []
        # Batch download is faster
        data = yf.download(
            symbols,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False,
        )
        if data is None or data.empty:
            return out

        multi = isinstance(data.columns, pd.MultiIndex)
        for symbol in symbols:
            try:
                if multi:
                    if symbol not in data.columns.get_level_values(0):
                        # single-symbol download flattens sometimes
                        hist = data if len(symbols) == 1 else None
                        if hist is None:
                            continue
                    else:
                        hist = data[symbol].dropna(how="all")
                else:
                    hist = data.dropna(how="all")
                if hist is None or hist.empty:
                    continue
                for ts, row in hist.iterrows():
                    if pd.isna(row.get("Close", float("nan"))):
                        continue
                    session = ts.date() if hasattr(ts, "date") else pd.Timestamp(ts).date()
                    adj = row["Adj Close"] if "Adj Close" in row.index and not pd.isna(row["Adj Close"]) else row["Close"]
                    vol = row.get("Volume", 0)
                    out.append(
                        Bar(
                            symbol=symbol,
                            session=session,
                            open=Decimal(str(round(float(row["Open"]), 6))),
                            high=Decimal(str(round(float(row["High"]), 6))),
                            low=Decimal(str(round(float(row["Low"]), 6))),
                            close=Decimal(str(round(float(row["Close"]), 6))),
                            adj_close=Decimal(str(round(float(adj), 6))),
                            volume=int(vol) if not pd.isna(vol) else 0,
                            observed_at=observed_at_for_session(session, lag_minutes),
                        )
                    )
            except Exception:
                # Per-symbol fallback
                try:
                    hist = yf.Ticker(symbol).history(
                        start=start.isoformat(),
                        end=(end + timedelta(days=1)).isoformat(),
                        auto_adjust=False,
                    )
                except Exception:
                    continue
                if hist.empty:
                    continue
                for ts, row in hist.iterrows():
                    session = ts.date() if hasattr(ts, "date") else pd.Timestamp(ts).date()
                    adj = row.get("Adj Close", row["Close"])
                    out.append(
                        Bar(
                            symbol=symbol,
                            session=session,
                            open=Decimal(str(round(float(row["Open"]), 6))),
                            high=Decimal(str(round(float(row["High"]), 6))),
                            low=Decimal(str(round(float(row["Low"]), 6))),
                            close=Decimal(str(round(float(row["Close"]), 6))),
                            adj_close=Decimal(str(round(float(adj), 6))),
                            volume=int(row["Volume"]),
                            observed_at=observed_at_for_session(session, lag_minutes),
                        )
                    )
        return out


class StooqProvider:
    """Secondary free source for cross-validation (CSV)."""

    name = "stooq"

    def fetch(
        self,
        symbols: list[str],
        start: date,
        end: date,
        *,
        lag_minutes: int = 15,
    ) -> list[Bar]:
        import httpx

        out: list[Bar] = []
        for symbol in symbols:
            # Stooq US equities: aapl.us
            slug = f"{symbol.lower()}.us"
            url = f"https://stooq.com/q/d/l/?s={slug}&i=d"
            try:
                with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                    resp = client.get(url)
                    resp.raise_for_status()
                    text = resp.text
            except Exception:
                continue
            if not text or "Date" not in text[:50]:
                continue
            from io import StringIO

            df = pd.read_csv(StringIO(text))
            if df.empty or "Date" not in df.columns:
                continue
            df["Date"] = pd.to_datetime(df["Date"]).dt.date
            df = df[(df["Date"] >= start) & (df["Date"] <= end)]
            for _, row in df.iterrows():
                session = row["Date"]
                c = float(row["Close"])
                out.append(
                    Bar(
                        symbol=symbol,
                        session=session,
                        open=Decimal(str(round(float(row["Open"]), 6))),
                        high=Decimal(str(round(float(row["High"]), 6))),
                        low=Decimal(str(round(float(row["Low"]), 6))),
                        close=Decimal(str(round(c, 6))),
                        adj_close=Decimal(str(round(c, 6))),  # stooq may already adjust
                        volume=int(row.get("Volume", 0) or 0),
                        observed_at=observed_at_for_session(session, lag_minutes),
                    )
                )
        return out


class SyntheticProvider:
    """Deterministic offline provider for tests."""

    name = "synthetic"

    def fetch(
        self,
        symbols: list[str],
        start: date,
        end: date,
        *,
        lag_minutes: int = 15,
    ) -> list[Bar]:
        import hashlib

        from fund.clock import trading_days

        sessions = trading_days(start, end)
        out: list[Bar] = []
        for symbol in symbols:
            seed = int(hashlib.sha256(symbol.encode()).hexdigest()[:8], 16)
            price = 50.0 + (seed % 200)
            for i, session in enumerate(sessions):
                delta = ((seed + i * 17) % 21 - 10) / 100.0
                o = price
                c = max(1.0, price * (1 + delta))
                h = max(o, c) * 1.005
                l = min(o, c) * 0.995
                vol = 1_000_000 + (seed + i) % 500_000
                out.append(
                    Bar(
                        symbol=symbol,
                        session=session,
                        open=Decimal(str(round(o, 4))),
                        high=Decimal(str(round(h, 4))),
                        low=Decimal(str(round(l, 4))),
                        close=Decimal(str(round(c, 4))),
                        adj_close=Decimal(str(round(c, 4))),
                        volume=vol,
                        observed_at=observed_at_for_session(session, lag_minutes),
                    )
                )
                price = c
        return out


def get_provider(name: str) -> BarProvider:
    key = (name or "yfinance").lower()
    if key == "yfinance":
        try:
            import yfinance  # noqa: F401

            return YFinanceProvider()
        except ImportError:
            return SyntheticProvider()
    if key == "stooq":
        return StooqProvider()
    if key == "synthetic":
        return SyntheticProvider()
    raise ValueError(f"unknown bar provider: {name}")


def bars_to_frame(bars: list[Bar]) -> pd.DataFrame:
    if not bars:
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
    return pd.DataFrame([b.model_dump(mode="python") for b in bars])
