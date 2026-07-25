"""ALL time goes through here.

session (trading date) vs observed_at (wall-clock instant) must stay distinct.
Never use datetime.now() or date.today() outside this module.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas_market_calendars as mcal

ET = ZoneInfo("America/New_York")
UTC = timezone.utc

_NYSE = mcal.get_calendar("NYSE")


class Clock:
    """Injectable clock for production and deterministic tests."""

    def __init__(self, now: datetime | None = None) -> None:
        self._fixed = now

    def now(self) -> datetime:
        if self._fixed is not None:
            return self._fixed
        return datetime.now(tz=ET)

    def now_utc(self) -> datetime:
        return self.now().astimezone(UTC)

    def set_fixed(self, when: datetime) -> None:
        if when.tzinfo is None:
            when = when.replace(tzinfo=ET)
        self._fixed = when

    def advance(self, **kwargs: float) -> None:
        if self._fixed is None:
            raise RuntimeError("advance() requires a fixed clock")
        self._fixed = self._fixed + timedelta(**kwargs)

    def today(self) -> date:
        return self.now().date()

    def session_close_as_of(
        self,
        session: date,
        lag_minutes: int = 15,
    ) -> datetime:
        """Point-in-time as_of for daily bar decisions: session close + lag."""
        close_dt = datetime.combine(session, time(16, 0), tzinfo=ET)
        return close_dt + timedelta(minutes=lag_minutes)


def ensure_tz(dt: datetime, tz: ZoneInfo = ET) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=tz)
    return dt.astimezone(tz)


def is_trading_day(d: date) -> bool:
    schedule = _NYSE.schedule(start_date=d, end_date=d)
    return len(schedule) > 0


def is_half_day(d: date) -> bool:
    """True if NYSE early close (e.g. day before Thanksgiving)."""
    schedule = _NYSE.schedule(start_date=d, end_date=d)
    if schedule.empty:
        return False
    market_close = schedule.iloc[0]["market_close"]
    # Early close is typically 13:00 ET
    close_et = market_close.tz_convert(ET) if hasattr(market_close, "tz_convert") else market_close
    return close_et.hour < 16


def next_trading_day(d: date) -> date:
    candidate = d + timedelta(days=1)
    for _ in range(15):
        if is_trading_day(candidate):
            return candidate
        candidate += timedelta(days=1)
    raise RuntimeError(f"no trading day found after {d}")


def prev_trading_day(d: date) -> date:
    candidate = d - timedelta(days=1)
    for _ in range(15):
        if is_trading_day(candidate):
            return candidate
        candidate -= timedelta(days=1)
    raise RuntimeError(f"no trading day found before {d}")


def trading_days(start: date, end: date) -> list[date]:
    schedule = _NYSE.valid_days(start_date=start, end_date=end)
    return [ts.date() for ts in schedule]


def market_open_close(session: date) -> tuple[datetime, datetime]:
    schedule = _NYSE.schedule(start_date=session, end_date=session)
    if schedule.empty:
        raise ValueError(f"{session} is not a trading day")
    open_ts = schedule.iloc[0]["market_open"].tz_convert(ET).to_pydatetime()
    close_ts = schedule.iloc[0]["market_close"].tz_convert(ET).to_pydatetime()
    return open_ts, close_ts
