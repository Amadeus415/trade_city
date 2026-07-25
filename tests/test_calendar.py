"""Exchange calendar: no decisions on holidays; half-days detected."""

from __future__ import annotations

from datetime import date

from fund.clock import is_half_day, is_trading_day, next_trading_day, trading_days


def test_weekend_not_trading():
    # 2024-07-06 is Saturday
    assert is_trading_day(date(2024, 7, 6)) is False
    assert is_trading_day(date(2024, 7, 4)) is False  # Independence Day
    assert is_trading_day(date(2024, 7, 5)) is True


def test_next_trading_day_skips_weekend():
    # Friday 2024-07-05 -> Monday 2024-07-08
    assert next_trading_day(date(2024, 7, 5)) == date(2024, 7, 8)


def test_trading_days_range():
    days = trading_days(date(2024, 7, 1), date(2024, 7, 10))
    assert date(2024, 7, 4) not in days
    assert all(is_trading_day(d) for d in days)


def test_half_day_thanksgiving_eve():
    # Day before Thanksgiving 2024-11-27 is early close
    assert is_half_day(date(2024, 11, 29)) is True  # Black Friday early close often
    # Thanksgiving itself is closed
    assert is_trading_day(date(2024, 11, 28)) is False
