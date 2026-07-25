"""INV-1: point-in-time integrity gates."""

from __future__ import annotations

import inspect
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from fund.store import bars as bars_mod
from fund.store import fundamentals as fund_mod
from fund.store import news as news_mod
from fund.store.bars import BarStore
from fund.store.fundamentals import FundamentalsStore
from fund.store.news import NewsStore
from fund.store.pit import assert_as_of_param, list_public_readers
from fund.store.universe import UniverseStore
from fund.types import Bar


def test_no_reader_without_as_of():
    modules = [bars_mod, fund_mod, news_mod]
    for mod in modules:
        for name, fn in list_public_readers(mod):
            # skip non-instance write helpers if any
            if "write" in name.lower():
                continue
            try:
                assert_as_of_param(fn, name)
            except AssertionError:
                # Methods bound without self signature still have as_of
                sig = inspect.signature(fn)
                if "as_of" not in sig.parameters:
                    raise


def test_future_bar_invisible(tmp_path: Path):
    store = BarStore(tmp_path)
    past = datetime(2024, 1, 2, 16, 15, tzinfo=timezone.utc)
    future = datetime(2024, 1, 10, 16, 15, tzinfo=timezone.utc)
    store.write_bars(
        [
            Bar(
                symbol="AAA",
                session=date(2024, 1, 2),
                open=Decimal("10"),
                high=Decimal("11"),
                low=Decimal("9"),
                close=Decimal("10.5"),
                adj_close=Decimal("10.5"),
                volume=1_000_000,
                observed_at=past,
            ),
            Bar(
                symbol="AAA",
                session=date(2024, 1, 10),
                open=Decimal("20"),
                high=Decimal("21"),
                low=Decimal("19"),
                close=Decimal("20.5"),
                adj_close=Decimal("20.5"),
                volume=1_000_000,
                observed_at=future,
            ),
        ]
    )
    as_of = datetime(2024, 1, 5, 16, 15, tzinfo=timezone.utc)
    df = store.get_bars(["AAA"], as_of=as_of)
    assert len(df) == 1
    assert df.iloc[0]["session"] == date(2024, 1, 2)
    assert float(df.iloc[0]["close"]) == 10.5


def test_fundamental_vintage(tmp_path: Path):
    store = FundamentalsStore(tmp_path)
    period = date(2022, 12, 31)
    first = datetime(2023, 2, 15, tzinfo=timezone.utc)
    restate = datetime(2024, 2, 20, tzinfo=timezone.utc)
    store.write_facts(
        [
            {
                "symbol": "AAA",
                "metric": "pe_ttm",
                "period_end": period,
                "vintage": "2023-02-15",
                "value": 20.0,
                "observed_at": first,
            },
            {
                "symbol": "AAA",
                "metric": "pe_ttm",
                "period_end": period,
                "vintage": "2024-02-20",
                "value": 15.0,  # restated
                "observed_at": restate,
            },
        ]
    )
    early = store.get_metric_as_of(
        "AAA", "pe_ttm", as_of=datetime(2023, 6, 1, tzinfo=timezone.utc)
    )
    late = store.get_metric_as_of(
        "AAA", "pe_ttm", as_of=datetime(2024, 6, 1, tzinfo=timezone.utc)
    )
    assert early == 20.0
    assert late == 15.0


def test_survivorship(tmp_path: Path):
    store = UniverseStore(tmp_path)
    store.write_membership(
        [
            {
                "symbol": "DEAD",
                "index": "test",
                "start_date": date(2020, 1, 1),
                "end_date": date(2021, 6, 1),
            },
            {
                "symbol": "LIVE",
                "index": "test",
                "start_date": date(2020, 1, 1),
                "end_date": None,
            },
        ]
    )
    mid = store.get_universe(datetime(2021, 1, 1, tzinfo=timezone.utc), index="test")
    after = store.get_universe(datetime(2022, 1, 1, tzinfo=timezone.utc), index="test")
    assert "DEAD" in mid
    assert "LIVE" in mid
    assert "DEAD" not in after
    assert "LIVE" in after


def test_news_pit(tmp_path: Path):
    store = NewsStore(tmp_path)
    t0 = datetime(2024, 3, 1, 12, 0, tzinfo=timezone.utc)
    store.write_articles(
        [
            {
                "article_id": "a1",
                "symbols": "AAA",
                "headline": "old",
                "body": "",
                "event_time": t0,
                "observed_at": t0 + timedelta(minutes=2),
                "source": "test",
            },
            {
                "article_id": "a2",
                "symbols": "AAA",
                "headline": "future",
                "body": "",
                "event_time": t0 + timedelta(days=10),
                "observed_at": t0 + timedelta(days=10, minutes=2),
                "source": "test",
            },
        ]
    )
    df = store.get_news(["AAA"], as_of=t0 + timedelta(days=1), lookback_days=30)
    assert len(df) == 1
    assert df.iloc[0]["headline"] == "old"
