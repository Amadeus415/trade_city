"""Cost model: limit non-fill when price does not cross."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from fund.config import CostConfig
from fund.execution.simulated import SimulatedBroker
from fund.types import OrderIntent, Side


def test_limit_nonfill():
    broker = SimulatedBroker(
        Decimal("10000"),
        CostConfig(),
        seed=1,
        nonfill_rate=0.0,  # isolate price-cross rule
    )
    intent = OrderIntent(
        client_order_id="c1",
        decision_id="d1",
        symbol="AAA",
        side=Side.BUY,
        quantity=Decimal("10"),
        limit_price=Decimal("50"),  # buy limit below the day's low
        created_at=datetime.now(tz=timezone.utc),
    )
    broker.place_order(intent)
    filled, px, fees = broker.try_fill(
        intent,
        {"open": 60, "high": 62, "low": 55, "close": 58},  # low 55 > limit 50
    )
    assert filled is False
    assert px == 0


def test_limit_fill_when_crosses():
    broker = SimulatedBroker(
        Decimal("10000"),
        CostConfig(),
        seed=0,
        nonfill_rate=0.0,
    )
    intent = OrderIntent(
        client_order_id="c2",
        decision_id="d2",
        symbol="AAA",
        side=Side.BUY,
        quantity=Decimal("10"),
        limit_price=Decimal("60"),
        created_at=datetime.now(tz=timezone.utc),
    )
    broker.place_order(intent)
    filled, px, fees = broker.try_fill(
        intent,
        {"open": 58, "high": 62, "low": 55, "close": 59},
    )
    assert filled is True
    assert px > 0
    assert fees >= 0
