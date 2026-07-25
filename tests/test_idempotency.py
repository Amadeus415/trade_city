"""Order idempotency and timeout-then-reconcile."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from fund.execution.idempotency import client_order_id
from fund.execution.reconcile import reconcile
from fund.execution.simulated import SimulatedBroker
from fund.store.journal import Journal
from fund.types import OrderIntent, OrderStatus, Side
from fund.config import CostConfig


def test_duplicate_intent_rejected(tmp_path: Path):
    j = Journal(tmp_path / "j.db")
    j.migrate()
    coid = client_order_id("dec1", "AAA", Side.BUY, Decimal("10"))
    intent = OrderIntent(
        client_order_id=coid,
        decision_id="dec1",
        symbol="AAA",
        side=Side.BUY,
        quantity=Decimal("10"),
        limit_price=Decimal("100"),
        created_at=datetime.now(tz=timezone.utc),
    )
    assert j.insert_order_intent(intent) is True
    assert j.insert_order_intent(intent) is False
    j.close()


def test_timeout_then_reconcile(tmp_path: Path):
    """Simulated: order exists at broker after 'timeout' ⇒ no second place."""
    j = Journal(tmp_path / "j.db")
    j.migrate()
    broker = SimulatedBroker(Decimal("10000"), CostConfig())
    coid = client_order_id("dec2", "AAA", Side.BUY, Decimal("5"))
    intent = OrderIntent(
        client_order_id=coid,
        decision_id="dec2",
        symbol="AAA",
        side=Side.BUY,
        quantity=Decimal("5"),
        limit_price=Decimal("50"),
        created_at=datetime.now(tz=timezone.utc),
    )
    assert j.insert_order_intent(intent)
    # "Timeout" after send: broker has it, journal still intent
    ack = broker.place_order(intent)
    # Do NOT update journal (simulates crash/timeout)
    # Reconcile should pick up acknowledged from broker
    # First, mark broker order as filled for terminal resolve path
    from fund.types import BrokerOrderStatus

    broker._orders[coid] = BrokerOrderStatus(
        client_order_id=coid,
        broker_order_id=ack.broker_order_id,
        status=OrderStatus.FILLED,
        filled_qty=Decimal("5"),
        avg_fill_price=Decimal("50"),
    )
    result = reconcile(j, broker, since=datetime.now(tz=timezone.utc))
    row = j.get_order(coid)
    assert row is not None
    assert row["status"] == OrderStatus.FILLED.value
    # Second insert still blocked
    assert j.insert_order_intent(intent) is False
    j.close()
    assert result["resolved"] >= 1


def test_client_order_id_stable():
    a = client_order_id("d", "X", Side.BUY, Decimal("1.5"))
    b = client_order_id("d", "X", Side.BUY, Decimal("1.5"))
    c = client_order_id("d", "X", Side.SELL, Decimal("1.5"))
    assert a == b
    assert a != c
    assert len(a) == 32
