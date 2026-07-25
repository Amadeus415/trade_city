"""INV-3: identical execution path shape for sim vs mocked live adapter."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from fund.config import CostConfig
from fund.execution.idempotency import client_order_id
from fund.execution.simulated import SimulatedBroker
from fund.store.journal import Journal
from fund.types import (
    BrokerOrderStatus,
    OrderAck,
    OrderIntent,
    OrderReview,
    OrderStatus,
    PortfolioSnapshot,
    Quote,
    Side,
)


class MockRobinhood:
    """Minimal stand-in that implements BrokerAdapter without network."""

    def __init__(self) -> None:
        self._orders: dict[str, BrokerOrderStatus] = {}
        self.cash = Decimal("10000")

    def get_portfolio(self) -> PortfolioSnapshot:
        return PortfolioSnapshot(
            as_of=datetime.now(tz=timezone.utc),
            cash=self.cash,
            equity=self.cash,
            positions=[],
            peak_equity=self.cash,
        )

    def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        return {
            s: Quote(
                symbol=s,
                bid=Decimal("99"),
                ask=Decimal("101"),
                last=Decimal("100"),
                as_of=datetime.now(tz=timezone.utc),
            )
            for s in symbols
        }

    def review_order(self, intent: OrderIntent) -> OrderReview:
        return OrderReview(
            client_order_id=intent.client_order_id,
            approved=True,
            warnings=[],
            raw={"mock": True},
        )

    def place_order(self, intent: OrderIntent) -> OrderAck:
        bid = f"rh-{intent.client_order_id[:8]}"
        self._orders[intent.client_order_id] = BrokerOrderStatus(
            client_order_id=intent.client_order_id,
            broker_order_id=bid,
            status=OrderStatus.ACKNOWLEDGED,
        )
        return OrderAck(
            client_order_id=intent.client_order_id,
            broker_order_id=bid,
            status=OrderStatus.ACKNOWLEDGED,
            raw={"mock": True},
        )

    def cancel_order(self, broker_order_id: str) -> None:
        pass

    def list_orders(self, since: datetime) -> list[BrokerOrderStatus]:
        return list(self._orders.values())


def _run_path(journal: Journal, broker: Any, decision_id: str) -> dict:
    coid = client_order_id(decision_id, "AAA", Side.BUY, Decimal("3"))
    intent = OrderIntent(
        client_order_id=coid,
        decision_id=decision_id,
        symbol="AAA",
        side=Side.BUY,
        quantity=Decimal("3"),
        limit_price=Decimal("100"),
        created_at=datetime.now(tz=timezone.utc),
    )
    assert journal.insert_order_intent(intent)
    review = broker.review_order(intent)
    assert review.approved
    ack = broker.place_order(intent)
    journal.update_order(
        coid,
        status=ack.status,
        broker_order_id=ack.broker_order_id,
        review_response=str(review.raw),
    )
    return journal.get_order(coid)  # type: ignore[return-value]


def test_sim_vs_mock_rh_journal_shape(tmp_path: Path):
    j1 = Journal(tmp_path / "sim.db")
    j1.migrate()
    j2 = Journal(tmp_path / "rh.db")
    j2.migrate()

    sim = SimulatedBroker(Decimal("10000"), CostConfig())
    rh = MockRobinhood()

    r1 = _run_path(j1, sim, "dec-sim")
    r2 = _run_path(j2, rh, "dec-rh")

    # Same columns / status lifecycle
    for key in (
        "client_order_id",
        "decision_id",
        "symbol",
        "side",
        "quantity",
        "limit_price",
        "status",
        "broker_order_id",
    ):
        assert key in r1 and key in r2
    assert r1["status"] == OrderStatus.ACKNOWLEDGED.value
    assert r2["status"] == OrderStatus.ACKNOWLEDGED.value
    assert r1["symbol"] == r2["symbol"] == "AAA"
    j1.close()
    j2.close()
