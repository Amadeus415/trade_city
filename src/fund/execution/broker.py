"""BrokerAdapter protocol — INV-3: same path for backtest/paper/live."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from fund.types import (
    BrokerOrderStatus,
    OrderAck,
    OrderIntent,
    OrderReview,
    PortfolioSnapshot,
    Quote,
)


@runtime_checkable
class BrokerAdapter(Protocol):
    def get_portfolio(self) -> PortfolioSnapshot: ...

    def get_quotes(self, symbols: list[str]) -> dict[str, Quote]: ...

    def review_order(self, intent: OrderIntent) -> OrderReview: ...

    def place_order(self, intent: OrderIntent) -> OrderAck: ...

    def cancel_order(self, broker_order_id: str) -> None: ...

    def list_orders(self, since: datetime) -> list[BrokerOrderStatus]: ...
