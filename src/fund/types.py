"""Core domain models. Everything else references these.

Money and quantities use Decimal. Pydantic rejects float coercion on those fields.
session (trading date) and observed_at (wall-clock instant) must never be conflated.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _reject_float(v: Any) -> Any:
    if isinstance(v, float):
        raise TypeError("float is not allowed for money/quantity fields; use Decimal or str")
    return v


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


class Action(StrEnum):
    OPEN = "open"
    ADD = "add"
    TRIM = "trim"
    CLOSE = "close"
    HOLD = "hold"
    ABSTAIN = "abstain"  # explicitly "I have no view" — not the same as HOLD


class RiskVerdict(StrEnum):
    ACCEPTED = "accepted"
    CLAMPED = "clamped"
    REJECTED = "rejected"


class RiskStateName(StrEnum):
    NORMAL = "NORMAL"
    REDUCED = "REDUCED"
    HALTED = "HALTED"


class OrderStatus(StrEnum):
    INTENT = "intent"
    ACKNOWLEDGED = "acknowledged"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"
    REJECTED = "rejected"


class MaskingMode(StrEnum):
    BRIGHT = "bright"
    STOCK_BLIND = "stock_blind"
    DATE_BLIND = "date_blind"
    BLINDED = "blinded"


class DecimalModel(BaseModel):
    """Base model that forbids float coercion on Decimal fields via validators."""

    model_config = ConfigDict(extra="forbid", strict=False)


class Bar(DecimalModel):
    symbol: str
    session: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    adj_close: Decimal
    volume: int
    observed_at: datetime  # when WE learned it. INV-1 depends on this.

    @field_validator("open", "high", "low", "close", "adj_close", mode="before")
    @classmethod
    def _no_float_prices(cls, v: Any) -> Any:
        return _reject_float(v)


class Quote(DecimalModel):
    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    as_of: datetime

    @field_validator("bid", "ask", "last", mode="before")
    @classmethod
    def _no_float(cls, v: Any) -> Any:
        return _reject_float(v)

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / 2

    @property
    def spread_bps(self) -> Decimal:
        mid = self.mid
        if mid == 0:
            return Decimal("0")
        return ((self.ask - self.bid) / mid) * Decimal("10000")


class Position(DecimalModel):
    symbol: str
    quantity: Decimal
    avg_cost: Decimal
    market_value: Decimal
    opened_at: date
    sector: str = "unknown"
    cluster_id: str = "none"

    @field_validator("quantity", "avg_cost", "market_value", mode="before")
    @classmethod
    def _no_float(cls, v: Any) -> Any:
        return _reject_float(v)


class PortfolioSnapshot(DecimalModel):
    as_of: datetime
    cash: Decimal
    equity: Decimal
    positions: list[Position]
    peak_equity: Decimal

    @field_validator("cash", "equity", "peak_equity", mode="before")
    @classmethod
    def _no_float(cls, v: Any) -> Any:
        return _reject_float(v)

    def weight(self, symbol: str) -> Decimal:
        if self.equity <= 0:
            return Decimal("0")
        for p in self.positions:
            if p.symbol == symbol:
                return p.market_value / self.equity
        return Decimal("0")

    def position_map(self) -> dict[str, Position]:
        return {p.symbol: p for p in self.positions}


class Proposal(DecimalModel):
    """Output of the agent layer. NOT yet an order."""

    symbol: str
    action: Action
    target_weight: Decimal = Field(ge=0, le=1)
    confidence: Decimal = Field(ge=0, le=1)
    thesis: str = Field(max_length=1200)
    invalidation: str = Field(max_length=400)
    horizon_days: int = Field(ge=1, le=250)
    source_features: list[str]

    @field_validator("target_weight", "confidence", mode="before")
    @classmethod
    def _no_float(cls, v: Any) -> Any:
        return _reject_float(v)


class RiskDecision(DecimalModel):
    proposal: Proposal
    verdict: RiskVerdict
    final_weight: Decimal
    reasons: list[str]
    limits_version: str

    @field_validator("final_weight", mode="before")
    @classmethod
    def _no_float(cls, v: Any) -> Any:
        return _reject_float(v)


class OrderIntent(DecimalModel):
    client_order_id: str
    decision_id: str
    symbol: str
    side: Side
    quantity: Decimal
    limit_price: Decimal
    time_in_force: str = "day"
    created_at: datetime

    @field_validator("quantity", "limit_price", mode="before")
    @classmethod
    def _no_float(cls, v: Any) -> Any:
        return _reject_float(v)


class OrderReview(DecimalModel):
    client_order_id: str
    approved: bool
    warnings: list[str] = Field(default_factory=list)
    estimated_cost: Decimal | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class OrderAck(DecimalModel):
    client_order_id: str
    broker_order_id: str
    status: OrderStatus
    raw: dict[str, Any] = Field(default_factory=dict)


class BrokerOrderStatus(DecimalModel):
    client_order_id: str
    broker_order_id: str | None
    status: OrderStatus
    filled_qty: Decimal = Decimal("0")
    avg_fill_price: Decimal | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class Fill(DecimalModel):
    fill_id: str
    client_order_id: str
    quantity: Decimal
    price: Decimal
    fees: Decimal
    filled_at: datetime

    @field_validator("quantity", "price", "fees", mode="before")
    @classmethod
    def _no_float(cls, v: Any) -> Any:
        return _reject_float(v)


class MarketContext(DecimalModel):
    """Snapshot of market data the risk engine needs for order-level gates."""

    as_of: datetime
    last_prices: dict[str, Decimal] = Field(default_factory=dict)
    spreads_bps: dict[str, Decimal] = Field(default_factory=dict)
    adv_notional: dict[str, Decimal] = Field(default_factory=dict)
    sectors: dict[str, str] = Field(default_factory=dict)
    clusters: dict[str, str] = Field(default_factory=dict)
    halted_symbols: set[str] = Field(default_factory=set)
    days_to_earnings: dict[str, int] = Field(default_factory=dict)
    days_listed: dict[str, int] = Field(default_factory=dict)
    allowlist: set[str] = Field(default_factory=set)
    blocklist: set[str] = Field(default_factory=set)
    session_open_minutes: int = 0  # minutes since regular open
    session_close_minutes: int = 390  # minutes until regular close (or remaining)

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
