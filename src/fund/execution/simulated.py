"""Simulated broker for backtest + paper. Same journal shape as live."""

from __future__ import annotations

import random
import uuid
from datetime import date, datetime
from decimal import Decimal

from fund.clock import Clock, ET
from fund.config import CostConfig
from fund.execution.costs import apply_slippage_bps, slippage_bps, total_fees
from fund.types import (
    BrokerOrderStatus,
    OrderAck,
    OrderIntent,
    OrderReview,
    OrderStatus,
    PortfolioSnapshot,
    Position,
    Quote,
    Side,
)


class SimulatedBroker:
    def __init__(
        self,
        cash: Decimal,
        cost_cfg: CostConfig,
        clock: Clock | None = None,
        seed: int = 42,
        nonfill_rate: float | None = None,
    ) -> None:
        self.cash = cash
        self.cost_cfg = cost_cfg
        self.clock = clock or Clock()
        self.positions: dict[str, Position] = {}
        self.peak_equity = cash
        self._orders: dict[str, BrokerOrderStatus] = {}
        self._prices: dict[str, Decimal] = {}
        self._bars_today: dict[str, dict] = {}  # symbol -> ohlc
        self._rng = random.Random(seed)
        self.nonfill_rate = (
            nonfill_rate if nonfill_rate is not None else cost_cfg.limit_nonfill_rate
        )
        self.adv: dict[str, Decimal] = {}

    def set_market(
        self,
        prices: dict[str, Decimal],
        bars: dict[str, dict] | None = None,
        adv: dict[str, Decimal] | None = None,
    ) -> None:
        self._prices = dict(prices)
        if bars:
            self._bars_today = bars
        if adv:
            self.adv = adv
        self._mark()

    def _mark(self) -> None:
        for sym, pos in list(self.positions.items()):
            px = self._prices.get(sym)
            if px is not None:
                self.positions[sym] = Position(
                    symbol=sym,
                    quantity=pos.quantity,
                    avg_cost=pos.avg_cost,
                    market_value=pos.quantity * px,
                    opened_at=pos.opened_at,
                    sector=pos.sector,
                    cluster_id=pos.cluster_id,
                )

    def equity(self) -> Decimal:
        mv = sum((p.market_value for p in self.positions.values()), Decimal("0"))
        return self.cash + mv

    def get_portfolio(self) -> PortfolioSnapshot:
        self._mark()
        eq = self.equity()
        if eq > self.peak_equity:
            self.peak_equity = eq
        return PortfolioSnapshot(
            as_of=self.clock.now(),
            cash=self.cash,
            equity=eq,
            positions=list(self.positions.values()),
            peak_equity=self.peak_equity,
        )

    def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        out: dict[str, Quote] = {}
        for s in symbols:
            px = self._prices.get(s, Decimal("0"))
            spread = px * Decimal("0.0005")
            out[s] = Quote(
                symbol=s,
                bid=px - spread / 2,
                ask=px + spread / 2,
                last=px,
                as_of=self.clock.now(),
            )
        return out

    def review_order(self, intent: OrderIntent) -> OrderReview:
        return OrderReview(
            client_order_id=intent.client_order_id,
            approved=True,
            warnings=[],
            estimated_cost=intent.quantity * intent.limit_price,
            raw={"sim": True},
        )

    def place_order(self, intent: OrderIntent) -> OrderAck:
        broker_id = f"sim-{uuid.uuid4().hex[:12]}"
        status = BrokerOrderStatus(
            client_order_id=intent.client_order_id,
            broker_order_id=broker_id,
            status=OrderStatus.ACKNOWLEDGED,
            filled_qty=Decimal("0"),
        )
        self._orders[intent.client_order_id] = status
        return OrderAck(
            client_order_id=intent.client_order_id,
            broker_order_id=broker_id,
            status=OrderStatus.ACKNOWLEDGED,
            raw={"sim": True},
        )

    def cancel_order(self, broker_order_id: str) -> None:
        for coid, st in self._orders.items():
            if st.broker_order_id == broker_order_id:
                st.status = OrderStatus.CANCELLED
                self._orders[coid] = st

    def list_orders(self, since: datetime) -> list[BrokerOrderStatus]:
        return list(self._orders.values())

    def try_fill(
        self,
        intent: OrderIntent,
        session_bar: dict,
    ) -> tuple[bool, Decimal, Decimal]:
        """Attempt limit fill against session OHLC. Returns (filled, price, fees).

        Buy fills if day's low <= limit; sell if day's high >= limit.
        Models non-fill rate at the touch.
        """
        high = Decimal(str(session_bar["high"]))
        low = Decimal(str(session_bar["low"]))
        open_px = Decimal(str(session_bar["open"]))

        if intent.side == Side.BUY:
            if low > intent.limit_price:
                return False, Decimal("0"), Decimal("0")
        else:
            if high < intent.limit_price:
                return False, Decimal("0"), Decimal("0")

        # Non-fill rate even when price crosses
        if self._rng.random() < self.nonfill_rate:
            return False, Decimal("0"), Decimal("0")

        # Fill near open with slippage
        is_liquid = self.adv.get(intent.symbol, Decimal("0")) >= Decimal("5000000")
        order_notional = abs(intent.quantity) * intent.limit_price
        adv = self.adv.get(intent.symbol, Decimal("10000000"))
        slip = slippage_bps(order_notional, adv, self.cost_cfg)
        # half-spread + slip
        half_spread = Decimal(
            self.cost_cfg.default_spread_bps_liquid
            if is_liquid
            else self.cost_cfg.default_spread_bps_illiquid
        ) / Decimal("2")
        total_bps = slip + half_spread
        fill_px = apply_slippage_bps(open_px, intent.side, total_bps)
        # Cap at limit for buys / floor for sells
        if intent.side == Side.BUY:
            fill_px = min(fill_px, intent.limit_price)
        else:
            fill_px = max(fill_px, intent.limit_price)

        fees = total_fees(intent.side, intent.quantity, fill_px, self.cost_cfg)
        self._apply_fill(intent, fill_px, fees)
        st = self._orders.get(intent.client_order_id)
        if st:
            st.status = OrderStatus.FILLED
            st.filled_qty = intent.quantity
            st.avg_fill_price = fill_px
        return True, fill_px, fees

    def _apply_fill(self, intent: OrderIntent, price: Decimal, fees: Decimal) -> None:
        qty = intent.quantity
        if intent.side == Side.BUY:
            cost = qty * price + fees
            self.cash -= cost
            if intent.symbol in self.positions:
                pos = self.positions[intent.symbol]
                new_qty = pos.quantity + qty
                new_avg = (
                    (pos.avg_cost * pos.quantity + price * qty) / new_qty
                    if new_qty
                    else price
                )
                self.positions[intent.symbol] = Position(
                    symbol=intent.symbol,
                    quantity=new_qty,
                    avg_cost=new_avg,
                    market_value=new_qty * price,
                    opened_at=pos.opened_at,
                )
            else:
                self.positions[intent.symbol] = Position(
                    symbol=intent.symbol,
                    quantity=qty,
                    avg_cost=price,
                    market_value=qty * price,
                    opened_at=self.clock.today(),
                )
        else:
            proceeds = qty * price - fees
            self.cash += proceeds
            if intent.symbol in self.positions:
                pos = self.positions[intent.symbol]
                new_qty = pos.quantity - qty
                if new_qty <= 0:
                    del self.positions[intent.symbol]
                else:
                    self.positions[intent.symbol] = Position(
                        symbol=intent.symbol,
                        quantity=new_qty,
                        avg_cost=pos.avg_cost,
                        market_value=new_qty * price,
                        opened_at=pos.opened_at,
                    )
        self._prices[intent.symbol] = price
        self._mark()
