"""Weight → whole shares. Always round down."""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal


def shares_for_weight(
    target_weight: Decimal,
    equity: Decimal,
    price: Decimal,
    current_qty: Decimal = Decimal("0"),
) -> Decimal:
    """Return whole-share delta (positive = buy, negative = sell) to reach target_weight."""
    if price <= 0 or equity <= 0:
        return Decimal("0")
    target_notional = target_weight * equity
    target_qty = (target_notional / price).to_integral_value(rounding=ROUND_DOWN)
    delta = target_qty - current_qty
    return delta


def notional(qty: Decimal, price: Decimal) -> Decimal:
    return abs(qty) * price
