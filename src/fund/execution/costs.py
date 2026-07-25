"""Shared cost model for sim and live evaluation (INV-3 / §9.4)."""

from __future__ import annotations

from decimal import Decimal

from fund.config import CostConfig
from fund.types import Side


def commission(qty: Decimal, cfg: CostConfig) -> Decimal:
    return abs(qty) * cfg.commission_per_share


def spread_cost_bps(is_liquid: bool, cfg: CostConfig) -> Decimal:
    bps = cfg.default_spread_bps_liquid if is_liquid else cfg.default_spread_bps_illiquid
    return Decimal(bps) / Decimal("2")  # pay half-spread


def slippage_bps(
    order_notional: Decimal,
    adv_notional: Decimal,
    cfg: CostConfig,
) -> Decimal:
    base = Decimal(cfg.slippage_base_bps)
    if adv_notional <= 0:
        return base
    size_term = Decimal(cfg.slippage_size_coeff) * (order_notional / adv_notional)
    return base + size_term * Decimal("10000") if False else base + (
        Decimal(cfg.slippage_size_coeff) * (order_notional / adv_notional)
    )
    # size_term already in fraction of ADV; convert to bps:
    # 10 * (order/adv) bps → size_coeff * (order/adv)  [coeff is already in bps units]


def apply_slippage_bps(price: Decimal, side: Side, bps: Decimal) -> Decimal:
    mult = bps / Decimal("10000")
    if side == Side.BUY:
        return price * (Decimal("1") + mult)
    return price * (Decimal("1") - mult)


def sell_fees(qty: Decimal, price: Decimal, cfg: CostConfig) -> Decimal:
    notional = abs(qty) * price
    sec = notional * cfg.sec_fee_bps / Decimal("10000")
    taf = abs(qty) * cfg.finra_taf_per_share
    return sec + taf


def total_fees(
    side: Side,
    qty: Decimal,
    price: Decimal,
    cfg: CostConfig,
) -> Decimal:
    fees = commission(qty, cfg)
    if side == Side.SELL:
        fees += sell_fees(qty, price, cfg)
    return fees
