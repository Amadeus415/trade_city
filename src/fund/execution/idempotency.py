"""Deterministic client_order_id — SQLite UNIQUE is the guard."""

from __future__ import annotations

import hashlib
from decimal import Decimal

from fund.types import Side


def client_order_id(decision_id: str, symbol: str, side: Side, qty: Decimal) -> str:
    payload = f"{decision_id}|{symbol}|{side}|{qty:.6f}"
    return hashlib.sha256(payload.encode()).hexdigest()[:32]
