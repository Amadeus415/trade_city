"""Robinhood Agentic MCP adapter (live equities only).

Auth tokens live outside the repo (keyring / ~/.config/fund/).
Always review_equity_order before place_equity_order.
Timeouts are UNKNOWN — reconcile before any retry.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

from fund.logging_setup import get_logger
from fund.types import (
    BrokerOrderStatus,
    OrderAck,
    OrderIntent,
    OrderReview,
    OrderStatus,
    PortfolioSnapshot,
    Position,
    Quote,
)

log = get_logger(__name__)

DEFAULT_MCP_URL = "https://agent.robinhood.com/mcp/trading"
TOKEN_PATH = Path.home() / ".config" / "fund" / "robinhood_token.json"


class BrokerCircuitOpen(Exception):
    pass


class RobinhoodMCPBroker:
    """Live adapter. Requires OAuth token provisioned by the operator."""

    def __init__(
        self,
        token_path: Path | str = TOKEN_PATH,
        base_url: str = DEFAULT_MCP_URL,
        timeout: float = 30.0,
        failure_threshold: int = 3,
    ) -> None:
        self.token_path = Path(token_path)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.failure_threshold = failure_threshold
        self._consecutive_failures = 0
        self._token = self._load_token()

    def _load_token(self) -> str | None:
        if not self.token_path.exists():
            return None
        data = json.loads(self.token_path.read_text())
        return data.get("access_token")

    def _headers(self) -> dict[str, str]:
        if not self._token:
            raise RuntimeError(
                f"No Robinhood token at {self.token_path}. "
                "Place OAuth access_token JSON there (mode 0600)."
            )
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

    def _call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if self._consecutive_failures >= self.failure_threshold:
            raise BrokerCircuitOpen(
                f"{self._consecutive_failures} consecutive broker failures"
            )
        payload = {
            "jsonrpc": "2.0",
            "id": int(time.time() * 1000),
            "method": "tools/call",
            "params": {"name": method, "arguments": params},
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    self.base_url,
                    headers=self._headers(),
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
            self._consecutive_failures = 0
            log.info("robinhood_mcp_call", method=method, ok=True)
            return data
        except httpx.TimeoutException:
            self._consecutive_failures += 1
            log.error("robinhood_timeout", method=method)
            raise
        except Exception:
            self._consecutive_failures += 1
            log.exception("robinhood_error", method=method)
            raise

    def get_portfolio(self) -> PortfolioSnapshot:
        data = self._call("get_account_portfolio", {})
        # Adapter mapping is intentionally defensive — RH schema may evolve
        result = data.get("result", data)
        cash = Decimal(str(result.get("cash", "0")))
        equity = Decimal(str(result.get("equity", cash)))
        positions = []
        for p in result.get("positions", []):
            positions.append(
                Position(
                    symbol=p["symbol"],
                    quantity=Decimal(str(p["quantity"])),
                    avg_cost=Decimal(str(p.get("avg_cost", p.get("average_cost", "0")))),
                    market_value=Decimal(str(p.get("market_value", "0"))),
                    opened_at=datetime.fromisoformat(
                        p.get("opened_at", datetime.utcnow().isoformat())
                    ).date(),
                )
            )
        return PortfolioSnapshot(
            as_of=datetime.utcnow(),
            cash=cash,
            equity=equity,
            positions=positions,
            peak_equity=equity,
        )

    def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        data = self._call("get_quotes", {"symbols": symbols})
        result = data.get("result", data)
        out: dict[str, Quote] = {}
        for q in result.get("quotes", []):
            out[q["symbol"]] = Quote(
                symbol=q["symbol"],
                bid=Decimal(str(q.get("bid", q.get("last", 0)))),
                ask=Decimal(str(q.get("ask", q.get("last", 0)))),
                last=Decimal(str(q.get("last", 0))),
                as_of=datetime.utcnow(),
            )
        return out

    def review_order(self, intent: OrderIntent) -> OrderReview:
        data = self._call(
            "review_equity_order",
            {
                "symbol": intent.symbol,
                "side": intent.side.value,
                "quantity": str(intent.quantity),
                "limit_price": str(intent.limit_price),
                "client_order_id": intent.client_order_id,
                "time_in_force": intent.time_in_force,
            },
        )
        result = data.get("result", data)
        warnings = list(result.get("warnings", []))
        # Any warning = rejection unless allowlisted
        benign = {"market_hours_info"}
        bad = [w for w in warnings if w not in benign]
        return OrderReview(
            client_order_id=intent.client_order_id,
            approved=result.get("approved", len(bad) == 0),
            warnings=warnings,
            estimated_cost=Decimal(str(result["estimated_cost"]))
            if result.get("estimated_cost")
            else None,
            raw=result if isinstance(result, dict) else {"raw": result},
        )

    def place_order(self, intent: OrderIntent) -> OrderAck:
        # INV: always review first
        review = self.review_order(intent)
        if not review.approved:
            return OrderAck(
                client_order_id=intent.client_order_id,
                broker_order_id="",
                status=OrderStatus.REJECTED,
                raw={"review": review.raw, "warnings": review.warnings},
            )
        data = self._call(
            "place_equity_order",
            {
                "symbol": intent.symbol,
                "side": intent.side.value,
                "quantity": str(intent.quantity),
                "limit_price": str(intent.limit_price),
                "client_order_id": intent.client_order_id,
                "time_in_force": intent.time_in_force,
            },
        )
        result = data.get("result", data)
        return OrderAck(
            client_order_id=intent.client_order_id,
            broker_order_id=str(result.get("order_id", result.get("broker_order_id", ""))),
            status=OrderStatus.ACKNOWLEDGED,
            raw=result if isinstance(result, dict) else {"raw": result},
        )

    def cancel_order(self, broker_order_id: str) -> None:
        self._call("cancel_order", {"order_id": broker_order_id})

    def list_orders(self, since: datetime) -> list[BrokerOrderStatus]:
        data = self._call("list_orders", {"since": since.isoformat()})
        result = data.get("result", data)
        out: list[BrokerOrderStatus] = []
        for o in result.get("orders", []):
            out.append(
                BrokerOrderStatus(
                    client_order_id=o.get("client_order_id", ""),
                    broker_order_id=o.get("order_id"),
                    status=OrderStatus(o.get("status", "acknowledged")),
                    filled_qty=Decimal(str(o.get("filled_qty", "0"))),
                    avg_fill_price=Decimal(str(o["avg_fill_price"]))
                    if o.get("avg_fill_price")
                    else None,
                    raw=o,
                )
            )
        return out
