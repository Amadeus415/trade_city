"""Startup / cycle state repair for orders stuck in intent or acknowledged."""

from __future__ import annotations

from datetime import datetime

from fund.execution.broker import BrokerAdapter
from fund.logging_setup import get_logger
from fund.store.journal import Journal
from fund.types import OrderStatus, RiskStateName

log = get_logger(__name__)


def reconcile(
    journal: Journal,
    broker: BrokerAdapter,
    *,
    since: datetime,
    escalate_stuck: bool = True,
) -> dict:
    """Resolve open journal orders against broker state.

    Stuck `intent` for more than one cycle → REDUCED + alert.
    """
    open_orders = journal.get_open_orders()
    broker_orders = {o.client_order_id: o for o in broker.list_orders(since)}
    resolved = 0
    stuck = 0

    for row in open_orders:
        coid = row["client_order_id"]
        status = row["status"]
        bo = broker_orders.get(coid)
        if bo is None:
            if status == OrderStatus.INTENT.value:
                stuck += 1
                log.warning("order_stuck_intent", client_order_id=coid)
            continue
        if bo.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.FAILED,
            OrderStatus.REJECTED,
        ):
            journal.update_order(
                coid,
                status=bo.status,
                broker_order_id=bo.broker_order_id,
            )
            resolved += 1
        elif bo.status == OrderStatus.ACKNOWLEDGED:
            journal.update_order(
                coid,
                status=OrderStatus.ACKNOWLEDGED,
                broker_order_id=bo.broker_order_id,
            )

    if stuck and escalate_stuck:
        journal.transition_risk_state(
            RiskStateName.REDUCED,
            trigger_metric="stuck_intent_orders",
            trigger_value=str(stuck),
        )
        log.error("escalated_reduced_stuck_orders", count=stuck)

    return {"open": len(open_orders), "resolved": resolved, "stuck": stuck}
