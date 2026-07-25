"""Daily decision and execution cycles — separate commands (INV)."""

from __future__ import annotations

from datetime import datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from fund.agent.cache import LLMCache
from fund.agent.pm import PortfolioManager
from fund.clock import Clock, is_half_day, is_trading_day
from fund.config import Settings, load_universe
from fund.execution.idempotency import client_order_id
from fund.execution.reconcile import reconcile
from fund.execution.simulated import SimulatedBroker
from fund.features.packet import (
    build_feature_table,
    compute_all_features,
    feature_packet_hash,
    packet_to_json,
)
from fund.logging_setup import get_logger
from fund.risk.engine import RiskEngine
from fund.risk.limits import RiskLimits
from fund.risk.sizing import shares_for_weight
from fund.risk.state import KillSwitch
from fund.store.bars import BarStore
from fund.store.journal import Journal
from fund.store.universe import UniverseStore
from fund.types import (
    Action,
    MarketContext,
    OrderIntent,
    OrderStatus,
    RiskVerdict,
    Side,
)

log = get_logger(__name__)
ET = ZoneInfo("America/New_York")


def _broker_for_mode(settings: Settings, journal: Journal):
    if settings.mode == "live":
        from fund.execution.robinhood_mcp import RobinhoodMCPBroker

        return RobinhoodMCPBroker()
    # paper / backtest use simulated — paper needs cash from journal or settings
    cash = Decimal(str(settings.initial_cash))
    return SimulatedBroker(cash, settings.costs)


def decide(
    settings: Settings,
    *,
    as_of: datetime | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    clock = Clock(as_of)
    now = clock.now()
    if not is_trading_day(now.date()) or is_half_day(now.date()):
        log.info("noop_non_trading_or_half_day", date=str(now.date()))
        return {"status": "noop", "reason": "non_trading_or_half_day"}

    journal = Journal(settings.journal_path, clock=clock)
    journal.migrate()
    bars = BarStore(settings.data_dir)
    universe = UniverseStore(settings.data_dir)
    broker = _broker_for_mode(settings, journal)
    kill = KillSwitch(journal)
    state = kill.current()

    # Market data as_of
    if as_of is None:
        as_of = now
    symbols = universe.get_universe(as_of)
    bar_df = bars.get_bars(symbols + (["SPY"] if "SPY" not in symbols else []), as_of=as_of, lookback=300)
    portfolio = broker.get_portfolio()

    # Seed sim prices from bars
    if isinstance(broker, SimulatedBroker) and not bar_df.empty:
        last = bar_df.sort_values("session").groupby("symbol").tail(1)
        prices = {
            str(r["symbol"]): Decimal(str(r["close"])) for _, r in last.iterrows()
        }
        broker.set_market(prices)

    sectors = universe.sector_map()
    feats, clusters = compute_all_features(
        bar_df,
        [s for s in symbols if not bar_df.empty and s in set(bar_df["symbol"])],
        sectors=sectors,
    )
    packet = build_feature_table(
        feats,
        portfolio,
        as_of=as_of,
        sectors=sectors,
        clusters=clusters,
        constraints={
            "max_position": str(settings.risk.position.max_position_weight),
            "max_cluster": str(settings.risk.concentration.max_cluster_weight),
            "max_gross": str(settings.risk.account.max_gross_exposure),
            "max_orders": settings.risk.turnover.max_orders_per_day,
        },
        risk_state=state.value,
    )

    cache = (
        LLMCache(Path(settings.data_dir) / "llm_cache" / "cache.db")
        if settings.agent.cache_enabled
        else None
    )
    pm = PortfolioManager(settings.agent, cache=cache)
    proposals, meta = pm.run(packet)

    run_id = journal.start_run(
        mode=settings.mode,
        config_hash=settings.config_hash(),
        limits_version=settings.risk.limits_version,
    )
    decision_id = journal.record_decision(
        run_id=run_id,
        as_of=as_of,
        session=as_of.date().isoformat(),
        prompt_version=meta.get("prompt_version", ""),
        model=meta.get("model", ""),
        temperature=settings.agent.temperature,
        masking_mode=meta.get("masking_mode", settings.agent.masking_mode),
        feature_packet_hash=feature_packet_hash(packet),
        feature_packet_json=packet_to_json(feats, clusters),
        raw_llm_response=meta.get("raw", ""),
        latency_ms=meta.get("latency_ms"),
        cost_usd=meta.get("cost_usd"),
    )
    pids = journal.record_proposals(decision_id, proposals)

    last_prices = {}
    if not bar_df.empty:
        last = bar_df.sort_values("session").groupby("symbol").tail(1)
        last_prices = {
            str(r["symbol"]): Decimal(str(r["close"])) for _, r in last.iterrows()
        }

    market = MarketContext(
        as_of=as_of,
        last_prices=last_prices,
        spreads_bps={s: Decimal("5") for s in last_prices},
        adv_notional={s: Decimal("10000000") for s in last_prices},
        sectors=sectors,
        clusters=clusters,
        allowlist=set(universe.static_allowlist()),
        blocklist=set(settings.risk.universe.blocklist),
        session_open_minutes=60,
        session_close_minutes=300,
        days_listed={s: 500 for s in symbols},
    )
    engine = RiskEngine(RiskLimits(settings.risk))
    decisions = engine.evaluate(proposals, portfolio, market, state)
    if pids:
        n = min(len(pids), len(decisions))
        journal.record_risk_decisions(pids[:n], decisions[:n])

    # Queue intents only — execute is a separate command
    queued = 0
    for rd in decisions:
        if rd.verdict == RiskVerdict.REJECTED:
            continue
        if rd.proposal.action in (Action.HOLD, Action.ABSTAIN):
            continue
        sym = rd.proposal.symbol
        px = last_prices.get(sym)
        if not px:
            continue
        pos_map = portfolio.position_map()
        cur = pos_map[sym].quantity if sym in pos_map else Decimal("0")
        tw = Decimal("0") if rd.proposal.action == Action.CLOSE else rd.final_weight
        delta = shares_for_weight(tw, portfolio.equity, px, cur)
        if delta == 0:
            continue
        side = Side.BUY if delta > 0 else Side.SELL
        qty = abs(delta)
        offset = Decimal(settings.risk.order.max_limit_offset_bps) / Decimal("10000")
        limit = px * (1 + offset) if side == Side.BUY else px * (1 - offset)
        coid = client_order_id(decision_id, sym, side, qty)
        intent = OrderIntent(
            client_order_id=coid,
            decision_id=decision_id,
            symbol=sym,
            side=side,
            quantity=qty,
            limit_price=limit.quantize(Decimal("0.01")),
            created_at=as_of,
        )
        if not dry_run:
            journal.insert_order_intent(intent)
        queued += 1
        log.info("queued_order", **intent.model_dump(mode="json"))

    journal.close()
    return {
        "status": "ok",
        "run_id": run_id,
        "decision_id": decision_id,
        "proposals": len(proposals),
        "queued": queued,
        "risk_state": state.value,
    }


def execute(
    settings: Settings,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    clock = Clock()
    now = clock.now()
    if not is_trading_day(now.date()):
        return {"status": "noop", "reason": "non_trading_day"}

    journal = Journal(settings.journal_path, clock=clock)
    journal.migrate()
    broker = _broker_for_mode(settings, journal)

    recon = reconcile(journal, broker, since=now - timedelta(days=5))
    open_orders = [
        o for o in journal.get_open_orders() if o["status"] == OrderStatus.INTENT.value
    ]

    rejected_exist = False
    # dry-run exits non-zero if any risk decision was REJECTED recently — checked by CLI
    sent = 0
    payloads: list[dict] = []
    for row in open_orders:
        intent = OrderIntent(
            client_order_id=row["client_order_id"],
            decision_id=row["decision_id"],
            symbol=row["symbol"],
            side=Side(row["side"]),
            quantity=Decimal(row["quantity"]),
            limit_price=Decimal(row["limit_price"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )
        payloads.append(intent.model_dump(mode="json"))
        if dry_run:
            log.info("dry_run_would_send", **payloads[-1])
            continue
        try:
            review = broker.review_order(intent)
            journal.update_order(
                intent.client_order_id,
                review_response=str(review.raw),
            )
            if not review.approved:
                journal.update_order(
                    intent.client_order_id,
                    status=OrderStatus.REJECTED,
                    error=";".join(review.warnings),
                )
                continue
            ack = broker.place_order(intent)
            journal.update_order(
                intent.client_order_id,
                status=ack.status,
                broker_order_id=ack.broker_order_id,
            )
            sent += 1
        except Exception as e:
            # Timeout = unknown — do not retry without reconcile
            journal.update_order(
                intent.client_order_id,
                error=f"unknown:{e}",
            )
            log.exception("execute_error", client_order_id=intent.client_order_id)
            break

    journal.close()
    return {
        "status": "ok",
        "dry_run": dry_run,
        "reconcile": recon,
        "sent": sent,
        "payloads": payloads if dry_run else [],
    }
