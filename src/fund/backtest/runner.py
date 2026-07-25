"""Event-loop backtester. Orders execute next session open (never same-close)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from fund.agent.cache import LLMCache
from fund.agent.pm import PortfolioManager
from fund.clock import Clock, trading_days
from fund.config import Settings
from fund.execution.idempotency import client_order_id
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
    Fill,
    MarketContext,
    OrderIntent,
    RiskStateName,
    RiskVerdict,
    Side,
)

log = get_logger(__name__)
ET = ZoneInfo("America/New_York")


@dataclass
class PendingOrder:
    intent: OrderIntent
    decision_id: str


@dataclass
class BacktestResult:
    run_id: str
    equity_curve: pd.Series
    decisions: int = 0
    orders: int = 0
    fills: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


class BacktestRunner:
    def __init__(
        self,
        settings: Settings,
        bar_store: BarStore,
        universe: UniverseStore,
        journal: Journal,
        *,
        use_agent: bool = True,
        strategy: str = "agent",  # agent | momentum | equal_weight
    ) -> None:
        self.settings = settings
        self.bars = bar_store
        self.universe = universe
        self.journal = journal
        self.use_agent = use_agent
        self.strategy = strategy
        self.limits = RiskLimits(settings.risk)
        self.risk = RiskEngine(self.limits)
        self.kill = KillSwitch(journal)
        cache = None
        if settings.agent.cache_enabled:
            cache = LLMCache(Path(settings.data_dir) / "llm_cache" / "cache.db")
        self.pm = PortfolioManager(settings.agent, cache=cache) if use_agent else None

    def run(self, start: date, end: date) -> BacktestResult:
        sessions = trading_days(start, end)
        if not sessions:
            raise ValueError(f"no trading days in {start}..{end}")

        # Preload all bars once — backtests re-query every session
        try:
            syms = self.universe.static_allowlist()
            if "SPY" not in syms:
                syms = list(syms) + ["SPY"]
            n = self.bars.preload(syms)
            log.info("bars_preloaded", rows=n, symbols=len(syms))
        except Exception as e:
            log.warning("bars_preload_failed", error=str(e))

        run_id = self.journal.start_run(
            mode="backtest",
            config_hash=self.settings.config_hash(),
            limits_version=self.settings.risk.limits_version,
        )
        # Backtests own risk state for the run (live/paper persist across restarts).
        self.journal.transition_risk_state(
            RiskStateName.NORMAL,
            trigger_metric="backtest_start",
            trigger_value=run_id,
            operator_note="reset for isolated backtest",
            consecutive_loss_days=0,
        )
        cash = Decimal(str(self.settings.backtest.start_cash))
        clock = Clock(datetime.combine(sessions[0], time(16, 15), tzinfo=ET))
        broker = SimulatedBroker(cash, self.settings.costs, clock=clock)
        pending: list[PendingOrder] = []
        equity_points: list[tuple[date, float]] = []
        n_decisions = n_orders = n_fills = 0
        prev_equity = cash
        week_start_equity = cash

        lag = self.settings.availability.bar_lag_minutes
        lookback = 300

        for i, session in enumerate(sessions):
            as_of = datetime.combine(session, time(16, 0), tzinfo=ET) + timedelta(
                minutes=lag
            )
            clock.set_fixed(as_of)

            # Resolve prices for session
            symbols = self.universe.get_universe(as_of)
            if "SPY" not in symbols:
                symbols = symbols + ["SPY"]
            bar_df = self.bars.get_bars(symbols, as_of=as_of, lookback=lookback)
            if bar_df.empty:
                continue

            # Today's bars for fill simulation
            today = bar_df[bar_df["session"] == session]
            prices: dict[str, Decimal] = {}
            bars_ohlc: dict[str, dict] = {}
            for _, row in today.iterrows():
                sym = str(row["symbol"])
                prices[sym] = Decimal(str(row["close"]))
                bars_ohlc[sym] = {
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                }
            # ADV approx
            adv: dict[str, Decimal] = {}
            for sym in symbols:
                sb = bar_df[bar_df["symbol"] == sym].tail(21)
                if not sb.empty:
                    adv[sym] = Decimal(
                        str((sb["close"].astype(float) * sb["volume"].astype(float)).mean())
                    )
            broker.set_market(prices, bars_ohlc, adv)

            # Execute pending orders from prior decision (next open)
            still_pending: list[PendingOrder] = []
            for po in pending:
                bar = bars_ohlc.get(po.intent.symbol)
                if bar is None:
                    still_pending.append(po)
                    continue
                filled, px, fees = broker.try_fill(po.intent, bar)
                n_orders += 1
                if filled:
                    n_fills += 1
                    self.journal.update_order(
                        po.intent.client_order_id,
                        status=__import__("fund.types", fromlist=["OrderStatus"]).OrderStatus.FILLED,
                    )
                    self.journal.record_fill(
                        Fill(
                            fill_id=f"f-{po.intent.client_order_id[:12]}",
                            client_order_id=po.intent.client_order_id,
                            quantity=po.intent.quantity,
                            price=px,
                            fees=fees,
                            filled_at=as_of,
                        )
                    )
            pending = still_pending

            portfolio = broker.get_portfolio()
            equity_points.append((session, float(portfolio.equity)))
            self.journal.snapshot_equity(
                run_id,
                session.isoformat(),
                portfolio.cash,
                portfolio.equity,
                portfolio.peak_equity,
                [p.model_dump(mode="json") for p in portfolio.positions],
            )

            # Risk state from daily PnL
            if prev_equity > 0:
                daily_pnl = (portfolio.equity - prev_equity) / prev_equity
            else:
                daily_pnl = Decimal("0")
            # week approx: Monday reset
            if session.weekday() == 0:
                week_start_equity = prev_equity
            weekly_pnl = (
                (portfolio.equity - week_start_equity) / week_start_equity
                if week_start_equity > 0
                else Decimal("0")
            )
            dd = (
                (portfolio.peak_equity - portfolio.equity) / portfolio.peak_equity
                if portfolio.peak_equity > 0
                else Decimal("0")
            )
            ll = self.settings.risk.loss_limits
            self.kill.evaluate_pnl(
                daily_pnl_pct=daily_pnl,
                weekly_pnl_pct=weekly_pnl,
                drawdown_pct=dd,
                equity_recovered=daily_pnl > 0 and self.kill.current() == RiskStateName.REDUCED,
                soft_daily=ll.daily_loss_soft_pct,
                hard_daily=ll.daily_loss_hard_pct,
                hard_weekly=ll.weekly_loss_hard_pct,
                max_dd=ll.max_drawdown_halt_pct,
                consecutive_halt=ll.consecutive_loss_days_halt,
            )
            prev_equity = portfolio.equity
            state = self.kill.current()

            # Features + decision
            sectors = self.universe.sector_map()
            feats, clusters = compute_all_features(
                bar_df,
                [s for s in symbols if s in bar_df["symbol"].values],
                sectors=sectors,
                correlation_threshold=float(
                    self.settings.risk.concentration.correlation_threshold
                ),
            )
            constraints = {
                "max_position": str(self.settings.risk.position.max_position_weight),
                "max_cluster": str(self.settings.risk.concentration.max_cluster_weight),
                "max_gross": str(self.settings.risk.account.max_gross_exposure),
                "min_hold_days": self.settings.risk.turnover.min_holding_days,
                "max_orders": self.settings.risk.turnover.max_orders_per_day,
            }
            packet = build_feature_table(
                feats,
                portfolio,
                as_of=as_of,
                sectors=sectors,
                clusters=clusters,
                constraints=constraints,
                risk_state=state.value,
            )
            phash = feature_packet_hash(packet)

            if self.strategy == "momentum":
                proposals = self._momentum_proposals(feats, portfolio)
                meta = {
                    "prompt_version": "baseline:momentum",
                    "model": "none",
                    "raw": "[]",
                    "masking_mode": "bright",
                }
            elif self.strategy == "equal_weight":
                proposals = self._ew_proposals(list(feats.keys()), portfolio)
                meta = {
                    "prompt_version": "baseline:equal_weight",
                    "model": "none",
                    "raw": "[]",
                    "masking_mode": "bright",
                }
            else:
                assert self.pm is not None
                proposals, meta = self.pm.run(packet)

            n_decisions += 1
            decision_id = self.journal.record_decision(
                run_id=run_id,
                as_of=as_of,
                session=session.isoformat(),
                prompt_version=meta.get("prompt_version", ""),
                model=meta.get("model", ""),
                temperature=self.settings.agent.temperature,
                masking_mode=meta.get("masking_mode", self.settings.agent.masking_mode),
                feature_packet_hash=phash,
                feature_packet_json=packet_to_json(feats, clusters),
                raw_llm_response=meta.get("raw", ""),
                latency_ms=meta.get("latency_ms"),
                cost_usd=meta.get("cost_usd"),
            )
            pids = self.journal.record_proposals(decision_id, proposals)

            market = MarketContext(
                as_of=as_of,
                last_prices=prices,
                spreads_bps={s: Decimal("5") for s in prices},
                adv_notional=adv,
                sectors=sectors,
                clusters=clusters,
                allowlist=set(self.universe.static_allowlist()),
                blocklist=set(self.settings.risk.universe.blocklist),
                session_open_minutes=60,  # decision after close; execute next day open+15
                session_close_minutes=300,
                days_listed={s: 500 for s in symbols},
            )
            decisions = self.risk.evaluate(proposals, portfolio, market, state)
            if pids and decisions:
                # Align lengths: risk may have same count as proposals
                n = min(len(pids), len(decisions))
                self.journal.record_risk_decisions(pids[:n], decisions[:n])

            # Queue orders for next session
            for rd in decisions:
                if rd.verdict == RiskVerdict.REJECTED:
                    continue
                if rd.proposal.action in (Action.HOLD, Action.ABSTAIN):
                    continue
                sym = rd.proposal.symbol
                px = prices.get(sym)
                if px is None or px <= 0:
                    continue
                pos_map = portfolio.position_map()
                cur_qty = pos_map[sym].quantity if sym in pos_map else Decimal("0")
                target_w = rd.final_weight
                if rd.proposal.action == Action.CLOSE:
                    target_w = Decimal("0")
                delta = shares_for_weight(target_w, portfolio.equity, px, cur_qty)
                if delta == 0:
                    continue
                side = Side.BUY if delta > 0 else Side.SELL
                qty = abs(delta)
                # Limit offset
                offset = Decimal(self.settings.risk.order.max_limit_offset_bps) / Decimal(
                    "10000"
                )
                limit = px * (Decimal("1") + offset) if side == Side.BUY else px * (
                    Decimal("1") - offset
                )
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
                if self.journal.insert_order_intent(intent):
                    ack = broker.place_order(intent)
                    self.journal.update_order(
                        coid,
                        status=__import__(
                            "fund.types", fromlist=["OrderStatus"]
                        ).OrderStatus.ACKNOWLEDGED,
                        broker_order_id=ack.broker_order_id,
                    )
                    pending.append(PendingOrder(intent=intent, decision_id=decision_id))

            if (i + 1) % 50 == 0:
                log.info(
                    "backtest_progress",
                    session=str(session),
                    equity=str(portfolio.equity),
                    decisions=n_decisions,
                )

        curve = pd.Series(
            {d: e for d, e in equity_points},
            name="agent",
        )
        return BacktestResult(
            run_id=run_id,
            equity_curve=curve,
            decisions=n_decisions,
            orders=n_orders,
            fills=n_fills,
        )

    def _momentum_proposals(self, feats, portfolio):
        from fund.types import Proposal

        ranked = sorted(
            (
                (s, f.get("mom_63d"))
                for s, f in feats.items()
                if f.get("mom_63d") is not None and s != "SPY"
            ),
            key=lambda x: x[1] or -999,
            reverse=True,
        )[:5]
        proposals = []
        for s, m in ranked:
            proposals.append(
                Proposal(
                    symbol=s,
                    action=Action.OPEN if portfolio.weight(s) == 0 else Action.ADD,
                    target_weight=Decimal("0.08"),
                    confidence=Decimal("0.50"),
                    thesis=f"Momentum baseline: mom_63d={m:.3f}",
                    invalidation="mom_63d rank falls out of top half",
                    horizon_days=21,
                    source_features=["mom_63d"],
                )
            )
        return proposals

    def _ew_proposals(self, symbols, portfolio):
        from fund.types import Proposal

        syms = [s for s in symbols if s != "SPY"][:10]
        if not syms:
            return []
        w = (Decimal("0.80") / len(syms)).quantize(Decimal("0.0001"))
        return [
            Proposal(
                symbol=s,
                action=Action.OPEN if portfolio.weight(s) == 0 else Action.HOLD,
                target_weight=w,
                confidence=Decimal("0.30"),
                thesis="Equal weight baseline",
                invalidation="n/a",
                horizon_days=21,
                source_features=[],
            )
            for s in syms
        ]
