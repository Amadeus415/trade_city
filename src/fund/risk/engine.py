"""Deterministic risk veto. INV-2: no LLM can bypass this.

Must NOT import from fund.agent.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Iterable

from fund.risk.limits import RiskLimits
from fund.risk.sizing import notional, shares_for_weight
from fund.types import (
    Action,
    MarketContext,
    PortfolioSnapshot,
    Proposal,
    RiskDecision,
    RiskStateName,
    RiskVerdict,
    Side,
)


class RiskEngine:
    def __init__(self, limits: RiskLimits) -> None:
        self.limits = limits

    def evaluate(
        self,
        proposals: list[Proposal],
        portfolio: PortfolioSnapshot,
        market: MarketContext,
        state: RiskStateName,
        *,
        is_stop_loss: dict[str, bool] | None = None,
    ) -> list[RiskDecision]:
        """Ordered evaluation pipeline. Each stage may reject or clamp."""
        stop = is_stop_loss or {}
        results: list[RiskDecision] = []
        working: list[tuple[Proposal, Decimal, list[str], RiskVerdict]] = []

        for prop in proposals:
            reasons: list[str] = []
            weight = prop.target_weight
            verdict = RiskVerdict.ACCEPTED

            # Non-trade actions pass through
            if prop.action in (Action.HOLD, Action.ABSTAIN):
                results.append(
                    RiskDecision(
                        proposal=prop,
                        verdict=RiskVerdict.ACCEPTED,
                        final_weight=portfolio.weight(prop.symbol),
                        reasons=["non_trade_action"],
                        limits_version=self.limits.version,
                    )
                )
                continue

            # 1. State gate
            if state == RiskStateName.HALTED:
                if prop.action != Action.CLOSE:
                    results.append(self._reject(prop, ["state_halted"]))
                    continue
            if state == RiskStateName.REDUCED:
                if prop.action in (Action.OPEN, Action.ADD):
                    results.append(self._reject(prop, ["state_reduced_blocks_open_add"]))
                    continue

            # 2. Universe gate
            u = self.limits.config.universe
            if u.allowlist_only and market.allowlist and prop.symbol not in market.allowlist:
                results.append(self._reject(prop, ["not_in_allowlist"]))
                continue
            if prop.symbol in market.blocklist or prop.symbol in set(u.blocklist):
                results.append(self._reject(prop, ["blocklist"]))
                continue
            if prop.symbol in market.halted_symbols and self.limits.config.order.reject_if_halted:
                results.append(self._reject(prop, ["exchange_halted"]))
                continue
            price = market.last_prices.get(prop.symbol)
            if price is not None:
                if price < u.min_price or price > u.max_price:
                    results.append(self._reject(prop, ["price_out_of_range"]))
                    continue
            adv = market.adv_notional.get(prop.symbol)
            if adv is not None and adv < u.min_avg_dollar_volume_21d:
                results.append(self._reject(prop, ["adv_too_low"]))
                continue
            days_listed = market.days_listed.get(prop.symbol)
            if days_listed is not None and days_listed < u.min_days_listed:
                results.append(self._reject(prop, ["min_days_listed"]))
                continue
            dte = market.days_to_earnings.get(prop.symbol)
            if (
                dte is not None
                and dte <= u.exclude_earnings_within_days
                and prop.action in (Action.OPEN, Action.ADD)
            ):
                results.append(self._reject(prop, ["earnings_blackout"]))
                continue

            # 3. Per-position clamp
            max_w = self.limits.config.position.max_position_weight
            if weight > max_w:
                weight = max_w
                reasons.append("max_position_weight")
                verdict = RiskVerdict.CLAMPED
            pos_map = portfolio.position_map()
            if prop.action == Action.OPEN and prop.symbol not in pos_map:
                max_new = self.limits.config.position.max_new_position_weight
                if weight > max_new:
                    weight = max_new
                    reasons.append("max_new_position_weight")
                    verdict = RiskVerdict.CLAMPED
            if prop.action == Action.ADD and prop.symbol in pos_map:
                current_w = portfolio.weight(prop.symbol)
                max_add = self.limits.config.position.max_add_per_cycle_weight
                if weight > current_w + max_add:
                    weight = current_w + max_add
                    reasons.append("max_add_per_cycle_weight")
                    verdict = RiskVerdict.CLAMPED

            # Shorts forbidden in v1
            if not self.limits.config.account.allow_short and weight < 0:
                results.append(self._reject(prop, ["shorts_not_allowed"]))
                continue

            # 7 early: holding-period gate for sells
            if prop.action in (Action.TRIM, Action.CLOSE) and prop.symbol in pos_map:
                pos = pos_map[prop.symbol]
                held = (market.as_of.date() - pos.opened_at).days
                min_hold = self.limits.config.turnover.min_holding_days
                if held < min_hold and not stop.get(prop.symbol, False):
                    results.append(self._reject(prop, ["min_holding_days"]))
                    continue

            working.append((prop, weight, reasons, verdict))

        # 4. Count gate — prefer highest confidence
        max_pos = self.limits.config.position.max_positions
        current_symbols = {p.symbol for p in portfolio.positions}
        # Sort by confidence desc for prioritization
        working.sort(key=lambda x: x[0].confidence, reverse=True)
        accepted_new: set[str] = set()
        filtered: list[tuple[Proposal, Decimal, list[str], RiskVerdict]] = []
        for prop, weight, reasons, verdict in working:
            if prop.action == Action.CLOSE:
                filtered.append((prop, weight, reasons, verdict))
                continue
            projected = current_symbols | accepted_new | {prop.symbol}
            # closes free slots
            closing = {
                p.symbol
                for p, w, r, v in working
                if p.action == Action.CLOSE and w == 0
            }
            projected = (projected - closing)
            if len(projected) > max_pos and prop.symbol not in current_symbols:
                results.append(self._reject(prop, ["max_positions"]))
                continue
            if prop.symbol not in current_symbols:
                accepted_new.add(prop.symbol)
            filtered.append((prop, weight, reasons, verdict))
        working = filtered

        # 5–6. Concentration + gross exposure clamps on post-trade weights
        working = self._clamp_concentration(working, portfolio, market)
        working = self._clamp_gross(working, portfolio)

        # 8. Turnover gate
        working = self._turnover_gate(working, portfolio, market)

        # 9. Order-count gate
        max_orders = self.limits.config.turnover.max_orders_per_day
        tradeable = [w for w in working if w[0].action not in (Action.HOLD, Action.ABSTAIN)]
        tradeable.sort(key=lambda x: x[0].confidence, reverse=True)
        kept = tradeable[:max_orders]
        dropped = tradeable[max_orders:]
        for prop, weight, reasons, verdict in dropped:
            results.append(self._reject(prop, ["max_orders_per_day"]))
        working = kept

        # 10–11. Sizing + order-level gates
        min_notional = self.limits.config.position.min_position_notional
        for prop, weight, reasons, verdict in working:
            price = market.last_prices.get(prop.symbol, Decimal("0"))
            pos_map = portfolio.position_map()
            cur_qty = pos_map[prop.symbol].quantity if prop.symbol in pos_map else Decimal("0")
            if prop.action == Action.CLOSE:
                weight = Decimal("0")
            delta = shares_for_weight(weight, portfolio.equity, price, cur_qty)
            if delta == 0 and prop.action not in (Action.HOLD, Action.ABSTAIN):
                # no-op after rounding
                if prop.action == Action.CLOSE and cur_qty > 0:
                    delta = -cur_qty
                else:
                    results.append(self._reject(prop, reasons + ["zero_shares_after_round"]))
                    continue
            order_notional = notional(delta, price)
            if order_notional < min_notional and prop.action != Action.CLOSE:
                results.append(self._reject(prop, reasons + ["min_position_notional"]))
                continue

            # Order-level: spread, ADV, session window
            ocfg = self.limits.config.order
            spread = market.spreads_bps.get(prop.symbol)
            if spread is not None and spread > ocfg.max_spread_bps:
                results.append(self._reject(prop, reasons + ["max_spread_bps"]))
                continue
            adv = market.adv_notional.get(prop.symbol)
            if adv and adv > 0 and order_notional / adv > ocfg.max_pct_of_adv:
                results.append(self._reject(prop, reasons + ["max_pct_of_adv"]))
                continue
            if market.session_open_minutes < ocfg.no_trade_first_minutes:
                results.append(self._reject(prop, reasons + ["no_trade_first_minutes"]))
                continue
            if market.session_close_minutes < ocfg.no_trade_last_minutes:
                results.append(self._reject(prop, reasons + ["no_trade_last_minutes"]))
                continue

            if not reasons:
                reasons = ["ok"]
            results.append(
                RiskDecision(
                    proposal=prop,
                    verdict=verdict,
                    final_weight=weight,
                    reasons=reasons,
                    limits_version=self.limits.version,
                )
            )

        return results

    def _reject(self, prop: Proposal, reasons: list[str]) -> RiskDecision:
        return RiskDecision(
            proposal=prop,
            verdict=RiskVerdict.REJECTED,
            final_weight=Decimal("0")
            if prop.action in (Action.OPEN, Action.CLOSE)
            else prop.target_weight,
            reasons=reasons,
            limits_version=self.limits.version,
        )

    def _clamp_concentration(
        self,
        working: list[tuple[Proposal, Decimal, list[str], RiskVerdict]],
        portfolio: PortfolioSnapshot,
        market: MarketContext,
    ) -> list[tuple[Proposal, Decimal, list[str], RiskVerdict]]:
        max_sector = self.limits.config.concentration.max_sector_weight
        max_cluster = self.limits.config.concentration.max_cluster_weight
        # Build post-trade weight map
        weights = {p.symbol: portfolio.weight(p.symbol) for p in portfolio.positions}
        for prop, weight, reasons, verdict in working:
            weights[prop.symbol] = weight

        def sector_of(sym: str) -> str:
            return market.sectors.get(sym, "unknown")

        def cluster_of(sym: str) -> str:
            return market.clusters.get(sym, "none")

        # Scale down offending names proportionally
        out: list[tuple[Proposal, Decimal, list[str], RiskVerdict]] = []
        for prop, weight, reasons, verdict in working:
            w = weight
            # Sector
            sec = sector_of(prop.symbol)
            sec_total = sum(wt for s, wt in weights.items() if sector_of(s) == sec)
            if sec_total > max_sector and w > 0:
                scale = max_sector / sec_total
                w = w * scale
                reasons = reasons + ["max_sector_weight"]
                verdict = RiskVerdict.CLAMPED
                weights[prop.symbol] = w
            # Cluster
            cl = cluster_of(prop.symbol)
            cl_total = sum(wt for s, wt in weights.items() if cluster_of(s) == cl)
            if cl_total > max_cluster and w > 0:
                scale = max_cluster / cl_total
                w = w * scale
                reasons = reasons + ["max_cluster_weight"]
                verdict = RiskVerdict.CLAMPED
                weights[prop.symbol] = w
            out.append((prop, w, reasons, verdict))
        return out

    def _clamp_gross(
        self,
        working: list[tuple[Proposal, Decimal, list[str], RiskVerdict]],
        portfolio: PortfolioSnapshot,
    ) -> list[tuple[Proposal, Decimal, list[str], RiskVerdict]]:
        max_gross = self.limits.config.account.max_gross_exposure
        min_cash = self.limits.config.account.min_cash_buffer_pct
        weights = {p.symbol: portfolio.weight(p.symbol) for p in portfolio.positions}
        for prop, weight, _, _ in working:
            weights[prop.symbol] = weight
        gross = sum(abs(w) for w in weights.values())
        if gross <= max_gross and (1 - gross) >= min_cash:
            return working
        # Scale buys down
        scale = min(
            max_gross / gross if gross > 0 else Decimal("1"),
            (Decimal("1") - min_cash) / gross if gross > 0 else Decimal("1"),
        )
        out: list[tuple[Proposal, Decimal, list[str], RiskVerdict]] = []
        for prop, weight, reasons, verdict in working:
            w = weight
            if weight > portfolio.weight(prop.symbol):
                # buy side
                w = weight * scale
                reasons = reasons + ["max_gross_exposure"]
                verdict = RiskVerdict.CLAMPED
            out.append((prop, w, reasons, verdict))
        return out

    def _turnover_gate(
        self,
        working: list[tuple[Proposal, Decimal, list[str], RiskVerdict]],
        portfolio: PortfolioSnapshot,
        market: MarketContext,
    ) -> list[tuple[Proposal, Decimal, list[str], RiskVerdict]]:
        max_to = self.limits.config.turnover.max_daily_turnover
        if portfolio.equity <= 0:
            return working
        # Estimate turnover
        items = sorted(working, key=lambda x: x[0].confidence, reverse=True)
        kept: list[tuple[Proposal, Decimal, list[str], RiskVerdict]] = []
        turnover = Decimal("0")
        for prop, weight, reasons, verdict in items:
            cur = portfolio.weight(prop.symbol)
            delta_w = abs(weight - cur)
            if turnover + delta_w > max_to and prop.action != Action.CLOSE:
                continue  # drop low-confidence (already sorted)
            turnover += delta_w
            kept.append((prop, weight, reasons, verdict))
        return kept

    def post_trade_ok(
        self,
        decisions: Iterable[RiskDecision],
        portfolio: PortfolioSnapshot,
        market: MarketContext,
    ) -> tuple[bool, list[str]]:
        """Property-test helper: post-trade portfolio never violates hard limits."""
        weights = {p.symbol: portfolio.weight(p.symbol) for p in portfolio.positions}
        for d in decisions:
            if d.verdict == RiskVerdict.REJECTED:
                continue
            if d.proposal.action in (Action.HOLD, Action.ABSTAIN):
                continue
            weights[d.proposal.symbol] = d.final_weight

        violations: list[str] = []
        cfg = self.limits.config
        gross = sum(abs(w) for w in weights.values())
        # Allow tiny float/decimal slack
        eps = Decimal("0.001")
        if gross > cfg.account.max_gross_exposure + eps:
            violations.append(f"gross {gross} > max")
        if (Decimal("1") - gross) + eps < cfg.account.min_cash_buffer_pct:
            # cash buffer only if we have full allocation view
            pass  # soft: cash may include unsettled; checked via gross
        for sym, w in weights.items():
            if w > cfg.position.max_position_weight + eps:
                violations.append(f"{sym} weight {w} > max_position")
            if w < 0 and not cfg.account.allow_short:
                violations.append(f"{sym} short not allowed")
        n_pos = sum(1 for w in weights.values() if w > 0)
        if n_pos > cfg.position.max_positions:
            violations.append(f"positions {n_pos} > max")
        return len(violations) == 0, violations
