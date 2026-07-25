"""INV-2: risk engine properties + anti-tampering."""

from __future__ import annotations

import ast
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st
from pydantic import ValidationError

from fund.config import RiskConfig
from fund.risk.engine import RiskEngine
from fund.risk.limits import RiskLimits
from fund.types import (
    Action,
    MarketContext,
    PortfolioSnapshot,
    Position,
    Proposal,
    RiskStateName,
    RiskVerdict,
)


def _proposal(
    symbol: str = "AAA",
    action: Action = Action.OPEN,
    weight: str = "0.08",
    conf: str = "0.5",
) -> Proposal:
    return Proposal(
        symbol=symbol,
        action=action,
        target_weight=Decimal(weight),
        confidence=Decimal(conf),
        thesis="test thesis " * 5,
        invalidation="price breaks support",
        horizon_days=21,
        source_features=["mom_63d"],
    )


def _portfolio(cash: str = "5000", positions: list[Position] | None = None) -> PortfolioSnapshot:
    pos = positions or []
    mv = sum((p.market_value for p in pos), Decimal("0"))
    cash_d = Decimal(cash)
    eq = cash_d + mv
    return PortfolioSnapshot(
        as_of=datetime(2024, 6, 3, 16, 15, tzinfo=timezone.utc),
        cash=cash_d,
        equity=eq,
        positions=pos,
        peak_equity=eq,
    )


def _market(symbols: list[str] | None = None) -> MarketContext:
    symbols = symbols or ["AAA", "BBB", "CCC", "DDD", "EEE"]
    return MarketContext(
        as_of=datetime(2024, 6, 3, 16, 15, tzinfo=timezone.utc),
        last_prices={s: Decimal("100") for s in symbols},
        spreads_bps={s: Decimal("5") for s in symbols},
        adv_notional={s: Decimal("20000000") for s in symbols},
        sectors={s: "technology" if s in ("AAA", "BBB") else "healthcare" for s in symbols},
        clusters={s: "C1" if s in ("AAA", "BBB") else "C2" for s in symbols},
        allowlist=set(symbols),
        blocklist=set(),
        session_open_minutes=60,
        session_close_minutes=300,
        days_listed={s: 500 for s in symbols},
    )


def test_limits_frozen():
    cfg = RiskConfig()
    with pytest.raises(ValidationError):
        cfg.account.max_gross_exposure = Decimal("1.5")  # type: ignore[misc]


def test_no_agent_import():
    risk_dir = Path(__file__).resolve().parents[1] / "src" / "fund" / "risk"
    for path in risk_dir.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("fund.agent"), path
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("fund.agent"):
                    raise AssertionError(f"{path} imports {node.module}")


def test_halt_blocks_opens_allows_closes():
    engine = RiskEngine(RiskLimits(RiskConfig()))
    port = _portfolio(
        positions=[
            Position(
                symbol="AAA",
                quantity=Decimal("10"),
                avg_cost=Decimal("100"),
                market_value=Decimal("1000"),
                opened_at=date(2024, 1, 1),
            )
        ]
    )
    market = _market()
    open_p = _proposal("BBB", Action.OPEN, "0.05")
    close_p = _proposal("AAA", Action.CLOSE, "0")
    decisions = engine.evaluate(
        [open_p, close_p], port, market, RiskStateName.HALTED
    )
    by_sym = {d.proposal.symbol: d for d in decisions}
    assert by_sym["BBB"].verdict == RiskVerdict.REJECTED
    assert "state_halted" in by_sym["BBB"].reasons
    assert by_sym["AAA"].verdict in (RiskVerdict.ACCEPTED, RiskVerdict.CLAMPED)


def test_max_position_clamped():
    engine = RiskEngine(RiskLimits(RiskConfig()))
    port = _portfolio()
    market = _market()
    p = _proposal("AAA", Action.OPEN, "0.50")  # way over 10%
    decisions = engine.evaluate([p], port, market, RiskStateName.NORMAL)
    d = decisions[0]
    assert d.verdict in (RiskVerdict.CLAMPED, RiskVerdict.REJECTED)
    if d.verdict == RiskVerdict.CLAMPED:
        assert d.final_weight <= Decimal("0.10")


@given(
    n=st.integers(1, 8),
    weights=st.lists(
        st.decimals(min_value=Decimal("0.01"), max_value=Decimal("0.25"), places=2),
        min_size=1,
        max_size=8,
    ),
    confs=st.lists(
        st.decimals(min_value=Decimal("0.1"), max_value=Decimal("0.9"), places=2),
        min_size=1,
        max_size=8,
    ),
)
@settings(max_examples=100, deadline=None)
def test_post_trade_never_violates(n, weights, confs):
    engine = RiskEngine(RiskLimits(RiskConfig()))
    symbols = [f"S{i}" for i in range(max(n, len(weights), len(confs)))]
    proposals = []
    for i, sym in enumerate(symbols):
        w = weights[i % len(weights)]
        c = confs[i % len(confs)]
        proposals.append(
            Proposal(
                symbol=sym,
                action=Action.OPEN,
                target_weight=Decimal(str(w)),
                confidence=Decimal(str(c)),
                thesis="hypothesis generated thesis text for testing",
                invalidation="hypothesis invalidation condition text",
                horizon_days=21,
                source_features=["mom_63d"],
            )
        )
    port = _portfolio(cash="10000")
    market = _market(symbols)
    decisions = engine.evaluate(proposals, port, market, RiskStateName.NORMAL)
    ok, violations = engine.post_trade_ok(decisions, port, market)
    assert ok, violations
