"""Kill-switch persistence and semantics."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from fund.risk.state import KillSwitch
from fund.store.journal import Journal
from fund.types import RiskStateName


def test_halt_survives_restart(tmp_path: Path):
    db = tmp_path / "journal.db"
    j1 = Journal(db)
    j1.migrate()
    ks = KillSwitch(j1)
    j1.transition_risk_state(
        RiskStateName.HALTED,
        trigger_metric="daily_loss_hard_pct",
        trigger_value="-0.04",
    )
    assert ks.current() == RiskStateName.HALTED
    j1.close()

    j2 = Journal(db)
    j2.migrate()
    assert KillSwitch(j2).current() == RiskStateName.HALTED
    j2.close()


def test_halt_blocks_opens_allows_closes(tmp_path: Path):
    """State gate: HALTED rejects OPEN/ADD, allows CLOSE — via risk engine."""
    from datetime import date, datetime, timezone

    from fund.config import RiskConfig
    from fund.risk.engine import RiskEngine
    from fund.risk.limits import RiskLimits
    from fund.types import (
        Action,
        MarketContext,
        PortfolioSnapshot,
        Position,
        Proposal,
        RiskVerdict,
    )

    db = tmp_path / "j.db"
    j = Journal(db)
    j.migrate()
    j.transition_risk_state(RiskStateName.HALTED, trigger_metric="test", trigger_value="1")
    state = KillSwitch(j).current()
    assert state == RiskStateName.HALTED

    engine = RiskEngine(RiskLimits(RiskConfig()))
    port = PortfolioSnapshot(
        as_of=datetime.now(tz=timezone.utc),
        cash=Decimal("5000"),
        equity=Decimal("6000"),
        positions=[
            Position(
                symbol="AAA",
                quantity=Decimal("10"),
                avg_cost=Decimal("100"),
                market_value=Decimal("1000"),
                opened_at=date(2024, 1, 1),
            )
        ],
        peak_equity=Decimal("7000"),
    )
    market = MarketContext(
        as_of=datetime.now(tz=timezone.utc),
        last_prices={"AAA": Decimal("100"), "BBB": Decimal("50")},
        spreads_bps={"AAA": Decimal("5"), "BBB": Decimal("5")},
        adv_notional={"AAA": Decimal("1e7"), "BBB": Decimal("1e7")},
        allowlist={"AAA", "BBB"},
        session_open_minutes=60,
        session_close_minutes=300,
        days_listed={"AAA": 500, "BBB": 500},
    )
    props = [
        Proposal(
            symbol="BBB",
            action=Action.OPEN,
            target_weight=Decimal("0.05"),
            confidence=Decimal("0.5"),
            thesis="open new",
            invalidation="x",
            horizon_days=10,
            source_features=[],
        ),
        Proposal(
            symbol="AAA",
            action=Action.CLOSE,
            target_weight=Decimal("0"),
            confidence=Decimal("0.9"),
            thesis="exit",
            invalidation="x",
            horizon_days=1,
            source_features=[],
        ),
    ]
    dec = engine.evaluate(props, port, market, state)
    by = {d.proposal.symbol: d for d in dec}
    assert by["BBB"].verdict == RiskVerdict.REJECTED
    assert by["AAA"].verdict != RiskVerdict.REJECTED or "state_halted" not in by["AAA"].reasons
    j.close()


def test_manual_reset_requires_equity(tmp_path: Path):
    db = tmp_path / "j.db"
    j = Journal(db)
    j.migrate()
    j.transition_risk_state(RiskStateName.HALTED, trigger_metric="t", trigger_value="1")
    ks = KillSwitch(j)
    with pytest.raises(ValueError):
        ks.reset(Decimal("999.00"), Decimal("1000.00"))
    ks.reset(Decimal("1000.00"), Decimal("1000.00"))
    assert ks.current() == RiskStateName.NORMAL
    j.close()
