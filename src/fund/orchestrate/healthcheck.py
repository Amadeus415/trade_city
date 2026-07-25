"""Operational healthcheck — empty DB is OK if migrations applied."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fund.clock import Clock, is_trading_day
from fund.store.journal import Journal


def healthcheck(
    journal_path: str | Path,
    migrations_dir: str | Path = "migrations",
) -> dict[str, Any]:
    path = Path(journal_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    journal = Journal(path)
    try:
        journal.migrate(migrations_dir)
        state = journal.get_risk_state()
        clock = Clock()
        today = clock.today()
        ok = True
        issues: list[str] = []
        if state["state"] == "HALTED":
            issues.append("risk state is HALTED")
        result = {
            "ok": ok,
            "journal": str(path),
            "risk_state": state["state"],
            "consecutive_loss_days": state["consecutive_loss_days"],
            "is_trading_day": is_trading_day(today),
            "today": today.isoformat(),
            "issues": issues,
        }
        return result
    finally:
        journal.close()
