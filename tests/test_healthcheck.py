from __future__ import annotations

from pathlib import Path

from fund.orchestrate.healthcheck import healthcheck


def test_healthcheck_empty_db(tmp_path: Path):
    db = tmp_path / "journal.db"
    # migrations relative to repo root
    result = healthcheck(db, migrations_dir="migrations")
    assert result["ok"] is True
    assert result["risk_state"] == "NORMAL"
