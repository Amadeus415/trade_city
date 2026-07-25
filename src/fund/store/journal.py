"""SQLite journal — decisions, orders, fills, risk state (WAL mode)."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import uuid
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from fund.clock import Clock, ET
from fund.types import (
    Fill,
    OrderIntent,
    OrderStatus,
    Proposal,
    RiskDecision,
    RiskStateName,
)


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


class Journal:
    def __init__(self, path: str | Path, clock: Clock | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock or Clock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    def close(self) -> None:
        self._conn.close()

    def migrate(self, migrations_dir: str | Path = "migrations") -> None:
        mig_dir = Path(migrations_dir)
        for sql_file in sorted(mig_dir.glob("*.sql")):
            sql = sql_file.read_text()
            self._conn.executescript(sql)
        self._conn.commit()

    # ── runs ──────────────────────────────────────────────────────────

    def start_run(
        self,
        mode: str,
        config_hash: str,
        limits_version: str,
        run_id: str | None = None,
    ) -> str:
        rid = run_id or str(uuid.uuid4())
        self._conn.execute(
            """
            INSERT INTO runs (run_id, mode, started_at, config_hash, git_sha, limits_version)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                rid,
                mode,
                self.clock.now().isoformat(),
                config_hash,
                _git_sha(),
                limits_version,
            ),
        )
        self._conn.commit()
        return rid

    # ── decisions / proposals / risk ──────────────────────────────────

    def record_decision(
        self,
        *,
        run_id: str,
        as_of: datetime,
        session: str,
        prompt_version: str,
        model: str,
        temperature: float,
        masking_mode: str,
        feature_packet_hash: str,
        feature_packet_json: str,
        raw_llm_response: str,
        latency_ms: int | None = None,
        cost_usd: float | None = None,
        decision_id: str | None = None,
    ) -> str:
        did = decision_id or str(uuid.uuid4())
        self._conn.execute(
            """
            INSERT INTO decisions (
              decision_id, run_id, as_of, session, prompt_version, model,
              temperature, masking_mode, feature_packet_hash, feature_packet_json,
              raw_llm_response, latency_ms, cost_usd
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                did,
                run_id,
                as_of.isoformat(),
                session,
                prompt_version,
                model,
                temperature,
                masking_mode,
                feature_packet_hash,
                feature_packet_json,
                raw_llm_response,
                latency_ms,
                cost_usd,
            ),
        )
        self._conn.commit()
        return did

    def record_proposals(self, decision_id: str, proposals: list[Proposal]) -> list[str]:
        ids: list[str] = []
        for p in proposals:
            pid = str(uuid.uuid4())
            self._conn.execute(
                """
                INSERT INTO proposals (
                  proposal_id, decision_id, symbol, action, target_weight,
                  confidence, thesis, invalidation, horizon_days, source_features
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pid,
                    decision_id,
                    p.symbol,
                    p.action.value,
                    str(p.target_weight),
                    str(p.confidence),
                    p.thesis,
                    p.invalidation,
                    p.horizon_days,
                    json.dumps(p.source_features),
                ),
            )
            ids.append(pid)
        self._conn.commit()
        return ids

    def record_risk_decisions(
        self,
        proposal_ids: list[str],
        decisions: list[RiskDecision],
    ) -> None:
        now = self.clock.now().isoformat()
        for pid, rd in zip(proposal_ids, decisions, strict=True):
            self._conn.execute(
                """
                INSERT INTO risk_decisions (
                  proposal_id, verdict, final_weight, reasons, limits_version, evaluated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    pid,
                    rd.verdict.value,
                    str(rd.final_weight),
                    json.dumps(rd.reasons),
                    rd.limits_version,
                    now,
                ),
            )
        self._conn.commit()

    # ── orders / fills ────────────────────────────────────────────────

    def insert_order_intent(self, intent: OrderIntent) -> bool:
        """Insert intent row. Returns False if client_order_id already exists (idempotency)."""
        try:
            self._conn.execute(
                """
                INSERT INTO orders (
                  client_order_id, decision_id, symbol, side, quantity, limit_price,
                  status, broker_order_id, created_at, updated_at, review_response, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, NULL, NULL)
                """,
                (
                    intent.client_order_id,
                    intent.decision_id,
                    intent.symbol,
                    intent.side.value,
                    str(intent.quantity),
                    str(intent.limit_price),
                    OrderStatus.INTENT.value,
                    intent.created_at.isoformat(),
                    intent.created_at.isoformat(),
                ),
            )
            self._conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def update_order(
        self,
        client_order_id: str,
        *,
        status: OrderStatus | None = None,
        broker_order_id: str | None = None,
        review_response: str | None = None,
        error: str | None = None,
    ) -> None:
        fields: list[str] = ["updated_at = ?"]
        vals: list[Any] = [self.clock.now().isoformat()]
        if status is not None:
            fields.append("status = ?")
            vals.append(status.value)
        if broker_order_id is not None:
            fields.append("broker_order_id = ?")
            vals.append(broker_order_id)
        if review_response is not None:
            fields.append("review_response = ?")
            vals.append(review_response)
        if error is not None:
            fields.append("error = ?")
            vals.append(error)
        vals.append(client_order_id)
        self._conn.execute(
            f"UPDATE orders SET {', '.join(fields)} WHERE client_order_id = ?",
            vals,
        )
        self._conn.commit()

    def record_fill(self, fill: Fill) -> None:
        self._conn.execute(
            """
            INSERT INTO fills (fill_id, client_order_id, quantity, price, fees, filled_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                fill.fill_id,
                fill.client_order_id,
                str(fill.quantity),
                str(fill.price),
                str(fill.fees),
                fill.filled_at.isoformat(),
            ),
        )
        self._conn.commit()

    def get_open_orders(self) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            """
            SELECT * FROM orders
            WHERE status IN ('intent', 'acknowledged', 'partial')
            ORDER BY created_at
            """
        )
        return [dict(r) for r in cur.fetchall()]

    def get_order(self, client_order_id: str) -> dict[str, Any] | None:
        cur = self._conn.execute(
            "SELECT * FROM orders WHERE client_order_id = ?",
            (client_order_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    # ── equity ────────────────────────────────────────────────────────

    def snapshot_equity(
        self,
        run_id: str,
        session: str,
        cash: Decimal,
        equity: Decimal,
        peak_equity: Decimal,
        positions: list[dict[str, Any]],
    ) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO equity_snapshots
              (run_id, session, cash, equity, peak_equity, positions_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                session,
                str(cash),
                str(equity),
                str(peak_equity),
                json.dumps(positions, default=str),
            ),
        )
        self._conn.commit()

    def equity_curve(self, run_id: str) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            """
            SELECT session, cash, equity, peak_equity, positions_json
            FROM equity_snapshots WHERE run_id = ? ORDER BY session
            """,
            (run_id,),
        )
        return [dict(r) for r in cur.fetchall()]

    # ── risk state machine ────────────────────────────────────────────

    def get_risk_state(self) -> dict[str, Any]:
        cur = self._conn.execute("SELECT * FROM risk_state WHERE id = 1")
        row = cur.fetchone()
        if row is None:
            self._conn.execute(
                """
                INSERT INTO risk_state (id, state, since, consecutive_loss_days)
                VALUES (1, 'NORMAL', ?, 0)
                """,
                (self.clock.now().isoformat(),),
            )
            self._conn.commit()
            return {
                "id": 1,
                "state": RiskStateName.NORMAL.value,
                "since": self.clock.now().isoformat(),
                "trigger_metric": None,
                "trigger_value": None,
                "consecutive_loss_days": 0,
            }
        return dict(row)

    def transition_risk_state(
        self,
        to_state: RiskStateName,
        trigger_metric: str | None = None,
        trigger_value: str | None = None,
        operator_note: str | None = None,
        consecutive_loss_days: int | None = None,
    ) -> None:
        current = self.get_risk_state()
        from_state = current["state"]
        if from_state == to_state.value and consecutive_loss_days is None:
            return
        now = self.clock.now().isoformat()
        days = (
            consecutive_loss_days
            if consecutive_loss_days is not None
            else current["consecutive_loss_days"]
        )
        self._conn.execute(
            """
            UPDATE risk_state
            SET state = ?, since = ?, trigger_metric = ?, trigger_value = ?,
                consecutive_loss_days = ?
            WHERE id = 1
            """,
            (to_state.value, now, trigger_metric, trigger_value, days),
        )
        self._conn.execute(
            """
            INSERT INTO risk_state_transitions
              (from_state, to_state, at, trigger_metric, trigger_value, operator_note)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                from_state,
                to_state.value,
                now,
                trigger_metric,
                trigger_value,
                operator_note,
            ),
        )
        self._conn.commit()

    def set_consecutive_loss_days(self, n: int) -> None:
        self._conn.execute(
            "UPDATE risk_state SET consecutive_loss_days = ? WHERE id = 1",
            (n,),
        )
        self._conn.commit()

    def risk_transitions(self) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT * FROM risk_state_transitions ORDER BY id"
        )
        return [dict(r) for r in cur.fetchall()]

    # ── queries for eval ──────────────────────────────────────────────

    def list_runs(self) -> list[dict[str, Any]]:
        cur = self._conn.execute("SELECT * FROM runs ORDER BY started_at DESC")
        return [dict(r) for r in cur.fetchall()]

    def risk_rejections(self, run_id: str | None = None) -> list[dict[str, Any]]:
        if run_id:
            cur = self._conn.execute(
                """
                SELECT rd.*, p.symbol, p.action, d.run_id
                FROM risk_decisions rd
                JOIN proposals p ON p.proposal_id = rd.proposal_id
                JOIN decisions d ON d.decision_id = p.decision_id
                WHERE d.run_id = ? AND rd.verdict = 'rejected'
                """,
                (run_id,),
            )
        else:
            cur = self._conn.execute(
                """
                SELECT rd.*, p.symbol, p.action
                FROM risk_decisions rd
                JOIN proposals p ON p.proposal_id = rd.proposal_id
                WHERE rd.verdict = 'rejected'
                """
            )
        return [dict(r) for r in cur.fetchall()]

    def proposals_for_run(self, run_id: str) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            """
            SELECT p.*, rd.verdict, rd.final_weight, rd.reasons, d.session
            FROM proposals p
            JOIN decisions d ON d.decision_id = p.decision_id
            LEFT JOIN risk_decisions rd ON rd.proposal_id = p.proposal_id
            WHERE d.run_id = ?
            ORDER BY d.session, p.symbol
            """,
            (run_id,),
        )
        return [dict(r) for r in cur.fetchall()]
