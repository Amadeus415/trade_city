"""Kill-switch state machine. Persisted in SQLite — crash must not clear a halt."""

from __future__ import annotations

from decimal import Decimal

from fund.store.journal import Journal
from fund.types import RiskStateName


class KillSwitch:
    """
    NORMAL ──daily_loss_soft / 3 loss days──► REDUCED
    REDUCED ──equity recovery──► NORMAL
    NORMAL/REDUCED ──hard limits──► HALTED
    HALTED ──manual CLI reset only──► NORMAL
    """

    def __init__(self, journal: Journal) -> None:
        self.journal = journal

    def current(self) -> RiskStateName:
        row = self.journal.get_risk_state()
        return RiskStateName(row["state"])

    def consecutive_loss_days(self) -> int:
        return int(self.journal.get_risk_state()["consecutive_loss_days"])

    def evaluate_pnl(
        self,
        *,
        daily_pnl_pct: Decimal,
        weekly_pnl_pct: Decimal,
        drawdown_pct: Decimal,
        equity_recovered: bool,
        soft_daily: Decimal,
        hard_daily: Decimal,
        hard_weekly: Decimal,
        max_dd: Decimal,
        consecutive_halt: int = 5,
    ) -> RiskStateName:
        """Update state from latest P&L metrics. Returns new state."""
        state = self.current()
        loss_days = self.consecutive_loss_days()

        if daily_pnl_pct < 0:
            loss_days += 1
        else:
            loss_days = 0
        self.journal.set_consecutive_loss_days(loss_days)

        # Hard halt conditions (from NORMAL or REDUCED)
        if state != RiskStateName.HALTED:
            if daily_pnl_pct <= -hard_daily:
                self.journal.transition_risk_state(
                    RiskStateName.HALTED,
                    trigger_metric="daily_loss_hard_pct",
                    trigger_value=str(daily_pnl_pct),
                )
                return RiskStateName.HALTED
            if weekly_pnl_pct <= -hard_weekly:
                self.journal.transition_risk_state(
                    RiskStateName.HALTED,
                    trigger_metric="weekly_loss_hard_pct",
                    trigger_value=str(weekly_pnl_pct),
                )
                return RiskStateName.HALTED
            if drawdown_pct >= max_dd:
                self.journal.transition_risk_state(
                    RiskStateName.HALTED,
                    trigger_metric="max_drawdown_halt_pct",
                    trigger_value=str(drawdown_pct),
                )
                return RiskStateName.HALTED
            if loss_days >= consecutive_halt:
                self.journal.transition_risk_state(
                    RiskStateName.HALTED,
                    trigger_metric="consecutive_loss_days",
                    trigger_value=str(loss_days),
                    consecutive_loss_days=loss_days,
                )
                return RiskStateName.HALTED

        if state == RiskStateName.NORMAL:
            if daily_pnl_pct <= -soft_daily or loss_days >= 3:
                self.journal.transition_risk_state(
                    RiskStateName.REDUCED,
                    trigger_metric="daily_loss_soft_or_streak",
                    trigger_value=str(daily_pnl_pct),
                    consecutive_loss_days=loss_days,
                )
                return RiskStateName.REDUCED

        if state == RiskStateName.REDUCED and equity_recovered:
            self.journal.transition_risk_state(
                RiskStateName.NORMAL,
                trigger_metric="equity_recovery",
                trigger_value="true",
                consecutive_loss_days=loss_days,
            )
            return RiskStateName.NORMAL

        return self.current()

    def reset(self, confirm_equity: Decimal, actual_equity: Decimal) -> None:
        """Manual reset from HALTED. Operator must type current equity."""
        if self.current() != RiskStateName.HALTED:
            raise RuntimeError("risk state is not HALTED; reset not needed")
        # Require exact match to 2 decimal places
        if confirm_equity.quantize(Decimal("0.01")) != actual_equity.quantize(Decimal("0.01")):
            raise ValueError(
                f"confirm-equity {confirm_equity} does not match actual equity {actual_equity}"
            )
        self.journal.transition_risk_state(
            RiskStateName.NORMAL,
            trigger_metric="manual_reset",
            trigger_value=str(confirm_equity),
            operator_note="CLI reset with equity confirmation",
            consecutive_loss_days=0,
        )
