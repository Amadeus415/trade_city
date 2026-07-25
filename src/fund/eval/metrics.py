"""Performance metrics — honest evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Metrics:
    cagr: float
    total_return: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    drawdown_duration_days: int
    calmar: float
    hit_rate: float
    avg_win_loss_ratio: float
    profit_factor: float
    turnover: float | None
    avg_holding_period: float | None
    exposure_pct: float | None
    cost_drag: float | None
    rolling_sharpe_6m: list[float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "cagr": self.cagr,
            "total_return": self.total_return,
            "volatility": self.volatility,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "drawdown_duration_days": self.drawdown_duration_days,
            "calmar": self.calmar,
            "hit_rate": self.hit_rate,
            "avg_win_loss_ratio": self.avg_win_loss_ratio,
            "profit_factor": self.profit_factor,
            "turnover": self.turnover,
            "avg_holding_period": self.avg_holding_period,
            "exposure_pct": self.exposure_pct,
            "cost_drag": self.cost_drag,
        }


def compute_metrics(
    equity: pd.Series,
    *,
    periods_per_year: int = 252,
    gross_equity: pd.Series | None = None,
    turnover: float | None = None,
    avg_holding_period: float | None = None,
    exposure_pct: float | None = None,
) -> Metrics:
    eq = equity.dropna().astype(float)
    if len(eq) < 2:
        return Metrics(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, turnover, avg_holding_period, exposure_pct, None, []
        )
    rets = eq.pct_change().dropna()
    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1)
    n_years = len(eq) / periods_per_year
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / n_years) - 1) if n_years > 0 else 0.0
    vol = float(rets.std() * np.sqrt(periods_per_year))
    mean_r = float(rets.mean() * periods_per_year)
    sharpe = mean_r / vol if vol > 0 else 0.0
    downside = rets[rets < 0]
    down_std = float(downside.std() * np.sqrt(periods_per_year)) if len(downside) else 0.0
    sortino = mean_r / down_std if down_std > 0 else 0.0

    peak = eq.cummax()
    dd = eq / peak - 1
    max_dd = float(dd.min())
    # duration
    in_dd = dd < 0
    max_dur = 0
    cur = 0
    for v in in_dd:
        if v:
            cur += 1
            max_dur = max(max_dur, cur)
        else:
            cur = 0

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    hit = float(len(wins) / len(rets)) if len(rets) else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) else 0.0
    wl = avg_win / avg_loss if avg_loss > 0 else 0.0
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else 0.0

    cost_drag = None
    if gross_equity is not None and len(gross_equity) == len(eq):
        g = float(gross_equity.iloc[-1] / gross_equity.iloc[0] - 1)
        cost_drag = g - total_return

    # Rolling 6m Sharpe (~126 sessions)
    window = 126
    rolling: list[float] = []
    for i in range(window, len(rets) + 1):
        w = rets.iloc[i - window : i]
        wv = w.std() * np.sqrt(periods_per_year)
        wm = w.mean() * periods_per_year
        rolling.append(float(wm / wv) if wv > 0 else 0.0)

    return Metrics(
        cagr=cagr,
        total_return=total_return,
        volatility=vol,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        drawdown_duration_days=max_dur,
        calmar=calmar,
        hit_rate=hit,
        avg_win_loss_ratio=wl,
        profit_factor=pf,
        turnover=turnover,
        avg_holding_period=avg_holding_period,
        exposure_pct=exposure_pct,
        cost_drag=cost_drag,
        rolling_sharpe_6m=rolling,
    )
