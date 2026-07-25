"""Mandatory baselines: SPY B&H, equal-weight, random monkey, momentum."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from fund.clock import trading_days


def _equity_curve_from_weights(
    prices: pd.DataFrame,
    weights_by_session: dict[date, dict[str, float]],
    start_cash: float,
    cost_bps: float = 10.0,
) -> pd.Series:
    """prices: index=session date, columns=symbols."""
    sessions = list(prices.index)
    equity = start_cash
    prev_w: dict[str, float] = {}
    curve = []
    for sess in sessions:
        w = weights_by_session.get(sess, prev_w)
        # turnover cost
        symbols = set(prev_w) | set(w)
        turnover = sum(abs(w.get(s, 0) - prev_w.get(s, 0)) for s in symbols)
        equity *= 1 - turnover * cost_bps / 10000
        # daily return of portfolio
        if sess != sessions[0]:
            prev = sessions[sessions.index(sess) - 1]
            day_ret = 0.0
            for s, wt in prev_w.items():
                if s in prices.columns and prev in prices.index and sess in prices.index:
                    p0 = prices.at[prev, s]
                    p1 = prices.at[sess, s]
                    if p0 and p0 == p0 and p1 and p1 == p1 and p0 != 0:
                        day_ret += wt * (p1 / p0 - 1)
            equity *= 1 + day_ret
        prev_w = w
        curve.append(equity)
    return pd.Series(curve, index=sessions, name="equity")


def buy_and_hold_spy(
    prices: pd.DataFrame,
    start_cash: float = 10000.0,
) -> pd.Series:
    if "SPY" not in prices.columns:
        raise ValueError("SPY required for buy-and-hold baseline")
    spy = prices["SPY"].dropna()
    ret = spy.pct_change().fillna(0)
    equity = start_cash * (1 + ret).cumprod()
    equity.iloc[0] = start_cash
    return equity.rename("spy_bh")


def equal_weight_monthly(
    prices: pd.DataFrame,
    start_cash: float = 10000.0,
    cost_bps: float = 10.0,
) -> pd.Series:
    sessions = list(prices.index)
    symbols = [c for c in prices.columns if c != "SPY"]
    if not symbols:
        symbols = list(prices.columns)
    weights_by_session: dict[date, dict[str, float]] = {}
    current: dict[str, float] = {}
    last_month = None
    for sess in sessions:
        m = (sess.year, sess.month)
        if m != last_month:
            available = [s for s in symbols if not pd.isna(prices.at[sess, s])]
            n = len(available) or 1
            current = {s: 1.0 / n for s in available}
            last_month = m
        weights_by_session[sess] = current
    return _equity_curve_from_weights(prices, weights_by_session, start_cash, cost_bps).rename(
        "equal_weight"
    )


def momentum_top_n(
    prices: pd.DataFrame,
    start_cash: float = 10000.0,
    lookback: int = 63,
    top_n: int = 5,
    cost_bps: float = 10.0,
) -> pd.Series:
    sessions = list(prices.index)
    symbols = [c for c in prices.columns if c != "SPY"] or list(prices.columns)
    weights_by_session: dict[date, dict[str, float]] = {}
    current: dict[str, float] = {}
    last_month = None
    for i, sess in enumerate(sessions):
        m = (sess.year, sess.month)
        if m != last_month and i >= lookback:
            moms = {}
            for s in symbols:
                window = prices[s].iloc[i - lookback : i + 1].dropna()
                if len(window) >= lookback:
                    moms[s] = window.iloc[-1] / window.iloc[0] - 1
            ranked = sorted(moms, key=lambda s: moms[s], reverse=True)[:top_n]
            current = {s: 1.0 / len(ranked) for s in ranked} if ranked else {}
            last_month = m
        weights_by_session[sess] = current
    return _equity_curve_from_weights(prices, weights_by_session, start_cash, cost_bps).rename(
        "momentum"
    )


def random_monkey(
    prices: pd.DataFrame,
    start_cash: float = 10000.0,
    n_positions: int = 5,
    turnover_monthly: float = 0.3,
    n_runs: int = 200,
    seed: int = 42,
    cost_bps: float = 10.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (all_runs DataFrame, mean curve)."""
    rng = np.random.default_rng(seed)
    sessions = list(prices.index)
    symbols = [c for c in prices.columns if c != "SPY"] or list(prices.columns)
    runs = []
    for run in range(n_runs):
        weights_by_session: dict[date, dict[str, float]] = {}
        current: dict[str, float] = {}
        last_month = None
        for sess in sessions:
            m = (sess.year, sess.month)
            if m != last_month:
                pick = list(rng.choice(symbols, size=min(n_positions, len(symbols)), replace=False))
                current = {s: 1.0 / len(pick) for s in pick}
                last_month = m
            weights_by_session[sess] = current
        curve = _equity_curve_from_weights(
            prices, weights_by_session, start_cash, cost_bps
        )
        runs.append(curve.rename(f"monkey_{run}"))
    df = pd.concat(runs, axis=1)
    return df, df.mean(axis=1).rename("monkey_mean")


def monkey_percentile(agent_final: float, monkey_runs: pd.DataFrame) -> float:
    finals = monkey_runs.iloc[-1].values
    return float(np.mean(finals < agent_final) * 100)


def run_all_baselines(
    prices: pd.DataFrame,
    start_cash: float = 10000.0,
    monkey_runs: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    spy = buy_and_hold_spy(prices, start_cash)
    ew = equal_weight_monthly(prices, start_cash)
    mom = momentum_top_n(prices, start_cash)
    monkeys, monkey_mean = random_monkey(
        prices, start_cash, n_runs=monkey_runs, seed=seed
    )
    return {
        "spy_bh": spy,
        "equal_weight": ew,
        "momentum": mom,
        "monkey_mean": monkey_mean,
        "monkey_runs": monkeys,
    }
