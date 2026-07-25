"""Per-symbol price/vol features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fund.features.registry import feature


def _close(bars: pd.DataFrame) -> np.ndarray:
    return bars["adj_close"].astype(float).values if "adj_close" in bars.columns else bars["close"].astype(float).values


def _vol(bars: pd.DataFrame) -> np.ndarray:
    return bars["volume"].astype(float).values


@feature(name="mom_21d", version=1, requires=["close"], lookback=22)
def mom_21d(bars: pd.DataFrame) -> float | None:
    c = _close(bars)
    if len(c) < 22 or c[-22] == 0:
        return None
    return float(c[-1] / c[-22] - 1)


@feature(name="mom_63d", version=1, requires=["close"], lookback=64)
def mom_63d(bars: pd.DataFrame) -> float | None:
    c = _close(bars)
    if len(c) < 64 or c[-64] == 0:
        return None
    return float(c[-1] / c[-64] - 1)


@feature(name="mom_252d", version=1, requires=["close"], lookback=253)
def mom_252d(bars: pd.DataFrame) -> float | None:
    c = _close(bars)
    if len(c) < 253 or c[-253] == 0:
        return None
    return float(c[-1] / c[-253] - 1)


@feature(name="realized_vol_21d", version=1, requires=["close"], lookback=22)
def realized_vol_21d(bars: pd.DataFrame) -> float | None:
    c = _close(bars)
    if len(c) < 22:
        return None
    rets = np.diff(np.log(c[-22:]))
    return float(np.std(rets) * np.sqrt(252))


@feature(name="realized_vol_63d", version=1, requires=["close"], lookback=64)
def realized_vol_63d(bars: pd.DataFrame) -> float | None:
    c = _close(bars)
    if len(c) < 64:
        return None
    rets = np.diff(np.log(c[-64:]))
    return float(np.std(rets) * np.sqrt(252))


@feature(name="atr_14d_pct", version=1, requires=["high", "low", "close"], lookback=15)
def atr_14d_pct(bars: pd.DataFrame) -> float | None:
    if len(bars) < 15:
        return None
    h = bars["high"].astype(float).values
    l = bars["low"].astype(float).values
    c = bars["close"].astype(float).values
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    atr = np.mean(tr[-14:])
    if c[-1] == 0:
        return None
    return float(atr / c[-1])


@feature(name="drawdown_from_252d_high", version=1, requires=["close"], lookback=252)
def drawdown_from_252d_high(bars: pd.DataFrame) -> float | None:
    c = _close(bars)
    if len(c) < 2:
        return None
    window = c[-min(252, len(c)) :]
    peak = np.max(window)
    if peak == 0:
        return None
    return float(c[-1] / peak - 1)


@feature(name="dist_from_ma_50", version=1, requires=["close"], lookback=50)
def dist_from_ma_50(bars: pd.DataFrame) -> float | None:
    c = _close(bars)
    if len(c) < 50:
        return None
    ma = np.mean(c[-50:])
    if ma == 0:
        return None
    return float(c[-1] / ma - 1)


@feature(name="dist_from_ma_200", version=1, requires=["close"], lookback=200)
def dist_from_ma_200(bars: pd.DataFrame) -> float | None:
    c = _close(bars)
    if len(c) < 200:
        return None
    ma = np.mean(c[-200:])
    if ma == 0:
        return None
    return float(c[-1] / ma - 1)


@feature(name="rsi_14", version=1, requires=["close"], lookback=15)
def rsi_14(bars: pd.DataFrame) -> float | None:
    c = _close(bars)
    if len(c) < 15:
        return None
    deltas = np.diff(c[-15:])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


@feature(name="volume_ratio_21d", version=1, requires=["volume"], lookback=21)
def volume_ratio_21d(bars: pd.DataFrame) -> float | None:
    v = _vol(bars)
    if len(v) < 21:
        return None
    avg = np.mean(v[-21:])
    if avg == 0:
        return None
    return float(v[-1] / avg)


@feature(name="gap_overnight_pct", version=1, requires=["open", "close"], lookback=2)
def gap_overnight_pct(bars: pd.DataFrame) -> float | None:
    if len(bars) < 2:
        return None
    prev_close = float(bars.iloc[-2]["close"])
    o = float(bars.iloc[-1]["open"])
    if prev_close == 0:
        return None
    return float(o / prev_close - 1)


@feature(name="beta_to_spy_126d", version=1, requires=["close"], lookback=127)
def beta_to_spy_126d(bars: pd.DataFrame) -> float | None:
    """Requires columns close and spy_close (joined upstream)."""
    if "spy_close" not in bars.columns or len(bars) < 127:
        return None
    c = bars["adj_close"].astype(float).values if "adj_close" in bars.columns else bars["close"].astype(float).values
    s = bars["spy_close"].astype(float).values
    c, s = c[-127:], s[-127:]
    rc = np.diff(np.log(c))
    rs = np.diff(np.log(s))
    if np.var(rs) == 0:
        return None
    return float(np.cov(rc, rs)[0, 1] / np.var(rs))
