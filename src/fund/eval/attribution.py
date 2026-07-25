"""Return attribution: market beta / style / residual selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AttributionResult:
    market_beta: float
    market_explained: float
    style_explained: float
    residual_alpha_ann: float
    residual_r2: float


def attribute_returns(
    strategy_returns: pd.Series,
    market_returns: pd.Series,
    style_returns: pd.DataFrame | None = None,
) -> AttributionResult:
    """OLS: r_s = a + b * r_m + style factors + e."""
    df = pd.concat(
        [strategy_returns.rename("s"), market_returns.rename("m")],
        axis=1,
    ).dropna()
    if style_returns is not None:
        df = df.join(style_returns, how="inner").dropna()
    if len(df) < 30:
        return AttributionResult(0, 0, 0, 0, 0)

    y = df["s"].values
    cols = ["m"] + [c for c in df.columns if c not in ("s", "m")]
    X = df[cols].values
    X = np.column_stack([np.ones(len(X)), X])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return AttributionResult(0, 0, 0, 0, 0)

    pred = X @ beta
    resid = y - pred
    ss_tot = np.sum((y - y.mean()) ** 2)
    ss_res = np.sum(resid**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Market-only
    X_m = np.column_stack([np.ones(len(df)), df["m"].values])
    b_m, _, _, _ = np.linalg.lstsq(X_m, y, rcond=None)
    pred_m = X_m @ b_m
    r2_m = 1 - np.sum((y - pred_m) ** 2) / ss_tot if ss_tot > 0 else 0.0

    alpha_daily = float(beta[0])
    alpha_ann = alpha_daily * 252
    style_r2 = max(0.0, r2 - r2_m)

    return AttributionResult(
        market_beta=float(beta[1]),
        market_explained=float(r2_m),
        style_explained=float(style_r2),
        residual_alpha_ann=alpha_ann,
        residual_r2=float(1 - r2),
    )
