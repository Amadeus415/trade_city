"""Cross-sectional ranks and correlation clusters."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def rank_features(
    symbol_features: dict[str, dict[str, float | None]],
    feature_name: str,
    out_name: str | None = None,
) -> dict[str, dict[str, float | None]]:
    """Add percentile rank of feature_name within the universe."""
    out_name = out_name or f"{feature_name}_rank"
    pairs = [
        (sym, feats.get(feature_name))
        for sym, feats in symbol_features.items()
        if feats.get(feature_name) is not None
    ]
    if not pairs:
        return symbol_features
    pairs.sort(key=lambda x: x[1])  # type: ignore[arg-type]
    n = len(pairs)
    ranks = {sym: (i + 1) / n for i, (sym, _) in enumerate(pairs)}
    for sym, feats in symbol_features.items():
        feats[out_name] = ranks.get(sym)
    return symbol_features


def sector_relative_mom(
    symbol_features: dict[str, dict[str, float | None]],
    sectors: dict[str, str],
    mom_key: str = "mom_63d",
    out_name: str = "sector_relative_mom_63d",
) -> dict[str, dict[str, float | None]]:
    by_sector: dict[str, list[float]] = {}
    for sym, feats in symbol_features.items():
        m = feats.get(mom_key)
        if m is None:
            continue
        sec = sectors.get(sym, "unknown")
        by_sector.setdefault(sec, []).append(m)
    sector_med = {s: float(np.median(v)) for s, v in by_sector.items() if v}
    for sym, feats in symbol_features.items():
        m = feats.get(mom_key)
        if m is None:
            feats[out_name] = None
            continue
        med = sector_med.get(sectors.get(sym, "unknown"))
        feats[out_name] = None if med is None else m - med
    return symbol_features


def correlation_clusters(
    returns: pd.DataFrame,
    threshold: float = 0.70,
) -> dict[str, str]:
    """Average-linkage clustering at rho >= threshold.

    returns: columns = symbols, rows = sessions.
    """
    if returns.shape[1] < 2:
        return {c: "C0" for c in returns.columns}
    corr = returns.corr().fillna(0.0)
    # Distance = 1 - corr, clipped
    dist = 1 - corr.values
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0, 2)
    # Ensure symmetry
    dist = (dist + dist.T) / 2
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    # fcluster with distance criterion: merge if distance <= 1 - threshold
    labels = fcluster(Z, t=1 - threshold, criterion="distance")
    return {sym: f"C{lab}" for sym, lab in zip(returns.columns, labels, strict=True)}


def size_decile(market_caps: dict[str, float]) -> dict[str, int | None]:
    if not market_caps:
        return {}
    items = sorted(market_caps.items(), key=lambda x: x[1])
    n = len(items)
    out: dict[str, int | None] = {}
    for i, (sym, _) in enumerate(items):
        out[sym] = min(10, int(i / n * 10) + 1)
    return out
