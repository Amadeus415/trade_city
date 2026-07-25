"""Build the compact feature packet table for the agent."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from decimal import Decimal
from typing import Any

import pandas as pd

from fund.features.cross_sectional import (
    correlation_clusters,
    rank_features,
    sector_relative_mom,
)
from fund.features.registry import FeatureRegistry
from fund.types import PortfolioSnapshot


def build_feature_table(
    symbol_features: dict[str, dict[str, float | None]],
    portfolio: PortfolioSnapshot,
    *,
    as_of: datetime,
    sectors: dict[str, str],
    clusters: dict[str, str],
    constraints: dict[str, Any],
    risk_state: str,
    columns: list[str] | None = None,
) -> str:
    cols = columns or [
        "mom_63d",
        "mom_63d_rank",
        "realized_vol_21d",
        "drawdown_from_252d_high",
        "pe_ttm",
        "days_to_earnings",
        "news_count_7d",
    ]
    lines = [
        f"UNIVERSE SNAPSHOT — as_of {as_of.strftime('%Y-%m-%d %H:%M %Z')}".strip(),
        "sym   " + "  ".join(f"{c[:10]:>10}" for c in cols) + "  cluster",
    ]
    for sym in sorted(symbol_features.keys()):
        feats = symbol_features[sym]
        cells = []
        for c in cols:
            v = feats.get(c)
            if v is None:
                cells.append(f"{'n/a':>10}")
            elif "mom" in c or "dd" in c or "drawdown" in c or "vol" in c:
                if "rank" in c:
                    cells.append(f"{v:10.2f}")
                else:
                    cells.append(f"{v * 100:9.1f}%")
            else:
                cells.append(f"{v:10.2f}")
        cl = clusters.get(sym, "n/a")
        lines.append(f"{sym:<5} " + "  ".join(cells) + f"  {cl}")

    lines.append("")
    gross = sum(abs(p.market_value) for p in portfolio.positions)
    gross_pct = float(gross / portfolio.equity * 100) if portfolio.equity else 0.0
    lines.append(
        f"PORTFOLIO — equity ${portfolio.equity:,.2f}  cash ${portfolio.cash:,.2f}  "
        f"gross {gross_pct:.1f}%"
    )
    lines.append("sym  weight  days_held  cluster")
    for p in portfolio.positions:
        w = float(p.market_value / portfolio.equity * 100) if portfolio.equity else 0
        days = (as_of.date() - p.opened_at).days
        lines.append(
            f"{p.symbol:<5} {w:5.1f}%  {days:9d}  {clusters.get(p.symbol, p.cluster_id)}"
        )
    lines.append("")
    lines.append("CONSTRAINTS (enforced downstream, stated for your awareness)")
    cstr = "  ".join(f"{k} {v}" for k, v in constraints.items())
    lines.append(cstr)
    lines.append(f"RISK STATE: {risk_state}")
    return "\n".join(lines)


def feature_packet_hash(packet: str) -> str:
    return hashlib.sha256(packet.encode()).hexdigest()


def compute_all_features(
    bars: pd.DataFrame,
    symbols: list[str],
    *,
    sectors: dict[str, str],
    news_counts: dict[str, int] | None = None,
    fundamentals: dict[str, dict[str, float | None]] | None = None,
    correlation_threshold: float = 0.70,
) -> tuple[dict[str, dict[str, float | None]], dict[str, str]]:
    reg = FeatureRegistry()
    # Join SPY closes for beta if present
    if "SPY" in symbols or (bars["symbol"] == "SPY").any():
        spy = bars[bars["symbol"] == "SPY"][["session", "adj_close" if "adj_close" in bars.columns else "close"]].copy()
        spy.columns = ["session", "spy_close"]
        bars = bars.merge(spy, on="session", how="left")

    feats = reg.compute_universe(bars, symbols)
    rank_features(feats, "mom_63d", "mom_63d_rank")
    rank_features(feats, "realized_vol_21d", "vol_21d_rank")
    sector_relative_mom(feats, sectors)

    if news_counts:
        for sym, n in news_counts.items():
            if sym in feats:
                feats[sym]["news_count_7d"] = float(n)
    if fundamentals:
        for sym, fdict in fundamentals.items():
            if sym in feats:
                feats[sym].update(fdict)

    # Correlation clusters from returns
    close_col = "adj_close" if "adj_close" in bars.columns else "close"
    pivot = bars.pivot_table(index="session", columns="symbol", values=close_col, aggfunc="last")
    rets = pivot.pct_change().dropna(how="all").tail(126)
    clusters = correlation_clusters(rets, threshold=correlation_threshold) if not rets.empty else {}
    for sym in symbols:
        if sym in feats:
            feats[sym]["cluster_id"] = None  # informational only as float features
    return feats, clusters


def packet_to_json(
    symbol_features: dict[str, dict[str, float | None]],
    clusters: dict[str, str],
) -> str:
    payload = {
        "features": symbol_features,
        "clusters": clusters,
    }
    return json.dumps(payload, sort_keys=True, default=str)
