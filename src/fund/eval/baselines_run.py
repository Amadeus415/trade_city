"""Run all four mandatory baselines + metrics report from stored bars."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from fund.backtest.baselines import monkey_percentile, run_all_baselines
from fund.config import Settings
from fund.eval.metrics import compute_metrics
from fund.store.bars import BarStore

ET = ZoneInfo("America/New_York")


@dataclass
class BaselineBundle:
    prices: pd.DataFrame
    curves: dict[str, pd.Series]
    monkey_runs: pd.DataFrame
    metrics: dict[str, dict[str, Any]]
    report_path: Path


def load_price_matrix(
    store: BarStore,
    symbols: list[str],
    start: date,
    end: date,
    lag_minutes: int = 15,
) -> pd.DataFrame:
    from datetime import timedelta

    as_of = datetime.combine(end, time(16, 0), tzinfo=ET) + timedelta(minutes=lag_minutes)
    # ensure SPY present for B&H
    syms = list(dict.fromkeys(symbols + (["SPY"] if "SPY" not in symbols else [])))
    store.preload(syms)
    return store.price_matrix(syms, as_of=as_of, start=start, end=end)


def run_baselines_pipeline(
    settings: Settings,
    store: BarStore,
    symbols: list[str],
    start: date,
    end: date,
    out_dir: str | Path = "reports",
    *,
    monkey_runs: int | None = None,
    seed: int | None = None,
) -> BaselineBundle:
    prices = load_price_matrix(
        store,
        symbols,
        start,
        end,
        lag_minutes=settings.availability.bar_lag_minutes,
    )
    if prices.empty or len(prices) < 20:
        raise ValueError(
            f"insufficient price history for baselines ({len(prices)} sessions). "
            "Run `fund ingest backfill` first."
        )

    cash = float(settings.backtest.start_cash)
    n_monkey = monkey_runs if monkey_runs is not None else settings.backtest.monkey_runs
    rng_seed = seed if seed is not None else settings.backtest.random_seed

    result = run_all_baselines(
        prices,
        start_cash=cash,
        monkey_runs=n_monkey,
        seed=rng_seed,
    )
    monkey_df = result.pop("monkey_runs")
    curves = {k: v for k, v in result.items()}

    metrics = {name: compute_metrics(curve).as_dict() for name, curve in curves.items()}

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{start.isoformat()}_{end.isoformat()}"
    chart = out / f"baselines_{tag}.png"
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, curve in curves.items():
        curve.plot(ax=ax, label=name)
    ax.legend()
    ax.set_title(f"Baselines {start} → {end}")
    ax.set_ylabel("Equity ($)")
    fig.tight_layout()
    fig.savefig(chart, dpi=120)
    plt.close(fig)

    lines = [
        f"# Baseline Report — {start} → {end}",
        "",
        f"Start cash: ${cash:,.2f}",
        f"Sessions: {len(prices)}",
        f"Symbols: {len(prices.columns)}",
        f"Monkey runs: {n_monkey}",
        "",
        f"![baselines]({chart.name})",
        "",
        "## Metrics",
        "",
    ]
    # Metric table header
    names = list(curves.keys())
    lines.append("| Metric | " + " | ".join(names) + " |")
    lines.append("|---|" + "|".join(["---"] * len(names)) + "|")
    keys = [
        "cagr",
        "total_return",
        "volatility",
        "sharpe",
        "max_drawdown",
        "calmar",
    ]
    for k in keys:
        cells = []
        for n in names:
            v = metrics[n].get(k)
            cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append(f"| {k} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Gate notes",
        "",
        "- SPY CAGR should land in a historically plausible range for the window.",
        "- Cost model is embedded in equal_weight / momentum / monkey (10 bps turnover).",
        "- Agent must later beat momentum and clear ~90th monkey percentile in **blinded** mode.",
        "",
    ]
    report_path = out / f"baselines_{tag}.md"
    report_path.write_text("\n".join(lines))

    # Persist curves for leakage / agent comparison
    curves_path = out / f"baselines_curves_{tag}.parquet"
    pd.DataFrame(curves).to_parquet(curves_path)

    return BaselineBundle(
        prices=prices,
        curves=curves,
        monkey_runs=monkey_df,
        metrics=metrics,
        report_path=report_path,
    )


def agent_vs_baselines(
    agent_equity: pd.Series,
    bundle: BaselineBundle,
) -> dict[str, Any]:
    """Monkey percentile + simple ranking vs baselines."""
    agent = agent_equity.astype(float).dropna()
    if agent.empty:
        return {}
    final = float(agent.iloc[-1])
    pct = monkey_percentile(final, bundle.monkey_runs)
    rank = {}
    for name, curve in bundle.curves.items():
        c = curve.reindex(agent.index, method="ffill").dropna()
        if len(c):
            rank[name] = float(c.iloc[-1])
    rank["agent"] = final
    ordered = sorted(rank.items(), key=lambda x: x[1], reverse=True)
    return {
        "monkey_percentile": pct,
        "final_equity": rank,
        "ranking": [n for n, _ in ordered],
        "beats_momentum": final > rank.get("momentum", 0),
        "beats_spy": final > rank.get("spy_bh", 0),
        "above_90th_monkey": pct >= 90.0,
    }
