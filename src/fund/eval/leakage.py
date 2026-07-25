"""M5 leakage gate — run agent backtest under all four masking modes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from fund.backtest.runner import BacktestResult, BacktestRunner
from fund.config import Settings, load_settings
from fund.eval.baselines_run import (
    agent_vs_baselines,
    run_baselines_pipeline,
)
from fund.eval.metrics import compute_metrics
from fund.store.bars import BarStore
from fund.store.journal import Journal
from fund.store.universe import UniverseStore
from fund.types import MaskingMode


@dataclass
class LeakageResult:
    modes: dict[str, dict[str, Any]] = field(default_factory=dict)
    report_path: Path | None = None
    gate_pass: bool | None = None
    gate_notes: list[str] = field(default_factory=list)


def run_leakage_gate(
    *,
    start: date,
    end: date,
    config_dir: str = "config",
    out_dir: str | Path = "reports",
    modes: list[str] | None = None,
    configs_tried: int = 1,
    monkey_runs: int = 50,
) -> LeakageResult:
    """Run agent backtest once per masking mode and compare.

    Gate (from plan):
    - If blinded collapses to near-random while bright looks excellent → memorisation.
    - Agent must exceed ~90th monkey percentile in blinded mode to proceed.
    """
    modes = modes or [m.value for m in MaskingMode]
    base = load_settings(mode="backtest", config_dir=config_dir)
    Path(base.data_dir).mkdir(parents=True, exist_ok=True)

    journal = Journal(base.journal_path)
    journal.migrate()
    store = BarStore(base.data_dir)
    universe = UniverseStore(base.data_dir, config_dir)
    symbols = universe.static_allowlist()

    # Baselines once for the window
    baselines = run_baselines_pipeline(
        base,
        store,
        symbols,
        start,
        end,
        out_dir=out_dir,
        monkey_runs=monkey_runs,
    )

    result = LeakageResult()
    curves: dict[str, pd.Series] = {}

    for mode in modes:
        settings = load_settings(
            mode="backtest",
            config_dir=config_dir,
            overrides={"agent": {"masking_mode": mode, "provider": base.agent.provider}},
        )
        runner = BacktestRunner(
            settings,
            store,
            universe,
            journal,
            use_agent=True,
            strategy="agent",
        )
        # Preload once for speed
        store.preload(symbols + (["SPY"] if "SPY" not in symbols else []))
        bt: BacktestResult = runner.run(start, end)
        eq = bt.equity_curve
        curves[mode] = eq
        metrics = compute_metrics(eq).as_dict() if len(eq) else {}
        vs = agent_vs_baselines(eq, baselines) if len(eq) else {}
        result.modes[mode] = {
            "run_id": bt.run_id,
            "decisions": bt.decisions,
            "orders": bt.orders,
            "fills": bt.fills,
            "final_equity": float(eq.iloc[-1]) if len(eq) else None,
            "metrics": metrics,
            "vs_baselines": vs,
        }

    # Gate logic
    notes: list[str] = []
    bright = result.modes.get("bright", {})
    blinded = result.modes.get("blinded", {})
    bright_eq = bright.get("final_equity")
    blinded_eq = blinded.get("final_equity")
    start_cash = float(base.backtest.start_cash)

    if bright_eq and blinded_eq and start_cash:
        bright_ret = bright_eq / start_cash - 1
        blinded_ret = blinded_eq / start_cash - 1
        if bright_ret > 0.05 and abs(blinded_ret) < 0.02:
            notes.append(
                "FAIL: bright looks profitable but blinded is near-flat — "
                "alpha may be memorisation/leakage. Do not proceed to live."
            )
            result.gate_pass = False
        blinded_pct = (blinded.get("vs_baselines") or {}).get("monkey_percentile")
        if blinded_pct is not None and blinded_pct < 90:
            notes.append(
                f"FAIL: blinded monkey percentile {blinded_pct:.1f} < 90. "
                "Agent has not demonstrated skill under leakage control."
            )
            result.gate_pass = False
        if result.gate_pass is None:
            if blinded_pct is not None and blinded_pct >= 90:
                notes.append(
                    f"PASS candidate: blinded monkey percentile {blinded_pct:.1f} ≥ 90. "
                    "Still review attribution before paper capital."
                )
                result.gate_pass = True
            else:
                notes.append("INCONCLUSIVE: insufficient equity history for hard gate.")
                result.gate_pass = None
    else:
        notes.append("INCONCLUSIVE: missing bright/blinded equity curves.")
        result.gate_pass = None

    notes.append(f"Configs tried before this result: {configs_tried}")
    result.gate_notes = notes

    # Report
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{start.isoformat()}_{end.isoformat()}"
    chart = out / f"leakage_{tag}.png"
    fig, ax = plt.subplots(figsize=(11, 5))
    for mode, series in curves.items():
        if len(series):
            series.plot(ax=ax, label=mode)
    for name, series in baselines.curves.items():
        series.plot(ax=ax, label=name, alpha=0.5, linestyle="--")
    ax.legend(fontsize=8)
    ax.set_title(f"M5 Leakage gate {start} → {end}")
    fig.tight_layout()
    fig.savefig(chart, dpi=120)
    plt.close(fig)

    lines = [
        f"# M5 Leakage Gate — {start} → {end}",
        "",
        f"**Gate pass:** `{result.gate_pass}`",
        "",
        f"![leakage]({chart.name})",
        "",
        "## Notes",
        "",
    ]
    for n in notes:
        lines.append(f"- {n}")
    lines += ["", "## Per-mode results", ""]
    for mode, info in result.modes.items():
        lines.append(f"### `{mode}`")
        lines.append(f"- run_id: `{info['run_id']}`")
        lines.append(f"- final equity: {info.get('final_equity')}")
        lines.append(f"- decisions/orders/fills: {info['decisions']}/{info['orders']}/{info['fills']}")
        sharpe = (info.get("metrics") or {}).get("sharpe")
        if sharpe is not None:
            lines.append(f"- sharpe: {sharpe:.4f}")
        vs = info.get("vs_baselines") or {}
        if vs:
            lines.append(f"- monkey percentile: {vs.get('monkey_percentile')}")
            lines.append(f"- ranking: {vs.get('ranking')}")
        lines.append("")

    report_path = out / f"leakage_{tag}.md"
    report_path.write_text("\n".join(lines))
    result.report_path = report_path
    journal.close()
    return result
