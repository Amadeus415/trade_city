"""Markdown evaluation report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from fund.eval.attribution import AttributionResult
from fund.eval.metrics import Metrics, compute_metrics
from fund.store.journal import Journal


def write_report(
    journal: Journal,
    run_id: str,
    out_dir: str | Path,
    *,
    baselines: dict[str, pd.Series] | None = None,
    monkey_percentile: float | None = None,
    attribution: AttributionResult | None = None,
    configs_tried: int = 1,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = journal.equity_curve(run_id)
    if not rows:
        raise ValueError(f"no equity snapshots for run {run_id}")

    sessions = [r["session"] for r in rows]
    equity = pd.Series(
        [float(r["equity"]) for r in rows],
        index=pd.to_datetime(sessions),
        name="agent",
    )
    metrics = compute_metrics(equity)

    # Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    equity.plot(ax=ax, label="agent")
    if baselines:
        for name, series in baselines.items():
            if name == "monkey_runs":
                continue
            s = series.copy()
            s.index = pd.to_datetime(s.index)
            # align
            s = s.reindex(equity.index, method="ffill")
            s.plot(ax=ax, label=name, alpha=0.8)
    ax.legend()
    ax.set_title(f"Equity curves — run {run_id[:8]}")
    ax.set_ylabel("Equity ($)")
    chart_path = out / f"equity_{run_id[:8]}.png"
    fig.tight_layout()
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)

    proposals = journal.proposals_for_run(run_id)
    rejections = journal.risk_rejections(run_id)
    # Top theses by confidence among accepted
    accepted = [p for p in proposals if p.get("verdict") in ("accepted", "clamped")]
    accepted_sorted = sorted(
        accepted, key=lambda p: float(p.get("confidence") or 0), reverse=True
    )[:20]

    reject_counts: dict[str, int] = {}
    for r in rejections:
        import json

        reasons = json.loads(r["reasons"]) if isinstance(r["reasons"], str) else r["reasons"]
        for reason in reasons:
            reject_counts[reason] = reject_counts.get(reason, 0) + 1

    lines = [
        f"# Evaluation Report — `{run_id}`",
        "",
        f"**Configs tried before this result:** {configs_tried}",
        "",
        f"![equity]({chart_path.name})",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for k, v in metrics.as_dict().items():
        if v is None:
            continue
        if isinstance(v, float):
            lines.append(f"| {k} | {v:.4f} |")
        else:
            lines.append(f"| {k} | {v} |")

    if monkey_percentile is not None:
        lines += [
            "",
            f"**Random-monkey percentile:** {monkey_percentile:.1f}th "
            f"(gate: ≥ 90th in blinded mode)",
            "",
        ]

    if attribution:
        lines += [
            "## Attribution",
            "",
            f"- Market beta: {attribution.market_beta:.3f}",
            f"- Market R²: {attribution.market_explained:.3f}",
            f"- Style R²: {attribution.style_explained:.3f}",
            f"- Residual alpha (ann.): {attribution.residual_alpha_ann:.3%}",
            f"- Residual (1−R²): {attribution.residual_r2:.3f}",
            "",
        ]

    lines += ["## Top decisions by confidence", ""]
    for p in accepted_sorted:
        lines.append(
            f"- **{p['symbol']}** `{p['action']}` w={p.get('final_weight') or p['target_weight']} "
            f"conf={p['confidence']}: {p['thesis'][:200]}"
        )

    lines += ["", "## Risk rejections by reason", ""]
    if reject_counts:
        for reason, n in sorted(reject_counts.items(), key=lambda x: -x[1]):
            lines.append(f"- `{reason}`: {n}")
    else:
        lines.append("_none_")

    lines += [
        "",
        "## Honest notes",
        "",
        "- Success is not 'makes money'. Measure against SPY, equal-weight, momentum, and monkey.",
        "- If blinded mode collapses vs bright, alpha may be memorisation.",
        f"- Rolling 6m Sharpe samples: {len(metrics.rolling_sharpe_6m)}",
        "",
    ]

    report_path = out / f"report_{run_id[:8]}.md"
    report_path.write_text("\n".join(lines))
    return report_path
