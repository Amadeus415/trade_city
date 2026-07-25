"""End-to-end pipelines: research backtest stack and paper trading day."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from fund.config import Settings, load_settings, load_universe
from fund.eval.baselines_run import run_baselines_pipeline
from fund.eval.leakage import run_leakage_gate
from fund.eval.report import write_report
from fund.ingest.bars import BarIngestor
from fund.ingest.corporate_actions import CorporateActionsIngestor
from fund.ingest.fundamentals import FundamentalsIngestor
from fund.ingest.news import NewsIngestor
from fund.logging_setup import get_logger
from fund.orchestrate.alerts import send_alert
from fund.orchestrate.daily import decide, execute
from fund.orchestrate.healthcheck import healthcheck
from fund.store.bars import BarStore
from fund.store.fundamentals import FundamentalsStore
from fund.store.journal import Journal
from fund.store.news import NewsStore
from fund.store.universe import UniverseStore

log = get_logger(__name__)


@dataclass
class ResearchResult:
    steps: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    errors: list[str] = field(default_factory=list)


def research_pipeline(
    start: date,
    end: date,
    *,
    config_dir: str = "config",
    provider: str = "yfinance",
    out_dir: str = "reports",
    run_agent: bool = True,
    run_leakage: bool = False,
    monkey_runs: int = 50,
    symbols: list[str] | None = None,
    configs_tried: int = 1,
) -> ResearchResult:
    """Full research path:

    ingest → validate → (optional cross-validate) → baselines →
    agent backtest → report → optional M5 leakage gate
    """
    from fund.backtest.runner import BacktestRunner

    result = ResearchResult()
    settings = load_settings(mode="backtest", config_dir=config_dir)
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    uni = load_universe(config_dir)
    symbols = symbols or list(uni.get("symbols", []))

    # 1. Ingest
    store = BarStore(settings.data_dir)
    ingestor = BarIngestor(
        store,
        symbols,
        bar_lag_minutes=settings.availability.bar_lag_minutes,
        provider=provider,
    )
    n = ingestor.backfill(start, end)
    result.steps["ingest_bars"] = {"rows": n, "provider": provider}
    log.info("pipeline_ingest", rows=n, provider=provider)

    FundamentalsIngestor(FundamentalsStore(settings.data_dir), symbols).backfill(start, end)
    NewsIngestor(NewsStore(settings.data_dir), symbols).backfill(start, end)
    CorporateActionsIngestor(settings.data_dir, symbols).backfill(start, end)
    UniverseStore(settings.data_dir, config_dir).seed_from_allowlist(start)
    result.steps["universe_seeded"] = True

    # 2. Validate
    validation = ingestor.validate()
    result.steps["validate"] = {
        "ok": validation.ok,
        "errors": validation.errors,
        "warnings": validation.warnings,
    }
    if not validation.ok:
        result.ok = False
        result.errors.extend(validation.errors)
        return result

    # 3. Cross-validate real sources only (synthetic has no peer)
    if provider != "synthetic":
        try:
            xv = ingestor.cross_validate(symbols=symbols[:20], start=start, end=end)
            result.steps["cross_validate"] = {
                "ok": xv.ok,
                "errors": xv.errors,
                "warnings": xv.warnings,
            }
            if not xv.ok:
                # Soft fail: warn but continue (adj-close / vendor diffs common)
                result.steps["cross_validate"]["soft_fail"] = True
                log.warning("cross_validate_soft_fail", errors=xv.errors)
        except Exception as e:
            result.steps["cross_validate"] = {"ok": False, "error": str(e)}
    else:
        result.steps["cross_validate"] = {"ok": True, "skipped": "synthetic provider"}

    # 4. Baselines
    baselines = run_baselines_pipeline(
        settings,
        store,
        symbols,
        start,
        end,
        out_dir=out_dir,
        monkey_runs=monkey_runs,
    )
    result.steps["baselines"] = {
        "report": str(baselines.report_path),
        "metrics": baselines.metrics,
    }

    # 5. Agent backtest (default mock provider unless configured)
    if run_agent:
        journal = Journal(settings.journal_path)
        journal.migrate()
        store.preload(symbols + (["SPY"] if "SPY" not in symbols else []))
        runner = BacktestRunner(
            settings,
            store,
            UniverseStore(settings.data_dir, config_dir),
            journal,
            use_agent=True,
            strategy="agent",
        )
        bt = runner.run(start, end)
        report = write_report(
            journal,
            bt.run_id,
            out_dir,
            baselines={k: v for k, v in baselines.curves.items()},
            monkey_percentile=None,
            configs_tried=configs_tried,
        )
        # Attach monkey percentile
        from fund.eval.baselines_run import agent_vs_baselines

        vs = agent_vs_baselines(bt.equity_curve, baselines)
        result.steps["agent"] = {
            "run_id": bt.run_id,
            "decisions": bt.decisions,
            "orders": bt.orders,
            "fills": bt.fills,
            "final_equity": float(bt.equity_curve.iloc[-1]) if len(bt.equity_curve) else None,
            "report": str(report),
            "vs_baselines": vs,
        }
        journal.close()

    # 6. Optional leakage gate
    if run_leakage:
        leak = run_leakage_gate(
            start=start,
            end=end,
            config_dir=config_dir,
            out_dir=out_dir,
            monkey_runs=min(monkey_runs, 50),
            configs_tried=configs_tried,
        )
        result.steps["leakage"] = {
            "gate_pass": leak.gate_pass,
            "notes": leak.gate_notes,
            "report": str(leak.report_path) if leak.report_path else None,
            "modes": {
                m: {
                    "final_equity": info.get("final_equity"),
                    "monkey_percentile": (info.get("vs_baselines") or {}).get(
                        "monkey_percentile"
                    ),
                }
                for m, info in leak.modes.items()
            },
        }
        if leak.gate_pass is False:
            result.ok = False
            result.errors.append("M5 leakage gate failed")

    result.steps["healthcheck"] = healthcheck(settings.journal_path)
    return result


def paper_day(
    *,
    config_dir: str = "config",
    provider: str = "yfinance",
    dry_run_execute: bool = True,
    skip_execute: bool = False,
) -> dict[str, Any]:
    """One paper-trading day: ingest → validate → decide → (optional) execute dry-run."""
    settings = load_settings(mode="paper", config_dir=config_dir)
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    uni = load_universe(config_dir)
    symbols = list(uni.get("symbols", []))
    out: dict[str, Any] = {}

    # Health first
    hc = healthcheck(settings.journal_path)
    out["healthcheck"] = hc
    if hc.get("risk_state") == "HALTED":
        send_alert(
            settings.alerting.webhook_url,
            "Risk HALTED",
            "Paper day aborted: risk state is HALTED",
            severity="critical",
        )
        out["status"] = "aborted_halted"
        return out

    if not hc.get("is_trading_day"):
        out["status"] = "noop_non_trading_day"
        return out

    store = BarStore(settings.data_dir)
    ingestor = BarIngestor(
        store,
        symbols,
        bar_lag_minutes=settings.availability.bar_lag_minutes,
        provider=provider,
    )
    out["ingest"] = {"rows": ingestor.incremental()}
    val = ingestor.validate()
    out["validate"] = {"ok": val.ok, "errors": val.errors, "warnings": val.warnings}
    if not val.ok:
        send_alert(
            settings.alerting.webhook_url,
            "Ingest validation failed",
            "; ".join(val.errors),
            severity="error",
        )
        out["status"] = "validate_failed"
        return out

    FundamentalsIngestor(FundamentalsStore(settings.data_dir), symbols).incremental()
    NewsIngestor(NewsStore(settings.data_dir), symbols).incremental()

    decision = decide(settings, dry_run=False)
    out["decide"] = decision

    if not skip_execute:
        ex = execute(settings, dry_run=dry_run_execute)
        out["execute"] = ex

    # Alert on risk state change snapshot
    j = Journal(settings.journal_path)
    j.migrate()
    state = j.get_risk_state()
    j.close()
    out["risk_state"] = state
    if state["state"] != "NORMAL":
        send_alert(
            settings.alerting.webhook_url,
            f"Risk state {state['state']}",
            f"trigger={state.get('trigger_metric')} value={state.get('trigger_value')}",
            severity="warning",
        )

    out["status"] = "ok"
    return out
