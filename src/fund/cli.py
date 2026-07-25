"""Single CLI entrypoint — cron invokes these subcommands."""

from __future__ import annotations

import sys
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import click

from fund.logging_setup import configure_logging, get_logger


@click.group()
@click.option("--log-level", default="INFO")
def main(log_level: str) -> None:
    configure_logging(log_level)


@main.group()
def ingest() -> None:
    """Data ingestion."""


@ingest.command("incremental")
@click.option("--config-dir", default="config")
@click.option("--mode", default="paper")
@click.option(
    "--provider",
    default="yfinance",
    type=click.Choice(["yfinance", "stooq", "synthetic"]),
)
def ingest_incremental(config_dir: str, mode: str, provider: str) -> None:
    from fund.config import load_settings, load_universe
    from fund.ingest.bars import BarIngestor
    from fund.ingest.corporate_actions import CorporateActionsIngestor
    from fund.ingest.fundamentals import FundamentalsIngestor
    from fund.ingest.news import NewsIngestor
    from fund.store.bars import BarStore
    from fund.store.fundamentals import FundamentalsStore
    from fund.store.news import NewsStore

    settings = load_settings(mode=mode, config_dir=config_dir)
    uni = load_universe(config_dir)
    symbols = uni.get("symbols", [])
    log = get_logger("ingest")
    n = BarIngestor(
        BarStore(settings.data_dir),
        symbols,
        bar_lag_minutes=settings.availability.bar_lag_minutes,
        provider=provider,
    ).incremental()
    log.info("bars_incremental", rows=n, provider=provider)
    n = FundamentalsIngestor(FundamentalsStore(settings.data_dir), symbols).incremental()
    log.info("fundamentals_incremental", rows=n)
    n = NewsIngestor(NewsStore(settings.data_dir), symbols).incremental()
    log.info("news_incremental", rows=n)
    CorporateActionsIngestor(settings.data_dir, symbols).incremental()


@ingest.command("backfill")
@click.option("--start", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--config-dir", default="config")
@click.option("--mode", default="backtest")
@click.option(
    "--provider",
    default="yfinance",
    type=click.Choice(["yfinance", "stooq", "synthetic"]),
)
def ingest_backfill(
    start: datetime, end: datetime, config_dir: str, mode: str, provider: str
) -> None:
    from fund.config import load_settings, load_universe
    from fund.ingest.bars import BarIngestor
    from fund.ingest.fundamentals import FundamentalsIngestor
    from fund.store.bars import BarStore
    from fund.store.fundamentals import FundamentalsStore
    from fund.store.universe import UniverseStore

    settings = load_settings(mode=mode, config_dir=config_dir)
    uni = load_universe(config_dir)
    symbols = uni.get("symbols", [])
    log = get_logger("ingest")
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    n = BarIngestor(
        BarStore(settings.data_dir),
        symbols,
        bar_lag_minutes=settings.availability.bar_lag_minutes,
        provider=provider,
    ).backfill(start.date(), end.date())
    log.info("bars_backfill", rows=n, provider=provider)
    FundamentalsIngestor(FundamentalsStore(settings.data_dir), symbols).backfill(
        start.date(), end.date()
    )
    UniverseStore(settings.data_dir, config_dir).seed_from_allowlist(start.date())
    log.info("universe_seeded")


@ingest.command("validate")
@click.option("--config-dir", default="config")
@click.option("--mode", default="paper")
def ingest_validate(config_dir: str, mode: str) -> None:
    from fund.config import load_settings, load_universe
    from fund.ingest.bars import BarIngestor
    from fund.store.bars import BarStore

    settings = load_settings(mode=mode, config_dir=config_dir)
    symbols = load_universe(config_dir).get("symbols", [])
    result = BarIngestor(BarStore(settings.data_dir), symbols).validate()
    for w in result.warnings:
        click.echo(f"WARN: {w}")
    for e in result.errors:
        click.echo(f"ERROR: {e}")
    if not result.ok:
        sys.exit(1)
    click.echo("validation ok")


@ingest.command("cross-validate")
@click.option("--config-dir", default="config")
@click.option("--mode", default="backtest")
@click.option("--secondary", default="stooq")
@click.option("--sample", default=20, type=int)
@click.option("--max-diff-pct", default=0.5, type=float)
def ingest_cross_validate(
    config_dir: str, mode: str, secondary: str, sample: int, max_diff_pct: float
) -> None:
    """Compare primary bars vs secondary source (spec: fail if >0.5% close gap)."""
    from fund.config import load_settings, load_universe
    from fund.ingest.bars import BarIngestor
    from fund.store.bars import BarStore

    settings = load_settings(mode=mode, config_dir=config_dir)
    symbols = load_universe(config_dir).get("symbols", [])
    result = BarIngestor(BarStore(settings.data_dir), symbols).cross_validate(
        symbols=symbols,
        secondary=secondary,
        max_close_diff_pct=max_diff_pct,
        sample_n=sample,
    )
    for w in result.warnings:
        click.echo(f"WARN: {w}")
    for e in result.errors:
        click.echo(f"ERROR: {e}")
    if not result.ok:
        sys.exit(1)
    click.echo("cross-validation ok")


@main.command()
@click.option("--mode", type=click.Choice(["backtest", "paper", "live"]), default="paper")
@click.option("--as-of", "as_of", default=None)
@click.option("--config-dir", default="config")
@click.option("--dry-run", is_flag=True)
def decide(mode: str, as_of: str | None, config_dir: str, dry_run: bool) -> None:
    from fund.config import load_settings
    from fund.orchestrate.daily import decide as do_decide

    settings = load_settings(mode=mode, config_dir=config_dir)
    ts = datetime.fromisoformat(as_of) if as_of else None
    result = do_decide(settings, as_of=ts, dry_run=dry_run)
    click.echo(result)


@main.command()
@click.option("--mode", type=click.Choice(["paper", "live"]), default="paper")
@click.option("--config-dir", default="config")
@click.option("--dry-run", is_flag=True)
def execute(mode: str, config_dir: str, dry_run: bool) -> None:
    from fund.config import load_settings
    from fund.orchestrate.daily import execute as do_execute
    from fund.store.journal import Journal

    settings = load_settings(mode=mode, config_dir=config_dir)
    result = do_execute(settings, dry_run=dry_run)
    click.echo(result)
    if dry_run:
        # exit non-zero if any recent risk rejection
        j = Journal(settings.journal_path)
        j.migrate()
        rejs = j.risk_rejections()
        j.close()
        if rejs:
            click.echo(f"dry-run: {len(rejs)} risk rejections present", err=True)
            sys.exit(1)


@main.command()
@click.option("--mode", type=click.Choice(["paper", "live", "backtest"]), default="paper")
@click.option("--config-dir", default="config")
def reconcile(mode: str, config_dir: str) -> None:
    from datetime import timedelta

    from fund.config import load_settings
    from fund.execution.reconcile import reconcile as do_reconcile
    from fund.orchestrate.daily import _broker_for_mode
    from fund.store.journal import Journal

    settings = load_settings(mode=mode, config_dir=config_dir)
    journal = Journal(settings.journal_path)
    journal.migrate()
    broker = _broker_for_mode(settings, journal)
    result = do_reconcile(
        journal, broker, since=datetime.now() - timedelta(days=7)
    )
    journal.close()
    click.echo(result)


@main.command()
@click.option("--config-dir", default="config")
@click.option("--mode", default="paper")
def healthcheck(config_dir: str, mode: str) -> None:
    from fund.config import load_settings
    from fund.orchestrate.healthcheck import healthcheck as do_health

    settings = load_settings(mode=mode, config_dir=config_dir)
    result = do_health(settings.journal_path)
    click.echo(result)
    if not result.get("ok"):
        sys.exit(1)


@main.group()
def risk() -> None:
    """Risk state controls."""


@risk.command("status")
@click.option("--config-dir", default="config")
@click.option("--mode", default="paper")
def risk_status(config_dir: str, mode: str) -> None:
    from fund.config import load_settings
    from fund.store.journal import Journal

    settings = load_settings(mode=mode, config_dir=config_dir)
    j = Journal(settings.journal_path)
    j.migrate()
    click.echo(j.get_risk_state())
    j.close()


@risk.command("reset")
@click.option("--confirm-equity", required=True, type=str)
@click.option("--config-dir", default="config")
@click.option("--mode", default="paper")
def risk_reset(confirm_equity: str, config_dir: str, mode: str) -> None:
    from fund.config import load_settings
    from fund.orchestrate.daily import _broker_for_mode
    from fund.risk.state import KillSwitch
    from fund.store.journal import Journal

    settings = load_settings(mode=mode, config_dir=config_dir)
    j = Journal(settings.journal_path)
    j.migrate()
    broker = _broker_for_mode(settings, j)
    actual = broker.get_portfolio().equity
    KillSwitch(j).reset(Decimal(confirm_equity), actual)
    click.echo({"status": "reset", "equity": str(actual)})
    j.close()


@main.group()
def backtest() -> None:
    """Backtest runner."""


@backtest.command("run")
@click.option("--config", "config_path", default="config/backtest.yaml")
@click.option("--start", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--strategy", default="agent", type=click.Choice(["agent", "momentum", "equal_weight"]))
@click.option("--config-dir", default="config")
def backtest_run(
    config_path: str,
    start: datetime,
    end: datetime,
    strategy: str,
    config_dir: str,
) -> None:
    from fund.backtest.runner import BacktestRunner
    from fund.config import load_settings
    from fund.store.bars import BarStore
    from fund.store.journal import Journal
    from fund.store.universe import UniverseStore

    settings = load_settings(mode="backtest", config_dir=config_dir)
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    journal = Journal(settings.journal_path)
    journal.migrate()
    runner = BacktestRunner(
        settings,
        BarStore(settings.data_dir),
        UniverseStore(settings.data_dir, config_dir),
        journal,
        use_agent=(strategy == "agent"),
        strategy=strategy,
    )
    result = runner.run(start.date(), end.date())
    click.echo(
        {
            "run_id": result.run_id,
            "decisions": result.decisions,
            "orders": result.orders,
            "fills": result.fills,
            "final_equity": float(result.equity_curve.iloc[-1])
            if len(result.equity_curve)
            else None,
        }
    )
    journal.close()


@main.group()
def eval() -> None:
    """Evaluation reports."""


@eval.command("report")
@click.option("--run", "run_id", required=True)
@click.option("--out", "out_dir", default="reports")
@click.option("--config-dir", default="config")
@click.option("--mode", default="backtest")
def eval_report(run_id: str, out_dir: str, config_dir: str, mode: str) -> None:
    from fund.config import load_settings
    from fund.eval.report import write_report
    from fund.store.journal import Journal

    settings = load_settings(mode=mode, config_dir=config_dir)
    j = Journal(settings.journal_path)
    j.migrate()
    path = write_report(j, run_id, out_dir)
    click.echo({"report": str(path)})
    j.close()


@eval.command("baselines")
@click.option("--start", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--out", "out_dir", default="reports")
@click.option("--config-dir", default="config")
@click.option("--monkey-runs", default=None, type=int)
def eval_baselines(
    start: datetime,
    end: datetime,
    out_dir: str,
    config_dir: str,
    monkey_runs: int | None,
) -> None:
    """Run SPY / equal-weight / momentum / random-monkey baselines."""
    from fund.config import load_settings, load_universe
    from fund.eval.baselines_run import run_baselines_pipeline
    from fund.store.bars import BarStore

    settings = load_settings(mode="backtest", config_dir=config_dir)
    symbols = load_universe(config_dir).get("symbols", [])
    bundle = run_baselines_pipeline(
        settings,
        BarStore(settings.data_dir),
        symbols,
        start.date(),
        end.date(),
        out_dir=out_dir,
        monkey_runs=monkey_runs,
    )
    click.echo(
        {
            "report": str(bundle.report_path),
            "metrics": {k: v.get("sharpe") for k, v in bundle.metrics.items()},
        }
    )


@eval.command("leakage")
@click.option("--start", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--out", "out_dir", default="reports")
@click.option("--config-dir", default="config")
@click.option("--monkey-runs", default=50, type=int)
@click.option("--configs-tried", default=1, type=int)
def eval_leakage(
    start: datetime,
    end: datetime,
    out_dir: str,
    config_dir: str,
    monkey_runs: int,
    configs_tried: int,
) -> None:
    """M5 leakage gate: agent under all four masking modes."""
    from fund.eval.leakage import run_leakage_gate

    result = run_leakage_gate(
        start=start.date(),
        end=end.date(),
        config_dir=config_dir,
        out_dir=out_dir,
        monkey_runs=monkey_runs,
        configs_tried=configs_tried,
    )
    click.echo(
        {
            "gate_pass": result.gate_pass,
            "notes": result.gate_notes,
            "report": str(result.report_path) if result.report_path else None,
            "modes": {
                m: info.get("final_equity") for m, info in result.modes.items()
            },
        }
    )
    if result.gate_pass is False:
        sys.exit(2)


@main.group()
def pipeline() -> None:
    """End-to-end research and paper-day pipelines."""


@pipeline.command("research")
@click.option("--start", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--config-dir", default="config")
@click.option(
    "--provider",
    default="yfinance",
    type=click.Choice(["yfinance", "stooq", "synthetic"]),
)
@click.option("--out", "out_dir", default="reports")
@click.option("--no-agent", is_flag=True, help="Baselines only")
@click.option("--leakage", is_flag=True, help="Also run M5 leakage gate")
@click.option("--monkey-runs", default=50, type=int)
@click.option("--configs-tried", default=1, type=int)
@click.option("--symbols", default=None, help="Comma-separated subset for faster runs")
def pipeline_research(
    start: datetime,
    end: datetime,
    config_dir: str,
    provider: str,
    out_dir: str,
    no_agent: bool,
    leakage: bool,
    monkey_runs: int,
    configs_tried: int,
    symbols: str | None,
) -> None:
    """Ingest → validate → baselines → agent backtest → report [→ leakage]."""
    from fund.orchestrate.pipeline import research_pipeline

    sym_list = [s.strip() for s in symbols.split(",")] if symbols else None
    result = research_pipeline(
        start.date(),
        end.date(),
        config_dir=config_dir,
        provider=provider,
        out_dir=out_dir,
        run_agent=not no_agent,
        run_leakage=leakage,
        monkey_runs=monkey_runs,
        symbols=sym_list,
        configs_tried=configs_tried,
    )
    click.echo(
        {
            "ok": result.ok,
            "errors": result.errors,
            "steps": {
                k: (v if not isinstance(v, dict) or "metrics" not in v else {
                    **{kk: vv for kk, vv in v.items() if kk != "metrics"},
                    "sharpes": {
                        n: m.get("sharpe") for n, m in v.get("metrics", {}).items()
                    },
                })
                for k, v in result.steps.items()
            },
        }
    )
    if not result.ok:
        sys.exit(1)


@pipeline.command("paper-day")
@click.option("--config-dir", default="config")
@click.option(
    "--provider",
    default="yfinance",
    type=click.Choice(["yfinance", "stooq", "synthetic"]),
)
@click.option("--live-execute", is_flag=True, help="Actually send (still simulated in paper)")
@click.option("--skip-execute", is_flag=True)
def pipeline_paper_day(
    config_dir: str, provider: str, live_execute: bool, skip_execute: bool
) -> None:
    """Single paper session: ingest → validate → decide → execute(--dry-run)."""
    from fund.orchestrate.pipeline import paper_day

    result = paper_day(
        config_dir=config_dir,
        provider=provider,
        dry_run_execute=not live_execute,
        skip_execute=skip_execute,
    )
    click.echo(result)
    if result.get("status") not in ("ok", "noop_non_trading_day"):
        sys.exit(1)


if __name__ == "__main__":
    main()
