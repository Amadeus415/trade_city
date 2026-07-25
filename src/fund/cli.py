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
def ingest_incremental(config_dir: str, mode: str) -> None:
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
    n = BarIngestor(BarStore(settings.data_dir), symbols).incremental()
    log.info("bars_incremental", rows=n)
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
def ingest_backfill(start: datetime, end: datetime, config_dir: str, mode: str) -> None:
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
    n = BarIngestor(BarStore(settings.data_dir), symbols).backfill(
        start.date(), end.date()
    )
    log.info("bars_backfill", rows=n)
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


if __name__ == "__main__":
    main()
