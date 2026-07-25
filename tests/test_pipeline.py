"""End-to-end research pipeline and baselines (synthetic data)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from fund.config import load_settings
from fund.eval.baselines_run import run_baselines_pipeline
from fund.ingest.bars import BarIngestor
from fund.orchestrate.pipeline import research_pipeline
from fund.store.bars import BarStore
from fund.store.universe import UniverseStore


def test_research_pipeline_synthetic(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "data"
    cfg = tmp_path / "config"
    cfg.mkdir()
    # Minimal universe + base config pointing at tmp data
    (cfg / "universe.yaml").write_text(
        "symbols: [AAA, BBB, SPY]\nsectors: {AAA: tech, BBB: tech, SPY: etf}\n"
    )
    (cfg / "base.yaml").write_text(
        f"""
mode: backtest
data_dir: {data_dir}
journal_path: {data_dir / "journal.db"}
initial_cash: "10000.00"
availability:
  bar_lag_minutes: 15
risk:
  limits_version: "1.0.0"
  account:
    max_gross_exposure: 0.95
    min_cash_buffer_pct: 0.05
    allow_short: false
  position:
    max_position_weight: 0.10
    max_new_position_weight: 0.05
    max_positions: 15
    min_position_notional: 50
    max_add_per_cycle_weight: 0.02
  concentration:
    max_sector_weight: 0.50
    max_cluster_weight: 0.50
    correlation_threshold: 0.70
  turnover:
    max_daily_turnover: 0.50
    max_orders_per_day: 10
    min_holding_days: 1
    max_round_trips_per_week: 10
  loss_limits:
    daily_loss_soft_pct: 0.05
    daily_loss_hard_pct: 0.10
    weekly_loss_hard_pct: 0.20
    max_drawdown_halt_pct: 0.50
    consecutive_loss_days_halt: 20
  order:
    limit_orders_only: true
    max_limit_offset_bps: 50
    max_spread_bps: 100
    max_pct_of_adv: 0.1
    reject_if_halted: true
    no_trade_first_minutes: 0
    no_trade_last_minutes: 0
  universe:
    allowlist_only: true
    min_price: 1
    max_price: 10000
    min_avg_dollar_volume_21d: 1
    min_days_listed: 1
    exclude_earnings_within_days: 0
    blocklist: []
agent:
  provider: mock
  analyst_model: mock
  pm_model: mock
  temperature: 0.0
  masking_mode: bright
  cache_enabled: false
costs:
  commission_per_share: "0.01"
  default_spread_bps_liquid: 5
  default_spread_bps_illiquid: 15
  slippage_base_bps: 5
  slippage_size_coeff: 10
  limit_nonfill_rate: 0.0
  sec_fee_bps: "0.08"
  finra_taf_per_share: "0.000166"
backtest:
  start_cash: "10000.00"
  monkey_runs: 10
  random_seed: 1
alerting:
  webhook_url: null
"""
    )
    (cfg / "backtest.yaml").write_text("mode: backtest\n")

    out = tmp_path / "reports"
    result = research_pipeline(
        date(2024, 1, 2),
        date(2024, 3, 29),
        config_dir=str(cfg),
        provider="synthetic",
        out_dir=str(out),
        run_agent=True,
        run_leakage=False,
        monkey_runs=10,
        symbols=["AAA", "BBB", "SPY"],
    )
    assert result.ok, result.errors
    assert "baselines" in result.steps
    assert result.steps["baselines"]["report"]
    assert Path(result.steps["baselines"]["report"]).exists()
    assert "agent" in result.steps
    assert result.steps["agent"]["run_id"]
    assert result.steps["agent"]["final_equity"] is not None


def test_baselines_only(tmp_path: Path):
    data = tmp_path / "data"
    store = BarStore(data)
    symbols = ["AAA", "BBB", "SPY"]
    BarIngestor(store, symbols, provider="synthetic").backfill(
        date(2023, 1, 3), date(2023, 6, 30)
    )
    # Minimal settings via load from real config with overrides is heavy;
    # use research path settings by writing tiny config
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "universe.yaml").write_text(
        "symbols: [AAA, BBB, SPY]\nsectors: {AAA: t, BBB: t, SPY: etf}\n"
    )
    # Reuse project base merged with data_dir override
    settings = load_settings(
        mode="backtest",
        config_dir="config",
        overrides={
            "data_dir": str(data),
            "journal_path": str(data / "j.db"),
            "backtest": {"start_cash": "10000", "monkey_runs": 5, "random_seed": 0},
        },
    )
    bundle = run_baselines_pipeline(
        settings,
        store,
        symbols,
        date(2023, 1, 3),
        date(2023, 6, 30),
        out_dir=tmp_path / "reports",
        monkey_runs=5,
    )
    assert "spy_bh" in bundle.curves
    assert "momentum" in bundle.curves
    assert bundle.report_path.exists()
    assert bundle.metrics["spy_bh"]["total_return"] is not None
