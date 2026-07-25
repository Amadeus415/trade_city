"""YAML + Pydantic settings. Limits are frozen after load."""

from __future__ import annotations

import hashlib
import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AccountLimits(BaseModel):
    model_config = ConfigDict(frozen=True)
    max_gross_exposure: Decimal = Decimal("0.95")
    max_net_exposure: Decimal = Decimal("0.95")
    min_cash_buffer_pct: Decimal = Decimal("0.05")
    allow_short: bool = False
    allow_margin: bool = False


class PositionLimits(BaseModel):
    model_config = ConfigDict(frozen=True)
    max_position_weight: Decimal = Decimal("0.10")
    max_new_position_weight: Decimal = Decimal("0.05")
    max_positions: int = 15
    min_position_notional: Decimal = Decimal("50.00")
    max_add_per_cycle_weight: Decimal = Decimal("0.02")


class ConcentrationLimits(BaseModel):
    model_config = ConfigDict(frozen=True)
    max_sector_weight: Decimal = Decimal("0.30")
    max_cluster_weight: Decimal = Decimal("0.35")
    correlation_threshold: Decimal = Decimal("0.70")


class TurnoverLimits(BaseModel):
    model_config = ConfigDict(frozen=True)
    max_daily_turnover: Decimal = Decimal("0.25")
    max_orders_per_day: int = 10
    min_holding_days: int = 2
    max_round_trips_per_week: int = 3


class LossLimits(BaseModel):
    model_config = ConfigDict(frozen=True)
    daily_loss_soft_pct: Decimal = Decimal("0.02")
    daily_loss_hard_pct: Decimal = Decimal("0.03")
    weekly_loss_hard_pct: Decimal = Decimal("0.07")
    max_drawdown_halt_pct: Decimal = Decimal("0.15")
    consecutive_loss_days_halt: int = 5


class OrderLimits(BaseModel):
    model_config = ConfigDict(frozen=True)
    limit_orders_only: bool = True
    max_limit_offset_bps: int = 50
    max_spread_bps: int = 30
    max_pct_of_adv: Decimal = Decimal("0.001")
    reject_if_halted: bool = True
    no_trade_first_minutes: int = 15
    no_trade_last_minutes: int = 10


class UniverseLimits(BaseModel):
    model_config = ConfigDict(frozen=True)
    allowlist_only: bool = True
    min_price: Decimal = Decimal("5.00")
    max_price: Decimal = Decimal("2000.00")
    min_avg_dollar_volume_21d: Decimal = Decimal("5000000")
    min_days_listed: int = 250
    exclude_earnings_within_days: int = 2
    blocklist: list[str] = Field(default_factory=list)


class RiskConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    limits_version: str = "1.0.0"
    account: AccountLimits = Field(default_factory=AccountLimits)
    position: PositionLimits = Field(default_factory=PositionLimits)
    concentration: ConcentrationLimits = Field(default_factory=ConcentrationLimits)
    turnover: TurnoverLimits = Field(default_factory=TurnoverLimits)
    loss_limits: LossLimits = Field(default_factory=LossLimits)
    order: OrderLimits = Field(default_factory=OrderLimits)
    universe: UniverseLimits = Field(default_factory=UniverseLimits)


class AgentConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    provider: str = "mock"
    analyst_model: str = "mock-analyst"
    pm_model: str = "mock-pm"
    temperature: float = 0.1
    masking_mode: str = "bright"
    max_retries: int = 1
    cache_enabled: bool = True
    cache_dir: str = "./data/llm_cache"
    api_key: str | None = None
    base_url: str | None = None


class CostConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    commission_per_share: Decimal = Decimal("0.01")
    default_spread_bps_liquid: int = 5
    default_spread_bps_illiquid: int = 15
    slippage_base_bps: int = 5
    slippage_size_coeff: int = 10
    limit_nonfill_rate: float = 0.20
    sec_fee_bps: Decimal = Decimal("0.08")
    finra_taf_per_share: Decimal = Decimal("0.000166")


class BacktestConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    start_cash: Decimal = Decimal("10000.00")
    monkey_runs: int = 200
    random_seed: int = 42


class AvailabilityConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    bar_lag_minutes: int = 15
    news_lag_minutes: int = 2


class AlertConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    webhook_url: str | None = None


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)
    mode: str = "paper"
    data_dir: str = "./data"
    journal_path: str = "./data/journal.db"
    timezone: str = "America/New_York"
    initial_cash: Decimal = Decimal("10000.00")
    limits_version: str = "1.0.0"
    availability: AvailabilityConfig = Field(default_factory=AvailabilityConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    costs: CostConfig = Field(default_factory=CostConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    alerting: AlertConfig = Field(default_factory=AlertConfig)

    def config_hash(self) -> str:
        payload = self.model_dump(mode="json")
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    llm_provider: str | None = None
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_analyst_model: str | None = None
    llm_pm_model: str | None = None
    alert_webhook_url: str | None = None
    fund_data_dir: str | None = None
    fund_config_dir: str | None = None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return data


def load_settings(
    mode: str | None = None,
    config_dir: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    """Load base.yaml + {mode}.yaml + env overrides."""
    cfg_dir = Path(config_dir or os.environ.get("FUND_CONFIG_DIR", "config"))
    base = _load_yaml(cfg_dir / "base.yaml")
    mode_name = mode or base.get("mode", "paper")
    mode_cfg = _load_yaml(cfg_dir / f"{mode_name}.yaml")
    merged = _deep_merge(base, mode_cfg)
    if overrides:
        merged = _deep_merge(merged, overrides)

    env = EnvSettings()
    agent = merged.setdefault("agent", {})
    if env.llm_provider:
        agent["provider"] = env.llm_provider
    if env.llm_api_key:
        agent["api_key"] = env.llm_api_key
    if env.llm_base_url:
        agent["base_url"] = env.llm_base_url
    if env.llm_analyst_model:
        agent["analyst_model"] = env.llm_analyst_model
    if env.llm_pm_model:
        agent["pm_model"] = env.llm_pm_model
    if env.alert_webhook_url:
        merged.setdefault("alerting", {})["webhook_url"] = env.alert_webhook_url
    if env.fund_data_dir:
        merged["data_dir"] = env.fund_data_dir
        merged["journal_path"] = str(Path(env.fund_data_dir) / "journal.db")

    if "mode" not in merged:
        merged["mode"] = mode_name

    return Settings.model_validate(merged)


def load_universe(config_dir: str | Path | None = None) -> dict[str, Any]:
    cfg_dir = Path(config_dir or os.environ.get("FUND_CONFIG_DIR", "config"))
    return _load_yaml(cfg_dir / "universe.yaml")
