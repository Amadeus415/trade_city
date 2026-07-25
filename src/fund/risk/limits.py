"""Typed, frozen risk limits. Loaded from config; never mutated at runtime."""

from __future__ import annotations

from fund.config import RiskConfig


class RiskLimits:
    """Thin wrapper around frozen RiskConfig for the engine."""

    def __init__(self, config: RiskConfig) -> None:
        self._config = config

    @property
    def config(self) -> RiskConfig:
        return self._config

    @property
    def version(self) -> str:
        return self._config.limits_version

    def __getattr__(self, name: str):
        return getattr(self._config, name)
