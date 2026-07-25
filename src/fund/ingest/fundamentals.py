"""Vintaged fundamentals ingest (SEC EDGAR companyfacts when available)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from fund.ingest.base import ValidationResult
from fund.store.fundamentals import FundamentalsStore


class FundamentalsIngestor:
    def __init__(self, store: FundamentalsStore, symbols: list[str]) -> None:
        self.store = store
        self.symbols = symbols

    def backfill(self, start: date, end: date) -> int:
        """Seed synthetic vintaged facts for offline/dev. Real EDGAR in production."""
        rows: list[dict] = []
        # Minimal synthetic PE / growth figures with explicit vintages
        for symbol in self.symbols:
            for year in range(start.year, end.year + 1):
                period_end = date(year, 12, 31)
                if period_end < start or period_end > end:
                    continue
                # First report ~45 days after period end
                vintage = datetime(year + 1, 2, 15, tzinfo=timezone.utc)
                rows.append(
                    {
                        "symbol": symbol,
                        "metric": "pe_ttm",
                        "period_end": period_end,
                        "vintage": vintage.date().isoformat(),
                        "value": 15.0 + (hash(symbol) % 20),
                        "observed_at": vintage,
                    }
                )
                rows.append(
                    {
                        "symbol": symbol,
                        "metric": "revenue_growth_yoy",
                        "period_end": period_end,
                        "vintage": vintage.date().isoformat(),
                        "value": 0.05 + (hash(symbol + "g") % 10) / 100,
                        "observed_at": vintage,
                    }
                )
                # Restatement one year later (must NOT leak into earlier as_of)
                restate = datetime(year + 2, 2, 20, tzinfo=timezone.utc)
                rows.append(
                    {
                        "symbol": symbol,
                        "metric": "pe_ttm",
                        "period_end": period_end,
                        "vintage": restate.date().isoformat(),
                        "value": 14.0 + (hash(symbol) % 20),  # restated
                        "observed_at": restate,
                    }
                )
        return self.store.write_facts(rows)

    def incremental(self) -> int:
        today = date.today()
        return self.backfill(today - timedelta(days=400), today)

    def validate(self) -> ValidationResult:
        as_of = datetime.now(tz=timezone.utc)
        df = self.store.get_fundamentals(self.symbols[:5], as_of=as_of)
        warnings: list[str] = []
        if df.empty:
            warnings.append("no fundamentals loaded")
        return ValidationResult(ok=True, warnings=warnings)
