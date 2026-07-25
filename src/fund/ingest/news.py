"""News ingest — RSS/API with raw text retention."""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta, timezone

from fund.ingest.base import ValidationResult
from fund.store.news import NewsStore


class NewsIngestor:
    def __init__(
        self,
        store: NewsStore,
        symbols: list[str],
        lag_minutes: int = 2,
    ) -> None:
        self.store = store
        self.symbols = symbols
        self.lag_minutes = lag_minutes

    def backfill(self, start: date, end: date) -> int:
        """Placeholder: empty news for offline. Wire a real RSS/API later."""
        return 0

    def incremental(self) -> int:
        return self.backfill(date.today() - timedelta(days=3), date.today())

    def write_test_article(
        self,
        symbol: str,
        headline: str,
        event_time: datetime,
        body: str = "",
    ) -> int:
        art_id = hashlib.sha256(f"{symbol}|{headline}|{event_time.isoformat()}".encode()).hexdigest()[
            :16
        ]
        observed = event_time + timedelta(minutes=self.lag_minutes)
        return self.store.write_articles(
            [
                {
                    "article_id": art_id,
                    "symbols": symbol,
                    "headline": headline,
                    "body": body,
                    "event_time": event_time,
                    "observed_at": observed,
                    "source": "test",
                }
            ]
        )

    def validate(self) -> ValidationResult:
        return ValidationResult(ok=True)
