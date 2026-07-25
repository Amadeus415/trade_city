"""Content-hash LLM cache — reproducible backtests, lower cost."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class LLMCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
              cache_key TEXT PRIMARY KEY,
              provider TEXT NOT NULL,
              model TEXT NOT NULL,
              prompt_version TEXT NOT NULL,
              response_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    @staticmethod
    def make_key(
        prompt_version: str,
        model: str,
        feature_packet: str,
        system_prompt: str = "",
    ) -> str:
        payload = f"{prompt_version}|{model}|{system_prompt}|{feature_packet}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> str | None:
        cur = self._conn.execute(
            "SELECT response_json FROM cache WHERE cache_key = ?", (key,)
        )
        row = cur.fetchone()
        return row[0] if row else None

    def put(
        self,
        key: str,
        *,
        provider: str,
        model: str,
        prompt_version: str,
        response: str,
    ) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO cache
              (cache_key, provider, model, prompt_version, response_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                key,
                provider,
                model,
                prompt_version,
                response,
                datetime.now(tz=timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
