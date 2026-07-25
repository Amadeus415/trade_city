"""Historical universe membership — survivorship-safe."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yaml


class UniverseStore:
    def __init__(self, data_dir: str | Path, config_dir: str | Path = "config") -> None:
        self.root = Path(data_dir) / "universe"
        self.root.mkdir(parents=True, exist_ok=True)
        self._membership = self.root / "membership.parquet"
        self.config_dir = Path(config_dir)

    def write_membership(self, rows: list[dict] | pd.DataFrame) -> int:
        if isinstance(rows, list):
            if not rows:
                return 0
            df = pd.DataFrame(rows)
        else:
            df = rows.copy()
        df["start_date"] = pd.to_datetime(df["start_date"]).dt.date
        df["end_date"] = df["end_date"].apply(
            lambda x: None if pd.isna(x) or x is None else pd.to_datetime(x).date()
        )
        if self._membership.exists():
            existing = pd.read_parquet(self._membership)
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["symbol", "index", "start_date"], keep="last"
            )
        else:
            combined = df
        combined.to_parquet(self._membership, index=False)
        return len(df)

    def get_universe(self, as_of: datetime, index: str | None = None) -> list[str]:
        """Resolve membership as of a decision timestamp (uses session date)."""
        d = as_of.date() if isinstance(as_of, datetime) else as_of
        if self._membership.exists():
            df = pd.read_parquet(self._membership)
            if index:
                df = df[df["index"] == index]
            members: list[str] = []
            for _, row in df.iterrows():
                start = row["start_date"]
                end = row["end_date"]
                if isinstance(start, str):
                    start = date.fromisoformat(start)
                if isinstance(end, str):
                    end = date.fromisoformat(end)
                if start <= d and (end is None or pd.isna(end) or end >= d):
                    members.append(str(row["symbol"]))
            if members:
                return sorted(set(members))
        # Fallback to static allowlist (current file) — document as non-historical
        return self.static_allowlist()

    def static_allowlist(self) -> list[str]:
        path = self.config_dir / "universe.yaml"
        if not path.exists():
            return []
        with path.open() as f:
            data = yaml.safe_load(f) or {}
        return list(data.get("symbols", []))

    def sector_map(self) -> dict[str, str]:
        path = self.config_dir / "universe.yaml"
        if not path.exists():
            return {}
        with path.open() as f:
            data = yaml.safe_load(f) or {}
        return dict(data.get("sectors", {}))

    def seed_from_allowlist(
        self,
        start_date: date,
        index: str = "custom_v1",
    ) -> int:
        """Bootstrap historical membership from current allowlist (research only).

        For production historical accuracy, replace with true index membership.
        """
        symbols = self.static_allowlist()
        rows = [
            {
                "symbol": s,
                "index": index,
                "start_date": start_date,
                "end_date": None,
            }
            for s in symbols
        ]
        return self.write_membership(rows)
