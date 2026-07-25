"""Ingestor protocol and shared validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Protocol, runtime_checkable


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise ValueError("ingest validation failed: " + "; ".join(self.errors))


@runtime_checkable
class Ingestor(Protocol):
    def backfill(self, start: date, end: date) -> int: ...

    def incremental(self) -> int: ...

    def validate(self) -> ValidationResult: ...
