"""LLM I/O contracts — structured output only."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field, field_validator

from fund.types import Action, Proposal


class SymbolView(BaseModel):
    symbol: str
    stance: str = Field(description="bullish | bearish | neutral | abstain")
    score: Decimal = Field(ge=-1, le=1)
    confidence: Decimal = Field(ge=0, le=1)
    thesis: str = Field(max_length=800)
    key_features: list[str] = Field(default_factory=list)


class AnalystOutput(BaseModel):
    analyst: str
    views: list[SymbolView]
    notes: str = ""


class PMProposalRaw(BaseModel):
    symbol: str
    action: Action
    target_weight: Decimal = Field(ge=0, le=1)
    confidence: Decimal = Field(ge=0, le=1)
    thesis: str = Field(max_length=1200)
    invalidation: str = Field(max_length=400)
    horizon_days: int = Field(ge=1, le=250)
    source_features: list[str]


class PMOutput(BaseModel):
    proposals: list[PMProposalRaw]
    abstain_reason: str | None = None

    def to_proposals(self) -> list[Proposal]:
        return [
            Proposal(
                symbol=p.symbol,
                action=p.action,
                target_weight=p.target_weight,
                confidence=p.confidence,
                thesis=p.thesis,
                invalidation=p.invalidation,
                horizon_days=p.horizon_days,
                source_features=p.source_features,
            )
            for p in self.proposals
        ]


ANALYST_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "analyst": {"type": "string"},
        "views": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "stance": {"type": "string"},
                    "score": {"type": "string"},
                    "confidence": {"type": "string"},
                    "thesis": {"type": "string"},
                    "key_features": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["symbol", "stance", "score", "confidence", "thesis"],
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["analyst", "views"],
}

PM_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "action": {
                        "type": "string",
                        "enum": ["open", "add", "trim", "close", "hold", "abstain"],
                    },
                    "target_weight": {"type": "string"},
                    "confidence": {"type": "string"},
                    "thesis": {"type": "string"},
                    "invalidation": {"type": "string"},
                    "horizon_days": {"type": "integer"},
                    "source_features": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "symbol",
                    "action",
                    "target_weight",
                    "confidence",
                    "thesis",
                    "invalidation",
                    "horizon_days",
                    "source_features",
                ],
            },
        },
        "abstain_reason": {"type": ["string", "null"]},
    },
    "required": ["proposals"],
}
