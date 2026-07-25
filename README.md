# Agentic Fund

Point-in-time disciplined, model-agnostic equity trading system.

Built from `AGENTIC_TRADING_PLAN.md`. **Success is not "makes money."** Success is honest measurement against baselines, hard risk vetoes, and no look-ahead bias.

## Invariants

1. **INV-1 — Point-in-time integrity.** Every store reader requires `as_of`. No future `observed_at` data is visible.
2. **INV-2 — Deterministic risk veto.** `fund.risk` never imports `fund.agent`. Limits are frozen. LLM proposes; risk disposes.
3. **INV-3 — Identical execution path.** Backtest / paper / live differ only by `BrokerAdapter`.

`HOLD` / `ABSTAIN` are first-class actions.

## Stack

- Python 3.12, `uv`, Pydantic v2, Parquet + DuckDB, SQLite (WAL), YAML config, `structlog`, `pytest` + `hypothesis`

## Quick start

```bash
uv sync --all-extras
cp .env.example .env

# M0 gate
uv run fund healthcheck

# ── Full research pipeline (recommended) ──────────────────────────
# ingest → validate → baselines → agent backtest → reports
uv run fund pipeline research \
  --start 2023-01-01 --end 2024-12-31 \
  --provider yfinance \
  --monkey-runs 50

# Optional: M5 leakage gate (all four masking modes)
uv run fund pipeline research \
  --start 2024-01-01 --end 2024-06-30 \
  --provider yfinance --leakage --monkey-runs 50

# Or step-by-step:
uv run fund ingest backfill --start 2023-01-01 --end 2024-12-31 --provider yfinance
uv run fund ingest validate
uv run fund ingest cross-validate --sample 20
uv run fund eval baselines --start 2023-01-01 --end 2024-12-31
uv run fund backtest run --start 2024-01-01 --end 2024-06-30 --strategy agent
uv run fund eval report --run <run_id>
uv run fund eval leakage --start 2024-01-01 --end 2024-06-30

# Paper day (cron-friendly)
uv run fund pipeline paper-day          # dry-run execute
./scripts/cron_paper.sh after-close     # post-close jobs
./scripts/cron_paper.sh after-open      # post-open jobs
```

## Model-agnostic LLM layer

Set in `.env` or `config/*.yaml` → `agent:`:

| Provider | Notes |
|----------|--------|
| `mock` | Deterministic offline (default). |
| `openai` | OpenAI API |
| `anthropic` | Anthropic Messages API |
| `xai` | xAI (OpenAI-compatible) |
| `ollama` | Local OpenAI-compatible |
| `openai_compatible` | Any base URL (Groq, Together, vLLM, …) |

```bash
export LLM_PROVIDER=openai_compatible
export LLM_BASE_URL=https://api.your-host.com/v1
export LLM_API_KEY=...
export LLM_ANALYST_MODEL=...
export LLM_PM_MODEL=...
```

Providers implement `LLMProvider.complete(messages, response_schema=...)`. Swap models without touching risk, execution, or journal code.

## CLI

```
fund ingest incremental|backfill|validate
fund decide --mode {backtest,paper,live} [--as-of] [--dry-run]
fund execute --mode {paper,live} [--dry-run]
fund reconcile
fund healthcheck
fund risk status|reset --confirm-equity <value>
fund backtest run --start ... --end ... [--strategy agent|momentum|equal_weight]
fund eval report --run <id>
```

Daily schedule (US equities): decide after close; execute after the open settles. See the plan §11.

## Milestones

| Gate | Status in repo |
|------|----------------|
| M0 Scaffolding | Implemented |
| M1 Data + PIT | Implemented (+ tests) |
| M2 Backtester + baselines | Implemented |
| M3 Risk engine | Implemented (+ hypothesis) |
| M4 Features + agent | Implemented (mock + multi-provider) |
| M5 Leakage gate | Tooling ready (`masking_mode`); run empirically before live |
| M6 Paper (3 mo) | Calendar-bound — not skippable |
| M7 Live ≤ $500 | Robinhood MCP adapter stub; token outside repo |
| M8 Scaling policy | `SCALING_POLICY.md` |

**Do not deploy real money until M0–M5 gates pass and M6 paper history exists.**

## Layout

See plan §3. Package lives under `src/fund/`.

## Tests

```bash
uv run pytest -q
```

Critical gates: `test_pit`, `test_risk_properties`, `test_killswitch`, `test_idempotency`, `test_parity`, `test_costs`, `test_calendar`.

## Secrets

- `.env` gitignored; OAuth for Robinhood in `~/.config/fund/` mode `0600`
- Never journal tokens

## License

Personal / hobby use. Not investment advice. Not for managing other people's money.
