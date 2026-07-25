# Capital Scaling Policy (M8)

Written before it is needed. Future-you will want to break this.

## Rule

Increase capital by **at most 2×** after each **3-month period** that satisfies **all** of:

1. Positive rolling Sharpe over the period
2. No `HALTED` risk-state transitions
3. Live-vs-backtest tracking error inside tolerance (realised Sharpe within ~50% of backtest)

## Hard constraints

- **Never** increase after a single good month.
- **Never** increase to recover a loss.
- On any `HALTED` event, capital returns to the **prior tier**.
- Live starts at **$500 or less** (see `config/live.yaml`).

## Tiers (example)

| Tier | Capital | Prerequisite |
|------|---------|--------------|
| 0 | ≤ $500 | M7 gate passed |
| 1 | ≤ $1,000 | 3 months clean at tier 0 |
| 2 | ≤ $2,000 | 3 months clean at tier 1 |
| … | 2× prior | same |

This document is committed to the repo so the policy is reviewable and diffable.
