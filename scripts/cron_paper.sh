#!/usr/bin/env bash
# Paper-trading cron helpers (US equities, America/New_York).
# Install example:
#   15 16 * * 1-5  cd /path/to/trade_city && ./scripts/cron_paper.sh after-close
#   32 9  * * 1-5  cd /path/to/trade_city && ./scripts/cron_paper.sh after-open
#   45 9  * * 1-5  cd /path/to/trade_city && ./scripts/cron_paper.sh health
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export FUND_CONFIG_DIR="${FUND_CONFIG_DIR:-$ROOT/config}"
export FUND_DATA_DIR="${FUND_DATA_DIR:-$ROOT/data}"

CMD="${1:-help}"
case "$CMD" in
  after-close)
    # 16:15 ingest + validate, 16:30 decide
    uv run fund ingest incremental --mode paper
    uv run fund ingest validate --mode paper
    uv run fund decide --mode paper
    ;;
  after-open)
    # 09:32 reconcile + execute (dry-run until you trust it)
    uv run fund reconcile --mode paper
    uv run fund execute --mode paper --dry-run
    # When ready: remove --dry-run or use:
    # uv run fund pipeline paper-day --live-execute
    ;;
  health)
    uv run fund healthcheck --mode paper
    uv run fund risk status --mode paper
    ;;
  paper-day)
    # Full one-shot (useful for manual / testing)
    uv run fund pipeline paper-day --provider yfinance
    ;;
  *)
    echo "Usage: $0 {after-close|after-open|health|paper-day}"
    exit 1
    ;;
esac
