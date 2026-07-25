#!/usr/bin/env bash
# Full research pipeline (ingest → baselines → agent → optional leakage).
# Usage:
#   ./scripts/run_research.sh 2023-01-01 2024-06-30
#   ./scripts/run_research.sh 2024-01-01 2024-03-31 --leakage --provider synthetic
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

START="${1:?start date YYYY-MM-DD}"
END="${2:?end date YYYY-MM-DD}"
shift 2 || true

uv run fund pipeline research --start "$START" --end "$END" "$@"
