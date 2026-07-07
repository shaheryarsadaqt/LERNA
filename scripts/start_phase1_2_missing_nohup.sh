#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="logs/phase1_2_missing_${STAMP}.log"
PID_FILE="logs/phase1_2_missing_${STAMP}.pid"

nohup python -u scripts/run_phase1_2_missing.py "$@" > "$LOG" 2>&1 &
PID="$!"
echo "$PID" > "$PID_FILE"

echo "Started Phase 1.2 missing-run resume process"
echo "PID: $PID"
echo "Log: $ROOT/$LOG"
echo "PID file: $ROOT/$PID_FILE"
echo "Monitor: tail -f '$ROOT/$LOG'"
