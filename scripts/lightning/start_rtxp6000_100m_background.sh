#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/teamspace/studios/this_studio}"
REPO="${REPO:-$ROOT/LaflaAi-Core}"
WORK="${WORK:-$ROOT/LaflaAI100M}"
REPORTS="$WORK/reports"
PID_FILE="$REPORTS/lightning-rtxp6000-100m.pid"
NOHUP_LOG="$REPORTS/lightning-rtxp6000-100m-nohup.log"

mkdir -p "$REPORTS"
test -d "$REPO" || { echo "Repo bulunamadi: $REPO" >&2; exit 2; }

if pgrep -af "lafla_ai_core.cli.train_pretrain" >/dev/null 2>&1; then
  echo "[lafla] egitim zaten calisiyor:"
  pgrep -af "lafla_ai_core.cli.train_pretrain"
  exit 0
fi

cd "$REPO"
nohup bash scripts/lightning/start_rtxp6000_100m.sh > "$NOHUP_LOG" 2>&1 &
PID="$!"
printf "%s\n" "$PID" > "$PID_FILE"

echo "[lafla] background_pid=$PID"
echo "[lafla] pid_file=$PID_FILE"
echo "[lafla] nohup_log=$NOHUP_LOG"
echo "[lafla] monitor: bash scripts/lightning/monitor_100m.sh --watch 10"
