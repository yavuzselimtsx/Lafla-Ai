#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/teamspace/studios/this_studio}"
WORK="${WORK:-$ROOT/LaflaAI100M}"
REPORTS="$WORK/reports"
CHECKPOINTS="$WORK/checkpoints"
CHECKPOINT_BACKUPS="$WORK/checkpoint-backups"
HEALTH_LOG="$REPORTS/train-health.jsonl"
NOHUP_LOG="${NOHUP_LOG:-$REPORTS/lightning-t4-100m-nohup.log}"

INTERVAL=0
if [ "${1:-}" = "--watch" ]; then
  INTERVAL="${2:-10}"
fi

render_once() {
  echo "===== TIME ====="
  date -Is
  echo
  echo "===== PROCESS ====="
  pgrep -af "lafla_ai_core.cli.train_pretrain" || echo "egitim prosesi yok"
  echo
  echo "===== GPU ====="
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv || true
  else
    echo "nvidia-smi yok"
  fi
  echo
  echo "===== SON TRAIN HEALTH ====="
  if [ -s "$HEALTH_LOG" ]; then
    tail -n 8 "$HEALTH_LOG"
  else
    echo "health log yok: $HEALTH_LOG"
  fi
  echo
  echo "===== SON LAUNCH LOG ====="
  if [ -s "$NOHUP_LOG" ]; then
    tail -n 12 "$NOHUP_LOG"
  else
    echo "nohup log yok: $NOHUP_LOG"
  fi
  echo
  echo "===== CHECKPOINTS ====="
  if [ -d "$CHECKPOINTS" ]; then
    ls -1 "$CHECKPOINTS" | tail -n 12
  else
    echo "checkpoint klasoru yok: $CHECKPOINTS"
  fi
  echo
  echo "===== CHECKPOINT BACKUPS ====="
  if [ -d "$CHECKPOINT_BACKUPS" ]; then
    ls -1 "$CHECKPOINT_BACKUPS" | tail -n 12
  else
    echo "checkpoint backup klasoru yok: $CHECKPOINT_BACKUPS"
  fi
}

if [ "$INTERVAL" = "0" ]; then
  render_once
else
  while true; do
    printf "\033c"
    render_once
    sleep "$INTERVAL"
  done
fi
