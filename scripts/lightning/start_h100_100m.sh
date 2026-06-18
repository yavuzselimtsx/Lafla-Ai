#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/teamspace/studios/this_studio}"
WORK="${WORK:-$ROOT/LaflaAI100M}"
REPORTS="$WORK/reports"

export PROFILE_NAME="h100"
export LAFLA_TRAINING_CONFIG="configs/training/lightning/lightning-h100-100m-quality-fast.yaml"
export VENV="${VENV:-$ROOT/.venvs/lafla-100m-h100}"
export LOG="${LOG:-$REPORTS/lightning-h100-100m.log}"
export DEVICE_LABEL="H100"
export DATASET_VERSION="lafla-100m-lightning-h100-realdata-2026-06"
export ARCHIVE_NAME="lafla-100m-thinking-lightning-h100-run.tar.gz"
export PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
export PYTORCH_VERSION="${LAFLA_TORCH_VERSION:-2.11.0}"
export EXPECTED_GPU_NAME="NVIDIA H100"
export REQUIRE_BF16="1"

mkdir -p "$REPORTS"
last_status=1
for CUDA_BATCH_SCALE in 1.0 0.5 0.25; do
  export CUDA_BATCH_SCALE
  echo "[lafla] H100 attempt cuda_batch_scale=$CUDA_BATCH_SCALE"
  if bash scripts/lightning/start_t4_100m.sh; then
    exit 0
  else
    last_status=$?
  fi
  if ! tail -n 200 "$LOG" | grep -Eq "torch\.OutOfMemoryError|CUDA out of memory"; then
    echo "[lafla] CUDA OOM disi hata; otomatik retry yapilmadi" >&2
    exit "$last_status"
  fi
  echo "[lafla] CUDA OOM; son READY checkpoint daha kucuk batch ile yeniden acilacak" >&2
done

echo "[lafla] H100 CUDA batch fallback profilleri tukendi" >&2
exit "$last_status"
