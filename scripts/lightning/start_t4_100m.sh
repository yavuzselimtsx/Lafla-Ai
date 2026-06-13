#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/teamspace/studios/this_studio}"
REPO="${REPO:-$ROOT/LaflaAi-Core}"
WORK="${WORK:-$ROOT/LaflaAI100M}"
VENV="${VENV:-$ROOT/.venvs/lafla-100m-t4}"
DATA_JSONL="${DATA_JSONL:-$WORK/data/train.jsonl}"
MANIFEST="${MANIFEST:-$WORK/data/veri_manifesti.json}"
REPORTS="$WORK/reports"
TOKENIZER="$WORK/tokenizer/lafla-tokenizer.json"
CHECKPOINTS="$WORK/checkpoints"
HF_PACKAGE="$WORK/hf-package"
ARCHIVES="$WORK/archives"
LOG="$REPORTS/lightning-t4-100m.log"

TARGET_CHARS="${TARGET_CHARS:-200000000}"
MIN_CHARS="${MIN_CHARS:-150000000}"
MAX_RECORD_CHARS="${MAX_RECORD_CHARS:-12000}"

mkdir -p "$WORK/data" "$REPORTS" "$WORK/tokenizer" "$CHECKPOINTS" "$HF_PACKAGE" "$ARCHIVES" "$(dirname "$VENV")"
exec > >(tee -a "$LOG") 2>&1

echo "[lafla] start=$(date -Is)"
echo "[lafla] repo=$REPO"
echo "[lafla] work=$WORK"
echo "[lafla] data=$DATA_JSONL"
echo "[lafla] manifest=$MANIFEST"
echo "[lafla] venv=$VENV"

test -d "$REPO" || { echo "Repo bulunamadi: $REPO" >&2; exit 2; }
cd "$REPO"

ensure_python_bootstrap() {
  if ! python3 -m pip --version >/dev/null 2>&1; then
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/lafla-get-pip.py
    python3 /tmp/lafla-get-pip.py --user
  fi
  python3 -m pip install --user --upgrade virtualenv
}

create_virtualenv() {
  if [ -x "$VENV/bin/python" ]; then
    return
  fi
  if [ -e "$VENV" ]; then
    case "$VENV" in
      "$ROOT"/.venvs/*) rm -rf "$VENV" ;;
      *) echo "Guvenli olmayan VENV yolu temizlenmedi: $VENV" >&2; exit 2 ;;
    esac
  fi
  if python3 -m virtualenv "$VENV"; then
    return
  fi
  if python3 -m venv "$VENV"; then
    return
  fi
  if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y python3.12-venv python3-pip
    python3 -m venv "$VENV"
    return
  fi
  echo "venv olusturulamadi; virtualenv, python3-venv veya passwordless sudo gerekli" >&2
  exit 2
}

ensure_python_bootstrap
create_virtualenv
source "$VENV/bin/activate"

python -m pip install --upgrade pip wheel setuptools
if ! python - <<'PY'
import torch
print("torch_exists=", torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())
PY
then
  python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch
fi
python -m pip install -r requirements/kaggle-gpu.txt

export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=true

CUDA_DEVICE_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
echo "[lafla] CUDA_DEVICE_COUNT=$CUDA_DEVICE_COUNT"
test "$CUDA_DEVICE_COUNT" -ge 1 || { echo "Lightning T4 launcher en az bir CUDA cihazi gerektirir" >&2; exit 2; }

python - <<'PY'
import torch

print("cuda_available=", torch.cuda.is_available())
print("cuda_device_count=", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(f"cuda:{index}=", torch.cuda.get_device_name(index))
PY

python -m lafla_ai_core.cli.check_environment --optimizer adamw --accelerator cuda
python -m lafla_ai_core.cli.quality_scan --root .
python -m lafla_ai_core.cli.preflight \
  configs/model/lafla-100m-thinking.yaml \
  configs/training/kaggle/kaggle-gpu-100m.yaml \
  configs/tokenizer/turkish-german-thinking-bpe.yaml \
  configs/runtime/desktop-i3-int8-100m.yaml \
  configs/post_training/lafla-thinking-sft.yaml

if [ ! -s "$DATA_JSONL" ] || [ ! -s "$MANIFEST" ]; then
  echo "[lafla] real data missing; preparing real data"
  python scripts/data/prepare_real_data.py \
    --output "$DATA_JSONL" \
    --manifest "$MANIFEST" \
    --report "$REPORTS/data-prepare-report.json" \
    --identity-jsonl configs/data/identity/lafla-model-identity-100m.jsonl \
    --dataset-version lafla-100m-lightning-t4-realdata-2026-06 \
    --target-chars "$TARGET_CHARS" \
    --min-chars "$MIN_CHARS" \
    --max-record-chars "$MAX_RECORD_CHARS"
else
  echo "[lafla] existing real data found"
fi

python -m lafla_ai_core.cli.data_audit \
  --manifest "$MANIFEST" \
  --report "$REPORTS/data-audit.json"
python -m lafla_ai_core.cli.validate_pretraining_data \
  --data-jsonl configs/data/identity/lafla-model-identity-100m.jsonl \
  --data-jsonl "$DATA_JSONL" \
  --report "$REPORTS/pretraining-data-validation.json"

if [ ! -s "$TOKENIZER" ]; then
  python -m lafla_ai_core.cli.tokenizer_train \
    --config configs/tokenizer/turkish-german-thinking-bpe.yaml \
    --output "$TOKENIZER" \
    --report "$REPORTS/tokenizer-quality.json" \
    configs/data/identity/lafla-model-identity-100m.jsonl \
    "$DATA_JSONL"
else
  echo "[lafla] existing tokenizer=$TOKENIZER"
fi

python -m lafla_ai_core.cli.hf_package \
  --tokenizer-json "$TOKENIZER" \
  --output-dir "$HF_PACKAGE" \
  --model-config configs/model/lafla-100m-thinking.yaml \
  --model-name lafla-100m-thinking

AUTO_RESUME=""
if [ -z "${RESUME_FROM:-}" ]; then
  AUTO_RESUME="$(CHECKPOINTS="$CHECKPOINTS" python - <<'PY'
import os
from pathlib import Path

candidates: list[tuple[int, Path]] = []
for path in Path(os.environ["CHECKPOINTS"]).glob("lafla-step-*"):
    if not (path / "READY.json").exists():
        continue
    try:
        step = int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        continue
    candidates.append((step, path))
if candidates:
    print(max(candidates)[1])
PY
)"
fi
ACTIVE_RESUME="${RESUME_FROM:-$AUTO_RESUME}"

TRAIN_ARGS=(
  -m lafla_ai_core.cli.train_pretrain
  --model-config configs/model/lafla-100m-thinking.yaml
  --training-config configs/training/kaggle/kaggle-gpu-100m.yaml
  --tokenizer-path "$TOKENIZER"
  --checkpoint-dir "$CHECKPOINTS"
  --health-log "$REPORTS/train-health.jsonl"
  --data-jsonl configs/data/identity/lafla-model-identity-100m.jsonl
  --data-jsonl "$DATA_JSONL"
)
if [ -n "$ACTIVE_RESUME" ]; then
  test -d "$ACTIVE_RESUME" || { echo "RESUME_FROM checkpoint bulunamadi: $ACTIVE_RESUME" >&2; exit 2; }
  TRAIN_ARGS+=(--resume-from "$ACTIVE_RESUME")
  echo "[lafla] resume_from=$ACTIVE_RESUME"
fi
python "${TRAIN_ARGS[@]}"

python -m lafla_ai_core.cli.artifact_manifest \
  --root "$WORK" \
  --output "$REPORTS/artifact-manifest.json"
tar -czf "$ARCHIVES/lafla-100m-thinking-lightning-t4-run.tar.gz" \
  -C "$CHECKPOINTS" lafla-final \
  -C "$WORK" tokenizer reports hf-package
sync
ls -lh "$ARCHIVES"
echo "[lafla] finished=$(date -Is)"
