#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/content}"
REPO="${REPO:-$ROOT/LaflaAi-Core}"
WORK="${WORK:-$ROOT/LaflaAI100M}"
DRIVE_ROOT="${DRIVE_ROOT:-$ROOT/gdrive/MyDrive/LaflaAI100M}"
DATA_JSONL="${DATA_JSONL:-$WORK/data/train.jsonl}"
MANIFEST="${MANIFEST:-$WORK/data/veri_manifesti.json}"
REPORTS="$WORK/reports"
TOKENIZER="$WORK/tokenizer/lafla-tokenizer.json"
CHECKPOINTS="$WORK/checkpoints"
HF_PACKAGE="$WORK/hf-package"
LOG="$REPORTS/colab-tpu-launch.log"

mkdir -p "$REPORTS" "$WORK/tokenizer" "$CHECKPOINTS" "$HF_PACKAGE"
exec > >(tee -a "$LOG") 2>&1

echo "[lafla] start=$(date -Is)"
echo "[lafla] repo=$REPO"
echo "[lafla] data=$DATA_JSONL"
echo "[lafla] manifest=$MANIFEST"

test -d "$REPO" || { echo "Repo bulunamadi: $REPO" >&2; exit 2; }
test -s "$DATA_JSONL" || { echo "Gercek train.jsonl bulunamadi veya bos: $DATA_JSONL" >&2; exit 2; }
test -s "$MANIFEST" || { echo "veri_manifesti.json bulunamadi veya bos: $MANIFEST" >&2; exit 2; }
test -s "$REPO/configs/data/lafla-100m-source-plan.json" || {
  echo "100M source plan bulunamadi" >&2
  exit 2
}

python - <<'PY'
from google.colab import drive
drive.mount("/content/gdrive", force_remount=False)
PY

cd "$REPO"
python -m pip install --upgrade pip wheel setuptools
python -m pip install torch "torch_xla[tpu]"
python -m pip install -r requirements/colab-tpu.txt

export PYTHONPATH=src
export PJRT_DEVICE=TPU
export XLA_USE_BF16=1
export TOKENIZERS_PARALLELISM=true

python - <<'PY'
import torch_xla
print("xla_device=", torch_xla.device())
PY

python -m lafla_ai_core.cli.check_environment --optimizer adamw --accelerator xla
python -m lafla_ai_core.cli.quality_scan --root .
python -m lafla_ai_core.cli.preflight \
  configs/model/lafla-100m-thinking.yaml \
  configs/training/colab-tpu-v5e-100m.yaml \
  configs/tokenizer/turkish-german-thinking-bpe.yaml \
  configs/runtime/desktop-i3-int8-100m.yaml \
  configs/post_training/lafla-thinking-sft.yaml
python -m lafla_ai_core.cli.data_audit \
  --manifest "$MANIFEST" \
  --report "$REPORTS/data-audit.json"

if [ ! -s "$TOKENIZER" ]; then
  python -m lafla_ai_core.cli.tokenizer_train \
    --config configs/tokenizer/turkish-german-thinking-bpe.yaml \
    --output "$TOKENIZER" \
    --report "$REPORTS/tokenizer-quality.json" \
    configs/data/lafla-model-identity-100m.jsonl \
    "$DATA_JSONL"
else
  echo "[lafla] existing tokenizer=$TOKENIZER"
fi

python -m lafla_ai_core.cli.hf_package \
  --tokenizer-json "$TOKENIZER" \
  --output-dir "$HF_PACKAGE" \
  --model-config configs/model/lafla-100m-thinking.yaml \
  --model-name lafla-100m-thinking

THINKING_SFT_JSONLS=(
  datasets/synthetic/lafla-100m-thinking-chat-seed-20k.jsonl
  datasets/synthetic/lafla-100m-safety-jailbreak-seed-10k.jsonl
)
for index in "${!THINKING_SFT_JSONLS[@]}"; do
  sft_jsonl="${THINKING_SFT_JSONLS[$index]}"
  report_index="$(printf "%03d" "$((index + 1))")"
  test -s "$sft_jsonl" || { echo "Thinking SFT seed bulunamadi veya bos: $sft_jsonl" >&2; exit 2; }
  python -m lafla_ai_core.cli.validate_thinking_sft \
    --input "$sft_jsonl" \
    --report "$REPORTS/thinking-sft-audit-$report_index.json"
done
SFT_MANIFEST="$REPORTS/post-training-sft-inputs.json" python - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "stage": "post_training_thinking_sft",
    "usage": "validate_then_use_after_base_pretrain_checkpoint",
    "allowed_for_pretraining": False,
    "inputs": [
        "datasets/synthetic/lafla-100m-thinking-chat-seed-20k.jsonl",
        "datasets/synthetic/lafla-100m-safety-jailbreak-seed-10k.jsonl",
    ],
}
path = Path(os.environ["SFT_MANIFEST"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

TRAIN_ARGS=(
  python -m lafla_ai_core.cli.train_pretrain
  --model-config configs/model/lafla-100m-thinking.yaml
  --training-config configs/training/colab-tpu-v5e-100m.yaml
  --tokenizer-path "$TOKENIZER"
  --checkpoint-dir "$CHECKPOINTS"
  --health-log "$REPORTS/train-health.jsonl"
  --data-jsonl configs/data/lafla-model-identity-100m.jsonl
  --data-jsonl "$DATA_JSONL"
)
if [ -n "${RESUME_FROM:-}" ]; then
  test -d "$RESUME_FROM" || { echo "RESUME_FROM checkpoint bulunamadi: $RESUME_FROM" >&2; exit 2; }
  TRAIN_ARGS+=(--resume-from "$RESUME_FROM")
fi
"${TRAIN_ARGS[@]}"

python -m lafla_ai_core.cli.artifact_manifest \
  --root "$WORK" \
  --output "$REPORTS/artifact-manifest.json"
mkdir -p "$DRIVE_ROOT/final-checkpoint"
tar -czf "$DRIVE_ROOT/final-checkpoint/lafla-100m-thinking-colab-tpu-v5e-run.tar.gz" \
  -C "$CHECKPOINTS" lafla-final \
  -C "$WORK" tokenizer reports hf-package
sync
ls -lh "$DRIVE_ROOT/final-checkpoint"
echo "[lafla] finished=$(date -Is)"
