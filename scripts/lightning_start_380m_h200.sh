#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/teamspace/studios/this_studio}"
ZIP="${ZIP:-$ROOT/LaflaAi-Core-src-lightning-h200-380m.zip}"
REPO="${REPO:-$ROOT/LaflaAi-Core}"
WORK="${WORK:-$ROOT/LaflaAI380M}"
RAW_DATA="$WORK/data/train.raw.jsonl"
CLEAN_DATA="$WORK/data/train.clean.jsonl"
TOKENIZER_SAMPLE="$WORK/data/tokenizer.sample.jsonl"
MANIFEST="$WORK/data/veri_manifesti.json"
REPORTS="$WORK/reports"
LOG="$REPORTS/launch.log"
export RAW_DATA CLEAN_DATA TOKENIZER_SAMPLE

mkdir -p "$WORK/data" "$REPORTS" "$WORK/tokenizer" "$WORK/checkpoints" "$WORK/hf-package"
exec > >(tee -a "$LOG") 2>&1

echo "[lafla] root=$ROOT"
echo "[lafla] work=$WORK"
echo "[lafla] started=$(date -Is)"
nvidia-smi || true

if [ -f "$ZIP" ]; then
  echo "[lafla] extracting $ZIP to $REPO"
  rm -rf "$REPO"
  export ZIP REPO
  python - <<'PY'
import os
import shutil
import zipfile
from pathlib import Path

zip_path = Path(os.environ["ZIP"]).resolve()
repo = Path(os.environ["REPO"]).resolve()
repo.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_path) as archive:
    entries = [(item, item.filename.replace("\\", "/")) for item in archive.infolist()]
    file_names = [name for item, name in entries if name and not name.endswith("/") and not item.is_dir()]
    first = file_names[0].split("/")[0] if file_names else ""
    strip_prefix = first if first and all(name.startswith(first + "/") for name in file_names) else ""
    for item, name in entries:
        if strip_prefix and name.startswith(strip_prefix + "/"):
            name = name[len(strip_prefix) + 1 :]
        if not name:
            continue
        is_dir = item.is_dir() or name.endswith("/")
        clean_name = name.rstrip("/") if is_dir else name
        if not clean_name:
            continue
        target = (repo / clean_name).resolve()
        if repo not in target.parents and target != repo:
            raise RuntimeError(f"zip path outside repo: {item.filename}")
        if is_dir:
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(item) as source, target.open("wb") as dest:
            shutil.copyfileobj(source, dest)
PY
else
  echo "[lafla] zip not found, using existing repo: $REPO"
fi

cd "$REPO"
python -m venv "$WORK/.venv"
source "$WORK/.venv/bin/activate"
python -m pip install --upgrade pip wheel setuptools
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch
python -m pip install -r requirements/colab.txt

export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m lafla_ai_core.cli.check_environment --optimizer adamw
python -m lafla_ai_core.cli.quality_scan --root .
python -m lafla_ai_core.cli.preflight \
  configs/model/lafla-380m-thinking.yaml \
  configs/training/lightning-h200-380m-50000.yaml \
  configs/tokenizer/turkish-thinking-bpe.yaml \
  configs/runtime/desktop-phone-fp16-380m.yaml \
  configs/post_training/lafla-thinking-sft.yaml

if [ ! -s "$RAW_DATA" ]; then
  python scripts/lightning_prepare_real_data.py \
    --output "$RAW_DATA" \
    --manifest "$MANIFEST" \
    --report "$REPORTS/data-prepare-report.json" \
    --identity-jsonl configs/data/lafla-model-identity-380m.jsonl \
    --dataset-version lafla-380m-thinking-realdata-2026-06 \
    --target-chars "${TARGET_CHARS:-2200000000}" \
    --min-chars "${MIN_CHARS:-900000000}"
else
  echo "[lafla] using existing raw data: $RAW_DATA"
fi

python - <<'PY'
import json
import os
from pathlib import Path

from lafla_ai_core.tokenizer.quality import validate_clean_text

raw = Path(os.environ["RAW_DATA"])
clean = Path(os.environ["CLEAN_DATA"])
sample = Path(os.environ["TOKENIZER_SAMPLE"])
sample_char_limit = int(os.environ.get("TOKENIZER_SAMPLE_CHARS", "60000000"))
clean.parent.mkdir(parents=True, exist_ok=True)
kept = 0
rejected = 0
sample_chars = 0
with raw.open("r", encoding="utf-8") as source, clean.open("w", encoding="utf-8", newline="\n") as dest, sample.open(
    "w", encoding="utf-8", newline="\n"
) as sample_dest:
    for line_number, line in enumerate(source, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            text = validate_clean_text(str(record.get("text", "")), f"{raw}:{line_number}")
            if text.strip():
                output_line = json.dumps({"text": text}, ensure_ascii=False) + "\n"
                dest.write(output_line)
                if sample_chars < sample_char_limit:
                    sample_dest.write(output_line)
                    sample_chars += len(text)
                kept += 1
            else:
                rejected += 1
        except Exception:
            rejected += 1
print(
    json.dumps(
        {"clean_data": str(clean), "tokenizer_sample": str(sample), "sample_chars": sample_chars, "kept": kept, "rejected": rejected},
        ensure_ascii=False,
        sort_keys=True,
    )
)
if kept <= 0:
    raise SystemExit("clean data is empty")
PY

python -m lafla_ai_core.cli.data_audit --manifest "$MANIFEST" --report "$REPORTS/data-audit.json"
python -m lafla_ai_core.cli.tokenizer_train \
  --config configs/tokenizer/turkish-thinking-bpe.yaml \
  --output "$WORK/tokenizer/lafla-tokenizer.json" \
  --report "$REPORTS/tokenizer-quality.json" \
  configs/data/lafla-model-identity-380m.jsonl \
  "$TOKENIZER_SAMPLE"
python -m lafla_ai_core.cli.hf_package \
  --tokenizer-json "$WORK/tokenizer/lafla-tokenizer.json" \
  --output-dir "$WORK/hf-package" \
  --model-config configs/model/lafla-380m-thinking.yaml \
  --model-name lafla-380m-thinking

TRAIN_ARGS=(
  python -m lafla_ai_core.cli.train_pretrain
  --model-config configs/model/lafla-380m-thinking.yaml
  --training-config configs/training/lightning-h200-380m-50000.yaml
  --tokenizer-path "$WORK/tokenizer/lafla-tokenizer.json"
  --checkpoint-dir "$WORK/checkpoints"
  --health-log "$REPORTS/train-health.jsonl"
  --data-jsonl configs/data/lafla-model-identity-380m.jsonl
  --data-jsonl "$CLEAN_DATA"
)
if [ -n "${RESUME_FROM:-}" ]; then
  TRAIN_ARGS+=(--resume-from "$RESUME_FROM")
fi
"${TRAIN_ARGS[@]}"

python -m lafla_ai_core.cli.artifact_manifest --root "$WORK" --output "$REPORTS/artifact-manifest.json"
mkdir -p "$WORK/final-checkpoint"
tar -czf "$WORK/final-checkpoint/lafla-380m-thinking-h200-50000-step-run.tar.gz" \
  -C "$WORK/checkpoints" lafla-final \
  -C "$WORK" tokenizer reports hf-package
sync
ls -lh "$WORK/final-checkpoint"
echo "[lafla] finished=$(date -Is)"
