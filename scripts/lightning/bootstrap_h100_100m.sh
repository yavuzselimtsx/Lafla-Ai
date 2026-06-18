#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/teamspace/studios/this_studio}"
WORK="${WORK:-$ROOT/LaflaAI100M}"
RELEASE_REPO="${RELEASE_REPO:-yavuzselimtsx/Lafla-Ai}"
RELEASE_TAG="${RELEASE_TAG:-lafla-100m-backup-20260618}"
BASE_URL="https://github.com/$RELEASE_REPO/releases/download/$RELEASE_TAG"

PRETRAIN_ASSET="pretrain-lafla-step-019500.tar.zst"
RUNTIME_ASSET="runtime-metadata.tar.zst"
CHECKSUM_ASSET="SHA256SUMS"

for command_name in curl sha256sum zstd tar python3 nvidia-smi; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "Gerekli komut bulunamadi: $command_name" >&2
    exit 2
  }
done

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
case "$GPU_NAME" in
  *"NVIDIA H100"*) ;;
  *) echo "H100 bekleniyordu, bulunan GPU: $GPU_NAME" >&2; exit 2 ;;
esac

if [ -e "$WORK" ]; then
  echo "Restore hedefi zaten var; uzerine yazilmadi: $WORK" >&2
  exit 2
fi

STAGING="$(mktemp -d "$ROOT/.lafla-h100-restore.XXXXXX")"
trap 'rm -rf "$STAGING"' EXIT
DOWNLOADS="$STAGING/downloads"
EXTRACTED="$STAGING/extracted"
PREPARED="$STAGING/LaflaAI100M"
mkdir -p "$DOWNLOADS" "$EXTRACTED" "$PREPARED/checkpoints"

download_asset() {
  local name="$1"
  echo "[lafla] download=$name"
  curl --fail --location --retry 5 --retry-all-errors \
    --output "$DOWNLOADS/$name" "$BASE_URL/$name"
}

download_asset "$CHECKSUM_ASSET"
download_asset "$PRETRAIN_ASSET"
download_asset "$RUNTIME_ASSET"

: > "$DOWNLOADS/SHA256SUMS.required"
for asset in "$PRETRAIN_ASSET" "$RUNTIME_ASSET"; do
  awk -v name="$asset" '$2 == name {print}' "$DOWNLOADS/$CHECKSUM_ASSET" \
    >> "$DOWNLOADS/SHA256SUMS.required"
done
test "$(wc -l < "$DOWNLOADS/SHA256SUMS.required")" -eq 2 || {
  echo "Release checksum manifest gerekli iki artifact'i icermiyor" >&2
  exit 2
}
(
  cd "$DOWNLOADS"
  sha256sum -c SHA256SUMS.required
)
zstd -t "$DOWNLOADS/$PRETRAIN_ASSET"
zstd -t "$DOWNLOADS/$RUNTIME_ASSET"

zstd -dc "$DOWNLOADS/$PRETRAIN_ASSET" | tar -C "$EXTRACTED" -xf -
zstd -dc "$DOWNLOADS/$RUNTIME_ASSET" | tar -C "$EXTRACTED" -xf -

CHECKPOINT="$EXTRACTED/lafla-step-019500"
RUNTIME="$EXTRACTED/runtime-metadata"
for required_file in READY.json config.json model.pt optimizer.pt rng.pt trainer_state.json; do
  test -s "$CHECKPOINT/$required_file" || {
    echo "Checkpoint dosyasi eksik: $required_file" >&2
    exit 2
  }
done
test -s "$RUNTIME/tokenizer/lafla-tokenizer.json" || { echo "Tokenizer eksik" >&2; exit 2; }
test -s "$RUNTIME/data/train.jsonl" || { echo "Gercek pretraining verisi eksik" >&2; exit 2; }
test -s "$RUNTIME/data/veri_manifesti.json" || { echo "Veri manifesti eksik" >&2; exit 2; }

TRAINER_STATE="$CHECKPOINT/trainer_state.json" python3 - <<'PY'
import json
import os
from pathlib import Path

state = json.loads(Path(os.environ["TRAINER_STATE"]).read_text(encoding="utf-8"))
state_format = state.get("format")
if state_format == "lafla-thinking-sft-state-v1":
    raise SystemExit("SFT checkpoint pretraining resume kaynagi olamaz")
if state_format != "lafla-trainer-state-v2":
    raise SystemExit(f"pretraining trainer state formati gecersiz: {state_format!r}")
if bool(state.get("smoke")):
    raise SystemExit("smoke checkpoint gercek egitimde kullanilamaz")
if int(state.get("step", -1)) != 19500:
    raise SystemExit(f"beklenmeyen checkpoint step: {state.get('step')!r}")
if int(state.get("cumulative_tokens", 0)) <= 0:
    raise SystemExit("checkpoint cumulative_tokens pozitif olmali")
PY

mv "$CHECKPOINT" "$PREPARED/checkpoints/"
mv "$RUNTIME/tokenizer" "$PREPARED/"
mv "$RUNTIME/data" "$PREPARED/"
if [ -d "$RUNTIME/reports" ]; then
  mv "$RUNTIME/reports" "$PREPARED/"
else
  mkdir "$PREPARED/reports"
fi
mv "$PREPARED" "$WORK"
sync

echo "[lafla] restore_ok=$WORK"
echo "[lafla] checkpoint=$WORK/checkpoints/lafla-step-019500"
echo "[lafla] tokenizer=$WORK/tokenizer/lafla-tokenizer.json"
echo "[lafla] data=$WORK/data/train.jsonl"
