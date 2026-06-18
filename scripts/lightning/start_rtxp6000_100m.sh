#!/usr/bin/env bash
set -Eeuo pipefail

export LAFLA_TRAINING_CONFIG="${LAFLA_TRAINING_CONFIG:-configs/training/lightning/lightning-rtx-pro-6000-100m-quality-fast.yaml}"
exec bash scripts/lightning/start_t4_100m.sh
