#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/teamspace/studios/this_studio}"
WORK="${WORK:-$ROOT/LaflaAI100M}"
export NOHUP_LOG="${NOHUP_LOG:-$WORK/reports/lightning-h100-100m-nohup.log}"

exec bash scripts/lightning/monitor_100m.sh "$@"
