#!/usr/bin/env bash
set -euo pipefail

SF="${1:-0.01}"
OUTPUT_DIR="${2:-/tmp/dodam-tpchgen-sf${SF//./_}}"
THREADS="${TPCHGEN_THREADS:-8}"

if ! command -v tpchgen-cli >/dev/null 2>&1; then
  echo "tpchgen-cli is required. Install it with:" >&2
  echo "  cargo install tpchgen-cli --version 3.0.0 --locked" >&2
  exit 1
fi

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

tpchgen-cli parquet \
  --scale-factor "${SF}" \
  --output-dir="${OUTPUT_DIR}" \
  --num-threads="${THREADS}" \
  --no-progress

echo "${OUTPUT_DIR}"
