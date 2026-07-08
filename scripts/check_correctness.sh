#!/usr/bin/env bash
set -euo pipefail

mode="${1:-quick}"

cargo fmt --all --check
cargo check -q
cargo test -q

case "$mode" in
  quick)
    ;;
  long)
    DODAM_LONG_DIFF=1 cargo test -q --test duckdb_differential \
      duckdb_differential_long_run_seeded_randomized -- --nocapture
    ;;
  *)
    echo "usage: $0 [quick|long]" >&2
    exit 2
    ;;
esac
