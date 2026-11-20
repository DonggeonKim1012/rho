#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
for batch in 1 2 4 8 16 32 64 128 256 512; do
  echo "PROBE batch=$batch"
  if OPENBLAS_NUM_THREADS=1 .venv/bin/python sh5/train.py --batch "$batch" --probe --steps 12 > "work/sh5/probe${batch}.log" 2>&1; then
    cat "work/sh5/probe_$batch/completion.json"
  else
    tail -25 "work/sh5/probe${batch}.log"
    exit 1
  fi
done
