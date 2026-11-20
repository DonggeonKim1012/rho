#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false WANDB_MODE=online
exec .venv/bin/python -u sh5/train.py --batch "${1:?batch required}" --hours 48 "${@:2}"
