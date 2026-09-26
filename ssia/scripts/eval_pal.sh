#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; DATA_ROOT="${SSIA_DATA_ROOT:-${ROOT}/data/ssia_prepared}"; RUN_ROOT="${SSIA_RUN_ROOT:-${ROOT}}"; RUN_ID="${SSIA_PAL_RUN_ID:-ssia_pal}"
CKPT="${SSIA_PAL_CHECKPOINT:-${RUN_ROOT}/runs/${RUN_ID}/checkpoints/best.pt}"; OUT="${SSIA_PAL_TEST_OUTPUT:-${RUN_ROOT}/runs/test/${RUN_ID}_test}"
cd "${ROOT}"; export PYTHONPATH="${ROOT}/common"; CUDA_VISIBLE_DEVICES="${SSIA_CUDA_VISIBLE_DEVICES:-0,1,2,3}" torchrun --standalone --nproc_per_node="${SSIA_NPROC_PER_NODE:-4}" ssia/scripts/evaluate.py --run-dir "${RUN_ROOT}/runs/${RUN_ID}" --checkpoint "${CKPT}" --split test --output-dir "${OUT}" --data-root "${DATA_ROOT}" --weights ema --batch-size "${SSIA_TEST_BATCH_PER_GPU:-8}" --num-workers "${SSIA_NUM_WORKERS:-4}" --prefetch-factor 2 --worker-threads 1 --qualitative-count 0 --export-predictions
