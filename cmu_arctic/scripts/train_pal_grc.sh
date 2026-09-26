#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${CMU_DATA_ROOT:-${ROOT}/data/cmu_arctic_mfcc39_k4_full_alignment_v1}"
FEATURE_ROOT="${CMU_FEATURE_ROOT:-${ROOT}/data/cmu_arctic_native_mfcc_protocol_audit_v2}"
SOURCE="${CMU_SOURCE_CHECKPOINT:?Set CMU_SOURCE_CHECKPOINT to the same audited epoch-60 checkpoint used by PAL}"
RUN_ROOT="${CMU_RUN_ROOT:-${ROOT}}"; RUN_ID="${CMU_PAL_GRC_RUN_ID:-cmu_pal_grc}"
GPUS="${CMU_CUDA_VISIBLE_DEVICES:-0,1,2,3}"; NPROC="${CMU_NPROC_PER_NODE:-4}"
LAMBDA="${CMU_GRC_WEIGHT:-1.0}"
cd "${ROOT}"; export PYTHONPATH="${ROOT}/common:${ROOT}/cmu_arctic/src"
CUDA_VISIBLE_DEVICES="${GPUS}" torchrun --standalone --nproc_per_node="${NPROC}" \
  cmu_arctic/scripts/train.py --data-root "${DATA_ROOT}" --feature-root "${FEATURE_ROOT}" \
  --model-config "${ROOT}/cmu_arctic/configs/cmu_pal_grc.yaml" --run-root "${RUN_ROOT}" \
  --run-id "${RUN_ID}" --resume "${SOURCE}" --resume-mode staged_continuation \
  --staged-condition pal_grc --group-psd-weight "${LAMBDA}" --seed "${CMU_SEED:-20260916}" \
  --max-steps "${CMU_MAX_STEPS:-1200}" --per-rank-batch-size "${CMU_BATCH_PER_GPU:-96}" \
  --num-workers "${CMU_NUM_WORKERS:-8}" --prefetch-factor 4 --preload-train-to-memory \
  --preload-val-to-memory --amp bf16 --learning-rate 3e-4 --warmup-steps 222 \
  --min-lr-ratio 0.1 --grad-clip-norm 1.0 --ema-decay 0.9995 --ema-step-scale 1.5 \
  --early-stop-patience 10 --early-stop-min-delta 0 --early-stop-min-epochs 10 \
  --no-latest-checkpoint-on-validation --val-every 6 --checkpoint-every-epochs 10 \
  --ddp --controlled-experiment E1
