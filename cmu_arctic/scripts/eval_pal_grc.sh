#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${ROOT}"
DATA_ROOT="${CMU_DATA_ROOT:-${ROOT}/data/cmu_arctic_mfcc39_k4_full_alignment_v1}"; FEATURE_ROOT="${CMU_FEATURE_ROOT:-${ROOT}/data/cmu_arctic_native_mfcc_protocol_audit_v2}"
RUN_ROOT="${CMU_RUN_ROOT:-${ROOT}}"; RUN_ID="${CMU_PAL_GRC_RUN_ID:-cmu_pal_grc}"; CKPT="${CMU_PAL_GRC_CHECKPOINT:-${RUN_ROOT}/runs/main/${RUN_ID}/checkpoints/best.pt}"
export PYTHONPATH="${ROOT}/common:${ROOT}/cmu_arctic/src"; CUDA_VISIBLE_DEVICES="${CMU_CUDA_VISIBLE_DEVICES:-0,1,2,3}" torchrun --standalone --nproc_per_node="${CMU_NPROC_PER_NODE:-4}" cmu_arctic/scripts/test.py --root "${DATA_ROOT}" --feature-root "${FEATURE_ROOT}" --checkpoint "${CKPT}" --model-config "${ROOT}/cmu_arctic/configs/cmu_pal_grc.yaml" --run-root "${RUN_ROOT}" --run-id "${RUN_ID}_test" --split test --weights ema --batch-size "${CMU_TEST_BATCH_PER_GPU:-96}" --num-workers "${CMU_NUM_WORKERS:-8}" --ddp
