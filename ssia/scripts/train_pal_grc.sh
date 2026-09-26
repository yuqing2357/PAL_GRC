#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${SSIA_DATA_ROOT:-${ROOT}/data/ssia_prepared}"; SOURCE="${SSIA_SOURCE_CHECKPOINT:?Set SSIA_SOURCE_CHECKPOINT to the audited epoch-10 checkpoint}"
RUN_ROOT="${SSIA_RUN_ROOT:-${ROOT}}"; RUN_ID="${SSIA_PAL_GRC_RUN_ID:-ssia_pal_grc}"; GPUS="${SSIA_CUDA_VISIBLE_DEVICES:-0,1,2,3}"; NPROC="${SSIA_NPROC_PER_NODE:-4}"
LAMBDA="${SSIA_GRC_WEIGHT:-1.0}"
cd "${ROOT}"; export PYTHONPATH="${ROOT}/common"
CUDA_VISIBLE_DEVICES="${GPUS}" torchrun --standalone --nproc_per_node="${NPROC}" ssia/scripts/train.py \
  --config "${ROOT}/ssia/configs/ssia_pal_grc.yaml" --resume "${SOURCE}" \
  --set "seed=${SSIA_SEED:-20260916}" "data.root=${DATA_ROOT}" "paths.out_dir=${RUN_ROOT}/runs/${RUN_ID}" "loss.group_psd_weight=${LAMBDA}"
