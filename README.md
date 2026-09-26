# Anonymous reproduction code

Minimal training and evaluation entry points for the paper's two domains.

## Install

```bash
python -m pip install -r requirements.txt
```

The experiments use Python 3.10+ and CUDA-enabled PyTorch.  Launchers use
four GPUs by default; set `CMU_CUDA_VISIBLE_DEVICES` or
`SSIA_CUDA_VISIBLE_DEVICES` to change this.

## Data

Data are not included.  Set paths with environment variables:

```bash
export CMU_DATA_ROOT=/path/to/cmu_arctic_mfcc39_k4_full_alignment_v1
export CMU_FEATURE_ROOT=/path/to/cmu_arctic_native_mfcc_protocol_audit_v2
export SSIA_DATA_ROOT=/path/to/prepared_ssia
```

CMU uses the prepared K=4 native-MFCC manifests. SSIA uses the prepared K=8
impedance/RGT arrays with `train`, `validation`, and `test` splits.

## CMU ARCTIC

Configs:

```text
cmu_arctic/configs/cmu_pal.yaml
cmu_arctic/configs/cmu_pal_grc.yaml
```

PAL+GRC is a continuation from the same PAL source checkpoint, not a separate
from-scratch model:

```bash
export CMU_SOURCE_CHECKPOINT=/path/to/pal_source_checkpoint.pt
bash cmu_arctic/scripts/train_pal.sh
bash cmu_arctic/scripts/train_pal_grc.sh
bash cmu_arctic/scripts/eval_pal.sh
bash cmu_arctic/scripts/eval_pal_grc.sh
```

Use `CMU_PAL_CHECKPOINT` and `CMU_PAL_GRC_CHECKPOINT` to evaluate an existing
checkpoint without retraining.

## SSIA

Configs:

```text
ssia/configs/ssia_pal.yaml
ssia/configs/ssia_pal_grc.yaml
```

```bash
export SSIA_SOURCE_CHECKPOINT=/path/to/pal_source_checkpoint.pt
bash ssia/scripts/train_pal.sh
bash ssia/scripts/train_pal_grc.sh
bash ssia/scripts/eval_pal.sh
bash ssia/scripts/eval_pal_grc.sh
```

Use `SSIA_PAL_CHECKPOINT` and `SSIA_PAL_GRC_CHECKPOINT` for frozen evaluation.

## Outputs

Each evaluation writes `summary.json`, pairwise metrics, transitive metrics,
F@0--F@20, IoU, PathMAE, TransitiveMAE, TransitiveP95, and prediction files
under the selected run root.

The repository contains no raw data, checkpoints, logs, caches, author
metadata, or machine-specific absolute paths.
