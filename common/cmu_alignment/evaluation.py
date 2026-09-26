"""D34-style index metrics on CMU's native-length K=4 alignments.

Predictions are decoded exactly once from the CRF. Ground truth enters only
after decoding, for metric computation. The implementation mirrors D34's
continuous pair/triple metric definitions while replacing its K=8,
volume-level hierarchy with CMU's K=4, parent-prompt hierarchy.  All formal
metrics use the K=4 intersection of phoneme-labelled content domains; raw
leading/trailing recording context never enters a reported error.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from cmu_alignment.data import PAIR_ORDER
from cmu_alignment.partial_overlap_crf import calibrated_pair_scores, is_directional_emission_mode, ragged_pair_scoring_lattices, viterbi_decode_one


F1_TOLERANCES = tuple(range(21))
EVALUATION_PROTOCOL = "CMU_CONTINUOUS_ALIGNMENT_METRICS_V6_COMMON_K4_CONTENT_NATIVE_FRAME_INDEX"


def _iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    if min(*a, *b) < 0:
        return 0.0
    left, right = max(a[0], b[0]), min(a[1], b[1])
    return float(max(0.0, right - left + 1.0) / max(1.0, max(a[1], b[1]) - min(a[0], b[0]) + 1.0))


def _bounds(path: np.ndarray) -> tuple[float, float]:
    active = np.flatnonzero(np.asarray(path) >= 0)
    return (-1.0, -1.0) if not len(active) else (float(active[0]), float(active[-1]))


def _target_bounds(path: np.ndarray, active: np.ndarray) -> tuple[float, float]:
    return (-1.0, -1.0) if not active.any() else (float(np.min(path[active])), float(np.max(path[active])))


def d34_pair_metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    """Literal D34 pair-metric logic, with CMU native frame indices as units."""
    pred = np.asarray(pred, np.float32)
    gt = np.asarray(gt, np.float32)
    valid = np.asarray(valid, bool)
    active = pred >= 0
    matched = active & valid
    result: dict[str, Any] = {"path_mae_index": float(np.abs(pred[matched] - gt[matched]).mean()) if matched.any() else float("nan")}
    for tolerance in F1_TOLERANCES:
        correct = matched & (np.abs(pred - gt) <= tolerance)
        tp = int(correct.sum())
        fp = int((active & ~correct).sum())
        fn = int((valid & ~correct).sum())
        result.update({
            f"f1_tol_{tolerance:02d}_index": 2.0 * tp / max(1, 2 * tp + fp + fn),
            f"tp_tol_{tolerance:02d}": tp,
            f"fp_tol_{tolerance:02d}": fp,
            f"fn_tol_{tolerance:02d}": fn,
        })
    source_pred, source_gt = _bounds(pred), _bounds(np.where(valid, gt, -1.0))
    target_pred, target_gt = _target_bounds(pred, active), _target_bounds(gt, valid)
    source_iou, target_iou = _iou(source_pred, source_gt), _iou(target_pred, target_gt)
    result.update({
        "source_iou": source_iou,
        "target_iou": target_iou,
        "interval_iou": 0.5 * (source_iou + target_iou),
        "predicted_has_overlap": int(active.any()),
        "predicted_source_bounds_inclusive": list(source_pred),
        "predicted_target_bounds_inclusive": list(target_pred),
        "gt_source_bounds_inclusive": list(source_gt),
        "gt_target_bounds_inclusive": list(target_gt),
    })
    return result


def cmu_prompt_interval_metrics(pred: np.ndarray, source_prompt_valid: np.ndarray, target_prompt_valid: np.ndarray) -> dict[str, Any]:
    """CMU interval metrics on the supplied native content domains.

    Formal V6 passes the K=4-common phoneme-labelled content masks here.
    The historical function name is retained only for callers that consume
    the old field names; neither input is cropped or renumbered.
    """
    pred = np.asarray(pred, np.float32)
    source_prompt_valid = np.asarray(source_prompt_valid, bool)
    target_prompt_valid = np.asarray(target_prompt_valid, bool)
    source_gt = _bounds(np.where(source_prompt_valid, 0.0, -1.0))
    target_gt = _bounds(np.where(target_prompt_valid, 0.0, -1.0))
    if min(*source_gt, *target_gt) < 0:
        raise ValueError("every CMU evaluation sequence requires a nonempty common content domain")
    active = pred >= 0
    source_pred = _bounds(pred)
    target_pred = _target_bounds(pred, active)
    source_iou, target_iou = _iou(source_pred, source_gt), _iou(target_pred, target_gt)
    if active.any():
        source_boundary = 0.5 * (abs(source_pred[0] - source_gt[0]) + abs(source_pred[1] - source_gt[1]))
        target_boundary = 0.5 * (abs(target_pred[0] - target_gt[0]) + abs(target_pred[1] - target_gt[1]))
    else:
        # There are no predicted endpoints.  Penalize the known common
        # content extent, never the padded/native waveform length.
        source_boundary = source_gt[1] - source_gt[0] + 1.0
        target_boundary = target_gt[1] - target_gt[0] + 1.0
    boundary = 0.5 * (source_boundary + target_boundary)
    return {
        "source_iou": source_iou,
        "target_iou": target_iou,
        "interval_iou": 0.5 * (source_iou + target_iou),
        "common_content_interval_iou": 0.5 * (source_iou + target_iou),
        "source_prompt_boundary_mae_index": float(source_boundary),
        "target_prompt_boundary_mae_index": float(target_boundary),
        "prompt_boundary_mae_index": float(boundary),
        # Existing CMU reports used the following names.  Keep them as exact
        # aliases, now with prompt-domain semantics rather than pair-overlap.
        "boundary_mae_frames": float(boundary),
        "boundary_mae_ms": float(boundary * 10.0),
        "predicted_source_bounds_inclusive": list(source_pred),
        "predicted_target_bounds_inclusive": list(target_pred),
        "source_prompt_bounds_inclusive": list(source_gt),
        "target_prompt_bounds_inclusive": list(target_gt),
    }


def _compose_continuously(qij: np.ndarray, qjk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """D34's continuous q_jk(q_ij(p)) composition; never integer-cast paths."""
    qij = np.asarray(qij, np.float32)
    qjk = np.asarray(qjk, np.float32)
    value = np.full_like(qij, -1.0)
    inside = (qij >= 0) & (qij <= len(qjk) - 1)
    lo = np.floor(np.clip(qij, 0, len(qjk) - 1)).astype(np.int64)
    hi = np.ceil(np.clip(qij, 0, len(qjk) - 1)).astype(np.int64)
    usable = inside & (qjk[lo] >= 0) & (qjk[hi] >= 0)
    weight = qij - lo
    value[usable] = (1.0 - weight[usable]) * qjk[lo[usable]] + weight[usable] * qjk[hi[usable]]
    return value, usable


def d34_transitive_metric(qij: np.ndarray, qjk: np.ndarray, gtik: np.ndarray, validik: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    """D34's continuous ordered-triple transitive error definition."""
    composed, composed_valid = _compose_continuously(qij, qjk)
    gtik = np.asarray(gtik, np.float32)
    validik = np.asarray(validik, bool)
    domain = validik & composed_valid
    errors = np.abs(composed[domain] - gtik[domain]).astype(np.float32, copy=False)
    return {
        "transitive_mae_index": float(errors.mean()) if len(errors) else float("nan"),
        "transitive_p95_index": float(np.percentile(errors, 95)) if len(errors) else float("nan"),
        "transitive_valid_points": int(len(errors)),
        "transitive_gt_valid_points": int(validik.sum()),
        "transitive_composed_valid_points": int(composed_valid.sum()),
    }, errors


# Compatibility helpers retained for existing tiny-overfit/smoke consumers.
def path_metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, predicted_source: tuple[int, int], predicted_target: tuple[int, int], gt_source: tuple[int, int], gt_target: tuple[int, int], tolerance: float = 2.0) -> dict[str, float | int]:
    active = pred >= 0
    correct = valid & active & (np.abs(pred - gt) <= tolerance)
    tp, fp, fn = int(correct.sum()), int((active & ~correct).sum()), int((valid & ~correct).sum())
    precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
    matched = valid & active
    mae = float(np.abs(pred[matched] - gt[matched]).mean()) if matched.any() else float("nan")
    missed = not active.any()
    boundary = float(len(gt)) if missed else float((abs(predicted_source[0] - gt_source[0]) + abs(predicted_source[1] - gt_source[1]) + abs(predicted_target[0] - gt_target[0]) + abs(predicted_target[1] - gt_target[1])) / 4)
    return {"tp": tp, "fp": fp, "fn": fn, "path_f1": float(2 * precision * recall / max(1e-12, precision + recall)), "path_mae_frames": mae, "path_mae_ms": mae * 10, "boundary_mae_frames": boundary, "boundary_mae_ms": boundary * 10, "source_iou": _iou(predicted_source, gt_source), "target_iou": _iou(predicted_target, gt_target), "interval_iou": .5 * (_iou(predicted_source, gt_source) + _iou(predicted_target, gt_target)), "predicted_has_overlap": int(not missed)}


def cycle_closure_mae(q_ij: np.ndarray, q_jk: np.ndarray, q_ik: np.ndarray) -> tuple[float, float]:
    """Legacy integer-index cycle metric; the formal metric is transitive MAE."""
    source = np.flatnonzero((q_ij >= 0) & (q_ik >= 0))
    if not len(source):
        return float("nan"), 0.0
    via = q_ij[source].astype(int)
    keep = (via >= 0) & (via < len(q_jk)) & (q_jk[via] >= 0)
    if not keep.any():
        return float("nan"), 0.0
    return float(np.abs(q_jk[via[keep]] - q_ik[source[keep]]).mean()), float(keep.mean())


def parent_balanced(rows: list[dict], field: str) -> float:
    groups = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is None:
            continue
        value = float(value)
        if np.isfinite(value):
            groups[row["parent_prompt_id"]].append(value)
    return float(np.mean([np.mean(values) for values in groups.values()]))


def _hierarchical_parent_mean(rows: list[dict[str, Any]], field: str) -> float:
    """D34 hierarchy adapted as pair → K4 group → parent prompt → dataset."""
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[field])
        if np.isfinite(value):
            groups[(str(row["parent_prompt_id"]), str(row["k4_group_id"]))].append(value)
    parents: dict[str, list[float]] = defaultdict(list)
    for (parent, _group), values in groups.items():
        parents[parent].append(float(np.mean(values)))
    return float(np.mean([np.mean(values) for values in parents.values()])) if parents else float("nan")


def summarize_d34_index_metrics(pair_rows: list[dict[str, Any]], triple_rows: list[dict[str, Any]], transitive_errors: np.ndarray, *, groups: int) -> dict[str, Any]:
    """D34 summary schema with CMU's prompt hierarchy and native-frame units."""
    if not pair_rows:
        raise ValueError("cannot summarize zero evaluated groups")
    errors = np.asarray(transitive_errors, np.float32)
    f1 = {str(tolerance): _hierarchical_parent_mean(pair_rows, f"f1_tol_{tolerance:02d}_index") for tolerance in F1_TOLERANCES}
    common_content_interval_iou = _hierarchical_parent_mean(pair_rows, "interval_iou")
    return {
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "coordinate_unit": "native MFCC frame index",
        "groups": int(groups),
        "directed_pairs": len(pair_rows),
        "ordered_triples": len(triple_rows),
        "pairwise_aggregation": "position→directed pair→K4 subgroup→parent prompt→dataset (equal pair weight)",
        "groupwise_aggregation": "all valid pointwise transitive errors pooled across ordered triples; MAE and P95 use the identical pool",
        "path_mae_index": _hierarchical_parent_mean(pair_rows, "path_mae_index"),
        "f1_tolerance_0_20": f1,
        "common_content_interval_iou": common_content_interval_iou,
        # Compatibility alias retained for existing result readers.  In V6,
        # unqualified interval_iou is the K=4-common labelled-content IoU.
        "prompt_interval_iou": common_content_interval_iou,
        "interval_iou": common_content_interval_iou,
        "prompt_boundary_mae_index": _hierarchical_parent_mean(pair_rows, "prompt_boundary_mae_index"),
        "transitive_error_points": int(len(errors)),
        "transitive_mae_index": float(errors.mean()) if len(errors) else float("nan"),
        "transitive_p95_index": float(np.percentile(errors, 95)) if len(errors) else float("nan"),
        "predicted_has_overlap_rate": _hierarchical_parent_mean(pair_rows, "predicted_has_overlap"),
    }


@dataclass
class D34EvaluationBatch:
    pair_rows: list[dict[str, Any]]
    triple_rows: list[dict[str, Any]]
    transitive_errors: np.ndarray


def evaluate_decoded_paths_d34(batch: dict[str, Any], decoded_paths: list[dict[tuple[int, int], np.ndarray]]) -> D34EvaluationBatch:
    """Evaluate externally decoded native-frame paths with the frozen CMU V5 protocol.

    This is deliberately model-agnostic: a learned CRF decoder, a classical
    DTW baseline, or a later pairwise model supplies one ``[T_source]``
    continuous target-index path for every directed K=4 pair.  Ground-truth
    arrays are read only here, after those paths already exist.
    """
    if len(decoded_paths) != len(batch["k4_group_id"]):
        raise ValueError("one decoded-path dictionary is required per K4 group")
    pair_rows: list[dict[str, Any]] = []
    triple_rows: list[dict[str, Any]] = []
    error_chunks: list[np.ndarray] = []
    required = set(PAIR_ORDER)
    evaluation_pair_valid = batch.get("evaluation_pair_valid", batch["pair_valid"])
    for b, decoded in enumerate(decoded_paths):
        if set(decoded) != required:
            raise ValueError("decoded paths must contain every one of the 12 directed K4 pairs")
        truth: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        normalized: dict[tuple[int, int], np.ndarray] = {}
        for i, j in PAIR_ORDER:
            source_length = int(batch["lengths"][b, i])
            target_length = int(batch["lengths"][b, j])
            path = np.asarray(decoded[i, j], dtype=np.float32)
            if path.shape != (source_length,):
                raise ValueError(f"decoded path {(i, j)} has {path.shape}, expected {(source_length,)}")
            active = path >= 0
            if not np.isfinite(path).all() or (active.any() and (path[active].max() > target_length - 1 or path[active].min() < 0)):
                raise ValueError(f"decoded path {(i, j)} contains invalid target coordinates")
            normalized[i, j] = path
            truth[i, j] = (
                batch["q_of_p"][b, i, j, :source_length].detach().cpu().numpy(),
                evaluation_pair_valid[b, i, j, :source_length].detach().cpu().numpy(),
            )
        common = {"k4_group_id": batch["k4_group_id"][b], "parent_prompt_id": batch["parent_prompt_id"][b]}
        for i, j in PAIR_ORDER:
            path = normalized[i, j]
            q, valid = truth[i, j]
            active = path >= 0
            # ``evaluation_pair_valid`` restricts point metrics to the
            # K4-common domain.  Use that identical domain for interval IoU
            # and boundary diagnostics rather than the complete recording
            # content span of an individual speaker.
            group_common = batch.get("group_common_valid", batch["reference_valid"])
            source_prompt_valid = group_common[b, i, :int(batch["lengths"][b, i])].detach().cpu().numpy()
            target_prompt_valid = group_common[b, j, :int(batch["lengths"][b, j])].detach().cpu().numpy()
            ids = np.flatnonzero(valid)
            if not len(ids):
                raise ValueError("formal CMU pair ground truth must have nonempty support")
            gt_source = (int(ids[0]), int(ids[-1]))
            gt_target = (int(round(q[ids].min())), int(round(q[ids].max())))
            metric = path_metrics(path, q, valid, _bounds(path), _target_bounds(path, active), gt_source, gt_target, tolerance=2.0)
            metric.update(d34_pair_metrics(path, q, valid))
            # Preserve pair-overlap diagnostics while making the unqualified
            # interval/boundary fields use the K4-common content domain.
            metric.update({f"pair_overlap_{key}": metric[key] for key in ("source_iou", "target_iou", "interval_iou", "gt_source_bounds_inclusive", "gt_target_bounds_inclusive")})
            metric.update(cmu_prompt_interval_metrics(path, source_prompt_valid, target_prompt_valid))
            metric.update(common | {"source": i, "target": j, "predicted_q": path.tolist()})
            pair_rows.append(metric)
        for source in range(4):
            for via in range(4):
                for target in range(4):
                    if len({source, via, target}) < 3:
                        continue
                    gt, valid = truth[source, target]
                    metric, errors = d34_transitive_metric(normalized[source, via], normalized[via, target], gt, valid)
                    metric.update(common | {"source": source, "via": via, "target": target})
                    triple_rows.append(metric)
                    if len(errors):
                        error_chunks.append(errors)
    return D34EvaluationBatch(pair_rows, triple_rows, np.concatenate(error_chunks) if error_chunks else np.empty(0, dtype=np.float32))


@torch.no_grad()
def evaluate_batch_d34(
    model,
    batch: dict,
    device: torch.device,
    crf_reference_length: float,
    *,
    allow_no_overlap_output: bool = True,
    emission_mode: str = "legacy",
    similarity_eps: float = 1e-6,
    normalization_eps: float = 1e-8,
) -> D34EvaluationBatch:
    """Decode one true-length CMU batch and materialize D34-style pair/triple rows."""
    model.eval()
    tensor = {key: value.to(device, non_blocking=device.type == "cuda") for key, value in batch.items() if isinstance(value, torch.Tensor)}
    out = model(tensor["x"], tensor["sequence_valid_mask"], tensor["lengths"])
    directional = is_directional_emission_mode(emission_mode)
    # Directional modes must preserve the one-unordered-GEMM contract at
    # inference too.  Decode all routed views from this shared score object.
    scoring = ragged_pair_scoring_lattices(
        out.matching_features, tensor["lengths"], out.group_crf_alpha, out.group_crf_beta,
        reference_length=crf_reference_length, similarity_mode=out.matching_mode,
        emission_mode=emission_mode, crf_temperature=getattr(out, "group_crf_temperature", None),
        gamma=getattr(out, "group_crf_gamma", None), eps=similarity_eps,
        normalization_eps=normalization_eps,
    ) if directional else None
    decoded_batches: list[dict[tuple[int, int], np.ndarray]] = []
    for b in range(len(batch["k4_group_id"])):
        decoded: dict[tuple[int, int], dict[str, Any]] = {}
        for pair_index, (i, j) in enumerate(PAIR_ORDER):
            source_length, target_length = int(tensor["lengths"][b, i]), int(tensor["lengths"][b, j])
            score = scoring.crf_input[b, pair_index, :source_length, :target_length] if directional else calibrated_pair_scores(
                out.matching_features[b, i, :source_length], out.matching_features[b, j, :target_length],
                out.group_crf_alpha, out.group_crf_beta, crf_reference_length,
                similarity_mode=out.matching_mode, emission_mode=emission_mode,
                crf_temperature=getattr(out, "group_crf_temperature", None),
                gamma=getattr(out, "group_crf_gamma", None),
                eps=similarity_eps, normalization_eps=normalization_eps,
            )
            decoded[i, j] = viterbi_decode_one(
                score, allow_empty=allow_no_overlap_output,
                input_is_emission=directional,
            )
        decoded_batches.append({pair: path["q"] for pair, path in decoded.items()})
    return evaluate_decoded_paths_d34(batch, decoded_batches)


@torch.no_grad()
def evaluate_batch(model, batch: dict, device: torch.device, crf_reference_length: float, tolerance: float = 2.0, *, allow_no_overlap_output: bool = True, emission_mode: str = "legacy", similarity_eps: float = 1e-6, normalization_eps: float = 1e-8) -> list[dict[str, Any]]:
    """Backward-compatible pair-row interface; formal F1 tolerance is fixed at 2."""
    if tolerance != 2.0:
        raise ValueError("formal CMU D34-style evaluation fixes F1 tolerance at 2 index points")
    return evaluate_batch_d34(model, batch, device, crf_reference_length, allow_no_overlap_output=allow_no_overlap_output, emission_mode=emission_mode, similarity_eps=similarity_eps, normalization_eps=normalization_eps).pair_rows
