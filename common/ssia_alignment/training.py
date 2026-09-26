"""DDP trainer for full-dataset, non-curriculum D3--D4 learning."""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .data.dataset import D34BridgeDataset, D34PreparedImpRGTDataset, EpochShuffleDDPSampler, FixedIndexBatchSampler, build_loader
from .end.loss import GroupwiseEndLoss, build_end_loss
from .end.model import build_end_model
from .end.partial_overlap_crf import viterbi_decode_batch
from .evaluation import EVALUATION_PROTOCOL, evaluate_predictions
from .runtime import CUDAPrefetcher, ModelEMA, barrier, build_optimizer, build_scheduler, collect_rng_states, ddp_setup, maybe_warm_page_cache, restore_rng_state, seed


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    # Fresh controlled pretraining runs write their initialization manifest
    # before the ordinary run-artifact setup below. Ensure this first write is
    # valid for a previously nonexistent output directory.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_state_sha256(model: torch.nn.Module) -> str:
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf8")); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=True) + "\n")


def _dict(cfg: Any) -> dict[str, Any]:
    return cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)


def _objective(loss_fn: torch.nn.Module) -> str:
    if getattr(loss_fn, "method_version", "").find("direct_cosine") >= 0:
        if float(getattr(loss_fn, "group_psd_weight", 0.0)) > 0.0:
            start_epoch = int(getattr(loss_fn, "group_psd_start_epoch", 1))
            return f"Simplified Always-Overlap Partial-Path CRF (exact r=0) + prediction-only Soft-Route Group PSD from epoch {start_epoch}"
        return "Simplified Always-Overlap Partial-Path CRF (exact r=0)"
    return "OriginalCorridorCRF + Projector" if loss_fn.lambda_projector else "OriginalCorridorCRF"


def _configure_cuda(cfg: Any) -> tuple[bool, torch.dtype]:
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.train.get("tf32", True))
    torch.backends.cudnn.allow_tf32 = bool(cfg.train.get("tf32", True))
    torch.backends.cudnn.benchmark = bool(cfg.train.get("cudnn_benchmark", True))
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    amp = str(cfg.train.get("amp", "bf16")).lower()
    return amp in {"bf16", "fp16"}, torch.bfloat16 if amp == "bf16" else torch.float16


def _loss_output(output: Mapping[str, torch.Tensor], *, optimized: bool) -> dict[str, torch.Tensor]:
    if output.get("matching_mode") == "direct_cosine":
        prepared = dict(output)
        for key in ("matching_features", "group_crf_alpha", "group_crf_beta"):
            value = output.get(key)
            if torch.is_tensor(value) and value.is_floating_point():
                prepared[key] = value.float()
        return prepared
    if not optimized:
        return {key: value.float() if torch.is_tensor(value) and value.is_floating_point() else value for key, value in output.items()}
    membership = output["group_membership"].float()
    prepared = dict(output)
    prepared["group_membership"] = membership
    if output.get("group_assignments") is not None:
        prepared["group_assignments"] = membership
    for key in ("group_crf_alpha", "group_crf_beta"):
        if key in output and torch.is_tensor(output[key]) and output[key].is_floating_point():
            prepared[key] = output[key].float()
    return prepared


def _reduce_logs(logs: Mapping[str, torch.Tensor], distributed: bool) -> dict[str, float]:
    keys = sorted(logs)
    values = torch.stack([logs[key].detach().float() for key in keys])
    if distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= dist.get_world_size()
    return {key: float(value) for key, value in zip(keys, values.cpu())}


def _reduce_weighted_epoch_logs(
    sums: Mapping[str, torch.Tensor], local_groups: int, distributed: bool
) -> dict[str, float]:
    """Return group-weighted epoch means with one DDP reduction."""
    if local_groups < 1:
        raise ValueError("cannot summarize an epoch without training groups")
    keys = sorted(sums)
    values = torch.stack([sums[key].detach().float() for key in keys])
    count = values.new_tensor(float(local_groups))
    packed = torch.cat((values, count[None]))
    if distributed:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    return {key: float(value / packed[-1]) for key, value in zip(keys, packed[:-1])}


def _projector_scale(cfg: Any, step: int) -> float:
    warmup, ramp = int(cfg.loss.get("projector_warmup_steps", 0)), int(cfg.loss.get("projector_ramp_steps", 1))
    return 0.0 if step <= warmup else min(1.0, (step - warmup) / max(1, ramp))


def _model_metadata(cfg: Any, model: torch.nn.Module) -> dict[str, Any]:
    section = dict(cfg["model"] if isinstance(cfg, Mapping) else cfg.model)
    experiment = cfg["experiment"] if isinstance(cfg, Mapping) else cfg.experiment
    if str(section.get("groupwise_mode", "")).lower() == "interaction_direct_cosine":
        return {"model_scale": experiment["model_scale"], "encoder_channels": list(section["channels"]), "embedding_dim": section["proj_dim"], "interaction_heads": section["group_set_heads"], "group_interaction_blocks": section["group_interaction_blocks"], "matching": "raw_direct_cosine", "group_slots": 0, "trainable_param_count": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    group = dict(section["group_alignment"])
    return {"model_scale": experiment["model_scale"], "encoder_channels": list(section["channels"]), "embedding_dim": section["proj_dim"], "slot_dim": group["slot_dim"], "assignment_dim": group["assignment_dim"], "interaction_heads": section["group_set_heads"], "slot_attention_heads": group["heads"], "num_slots": group["num_slots"], "trainable_param_count": sum(p.numel() for p in model.parameters() if p.requires_grad)}


@contextlib.contextmanager
def _ema_weights(model: torch.nn.Module, ema: ModelEMA | None):
    if ema is None:
        yield "raw_model"
        return
    raw = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(ema.state, strict=True)
    try:
        yield "ema_model"
    finally:
        model.load_state_dict(raw, strict=True)


def _decode_paths(output: Mapping[str, torch.Tensor]) -> np.ndarray:
    """Decode every directed END pair exactly as the frozen V4 evaluator does."""
    if output.get("matching_mode") == "direct_cosine":
        features = output["matching_features"].float()
        batch, curves, length, dimension = features.shape
        ids = torch.arange(curves, device=features.device)
        source, target = ids[:, None].expand(curves, curves), ids[None, :].expand(curves, curves)
        select = source != target
        source, target = source[select], target[select]
        forward = source < target
        left_id, right_id = source[forward], target[forward]
        canonical_left, canonical_right = torch.minimum(source, target), torch.maximum(source, target)
        route = ((canonical_left[:, None] == left_id[None, :]) & (canonical_right[:, None] == right_id[None, :])).to(torch.long).argmax(dim=1)
        left = features.index_select(1, left_id).reshape(batch * len(left_id), length, dimension)
        right = features.index_select(1, right_id).reshape(batch * len(right_id), length, dimension)
        unordered = torch.bmm(left, right.transpose(1, 2)).reshape(batch, len(left_id), length, length)
        similarity = unordered.index_select(1, route)
        similarity = torch.where(forward[None, :, None, None], similarity, similarity.transpose(-1, -2))
        score = output["group_crf_alpha"].float() * similarity.clamp(-1.0, 1.0) + output["group_crf_beta"].float()
        decoded = viterbi_decode_batch(score.reshape(batch * len(source), length, length), match_bias=0.0, segment_bias=0.0, allow_empty=False).q_of_p.numpy().reshape(batch, len(source), length)
        paths = np.full((batch, curves, curves, length), -1.0, dtype=np.float32)
        paths[:, source.detach().cpu().numpy(), target.detach().cpu().numpy()] = decoded
        return paths
    membership = output["group_membership"].float()
    batch, curves, length, _ = membership.shape
    ids = torch.arange(curves, device=membership.device)
    source, target = ids[:, None].expand(curves, curves), ids[None, :].expand(curves, curves)
    select = source != target
    source, target = source[select], target[select]
    left = membership.index_select(1, source).reshape(batch * len(source), length, -1)
    right = membership.index_select(1, target).reshape(batch * len(target), length, -1)
    correspondence = torch.bmm(left, right.transpose(1, 2))
    score = output["group_crf_alpha"].float() * correspondence.clamp_min(1.0e-6).log() + output["group_crf_beta"].float()
    decoded = viterbi_decode_batch(score, match_bias=0.0, segment_bias=None).q_of_p.numpy().reshape(batch, len(source), length)
    paths = np.full((batch, curves, curves, length), -1.0, dtype=np.float32)
    paths[:, source.detach().cpu().numpy(), target.detach().cpu().numpy()] = decoded
    return paths


def _distributed_transitive_stats(errors: np.ndarray, device: torch.device, world_size: int) -> dict[str, Any]:
    """Exact pooled MAE/P95, matching the final V4 checkpoint evaluator."""
    values = np.asarray(errors, np.float32).reshape(-1)
    values = np.sort(values[np.isfinite(values)])
    totals = torch.tensor([float(values.sum(dtype=np.float64)), float(len(values))], device=device, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    count = int(totals[1].item())
    if count == 0:
        return {"transitive_error_points": 0, "transitive_mae_index": float("nan"), "transitive_p95_index": float("nan")}
    if world_size == 1:
        return {"transitive_error_points": count, "transitive_mae_index": float(totals[0].item() / count), "transitive_p95_index": float(np.percentile(values, 95))}
    bits = values.view(np.uint32)
    lower = torch.tensor(int(bits.min()) if len(bits) else 2**32 - 1, device=device, dtype=torch.int64)
    upper = torch.tensor(int(bits.max()) if len(bits) else 0, device=device, dtype=torch.int64)
    dist.all_reduce(lower, op=dist.ReduceOp.MIN); dist.all_reduce(upper, op=dist.ReduceOp.MAX)

    def kth(rank: int) -> float:
        lo, hi = int(lower.item()), int(upper.item())
        while lo < hi:
            middle = (lo + hi) // 2
            value = np.array([middle], dtype=np.uint32).view(np.float32)[0]
            seen = torch.tensor(int(np.searchsorted(values, value, side="right")), device=device, dtype=torch.int64)
            dist.all_reduce(seen, op=dist.ReduceOp.SUM)
            if int(seen.item()) > rank:
                hi = middle
            else:
                lo = middle + 1
        return float(np.array([lo], dtype=np.uint32).view(np.float32)[0])

    position = 0.95 * (count - 1)
    left, right = int(math.floor(position)), int(math.ceil(position))
    left_value, right_value = kth(left), kth(right)
    return {"transitive_error_points": count, "transitive_mae_index": float(totals[0].item() / count), "transitive_p95_index": left_value + (position - left) * (right_value - left_value)}


def _hierarchical_group_mean(rows: list[dict[str, Any]], key: str) -> float:
    by_volume: dict[int, list[float]] = {}
    for row in rows:
        value = float(row[key])
        if np.isfinite(value):
            by_volume.setdefault(int(row["volume_id"]), []).append(value)
    return float(np.mean([np.mean(values) for values in by_volume.values()])) if by_volume else float("nan")


@torch.no_grad()
def evaluate_validation_loss(model: torch.nn.Module, loss_fn: torch.nn.Module, loader, device: torch.device, *, amp_enabled: bool, amp_dtype: torch.dtype, distributed: bool, ema: ModelEMA | None, expected_groups: int, report_f1_tolerance_index: int, metric_decode_batch_size: int) -> dict[str, Any]:
    """Full EMA validation loss plus the frozen five V4 monitoring metrics."""
    started = time.monotonic()
    was_training, prior_scale = model.training, loss_fn.projector_scale
    model.eval(); loss_fn.set_projector_scale(1.0)
    total = torch.zeros((), device=device, dtype=torch.float64)
    crf = torch.zeros((), device=device, dtype=torch.float64)
    count = torch.zeros((), device=device, dtype=torch.float64)
    group_metrics: dict[tuple[int, int], dict[str, list[float] | int]] = {}
    error_chunks: list[np.ndarray] = []
    with _ema_weights(model, ema) as weights:
        prefetcher = CUDAPrefetcher(loader, device)
        while True:
            try:
                batch = prefetcher.next()
            except StopIteration:
                break
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                output = model(batch["z"], batch["mask"])
            value, logs = loss_fn(_loss_output(output, optimized=loss_fn.performance_optimization_enabled), batch, compute_diagnostics=False)
            groups = int(batch["z"].shape[0])
            total += value.double() * groups; crf += logs["l_crf"].double() * groups; count += groups
            # Loss uses the established validation batch.  Decode in smaller
            # chunks because Viterbi retains [pairs, 512, 512] predecessor
            # state; this protects the training allocation from an OOM while
            # still evaluating every validation group exactly once.
            for start in range(0, groups, metric_decode_batch_size):
                stop = min(groups, start + metric_decode_batch_size)
                metric_output = ({"matching_mode": "direct_cosine", "matching_features": output["matching_features"][start:stop], "group_crf_alpha": output["group_crf_alpha"], "group_crf_beta": output["group_crf_beta"]} if output.get("matching_mode") == "direct_cosine" else {"group_membership": output["group_membership"][start:stop], "group_crf_alpha": output["group_crf_alpha"], "group_crf_beta": output["group_crf_beta"]})
                paths = _decode_paths(metric_output)
                metadata = [{"volume_id": int(batch["volume_id"][index]), "coordinate_group_id": int(batch["coordinate_group_id"][index])} for index in range(start, stop)]
                pair_rows, triple_rows, errors = evaluate_predictions(paths, batch["q_of_p"][start:stop].detach().cpu().numpy(), batch["valid_pair"][start:stop].detach().cpu().numpy(), metadata)
                for row in pair_rows:
                    key = (int(row["volume_id"]), int(row["coordinate_group_id"]))
                    record = group_metrics.setdefault(key, {"volume_id": key[0], "path_mae_index": [], "f1": [], "iou": []})
                    record["path_mae_index"].append(float(row["path_mae_index"]))  # type: ignore[index]
                    record["f1"].append(float(row[f"f1_tol_{report_f1_tolerance_index:02d}_index"]))  # type: ignore[index]
                    record["iou"].append(float(row["iou"]))  # type: ignore[index]
                if len(errors):
                    error_chunks.append(errors)
            del batch, output, value, logs
    del prefetcher
    if distributed:
        dist.all_reduce(total); dist.all_reduce(crf); dist.all_reduce(count)
    if int(count.item()) != int(expected_groups):
        raise RuntimeError(f"validation covered {int(count.item())} groups, expected {expected_groups}")
    compact_groups = [{"volume_id": int(record["volume_id"]), "path_mae_index": float(np.nanmean(record["path_mae_index"])), "f1": float(np.nanmean(record["f1"])), "iou": float(np.nanmean(record["iou"]))} for record in group_metrics.values()]
    if distributed:
        gathered: list[list[dict[str, Any]] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, compact_groups)
        all_groups = [row for shard in gathered for row in (shard or [])] if dist.get_rank() == 0 else []
    else:
        all_groups = compact_groups
    errors = np.concatenate(error_chunks) if error_chunks else np.empty(0, dtype=np.float32)
    transitive = _distributed_transitive_stats(errors, device, dist.get_world_size() if distributed else 1)
    loss_fn.set_projector_scale(prior_scale)
    if was_training:
        model.train()
    torch.cuda.synchronize(device); torch.cuda.empty_cache()
    metrics = {}
    if not distributed or dist.get_rank() == 0:
        if len(all_groups) != int(expected_groups):
            raise RuntimeError(f"validation metrics covered {len(all_groups)} groups, expected {expected_groups}")
        metrics = {"metrics_protocol": EVALUATION_PROTOCOL, "evaluation_domain": "source positions with GT correspondence to every other curve in the K=8 group", "path_mae_index": _hierarchical_group_mean(all_groups, "path_mae_index"), f"f1_tol_{report_f1_tolerance_index:02d}_index": _hierarchical_group_mean(all_groups, "f1"), "iou": _hierarchical_group_mean(all_groups, "iou"), **transitive}
    return {"loss": float((total / count).item()), "crf_loss": float((crf / count).item()), "groups": int(count.item()), "checkpoint_selection_metric": "validation_crf_loss_ema", "validation_weights": weights, "evaluation_projector_scale": 1.0, "runtime_seconds": time.monotonic() - started, **metrics}


def _replace_link(link: Path, target: Path) -> None:
    temporary = link.with_suffix(link.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        temporary.symlink_to(target.name)
    except OSError:
        shutil.copy2(target, temporary)
    temporary.replace(link)


def _save_checkpoint(out_dir: Path, *, rank: int, world_size: int, local_rank: int, model: torch.nn.Module, loss_fn: GroupwiseEndLoss, optimizer, scheduler, ema: ModelEMA | None, step: int, best_metric: float | None, schedule: dict[str, Any], config: dict[str, Any], is_best: bool) -> None:
    states = collect_rng_states(local_rank, rank, world_size)
    if rank == 0:
        loss_fn.sync_persistent_control_state()
        root = out_dir / "checkpoints"; root.mkdir(parents=True, exist_ok=True)
        path = root / f"step_{step:06d}.pt"; temporary = path.with_suffix(".pt.tmp")
        torch.save({"format_version": "d34_bridge_controlled_end_checkpoint_v1", "method_version": loss_fn.method_version, "objective": _objective(loss_fn), "model": model.state_dict(), "ema_model": None if ema is None else ema.state, "loss_state": loss_fn.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "global_step": step, "best_metric": best_metric, "metric_name": "validation_crf_loss_ema", "world_size": world_size, "rng_states": states, "schedule": schedule, "config": config, "model_scaling": _model_metadata(config, model)}, temporary)
        temporary.replace(path); _replace_link(root / "last.pt", path)
        if is_best:
            _replace_link(root / "best.pt", path)
        print(f"checkpoint saved: {path}", flush=True)
    barrier(world_size > 1)


def _load_resume(path: Path, model, loss_fn, optimizer, scheduler, ema, device, rank: int, world_size: int, *, resume_mode: str, target_config: Mapping[str, Any]) -> tuple[int, float | None, dict[str, Any], dict | None]:
    payload = torch.load(path, map_location=device, weights_only=False)
    method_matches = payload.get("method_version") == loss_fn.method_version
    controlled = resume_mode == "controlled_continuation"
    # New paper continuations retain the production restart state but follow
    # independent validation-driven early stopping.  Keep the older
    # ``controlled_continuation`` mode intact for already-created fixed-budget
    # controls.
    staged = resume_mode == "staged_continuation"
    group_psd_continuation = resume_mode == "group_psd_continuation"
    if not method_matches and (group_psd_continuation or controlled or staged):
        source_config = payload.get("config")
        if not isinstance(source_config, Mapping) or dict(source_config.get("model", {})) != dict(target_config.get("model", {})):
            raise RuntimeError("Group-PSD continuation requires an architecture-identical source checkpoint")
        source_loss, target_loss = dict(source_config.get("loss", {})), dict(target_config.get("loss", {}))
        invariant_loss_keys = ("lambda_crf", "lambda_projector", "lambda_similarity", "lambda_group_score", "allow_no_overlap_output", "align_corridor_radius", "zero_radius_continuous_target_policy", "segment_existence_bias", "target_length_calibration")
        if any(source_loss.get(key) != target_loss.get(key) for key in invariant_loss_keys):
            raise RuntimeError("Group-PSD continuation may change only the optional Group-PSD loss controls")
        if float(source_loss.get("group_psd_weight", 0.0)) != 0.0:
            raise RuntimeError("continuation must fork from CRF-only weights")
        if group_psd_continuation and float(target_loss.get("group_psd_weight", 0.0)) <= 0.0:
            raise RuntimeError("Group-PSD continuation must enable a positive Group-PSD weight")
    elif not method_matches:
        raise RuntimeError("resume checkpoint does not match model identity")
    if int(payload.get("world_size", world_size)) != world_size:
        raise RuntimeError("resume checkpoint does not match model identity or DDP world size")
    model.load_state_dict(payload["model"], strict=True); loss_fn.load_state_dict(payload.get("loss_state", {}), strict=False); loss_fn.sync_runtime_control_state()
    optimizer.load_state_dict(payload["optimizer"]); scheduler.load_state_dict(payload["scheduler"])
    if ema is not None and payload.get("ema_model") is not None:
        ema.load_state_dict(payload["ema_model"])
    states = payload.get("rng_states")
    return int(payload["global_step"]), payload.get("best_metric"), payload.get("schedule", {}), None if states is None else states[rank]


def train(cfg: Any, *, resume_override: str | None = None) -> None:
    rank, world_size, local_rank, distributed = ddp_setup()
    main, device = rank == 0, torch.device(f"cuda:{local_rank}")
    if world_size != int(cfg.train.expected_world_size):
        raise RuntimeError(f"expected {cfg.train.expected_world_size} DDP ranks, got {world_size}")
    if int(cfg.train.get("grad_accum_steps", 1)) != 1:
        raise ValueError("the verified full-epoch sampler currently requires grad_accum_steps=1")
    torch.set_num_threads(int(cfg.train.get("main_threads", 2)))
    amp_enabled, amp_dtype = _configure_cuda(cfg)
    seed(int(cfg.seed), 0)
    data_root, out_dir = Path(cfg.data.root).resolve(), Path(cfg.paths.out_dir).resolve()
    use_bridge_gap_mask = bool(cfg.data.get("use_bridge_gap_mask", False))
    prepared_imp = str(cfg.data.get("format", "")).lower() == "prepared_imp_rgt_v1"
    dataset_cls = D34PreparedImpRGTDataset if prepared_imp else D34BridgeDataset
    if prepared_imp:
        train_set = dataset_cls(data_root, "train", min_overlap_fraction=float(cfg.data.min_overlap_fraction), coordinate_tolerance=float(cfg.data.coordinate_tolerance))
        val_set = dataset_cls(data_root, "validation", min_overlap_fraction=float(cfg.data.min_overlap_fraction), coordinate_tolerance=float(cfg.data.coordinate_tolerance))
    else:
        train_set = dataset_cls(data_root, "train", min_overlap_fraction=float(cfg.data.min_overlap_fraction), coordinate_tolerance=float(cfg.data.coordinate_tolerance), use_bridge_gap_mask=use_bridge_gap_mask)
        val_set = dataset_cls(data_root, "validation", min_overlap_fraction=float(cfg.data.min_overlap_fraction), coordinate_tolerance=float(cfg.data.coordinate_tolerance), use_bridge_gap_mask=use_bridge_gap_mask)
    if len(train_set) != int(cfg.data.expected_train_groups) or len(val_set) != int(cfg.validation.expected_groups):
        raise RuntimeError(f"unexpected data size: train={len(train_set)}, validation={len(val_set)}")
    local_batch, epochs = int(cfg.train.batch_size_per_gpu), int(cfg.train.epochs)
    train_sampler = EpochShuffleDDPSampler(len(train_set), rank=rank, world_size=world_size, local_batch_size=local_batch, epochs=epochs, seed=int(cfg.seed))
    max_steps = len(train_sampler)
    # A hardware-only microbatch rebase may retain the source scheduler's
    # absolute update horizon while changing steps per data epoch.  This is
    # intentionally opt-in: ordinary and historical runs remain untouched.
    configured_max_step = cfg.train.get("continuation_max_global_step", None)
    if configured_max_step not in (None, "", 0):
        max_steps = int(configured_max_step)
        if max_steps < 1:
            raise ValueError("continuation_max_global_step must be positive")
    validation_cadence = str(cfg.validation.get("cadence", "every_epoch"))
    if validation_cadence == "every_epoch":
        validation_every_epochs = 1
    elif validation_cadence == "every_n_epochs":
        validation_every_epochs = int(cfg.validation.get("every_n_epochs", 0))
        if validation_every_epochs < 1:
            raise ValueError("validation.every_n_epochs must be positive")
    else:
        raise ValueError("validation.cadence must be every_epoch or every_n_epochs")
    report_f1_tolerance_index = int(cfg.validation.get("report_f1_tolerance_index", 2))
    if not 0 <= report_f1_tolerance_index <= 20:
        raise ValueError("validation.report_f1_tolerance_index must be in [0, 20]")
    metric_decode_batch_size = int(cfg.validation.get("metric_decode_batch_size_per_gpu", 4))
    if metric_decode_batch_size < 1:
        raise ValueError("validation.metric_decode_batch_size_per_gpu must be positive")
    validation_steps = train_sampler.steps_per_epoch * validation_every_epochs
    schedule = {"name": "full_dataset_epoch_shuffle_no_curriculum", "epochs": epochs, "groups_per_epoch": len(train_set), "global_batch_size": local_batch * world_size, "steps_per_epoch": train_sampler.steps_per_epoch, "total_optimizer_steps": max_steps, "validation_cadence": validation_cadence, "validation_every_epochs": validation_every_epochs, "validation_steps": validation_steps, "validation_metric_decode_batch_size_per_gpu": metric_decode_batch_size, "validation_metric_protocol": EVALUATION_PROTOCOL, "validation_report_metrics": ["path_mae_index", f"f1_tol_{report_f1_tolerance_index:02d}_index", "iou", "transitive_mae_index", "transitive_p95_index"], "supervision": "previous_END_online_pairwise_labels_from_rgt", "continuation_batch_rebase": bool(cfg.train.get("continuation_batch_rebase", False)), "continuation_max_global_step": max_steps if configured_max_step not in (None, "", 0) else None, "sha256": hashlib.sha256(json.dumps({"seed": int(cfg.seed), "epochs": epochs, "groups": len(train_set), "world": world_size, "batch": local_batch}, sort_keys=True).encode()).hexdigest()}
    maybe_warm_page_cache(cfg, data_root, rank=rank, distributed=distributed)
    raw_model = build_end_model(cfg).to(device); loss_fn = build_end_loss(cfg).to(device)
    optimizer = build_optimizer(raw_model, cfg)
    scheduler = build_scheduler(cfg, optimizer, max_steps)
    ema = ModelEMA(raw_model, float(cfg.train.ema_decay)) if bool(cfg.train.ema) else None
    resume = resume_override or cfg.train.get("resume", None)
    continuation_mode = str(cfg.train.get("resume_mode", "exact")).lower()
    allow_group_psd_continuation = continuation_mode == "group_psd_continuation"
    controlled_continuation = continuation_mode == "controlled_continuation"
    staged_continuation = continuation_mode == "staged_continuation"
    step, best_metric, previous_schedule, restored_rng = 0, None, {}, None
    if resume not in (None, "", "none"):
        step, best_metric, previous_schedule, restored_rng = _load_resume(Path(resume), raw_model, loss_fn, optimizer, scheduler, ema, device, rank, world_size, resume_mode=continuation_mode, target_config=_dict(cfg))
        if previous_schedule.get("sha256") != schedule["sha256"] and not (bool(cfg.train.get("e1_preflight", False)) or bool(cfg.train.get("e2_controlled_pretraining", False)) or bool(cfg.train.get("continuation_batch_rebase", False))):
            raise RuntimeError("resume schedule differs from this configuration")
        expected_source_step = cfg.train.get("continuation_expected_source_global_step", None)
        if expected_source_step not in (None, "", 0) and int(step) != int(expected_source_step):
            raise RuntimeError(f"continuation batch-rebase expected source global step {expected_source_step}, received {step}")
    if main:
        if resume in (None, "", "none"):
            _atomic_json(out_dir / "initialization_manifest.json", {"seed":int(cfg.seed),"initial_model_state_sha256":_model_state_sha256(raw_model),"cross_sequence_communication":bool(cfg.model.get("cross_sequence_communication",True)),"parameter_count":sum(p.numel() for p in raw_model.parameters()),"trainable_parameter_count":sum(p.numel() for p in raw_model.parameters() if p.requires_grad)})
        out_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(out_dir / "config_resolved.json", _dict(cfg)); _atomic_json(out_dir / "training_schedule.json", schedule)
        _atomic_json(out_dir / "data_protocol.json", {"root": str(data_root), "train_groups": len(train_set), "validation_groups": len(val_set), "input": "prepared imp.f16.npy with dataloader per-curve z-score + P99 clipping/scaling" if prepared_imp else "preprocessed_p99_per_curve_f32_v1/z.f32.npy", "rgt_target": "online pairwise labels from RGT; nearest-integer exact lattice target for r=0" if prepared_imp else "previous END online pairwise labels from RGT", "use_bridge_gap_mask": use_bridge_gap_mask})
        if (allow_group_psd_continuation or controlled_continuation or staged_continuation) and resume not in (None, "", "none") and best_metric is not None:
            carried_best = out_dir / "checkpoints" / "best.pt"
            carried_best.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(resume), carried_best)
            _atomic_json(out_dir / "continuation_protocol.json", {"mode": continuation_mode, "source_checkpoint": str(Path(resume).resolve()), "source_global_step": step, "source_best_validation_crf_loss_ema": best_metric, "state_restore": ["raw_model", "optimizer", "scheduler", "EMA", "per-rank RNG"], "initial_best_checkpoint": str(carried_best)})
        if controlled_continuation:
            fork=Path(resume).resolve()
            digest=_sha256_file(fork)
            controlled_name=str(cfg.experiment.get("controlled_experiment","E1"))
            _atomic_json(out_dir / f"{controlled_name.lower()}_control_manifest.json", {"experiment":f"{controlled_name} controlled continuation","condition":str(cfg.experiment.get("controlled_condition","pal_grc" if float(cfg.loss.get("group_psd_weight",0.0))>0 else "pal")),"fork_checkpoint":str(fork),"fork_checkpoint_sha256":digest,"resume_mode":continuation_mode,"state_restore":["raw_model","optimizer","scheduler","EMA","per-rank RNG","loss_control_state"],"start_global_update":step,"budget_end_global_update":max_steps,"post_fork_optimizer_updates":max_steps-step,"microbatch_per_rank":local_batch,"world_size":world_size,"effective_global_batch":local_batch*world_size,"validation_every_updates":validation_steps,"validation_candidates_including_fork":1+(max_steps-step)//validation_steps,"selection_metric":"EMA validation CRF loss","early_stopping":"disabled; fixed optimizer-update budget","data_order":"deterministic EpochShuffleDDPSampler(seed=experiment seed, epoch) from same start step","lr_control":"same resumed scheduler state and identical max_steps/LambdaLR contract"})
        if staged_continuation:
            fork = Path(resume).resolve()
            _atomic_json(out_dir / "staged_continuation_manifest.json", {
                "experiment": str(cfg.experiment.get("controlled_experiment", "staged")),
                "condition": str(cfg.experiment.get("controlled_condition", "pal_grc" if float(cfg.loss.get("group_psd_weight", 0.0)) > 0 else "pal")),
                "fork_checkpoint": str(fork), "fork_checkpoint_sha256": _sha256_file(fork),
                "resume_mode": continuation_mode,
                "state_restore": ["raw_model", "optimizer", "scheduler", "EMA", "per-rank RNG", "loss_control_state"],
                "start_global_update": step, "maximum_global_update": max_steps,
                "maximum_post_fork_optimizer_updates": max_steps - step,
                "microbatch_per_rank": local_batch, "world_size": world_size,
                "effective_global_batch": local_batch * world_size,
                "validation_every_updates": validation_steps,
                "selection_metric": "EMA validation CRF loss",
                "early_stopping": {"enabled": bool(cfg.train.get("early_stop_patience", 0)), "patience": int(cfg.train.get("early_stop_patience", 0)), "min_delta": float(cfg.train.get("early_stop_min_delta", 0.0))},
                "data_order": "EpochShuffleDDPSampler(seed=experiment seed, epoch); restored per-rank RNG",
                "lr_control": "resumed scheduler state with production absolute max-step horizon",
            })
    model: torch.nn.Module = DDP(raw_model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False, gradient_as_bucket_view=True, static_graph=bool(cfg.train.static_graph)) if distributed else raw_model
    if restored_rng is not None:
        restore_rng_state(restored_rng, local_rank)
    else:
        seed(int(cfg.seed), rank)
    continuation_batch_rebase = bool(cfg.train.get("continuation_batch_rebase", False))
    logical_epoch_offset = 0
    if continuation_batch_rebase:
        if resume in (None, "", "none"):
            raise RuntimeError("continuation_batch_rebase requires a resumed epoch-10 source checkpoint")
        logical_epoch_offset = int(cfg.train.get("continuation_source_completed_epochs", 0))
        if logical_epoch_offset < 1 or max_steps <= step:
            raise RuntimeError("invalid continuation batch-rebase epoch offset or update horizon")
        # Do not reinterpret source step=350 under the smaller batch.  Begin
        # data consumption at the first B=40 epoch after the source's ten
        # completed epochs, while retaining global update/scheduler step 350.
        data_start_step = logical_epoch_offset * train_sampler.steps_per_epoch
        remaining_updates = max_steps - step
        sampler_epochs = math.ceil((data_start_step + remaining_updates) / train_sampler.steps_per_epoch)
        train_sampler = EpochShuffleDDPSampler(len(train_set), rank=rank, world_size=world_size, local_batch_size=local_batch, epochs=sampler_epochs, seed=int(cfg.seed), start_step=data_start_step)
        display_epochs = logical_epoch_offset + math.ceil(remaining_updates / train_sampler.steps_per_epoch)
        schedule.update({
            "continuation_batch_rebase": True,
            "source_completed_logical_epochs": logical_epoch_offset,
            "source_global_step": step,
            "data_sampler_start_step": data_start_step,
            "data_sampler_epochs": sampler_epochs,
            "post_fork_optimizer_updates": remaining_updates,
            "logical_final_epoch": display_epochs,
        })
    else:
        train_sampler = EpochShuffleDDPSampler(len(train_set), rank=rank, world_size=world_size, local_batch_size=local_batch, epochs=epochs, seed=int(cfg.seed), start_step=step)
        display_epochs = epochs
    loader = build_loader(train_set, train_sampler, rank=rank, workers=int(cfg.train.num_workers_per_rank), prefetch_factor=int(cfg.train.prefetch_factor), worker_threads=int(cfg.train.worker_threads))
    val_sampler = FixedIndexBatchSampler(len(val_set), rank=rank, world_size=world_size, batch_size=int(cfg.validation.batch_size_per_gpu))
    val_loader = build_loader(val_set, val_sampler, rank=rank, workers=int(cfg.validation.num_workers_per_rank), prefetch_factor=int(cfg.validation.prefetch_factor), worker_threads=int(cfg.validation.worker_threads), validation=True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    prefetcher = CUDAPrefetcher(loader, device)
    log_every = int(cfg.train.log_every)
    log_cadence = str(cfg.train.get("train_log_cadence", "step")).strip().lower()
    if log_cadence not in {"step", "every_epoch"}:
        raise ValueError("train_log_cadence must be 'step' or 'every_epoch'")
    timing_every = int(cfg.train.get("timing_log_every", log_every))
    if main:
        slots = 0 if prepared_imp else cfg.model.group_alignment.num_slots
        print(f"[D34 Start] data={data_root} format={'prepared_imp_rgt_v1' if prepared_imp else 'legacy'} bridge_gap_mask={use_bridge_gap_mask} scale={cfg.experiment.model_scale} slots={slots} train_groups={len(train_set)} validation_groups={len(val_set)} GPUs={world_size} B/GPU={local_batch} epochs={display_epochs} steps={max_steps} validation=every_{validation_every_epochs}_epochs({validation_steps} steps) metric_decode_B/GPU={metric_decode_batch_size} metrics=PathMAE,F1@{report_f1_tolerance_index},IoU,TransitiveMAE,TransitiveP95 | {_objective(loss_fn)}", flush=True)
    early_stop_patience = int(cfg.train.get("early_stop_patience", 0))
    early_stop_delta = float(cfg.train.get("early_stop_min_delta", 0.0))
    material_best, stale_validations = (best_metric if (allow_group_psd_continuation or controlled_continuation or staged_continuation) else None), 0
    model.train(); window_start = run_start = time.monotonic(); window_samples = 0
    epoch_train_start = time.monotonic()
    epoch_start_step = step + 1
    epoch_log_sums: dict[str, torch.Tensor] = {}
    epoch_local_groups = 0
    stop_after_epochs = cfg.train.get("stop_after_epochs", None)
    end_step = max_steps if stop_after_epochs in (None, "", 0) else min(max_steps, int(stop_after_epochs) * train_sampler.steps_per_epoch)
    if end_step < step:
        raise RuntimeError("stop_after_epochs precedes the resumed global step")
    for current_step in range(step + 1, end_step + 1):
        if continuation_batch_rebase:
            continuation_updates = current_step - step
            current_epoch = logical_epoch_offset + (continuation_updates - 1) // train_sampler.steps_per_epoch + 1
            epoch_end = continuation_updates % train_sampler.steps_per_epoch == 0 or current_step == end_step
        else:
            current_epoch = (current_step - 1) // train_sampler.steps_per_epoch + 1
            epoch_end = current_step % train_sampler.steps_per_epoch == 0 or current_step == end_step
        if hasattr(loss_fn, "set_training_epoch"):
            loss_fn.set_training_epoch(current_epoch)
        optimizer.zero_grad(set_to_none=True); loss_fn.set_projector_scale(_projector_scale(cfg, current_step)); loss_fn.set_global_group_offset(current_step - 1)
        profile_timing = log_cadence == "step" and timing_every > 0 and (current_step == 1 or current_step % timing_every == 0)
        if profile_timing:
            torch.cuda.reset_peak_memory_stats(device)
        data_wait_start = time.monotonic()
        batch = prefetcher.next()
        data_wait_seconds = time.monotonic() - data_wait_start
        if profile_timing:
            forward_start, forward_end, loss_end, update_end = (torch.cuda.Event(enable_timing=True) for _ in range(4))
            forward_start.record()
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
            output = model(batch["z"], batch["mask"])
        if profile_timing:
            forward_end.record()
        loss, logs = loss_fn(_loss_output(output, optimized=loss_fn.performance_optimization_enabled), batch, compute_diagnostics=(current_step == 1 or current_step % log_every == 0))
        if profile_timing:
            loss_end.record()
        scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(raw_model.parameters(), float(cfg.train.grad_clip))
        scaler.step(optimizer); scaler.update(); scheduler.step()
        if ema is not None:
            ema.update(raw_model, current_step)
        if bool(getattr(loss_fn, "group_psd_is_active", False)):
            group_count = int(batch["z"].shape[0]) if getattr(loss_fn, "group_psd_batch_groups", None) is None else min(int(batch["z"].shape[0]), int(loss_fn.group_psd_batch_groups))
            loss_fn.advance_group_psd_group_offset(group_count)
        local_groups = int(batch["z"].shape[0])
        if log_cadence == "every_epoch":
            epoch_local_groups += local_groups
            for key, value in logs.items():
                if torch.is_tensor(value):
                    weighted = value.detach().float() * local_groups
                    epoch_log_sums[key] = weighted if key not in epoch_log_sums else epoch_log_sums[key] + weighted
        timing: dict[str, float] = {}
        if profile_timing:
            update_end.record(); update_end.synchronize()
            timing = {
                "data_wait_seconds": data_wait_seconds,
                "model_forward_ms": forward_start.elapsed_time(forward_end),
                "loss_forward_ms": forward_end.elapsed_time(loss_end),
                "backward_optimizer_ema_ms": loss_end.elapsed_time(update_end),
                "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0),
            }
        window_samples += local_groups * world_size
        if log_cadence == "every_epoch" and epoch_end:
            reduced = _reduce_weighted_epoch_logs(epoch_log_sums, epoch_local_groups, distributed)
            if main:
                epoch_seconds = max(1e-9, time.monotonic() - epoch_train_start)
                global_groups = epoch_local_groups * world_size
                payload = {
                    "type": "train_epoch", "epoch": current_epoch,
                    "step": current_step, "epoch_start_step": epoch_start_step,
                    "epoch_end_step": current_step, "steps_this_epoch": train_sampler.steps_per_epoch,
                    "max_steps": max_steps, "global_groups": global_groups,
                    "epoch_seconds": epoch_seconds, "groups_per_second": global_groups / epoch_seconds,
                    "lr": optimizer.param_groups[0]["lr"],
                    "num_slots": 0 if prepared_imp else int(cfg.model.group_alignment.num_slots),
                    **reduced,
                }
                _append_jsonl(out_dir / "train_log.jsonl", payload)
                print(
                    f"[D34 Train Epoch] {current_epoch}/{epochs} steps={payload['epoch_start_step']}-{current_step} "
                    f"L_total={payload['l_total']:.4f} L_crf={payload['l_crf']:.4f} "
                    f"L_group={payload['l_group_psd']:.3e} lambda_group_L_group={payload['l_group_psd_weighted']:.4f} | "
                    f"group_directional_energy={payload['group_directional_energy']:.3e} "
                    f"group_negative_spectral_energy={payload['group_negative_spectral_energy']:.3e} | "
                    f"{payload['groups_per_second']:.1f} groups/s epoch={epoch_seconds:.1f}s lr={payload['lr']:.3e}",
                    flush=True,
                )
            epoch_train_start = time.monotonic()
            epoch_start_step = current_step + 1
            epoch_log_sums = {}
            epoch_local_groups = 0
        elif log_cadence == "step" and (current_step == 1 or current_step % log_every == 0):
            reduced = _reduce_logs(logs, distributed)
            if main:
                elapsed, total_elapsed = max(1e-9, time.monotonic() - window_start), time.monotonic() - run_start
                rate, eta = window_samples / elapsed, (max_steps - current_step) / max(current_step / max(total_elapsed, 1e-9), 1e-9)
                payload = {"type": "train", "step": current_step, "max_steps": max_steps, "epoch": current_epoch, "percent": 100.0 * current_step / max_steps, "samples_per_second": rate, "elapsed_seconds": total_elapsed, "eta_seconds": eta, "lr": optimizer.param_groups[0]["lr"], "num_slots": 0 if prepared_imp else int(cfg.model.group_alignment.num_slots), **timing, **reduced}
                _append_jsonl(out_dir / "train_log.jsonl", payload)
                print(f"[D34 Train] {current_step}/{max_steps} ({payload['percent']:.1f}%) L_total={payload['l_total']:.4f} L_crf={payload['l_crf']:.4f} L_group={payload['l_group_psd']:.4f} lambda_group_L_group={payload['l_group_psd_weighted']:.4f} | {rate:.1f} groups/s ETA {eta:.0f}s", flush=True)
            window_start, window_samples = time.monotonic(), 0
        due = ((current_step - step) % validation_steps == 0 if continuation_batch_rebase else current_step % validation_steps == 0) or current_step == end_step
        if due:
            validation = evaluate_validation_loss(raw_model, loss_fn, val_loader, device, amp_enabled=amp_enabled, amp_dtype=amp_dtype, distributed=distributed, ema=ema, expected_groups=len(val_set), report_f1_tolerance_index=report_f1_tolerance_index, metric_decode_batch_size=metric_decode_batch_size)
            is_best = best_metric is None or validation["crf_loss"] < best_metric
            if is_best:
                best_metric = validation["crf_loss"]
            material = material_best is None or validation["crf_loss"] < material_best - early_stop_delta
            if material:
                material_best, stale_validations = validation["crf_loss"], 0
            else:
                stale_validations += 1
            if main:
                epoch = current_epoch
                _append_jsonl(out_dir / "validation_log.jsonl", {"type": "validation", "epoch": epoch, "step": current_step, "is_best": is_best, "material_improvement": material, "stale_validations": stale_validations, "early_stop_patience": early_stop_patience, "early_stop_min_delta": early_stop_delta, **validation})
                print(f"[D34 Validation] epoch={epoch}/{epochs} step={current_step} EMA crf={validation['crf_loss']:.6f} total={validation['loss']:.6f} | PathMAE={validation['path_mae_index']:.4f} F1@{report_f1_tolerance_index}={validation[f'f1_tol_{report_f1_tolerance_index:02d}_index']:.4f} IoU={validation['iou']:.4f} TransitiveMAE={validation['transitive_mae_index']:.4f} TransitiveP95={validation['transitive_p95_index']:.4f} | best_crf={best_metric:.6f}", flush=True)
            checkpoint_started = time.monotonic()
            _save_checkpoint(out_dir, rank=rank, world_size=world_size, local_rank=local_rank, model=raw_model, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, ema=ema, step=current_step, best_metric=best_metric, schedule=schedule, config=_dict(cfg), is_best=is_best)
            if main:
                print(f"[D34 Validation Timing] validation={validation['runtime_seconds']:.1f}s checkpoint={time.monotonic() - checkpoint_started:.1f}s", flush=True)
            # The next train-rate window must not include validation or disk
            # I/O. The old placement made the first post-validation rate look
            # like a training regression.
            window_start, window_samples = time.monotonic(), 0
            if early_stop_patience and stale_validations >= early_stop_patience:
                if main:
                    print(f"[D34 EarlyStop] no validation CRF improvement >= {early_stop_delta:g} for {stale_validations} validations", flush=True)
                break
        del batch, output, loss, logs
    if main:
        _atomic_json(out_dir / "run_metadata.json", {"model_scale": cfg.experiment.model_scale, "runtime_seconds": time.monotonic() - run_start, "best_validation_crf_loss_ema": best_metric, "world_size": world_size, "training_schedule": schedule["name"], "curriculum_learning": False, "initial_checkpoint": None if resume in (None, "", "none") else str(Path(resume).resolve()), "resume_mode": continuation_mode, "model_scaling": _model_metadata(cfg, raw_model)})
    barrier(distributed)
