"""The frozen END objective: original corridor CRF plus Projector.

The final END method uses one shared membership ``H`` for both branches:
``H_i H_j^T`` supplies pairwise correspondence to the original score-
conditioned partial-overlap corridor CRF, while stacked ``H`` supplies the
whole-group relation for the idempotent Projector.  No GeoCRF, PreciseCRF,
VCGR, RankFree, Structural Sharing, or auxiliary task loss is accepted.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Mapping

import torch
import torch.nn as nn

from ssia_alignment.loss_parts.group_spectral_regularizer import group_spectral_regularizer

from .diagnostics import membership_diagnostics
from .partial_overlap_crf import partial_overlap_crf_loss


def _value(cfg: Any, key: str, default: Any) -> Any:
    return cfg.get(key, default) if isinstance(cfg, Mapping) else getattr(cfg, key, default)


def _section(cfg: Any) -> Any:
    return cfg["loss"] if isinstance(cfg, Mapping) and "loss" in cfg else getattr(cfg, "loss", cfg)


def _performance_optimization_enabled(cfg: Any) -> bool:
    train = cfg.get("train", {}) if isinstance(cfg, Mapping) else getattr(cfg, "train", {})
    option = _value(train, "performance_optimization", False)
    return bool(_value(option, "enabled", option))


class GroupwiseEndLoss(nn.Module):
    """``L_END = L_corridor_CRF + scale_P * lambda_P * L_Projector``."""

    method_version = "end_corridor_crf_projector_v1"
    supports_component_gradient_diagnostics = False

    def __init__(self, cfg: Mapping[str, Any] | Any) -> None:
        super().__init__()
        loss = _section(cfg)
        self.lambda_crf = float(_value(loss, "lambda_crf", 1.0))
        self.lambda_projector = float(_value(loss, "lambda_projector", 0.1))
        if self.lambda_crf != 1.0:
            raise ValueError("END fixes loss.lambda_crf=1.0")
        forbidden = {
            "geo_margin_gamma": _value(loss, "geo_margin_gamma", 0.0),
            "lambda_vcgr": _value(loss, "lambda_vcgr", 0.0),
            "lambda_rank": _value(loss, "lambda_rank", 0.0),
            "lambda_rankfree": _value(loss, "lambda_rankfree", 0.0),
            "lambda_share": _value(loss, "lambda_share", 0.0),
        }
        active = {name: value for name, value in forbidden.items() if value not in (None, 0, 0.0, False)}
        if bool(_value(loss, "geo_margin_enabled", False)):
            active["geo_margin_enabled"] = True
        if bool(_value(loss, "precise_target", False)) or str(_value(loss, "crf_target_type", "original_partial_overlap_corridor")).lower() not in {"original_partial_overlap_corridor", "corridor", "legacy_corridor"}:
            active["precise_target"] = True
        if active:
            raise ValueError(f"END only permits original CorridorCRF + Projector; forbidden settings: {active}")
        self.global_stride = max(1, int(_value(loss, "global_stride", 8)))
        self.global_batch_groups = max(1, int(_value(loss, "global_batch_groups", 3)))
        self.group_pair_eps = float(_value(loss, "group_pair_eps", 1.0e-6))
        self.corridor_radius = float(_value(loss, "align_corridor_radius", 2.0))
        # The END ablation protocol intentionally varies only this supervised
        # corridor radius (r=1,2,3) and the Projector weight.  All use the
        # same original, score-conditioned corridor CRF implementation.
        if self.corridor_radius not in {1.0, 2.0, 3.0}:
            raise ValueError("END ablations permit loss.align_corridor_radius in {1.0, 2.0, 3.0}")
        segment = _value(loss, "align_segment_bias", None)
        self.align_segment_bias = None if segment in (None, "null", "none") else float(segment)
        model = cfg.get("model", {}) if isinstance(cfg, Mapping) else getattr(cfg, "model", {})
        self.assume_full_length = bool(_value(model, "assume_full_length", True))
        self.performance_optimization_enabled = _performance_optimization_enabled(cfg)
        self.register_buffer("projector_scale_value", torch.zeros(()), persistent=True)
        self.register_buffer("global_group_offset", torch.zeros((), dtype=torch.long), persistent=True)
        self._projector_scale_py = 0.0
        self._global_group_offset_py = 0
        self._pair_cache: dict[tuple[int, str, int | None], tuple[torch.Tensor, torch.Tensor]] = {}
        self._unordered_pair_cache: dict[tuple[int, str, int | None], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        experiment = cfg.get("experiment", {}) if isinstance(cfg, Mapping) else getattr(cfg, "experiment", {})
        requested = str(_value(experiment, "method_version", "")).strip()
        if requested:
            self.method_version = requested

    @property
    def projector_scale(self) -> float:
        return self._projector_scale_py if self.performance_optimization_enabled else float(self.projector_scale_value.item())

    def set_projector_scale(self, scale: float) -> None:
        self._projector_scale_py = float(scale)
        if not self.performance_optimization_enabled:
            self.projector_scale_value.fill_(float(scale))

    def set_global_group_offset(self, offset: int) -> None:
        self._global_group_offset_py = int(offset)
        if not self.performance_optimization_enabled:
            self.global_group_offset.fill_(int(offset))

    def sync_persistent_control_state(self) -> None:
        if self.performance_optimization_enabled:
            self.projector_scale_value.fill_(self._projector_scale_py)
            self.global_group_offset.fill_(self._global_group_offset_py)

    def sync_runtime_control_state(self) -> None:
        self._projector_scale_py = float(self.projector_scale_value.item())
        self._global_group_offset_py = int(self.global_group_offset.item())

    def _directed_pairs(self, curves: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key = (curves, device.type, device.index)
        if key not in self._pair_cache:
            ids = torch.arange(curves, device=device)
            source, target = ids[:, None].expand(curves, curves), ids[None, :].expand(curves, curves)
            self._pair_cache[key] = source[source != target], target[source != target]
        return self._pair_cache[key]

    def _unordered_pairs(self, curves: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (curves, device.type, device.index)
        if key not in self._unordered_pair_cache:
            source, target = self._directed_pairs(curves, device)
            forward = source < target
            left, right = source[forward], target[forward]
            canonical_left, canonical_right = torch.minimum(source, target), torch.maximum(source, target)
            route = ((canonical_left[:, None] == left[None, :]) & (canonical_right[:, None] == right[None, :])).to(torch.long).argmax(dim=1)
            self._unordered_pair_cache[key] = left, right, route, forward
        return self._unordered_pair_cache[key]

    def _correspondence(self, membership: torch.Tensor, profile_timer=None) -> torch.Tensor:
        batch, curves, length, slots = membership.shape
        source, target = self._directed_pairs(curves, membership.device)
        with (profile_timer("correspondence_build") if profile_timer else nullcontext()):
            if self.performance_optimization_enabled:
                pair_major = membership.permute(1, 0, 2, 3)
                left_id, right_id, route, forward = self._unordered_pairs(curves, membership.device)
                left = pair_major.index_select(0, left_id).reshape(left_id.numel() * batch, length, slots)
                right = pair_major.index_select(0, right_id).reshape(right_id.numel() * batch, length, slots)
                unordered = torch.bmm(left, right.transpose(1, 2)).reshape(left_id.numel(), batch, length, length)
                directed = unordered.index_select(0, route)
                directed = torch.where(forward[:, None, None, None], directed, directed.transpose(-1, -2))
                return directed.permute(1, 0, 2, 3)
            left = membership.index_select(1, source).reshape(batch * source.numel(), length, slots)
            right = membership.index_select(1, target).reshape(batch * target.numel(), length, slots)
            return torch.bmm(left, right.transpose(1, 2)).reshape(batch, source.numel(), length, length)

    def _crf(self, correspondence: torch.Tensor, out: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor], profile_timer=None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        source, target = self._directed_pairs(batch["mask"].shape[1], batch["mask"].device)
        batch_size, pairs, length, _ = correspondence.shape
        with (profile_timer("crf_forward") if profile_timer else nullcontext()):
            if self.performance_optimization_enabled:
                score = out["group_crf_alpha"] * correspondence.permute(1, 0, 2, 3).clamp_min(self.group_pair_eps).log() + out["group_crf_beta"]
                packed = score.reshape(pairs * batch_size, length, length)
                q_of_p = batch["q_of_p"].permute(1, 2, 0, 3)[source, target].reshape(pairs * batch_size, length)
                valid = batch["valid_pair"].permute(1, 2, 0, 3)[source, target].reshape(pairs * batch_size, length).bool()
            else:
                score = out["group_crf_alpha"] * correspondence.clamp_min(self.group_pair_eps).log() + out["group_crf_beta"]
                packed = score.permute(1, 0, 2, 3).reshape(pairs * batch_size, length, length)
                q_of_p = batch["q_of_p"][:, source, target].permute(1, 0, 2).reshape(pairs * batch_size, length)
                valid = batch["valid_pair"][:, source, target].permute(1, 0, 2).reshape(pairs * batch_size, length).bool()
            return partial_overlap_crf_loss(packed, q_of_p, valid, match_bias=0.0, segment_bias=self.align_segment_bias, corridor_radius=self.corridor_radius)

    def _selected_groups(self, membership: torch.Tensor) -> list[int]:
        count = min(membership.shape[0], self.global_batch_groups)
        offset = self._global_group_offset_py if self.performance_optimization_enabled else int(self.global_group_offset.item())
        return [(offset + item) % membership.shape[0] for item in range(count)]

    def forward(self, out: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor], *, compute_diagnostics: bool = True, profile_timer=None, **_ignored: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        membership, mask = out["group_membership"], batch["mask"]
        if out.get("null_probability") is not None or (out.get("group_assignments") is not None and out["group_assignments"].shape[-1] != membership.shape[-1]):
            raise RuntimeError("GroupwiseEndLoss requires the frozen no-NULL hierarchical contract")
        correspondence = self._correspondence(membership, profile_timer)
        crf, crf_logs = self._crf(correspondence, out, batch, profile_timer)
        if self.projector_scale and self.lambda_projector:
            projector, _rankfree, projector_logs = group_spectral_regularizer(membership, mask, stride=self.global_stride, batch_indices=self._selected_groups(membership), assume_full_length=self.assume_full_length, compute_rankfree=False, profile_timer=profile_timer)
        else:
            projector, projector_logs = membership.sum() * 0.0, {"global_groups_used": membership.new_zeros(())}
        total = self.lambda_crf * crf + self.projector_scale * self.lambda_projector * projector
        logs: dict[str, torch.Tensor] = {
            "l_crf": crf.detach(), "l_crf_weighted": (self.lambda_crf * crf).detach(),
            "l_projector": projector.detach(), "l_projector_weighted": (self.projector_scale * self.lambda_projector * projector).detach(),
            "projector_scale": total.new_tensor(self.projector_scale), **crf_logs, **projector_logs,
        }
        if compute_diagnostics:
            logs.update(membership_diagnostics(membership, mask, correspondence, dict(out), batch["valid_pair"]))
        logs["loss"] = total.detach()
        return total, logs


def build_end_loss(cfg: Mapping[str, Any] | Any) -> nn.Module:
    """Select the frozen legacy loss or the isolated slot-free main loss."""
    model = cfg.get("model", {}) if isinstance(cfg, Mapping) else getattr(cfg, "model", {})
    mode = _value(model, "groupwise_mode", "")
    if str(mode).lower() == "interaction_direct_cosine":
        from .direct_cosine_loss import DirectCosineAlwaysOverlapCRFLoss
        return DirectCosineAlwaysOverlapCRFLoss(cfg)
    return GroupwiseEndLoss(cfg)
