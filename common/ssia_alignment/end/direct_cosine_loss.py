"""CRF-only objective for D34's slot-free direct-cosine main route."""
from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn

from .partial_overlap_crf import always_overlap_crf_loss


def _value(cfg: Any, key: str, default: Any) -> Any:
    return cfg.get(key, default) if isinstance(cfg, Mapping) else getattr(cfg, key, default)


class DirectCosineAlwaysOverlapCRFLoss(nn.Module):
    """Exact r=0 partial-path CRF over all 56 directed D34 curve pairs."""

    method_version = "ssia_pal"
    lambda_projector = 0.0
    performance_optimization_enabled = False
    projector_scale = 0.0

    def __init__(self, cfg: Mapping[str, Any] | Any) -> None:
        super().__init__()
        loss = cfg.get("loss", {}) if isinstance(cfg, Mapping) else cfg.loss
        model = cfg.get("model", {}) if isinstance(cfg, Mapping) else cfg.model
        if str(_value(model, "groupwise_mode", "")).lower() != "interaction_direct_cosine":
            raise ValueError("direct CRF loss requires the slot-free direct-cosine model")
        if bool(_value(loss, "allow_no_overlap_output", True)):
            raise ValueError("D34 main route requires allow_no_overlap_output=false")
        if float(_value(loss, "align_corridor_radius", -1.0)) != 0.0:
            raise ValueError("D34 main route requires exact r=0 supervision")
        if str(_value(loss, "zero_radius_continuous_target_policy", "")).lower() != "nearest_integer":
            raise ValueError("r=0 requires nearest_integer GT discretization")
        if float(_value(loss, "lambda_projector", 0.0)) != 0.0 or float(_value(loss, "lambda_similarity", 0.0)) != 0.0 or float(_value(loss, "lambda_group_score", 0.0)) != 0.0:
            raise ValueError("the main route is CRF-only")
        self.group_psd_weight = float(_value(loss, "group_psd_weight", 0.0))
        self.group_psd_enabled = bool(_value(loss, "group_psd_enabled", False))
        self.group_psd_start_epoch = int(_value(loss, "group_psd_start_epoch", 1))
        # ``group_psd_num_bins`` is a backward-compatible alias only.  The
        # directed PSD branch samples this many *real curve positions*; it
        # does not construct bins, pools, or group nodes.
        self.group_psd_num_positions = int(
            _value(loss, "group_psd_num_positions", _value(loss, "group_psd_num_bins", 32))
        )
        self.group_psd_num_bins = self.group_psd_num_positions
        self.group_psd_objective_mode = str(_value(loss, "group_psd_objective_mode", "full")).lower()
        self.group_psd_cardinality = int(_value(loss, "group_psd_cardinality", 8))
        self.group_psd_subset_namespace = str(_value(loss, "group_psd_subset_namespace", "ssia_grc"))
        batch_groups = _value(loss, "group_psd_batch_groups", None)
        self.group_psd_batch_groups = None if batch_groups in (None, "none") else int(batch_groups)
        if self.group_psd_weight < 0.0 or self.group_psd_start_epoch < 1 or self.group_psd_num_positions < 1 or not 2 <= self.group_psd_cardinality <= 8 or self.group_psd_objective_mode not in {"full", "direction", "spectral"} or (self.group_psd_batch_groups is not None and self.group_psd_batch_groups < 1):
            raise ValueError("invalid Group-PSD configuration")
        if self.group_psd_weight > 0.0 and not self.group_psd_enabled:
            raise ValueError("a positive Group-PSD weight requires group_psd_enabled=true")
        # The established PSD continuation computes its expensive posterior
        # branch on a deterministic rotating subset, while CRF still covers
        # every group.  Keep this state checkpointable without synchronizing a
        # GPU scalar in the ordinary update path.
        self.register_buffer("group_psd_offset", torch.zeros((), dtype=torch.long))
        self._group_psd_offset_py = 0
        self._group_psd_sampling_draw = 0
        self._group_psd_active = self.group_psd_enabled and self.group_psd_start_epoch <= 1
        self._pair_cache: dict[tuple[int, str, int | None], tuple[torch.Tensor, ...]] = {}
        experiment = cfg.get("experiment", {}) if isinstance(cfg, Mapping) else cfg.experiment
        self._experiment_seed = int(_value(cfg, "seed", 0))
        requested = str(_value(experiment, "method_version", "")).strip()
        if requested:
            self.method_version = requested

    def set_projector_scale(self, _scale: float) -> None:
        return None

    def set_global_group_offset(self, _offset: int) -> None:
        return None

    @property
    def effective_group_psd_weight(self) -> float:
        return self.group_psd_weight if self._group_psd_active else 0.0

    @property
    def group_psd_is_active(self) -> bool:
        return self._group_psd_active

    def set_training_epoch(self, epoch: int) -> None:
        """Enable the existing Group-PSD term only at its configured epoch."""
        if epoch < 1:
            raise ValueError("training epoch must be positive")
        self._group_psd_active = self.group_psd_enabled and epoch >= self.group_psd_start_epoch

    def advance_group_psd_group_offset(self, count: int) -> None:
        self._group_psd_offset_py += int(count)
        if self.group_psd_offset.device.type == "cpu":
            self.group_psd_offset.add_(int(count))

    def sync_persistent_control_state(self) -> None:
        self.group_psd_offset.fill_(self._group_psd_offset_py)

    def sync_runtime_control_state(self) -> None:
        self._group_psd_offset_py = int(self.group_psd_offset.detach().cpu())

    def _pairs(self, curves: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        """Directed pair layout plus the minimal 28-GEMM routing plan.

        Direct cosine is symmetric before direction-specific path decoding:
        S(j,i) is S(i,j).T.  Build only unordered blocks, then route their
        views to the 56 directed CRF lattices.  This is the slot-free version
        of the established END correspondence optimization.
        """
        key = (curves, device.type, device.index)
        if key not in self._pair_cache:
            ids = torch.arange(curves, device=device)
            source, target = ids[:, None].expand(curves, curves), ids[None, :].expand(curves, curves)
            source, target = source[source != target], target[source != target]
            forward = source < target
            left_id, right_id = source[forward], target[forward]
            canonical_left, canonical_right = torch.minimum(source, target), torch.maximum(source, target)
            route = ((canonical_left[:, None] == left_id[None, :]) & (canonical_right[:, None] == right_id[None, :])).to(torch.long).argmax(dim=1)
            self._pair_cache[key] = source, target, left_id, right_id, route, forward
        return self._pair_cache[key]

    @staticmethod
    def _nearest_integer(q: torch.Tensor, valid: torch.Tensor, length: int) -> torch.Tensor:
        rounded = torch.floor(q.float() + 0.5).clamp_(0, length - 1)
        return torch.where(valid.bool(), rounded, q.float())

    def _directed_similarity(self, out: Mapping[str, torch.Tensor | str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = out["matching_features"]
        if not torch.is_tensor(features) or features.ndim != 4 or features.shape[1:] != (8, 512, features.shape[-1]):
            raise ValueError("matching_features must be [B,8,512,D]")
        if out.get("matching_mode") != "direct_cosine":
            raise ValueError("slot-free loss only accepts raw direct cosine matching")
        batch_size, curves, length, dimension = features.shape
        source, target, left_id, right_id, route, forward = self._pairs(curves, features.device)
        left = features.index_select(1, left_id).reshape(batch_size * len(left_id), length, dimension).float()
        right = features.index_select(1, right_id).reshape(batch_size * len(right_id), length, dimension).float()
        unordered = torch.bmm(left, right.transpose(1, 2)).reshape(batch_size, len(left_id), length, length)
        similarity = unordered.index_select(1, route)
        similarity = torch.where(forward[None, :, None, None], similarity, similarity.transpose(-1, -2))
        return similarity, source, target

    @staticmethod
    def _subset_seed(seed: int, group_id: int, occurrence: int, namespace: str) -> int:
        """Stable stateless RNG key; cardinality is intentionally excluded."""
        text = f"{int(seed)}:{int(group_id)}:{int(occurrence)}:{namespace}".encode("utf-8")
        value = 1469598103934665603
        for byte in text:
            value = ((value ^ byte) * 1099511628211) & ((1 << 63) - 1)
        return value

    def set_group_psd_calibration_draw(self, draw: int) -> None:
        self._group_psd_sampling_draw = int(draw)

    def _selected_curve_ids(self, *, group_id: int, occurrence: int, device: torch.device) -> torch.Tensor:
        # k=8 must be bit-for-bit identical to the historical Full relation
        # ordering: [0,...,7].  For k<8, a common random permutation is drawn
        # without replacement and then sorted only for block-order stability.
        if self.group_psd_cardinality == 8:
            return torch.arange(8, device=device)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._subset_seed(self._experiment_seed, group_id, occurrence, self.group_psd_subset_namespace))
        selected = torch.randperm(8, generator=generator)[: self.group_psd_cardinality].sort().values
        return selected.to(device=device)

    def prediction_only_group_psd_loss(
        self, out: Mapping[str, torch.Tensor | str], similarity: torch.Tensor | None = None,
        *, group_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Existing prediction-only PSD branch, exposed for gradient calibration."""
        if similarity is None:
            similarity, _source, _target = self._directed_similarity(out)
        batch_size, _pairs, length, _width = similarity.shape
        count = batch_size if self.group_psd_batch_groups is None else min(batch_size, self.group_psd_batch_groups)
        indices = (torch.arange(count, device=similarity.device) + (self._group_psd_offset_py % batch_size)).remainder(batch_size)
        selected = similarity.index_select(0, indices)
        if group_ids is None:
            group_ids = torch.arange(batch_size, device=similarity.device, dtype=torch.long)
        group_ids = group_ids.to(device=similarity.device, dtype=torch.long).index_select(0, indices)
        # Match the completed CMU direct-cosine continuation: Group PSD sees
        # the exact additive emission but does not tune its alpha/beta scalars.
        emission = (
            out["group_crf_alpha"].detach().float() * selected.clamp(-1.0, 1.0)
            + out["group_crf_beta"].detach().float()
            - math.log(float(length))
        )
        from ssia_alignment.loss_parts.direct_cosine_group_psd import soft_route_psd_group_loss
        # The two outer GRC groups retain independent stateless subsets, but
        # their selected full-resolution occupancies are computed together.
        # This preserves the objective while avoiding a second Python 512-step
        # CRF-DP loop for each training batch.
        members = torch.stack([
            self._selected_curve_ids(
                group_id=group_id,
                occurrence=self._group_psd_offset_py + local_index + self._group_psd_sampling_draw * 1_000_003,
                device=emission.device,
            )
            for local_index, group_id in enumerate(group_ids.tolist())
        ])
        result = soft_route_psd_group_loss(
            emission, curves=8, num_positions=self.group_psd_num_positions,
            objective_mode=self.group_psd_objective_mode, selected_curves=members,
        )
        per_group = result.per_group_loss
        # Every diagnostic below is an arithmetic mean over precisely the same
        # outer groups as established E1; no small-k reweighting is introduced.
        def mean_result(name: str) -> torch.Tensor:
            return getattr(result, name)
        result_loss = per_group.mean()
        return result_loss, {
            "group_psd_groups_used": emission.new_tensor(float(count)),
            "group_psd_offset": emission.new_tensor(float(self._group_psd_offset_py)),
            "group_psd_positions_used": emission.new_tensor(float(self.group_psd_num_positions)),
            "group_psd_cardinality": emission.new_tensor(float(self.group_psd_cardinality)),
            "group_psd_min_eigenvalue": mean_result("mean_min_eigenvalue").detach(),
            "group_psd_negative_eigenvalue_count": mean_result("mean_negative_eigenvalue_count").detach(),
            "group_psd_directional_energy": mean_result("mean_directional_energy").detach(),
            "group_psd_negative_spectral_energy": mean_result("mean_negative_spectral_energy").detach(),
            "group_psd_offdiagonal_relation_mass": mean_result("mean_offdiagonal_relation_mass").detach(),
            "group_psd_total_relation_mass": mean_result("mean_total_relation_mass").detach(),
            # Concise aliases make the two components of the one unified
            # Group Loss easy to inspect in ordinary training logs.
            "group_groups_used": emission.new_tensor(float(count)),
            "group_positions_used": emission.new_tensor(float(self.group_psd_num_positions)),
            "group_min_eigenvalue": mean_result("mean_min_eigenvalue").detach(),
            "group_negative_eigenvalue_count": mean_result("mean_negative_eigenvalue_count").detach(),
            "group_directional_energy": mean_result("mean_directional_energy").detach(),
            "group_negative_spectral_energy": mean_result("mean_negative_spectral_energy").detach(),
            "group_offdiagonal_relation_mass": mean_result("mean_offdiagonal_relation_mass").detach(),
            "group_total_relation_mass": mean_result("mean_total_relation_mass").detach(),
        }

    def forward(self, out: Mapping[str, torch.Tensor | str], batch: Mapping[str, torch.Tensor], *, compute_diagnostics: bool = True, profile_timer=None, **_ignored: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del compute_diagnostics, profile_timer
        similarity, source, target = self._directed_similarity(out)
        batch_size, _pairs, length, _width = similarity.shape
        score = out["group_crf_alpha"].float() * similarity.clamp(-1.0, 1.0) + out["group_crf_beta"].float()
        # Keep the batch-major order used by ``score.reshape``.  The former
        # pair-major label layout silently mismatched scores and GT whenever
        # batch_size > 1.
        packed_score = score.reshape(batch_size * len(source), length, length)
        q = batch["q_of_p"][:, source, target].reshape(batch_size * len(source), length)
        valid = batch["valid_pair"][:, source, target].reshape(batch_size * len(source), length).bool()
        q = self._nearest_integer(q, valid, length)
        crf, crf_logs = always_overlap_crf_loss(packed_score, q, valid, corridor_radius=0.0)
        effective_group_psd_weight = self.effective_group_psd_weight
        if effective_group_psd_weight:
            group_psd, psd_logs = self.prediction_only_group_psd_loss(out, similarity, group_ids=batch.get("storage_index"))
            total = crf + effective_group_psd_weight * group_psd
        else:
            group_psd = crf * 0.0
            total = crf
            psd_logs = {
                "group_psd_groups_used": crf.detach() * 0.0,
                "group_psd_offset": crf.detach() * 0.0,
                "group_psd_positions_used": crf.detach() * 0.0,
                "group_psd_cardinality": crf.detach() * 0.0,
                "group_psd_min_eigenvalue": crf.detach() * 0.0,
                "group_psd_negative_eigenvalue_count": crf.detach() * 0.0,
                "group_psd_directional_energy": crf.detach() * 0.0,
                "group_psd_negative_spectral_energy": crf.detach() * 0.0,
                "group_psd_offdiagonal_relation_mass": crf.detach() * 0.0,
                "group_psd_total_relation_mass": crf.detach() * 0.0,
                "group_groups_used": crf.detach() * 0.0,
                "group_positions_used": crf.detach() * 0.0,
                "group_min_eigenvalue": crf.detach() * 0.0,
                "group_negative_eigenvalue_count": crf.detach() * 0.0,
                "group_directional_energy": crf.detach() * 0.0,
                "group_negative_spectral_energy": crf.detach() * 0.0,
                "group_offdiagonal_relation_mass": crf.detach() * 0.0,
                "group_total_relation_mass": crf.detach() * 0.0,
            }
        zero = crf.detach() * 0.0
        return total, {
            "l_crf": crf.detach(), "l_crf_weighted": crf.detach(),
            "l_projector": zero, "l_projector_weighted": zero,
            "l_group_psd": group_psd.detach(), "l_group_psd_weighted": (effective_group_psd_weight * group_psd).detach(),
            "group_psd_loss": group_psd.detach(),
            "l_total": total.detach(), "projector_scale": zero,
            "group_psd_weight": total.new_tensor(effective_group_psd_weight),
            "group_psd_configured_weight": total.new_tensor(self.group_psd_weight),
            "group_psd_active": total.new_tensor(float(self.group_psd_is_active)),
            **crf_logs, **psd_logs, "loss": total.detach(),
        }
