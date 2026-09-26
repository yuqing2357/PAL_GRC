"""END CorridorCRF plus exact low-rank Projector on selected temporal nodes."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from cmu_alignment.data import PAIR_ORDER
from cmu_alignment.partial_overlap_crf import (
    _pair_tensors,
    _zero_radius_lattice_targets,
    is_directional_emission_mode,
    ragged_alignment_loss,
    ragged_pair_scoring_lattices,
)
from cmu_alignment.soft_route_psd import soft_route_psd_group_loss
from cmu_alignment.direct_cosine_group_psd import direct_cosine_group_psd_loss


_PAIR_INDEX = {pair: index for index, pair in enumerate(PAIR_ORDER)}


def gt_log_similarity_gaussian_ce_loss(
    raw_similarity: torch.Tensor,
    lengths: torch.Tensor,
    q_of_p: torch.Tensor,
    pair_valid: torch.Tensor,
    *,
    similarity_eps: float = 1e-6,
    zero_radius_continuous_target_policy: str = "nearest_integer",
    target_radius: int = 2,
    target_sigma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GT-path Gaussian similarity CE from the CRF's *same* raw lattice.

    The routed tensor contains twelve source-by-target views assembled from
    six unordered membership Gram products.  Applying target-axis log-softmax
    to each view is therefore row normalization for ``i -> j`` and, without
    another GEMM, column normalization of the canonical block for ``j -> i``.
    Each valid source row has a discrete r=0 GT centre.  Its target is a
    truncated, renormalized Gaussian over integer offsets
    ``[-target_radius, ..., +target_radius]``.  Only GT-supported source rows
    contribute; non-overlap rows have no label and are deliberately absent.
    """
    if raw_similarity.ndim != 4 or raw_similarity.shape[1] != len(PAIR_ORDER):
        raise ValueError("raw_similarity must be [B,12,L,L]")
    if similarity_eps <= 0 or target_radius < 0 or target_sigma <= 0:
        raise ValueError("similarity_eps and target_sigma must be positive; target_radius must be non-negative")
    batch, _pairs, width, _ = raw_similarity.shape
    if lengths.shape != (batch, 4) or q_of_p.shape[:3] != (batch, 4, 4) or pair_valid.shape[:3] != (batch, 4, 4):
        raise ValueError("K4 lengths, q_of_p, and pair_valid are required")
    pair_index, _unordered, _route, _forward = _pair_tensors(raw_similarity.device)
    source_ids, target_ids = pair_index[:, 0], pair_index[:, 1]
    target_length = lengths[:, target_ids]
    target_mask = torch.arange(width, device=raw_similarity.device)[None, None, None] < target_length[:, :, None, None]
    log_similarity = raw_similarity.float().clamp_min(similarity_eps).log()
    log_probability = F.log_softmax(log_similarity.masked_fill(~target_mask, -torch.inf), dim=-1)
    target_coordinate = q_of_p[:, source_ids, target_ids].reshape(batch * len(PAIR_ORDER), width)
    valid = pair_valid[:, source_ids, target_ids].reshape(batch * len(PAIR_ORDER), width).bool()
    target_coordinate = _zero_radius_lattice_targets(
        target_coordinate, valid, target_length.reshape(-1), zero_radius_continuous_target_policy,
    ).reshape(batch, len(PAIR_ORDER), width).long()
    valid = valid.reshape(batch, len(PAIR_ORDER), width)
    # Invalid source rows have no label.  Valid GT targets must already lie
    # in the true target prefix; this check makes a data/support mismatch loud.
    if raw_similarity.device.type == "cpu" and (valid & ((target_coordinate < 0) | (target_coordinate >= target_length[:, :, None]))).any():
        raise ValueError("valid GT target is outside the true target length")
    target_coordinate = target_coordinate.clamp_(0, width - 1)
    offsets = torch.arange(-target_radius, target_radius + 1, device=raw_similarity.device)
    candidate_coordinate = target_coordinate[..., None] + offsets
    candidate_valid = (candidate_coordinate >= 0) & (candidate_coordinate < target_length[:, :, None, None])
    candidate_coordinate = candidate_coordinate.clamp_(0, width - 1)
    gaussian = torch.exp(-0.5 * (offsets.to(dtype=log_probability.dtype) / float(target_sigma)).square())
    gaussian = gaussian.view(1, 1, 1, -1) * candidate_valid.to(dtype=log_probability.dtype)
    gaussian = gaussian / gaussian.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(log_probability.dtype).tiny)
    negative_log_probability = -(log_probability.gather(-1, candidate_coordinate) * gaussian).sum(dim=-1)
    weight = valid.to(dtype=negative_log_probability.dtype)
    per_pair = (negative_log_probability * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1)
    # The CMU K=4 contract has a legal GT path for every directed pair.  The
    # clamp makes malformed/empty reference rows contribute an exact zero
    # rather than changing any other pair's weighting.
    has_label = weight.sum(dim=-1) > 0
    per_pair = torch.where(has_label, per_pair, torch.zeros_like(per_pair))
    per_group = per_pair.sum(dim=-1) / has_label.sum(dim=-1).clamp_min(1)
    return per_group, weight.sum()


def gt_log_similarity_ce_loss(
    raw_similarity: torch.Tensor,
    lengths: torch.Tensor,
    q_of_p: torch.Tensor,
    pair_valid: torch.Tensor,
    *,
    similarity_eps: float = 1e-6,
    zero_radius_continuous_target_policy: str = "nearest_integer",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact-point r=0 compatibility wrapper for the original CE control."""
    return gt_log_similarity_gaussian_ce_loss(
        raw_similarity, lengths, q_of_p, pair_valid,
        similarity_eps=similarity_eps,
        zero_radius_continuous_target_policy=zero_radius_continuous_target_policy,
        target_radius=0, target_sigma=1.0,
    )


def _nearest_lattice_coordinate(value: torch.Tensor, limit: int) -> torch.Tensor:
    """Use the same discrete GT convention as the formal r=0 CRF target."""
    return torch.floor(value + .5).long().clamp_(0, max(0, limit - 1))


def gt_anchor_group_score_loss(
    score_lattices: torch.Tensor,
    lengths: torch.Tensor,
    q_of_p: torch.Tensor,
    pair_valid: torch.Tensor,
    *,
    radius: int,
    lambda_variance: float,
    variance_epsilon: float,
    anchors: torch.Tensor | None = None,
    lattices_are_emissions: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """GT-only anchor/witness consistency on an explicit pre-CRF lattice.

    Legacy calls supply CRF score tensors and read their existing
    ``score-log(Q)`` emissions.  Directional-rowmean calls supply
    ``normalized_similarity`` directly; no CRF energy, decoder, posterior,
    softmax, or normalized probability representation is used in that path.
    """
    if score_lattices.ndim != 4 or score_lattices.shape[1] != len(PAIR_ORDER):
        raise ValueError("score_lattices must be [B,12,L,L]")
    if radius < 0 or lambda_variance < 0 or variance_epsilon <= 0:
        raise ValueError("invalid group-score loss hyperparameter")
    batch, _pairs, width, _ = score_lattices.shape
    groups = lengths.shape[1]
    if groups < 4 or q_of_p.shape[:3] != (batch, groups, groups):
        raise ValueError("K>=4 GT correspondence tensors are required")
    if anchors is None:
        # CPU sampling avoids a GPU scalar synchronization in the per-group
        # construction loop; worker/rank seeds make this reproducible.
        first = torch.randint(groups, (batch,), device="cpu")
        second = (first + 1 + torch.randint(groups - 1, (batch,), device="cpu")) % groups
        anchors = torch.stack((first, second), dim=1)
    if anchors.shape != (batch, 2):
        raise ValueError("anchors must have shape [B,2]")

    offsets = torch.arange(-radius, radius + 1, device=score_lattices.device)
    source_positions = torch.arange(width, device=score_lattices.device)
    per_group, per_mean, per_variance = [], [], []
    point_count = score_lattices.new_zeros(())
    group_count = score_lattices.new_zeros(())
    for row, (anchor_a, anchor_b) in enumerate(anchors.tolist()):
        witnesses = [index for index in range(groups) if index not in {anchor_a, anchor_b}]
        q_ab = q_of_p[row, anchor_a, anchor_b]
        source_b = _nearest_lattice_coordinate(q_ab, width)
        valid_ab = pair_valid[row, anchor_a, anchor_b]
        witness_a, witness_b, witness_masks = [], [], []
        for witness in witnesses:
            target_length = lengths[row, witness]
            center = _nearest_lattice_coordinate(q_of_p[row, anchor_a, witness], width)
            target = (center[:, None] + offsets[None]).clamp_(0, width - 1)
            full_window = (
                valid_ab
                & pair_valid[row, anchor_a, witness]
                & (center - radius >= 0)
                & (center + radius < target_length)
                & (source_b < lengths[row, anchor_b])
            )
            emission_offset = torch.log(lengths[row, witness].to(dtype=score_lattices.dtype)) if lattices_are_emissions else score_lattices.new_zeros(())
            matrix_a = score_lattices[row, _PAIR_INDEX[(anchor_a, witness)]]
            matrix_b = score_lattices[row, _PAIR_INDEX[(anchor_b, witness)]]
            witness_a.append(matrix_a[source_positions[:, None], target] - emission_offset)
            witness_b.append(matrix_b[source_b[:, None], target] - emission_offset)
            witness_masks.append(full_window)
        curves_a, curves_b = torch.stack(witness_a), torch.stack(witness_b)
        masks = torch.stack(witness_masks)
        count = masks.sum(dim=0)
        eligible = count >= 2
        weight = masks.to(dtype=score_lattices.dtype)[:, :, None]
        denominator = count.clamp_min(1).to(dtype=score_lattices.dtype)[:, None]
        mean_a = (curves_a * weight).sum(dim=0) / denominator
        mean_b = (curves_b * weight).sum(dim=0) / denominator
        centered_a = mean_a - mean_a.mean(dim=-1, keepdim=True)
        centered_b = mean_b - mean_b.mean(dim=-1, keepdim=True)
        mean_point = F.smooth_l1_loss(centered_a, centered_b, reduction="none").mean(dim=-1)
        variance_a = ((curves_a - mean_a[None]).square() * weight).sum(dim=0) / denominator
        variance_b = ((curves_b - mean_b[None]).square() * weight).sum(dim=0) / denominator
        variance_point = F.smooth_l1_loss(
            torch.log(variance_a + variance_epsilon),
            torch.log(variance_b + variance_epsilon), reduction="none",
        ).mean(dim=-1)
        eligible_weight = eligible.to(dtype=score_lattices.dtype)
        count_eligible = eligible_weight.sum()
        # A group with no eligible point contributes an exact differentiable
        # zero.  Keeping this tensor-only avoids one CPU/GPU synchronization
        # per K4 group in the training hot path.
        per_mean.append((mean_point * eligible_weight).sum() / count_eligible.clamp_min(1))
        per_variance.append((variance_point * eligible_weight).sum() / count_eligible.clamp_min(1))
        per_group.append(per_mean[-1] + float(lambda_variance) * per_variance[-1])
        point_count = point_count + count_eligible
        group_count = group_count + (count_eligible > 0).to(score_lattices.dtype)
    return (torch.stack(per_group), torch.stack(per_mean), torch.stack(per_variance), point_count, group_count)


def _projector_nodes(H: torch.Tensor, valid: torch.Tensor, stride: int) -> tuple[torch.Tensor, torch.Tensor]:
    if stride < 1: raise ValueError("projector stride must be >= 1")
    return H[:, :, ::stride], valid[:, :, ::stride]


def projector_explicit(H: torch.Tensor, valid: torch.Tensor, eps: float = 1e-12, stride: int = 1, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Reference O(N²) Projector used only by correctness tests."""
    values = []
    H, valid = _projector_nodes(H, valid, stride)
    for h, mask in zip(H, valid, strict=True):
        x = h[mask].to(dtype)
        if len(x) < 2:
            values.append(h.sum() * 0.0); continue
        relation = x @ x.T
        degree = relation.sum(-1).clamp_min(eps)
        normalized = relation * torch.rsqrt(degree)[:, None] * torch.rsqrt(degree)[None, :]
        values.append(((normalized @ normalized - normalized).square().sum() / normalized.square().sum().clamp_min(eps)).to(h.dtype))
    return torch.stack(values)


def exact_low_rank_projector(H: torch.Tensor, valid: torch.Tensor, eps: float = 1e-12, stride: int = 1, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Exact Projector through the S×S Gram matrix on nodes selected by stride.

    For A=D^-1/2 H and G=AᵀA, ||AAᵀ||²=tr(G²) and
    ||(AAᵀ)²-AAᵀ||²=tr(G⁴)-2tr(G³)+tr(G²). Low-rank evaluation adds no
    approximation beyond the explicitly configured temporal sampling.
    """
    values = []
    H, valid = _projector_nodes(H, valid, stride)
    for h, mask in zip(H, valid, strict=True):
        x = h[mask].to(dtype)
        if len(x) < 2:
            values.append(h.sum() * 0.0); continue
        column_mass = x.T @ torch.ones(len(x), device=x.device, dtype=x.dtype)
        degree = (x @ column_mass).clamp_min(eps)
        A = x * torch.rsqrt(degree)[:, None]
        G = A.T @ A
        G2 = G @ G
        tr2 = torch.trace(G2)
        tr3 = torch.trace(G2 @ G)
        tr4 = torch.trace(G2 @ G2)
        numerator = (tr4 - 2 * tr3 + tr2).clamp_min(0)
        values.append((numerator / tr2.clamp_min(eps)).to(h.dtype))
    return torch.stack(values)


class ENDCriterion(torch.nn.Module):
    def __init__(
        self,
        corridor_radius: float = 2.0,
        lambda_projector: float = .1,
        global_mean_sample_weight: float = 1.0,
        crf_reference_length: float | None = None,
        projector_stride: int = 1,
        projector_batch_groups: int | None = None,
        projector_dtype: torch.dtype = torch.float32,
        allow_no_overlap_output: bool = True,
        emission_mode: str = "legacy",
        zero_radius_continuous_target_policy: str = "error",
        lambda_group_score: float = 0.0,
        group_score_radius: int = 2,
        lambda_group_variance: float = 1.0,
        group_score_variance_epsilon: float = 1e-4,
        lambda_similarity: float = 0.0,
        similarity_target_radius: int = 0,
        similarity_target_sigma: float = 1.0,
        similarity_eps: float = 1e-6,
        normalization_eps: float = 1e-8,
        lambda_group_psd: float = 0.0,
        group_psd_num_bins: int = 32,
        group_psd_position_fraction: float | None = None,
        group_psd_batch_groups: int | None = None,
        group_psd_enabled: bool | None = None,
        group_psd_start_epoch: int = 1,
        group_psd_objective_mode: str = "full",
    ):
        super().__init__()
        if global_mean_sample_weight <= 0 or crf_reference_length is None: raise ValueError("positive global parent weight and crf_reference_length are required")
        self.radius, self.lambda_projector = float(corridor_radius), float(lambda_projector)
        self.crf_reference_length = float(crf_reference_length)
        self.projector_stride = int(projector_stride)
        self.projector_batch_groups = projector_batch_groups
        self.projector_dtype = projector_dtype
        self.allow_no_overlap_output = bool(allow_no_overlap_output)
        self.emission_mode = str(emission_mode)
        self.zero_radius_continuous_target_policy = str(zero_radius_continuous_target_policy)
        self.lambda_group_score = float(lambda_group_score)
        self.group_score_radius = int(group_score_radius)
        self.lambda_group_variance = float(lambda_group_variance)
        self.group_score_variance_epsilon = float(group_score_variance_epsilon)
        self.lambda_similarity = float(lambda_similarity)
        self.similarity_target_radius = int(similarity_target_radius)
        self.similarity_target_sigma = float(similarity_target_sigma)
        self.similarity_eps = float(similarity_eps)
        self.normalization_eps = float(normalization_eps)
        self.lambda_group_psd = float(lambda_group_psd)
        self.group_psd_num_bins = int(group_psd_num_bins)
        self.group_psd_position_fraction = None if group_psd_position_fraction is None else float(group_psd_position_fraction)
        self.group_psd_batch_groups = group_psd_batch_groups
        # ``None`` preserves prior direct callers: a positive lambda enables
        # Group-PSD immediately.  Formal CMU continuation configurations set
        # this explicitly and activate it only after their CRF-only stage.
        self.group_psd_enabled = self.lambda_group_psd > 0.0 if group_psd_enabled is None else bool(group_psd_enabled)
        self.group_psd_start_epoch = int(group_psd_start_epoch)
        self.group_psd_objective_mode = str(group_psd_objective_mode).lower()
        self._group_psd_active = self.group_psd_enabled and self.lambda_group_psd > 0.0 and self.group_psd_start_epoch <= 1
        if self.lambda_group_score < 0 or self.group_score_radius < 0 or self.lambda_group_variance < 0 or self.group_score_variance_epsilon <= 0 or self.lambda_similarity < 0 or self.similarity_target_radius < 0 or self.similarity_target_sigma <= 0 or self.similarity_eps <= 0 or self.normalization_eps <= 0 or self.lambda_group_psd < 0 or self.group_psd_num_bins < 1 or self.group_psd_start_epoch < 1 or self.group_psd_objective_mode not in {"full", "direction", "spectral"} or (self.group_psd_position_fraction is not None and not 0.0 < self.group_psd_position_fraction <= 1.0) or (self.group_psd_batch_groups is not None and int(self.group_psd_batch_groups) < 1):
            raise ValueError("invalid auxiliary loss configuration")
        if self.lambda_group_psd > 0.0 and not self.group_psd_enabled:
            raise ValueError("a positive Group-PSD weight requires group_psd_enabled=true")
        self.register_buffer("projector_offset", torch.zeros((), dtype=torch.long))
        self.register_buffer("group_psd_offset", torch.zeros((), dtype=torch.long))
        # Match D34's performance path: this scalar controls a deterministic
        # rotating estimator, so keeping it as Python state during ordinary
        # updates avoids a tiny CUDA op (and an eventual scalar read) per
        # training step.  Checkpoints use the explicit accessor below.
        self._projector_offset_py = 0
        self._group_psd_offset_py = 0
        self.register_buffer("global_mean_sample_weight", torch.tensor(float(global_mean_sample_weight)))

    def set_projector_group_offset(self, offset: int) -> None:
        self._projector_offset_py = int(offset)
        self.projector_offset.fill_(self._projector_offset_py)

    def advance_projector_group_offset(self, count: int) -> None:
        increment = int(count)
        self._projector_offset_py += increment
        # CPU unit tests and CPU-only callers retain an immediately observable
        # persistent buffer.  CUDA training uses the D34 Python control path
        # and materializes this buffer only at checkpoint time.
        if self.projector_offset.device.type == "cpu":
            self.projector_offset.add_(increment)

    def set_group_psd_group_offset(self, offset: int) -> None:
        self._group_psd_offset_py = int(offset)
        self.group_psd_offset.fill_(self._group_psd_offset_py)

    def advance_group_psd_group_offset(self, count: int) -> None:
        increment = int(count)
        self._group_psd_offset_py += increment
        if self.group_psd_offset.device.type == "cpu":
            self.group_psd_offset.add_(increment)

    @property
    def group_psd_is_active(self) -> bool:
        return self._group_psd_active

    def set_training_epoch(self, epoch: int) -> None:
        """Activate Group-PSD at the declared training epoch only."""
        if epoch < 1:
            raise ValueError("training epoch must be positive")
        self._group_psd_active = (
            self.group_psd_enabled
            and self.lambda_group_psd > 0.0
            and epoch >= self.group_psd_start_epoch
        )

    @property
    def projector_offset_value(self) -> int:
        """Current control state without a GPU scalar synchronization."""
        return self._projector_offset_py

    @property
    def group_psd_offset_value(self) -> int:
        """Current rotating PSD-estimator control state without GPU sync."""
        return self._group_psd_offset_py

    def sync_persistent_control_state(self) -> None:
        """Materialize Python control state only when serializing a checkpoint."""
        self.projector_offset.fill_(self._projector_offset_py)
        self.group_psd_offset.fill_(self._group_psd_offset_py)

    def prediction_only_group_psd_loss(self, out):
        """Return the validated direct-cosine Group-PSD term without GT.

        This public calibration hook intentionally detaches CRF alpha/beta.
        It therefore measures only the matching-backbone gradient that the
        auxiliary branch contributes during continuation training.
        """
        matching_features = getattr(out, "matching_features", out.memberships)
        matching_mode = getattr(out, "matching_mode", "membership_log")
        if matching_mode != "direct_cosine":
            raise ValueError("the migrated direct relation Group-PSD is only defined for direct-cosine matching")
        if self.emission_mode not in {"legacy", "equivalent_log", "equivalent_log_stable"}:
            raise ValueError("direct-cosine Group-PSD requires the established additive CRF emission")
        scoring = ragged_pair_scoring_lattices(
            matching_features, out.lengths, out.group_crf_alpha, out.group_crf_beta,
            reference_length=self.crf_reference_length, eps=self.similarity_eps,
            similarity_mode=matching_mode, emission_mode=self.emission_mode,
            crf_temperature=getattr(out, "group_crf_temperature", None),
            gamma=getattr(out, "group_crf_gamma", None), normalization_eps=self.normalization_eps,
        )
        emission = (
            out.group_crf_alpha.detach().float() * scoring.raw_similarity.clamp(-1.0, 1.0)
            + out.group_crf_beta.detach().float()
            - torch.log(torch.as_tensor(self.crf_reference_length, device=matching_features.device, dtype=torch.float32))
        )
        return direct_cosine_group_psd_loss(emission, out.lengths, num_positions=self.group_psd_num_bins, position_fraction=self.group_psd_position_fraction, objective_mode=self.group_psd_objective_mode)

    def forward(self, out, batch: dict[str, torch.Tensor], projector_scale: float = 1.0, distributed_training: bool = False, compute_diagnostics: bool = True) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        # Small reference tests and external legacy callers may still supply
        # the historical membership-only output namespace.  Formal models
        # always expose the explicit matching/projector contracts below.
        matching_features=getattr(out,'matching_features',out.memberships)
        matching_mode=getattr(out,'matching_mode','membership_log')
        projector_features=getattr(out,'projector_features',out.memberships)
        group_score_active = self.lambda_group_score != 0.0 and torch.is_grad_enabled()
        group_psd_active = self.group_psd_is_active
        similarity_active = self.lambda_similarity != 0.0
        score_lattices = None
        scoring = None
        if group_score_active or group_psd_active or similarity_active:
            # Construct raw / directed relation / CRF output once.  The new
            # directional mode passes normalized similarity to Group Loss and
            # its separately calibrated emission to the exact CRF.
            scoring = ragged_pair_scoring_lattices(
                matching_features, out.lengths, out.group_crf_alpha, out.group_crf_beta,
                reference_length=self.crf_reference_length, eps=self.similarity_eps, similarity_mode=matching_mode,
                emission_mode=self.emission_mode,
                crf_temperature=getattr(out, "group_crf_temperature", None),
                gamma=getattr(out, "group_crf_gamma", None),
                normalization_eps=self.normalization_eps,
            )
            score_lattices = scoring.crf_input
        alignment, cells = ragged_alignment_loss(
            matching_features, out.lengths, batch["q_of_p"], batch["pair_valid"],
            out.group_crf_alpha, out.group_crf_beta, self.radius,
            reference_length=self.crf_reference_length, return_cells=compute_diagnostics,
            similarity_mode=matching_mode,
            allow_empty=self.allow_no_overlap_output,
            emission_mode=self.emission_mode,
            zero_radius_continuous_target_policy=self.zero_radius_continuous_target_policy,
            score_lattices=score_lattices,
            crf_temperature=getattr(out, "group_crf_temperature", None),
            gamma=getattr(out, "group_crf_gamma", None),
            eps=self.similarity_eps,
            normalization_eps=self.normalization_eps,
            score_lattices_are_emissions=is_directional_emission_mode(self.emission_mode),
        )
        B = len(matching_features)
        scale = batch["sample_weight"] * batch.get("schedule_weight",torch.ones_like(batch["sample_weight"])) / self.global_mean_sample_weight.to(alignment.dtype)
        # On the final DDP schedule step some ranks receive synthetic zero
        # weight entries.  Normalize by the *global real* group count so their
        # presence cannot reduce the contribution of the true final batch.
        normalizer=torch.tensor(float(B),device=alignment.device,dtype=alignment.dtype)
        if distributed_training:
            import torch.distributed as dist
            if not dist.is_initialized():raise RuntimeError("distributed_training requires an initialized process group")
            real=batch.get("schedule_weight",torch.ones_like(batch["sample_weight"])).sum().to(alignment.dtype)
            dist.all_reduce(real,op=dist.ReduceOp.SUM)
            normalizer=real/dist.get_world_size()
        weighted_alignment=(scale*alignment).sum()/normalizer.clamp_min(1)
        if similarity_active:
            if scoring is None:
                raise RuntimeError("similarity loss requires a shared scoring lattice")
            similarity, similarity_points = gt_log_similarity_gaussian_ce_loss(
                scoring.raw_similarity, out.lengths, batch["q_of_p"], batch["pair_valid"],
                similarity_eps=self.similarity_eps,
                zero_radius_continuous_target_policy=self.zero_radius_continuous_target_policy,
                target_radius=self.similarity_target_radius,
                target_sigma=self.similarity_target_sigma,
            )
            weighted_similarity = (scale * similarity).sum() / normalizer.clamp_min(1)
            unweighted_similarity = similarity.mean()
        else:
            similarity = alignment * 0.0
            weighted_similarity = alignment.sum() * 0.0
            unweighted_similarity = alignment.sum() * 0.0
            similarity_points = alignment.sum() * 0.0
        if group_score_active:
            group_lattice = scoring.normalized_similarity if is_directional_emission_mode(self.emission_mode) else score_lattices
            if group_lattice is None:
                raise RuntimeError("directional group loss requires normalized similarity")
            group_score, group_mean, group_variance, group_points, group_groups = gt_anchor_group_score_loss(
                group_lattice, out.lengths, batch["q_of_p"], batch["pair_valid"],
                radius=self.group_score_radius, lambda_variance=self.lambda_group_variance,
                variance_epsilon=self.group_score_variance_epsilon,
                lattices_are_emissions=not is_directional_emission_mode(self.emission_mode),
            )
            weighted_group_score=(scale*group_score).sum()/normalizer.clamp_min(1)
            unweighted_group_score=group_score.mean()
            unweighted_group_mean=group_mean.mean()
            unweighted_group_variance=group_variance.mean()
        else:
            group_score = alignment * 0.0
            weighted_group_score = alignment.sum() * 0.0
            unweighted_group_score = alignment.sum() * 0.0
            unweighted_group_mean = alignment.sum() * 0.0
            unweighted_group_variance = alignment.sum() * 0.0
            group_points = alignment.sum() * 0.0
            group_groups = alignment.sum() * 0.0
        if group_psd_active:
            if scoring is None:
                raise RuntimeError("group PSD loss requires the shared raw similarity lattice")
            if self.emission_mode not in {"legacy", "equivalent_log", "equivalent_log_stable"}:
                raise ValueError("prediction-only soft-route PSD requires an established additive CRF emission")
            # This is the unchanged simplified Always-Overlap emission.  The
            # group branch reads alpha/beta values but cannot optimize either
            # calibration scalar; only the matching representation receives
            # its second-order posterior gradient.
            # The expensive differentiable posterior/HVP is evaluated only on
            # a deterministic rotating subset.  The B/count factor below is
            # the unbiased all-group estimator; the pairwise CRF still uses
            # every group in this batch.
            group_psd_count = B if self.group_psd_batch_groups is None else min(B, int(self.group_psd_batch_groups))
            group_psd_indices = (torch.arange(group_psd_count, device=alignment.device) + (self._group_psd_offset_py % B)).remainder(B)
            raw_group_similarity = scoring.raw_similarity.index_select(0, group_psd_indices).float()
            group_lengths = out.lengths.index_select(0, group_psd_indices)
            if matching_mode == "direct_cosine":
                # Current D34 semantics, adapted only for CMU's ragged native
                # lengths: preserve all directed occupancy blocks exactly.
                # No reciprocal averaging, pooling, or degree normalization.
                group_emission = (
                    out.group_crf_alpha.detach().float() * raw_group_similarity.clamp(-1.0, 1.0)
                    + out.group_crf_beta.detach().float()
                    - torch.log(torch.as_tensor(self.crf_reference_length, device=alignment.device, dtype=torch.float32))
                )
                psd = direct_cosine_group_psd_loss(
                    group_emission, group_lengths, num_positions=self.group_psd_num_bins,
                    position_fraction=self.group_psd_position_fraction,
                    objective_mode=self.group_psd_objective_mode,
                )
            elif matching_mode == "membership_log":
                # Keep historical membership/slot experiments reproducible;
                # the new direct-relation implementation is deliberately not
                # retrofitted into a different method family.
                group_emission = (
                    out.group_crf_alpha.detach().float() * raw_group_similarity.clamp_min(self.similarity_eps).log()
                    + out.group_crf_beta.detach().float()
                    - torch.log(torch.as_tensor(self.crf_reference_length, device=alignment.device, dtype=torch.float32))
                )
                psd = soft_route_psd_group_loss(group_emission, group_lengths, num_bins=self.group_psd_num_bins)
            else:
                raise ValueError(f"prediction-only Group-PSD does not support matching mode {matching_mode!r}")
            group_psd = alignment.new_zeros((B,)).index_copy(0, group_psd_indices, psd.per_group_loss)
            weighted_group_psd = (scale.index_select(0, group_psd_indices) * psd.per_group_loss).sum() * (B / max(1, group_psd_count)) / normalizer.clamp_min(1)
            unweighted_group_psd = psd.per_group_loss.mean()
        else:
            group_psd = alignment * 0.0
            weighted_group_psd = alignment.sum() * 0.0
            unweighted_group_psd = alignment.sum() * 0.0
            psd = None
            group_psd_count = 0
            group_psd_indices = torch.empty(0, dtype=torch.long, device=alignment.device)
        if psd is None:
            psd_positions = alignment.sum() * 0.0
            psd_negative_energy = alignment.sum() * 0.0
            psd_directional_energy = alignment.sum() * 0.0
            psd_entropy = alignment.sum() * 0.0
            psd_maximum = alignment.sum() * 0.0
            psd_path_length = alignment.sum() * 0.0
            psd_overlap_fraction = alignment.sum() * 0.0
            psd_offdiagonal_mass = alignment.sum() * 0.0
            psd_total_mass = alignment.sum() * 0.0
        else:
            psd_positions = alignment.new_tensor(float(getattr(psd, "positions_used", self.group_psd_num_bins)))
            psd_negative_energy = getattr(psd, "mean_negative_spectral_energy", None)
            if psd_negative_energy is None:
                psd_negative_energy = psd.mean_negative_spectral_mass
            psd_directional_energy = getattr(psd, "mean_directional_energy", None)
            if psd_directional_energy is None:
                psd_directional_energy = psd.mean_directional_disagreement
            psd_entropy = getattr(psd, "mean_posterior_entropy", alignment.sum() * 0.0)
            psd_maximum = getattr(psd, "mean_max_posterior", alignment.sum() * 0.0)
            psd_path_length = getattr(psd, "mean_expected_path_length", alignment.sum() * 0.0)
            psd_overlap_fraction = getattr(psd, "mean_predicted_overlap_fraction", alignment.sum() * 0.0)
            psd_offdiagonal_mass = getattr(psd, "mean_offdiagonal_relation_mass", alignment.sum() * 0.0)
            psd_total_mass = getattr(psd, "mean_total_relation_mass", alignment.sum() * 0.0)
        # Match the original END schedule: during projector warmup its exact
        # coefficient is zero, so do not construct the expensive relation at
        # all.  This changes neither the value nor the gradient of the total.
        projector_active = bool(projector_scale) and self.lambda_projector != 0.0
        if projector_active:
            count = B if self.projector_batch_groups is None else min(B, int(self.projector_batch_groups))
            indices = [(self._projector_offset_py + i) % B for i in range(count)]
            selected = exact_low_rank_projector(projector_features[indices], out.sequence_valid_mask[indices], stride=self.projector_stride, dtype=self.projector_dtype)
            # The rotating subset is a deterministic batch estimator of the
            # all-group parent-weighted Projector objective.
            weighted_projection=(scale[indices]*selected).sum()*(B/max(1,count))/normalizer.clamp_min(1)
            unweighted_projection=selected.mean()
            node_count=sum(int(out.sequence_valid_mask[i,:,::self.projector_stride].sum()) for i in indices) if compute_diagnostics else 0
        else:
            count, indices, node_count = 0, [], 0
            selected = alignment.new_zeros((1,))
            weighted_projection = alignment.sum() * 0.0
            unweighted_projection = alignment.sum() * 0.0
        projector_contribution=float(projector_scale)*self.lambda_projector*weighted_projection
        group_score_contribution=self.lambda_group_score*weighted_group_score
        group_psd_contribution=self.lambda_group_psd*weighted_group_psd
        similarity_contribution=self.lambda_similarity*weighted_similarity
        total=weighted_alignment+projector_contribution+group_score_contribution+group_psd_contribution+similarity_contribution
        return {
            "total_loss": total, "weighted_total_loss": total, "weighted_alignment_loss": weighted_alignment, "weighted_projector_loss": weighted_projection, "weighted_projector_contribution": projector_contribution, "weighted_group_score_loss": weighted_group_score, "weighted_group_score_contribution": group_score_contribution, "weighted_group_psd_loss": weighted_group_psd, "weighted_group_psd_contribution": group_psd_contribution, "weighted_similarity_loss": weighted_similarity, "weighted_similarity_contribution": similarity_contribution,
            "unweighted_total_loss": alignment.mean()+float(projector_scale)*self.lambda_projector*unweighted_projection+self.lambda_group_score*unweighted_group_score+self.lambda_group_psd*unweighted_group_psd+self.lambda_similarity*unweighted_similarity, "unweighted_alignment_loss": alignment.mean(), "unweighted_projector_loss": unweighted_projection, "unweighted_group_score_loss": unweighted_group_score, "unweighted_group_mean_loss": unweighted_group_mean, "unweighted_group_variance_loss": unweighted_group_variance, "unweighted_group_psd_loss": unweighted_group_psd, "unweighted_similarity_loss": unweighted_similarity,
            "alignment_loss": alignment.mean(), "projector_loss": selected.mean(), "group_score_loss": unweighted_group_score, "group_psd_loss": unweighted_group_psd, "similarity_loss": unweighted_similarity, "per_sample_alignment_loss": alignment, "per_sample_group_score_loss": group_score, "per_sample_group_psd_loss": group_psd, "per_sample_similarity_loss": similarity, "selected_projector_loss": selected, "selected_projector_indices":torch.tensor(indices,device=total.device),
            "diagnostics": {"pair_cell_count": torch.tensor(float(cells), device=total.device), "global_mean_sample_weight": self.global_mean_sample_weight.detach(), "projector_stride": torch.tensor(self.projector_stride, device=total.device), "projector_groups_used": torch.tensor(count, device=total.device), "projector_offset":total.new_tensor(self._projector_offset_py),"projector_node_count":torch.tensor(node_count,device=total.device),"group_score_points":group_points,"group_score_groups":group_groups,"similarity_points":similarity_points,
                "group_psd_groups_used": torch.tensor(group_psd_count, device=total.device), "group_psd_offset":total.new_tensor(self._group_psd_offset_py), "group_psd_active": total.new_tensor(float(self.group_psd_is_active)), "group_psd_positions_used": psd_positions, "group_psd_min_eigenvalue": alignment.sum() * 0.0 if psd is None else psd.mean_min_eigenvalue, "group_psd_negative_eigenvalue_count": alignment.sum() * 0.0 if psd is None else psd.mean_negative_eigenvalue_count, "group_psd_negative_spectral_mass": psd_negative_energy, "group_psd_directional_disagreement": psd_directional_energy, "group_psd_offdiagonal_relation_mass": psd_offdiagonal_mass, "group_psd_total_relation_mass": psd_total_mass, "group_psd_posterior_entropy": psd_entropy, "group_psd_max_posterior": psd_maximum, "group_psd_expected_path_length": psd_path_length, "group_psd_predicted_overlap_fraction": psd_overlap_fraction},
        }
