"""Frozen verified reference for the partial-overlap monotone CRF.

The legal output set and score are frozen to the verified research reference:

* one no-overlap output with score zero;
* otherwise one contiguous source interval of length at least two;
* one integer target per matched source row, with non-decreasing target indices;
* score = segment_bias + sum(score[p,q] + match_bias - log(Q)).

The custom autograd implementation recomputes DP states during backward and
uses an explicit forward-backward recurrence.  This avoids retaining a
512-layer generic-autograd graph while preserving the exact reference model.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch


def default_segment_bias(source_length: int) -> float:
    """Uniform base measure over all source intervals with length >= 2."""

    candidates = source_length * (source_length - 1) // 2
    if candidates < 1:
        raise ValueError("partial-overlap CRF needs source_length >= 2")
    return -math.log(candidates)


def _validate_inputs(
    scores: torch.Tensor, q_of_p: torch.Tensor, valid_pair: torch.Tensor
) -> None:
    if scores.ndim != 3:
        raise ValueError(f"scores must be [N,P,Q], got {tuple(scores.shape)}")
    if q_of_p.shape != scores.shape[:2] or valid_pair.shape != scores.shape[:2]:
        raise ValueError("q_of_p and valid_pair must have shape [N,P]")
    if scores.shape[1] < 2 or scores.shape[2] < 1:
        raise ValueError("partial-overlap CRF requires P>=2 and Q>=1")


def _emission(scores: torch.Tensor, match_bias: float) -> torch.Tensor:
    return scores + float(match_bias) - math.log(scores.shape[-1])


def _all_forward(
    emission: torch.Tensor, segment_bias: float, *, store_states: bool = True, allow_empty: bool = True
) -> tuple[torch.Tensor, torch.Tensor | None]:
    items, source_length, target_length = emission.shape
    negative = emission.new_full((items, target_length), -torch.inf)
    multi_states = emission.new_full(emission.shape, -torch.inf) if store_states else None
    previous_single = negative
    previous_multi = negative
    endpoint_logsum = emission.new_full((items,), -torch.inf)
    for p in range(source_length):
        single = emission[:, p] + float(segment_bias)
        if p > 0:
            combined = torch.logaddexp(previous_single, previous_multi)
            multi = emission[:, p] + torch.logcumsumexp(combined, dim=1)
            if multi_states is not None:
                multi_states[:, p] = multi
            endpoint_logsum = torch.logaddexp(
                endpoint_logsum, torch.logsumexp(multi, dim=1)
            )
        else:
            multi = negative
        previous_single = single
        previous_multi = multi
    logz = torch.logaddexp(emission.new_zeros(items), endpoint_logsum) if allow_empty else endpoint_logsum
    return logz, multi_states


def _gt_forward(
    emission: torch.Tensor,
    q_of_p: torch.Tensor,
    valid_pair: torch.Tensor,
    segment_bias: float,
    corridor_radius: float,
    *,
    store_states: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    items, source_length, target_length = emission.shape
    valid = valid_pair.bool()
    has_overlap = valid.any(dim=1)
    columns = torch.arange(
        target_length, device=emission.device, dtype=q_of_p.dtype
    )[None, :]
    negative = emission.new_full((items, target_length), -torch.inf)
    states = emission.new_full(emission.shape, -torch.inf) if store_states else None
    state = negative
    previous_active = torch.zeros(items, device=emission.device, dtype=torch.bool)
    logz = emission.new_zeros(items)
    for p in range(source_length):
        active = valid[:, p]
        starts = active & ~previous_active
        start_score = emission[:, p] + float(segment_bias)
        continuation = emission[:, p] + torch.logcumsumexp(state, dim=1)
        current = torch.where(starts[:, None], start_score, continuation)
        allowed = (columns - q_of_p[:, p, None]).abs() <= float(corridor_radius)
        current = torch.where(active[:, None] & allowed, current, negative)
        if states is not None:
            states[:, p] = current
        ends = active & (
            True if p == source_length - 1 else ~valid[:, p + 1]
        )
        logz = torch.where(ends, torch.logsumexp(current, dim=1), logz)
        state = current
        previous_active = active
    if bool((has_overlap & ~torch.isfinite(logz)).any()):
        raise RuntimeError("GT corridor contains no legal monotone path")
    return logz, states


def _forward_values(
    scores: torch.Tensor,
    q_of_p: torch.Tensor,
    valid_pair: torch.Tensor,
    match_bias: float,
    segment_bias: float,
    corridor_radius: float,
    *,
    allow_empty: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    emission = _emission(scores, match_bias)
    logz_all, _ = _all_forward(emission, segment_bias, store_states=False, allow_empty=allow_empty)
    logz_gt, _ = _gt_forward(
        emission,
        q_of_p,
        valid_pair,
        segment_bias,
        corridor_radius,
        store_states=False,
    )
    per_item = (logz_all - logz_gt) / float(scores.shape[1])
    overlap_posterior = -torch.expm1(-logz_all).clamp(max=0.0) if allow_empty else torch.ones_like(logz_all)
    return per_item.mean(), logz_all, logz_gt, overlap_posterior


def partial_overlap_crf_loss_autograd(
    scores: torch.Tensor,
    q_of_p: torch.Tensor,
    valid_pair: torch.Tensor,
    *,
    match_bias: float = -2.0,
    segment_bias: float | None = None,
    corridor_radius: float = 2.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Batched generic-autograd form used only for equivalence checks."""

    _validate_inputs(scores, q_of_p, valid_pair)
    segment = (
        default_segment_bias(scores.shape[1])
        if segment_bias is None
        else float(segment_bias)
    )
    loss, logz_all, logz_gt, overlap = _forward_values(
        scores, q_of_p, valid_pair, match_bias, segment, corridor_radius
    )
    return loss, {
        "align_logz_all": logz_all.mean().detach(),
        "align_logz_gt": logz_gt.mean().detach(),
        "align_overlap_posterior": overlap.mean().detach(),
    }


def _suffix_adjoint(
    previous: torch.Tensor, prefix: torch.Tensor, adjoint: torch.Tensor
) -> torch.Tensor:
    negative = torch.full_like(adjoint, -torch.inf)
    log_terms = torch.where(
        adjoint > 0.0,
        adjoint.clamp_min(torch.finfo(adjoint.dtype).tiny).log() - prefix,
        negative,
    )
    log_suffix = torch.logcumsumexp(log_terms.flip(1), dim=1).flip(1)
    return torch.exp(previous + log_suffix)


@torch.no_grad()
def _explicit_gradient(
    scores: torch.Tensor,
    q_of_p: torch.Tensor,
    valid_pair: torch.Tensor,
    match_bias: float,
    segment_bias: float,
    corridor_radius: float,
    *,
    allow_empty: bool = True,
) -> torch.Tensor:
    emission = _emission(scores, match_bias)
    items, source_length, _target_length = emission.shape

    logz_all, multi = _all_forward(emission, segment_bias, store_states=True, allow_empty=allow_empty)
    assert multi is not None
    # Reuse the all-path forward table for the final gradient.  During the
    # reverse scan row p is never needed again after its adjoint has been
    # computed, so overwriting it saves one full [N, P, Q] allocation.
    adjoint_multi = torch.zeros_like(emission[:, 0])
    single_adjoint = torch.zeros_like(adjoint_multi)
    for p in range(source_length - 1, 0, -1):
        endpoint_adjoint = torch.exp(multi[:, p] - logz_all[:, None])
        state_adjoint = adjoint_multi + endpoint_adjoint
        emission_adjoint = state_adjoint + single_adjoint

        previous_single = emission[:, p - 1] + float(segment_bias)
        previous_multi = multi[:, p - 1]
        combined = torch.logaddexp(previous_single, previous_multi)
        prefix = multi[:, p] - emission[:, p]
        adjoint_combined = _suffix_adjoint(combined, prefix, state_adjoint)
        single_weight = torch.exp(previous_single - combined)
        multi_weight = torch.exp(previous_multi - combined)
        single_adjoint = adjoint_combined * single_weight
        adjoint_multi = adjoint_combined * multi_weight
        multi[:, p].copy_(emission_adjoint)
    multi[:, 0].copy_(single_adjoint)

    logz_gt, gt_states = _gt_forward(
        emission,
        q_of_p,
        valid_pair,
        segment_bias,
        corridor_radius,
        store_states=True,
    )
    assert gt_states is not None
    valid = valid_pair.bool()
    adjoint = torch.zeros_like(emission[:, 0])
    for p in range(source_length - 1, -1, -1):
        active = valid[:, p]
        ends = active & (
            True if p == source_length - 1 else ~valid[:, p + 1]
        )
        terminal = torch.exp(gt_states[:, p] - logz_gt[:, None])
        adjoint = adjoint + torch.where(ends[:, None], terminal, 0.0)
        adjoint = torch.where(active[:, None], adjoint, 0.0)
        multi[:, p].sub_(adjoint)
        if p > 0:
            continuation = active & valid[:, p - 1]
            prefix = gt_states[:, p] - emission[:, p]
            previous = gt_states[:, p - 1]
            previous_adjoint = _suffix_adjoint(previous, prefix, adjoint)
            adjoint = torch.where(continuation[:, None], previous_adjoint, 0.0)

    return multi / float(items * source_length)


class _PartialOverlapCRFFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        scores: torch.Tensor,
        q_of_p: torch.Tensor,
        valid_pair: torch.Tensor,
        match_bias: float,
        segment_bias: float,
        corridor_radius: float,
    ):
        with torch.no_grad():
            loss, logz_all, logz_gt, overlap = _forward_values(
                scores,
                q_of_p,
                valid_pair,
                float(match_bias),
                float(segment_bias),
                float(corridor_radius),
            )
        ctx.save_for_backward(scores, q_of_p, valid_pair)
        ctx.match_bias = float(match_bias)
        ctx.segment_bias = float(segment_bias)
        ctx.corridor_radius = float(corridor_radius)
        diagnostics = (
            logz_all.mean(),
            logz_gt.mean(),
            overlap.mean(),
        )
        ctx.mark_non_differentiable(*diagnostics)
        return (loss, *diagnostics)

    @staticmethod
    def backward(ctx, grad_loss, _grad_all, _grad_gt, _grad_overlap):
        scores, q_of_p, valid_pair = ctx.saved_tensors
        gradient = _explicit_gradient(
            scores,
            q_of_p,
            valid_pair,
            ctx.match_bias,
            ctx.segment_bias,
            ctx.corridor_radius,
        )
        return gradient * grad_loss, None, None, None, None, None


class _AlwaysOverlapCRFFunction(torch.autograd.Function):
    """Memory-bounded backward for the no-empty, zero-segment CRF variant."""

    @staticmethod
    def forward(ctx, scores: torch.Tensor, q_of_p: torch.Tensor, valid_pair: torch.Tensor):
        with torch.no_grad():
            loss, logz_all, logz_gt, overlap = _forward_values(
                scores, q_of_p, valid_pair, 0.0, 0.0, 0.0, allow_empty=False,
            )
        ctx.save_for_backward(scores, q_of_p, valid_pair)
        diagnostics = (logz_all.mean(), logz_gt.mean(), overlap.mean())
        ctx.mark_non_differentiable(*diagnostics)
        return (loss, *diagnostics)

    @staticmethod
    def backward(ctx, grad_loss, _grad_all, _grad_gt, _grad_overlap):
        scores, q_of_p, valid_pair = ctx.saved_tensors
        gradient = _explicit_gradient(
            scores, q_of_p, valid_pair, 0.0, 0.0, 0.0, allow_empty=False,
        )
        return gradient * grad_loss, None, None


def partial_overlap_crf_loss(
    scores: torch.Tensor,
    q_of_p: torch.Tensor,
    valid_pair: torch.Tensor,
    *,
    match_bias: float = -2.0,
    segment_bias: float | None = None,
    corridor_radius: float = 2.0,
    use_precomputed_corridor: bool = False,
    debug_checks: bool = False,
    backward_profile_timer=None,
    gt_reverse_backend: str = "pytorch",
    gt_reverse_num_warps: int = 8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Frozen reference entry point.

    The last three keyword arguments are accepted only so the standalone
    profiler can swap this module behind the optimized public interface.  The
    reference algorithm deliberately ignores them.
    """

    _validate_inputs(scores, q_of_p, valid_pair)
    segment = (
        default_segment_bias(scores.shape[1])
        if segment_bias is None
        else float(segment_bias)
    )
    loss, logz_all, logz_gt, overlap = _PartialOverlapCRFFunction.apply(
        scores,
        q_of_p,
        valid_pair,
        float(match_bias),
        segment,
        float(corridor_radius),
    )
    return loss, {
        "align_logz_all": logz_all.detach(),
        "align_logz_gt": logz_gt.detach(),
        "align_overlap_posterior": overlap.detach(),
        "align_match_bias": loss.new_tensor(float(match_bias)),
        "align_segment_bias": loss.new_tensor(segment),
    }


def always_overlap_crf_loss(
    scores: torch.Tensor,
    q_of_p: torch.Tensor,
    valid_pair: torch.Tensor,
    *,
    corridor_radius: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Always-Overlap partial-path CRF used by the slot-free direct model.

    This keeps the established monotone partial-path state graph, but removes
    the empty output and the segment-existence prior.  The implementation is
    intentionally separate from :func:`partial_overlap_crf_loss`: historical
    END calls retain their exact empty-state behavior.
    """
    _validate_inputs(scores, q_of_p, valid_pair)
    if float(corridor_radius) != 0.0:
        raise ValueError("the slot-free main route fixes exact r=0 supervision")
    loss, logz_all, logz_gt, overlap = _AlwaysOverlapCRFFunction.apply(scores, q_of_p, valid_pair)
    return loss, {
        "align_logz_all": logz_all.detach(),
        "align_logz_gt": logz_gt.detach(),
        "align_overlap_posterior": overlap.detach(),
        "align_match_bias": loss.new_zeros(()),
        "align_segment_bias": loss.new_zeros(()),
    }


@torch.no_grad()
def partial_overlap_log_partition(
    scores: torch.Tensor,
    *,
    match_bias: float = -2.0,
    segment_bias: float | None = None,
) -> torch.Tensor:
    """Per-item logZ_all for evaluation/calibration diagnostics."""

    if scores.ndim != 3:
        raise ValueError("scores must be [N,P,Q]")
    segment = (
        default_segment_bias(scores.shape[1])
        if segment_bias is None
        else float(segment_bias)
    )
    logz, _ = _all_forward(
        _emission(scores, match_bias), segment, store_states=False
    )
    return logz


@dataclass(frozen=True)
class BatchedViterbiResult:
    score: torch.Tensor
    has_overlap: torch.Tensor
    bound_source: torch.Tensor
    bound_target: torch.Tensor
    q_of_p: torch.Tensor


@torch.no_grad()
def viterbi_decode_batch(
    scores: torch.Tensor,
    *,
    match_bias: float = -2.0,
    segment_bias: float | None = None,
    allow_empty: bool = True,
) -> BatchedViterbiResult:
    """GPU batched max-product decode over the training CRF state graph."""

    if scores.ndim != 3:
        raise ValueError("scores must be [N,P,Q]")
    items, source_length, target_length = scores.shape
    segment = (default_segment_bias(source_length) if segment_bias is None else float(segment_bias)) if allow_empty else 0.0
    emission = _emission(scores, match_bias)
    negative = emission.new_full((items, target_length), -torch.inf)
    previous_single = negative
    previous_multi = negative
    predecessor_q = torch.full(
        (items, source_length, target_length),
        -1,
        dtype=torch.int16,
        device=scores.device,
    )
    predecessor_multi = torch.zeros(
        (items, source_length, target_length),
        dtype=torch.bool,
        device=scores.device,
    )
    best_score = emission.new_full((items,), -torch.inf)
    best_p = torch.full((items,), -1, dtype=torch.int16, device=scores.device)
    best_q = torch.full((items,), -1, dtype=torch.int16, device=scores.device)

    for p in range(source_length):
        single = emission[:, p] + segment
        if p > 0:
            from_multi = previous_multi > previous_single
            combined = torch.maximum(previous_single, previous_multi)
            prefix, prefix_q = torch.cummax(combined, dim=1)
            multi = emission[:, p] + prefix
            predecessor_q[:, p] = prefix_q.to(torch.int16)
            predecessor_multi[:, p] = torch.gather(from_multi, 1, prefix_q)
            row_score, row_q = multi.max(dim=1)
            improve = row_score > best_score
            best_score = torch.where(improve, row_score, best_score)
            best_p = torch.where(
                improve, torch.full_like(best_p, p), best_p
            )
            best_q = torch.where(improve, row_q.to(torch.int16), best_q)
        else:
            multi = negative
        previous_single = single
        previous_multi = multi

    has_overlap = best_score > 0.0 if allow_empty else torch.ones(items, dtype=torch.bool, device=scores.device)
    q_output = torch.full(
        (items, source_length), -1.0, dtype=torch.float32, device=scores.device
    )
    source_bounds = torch.full(
        (items, 2), -1, dtype=torch.int32, device=scores.device
    )
    target_bounds = torch.full(
        (items, 2), -1, dtype=torch.int32, device=scores.device
    )
    item_ids = torch.arange(items, device=scores.device)
    current_p = best_p.long()
    current_q = best_q.long()
    active = has_overlap.clone()
    source_bounds[:, 1] = torch.where(active, current_p, -1).to(torch.int32)
    target_bounds[:, 1] = torch.where(active, current_q, -1).to(torch.int32)
    for _ in range(source_length):
        if not bool(active.any()):
            break
        active_ids = item_ids[active]
        p = current_p[active]
        q = current_q[active]
        q_output[active_ids, p] = q.float()
        previous_q = predecessor_q[active_ids, p, q].long()
        from_multi = predecessor_multi[active_ids, p, q]
        finishing_ids = active_ids[~from_multi]
        if finishing_ids.numel():
            finishing_p = p[~from_multi] - 1
            finishing_q = previous_q[~from_multi]
            q_output[finishing_ids, finishing_p] = finishing_q.float()
            source_bounds[finishing_ids, 0] = finishing_p.to(torch.int32)
            target_bounds[finishing_ids, 0] = finishing_q.to(torch.int32)
        next_active = active.clone()
        next_active[active_ids[~from_multi]] = False
        current_p[active_ids[from_multi]] -= 1
        current_q[active_ids] = previous_q
        active = next_active
    return BatchedViterbiResult(
        score=torch.where(has_overlap, best_score, 0.0).detach().cpu(),
        has_overlap=has_overlap.detach().cpu(),
        bound_source=source_bounds.detach().cpu(),
        bound_target=target_bounds.detach().cpu(),
        q_of_p=q_output.detach().cpu(),
    )
