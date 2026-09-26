"""Frozen END CorridorCRF on true, variable ``P x Q`` lattices.

Membership has no NULL slot. The CRF deliberately has a distinct, zero-score
no-overlap output, exactly as the frozen END implementation does.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
import torch

from cmu_alignment.data import PAIR_ORDER

# Computing C_ij = H_i H_j^T for both directions duplicates a large GEMM:
# C_ji is exactly C_ij^T.  Keep the public directed-pair layout unchanged,
# but build the six unordered K=4 matrices once and route/transposed-view them
# back into the frozen 12-directed-pair contract.
UNORDERED_PAIR_ORDER = tuple((i, j) for i in range(4) for j in range(i + 1, 4))
_UNORDERED_LOOKUP = {pair: index for index, pair in enumerate(UNORDERED_PAIR_ORDER)}
DIRECTED_TO_UNORDERED = tuple(_UNORDERED_LOOKUP[(min(i, j), max(i, j))] for i, j in PAIR_ORDER)
DIRECTED_IS_FORWARD = tuple(i < j for i, j in PAIR_ORDER)
_PAIR_TENSOR_CACHE: dict[tuple[str, int | None], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_DIRECTED_EMISSION_MODES = frozenset({'directional_rowmean_log', 'directional_rowcol_softmax'})


def is_directional_emission_mode(emission_mode: str) -> bool:
    """Whether ``emission_mode`` supplies an additive emission to the CRF.

    Historical modes return ``emission + log(target_length)`` because the
    frozen recurrence removes that term.  Directional modes return the final
    additive lattice directly, so that historical length correction is never
    applied a second time.
    """
    return emission_mode in _DIRECTED_EMISSION_MODES


@dataclass(frozen=True)
class PairScoringLattices:
    """Explicit boundary between a raw affinity, directed relation, and CRF.

    ``raw_similarity`` preserves the symmetric membership Gram blocks.
    ``normalized_similarity`` is intentionally directional in the
    directional modes.  ``emission`` is the additive CRF
    lattice.  Legacy modes retain their historical score lattice in
    ``crf_input``; the directional mode stores its already-calibrated
    emission there and marks it accordingly.
    """
    raw_similarity: torch.Tensor
    normalized_similarity: torch.Tensor | None
    log_normalized_similarity: torch.Tensor | None
    emission: torch.Tensor
    crf_input: torch.Tensor
    crf_input_is_emission: bool


def _pair_tensors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cache tiny pair-routing tensors to avoid repeated host→device copies."""
    key = (device.type, device.index)
    if key not in _PAIR_TENSOR_CACHE:
        _PAIR_TENSOR_CACHE[key] = (
            torch.as_tensor(PAIR_ORDER, device=device, dtype=torch.long),
            torch.as_tensor(UNORDERED_PAIR_ORDER, device=device, dtype=torch.long),
            torch.as_tensor(DIRECTED_TO_UNORDERED, device=device, dtype=torch.long),
            torch.as_tensor(DIRECTED_IS_FORWARD, device=device, dtype=torch.bool),
        )
    return _PAIR_TENSOR_CACHE[key]


def _zero_radius_lattice_targets(
    q: torch.Tensor,
    valid: torch.Tensor,
    target_length: torch.Tensor,
    policy: str,
) -> torch.Tensor:
    """Make a strict ``r=0`` corridor representable on an integer lattice.

    CMU teachers are continuous native-frame coordinates whereas the CRF
    lattice has integer target columns.  ``nearest_integer`` is therefore a
    pre-registered discrete-label convention for the r=0 ablation; it uses
    nearest integer, ties upward, and clips only to the legal target domain.
    Other radii retain the original continuous teacher coordinates exactly.
    """
    if policy == "error":
        fractional = valid.bool() & ((q - q.round()).abs() > 1e-6)
        if fractional.any():
            raise ValueError("r=0 requires an explicit continuous-target policy for fractional GT coordinates")
        return q
    if policy != "nearest_integer":
        raise ValueError(f"unknown zero-radius continuous-target policy: {policy}")
    rounded = torch.floor(q + 0.5).clamp_min(0)
    rounded = torch.minimum(rounded, (target_length - 1).clamp_min(0)[:, None].to(rounded.dtype))
    return torch.where(valid.bool(), rounded, q)


def segment_bias(source_length: int) -> float:
    if source_length < 2:
        raise ValueError("CorridorCRF requires P >= 2")
    return -math.log(source_length * (source_length - 1) // 2)


def _all(score: torch.Tensor, segment: float = 0.0, *, allow_empty: bool = True, input_is_emission: bool = False) -> torch.Tensor:
    """Partition over legal partial monotone paths.

    ``allow_empty=True`` preserves the frozen END model: a zero-score empty
    output competes with every non-empty segment and ``segment`` is its
    calibrated segment-existence prior.  The Always-Overlap variant instead
    uses ``allow_empty=False, segment=0``: every candidate is non-empty and
    the former path-start constant is removed.
    """
    P, Q = score.shape
    emission = score if input_is_emission else score - math.log(Q)
    single = emission.new_full((Q,), -torch.inf)
    multi = emission.new_full((Q,), -torch.inf)
    endpoint = emission.new_tensor(-torch.inf)
    for p in range(P):
        new_single = emission[p] + segment
        if p:
            new_multi = emission[p] + torch.logcumsumexp(torch.logaddexp(single, multi), dim=0)
            endpoint = torch.logaddexp(endpoint, torch.logsumexp(new_multi, dim=0))
        else:
            new_multi = multi
        single, multi = new_single, new_multi
    return torch.logaddexp(emission.new_zeros(()), endpoint) if allow_empty else endpoint


def _gt(score: torch.Tensor, q: torch.Tensor, valid: torch.Tensor, segment: float, radius: float, *, input_is_emission: bool = False) -> torch.Tensor:
    """Nonempty GT partition over its exact contiguous source support."""
    _P, Q = score.shape
    ids = torch.nonzero(valid.bool(), as_tuple=False).flatten()
    if len(ids) < 2 or not valid[ids[0]:ids[-1] + 1].bool().all():
        raise ValueError("GT support must be contiguous and have at least two frames")
    emission = score if input_is_emission else score - math.log(Q)
    columns = torch.arange(Q, device=score.device, dtype=q.dtype)
    state = emission.new_full((Q,), -torch.inf)
    for p in range(int(ids[0]), int(ids[-1]) + 1):
        state = emission[p] + (segment if p == ids[0] else torch.logcumsumexp(state, dim=0))
        state = state.masked_fill((columns - q[p]).abs() > radius, -torch.inf)
    result = torch.logsumexp(state, dim=0)
    if not torch.isfinite(result):
        raise RuntimeError("GT corridor contains no legal monotone path")
    return result


def corridor_loss_one(
    score: torch.Tensor,
    q: torch.Tensor,
    valid: torch.Tensor,
    radius: float = 2.0,
    *,
    allow_empty: bool = True,
    input_is_emission: bool = False,
) -> torch.Tensor:
    if score.ndim != 2 or len(q) != score.shape[0] or len(valid) != score.shape[0]:
        raise ValueError("score [P,Q], q [P], valid [P] required")
    segment = segment_bias(score.shape[0]) if allow_empty else 0.0
    return (_all(score, segment, allow_empty=allow_empty, input_is_emission=input_is_emission) - _gt(score, q, valid, segment, radius, input_is_emission=input_is_emission)) / score.shape[0]


def directional_rowmean_log_normalize(
    raw_similarity: torch.Tensor,
    target_length: torch.Tensor | int,
    *,
    similarity_eps: float = 1e-6,
    normalization_eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-domain target-row-mean normalization for each directed lattice.

    A raw forward block may be transposed to obtain its raw reverse block,
    but this routine must be called again with the reverse target lengths.
    Consequently the returned relation is intentionally not transpose tied.
    Padding is excluded from the target-axis logsumexp denominator.
    """
    if raw_similarity.ndim < 2:
        raise ValueError("raw_similarity must have source and target dimensions")
    if similarity_eps <= 0 or normalization_eps <= 0:
        raise ValueError("similarity_eps and normalization_eps must be positive")
    width = raw_similarity.shape[-1]
    target_length = torch.as_tensor(target_length, device=raw_similarity.device, dtype=torch.long)
    if target_length.ndim == 0:
        target_length = target_length.reshape(1)
    batch_shape = raw_similarity.shape[:-2]
    if target_length.numel() != int(math.prod(batch_shape)):
        raise ValueError("target_length must provide one value per directed lattice")
    length = target_length.reshape(batch_shape)
    if (length < 1).any() or (length > width).any():
        raise ValueError("target lengths must be within the padded lattice width")
    positive_log = raw_similarity.float().clamp_min(similarity_eps).log()
    columns = torch.arange(width, device=raw_similarity.device)
    target_valid = columns < length[..., None]
    log_total = torch.logsumexp(positive_log.masked_fill(~target_valid[..., None, :], -torch.inf), dim=-1, keepdim=True)
    log_mean = log_total - length.to(positive_log.dtype).log()[..., None, None]
    # log(mean(S+) + eps_n), evaluated without materializing row means.
    log_denom = torch.logaddexp(log_mean, log_mean.new_full((), math.log(normalization_eps)))
    log_normalized = positive_log - log_denom
    return log_normalized, log_normalized.exp()


def directional_rowcol_log_softmax(
    raw_similarity: torch.Tensor,
    target_length: torch.Tensor | int,
    *,
    crf_temperature: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return directed target-axis log probabilities from a raw Gram lattice.

    The caller routes a single canonical unordered block ``S_AB`` into its
    forward view ``S_AB`` and reverse view ``S_AB.T``.  Applying this helper
    independently to each routed view exactly implements row softmax for
    ``A -> B`` and column softmax followed by transpose for ``B -> A``.
    Padded target columns are excluded before log-softmax.
    """
    if raw_similarity.ndim < 2:
        raise ValueError("raw_similarity must have source and target dimensions")
    width = raw_similarity.shape[-1]
    target_length = torch.as_tensor(target_length, device=raw_similarity.device, dtype=torch.long)
    if target_length.ndim == 0:
        target_length = target_length.reshape(1)
    batch_shape = raw_similarity.shape[:-2]
    if target_length.numel() != int(math.prod(batch_shape)):
        raise ValueError("target_length must provide one value per directed lattice")
    length = target_length.reshape(batch_shape)
    # Formal CUDA callers receive positive temperatures from the model and
    # contiguous data-loader lengths by construction.  Keep malformed-input
    # checks for CPU/reference callers without introducing a GPU scalar sync
    # into every training step.
    if raw_similarity.device.type == "cpu" and ((length < 1).any() or (length > width).any()):
        raise ValueError("target lengths must be within the padded lattice width")
    columns = torch.arange(width, device=raw_similarity.device)
    target_valid = columns < length[..., None]
    logits = raw_similarity.float() / crf_temperature.float()
    log_probability = torch.log_softmax(logits.masked_fill(~target_valid[..., None, :], -torch.inf), dim=-1)
    return log_probability, log_probability.exp()


def calibrated_pair_scores(
    left: torch.Tensor,
    right: torch.Tensor,
    alpha: torch.Tensor | None,
    beta: torch.Tensor | None,
    reference_length: float,
    eps: float = 1e-6,
    similarity_mode: str = 'membership_log',
    emission_mode: str = "legacy",
    crf_temperature: torch.Tensor | None = None,
    gamma: torch.Tensor | None = None,
    normalization_eps: float = 1e-8,
) -> torch.Tensor:
    """CRF score with the frozen pair-specific ``log(Q/L_ref)`` correction.

    ``membership_log`` is the original END emission ``log(H_i H_j^T)``.
    ``direct_cosine`` instead uses L2-normalised full-resolution embeddings:
    ``alpha * E_i E_j^T + beta``.  The partial-overlap CRF itself is shared.
    """
    Q = right.shape[0]
    similarity = left.float() @ right.float().T
    return _score_from_similarity(
        similarity,
        torch.as_tensor(Q, device=similarity.device),
        alpha,
        beta,
        reference_length=reference_length,
        eps=eps,
        similarity_mode=similarity_mode,
        emission_mode=emission_mode,
        crf_temperature=crf_temperature,
        gamma=gamma,
        normalization_eps=normalization_eps,
    ).reshape_as(similarity)


def _score_from_similarity(
    similarity: torch.Tensor,
    target_length: torch.Tensor,
    alpha: torch.Tensor | None,
    beta: torch.Tensor | None,
    *,
    reference_length: float,
    eps: float,
    similarity_mode: str,
    emission_mode: str,
    linear_a: torch.Tensor | float | None = None,
    linear_c: torch.Tensor | float | None = None,
    crf_temperature: torch.Tensor | None = None,
    gamma: torch.Tensor | None = None,
    normalization_eps: float = 1e-8,
) -> torch.Tensor:
    """Build a CRF *score* lattice from a routed similarity lattice.

    The CorridorCRF recurrence consumes ``score - log(Q)``.  ``legacy`` is
    the established public calibration; ``equivalent_log`` exposes its exact
    emission simplification, and ``linear`` is reserved for frozen score-form
    audits.  All three return the same score-lattice contract so the path
    space, END recurrence, empty state, and corridor code remain untouched.
    """
    if emission_mode == "directional_rowmean_log":
        if similarity_mode != "membership_log":
            raise ValueError("directional_rowmean_log requires nonnegative membership_log similarity")
        if crf_temperature is None or gamma is None:
            raise ValueError("directional_rowmean_log requires crf_temperature and gamma")
        if not bool((crf_temperature.detach() > 0).all()):
            raise ValueError("crf_temperature must be positive")
        log_normalized, _normalized = directional_rowmean_log_normalize(
            similarity, target_length, similarity_eps=eps, normalization_eps=normalization_eps,
        )
        # New CRF mode: this is an additive emission, not a score awaiting
        # the historical recurrence's -log(Q) correction.
        return log_normalized / crf_temperature.float() + gamma.float()
    if emission_mode == "directional_rowcol_softmax":
        if crf_temperature is None or gamma is None:
            raise ValueError("directional_rowcol_softmax requires crf_temperature and gamma")
        log_probability, _probability = directional_rowcol_log_softmax(
            similarity, target_length, crf_temperature=crf_temperature,
        )
        # +log(Q) is the sole target-length correction in this mode: a
        # uniform conditional relation has zero evidence before gamma.
        return log_probability + target_length.to(log_probability.dtype).log()[..., None, None] + gamma.float()
    if similarity_mode == "direct_cosine" and emission_mode != "legacy":
        raise ValueError("emission simplification audit is defined only for membership_log")
    target_log = torch.log(target_length.to(dtype=similarity.dtype))[..., None, None]
    if emission_mode in {"legacy", "equivalent_log_stable"}:
        if similarity_mode == "membership_log":
            emission = alpha.float() * similarity.clamp_min(eps).log() + beta.float()
        elif similarity_mode == "direct_cosine":
            emission = alpha.float() * similarity.clamp(-1.0, 1.0) + beta.float()
        else:
            raise ValueError(f"unknown similarity_mode: {similarity_mode}")
        # ``equivalent_log_stable`` deliberately retains this established
        # arithmetic schedule.  It is the bit-identical control for audits;
        # ``equivalent_log`` below exposes the cancelled formula itself.
        # Keeping both prevents long FP32 DP scans from confusing roundoff
        # with a semantic CRF change.
        return emission + torch.log(target_length.to(dtype=similarity.dtype) / float(reference_length))[..., None, None]
    if emission_mode == "equivalent_log":
        # alpha*log(x) + beta - log(L_ref), followed by score=E+log(Q).
        emission = alpha.float() * similarity.clamp_min(eps).log() + beta.float() - math.log(float(reference_length))
        return emission + target_log
    if emission_mode == "linear":
        if linear_a is None or linear_c is None:
            raise ValueError("linear emission requires both linear_a and linear_c")
        emission = torch.as_tensor(linear_a, device=similarity.device, dtype=similarity.dtype) * similarity + torch.as_tensor(linear_c, device=similarity.device, dtype=similarity.dtype)
        return emission + target_log
    raise ValueError(f"unknown emission_mode: {emission_mode}")


def ragged_pair_similarity_lattices(H: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Return the routed 12-directed-pair raw membership similarities.

    This is intentionally a score-free helper for frozen calibration and
    diagnostics.  Padding remains present and must be masked by the caller.
    """
    if H.ndim != 4 or H.shape[1] != 4 or lengths.shape != H.shape[:2]:
        raise ValueError("H [B,4,L,S] and lengths [B,4] required")
    batch, _groups, width, slots = H.shape
    _index, unordered, route, forward = _pair_tensors(H.device)
    left = H[:, unordered[:, 0]].reshape(batch * len(UNORDERED_PAIR_ORDER), width, slots)
    right = H[:, unordered[:, 1]].reshape(batch * len(UNORDERED_PAIR_ORDER), width, slots)
    unordered_similarity = torch.bmm(left.float(), right.float().transpose(1, 2)).reshape(batch, len(UNORDERED_PAIR_ORDER), width, width)
    similarity = unordered_similarity.index_select(1, route)
    return torch.where(forward[None, :, None, None], similarity, similarity.transpose(-1, -2))


def ragged_pair_score_lattices(
    H: torch.Tensor,
    lengths: torch.Tensor,
    alpha: torch.Tensor | None,
    beta: torch.Tensor | None,
    *,
    reference_length: float,
    eps: float = 1e-6,
    similarity_mode: str = "membership_log",
    emission_mode: str = "legacy",
    linear_a: torch.Tensor | float | None = None,
    linear_c: torch.Tensor | float | None = None,
    crf_temperature: torch.Tensor | None = None,
    gamma: torch.Tensor | None = None,
    normalization_eps: float = 1e-8,
) -> torch.Tensor:
    """Return the exact 12-directed-pair CRF score lattices.

    This is the shared score construction used by the production CorridorCRF
    loss and by post-CRF diagnostics.  The padded part of a lattice is kept in
    the tensor for batching; every CRF routine below masks it with ``lengths``
    before it becomes a legal state.
    """
    if H.ndim != 4 or H.shape[1] != 4 or lengths.shape != H.shape[:2]:
        raise ValueError("H [B,4,L,S] and lengths [B,4] required")
    return ragged_pair_scoring_lattices(
        H, lengths, alpha, beta, reference_length=reference_length, eps=eps,
        similarity_mode=similarity_mode, emission_mode=emission_mode,
        linear_a=linear_a, linear_c=linear_c, crf_temperature=crf_temperature,
        gamma=gamma, normalization_eps=normalization_eps,
    ).crf_input


def ragged_pair_scoring_lattices(
    H: torch.Tensor,
    lengths: torch.Tensor,
    alpha: torch.Tensor | None,
    beta: torch.Tensor | None,
    *,
    reference_length: float,
    eps: float = 1e-6,
    similarity_mode: str = "membership_log",
    emission_mode: str = "legacy",
    linear_a: torch.Tensor | float | None = None,
    linear_c: torch.Tensor | float | None = None,
    crf_temperature: torch.Tensor | None = None,
    gamma: torch.Tensor | None = None,
    normalization_eps: float = 1e-8,
    raw_similarity: torch.Tensor | None = None,
) -> PairScoringLattices:
    """Build raw, directional-normalized, and CRF lattices in one pass.

    In directional mode six unordered GEMMs are still sufficient for raw
    Gram affinity.  The twelve routed views are then normalized separately,
    so reverse blocks are never reconstructed by transposing a normalized
    forward block.
    """
    if H.ndim != 4 or H.shape[1] != 4 or lengths.shape != H.shape[:2]:
        raise ValueError("H [B,4,L,S] and lengths [B,4] required")
    _batch, _groups, _width, _slots = H.shape
    index, _unordered, _route, _forward = _pair_tensors(H.device)
    # Auxiliary objectives may already have constructed the routed twelve
    # directed views from the six unordered GEMMs.  Reusing that exact tensor
    # keeps every CRF mode tied to the same matching lattice and avoids a
    # second expensive similarity construction.
    raw = ragged_pair_similarity_lattices(H, lengths) if raw_similarity is None else raw_similarity
    if raw.shape != (H.shape[0], len(PAIR_ORDER), H.shape[2], H.shape[2]):
        raise ValueError("raw_similarity must have shape [B,12,L,L] matching H")
    target_length = lengths[:, index[:, 1]]
    if emission_mode == "directional_rowmean_log":
        if crf_temperature is None or gamma is None:
            raise ValueError("directional_rowmean_log requires crf_temperature and gamma")
        log_normalized, normalized = directional_rowmean_log_normalize(
            raw, target_length, similarity_eps=eps, normalization_eps=normalization_eps,
        )
        emission = log_normalized / crf_temperature.float() + gamma.float()
        return PairScoringLattices(raw, normalized, log_normalized, emission, emission, True)
    if emission_mode == "directional_rowcol_softmax":
        if crf_temperature is None or gamma is None:
            raise ValueError("directional_rowcol_softmax requires crf_temperature and gamma")
        # ``raw`` contains 12 routed views backed by six unordered GEMMs.
        # Forward views are S_AB; reverse views are S_AB.T.  Target-axis
        # softmax on the latter is precisely column-softmax(S_AB).T.
        log_probability, probability = directional_rowcol_log_softmax(
            raw, target_length, crf_temperature=crf_temperature,
        )
        emission = log_probability + target_length.to(log_probability.dtype).log()[..., None, None] + gamma.float()
        return PairScoringLattices(raw, probability, log_probability, emission, emission, True)
    score = _score_from_similarity(
        raw, target_length, alpha, beta, reference_length=reference_length, eps=eps,
        similarity_mode=similarity_mode, emission_mode=emission_mode,
        linear_a=linear_a, linear_c=linear_c,
    )
    # All historical modes pass score=emission+log(Q) into the frozen CRF.
    emission = score - target_length.to(score.dtype)[..., None, None].log()
    return PairScoringLattices(raw, None, None, emission, score, False)


def _ragged_alignment_loss_loop(H: torch.Tensor, lengths: torch.Tensor, q_of_p: torch.Tensor, pair_valid: torch.Tensor, alpha: torch.Tensor | None, beta: torch.Tensor | None, radius: float = 2.0, eps: float = 1e-6, reference_length: float | None = None, similarity_mode: str = 'membership_log', *, allow_empty: bool = True, emission_mode: str = "legacy", zero_radius_continuous_target_policy: str = "error", crf_temperature: torch.Tensor | None = None, gamma: torch.Tensor | None = None, normalization_eps: float = 1e-8) -> tuple[torch.Tensor, int]:
    """Literal reference used by tests for the batched production recurrence."""
    if reference_length is None: raise ValueError('formal crf_reference_length is required')
    values, cells = [], 0
    for b in range(len(H)):
        pair_losses = []
        for i, j in PAIR_ORDER:
            P, Q = int(lengths[b, i]), int(lengths[b, j])
            score = calibrated_pair_scores(H[b, i, :P], H[b, j, :Q], alpha, beta, reference_length, eps, similarity_mode, emission_mode, crf_temperature, gamma, normalization_eps)
            teacher = q_of_p[b, i, j, :P].float()
            if float(radius) == 0.0:
                teacher = _zero_radius_lattice_targets(teacher[None], pair_valid[b, i, j, :P][None], torch.as_tensor([Q], device=teacher.device), zero_radius_continuous_target_policy)[0]
            pair_losses.append(corridor_loss_one(score, teacher, pair_valid[b, i, j, :P], radius, allow_empty=allow_empty, input_is_emission=is_directional_emission_mode(emission_mode)))
            cells += P * Q
        values.append(torch.stack(pair_losses).mean())
    return torch.stack(values), cells


def _batched_all_partition(
    emission: torch.Tensor,
    source_length: torch.Tensor,
    *,
    allow_empty: bool = True,
    use_segment_bias: bool = True,
) -> torch.Tensor:
    """Exact batched form of :func:`_all` over padded source rows."""
    count, P, _Q = emission.shape
    single = emission.new_full((count, _Q), -torch.inf)
    multi = emission.new_full((count, _Q), -torch.inf)
    endpoint = emission.new_full((count,), -torch.inf)
    segment = -torch.log((source_length * (source_length - 1) // 2).to(emission.dtype)) if use_segment_bias else torch.zeros_like(source_length, dtype=emission.dtype)
    for p in range(P):
        active = p < source_length
        fresh = emission[:, p] + segment[:, None]
        if p:
            candidate = emission[:, p] + torch.logcumsumexp(torch.logaddexp(single, multi), dim=1)
            multi = torch.where(active[:, None], candidate, multi)
            endpoint = torch.where(active, torch.logaddexp(endpoint, torch.logsumexp(candidate, dim=1)), endpoint)
        single = torch.where(active[:, None], fresh, single)
    return torch.logaddexp(torch.zeros_like(endpoint), endpoint) if allow_empty else endpoint


def _batched_gt_partition(
    emission: torch.Tensor,
    source_length: torch.Tensor,
    q: torch.Tensor,
    valid: torch.Tensor,
    radius: float,
    *,
    use_segment_bias: bool = True,
) -> torch.Tensor:
    """Exact batched form of :func:`_gt` for contiguous full-utterance support."""
    count, P, Q = emission.shape
    valid = valid.bool()
    support_count = valid.sum(dim=1)
    if (support_count < 2).any():
        raise ValueError("GT support must be contiguous and have at least two frames")
    start = valid.to(torch.int64).argmax(dim=1)
    expected = (torch.arange(P, device=emission.device)[None] >= start[:, None]) & (torch.arange(P, device=emission.device)[None] < (start + support_count)[:, None])
    if not torch.equal(valid, expected):
        raise ValueError("GT support must be contiguous and have at least two frames")
    # The source lengths are part of the all-path segment prior.  Existing END
    # semantics use the same prior in the GT partition, not the valid interval.
    segment = -torch.log((source_length * (source_length - 1) // 2).to(emission.dtype)) if use_segment_bias else torch.zeros_like(source_length, dtype=emission.dtype)
    state = emission.new_full((count, Q), -torch.inf)
    columns = torch.arange(Q, device=emission.device, dtype=q.dtype)[None]
    for p in range(P):
        inside = expected[:, p]
        candidate = emission[:, p] + torch.logcumsumexp(state, dim=1)
        candidate = torch.where((p == start)[:, None], emission[:, p] + segment[:, None], candidate)
        candidate = candidate.masked_fill((columns - q[:, p:p + 1]).abs() > radius, -torch.inf)
        state = torch.where(inside[:, None], candidate, state)
    result = torch.logsumexp(state, dim=1)
    if not torch.isfinite(result).all():
        raise RuntimeError("GT corridor contains no legal monotone path")
    return result


def _batched_all_forward(
    emission: torch.Tensor,
    source_length: torch.Tensor,
    *,
    store_states: bool,
    allow_empty: bool = True,
    use_segment_bias: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """D34 END's all-path recurrence, generalized only over valid prefixes.

    The recurrence itself is unchanged.  ``source_length`` merely prevents a
    padded CMU row from becoming a legal source row.  When requested, retain
    the all-path table for the explicit backward used by the established END
    custom-autograd implementation.
    """
    count, P, Q = emission.shape
    negative = emission.new_full((count, Q), -torch.inf)
    states = emission.new_full(emission.shape, -torch.inf) if store_states else None
    single, multi = negative, negative
    endpoint = emission.new_full((count,), -torch.inf)
    segment = -torch.log((source_length * (source_length - 1) // 2).to(emission.dtype)) if use_segment_bias else torch.zeros_like(source_length, dtype=emission.dtype)
    for p in range(P):
        active = p < source_length
        fresh = emission[:, p] + segment[:, None]
        if p:
            candidate = emission[:, p] + torch.logcumsumexp(torch.logaddexp(single, multi), dim=1)
            if states is not None:
                states[:, p] = torch.where(active[:, None], candidate, negative)
            multi = torch.where(active[:, None], candidate, multi)
            endpoint = torch.where(active, torch.logaddexp(endpoint, torch.logsumexp(candidate, dim=1)), endpoint)
        single = torch.where(active[:, None], fresh, single)
    return (torch.logaddexp(torch.zeros_like(endpoint), endpoint) if allow_empty else endpoint), states


def _batched_gt_forward(
    emission: torch.Tensor,
    source_length: torch.Tensor,
    q: torch.Tensor,
    valid: torch.Tensor,
    radius: float,
    *,
    store_states: bool,
    use_segment_bias: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """D34 END's GT recurrence with the frozen CMU contiguous support."""
    count, P, Q = emission.shape
    valid = valid.bool()
    support_count = valid.sum(dim=1)
    if (support_count < 2).any():
        raise ValueError("GT support must be contiguous and have at least two frames")
    start = valid.to(torch.int64).argmax(dim=1)
    positions = torch.arange(P, device=emission.device)
    expected = (positions[None] >= start[:, None]) & (positions[None] < (start + support_count)[:, None])
    if not torch.equal(valid, expected):
        raise ValueError("GT support must be contiguous and have at least two frames")
    segment = -torch.log((source_length * (source_length - 1) // 2).to(emission.dtype)) if use_segment_bias else torch.zeros_like(source_length, dtype=emission.dtype)
    negative = emission.new_full((count, Q), -torch.inf)
    states = emission.new_full(emission.shape, -torch.inf) if store_states else None
    state = negative
    columns = torch.arange(Q, device=emission.device, dtype=q.dtype)[None]
    for p in range(P):
        inside = expected[:, p]
        candidate = emission[:, p] + torch.logcumsumexp(state, dim=1)
        candidate = torch.where((p == start)[:, None], emission[:, p] + segment[:, None], candidate)
        candidate = candidate.masked_fill((columns - q[:, p:p + 1]).abs() > radius, -torch.inf)
        state = torch.where(inside[:, None], candidate, state)
        if states is not None:
            states[:, p] = torch.where(inside[:, None], candidate, negative)
    result = torch.logsumexp(state, dim=1)
    if not torch.isfinite(result).all():
        raise RuntimeError("GT corridor contains no legal monotone path")
    return result, states


def _suffix_adjoint(previous: torch.Tensor, prefix: torch.Tensor, adjoint: torch.Tensor) -> torch.Tensor:
    """Verbatim END reverse-scan primitive from the D34 custom CRF."""
    negative = torch.full_like(adjoint, -torch.inf)
    terms = torch.where(adjoint > 0.0, adjoint.clamp_min(torch.finfo(adjoint.dtype).tiny).log() - prefix, negative)
    return torch.exp(previous + torch.logcumsumexp(terms.flip(1), dim=1).flip(1))


@torch.no_grad()
def _ragged_explicit_gradient(
    score: torch.Tensor,
    source_length: torch.Tensor,
    target_length: torch.Tensor,
    q: torch.Tensor,
    valid: torch.Tensor,
    radius: float,
    *,
    allow_empty: bool = True,
    use_segment_bias: bool = True,
    input_is_emission: bool = False,
) -> torch.Tensor:
    """Exact END forward-backward derivative on padded true-size lattices.

    This is the D34 verified explicit gradient, with only prefix masks added
    for CMU's native ``P x Q`` pairs.  It replaces generic autograd through
    hundreds of DP rows; neither a frame nor a target coordinate is resampled.
    """
    count, P, Q = score.shape
    columns = torch.arange(Q, device=score.device)[None]
    target_valid = columns < target_length[:, None]
    emission = (score if input_is_emission else score - torch.log(target_length.to(score.dtype))[:, None, None]).masked_fill(~target_valid[:, None], -torch.inf)
    logz_all, all_states = _batched_all_forward(
        emission, source_length, store_states=True,
        allow_empty=allow_empty, use_segment_bias=use_segment_bias,
    )
    logz_gt, gt_states = _batched_gt_forward(
        emission, source_length, q, valid, radius, store_states=True,
        use_segment_bias=use_segment_bias,
    )
    assert all_states is not None and gt_states is not None

    gradient = torch.zeros_like(emission)
    adjoint_multi = torch.zeros_like(emission[:, 0])
    adjoint_single = torch.zeros_like(adjoint_multi)
    segment = -torch.log((source_length * (source_length - 1) // 2).to(emission.dtype)) if use_segment_bias else torch.zeros_like(source_length, dtype=emission.dtype)
    for p in range(P - 1, 0, -1):
        active = p < source_length
        endpoint_adjoint = torch.where(active[:, None], torch.exp(all_states[:, p] - logz_all[:, None]), torch.zeros_like(adjoint_multi))
        state_adjoint = torch.where(active[:, None], adjoint_multi + endpoint_adjoint, torch.zeros_like(adjoint_multi))
        emission_adjoint = state_adjoint + adjoint_single
        gradient[:, p] = torch.where(active[:, None], emission_adjoint, gradient[:, p])
        previous_single = emission[:, p - 1] + segment[:, None]
        previous_multi = all_states[:, p - 1]
        combined = torch.logaddexp(previous_single, previous_multi)
        prefix = all_states[:, p] - emission[:, p]
        reverse = _suffix_adjoint(combined, prefix, state_adjoint)
        adjoint_single = torch.where(active[:, None], reverse * torch.exp(previous_single - combined), torch.zeros_like(adjoint_single))
        adjoint_multi = torch.where(active[:, None], reverse * torch.exp(previous_multi - combined), torch.zeros_like(adjoint_multi))
    gradient[:, 0] = adjoint_single

    valid = valid.bool()
    support_count = valid.sum(dim=1)
    start = valid.to(torch.int64).argmax(dim=1)
    positions = torch.arange(P, device=score.device)
    support = (positions[None] >= start[:, None]) & (positions[None] < (start + support_count)[:, None])
    adjoint = torch.zeros_like(emission[:, 0])
    for p in range(P - 1, -1, -1):
        active = support[:, p]
        ends = active & (True if p == P - 1 else ~support[:, p + 1])
        terminal = torch.exp(gt_states[:, p] - logz_gt[:, None])
        adjoint = adjoint + torch.where(ends[:, None], terminal, torch.zeros_like(adjoint))
        adjoint = torch.where(active[:, None], adjoint, torch.zeros_like(adjoint))
        gradient[:, p].sub_(adjoint)
        if p:
            continuation = active & support[:, p - 1]
            prefix = gt_states[:, p] - emission[:, p]
            reverse = _suffix_adjoint(gt_states[:, p - 1], prefix, adjoint)
            adjoint = torch.where(continuation[:, None], reverse, torch.zeros_like(adjoint))
    return gradient.masked_fill(~target_valid[:, None], 0.0) / source_length.to(score.dtype)[:, None, None]


class _RaggedPartialOverlapCRFFunction(torch.autograd.Function):
    """CMU variable-length adaptation of D34's verified custom CRF backward."""
    @staticmethod
    def forward(ctx, score, source_length, target_length, q, valid, radius, allow_empty=True, use_segment_bias=True, input_is_emission=False):
        with torch.no_grad():
            columns = torch.arange(score.shape[-1], device=score.device)[None]
            emission = (score if input_is_emission else score - torch.log(target_length.to(score.dtype))[:, None, None]).masked_fill(columns[:, None] >= target_length[:, None, None], -torch.inf)
            all_partition, _ = _batched_all_forward(
                emission, source_length, store_states=False,
                allow_empty=bool(allow_empty), use_segment_bias=bool(use_segment_bias),
            )
            gt_partition, _ = _batched_gt_forward(
                emission, source_length, q, valid, float(radius), store_states=False,
                use_segment_bias=bool(use_segment_bias),
            )
            value = (all_partition - gt_partition) / source_length.to(score.dtype)
        ctx.save_for_backward(score, source_length, target_length, q, valid)
        ctx.radius = float(radius)
        ctx.allow_empty = bool(allow_empty)
        ctx.use_segment_bias = bool(use_segment_bias)
        ctx.input_is_emission = bool(input_is_emission)
        return value

    @staticmethod
    def backward(ctx, grad_value):
        score, source_length, target_length, q, valid = ctx.saved_tensors
        gradient = _ragged_explicit_gradient(
            score, source_length, target_length, q, valid, ctx.radius,
            allow_empty=ctx.allow_empty, use_segment_bias=ctx.use_segment_bias, input_is_emission=ctx.input_is_emission,
        )
        return gradient * grad_value[:, None, None], None, None, None, None, None, None, None, None


def ragged_alignment_loss(
    H: torch.Tensor,
    lengths: torch.Tensor,
    q_of_p: torch.Tensor,
    pair_valid: torch.Tensor,
    alpha: torch.Tensor | None,
    beta: torch.Tensor | None,
    radius: float = 2.0,
    eps: float = 1e-6,
    reference_length: float | None = None,
    return_cells: bool = True,
    similarity_mode: str = 'membership_log',
    score_lattices: torch.Tensor | None = None,
    *,
    allow_empty: bool = True,
    emission_mode: str = "legacy",
    zero_radius_continuous_target_policy: str = "error",
    crf_temperature: torch.Tensor | None = None,
    gamma: torch.Tensor | None = None,
    normalization_eps: float = 1e-8,
    score_lattices_are_emissions: bool | None = None,
) -> tuple[torch.Tensor, int]:
    """Batched, exact production implementation of the frozen ragged CRF.

    We retain the true P×Q lattice through masks; padding only lets all 12
    directed pairs of every K=4 group share GPU kernels.  No path, target, or
    supervision value is changed relative to ``_ragged_alignment_loss_loop``.
    """
    if reference_length is None: raise ValueError('formal crf_reference_length is required')
    if H.ndim != 4 or H.shape[1] != 4 or lengths.shape != H.shape[:2]:
        raise ValueError("H [B,4,L,S] and lengths [B,4] required")
    B, _K, Pmax, _S = H.shape
    index, _unordered, _route, _forward = _pair_tensors(H.device)
    source_ids, target_ids = index[:, 0], index[:, 1]
    source_length = lengths[:, source_ids].reshape(-1)
    target_length = lengths[:, target_ids].reshape(-1)
    q = q_of_p[:, source_ids, target_ids].reshape(B * len(PAIR_ORDER), Pmax).float()
    valid = pair_valid[:, source_ids, target_ids].reshape(B * len(PAIR_ORDER), Pmax)
    if float(radius) < 0:
        raise ValueError("corridor radius must be non-negative")
    if float(radius) == 0.0:
        q = _zero_radius_lattice_targets(q, valid, target_length, zero_radius_continuous_target_policy)
    if score_lattices is None:
        score_lattices = ragged_pair_score_lattices(
            H, lengths, alpha, beta, reference_length=float(reference_length),
            eps=eps, similarity_mode=similarity_mode, emission_mode=emission_mode,
            crf_temperature=crf_temperature, gamma=gamma, normalization_eps=normalization_eps,
        )
    if score_lattices.shape != (B, len(PAIR_ORDER), Pmax, Pmax):
        raise ValueError("score_lattices must have shape [B,12,L,L] matching H")
    score = score_lattices.reshape(B * len(PAIR_ORDER), Pmax, Pmax)
    if score_lattices_are_emissions is None:
        score_lattices_are_emissions = is_directional_emission_mode(emission_mode)
    # Reuse the established D34 END custom-autograd strategy.  The forward
    # and explicit backward are exact on this padded representation; masks
    # preserve each CMU pair's original native P x Q lattice.
    loss = _RaggedPartialOverlapCRFFunction.apply(
        score, source_length, target_length, q, valid, float(radius),
        bool(allow_empty), bool(allow_empty), bool(score_lattices_are_emissions),
    )
    # Cells are diagnostics only.  Avoid a GPU→CPU scalar read on normal
    # training steps; it would serialize the CUDA prefetch/update pipeline.
    cells = int((source_length * target_length).sum().item()) if return_cells else 0
    return loss.reshape(B, len(PAIR_ORDER)).mean(dim=1), cells


@torch.no_grad()
def _batched_partial_viterbi_paths(
    score: torch.Tensor,
    source_length: torch.Tensor,
    target_length: torch.Tensor,
    *,
    allow_empty: bool = True,
    input_is_emission: bool = False,
) -> torch.Tensor:
    """GPU-packed, exact END max-product paths on true-length lattices.

    This is deliberately the same recurrence as :func:`viterbi_decode_one`.
    It is local to this module because SRC needs only the detached discrete
    support/endpoints, never a gradient through Viterbi or another decoder.
    """
    count, source_max, target_max = score.shape
    emission = score.float() if input_is_emission else score.float() - torch.log(target_length.float()).view(-1, 1, 1)
    valid_target = torch.arange(target_max, device=score.device)[None, None] < target_length[:, None, None]
    emission = emission.masked_fill(~valid_target, -torch.inf)
    segment = -torch.log((source_length * (source_length - 1) // 2).clamp_min(1).float()) if allow_empty else torch.zeros_like(source_length, dtype=torch.float32)
    negative = emission.new_full((count, target_max), -torch.inf)
    single, multi = negative, negative
    predecessor = torch.zeros((count, source_max, target_max), device=score.device, dtype=torch.int16)
    predecessor_multi = torch.zeros((count, source_max, target_max), device=score.device, dtype=torch.bool)
    best = emission.new_full((count,), -torch.inf)
    best_p = torch.full((count,), -1, device=score.device, dtype=torch.long)
    best_q = torch.zeros(count, device=score.device, dtype=torch.long)
    for source in range(source_max):
        active = source < source_length
        fresh = emission[:, source] + segment[:, None]
        if source:
            combined = torch.maximum(single, multi)
            prefix, prefix_q = torch.cummax(combined, dim=1)
            candidate = emission[:, source] + prefix
            is_multi = multi > single
            multi = torch.where(active[:, None], candidate, multi)
            predecessor[:, source] = prefix_q.to(torch.int16)
            predecessor_multi[:, source] = is_multi.gather(1, prefix_q)
            value, index = multi.max(dim=1)
            take = active & (value > best)
            best = torch.where(take, value, best)
            best_p = torch.where(take, torch.full_like(best_p, source), best_p)
            best_q = torch.where(take, index, best_q)
        single = torch.where(active[:, None], fresh, single)
    has_overlap = best > 0.0 if allow_empty else torch.isfinite(best)
    output = score.new_full((count, source_max), -1.0)
    current_p, current_q = best_p.clone(), best_q.clone()
    alive = has_overlap.clone()
    for source in range(source_max - 1, 0, -1):
        active = alive & (current_p == source)
        # Keep the full packed batch on GPU.  The old indexed implementation
        # used ``nonzero`` followed by ``len(cuda_tensor)`` at every source
        # row, which forces thousands of host-visible shape decisions in an
        # SRC update.  These masked tensor updates are exactly the same END
        # traceback and have no data-dependent Python branch.
        prior = predecessor[:, source].gather(1, current_q[:, None]).squeeze(1).long()
        from_multi = predecessor_multi[:, source].gather(1, current_q[:, None]).squeeze(1)
        output[:, source] = torch.where(active, current_q.to(output.dtype), output[:, source])
        output[:, source - 1] = torch.where(active, prior.to(output.dtype), output[:, source - 1])
        continuing = active & from_multi
        finished = active & ~from_multi
        current_p = torch.where(continuing, current_p - 1, current_p)
        current_q = torch.where(continuing, prior, current_q)
        alive = alive & ~finished
    return output


def _conditional_endpoint_occupancy(
    score: torch.Tensor,
    source_length: torch.Tensor,
    target_length: torch.Tensor,
    path: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact soft occupancy conditional on detached END Viterbi endpoints.

    The discrete Viterbi support/endpoints are intentionally treated as a
    fixed selection.  Within those endpoints, the probability is the exact
    forward--backward marginal over every legal END monotone path: one source
    row per step and non-decreasing target coordinate.  This preserves the
    original transition family while keeping support selection outside SRC's
    backward graph.  Empty END outputs return an all-zero occupancy.
    """
    count, source_max, target_max = score.shape
    active = path >= 0
    has_overlap = active.any(dim=1)
    first_source = active.to(torch.long).argmax(dim=1)
    last_source = source_max - 1 - active.flip(1).to(torch.long).argmax(dim=1)
    rows = torch.arange(count, device=score.device)
    first_target = path[rows, first_source].round().long().clamp_min(0)
    last_target = path[rows, last_source].round().long().clamp_min(0)
    first_target = torch.minimum(first_target, (target_length - 1).clamp_min(0))
    last_target = torch.minimum(last_target, (target_length - 1).clamp_min(0))
    emission = score.float() - torch.log(target_length.float()).view(-1, 1, 1)
    source_grid = torch.arange(source_max, device=score.device)[None, :, None]
    target_grid = torch.arange(target_max, device=score.device)[None, None, :]
    inside = (
        has_overlap[:, None, None]
        & (source_grid >= first_source[:, None, None])
        & (source_grid <= last_source[:, None, None])
        & (target_grid >= first_target[:, None, None])
        & (target_grid <= last_target[:, None, None])
        & (source_grid < source_length[:, None, None])
        & (target_grid < target_length[:, None, None])
    )
    emission = emission.masked_fill(~inside, -torch.inf)
    negative = emission.new_full((count, target_max), -torch.inf)
    forward = emission.new_full(emission.shape, -torch.inf)
    state = negative
    for source in range(source_max):
        at_start = source == first_source
        candidate = emission[:, source] + torch.logcumsumexp(state, dim=1)
        start_candidate = emission[:, source].masked_fill(target_grid[:, 0] != first_target[:, None], -torch.inf)
        candidate = torch.where(at_start[:, None], start_candidate, candidate)
        state = torch.where((has_overlap & (source >= first_source) & (source <= last_source))[:, None], candidate, state)
        forward[:, source] = torch.where((has_overlap & (source >= first_source) & (source <= last_source))[:, None], state, negative)
    logz = forward[rows, last_source, last_target]
    backward = emission.new_full(emission.shape, -torch.inf)
    state = negative
    for source in range(source_max - 1, -1, -1):
        at_end = source == last_source
        # beta excludes the current emission.  The reversed prefix sum is the
        # exact q' >= q monotone continuation transition.
        next_state = torch.logcumsumexp((emission[:, source + 1] + state).flip(1), dim=1).flip(1) if source + 1 < source_max else negative
        terminal = torch.zeros_like(next_state).masked_fill(target_grid[:, 0] != last_target[:, None], -torch.inf)
        candidate = torch.where(at_end[:, None], terminal, next_state)
        valid_source = has_overlap & (source >= first_source) & (source <= last_source)
        state = torch.where(valid_source[:, None], candidate, state)
        backward[:, source] = torch.where(valid_source[:, None], state, negative)
    occupancy = torch.exp(forward + backward - logz[:, None, None]).masked_fill(~inside, 0.0)
    occupancy = torch.where(has_overlap[:, None, None], occupancy, torch.zeros_like(occupancy))
    return occupancy, has_overlap, (last_source - first_source + 1).clamp_min(0)


def src_relation_from_occupancy(occupancy: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return Frobenius-normalized source self-relations ``P P^T``."""
    relation = torch.bmm(occupancy, occupancy.transpose(1, 2))
    return relation / relation.flatten(1).norm(dim=1).clamp_min(eps)[:, None, None]


def src_loss_from_source_occupancies(
    occupancies: torch.Tensor,
    has_overlap: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Three-partner SRC reduction for one source sequence.

    ``occupancies`` is ``[B,3,L_source,L_target]``.  A comparison is included
    only when both relations have an ordinary nonempty END support; this makes
    an explicit no-overlap output a zero occupancy rather than fabricated
    correspondence evidence.
    """
    batch = occupancies.shape[0]
    relation = src_relation_from_occupancy(occupancies.reshape(-1, *occupancies.shape[-2:]), eps).reshape(batch, 3, occupancies.shape[-2], occupancies.shape[-2])
    # All three partner comparisons have identical shapes.  Computing them
    # together avoids three small reduction-launch chains per source.
    comparison = torch.stack((
        relation[:, 0] - relation[:, 1],
        relation[:, 0] - relation[:, 2],
        relation[:, 1] - relation[:, 2],
    ), dim=1)
    usable = torch.stack((
        has_overlap[:, 0] & has_overlap[:, 1],
        has_overlap[:, 0] & has_overlap[:, 2],
        has_overlap[:, 1] & has_overlap[:, 2],
    ), dim=1)
    difference = comparison.square().sum(dim=(2, 3))
    values = (difference * usable.to(difference.dtype)).sum(dim=1)
    counts = usable.sum(dim=1).to(values.dtype)
    loss = values / counts.clamp_min(1.0)
    entropy = -(occupancies.clamp_min(eps) * occupancies.clamp_min(eps).log()).sum(dim=3)
    active_rows = occupancies.sum(dim=3) > 0
    entropy = (entropy * active_rows).sum(dim=(1, 2)) / active_rows.sum(dim=(1, 2)).clamp_min(1).to(entropy.dtype)
    return loss, {
        "src_comparison_count": counts,
        "occupancy_entropy": entropy,
        "occupancy_support_coverage": active_rows.sum(dim=(1, 2)).to(entropy.dtype) / occupancies.shape[1] / occupancies.shape[2],
        "relation_overlap_fraction": has_overlap.to(entropy.dtype).mean(dim=1),
    }


def src_self_relation_loss(
    score_lattices: torch.Tensor,
    lengths: torch.Tensor,
    *,
    eps: float = 1e-8,
    return_occupancies: bool = False,
    return_paths: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Differentiable group SRC from exact post-CRF endpoint occupancies.

    The score lattice is ``[B,12,L,L]`` in ``PAIR_ORDER``.  Hard support and
    endpoints are selected once with detached original END Viterbi.  All
    gradients thereafter pass only through exact soft occupancies inside those
    selected legal CRF paths; neither raw scores nor a Sinkhorn relaxation is
    used as a substitute.
    """
    if score_lattices.ndim != 4 or score_lattices.shape[1] != len(PAIR_ORDER):
        raise ValueError("score_lattices [B,12,L,L] required")
    batch, _pairs, width, _ = score_lattices.shape
    pair_index, _unordered, _route, _forward = _pair_tensors(score_lattices.device)
    # PAIR_ORDER is source-major: [0->1,0->2,0->3,1->0,...].  Pack every
    # directed pair into one Viterbi / forward-backward call.  This preserves
    # the exact loss while replacing four under-filled GPU kernels with one
    # substantially larger batched operation.
    pair_score = score_lattices.reshape(batch * len(PAIR_ORDER), width, width)
    pair_source_length = lengths[:, pair_index[:, 0]].reshape(-1)
    pair_target_length = lengths[:, pair_index[:, 1]].reshape(-1)
    path = _batched_partial_viterbi_paths(pair_score.detach(), pair_source_length, pair_target_length)
    occupancy, has_overlap, _ = _conditional_endpoint_occupancy(pair_score, pair_source_length, pair_target_length, path)
    occupancy = occupancy.reshape(batch, 4, 3, width, width)
    has_overlap = has_overlap.reshape(batch, 4, 3)
    relation = src_relation_from_occupancy(occupancy.reshape(-1, width, width), eps).reshape(batch, 4, 3, width, width)
    comparison = torch.stack((
        relation[:, :, 0] - relation[:, :, 1],
        relation[:, :, 0] - relation[:, :, 2],
        relation[:, :, 1] - relation[:, :, 2],
    ), dim=2)
    usable = torch.stack((
        has_overlap[:, :, 0] & has_overlap[:, :, 1],
        has_overlap[:, :, 0] & has_overlap[:, :, 2],
        has_overlap[:, :, 1] & has_overlap[:, :, 2],
    ), dim=2)
    source_values = (comparison.square().sum(dim=(-1, -2)) * usable.to(relation.dtype)).sum(dim=2)
    source_counts = usable.sum(dim=2).to(relation.dtype)
    per_group = (source_values / source_counts.clamp_min(1.0)).mean(dim=1)
    row_entropy = -(occupancy.clamp_min(eps) * occupancy.clamp_min(eps).log()).sum(dim=-1)
    active_rows = occupancy.sum(dim=-1) > 0
    entropy_per_source = (row_entropy * active_rows).sum(dim=(2, 3)) / active_rows.sum(dim=(2, 3)).clamp_min(1).to(row_entropy.dtype)
    coverage_per_source = active_rows.sum(dim=(2, 3)).to(row_entropy.dtype) / float(3 * width)
    overlap_per_source = has_overlap.to(row_entropy.dtype).mean(dim=2)
    diagnostics: dict[str, torch.Tensor] = {
        "src_loss": per_group.mean(),
        "src_occupancy_entropy": entropy_per_source.mean(),
        "src_support_coverage": coverage_per_source.mean(),
        "src_relation_overlap_fraction": overlap_per_source.mean(),
    }
    if return_occupancies:
        # Source-major: entries source i contain its three directed partners
        # in their frozen PAIR_ORDER order.  Retained only by frozen audits.
        diagnostics["source_occupancies"] = [occupancy[:, source] for source in range(4)]
    if return_paths:
        diagnostics["hard_paths"] = path.reshape(batch, len(PAIR_ORDER), width)
    return per_group, diagnostics


def src_loss_from_retained_occupancies(source_occupancies: list[torch.Tensor], eps: float = 1e-8) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Evaluate SRC from source-major occupancy tensors used by frozen audits."""
    if len(source_occupancies) != 4:
        raise ValueError("four source-major occupancy tensors are required")
    values: list[torch.Tensor] = []
    entropy: list[torch.Tensor] = []
    coverage: list[torch.Tensor] = []
    overlap: list[torch.Tensor] = []
    for occupancy in source_occupancies:
        if occupancy.ndim != 4 or occupancy.shape[1] != 3:
            raise ValueError("each source occupancy must be [B,3,L,L]")
        has_overlap = occupancy.sum(dim=(2, 3)) > 0
        value, diagnostic = src_loss_from_source_occupancies(occupancy, has_overlap, eps=eps)
        values.append(value)
        entropy.append(diagnostic["occupancy_entropy"])
        coverage.append(diagnostic["occupancy_support_coverage"])
        overlap.append(diagnostic["relation_overlap_fraction"])
    per_group = torch.stack(values, dim=1).mean(dim=1)
    return per_group, {
        "src_loss": per_group.mean(),
        "src_occupancy_entropy": torch.stack(entropy, dim=1).mean(),
        "src_support_coverage": torch.stack(coverage, dim=1).mean(),
        "src_relation_overlap_fraction": torch.stack(overlap, dim=1).mean(),
    }


@torch.no_grad()
def hard_src_self_relation_loss(
    score_lattices: torch.Tensor,
    lengths: torch.Tensor,
    eps: float = 1e-8,
    paths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """SRC diagnostic from the ordinary hard END paths, with no soft proxy."""
    if score_lattices.ndim != 4 or score_lattices.shape[1] != len(PAIR_ORDER):
        raise ValueError("score_lattices [B,12,L,L] required")
    batch, _pairs, width, target_width = score_lattices.shape
    if paths is None:
        pair_index, _unordered, _route, _forward = _pair_tensors(score_lattices.device)
        source_lengths = lengths[:, pair_index[:, 0]].reshape(-1)
        target_lengths = lengths[:, pair_index[:, 1]].reshape(-1)
        paths = _batched_partial_viterbi_paths(score_lattices.detach().reshape(-1, width, target_width), source_lengths, target_lengths).reshape(batch, len(PAIR_ORDER), width)
    if paths.shape != (batch, len(PAIR_ORDER), width):
        raise ValueError("paths must be [B,12,L] matching score_lattices")
    source_major = paths.reshape(batch, 4, 3, width)
    occupancy = score_lattices.new_zeros((batch, 4, 3, width, target_width))
    valid = source_major >= 0
    occupancy.scatter_(4, source_major.clamp_min(0).long()[..., None], valid[..., None].to(occupancy.dtype))
    return src_loss_from_retained_occupancies([occupancy[:, source] for source in range(4)], eps=eps)


def viterbi_decode_one(score: torch.Tensor, *, allow_empty: bool = True, input_is_emission: bool = False) -> dict[str, np.ndarray | int | bool | float]:
    """Frozen END max-product recurrence on one true-size lattice.

    ``maximum``/prefix-max are deliberate: ``logaddexp`` belongs only to the
    partition DP. Public bounds are inclusive.
    """
    P, Q = score.shape
    if P < 2 or Q < 1:
        raise ValueError("Viterbi requires P>=2 and Q>=1")
    emission = (score if input_is_emission else score - math.log(Q)).detach().float().cpu().numpy()
    previous_single = np.full(Q, -np.inf)
    previous_multi = np.full(Q, -np.inf)
    predecessor_q = np.full((P, Q), -1, np.int64)
    predecessor_multi = np.zeros((P, Q), bool)
    best_score, best_p, best_q = -np.inf, -1, -1
    for p in range(P):
        single = emission[p] + (segment_bias(P) if allow_empty else 0.0)
        if p:
            from_multi = previous_multi > previous_single
            combined = np.maximum(previous_single, previous_multi)
            prefix_q = np.empty(Q, np.int64); best, arg = -np.inf, 0
            for q in range(Q):
                if combined[q] > best:
                    best, arg = combined[q], q
                prefix_q[q] = arg
            multi = emission[p] + combined[prefix_q]
            predecessor_q[p], predecessor_multi[p] = prefix_q, from_multi[prefix_q]
            q = int(multi.argmax())
            if multi[q] > best_score:
                best_score, best_p, best_q = float(multi[q]), p, q
        else:
            multi = previous_multi
        previous_single, previous_multi = single, multi
    if allow_empty and not best_score > 0.0:
        return {"score": 0.0, "has_overlap": False, "bound_source": (-1, -1), "bound_target": (-1, -1), "a": -1, "b": -1, "q": np.full(P, -1, np.int64)}
    if not np.isfinite(best_score):
        raise RuntimeError("Always-Overlap Viterbi found no legal non-empty partial path")
    path = np.full(P, -1, np.int64); p, q = best_p, best_q; path[p] = q
    while True:
        prior_q = int(predecessor_q[p, q]); path[p - 1] = prior_q
        if not predecessor_multi[p, q]:
            return {"score": best_score, "has_overlap": True, "bound_source": (p - 1, best_p), "bound_target": (prior_q, best_q), "a": p - 1, "b": best_p, "q": path}
        p, q = p - 1, prior_q
