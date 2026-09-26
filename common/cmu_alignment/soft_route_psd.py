"""Prediction-only soft-route PSD regularization.

This auxiliary branch never changes the production CRF loss or Viterbi
decoder.  It differentiates the Always-Overlap partial-path partition for
every directed pair, canonicalizes the model's own soft route mass, and
penalizes only negative spectrum of the K=4 group relation.  No function here
accepts GT paths, endpoints, supports, or a GT spectral tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from cmu_alignment.data import PAIR_ORDER


_PAIR_INDEX = {pair: index for index, pair in enumerate(PAIR_ORDER)}


@dataclass
class SoftRoutePSDResult:
    per_group_loss: torch.Tensor
    mean_min_eigenvalue: torch.Tensor
    mean_negative_eigenvalue_count: torch.Tensor
    mean_negative_spectral_mass: torch.Tensor
    mean_directional_disagreement: torch.Tensor
    mean_posterior_entropy: torch.Tensor
    mean_max_posterior: torch.Tensor
    mean_expected_path_length: torch.Tensor
    mean_predicted_overlap_fraction: torch.Tensor


def _masked_emission(emission: torch.Tensor, source_length: torch.Tensor, target_length: torch.Tensor) -> torch.Tensor:
    """Mask padded target cells without changing any legal CRF score.

    A finite sentinel is intentional.  Literal ``-inf`` is forward-correct
    but creates undefined second derivatives in PyTorch's logcumsumexp when a
    padded row is differentiated through the occupancy.  ``-1e4`` underflows
    to zero probability in float32/bfloat16 at CMU score scales while keeping
    the posterior Hessian finite.
    """
    source = torch.arange(emission.shape[-2], device=emission.device)[None, :, None]
    target = torch.arange(emission.shape[-1], device=emission.device)[None, None, :]
    invalid = (source >= source_length[:, None, None]) | (target >= target_length[:, None, None])
    return emission.float().masked_fill(invalid, -1.0e4)


def full_partial_path_log_partition(emission: torch.Tensor, source_length: torch.Tensor) -> torch.Tensor:
    """Exact Always-Overlap partial-path denominator for additive emissions.

    The family is the established ``allow_empty=False, segment=0`` CRF:
    contiguous source segments of at least two rows, with non-decreasing target
    coordinates.  No target-length correction is applied in this helper;
    ``emission`` is already the CRF additive score.
    """
    if emission.ndim != 3 or source_length.shape != (emission.shape[0],):
        raise ValueError("emission [B,P,Q] and source_length [B] are required")
    batch, source_max, target_max = emission.shape
    # A finite impossible-state sentinel avoids NaN second derivatives from
    # logaddexp/logcumsumexp at the initial empty ``multi`` state.  Its mass
    # is exactly underflowed at the score ranges used here.
    negative = emission.new_full((batch, target_max), -1.0e9)
    single, multi = negative, negative
    endpoint = emission.new_full((batch,), -1.0e9)
    for source in range(source_max):
        active = source < source_length
        fresh = emission[:, source]
        if source:
            candidate = emission[:, source] + torch.logcumsumexp(torch.logaddexp(single, multi), dim=1)
            multi = torch.where(active[:, None], candidate, multi)
            endpoint = torch.where(active, torch.logaddexp(endpoint, torch.logsumexp(candidate, dim=1)), endpoint)
        single = torch.where(active[:, None], fresh, single)
    if emission.device.type == "cpu" and not torch.isfinite(endpoint).all():
        raise RuntimeError("Always-Overlap partial-path partition contains no legal path")
    return endpoint


def full_partial_path_occupancy(
    emission: torch.Tensor,
    source_length: torch.Tensor,
    target_length: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Model-only structured soft occupancy over the complete CRF path space.

    This explicit forward--backward form is equivalent to ``d log Z / d E``
    for the Always-Overlap denominator.  Keeping it as ordinary differentiable
    tensor operations avoids the NaN-prone nested-autograd second derivative
    through padded ragged states.
    """
    if emission.ndim != 3 or source_length.shape != (emission.shape[0],) or target_length.shape != source_length.shape:
        raise ValueError("emission [B,P,Q], source_length [B], and target_length [B] are required")
    value = _masked_emission(emission, source_length, target_length)
    batch, source_max, target_max = value.shape
    negative = value.new_full((batch, target_max), -1.0e4)
    single, multi = negative, negative
    single_states, multi_states = [], []
    for source in range(source_max):
        active = source < source_length
        fresh = value[:, source]
        candidate = negative if source == 0 else value[:, source] + torch.logcumsumexp(torch.logaddexp(single, multi), dim=1)
        single = torch.where(active[:, None], fresh, negative)
        multi = torch.where(active[:, None], candidate, negative)
        single_states.append(single)
        multi_states.append(multi)
    single_table = torch.stack(single_states, dim=1)
    multi_table = torch.stack(multi_states, dim=1)
    logz = torch.logsumexp(multi_table.reshape(batch, -1), dim=1)

    suffix_multi = negative
    suffix_single_states, suffix_multi_states = [], []
    zero = value.new_zeros((batch, target_max))
    for source in range(source_max - 1, -1, -1):
        active = source < source_length
        if source + 1 < source_max:
            continuation = torch.logcumsumexp((value[:, source + 1] + suffix_multi).flip(1), dim=1).flip(1)
            continuation = torch.where((source + 1 < source_length)[:, None], continuation, negative)
        else:
            continuation = negative
        current_single_suffix = torch.where(active[:, None], continuation, negative)
        current_multi_suffix = torch.where(active[:, None], torch.logaddexp(zero, continuation), negative)
        suffix_single_states.append(current_single_suffix)
        suffix_multi_states.append(current_multi_suffix)
        suffix_multi = current_multi_suffix
    suffix_single_table = torch.stack(suffix_single_states[::-1], dim=1)
    suffix_multi_table = torch.stack(suffix_multi_states[::-1], dim=1)
    occupancy = torch.exp(single_table + suffix_single_table - logz[:, None, None]) + torch.exp(multi_table + suffix_multi_table - logz[:, None, None])
    source = torch.arange(emission.shape[-2], device=emission.device)[None, :, None]
    target = torch.arange(emission.shape[-1], device=emission.device)[None, None, :]
    legal = (source < source_length[:, None, None]) & (target < target_length[:, None, None])
    # Padded states are not legal CRF cells and receive an exact zero route
    # mass.  The other branch is finite, so this mask is gradient-safe.
    return torch.where(legal, occupancy, torch.zeros_like(occupancy)), logz


def normalize_directional_route_mass(occupancy: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Canonicalize a directed soft route to unit total posterior mass."""
    if occupancy.ndim < 2:
        raise ValueError("occupancy requires two route dimensions")
    return occupancy / occupancy.sum(dim=(-2, -1), keepdim=True).clamp_min(eps)


def degree_normalize_relation(mass: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Symmetric degree normalization of a nonnegative rectangular coupling."""
    left, right = mass.sum(dim=-1), mass.sum(dim=-2)
    left_inv = torch.where(left > eps, left.rsqrt(), torch.zeros_like(left))
    right_inv = torch.where(right > eps, right.rsqrt(), torch.zeros_like(right))
    return mass * left_inv[..., :, None] * right_inv[..., None, :]


def _bin_ids(length: int, count: int, device: torch.device) -> torch.Tensor:
    return torch.div(torch.arange(length, device=device) * count, length, rounding_mode="floor")


def pool_route_mass_to_bins(mass: torch.Tensor, source_length: int, target_length: int, *, num_bins: int) -> torch.Tensor:
    """Sum-pool one native-resolution coupling into shared sequence bins."""
    source_bins, target_bins = min(num_bins, source_length), min(num_bins, target_length)
    if source_bins < 1 or target_bins < 1:
        raise ValueError("sequence length must be positive")
    left, right = _bin_ids(source_length, source_bins, mass.device), _bin_ids(target_length, target_bins, mass.device)
    flat = left[:, None] * target_bins + right[None, :]
    output = mass.new_zeros((source_bins * target_bins,))
    output.scatter_add_(0, flat.reshape(-1), mass[:source_length, :target_length].reshape(-1))
    return output.reshape(source_bins, target_bins)


def assemble_group_relation_matrix(directed_mass: torch.Tensor, lengths: torch.Tensor, *, num_bins: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse directions, pool, degree-normalize, and assemble one K=4 matrix."""
    if directed_mass.shape[0] != len(PAIR_ORDER):
        raise ValueError("directed_mass must contain all directed K=4 pairs")
    lengths_list = [int(item) for item in lengths.detach().cpu().tolist()]
    bin_counts = [min(num_bins, length) for length in lengths_list]
    offsets = [0]
    for value in bin_counts:
        offsets.append(offsets[-1] + value)
    matrix = directed_mass.new_zeros((offsets[-1], offsets[-1]))
    for sequence, bins in enumerate(bin_counts):
        matrix[offsets[sequence]:offsets[sequence + 1], offsets[sequence]:offsets[sequence + 1]].fill_diagonal_(1.0)
    disagreement = directed_mass.new_zeros(())
    for left in range(4):
        for right in range(left + 1, 4):
            forward = directed_mass[_PAIR_INDEX[(left, right)]]
            reverse = directed_mass[_PAIR_INDEX[(right, left)]].transpose(0, 1)
            disagreement = disagreement + (forward - reverse).square().sum().sqrt() / ((forward.square().sum().sqrt() + reverse.square().sum().sqrt()) * .5).clamp_min(1e-12)
            merged = .5 * (forward + reverse)
            block = degree_normalize_relation(pool_route_mass_to_bins(merged, lengths_list[left], lengths_list[right], num_bins=num_bins))
            left_slice, right_slice = slice(offsets[left], offsets[left + 1]), slice(offsets[right], offsets[right + 1])
            matrix[left_slice, right_slice] = block
            matrix[right_slice, left_slice] = block.transpose(0, 1)
    return matrix, disagreement / 6


def _assemble_fixed_r32_batch(directed_mass: torch.Tensor, lengths: torch.Tensor, *, num_bins: int) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU-packed equivalent of :func:`assemble_group_relation_matrix`."""
    batch, _pairs, width, _ = directed_mass.shape
    device, dtype = directed_mass.device, directed_mass.dtype
    positions = torch.arange(width, device=device)
    bases = (positions[None, None, :] * num_bins // lengths.clamp_min(1)[..., None]).clamp_max(num_bins - 1)
    output = directed_mass.new_zeros((batch, 4 * num_bins, 4 * num_bins))
    eye = torch.eye(num_bins, device=device, dtype=dtype)
    for sequence in range(4):
        output[:, sequence * num_bins:(sequence + 1) * num_bins, sequence * num_bins:(sequence + 1) * num_bins] = eye
    disagreement = directed_mass.new_zeros((batch,))
    for left in range(4):
        for right in range(left + 1, 4):
            forward = directed_mass[:, _PAIR_INDEX[(left, right)]]
            reverse = directed_mass[:, _PAIR_INDEX[(right, left)]].transpose(-1, -2)
            disagreement = disagreement + (forward - reverse).square().sum(dim=(-2, -1)).sqrt() / ((forward.square().sum(dim=(-2, -1)).sqrt() + reverse.square().sum(dim=(-2, -1)).sqrt()) * .5).clamp_min(1e-12)
            merged = .5 * (forward + reverse)
            flat_index = bases[:, left, :, None] * num_bins + bases[:, right, None, :]
            pooled = merged.new_zeros((batch, num_bins * num_bins))
            pooled.scatter_add_(1, flat_index.reshape(batch, -1), merged.reshape(batch, -1))
            normalized = degree_normalize_relation(pooled.reshape(batch, num_bins, num_bins))
            output[:, left * num_bins:(left + 1) * num_bins, right * num_bins:(right + 1) * num_bins] = normalized
            output[:, right * num_bins:(right + 1) * num_bins, left * num_bins:(left + 1) * num_bins] = normalized.transpose(-1, -2)
    return output, disagreement / 6


def psd_projection(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Float32 projection onto the PSD cone for the detached loss target."""
    symmetric = .5 * (matrix.float() + matrix.float().transpose(-1, -2))
    values, vectors = torch.linalg.eigh(symmetric)
    return (vectors * values.clamp_min(0.0).unsqueeze(-2)) @ vectors.transpose(-1, -2), values


def psd_distance_loss(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Squared distance from the predicted group matrix to the PSD cone."""
    symmetric = .5 * (matrix.float() + matrix.float().transpose(-1, -2))
    with torch.no_grad():
        target, values = psd_projection(symmetric)
    loss = (symmetric - target).square().sum(dim=(-2, -1)) / symmetric.shape[-1]
    negative = values.clamp_max(0.0)
    return loss.to(matrix.dtype), values, target.to(matrix.dtype), negative.square().sum(dim=-1).to(matrix.dtype)


def _posterior_statistics(occupancy: torch.Tensor, source_length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    row_mass = occupancy.sum(dim=-1)
    active = row_mass > 0
    row_probability = torch.where(active[..., None], occupancy / row_mass[..., None].clamp_min(1e-12), torch.zeros_like(occupancy))
    row_entropy = -(row_probability.clamp_min(1e-12) * row_probability.clamp_min(1e-12).log()).sum(dim=-1)
    entropy = (row_entropy * active).sum() / active.sum().clamp_min(1)
    maximum = row_probability.max(dim=-1).values.masked_select(active).mean()
    path_length = occupancy.sum(dim=(-2, -1))
    return entropy, maximum, path_length.mean(), (path_length / source_length.to(occupancy.dtype).clamp_min(1)).mean()


def soft_route_psd_group_loss(group_emission: torch.Tensor, lengths: torch.Tensor, *, num_bins: int = 32) -> SoftRoutePSDResult:
    """Prediction-only Soft Correspondence → Pair Relation → PSD Group Loss."""
    if group_emission.ndim != 4 or group_emission.shape[1] != len(PAIR_ORDER):
        raise ValueError("group_emission must be [B,12,L,L]")
    batch, _pairs, width, _ = group_emission.shape
    if lengths.shape != (batch, 4):
        raise ValueError("lengths must be [B,4]")
    source_ids = torch.as_tensor([pair[0] for pair in PAIR_ORDER], device=group_emission.device)
    target_ids = torch.as_tensor([pair[1] for pair in PAIR_ORDER], device=group_emission.device)
    source_length = lengths[:, source_ids].reshape(-1)
    target_length = lengths[:, target_ids].reshape(-1)
    occupancy, _logz = full_partial_path_occupancy(group_emission.reshape(batch * len(PAIR_ORDER), width, width), source_length, target_length)
    directed_mass = normalize_directional_route_mass(occupancy).reshape(batch, len(PAIR_ORDER), width, width)
    entropy, maximum, expected_length, overlap_fraction = _posterior_statistics(occupancy, source_length)
    # The formal CMU data all have L>=32.  Preserve exact variable-block
    # semantics for short sequences in tests and future datasets.
    fixed_bins = not bool((lengths < num_bins).any().detach().cpu())
    if fixed_bins:
        predicted, disagreement = _assemble_fixed_r32_batch(directed_mass, lengths, num_bins=num_bins)
        loss, values, _target, negative_mass = psd_distance_loss(predicted)
        minimum = values.min(dim=-1).values.to(loss.dtype)
        negative_counts = (values < -1e-6).sum(dim=-1).to(loss.dtype)
    else:
        rows = [assemble_group_relation_matrix(directed_mass[index], lengths[index], num_bins=num_bins) for index in range(batch)]
        evaluated = [psd_distance_loss(matrix) for matrix, _disagreement in rows]
        loss = torch.stack([item[0] for item in evaluated])
        minimum = torch.stack([item[1].min().to(loss.dtype) for item in evaluated])
        negative_counts = torch.stack([(item[1] < -1e-6).sum().to(loss.dtype) for item in evaluated])
        negative_mass = torch.stack([item[3] for item in evaluated])
        disagreement = torch.stack([item[1] for item in rows])
    return SoftRoutePSDResult(
        per_group_loss=loss,
        mean_min_eigenvalue=minimum.mean(),
        mean_negative_eigenvalue_count=negative_counts.mean(),
        mean_negative_spectral_mass=negative_mass.mean(),
        mean_directional_disagreement=disagreement.mean(),
        mean_posterior_entropy=entropy,
        mean_max_posterior=maximum,
        mean_expected_path_length=expected_length,
        mean_predicted_overlap_fraction=overlap_fraction,
    )


@torch.no_grad()
def soft_route_psd_from_route_mass(predicted_mass: torch.Tensor, lengths: torch.Tensor, *, num_bins: int = 32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prediction-only diagnostic PSD value for supplied normalized routes."""
    predicted, disagreement = assemble_group_relation_matrix(predicted_mass, lengths, num_bins=num_bins)
    loss, values, _target, negative_mass = psd_distance_loss(predicted)
    return loss, values, disagreement, negative_mass
