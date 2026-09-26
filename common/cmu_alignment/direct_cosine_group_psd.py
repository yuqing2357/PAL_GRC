"""Directed-relation Group-PSD for CMU's variable-length direct-cosine CRF.

This is the CMU adaptation of the validated D34 Group-PSD implementation.
It operates on model-only Always-Overlap CRF occupancies at sampled *native
MFCC frame positions*.  In particular, it does not fuse reciprocal routes,
pool into bins, degree-normalize pair blocks, or consume ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from cmu_alignment.data import PAIR_ORDER
from cmu_alignment.soft_route_psd import full_partial_path_occupancy


@dataclass
class DirectCosineGroupPSDResult:
    """Per-group loss and detached diagnostic aggregates."""

    per_group_loss: torch.Tensor
    mean_directional_energy: torch.Tensor
    mean_negative_spectral_energy: torch.Tensor
    mean_min_eigenvalue: torch.Tensor
    mean_negative_eigenvalue_count: torch.Tensor
    positions_used: float
    mean_offdiagonal_relation_mass: torch.Tensor
    mean_total_relation_mass: torch.Tensor


def _native_positions(length: int, count: int, device: torch.device) -> torch.Tensor:
    """Evenly select existing native-frame indices from one sequence."""
    if not 1 <= count <= length:
        raise ValueError(f"position count must lie in [1, {length}], got {count}")
    if count == 1:
        return torch.zeros(1, dtype=torch.long, device=device)
    return torch.linspace(0, length - 1, steps=count, device=device).round().long()


def _group_positions(
    lengths: torch.Tensor, *, num_positions: int, position_fraction: float | None,
) -> tuple[torch.Tensor, ...]:
    """Return one native-index set per curve, preserving its own length.

    With ``position_fraction`` each curve i contributes round(p * L_i) real
    frame indices.  The fixed-count branch remains for legacy CMU runs.
    """
    native_lengths = [int(value) for value in lengths.detach().cpu().tolist()]
    if position_fraction is None:
        count = min(num_positions, min(native_lengths))
        counts = [count] * len(native_lengths)
    else:
        counts = [max(2, min(length, int(round(position_fraction * length)))) for length in native_lengths]
    return tuple(_native_positions(length, count, lengths.device) for length, count in zip(native_lengths, counts, strict=True))


def _assemble_directed_relation(occupancy: torch.Tensor, positions: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Build a block-ragged relation R from the 12 directed occupancy blocks."""
    if occupancy.ndim != 3 or occupancy.shape[0] != len(PAIR_ORDER) or len(positions) != 4:
        raise ValueError("expected occupancy [12,L,L] and four native-position sets")
    sizes = [len(item) for item in positions]
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    relation = occupancy.new_zeros((offsets[-1], offsets[-1]))
    for curve, size in enumerate(sizes):
        relation[offsets[curve]:offsets[curve + 1], offsets[curve]:offsets[curve + 1]] = torch.eye(size, device=occupancy.device, dtype=occupancy.dtype)
    for pair, (source, target) in enumerate(PAIR_ORDER):
        block = occupancy[pair].index_select(0, positions[source]).index_select(1, positions[target])
        relation[offsets[source]:offsets[source + 1], offsets[target]:offsets[target + 1]] = block
    return relation


def direct_cosine_group_psd_loss(
    group_emission: torch.Tensor,
    lengths: torch.Tensor,
    *,
    num_positions: int = 128,
    position_fraction: float | None = None,
    objective_mode: str = "full",
) -> DirectCosineGroupPSDResult:
    """Penalize directed inconsistency and non-PSD symmetric structure.

    CMU batches are padded to a local maximum length.  When
    ``position_fraction`` is set, curve i contributes round(p * L_i) actual
    native-frame indices, yielding rectangular m_i-by-m_j pair blocks.  No
    acoustic signal, ground truth, or native index is resampled or renumbered.
    """
    objective_mode = str(objective_mode).lower()
    if objective_mode not in {"direction", "spectral", "full"}:
        raise ValueError("objective_mode must be direction, spectral, or full")
    if group_emission.ndim != 4 or group_emission.shape[1] != len(PAIR_ORDER):
        raise ValueError("group_emission must be [B,12,L,L]")
    batch, _pairs, width, height = group_emission.shape
    if width != height or lengths.shape != (batch, 4):
        raise ValueError("CMU Group-PSD requires square padded lattices and lengths [B,4]")
    if num_positions < 1 or (position_fraction is not None and not 0.0 < position_fraction <= 1.0):
        raise ValueError("invalid Group-PSD position specification")
    if int(lengths.detach().min().cpu()) < 2:
        raise ValueError("CMU Group-PSD requires at least two native frames per curve")

    source_ids = torch.as_tensor([source for source, _target in PAIR_ORDER], device=group_emission.device)
    target_ids = torch.as_tensor([target for _source, target in PAIR_ORDER], device=group_emission.device)
    source_length = lengths[:, source_ids].reshape(-1)
    target_length = lengths[:, target_ids].reshape(-1)
    occupancy, _ = full_partial_path_occupancy(
        group_emission.reshape(batch * len(PAIR_ORDER), width, width), source_length, target_length
    )
    occupancy = occupancy.reshape(batch, len(PAIR_ORDER), width, width)
    losses, directional_terms, spectral_terms, minimums, negative_counts, mean_counts, offdiag_masses, total_masses = [], [], [], [], [], [], [], []
    for index in range(batch):
        positions = _group_positions(lengths[index], num_positions=num_positions, position_fraction=position_fraction)
        relation = _assemble_directed_relation(occupancy[index], positions).float()
        symmetric = 0.5 * (relation + relation.transpose(-1, -2))
        antisymmetric = 0.5 * (relation - relation.transpose(-1, -2))
        with torch.no_grad():
            values, vectors = torch.linalg.eigh(symmetric)
            target = (vectors * values.clamp_min(0.0).unsqueeze(-2)) @ vectors.transpose(-1, -2)
        size_squared = float(symmetric.shape[-1] ** 2)
        directional = antisymmetric.square().sum() / size_squared
        spectral = (symmetric - target).square().sum() / size_squared
        # Full retains the formal objective exactly.  The other modes only
        # expose its already-existing components for E3.
        losses.append(directional if objective_mode == "direction" else spectral if objective_mode == "spectral" else directional + spectral)
        directional_terms.append(directional)
        spectral_terms.append(spectral)
        minimums.append(values.min())
        negative_counts.append((values < -1.0e-6).sum().float())
        mean_counts.append(sum(len(item) for item in positions) / 4.0)
        mask = torch.eye(relation.shape[-1], dtype=torch.bool, device=relation.device)
        offdiag_masses.append(relation.masked_fill(mask, 0.0).abs().sum())
        total_masses.append(relation.abs().sum())
    loss = torch.stack(losses)
    directional_energy = torch.stack(directional_terms)
    negative_spectral_energy = torch.stack(spectral_terms)
    values_minimum = torch.stack(minimums)
    negative_eigenvalue_count = torch.stack(negative_counts)
    return DirectCosineGroupPSDResult(
        per_group_loss=loss.to(group_emission.dtype),
        mean_directional_energy=directional_energy.mean().detach().to(group_emission.dtype),
        mean_negative_spectral_energy=negative_spectral_energy.mean().detach().to(group_emission.dtype),
        mean_min_eigenvalue=values_minimum.mean().detach().to(group_emission.dtype),
        mean_negative_eigenvalue_count=negative_eigenvalue_count.mean().detach().to(group_emission.dtype),
        positions_used=float(sum(mean_counts) / len(mean_counts)),
        mean_offdiagonal_relation_mass=torch.stack(offdiag_masses).mean().detach().to(group_emission.dtype),
        mean_total_relation_mass=torch.stack(total_masses).mean().detach().to(group_emission.dtype),
    )
