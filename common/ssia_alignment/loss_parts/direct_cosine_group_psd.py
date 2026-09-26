"""Prediction-only directed-relation PSD regularizer for slot-free K=8 matching.

The regularizer operates directly on structured CRF occupancies at sampled
*original* curve positions. It deliberately keeps the two directed CRF
predictions independent: reciprocal agreement and global PSD consistency are
the quantities being learned, rather than preprocessing assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DirectCosineGroupPSDResult:
    """Per-group unified loss and detached diagnostics."""

    per_group_loss: torch.Tensor
    mean_directional_energy: torch.Tensor
    mean_negative_spectral_energy: torch.Tensor
    mean_min_eigenvalue: torch.Tensor
    mean_negative_eigenvalue_count: torch.Tensor
    # These are diagnostics of the *unprojected* directed relation.  They are
    # deliberately detached: they must never become an additional objective.
    mean_offdiagonal_relation_mass: torch.Tensor
    mean_total_relation_mass: torch.Tensor


def _pairs(curves: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.arange(curves, device=device)
    source, target = ids[:, None].expand(curves, curves), ids[None, :].expand(curves, curves)
    keep = source != target
    return source[keep], target[keep]


def _occupancy(emission: torch.Tensor) -> torch.Tensor:
    """Differentiable Always-Overlap CRF occupancy on fixed full lattices."""
    batch, source_length, target_length = emission.shape
    negative = emission.new_full((batch, target_length), -1.0e4)
    single, multi = negative, negative
    singles, multis = [], []
    for source in range(source_length):
        candidate = negative if source == 0 else emission[:, source] + torch.logcumsumexp(torch.logaddexp(single, multi), dim=1)
        single, multi = emission[:, source], candidate
        singles.append(single)
        multis.append(multi)
    single_table, multi_table = torch.stack(singles, dim=1), torch.stack(multis, dim=1)
    logz = torch.logsumexp(multi_table.reshape(batch, -1), dim=1)
    suffix_multi = negative
    suffix_single, suffix_many = [], []
    zero = emission.new_zeros((batch, target_length))
    for source in range(source_length - 1, -1, -1):
        continuation = negative if source + 1 == source_length else torch.logcumsumexp((emission[:, source + 1] + suffix_multi).flip(1), dim=1).flip(1)
        suffix_single.append(continuation)
        suffix_multi = torch.logaddexp(zero, continuation)
        suffix_many.append(suffix_multi)
    suffix_single_table = torch.stack(suffix_single[::-1], dim=1)
    suffix_multi_table = torch.stack(suffix_many[::-1], dim=1)
    return torch.exp(single_table + suffix_single_table - logz[:, None, None]) + torch.exp(multi_table + suffix_multi_table - logz[:, None, None])


def _sampled_positions(length: int, count: int, device: torch.device) -> torch.Tensor:
    """Deterministically choose evenly spaced, existing curve indices.

    These are samples of actual positions, never pooled bins or latent/group
    nodes. The same indices for every curve make the assembled relation a
    principal relation over real curve positions.
    """
    if not 1 <= count <= length:
        raise ValueError(f"group PSD positions must be in [1, {length}], got {count}")
    if count == 1:
        return torch.zeros(1, device=device, dtype=torch.long)
    return torch.linspace(0, length - 1, steps=count, device=device).round().to(torch.long)


def _directed_relation(
    occupancy: torch.Tensor, *, curves: int, positions: torch.Tensor,
    selected_curves: torch.Tensor | None = None,
    pair_source: torch.Tensor | None = None,
    pair_target: torch.Tensor | None = None,
) -> torch.Tensor:
    """Assemble a true ``[B,k*m,k*m]`` directed sub-relation.

    The model and PAL always predict/supervise the complete K=8 lattice.  For
    a proper subset, however, ``occupancy`` may contain only the selected
    directed pairs: each retained pair is still evaluated on its *full*
    512-by-512 CRF lattice before positions are extracted.  ``pair_source``
    and ``pair_target`` describe that compact pair layout.
    """
    batch, pair_count, _length, _width = occupancy.shape
    if (pair_source is None) != (pair_target is None):
        raise ValueError("pair_source and pair_target must be provided together")
    if pair_source is None:
        source, target = _pairs(curves, occupancy.device)
    else:
        source = pair_source.to(device=occupancy.device, dtype=torch.long)
        target = pair_target.to(device=occupancy.device, dtype=torch.long)
    if pair_count != len(source) or source.ndim != 1 or target.shape != source.shape:
        raise ValueError("occupancy pair dimension does not match its directed-pair layout")
    if selected_curves is None:
        selected_curves = torch.arange(curves, device=occupancy.device)
    selected_curves = selected_curves.to(device=occupancy.device, dtype=torch.long)
    if selected_curves.ndim != 1 or not 2 <= len(selected_curves) <= curves:
        raise ValueError("selected_curves must contain 2..curves distinct members")
    if len(torch.unique(selected_curves)) != len(selected_curves) or int(selected_curves.min()) < 0 or int(selected_curves.max()) >= curves:
        raise ValueError("selected_curves is not a valid subset of original curve identities")
    positions_count, selected_count = len(positions), len(selected_curves)
    relation = occupancy.new_zeros((batch, selected_count * positions_count, selected_count * positions_count))
    eye = torch.eye(positions_count, device=occupancy.device, dtype=occupancy.dtype)
    for local_curve in range(selected_count):
        start, end = local_curve * positions_count, (local_curve + 1) * positions_count
        relation[:, start:end, start:end] = eye

    # ``occupancy[:, pair]`` remains exactly M_ij. In particular we do not
    # normalize it, pool it, average it with M_ji.T, or set the reverse block
    # by transposition.
    sampled = occupancy.index_select(-2, positions).index_select(-1, positions)
    pair_lookup = {(int(left), int(right)): pair for pair, (left, right) in enumerate(zip(source.tolist(), target.tolist()))}
    for local_left, left in enumerate(selected_curves.tolist()):
        for local_right, right in enumerate(selected_curves.tolist()):
            if left == right:
                continue
            pair = pair_lookup[(left, right)]
            row_start, row_end = local_left * positions_count, (local_left + 1) * positions_count
            col_start, col_end = local_right * positions_count, (local_right + 1) * positions_count
            relation[:, row_start:row_end, col_start:col_end] = sampled[:, pair]
    return relation


def directed_relation_component_energies(relation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return formal directional/spectral energies and raw spectral facts.

    This public, small helper is used by E3's mathematical tests and
    post-hoc diagnostics.  The PSD target is intentionally detached exactly
    as it is in the formal training objective.
    """
    if relation.ndim < 2 or relation.shape[-1] != relation.shape[-2]:
        raise ValueError("relation must end in a square matrix")
    relation_float = relation.float()
    symmetric = 0.5 * (relation_float + relation_float.transpose(-1, -2))
    antisymmetric = 0.5 * (relation_float - relation_float.transpose(-1, -2))
    with torch.no_grad():
        values, vectors = torch.linalg.eigh(symmetric)
        target_matrix = (vectors * values.clamp_min(0.0).unsqueeze(-2)) @ vectors.transpose(-1, -2)
    size_squared = float(symmetric.shape[-1] ** 2)
    directional_energy = antisymmetric.square().sum(dim=(-2, -1)) / size_squared
    negative_spectral_energy = (symmetric - target_matrix).square().sum(dim=(-2, -1)) / size_squared
    return directional_energy, negative_spectral_energy, values, symmetric


def soft_route_psd_group_loss(
    group_emission: torch.Tensor,
    *,
    curves: int = 8,
    num_positions: int | None = None,
    num_bins: int | None = None,
    objective_mode: str = "full",
    selected_curves: torch.Tensor | None = None,
) -> DirectCosineGroupPSDResult:
    """Distance of the directed CRF relation to the symmetric PSD cone.

    ``group_emission`` is the additive CRF emission ``[B,K*(K-1),L,L]``.
    ``num_bins`` is retained solely as a deprecated configuration alias for
    ``num_positions``; it no longer triggers binning or pooling.
    """
    objective_mode = str(objective_mode).lower()
    if objective_mode not in {"direction", "spectral", "full"}:
        raise ValueError("objective_mode must be direction, spectral, or full")
    if num_positions is None:
        num_positions = 32 if num_bins is None else num_bins
    elif num_bins is not None and num_bins != num_positions:
        raise ValueError("num_positions and deprecated num_bins alias disagree")
    if group_emission.ndim != 4 or group_emission.shape[1] != curves * (curves - 1):
        raise ValueError("group_emission must be [B,K*(K-1),L,L]")
    batch, _pairs_count, length, width = group_emission.shape
    if length != width:
        raise ValueError("D34 directed PSD regularizer expects square fixed-length lattices")

    full_source, full_target = _pairs(curves, group_emission.device)
    if selected_curves is None:
        selected_curves = torch.arange(curves, device=group_emission.device)
    selected_curves = selected_curves.to(device=group_emission.device, dtype=torch.long)
    if selected_curves.ndim == 1:
        selected_curves = selected_curves.unsqueeze(0).expand(batch, -1)
    if selected_curves.ndim != 2 or selected_curves.shape[0] != batch or not 2 <= selected_curves.shape[1] <= curves:
        raise ValueError("selected_curves must be [k] or [B,k], with 2 <= k <= curves")
    if (selected_curves < 0).any() or (selected_curves >= curves).any() or any(len(torch.unique(row)) != len(row) for row in selected_curves):
        raise ValueError("selected_curves is not a valid subset of original curve identities")

    # PAL remains full K=8 / 56 directed pairs.  Group-PSD is the only branch
    # allowed to restrict its CRF occupancy work to the relations it actually
    # uses.  Critically, this selects *pairs*, not positions: every retained
    # pair still receives the exact full-resolution 512x512 occupancy DP.
    selected_count = selected_curves.shape[1]
    all_members = torch.arange(curves, device=group_emission.device).expand(batch, -1)
    if selected_count == curves and torch.equal(selected_curves, all_members):
        pair_indices = torch.arange(len(full_source), device=group_emission.device).expand(batch, -1)
        selected_emission = group_emission
    else:
        member = torch.zeros((batch, curves), dtype=torch.bool, device=group_emission.device)
        member.scatter_(1, selected_curves, True)
        keep = member[:, full_source] & member[:, full_target]
        expected_pairs = selected_count * (selected_count - 1)
        if not torch.all(keep.sum(dim=1) == expected_pairs):
            raise RuntimeError("selected GRC subset did not yield k*(k-1) directed pairs")
        pair_indices = torch.nonzero(keep, as_tuple=False)[:, 1].reshape(batch, expected_pairs)
        selected_emission = group_emission.gather(
            1, pair_indices[:, :, None, None].expand(-1, -1, length, width)
        )
    occupancy = _occupancy(selected_emission.reshape(batch * selected_emission.shape[1], length, width))
    occupancy = occupancy.reshape(batch, selected_emission.shape[1], length, width)
    positions = _sampled_positions(length, int(num_positions), group_emission.device)
    # The two external GRC groups are batched through the expensive occupancy
    # DP above.  Their stateless subsets may differ, so relation block layout
    # is assembled independently afterwards; this tiny loop has no lattice DP.
    relation_parts = []
    for batch_index in range(batch):
        source = full_source.index_select(0, pair_indices[batch_index])
        target = full_target.index_select(0, pair_indices[batch_index])
        relation_parts.append(_directed_relation(
            occupancy[batch_index : batch_index + 1], curves=curves, positions=positions,
            selected_curves=selected_curves[batch_index], pair_source=source, pair_target=target,
        ))
    relation = torch.cat(relation_parts, dim=0)

    # R is intentionally non-symmetric. Its antisymmetric component is a
    # training signal for reciprocal correction, while only S is PSD-projected.
    directional_energy, negative_spectral_energy, values, _symmetric = directed_relation_component_energies(relation)
    # Keep the formal Full objective bit-for-bit identical to the previous
    # implementation.  E3 only selects an existing component; it does not
    # alter R, the projection target, or either normalization.
    if objective_mode == "direction":
        loss = directional_energy
    elif objective_mode == "spectral":
        loss = negative_spectral_energy
    else:
        loss = directional_energy + negative_spectral_energy
    negative = values.clamp_max(0.0)
    diagonal_mask = torch.eye(relation.shape[-1], dtype=torch.bool, device=relation.device)
    relation_float = relation.float()
    offdiagonal_mass = relation_float.masked_fill(diagonal_mask, 0.0).abs().sum(dim=(-2, -1))
    total_mass = relation_float.abs().sum(dim=(-2, -1))
    return DirectCosineGroupPSDResult(
        per_group_loss=loss.to(group_emission.dtype),
        mean_directional_energy=directional_energy.mean().detach().to(group_emission.dtype),
        mean_negative_spectral_energy=negative_spectral_energy.mean().detach().to(group_emission.dtype),
        mean_min_eigenvalue=values.min(dim=-1).values.mean().detach().to(group_emission.dtype),
        mean_negative_eigenvalue_count=(values < -1.0e-6).sum(dim=-1).float().mean().detach().to(group_emission.dtype),
        mean_offdiagonal_relation_mass=offdiagonal_mass.mean().detach().to(group_emission.dtype),
        mean_total_relation_mass=total_mass.mean().detach().to(group_emission.dtype),
    )
