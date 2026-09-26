"""Unified V3 global regularization on membership-induced relations only.

This module intentionally does not accept embeddings.  It constructs the
coarse global relation R=H_bar H_bar^T, degree-normalizes it to S, then uses
the *same* S for the V2 projector objective and V3 rank-free complexity term.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Iterable

import torch


def _timer(profile_timer, name: str):
    return profile_timer(name) if profile_timer is not None else nullcontext()


def _spectral_statistics(eigenvalues: torch.Tensor, eps: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return rank-free loss and diagnostics for ascending ``eigvalsh`` output.

    ``global_effective_rank`` is the spectral effective dimension of S, not
    the number of active assignment slots.  Broad uniform H can activate every
    slot yet induce an almost rank-one co-membership relation.
    """
    positive = eigenvalues.clamp_min(0.0).flip(-1)  # descending; clamp is numerical only.
    energy = positive.square()
    probability = energy / energy.sum(-1, keepdim=True).clamp_min(eps)
    nodes = eigenvalues.shape[-1]
    weights = torch.linspace(0.0, 1.0, nodes, device=eigenvalues.device, dtype=eigenvalues.dtype)
    rankfree = (probability * weights).sum(-1)
    entropy = -(probability * probability.clamp_min(eps).log()).sum(-1)
    participation = positive.sum(-1).square() / energy.sum(-1).clamp_min(eps)
    total_energy = eigenvalues.square().sum(-1).clamp_min(eps)
    negative_energy = eigenvalues.clamp_max(0.0).square().sum(-1) / total_energy
    return rankfree, {
        "global_eigenvalue_max": eigenvalues[..., -1],
        "global_eigenvalue_min": eigenvalues[..., 0],
        "global_top1_energy": probability[..., 0],
        "global_top5_energy": probability[..., : min(5, nodes)].sum(-1),
        "global_effective_rank": entropy.exp(),
        "global_participation_rank": participation,
        "global_spectral_entropy": entropy,
        "global_numerical_negative_energy": negative_energy,
    }


def _relation_to_losses(relation: torch.Tensor, eps: float, profile_timer=None, *, compute_rankfree: bool = True) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor | None]:
    """Compute S, projector and rank-free terms for dense batched relations."""
    with _timer(profile_timer, "global_degree_normalization"):
        degree = relation.sum(-1).clamp_min(eps)
        inverse = torch.rsqrt(degree)
        normalized = inverse[..., :, None] * relation * inverse[..., None, :]
        normalized = 0.5 * (normalized + normalized.transpose(-1, -2))
    with _timer(profile_timer, "global_projector"):
        residual = torch.bmm(normalized, normalized) - normalized
        projector = residual.square().sum(dim=(-2, -1)) / normalized.square().sum(dim=(-2, -1)).clamp_min(eps)
    if compute_rankfree:
        with _timer(profile_timer, "global_eigvalsh"):
            eigenvalues = torch.linalg.eigvalsh(normalized)
        with _timer(profile_timer, "global_rankfree_terms"):
            rankfree, spectral = _spectral_statistics(eigenvalues, eps)
    else:
        eigenvalues = None
        rankfree = projector.new_zeros(projector.shape)
        spectral = {}
    symmetry = (normalized - normalized.transpose(-1, -2)).abs().amax(dim=(-2, -1))
    logs = {
        "global_projector_residual": projector,
        "global_degree_mean": degree.mean(-1),
        "global_degree_min": degree.min(-1).values,
        "global_degree_max": degree.max(-1).values,
        "global_symmetry_error": symmetry,
        **spectral,
    }
    # Concise aliases make the saved formal logs self-describing while the
    # ``global_*`` names remain the canonical diagnostic namespace.
    if eigenvalues is not None:
        logs["eigenvalue_min"] = eigenvalues[..., 0]
        logs["eigenvalue_max"] = eigenvalues[..., -1]
    return projector, rankfree, logs, normalized, eigenvalues


def group_spectral_regularizer(
    membership: torch.Tensor,
    mask: torch.Tensor,
    *,
    stride: int = 8,
    batch_indices: Iterable[int] | None = None,
    eps: float = 1.0e-8,
    assume_full_length: bool = True,
    compute_rankfree: bool = True,
    profile_timer=None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Return V3 projector and rank-free losses from coarse memberships.

    ``membership`` is [B,K,L,T].  The full-length path is fully batched.  The
    padded fallback preserves the same mathematics by compacting valid nodes
    per group before constructing its relation.
    """
    if membership.ndim != 4 or mask.shape != membership.shape[:3]:
        raise ValueError("membership must be [B,K,L,T] with mask [B,K,L]")
    batch, curves, length, slots = membership.shape
    selected = list(range(batch)) if batch_indices is None else [int(index) for index in batch_indices]
    if not selected:
        zero = membership.new_zeros((), dtype=torch.float32)
        return zero, zero, {"global_groups_used": zero}
    index = torch.as_tensor(selected, device=membership.device)
    positions = torch.arange(0, length, max(1, int(stride)), device=membership.device)
    with _timer(profile_timer, "global_coarse_membership_gather"):
        coarse = membership.index_select(0, index).index_select(2, positions).float()
        coarse_mask = mask.index_select(0, index).index_select(2, positions)

    if assume_full_length:
        with _timer(profile_timer, "global_relation_bmm"):
            flat = coarse.reshape(len(selected), curves * positions.numel(), slots)
            relation = torch.bmm(flat, flat.transpose(1, 2))
        projector, rankfree, logs, _normalized, _eigenvalues = _relation_to_losses(relation, eps, profile_timer, compute_rankfree=compute_rankfree)
        output = {key: value.mean().detach() for key, value in logs.items()}
        output["global_projector_loss"] = projector.mean().detach()
        output["global_rankfree_loss"] = rankfree.mean().detach()
        output["global_groups_used"] = projector.new_tensor(float(len(selected)))
        output["global_coarse_nodes"] = projector.new_tensor(float(curves * positions.numel()))
        return projector.mean(), rankfree.mean(), output

    # Exact compact-node fallback for a padded future dataset.  It is outside
    # the formal full-length hot path and intentionally favors correctness.
    projectors, rankfrees, entries = [], [], []
    for group in range(len(selected)):
        compact = coarse[group][~coarse_mask[group]]
        if compact.numel() == 0:
            continue
        with _timer(profile_timer, "global_relation_bmm"):
            relation = compact @ compact.transpose(0, 1)
        projector, rankfree, logs, _normalized, _eigenvalues = _relation_to_losses(relation[None], eps, profile_timer, compute_rankfree=compute_rankfree)
        projectors.append(projector[0]); rankfrees.append(rankfree[0]); entries.append({key: value[0] for key, value in logs.items()})
    if not projectors:
        zero = membership.new_zeros((), dtype=torch.float32)
        return zero, zero, {"global_groups_used": zero}
    output = {key: torch.stack([item[key] for item in entries]).mean().detach() for key in entries[0]}
    output["global_projector_loss"] = torch.stack(projectors).mean().detach()
    output["global_rankfree_loss"] = torch.stack(rankfrees).mean().detach()
    output["global_groups_used"] = membership.new_tensor(float(len(projectors)), dtype=torch.float32)
    output["global_coarse_nodes"] = membership.new_tensor(float(curves * positions.numel()), dtype=torch.float32)
    return torch.stack(projectors).mean(), torch.stack(rankfrees).mean(), output


def rankfree_projector_value(rank: int, nodes: int) -> float:
    """Analytic rank-free value of a rank-r ideal projector."""
    if not 1 <= rank <= nodes:
        raise ValueError("rank must be in [1, nodes]")
    return float(rank - 1) / float(2 * (nodes - 1))
