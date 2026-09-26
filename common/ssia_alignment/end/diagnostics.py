"""Detached END diagnostics; none of these values alter the gradient path."""
from __future__ import annotations

import torch


@torch.no_grad()
def membership_diagnostics(
    membership: torch.Tensor,
    mask: torch.Tensor,
    correspondence: torch.Tensor,
    output: dict[str, torch.Tensor],
    valid_pair: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Report the stable H diagnostics used by the V3 family without RankFree."""
    active = (~mask)[..., None].to(membership.dtype)
    mass = (membership.detach() * active).sum(dim=(1, 2))
    total = mass.sum(-1).clamp_min(1.0e-8)
    probability = mass / total[:, None]
    entropy = -(probability * probability.clamp_min(1.0e-8).log()).sum(-1)
    active_slots = (mass > 1.0e-4).sum(-1).to(membership.dtype)
    curves = membership.shape[1]
    self_pairs = torch.eye(curves, dtype=torch.bool, device=membership.device)[None, :, :, None]
    supported = (valid_pair.bool() & ~self_pairs).any(dim=2) & ~mask
    active_count = (~mask).sum().clamp_min(1)
    values = {
        "group_active_slot_count": active_slots.mean(),
        "group_empty_slot_count": (membership.shape[-1] - active_slots).mean(),
        "group_slot_mass_mean": mass.mean(),
        "group_slot_mass_max": mass.max(),
        "group_dominant_slot_fraction": (mass.max(-1).values / total).mean(),
        "group_slot_entropy": entropy.mean(),
        "group_pair_projection_mean": correspondence.mean(),
        "group_pair_projection_max": correspondence.max(),
        "group_pair_projection_std": correspondence.std(),
        "group_crf_alpha": output["group_crf_alpha"],
        "group_crf_beta": output["group_crf_beta"],
        "assignment_scale": output["assignment_scale"],
        "global_supported_node_fraction": supported.sum().to(membership.dtype) / active_count,
        "global_isolated_node_fraction": ((~supported) & ~mask).sum().to(membership.dtype) / active_count,
    }
    values.update(
        {
            "active_slot_count": values["group_active_slot_count"],
            "empty_slot_count": values["group_empty_slot_count"],
            "slot_entropy": values["group_slot_entropy"],
            "dominant_slot_fraction": values["group_dominant_slot_fraction"],
        }
    )
    return {key: value.detach() for key, value in values.items()}
