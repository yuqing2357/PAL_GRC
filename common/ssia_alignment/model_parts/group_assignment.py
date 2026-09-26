"""LEGACY / BASELINE-ONLY NULL assignment head.

It remains only as copied historical reference.  The V3 factory rejects this
path; V3 main uses ``CosineNoNullAssignmentHead`` instead.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GroupAssignmentHead(nn.Module):
    def __init__(self, embedding_dim: int, slot_dim: int, assignment_dim: int = 96, temperature: float = 1.0, null_logit_init: float = 0.0) -> None:
        super().__init__()
        self.embedding_projection = nn.Linear(embedding_dim, assignment_dim, bias=False)
        self.slot_projection = nn.Linear(slot_dim, assignment_dim, bias=False)
        self.register_buffer("temperature", torch.tensor(float(temperature)))
        self.null_logit = nn.Parameter(torch.tensor(float(null_logit_init)))

    def forward(self, embeddings: torch.Tensor, slots: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if embeddings.ndim != 4 or slots.ndim != 3:
            raise ValueError("expected embeddings [B,K,L,D] and slots [B,T,D]")
        left = self.embedding_projection(embeddings)
        right = self.slot_projection(slots)
        pre_temperature_logits = torch.einsum("bkld,btd->bklt", left, right)
        null = self.null_logit.expand(*pre_temperature_logits.shape[:-1], 1)
        raw_logits = torch.cat((pre_temperature_logits, null), dim=-1)
        logits = raw_logits / self.temperature.clamp_min(1.0e-4)
        assignments = torch.softmax(logits, dim=-1)
        if mask is not None:
            logits = logits.masked_fill(mask[..., None], 0.0)
            assignments = assignments.masked_fill(mask[..., None], 0.0)
        return {"group_assignment_logits": logits, "group_assignment_logits_pre_temperature": raw_logits, "group_assignments": assignments,
                "group_membership": assignments[..., :-1], "null_probability": assignments[..., -1]}
