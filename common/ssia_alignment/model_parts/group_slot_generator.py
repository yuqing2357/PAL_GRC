"""Input-conditioned unordered latent slots for Groupwise Multi-Curve Alignment."""
from __future__ import annotations

import torch
import torch.nn as nn


class _CrossSlotBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim), nn.Dropout(dropout))

    def forward(self, slots: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor | None) -> torch.Tensor:
        query = self.query_norm(slots)
        keys = self.context_norm(context)
        attended, _ = self.cross_attention(query, keys, keys, key_padding_mask=context_mask, need_weights=False)
        slots = slots + attended
        return slots + self.ffn(self.ffn_norm(slots))


class GroupSlotGenerator(nn.Module):
    """Latent queries become a set of slots conditioned on the current group.

    Query indices initialise an exchangeable latent set; no slot positional
    encoding, curve identity embedding, anchor curve or geological coordinate
    enters this module.
    """

    def __init__(self, context_dim: int, num_slots: int, slot_dim: int = 96, hidden_dim: int = 96, layers: int = 2, heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("group slot hidden_dim must be divisible by heads")
        self.num_slots = int(num_slots)
        self.context_projection = nn.Linear(context_dim, hidden_dim)
        self.latent_queries = nn.Parameter(torch.empty(self.num_slots, hidden_dim))
        nn.init.normal_(self.latent_queries, std=hidden_dim ** -0.5)
        self.blocks = nn.ModuleList(_CrossSlotBlock(hidden_dim, heads, dropout) for _ in range(int(layers)))
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, slot_dim)

    def forward(self, context: torch.Tensor, context_mask: torch.Tensor | None = None) -> torch.Tensor:
        if context.ndim != 3:
            raise ValueError("group context must have shape [B,N,C]")
        hidden = self.context_projection(context)
        slots = self.latent_queries.unsqueeze(0).expand(context.shape[0], -1, -1)
        for block in self.blocks:
            slots = block(slots, hidden, context_mask)
        return self.output_projection(self.output_norm(slots))
