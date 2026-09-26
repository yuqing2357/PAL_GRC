"""LEGACY / BASELINE-ONLY M3-Lite implementation.

V3 imports generic Conv1D/UNet helpers from this file, but never constructs
``M3LiteModel``: its historical NULL assignment semantics are not compatible
with the V3 main formulation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .group_assignment import GroupAssignmentHead
from .group_slot_generator import GroupSlotGenerator


def _value(cfg: Any, key: str, default: Any) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _model_section(cfg: Any) -> Any:
    if isinstance(cfg, M3LiteConfig):
        return cfg
    if isinstance(cfg, Mapping) and "model" in cfg:
        return cfg["model"]
    return getattr(cfg, "model", cfg)


def _group_count(channels: int, requested: int) -> int:
    groups = min(int(requested), int(channels))
    while channels % groups:
        groups -= 1
    return max(groups, 1)


def _masked_zero(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    return x if mask is None else x.masked_fill(mask[:, None, :], 0.0)


def _resize_mask(mask: torch.Tensor, length: int) -> torch.Tensor:
    """Mark an output position padded only when its full input bin is padded."""

    if mask.shape[-1] == length:
        return mask
    valid = (~mask).float()[:, None]
    pooled_valid = F.adaptive_max_pool1d(valid, length).squeeze(1)
    return pooled_valid < 0.5


@dataclass(frozen=True)
class M3LiteConfig:
    name: str = "m3_lite"
    input_channels: int = 1
    channels: tuple[int, ...] = (64, 96, 128, 192, 256)
    blocks_per_stage: tuple[int, ...] = (1, 1, 1, 1, 2)
    decoder_blocks_per_stage: int = 1
    groupnorm_groups: int = 8
    dropout: float = 0.0
    bottleneck_heads: int = 8
    bottleneck_dropout: float = 0.1
    interaction_gate_init: float = 0.1
    assume_full_length: bool = True
    proj_dim: int = 96
    emb_temp: float = 0.07
    group_alignment: Mapping[str, Any] | None = None


class ResidualBlock1d(nn.Module):
    def __init__(self, channels: int, groups: int, dropout: float) -> None:
        super().__init__()
        norm_groups = _group_count(channels, groups)
        self.norm1 = nn.GroupNorm(norm_groups, channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(norm_groups, channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden))))
        return _masked_zero(x + hidden, mask)


class ResidualStage1d(nn.Module):
    def __init__(self, channels: int, blocks: int, groups: int, dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            ResidualBlock1d(channels, groups, dropout) for _ in range(int(blocks))
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, mask)
        return x


class Downsample1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return _masked_zero(self.proj(x), mask)


class UpsampleFuse1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        blocks: int,
        groups: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.up_proj = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.fuse = nn.Conv1d(out_channels + skip_channels, out_channels, 1)
        self.stage = ResidualStage1d(out_channels, blocks, groups, dropout)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        x = self.up_proj(x)
        x = self.fuse(torch.cat((x, skip), dim=1))
        return self.stage(_masked_zero(x, mask), mask)


class PairwiseCurveInteraction(nn.Module):
    """Exchange one coarse message per partner, then average partners equally.

    Curve count is dynamic. No curve identity or anchor index is embedded, so
    reordering the curve dimension produces the same reordering at the output.
    """

    def __init__(
        self,
        channels: int,
        heads: int,
        dropout: float,
        gate_init: float,
        assume_full_length: bool,
    ) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("bottleneck channels must be divisible by bottleneck_heads")
        self.channels = int(channels)
        self.heads = int(heads)
        self.head_dim = self.channels // self.heads
        self.dropout_p = float(dropout)
        self.assume_full_length = bool(assume_full_length)
        self.norm_qkv = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels, bias=False)
        self.out_proj = nn.Linear(channels, channels)
        self.cross_gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.norm_ffn = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * channels, channels),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _pair_indices(curves: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        ids = torch.arange(curves, device=device)
        target_grid = ids[:, None].expand(curves, curves)
        source_grid = ids[None, :].expand(curves, curves)
        keep = target_grid != source_grid
        return target_grid[keep], source_grid[keep]

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, curves, length, channels = x.shape
        if channels != self.channels:
            raise ValueError(f"expected {self.channels} bottleneck channels, got {channels}")
        if curves < 2:
            output = x + self.ffn(self.norm_ffn(x))
            return output if mask is None else output.masked_fill(mask[..., None], 0.0)

        normalized = self.norm_qkv(x)
        qkv = self.qkv(normalized).reshape(
            batch, curves, length, 3, self.heads, self.head_dim
        )
        query, key, value = qkv.unbind(3)
        # [B, K, H, T, Dh]
        query = query.permute(0, 1, 3, 2, 4)
        key = key.permute(0, 1, 3, 2, 4)
        value = value.permute(0, 1, 3, 2, 4)

        if edge_index is None:
            targets, sources = self._pair_indices(curves, x.device)
        else:
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError("edge_index must have shape [2, E]")
            edge_index = edge_index.to(device=x.device, dtype=torch.long)
            targets, sources = edge_index.unbind(0)
        edges = int(targets.numel())
        if edges == 0:
            output = x + self.ffn(self.norm_ffn(x))
            return output if mask is None else output.masked_fill(mask[..., None], 0.0)
        query_pairs = query.index_select(1, targets).reshape(
            batch * edges, self.heads, length, self.head_dim
        )
        key_pairs = key.index_select(1, sources).reshape(
            batch * edges, self.heads, length, self.head_dim
        )
        value_pairs = value.index_select(1, sources).reshape(
            batch * edges, self.heads, length, self.head_dim
        )

        attention_mask = None
        if not self.assume_full_length and mask is not None:
            source_keep = (~mask).index_select(1, sources).reshape(batch * edges, length)
            attention_mask = source_keep[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            query_pairs,
            key_pairs,
            value_pairs,
            attn_mask=attention_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        attended = (
            attended.reshape(batch, edges, self.heads, length, self.head_dim)
            .permute(0, 1, 3, 2, 4)
            .reshape(batch, edges, length, channels)
        )
        attended = self.out_proj(attended)

        aggregated = torch.zeros_like(x).index_add(1, targets, attended)
        partner_counts = torch.zeros(curves, device=x.device, dtype=x.dtype).index_add(
            0, targets, torch.ones(edges, device=x.device, dtype=x.dtype)
        )
        aggregated = aggregated / partner_counts.clamp_min(1.0)[None, :, None, None]
        x = x + self.cross_gate * aggregated
        x = x + self.ffn(self.norm_ffn(x))
        return x if mask is None else x.masked_fill(mask[..., None], 0.0)


class M3LiteModel(nn.Module):
    """Shared multi-scale trace encoder with one coarse, anchor-free interaction."""

    def __init__(self, cfg: M3LiteConfig | Mapping[str, Any] | Any = M3LiteConfig()) -> None:
        super().__init__()
        cfg = _model_section(cfg)
        channels = tuple(int(item) for item in _value(cfg, "channels", (64, 96, 128, 192, 256)))
        if len(channels) != 5:
            raise ValueError("M3-Lite channels must contain five levels")
        blocks_raw = _value(cfg, "blocks_per_stage", (1, 1, 1, 1, 2))
        if isinstance(blocks_raw, int):
            blocks = (int(blocks_raw),) * len(channels)
        else:
            blocks = tuple(int(item) for item in blocks_raw)
        if len(blocks) != len(channels) or min(blocks) < 1:
            raise ValueError("blocks_per_stage must contain five positive values")

        groups = int(_value(cfg, "groupnorm_groups", 8))
        dropout = float(_value(cfg, "dropout", 0.0))
        input_channels = int(_value(cfg, "input_channels", 1))
        decoder_blocks = int(_value(cfg, "decoder_blocks_per_stage", 1))
        self.assume_full_length = bool(_value(cfg, "assume_full_length", True))

        self.stem = nn.Conv1d(input_channels, channels[0], 7, padding=3)
        self.encoder_stages = nn.ModuleList(
            ResidualStage1d(channel, count, groups, dropout)
            for channel, count in zip(channels, blocks)
        )
        self.downsamples = nn.ModuleList(
            Downsample1d(channels[index], channels[index + 1])
            for index in range(len(channels) - 1)
        )
        self.interaction = PairwiseCurveInteraction(
            channels=channels[-1],
            heads=int(_value(cfg, "bottleneck_heads", 8)),
            dropout=float(_value(cfg, "bottleneck_dropout", 0.1)),
            gate_init=float(_value(cfg, "interaction_gate_init", 0.1)),
            assume_full_length=self.assume_full_length,
        )
        self.decoder = nn.ModuleList(
            UpsampleFuse1d(
                in_channels=channels[index],
                skip_channels=channels[index - 1],
                out_channels=channels[index - 1],
                blocks=decoder_blocks,
                groups=groups,
                dropout=dropout,
            )
            for index in range(len(channels) - 1, 0, -1)
        )
        output_channels = int(_value(cfg, "proj_dim", 96))
        self.head = nn.Sequential(
            nn.GroupNorm(_group_count(channels[0], groups), channels[0]),
            nn.SiLU(),
            nn.Conv1d(channels[0], output_channels, 1),
        )
        self.register_buffer("emb_temp", torch.tensor(float(_value(cfg, "emb_temp", 0.07))))
        group_cfg = _value(cfg, "group_alignment", {}) or {}
        self.group_alignment_enabled = bool(_value(group_cfg, "enabled", False))
        self.group_slot_generator: GroupSlotGenerator | None = None
        self.group_assignment_head: GroupAssignmentHead | None = None
        self.group_crf_alpha_raw: nn.Parameter | None = None
        self.group_crf_beta: nn.Parameter | None = None
        if self.group_alignment_enabled:
            num_slots = int(_value(group_cfg, "num_slots", 512))
            if num_slots < 1:
                raise ValueError("group_alignment.num_slots must be positive")
            slot_dim = int(_value(group_cfg, "slot_dim", output_channels))
            self.group_slot_generator = GroupSlotGenerator(
                context_dim=channels[-1], num_slots=num_slots, slot_dim=slot_dim,
                hidden_dim=int(_value(group_cfg, "hidden_dim", slot_dim)),
                layers=int(_value(group_cfg, "layers", 2)),
                heads=int(_value(group_cfg, "heads", 4)),
                dropout=float(_value(group_cfg, "dropout", 0.0)),
            )
            self.group_assignment_head = GroupAssignmentHead(
                embedding_dim=output_channels, slot_dim=slot_dim,
                assignment_dim=int(_value(group_cfg, "assignment_dim", output_channels)),
                temperature=float(_value(group_cfg, "temperature", 1.0)),
                null_logit_init=float(_value(group_cfg, "null_logit_init", 0.0)),
            )
            if not bool(_value(group_cfg, "crf_score_learnable", True)):
                raise ValueError("formal groupwise CRF score must be learnable")
            alpha_init = float(_value(group_cfg, "crf_scale_init", 1.0))
            if alpha_init <= 0.0:
                raise ValueError("group_alignment.crf_scale_init must be positive")
            bias_init = str(_value(group_cfg, "crf_bias_init", "uniform_neutral")).lower()
            if bias_init != "uniform_neutral":
                raise ValueError("formal groupwise CRF only supports crf_bias_init=uniform_neutral")
            # softplus(alpha_raw)=alpha_init exactly, rather than treating the
            # unconstrained raw parameter itself as the desired positive scale.
            alpha_raw_init = math.log(math.expm1(alpha_init))
            # In the formal architecture num_slots equals canonical target
            # length. For uniform H, C=1/Q and beta=(alpha+1)log(Q) gives a
            # neutral CRF emission after the CRF's internal -log(Q) term.
            beta_init = (alpha_init + 1.0) * math.log(float(num_slots))
            self.group_crf_alpha_raw = nn.Parameter(torch.tensor(alpha_raw_init))
            self.group_crf_beta = nn.Parameter(torch.tensor(beta_init))

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
        edge_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if z.ndim != 3 or mask.shape != z.shape:
            raise ValueError("M3-Lite expects z and mask with shape [B, K, L]")
        batch, curves, length = z.shape
        if length % 16:
            raise ValueError("sequence length must be divisible by 16")
        flat_mask = mask.reshape(batch * curves, length)
        x = z.reshape(batch * curves, 1, length).masked_fill(flat_mask[:, None], 0.0)
        x = _masked_zero(self.stem(x), flat_mask)

        skips: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        current_mask = flat_mask
        for level, stage in enumerate(self.encoder_stages):
            x = stage(x, current_mask)
            skips.append(x)
            masks.append(current_mask)
            if level < len(self.downsamples):
                next_length = x.shape[-1] // 2
                current_mask = _resize_mask(current_mask, next_length)
                x = self.downsamples[level](x, current_mask)

        bottleneck_length = x.shape[-1]
        bottleneck_mask = current_mask.reshape(batch, curves, bottleneck_length)
        x = x.transpose(1, 2).reshape(batch, curves, bottleneck_length, -1)
        x = self.interaction(x, bottleneck_mask, edge_index=edge_index)
        group_context = x
        x = x.reshape(batch * curves, bottleneck_length, -1).transpose(1, 2)

        # skips[-1] is the pre-interaction bottleneck; decoding starts at skips[-2].
        for decoder_index, decoder in enumerate(self.decoder):
            skip_index = len(skips) - 2 - decoder_index
            x = decoder(x, skips[skip_index], masks[skip_index])

        embeddings = self.head(x).transpose(1, 2)
        embeddings = F.normalize(embeddings.float(), dim=-1).to(x.dtype)
        embeddings = embeddings.reshape(batch, curves, length, -1)
        embeddings = embeddings.masked_fill(mask[..., None], 0.0)
        output = {"emb": embeddings, "emb_temp": self.emb_temp}
        if self.group_alignment_enabled:
            if self.group_slot_generator is None or self.group_assignment_head is None:
                raise RuntimeError("group alignment modules were not initialized")
            if self.group_slot_generator.num_slots != length:
                raise ValueError(
                    "formal groupwise architecture defines num_group_slots "
                    "equal to canonical sequence length"
                )
            context = group_context.reshape(batch, curves * bottleneck_length, -1)
            context_mask = bottleneck_mask.reshape(batch, curves * bottleneck_length)
            slots = self.group_slot_generator(context, context_mask)
            assignment = self.group_assignment_head(embeddings, slots, mask)
            if self.group_crf_alpha_raw is None or self.group_crf_beta is None:
                raise RuntimeError("groupwise structured CRF parameters were not initialized")
            alpha = F.softplus(self.group_crf_alpha_raw)
            uniform_emission = -alpha * math.log(float(length)) + self.group_crf_beta - math.log(float(length))
            output.update({
                "group_context": group_context, "group_slots": slots,
                "group_crf_alpha": alpha, "group_crf_beta": self.group_crf_beta,
                "group_crf_uniform_emission": uniform_emission,
                **assignment,
            })
        return output
