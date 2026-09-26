"""Slot-free D34 model matching the latest two-GSI CMU direct-cosine route."""
from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from ssia_alignment.model_parts.hierarchical_groupwise import GroupSetInteraction
from ssia_alignment.model_parts.m3_lite import (
    Downsample1d, ResidualStage1d, UpsampleFuse1d, _group_count,
    _masked_zero, _model_section, _resize_mask, _value,
)


class DirectCosineTwoGSIEndModel(nn.Module):
    """K=8/L=512 backbone with two group interactions and no slot pathway."""

    method_version = "ssia_model"
    group_alignment_enabled = False
    group_alignment_no_null = True

    def __init__(self, cfg: Mapping[str, Any] | Any) -> None:
        super().__init__()
        section = _model_section(cfg)
        if str(_value(section, "groupwise_mode", "")).lower() != "interaction_direct_cosine":
            raise ValueError("slot-free D34 model requires groupwise_mode=interaction_direct_cosine")
        if "group_alignment" in section:
            raise ValueError("slot-free direct-cosine model must not declare group_alignment or slots")
        channels = tuple(int(item) for item in _value(section, "channels", (64, 128, 192, 256, 320)))
        if channels != (64, 128, 192, 256, 320) or tuple(_value(section, "blocks_per_stage", ())) != (1, 1, 2, 2, 2):
            raise ValueError("the D34 main route is the registered Large five-level backbone")
        if int(_value(section, "input_channels", 1)) != 1 or int(_value(section, "group_interaction_blocks", 0)) != 2:
            raise ValueError("D34 main route requires one impedance channel and exactly two GSI blocks")
        self.channels = channels
        self.proj_dim = int(_value(section, "proj_dim", 192))
        groups = int(_value(section, "groupnorm_groups", 8))
        dropout = float(_value(section, "dropout", 0.0))
        self.assume_full_length = bool(_value(section, "assume_full_length", True))
        self.cross_sequence_communication = bool(_value(section, "cross_sequence_communication", True))
        self.stem = nn.Conv1d(1, channels[0], 7, padding=3)
        self.encoder_stages = nn.ModuleList(ResidualStage1d(width, depth, groups, dropout) for width, depth in zip(channels, (1, 1, 2, 2, 2)))
        self.downsamples = nn.ModuleList(Downsample1d(channels[index], channels[index + 1]) for index in range(4))
        heads = int(_value(section, "group_set_heads", 8))
        interaction_dropout = float(_value(section, "group_interaction_dropout", 0.1))
        self.group_interaction32 = GroupSetInteraction(channels[-1], heads, interaction_dropout, self.assume_full_length)
        self.group_interaction64 = GroupSetInteraction(channels[-2], heads, interaction_dropout, self.assume_full_length)
        self.decoder = nn.ModuleList(UpsampleFuse1d(channels[index], channels[index - 1], channels[index - 1], 1, groups, dropout) for index in range(4, 0, -1))
        self.head = nn.Sequential(nn.GroupNorm(_group_count(channels[0], groups), channels[0]), nn.SiLU(), nn.Conv1d(channels[0], self.proj_dim, 1))
        direct = _value(section, "direct_matching", {}) or {}
        alpha = float(_value(direct, "crf_scale_init", 1.0))
        if alpha <= 0.0:
            raise ValueError("direct_matching.crf_scale_init must be positive")
        target_length = int(_value(direct, "crf_target_length", 512))
        if target_length != 512:
            raise ValueError("D34 direct-cosine CRF uses the fixed reference length 512")
        beta = float(_value(direct, "crf_bias_init", math.log(float(target_length))))
        self.group_crf_alpha_raw = nn.Parameter(torch.tensor(math.log(math.expm1(alpha))))
        self.group_crf_beta = nn.Parameter(torch.tensor(beta))

    def forward(self, z: torch.Tensor, mask: torch.Tensor, edge_index: Any = None) -> dict[str, torch.Tensor | str]:
        del edge_index
        if z.ndim != 3 or z.shape != mask.shape or z.shape[1:] != (8, 512):
            raise ValueError("slot-free D34 model expects z/mask [B,8,512]")
        batch, curves, length = z.shape
        flat_mask = mask.reshape(batch * curves, length)
        x = _masked_zero(self.stem(z.reshape(batch * curves, 1, length).masked_fill(flat_mask[:, None], 0.0)), flat_mask)
        skips, masks, current = [], [], flat_mask
        for level, stage in enumerate(self.encoder_stages):
            x = stage(x, current)
            skips.append(x); masks.append(current)
            if level < 4:
                current = _resize_mask(current, x.shape[-1] // 2)
                x = self.downsamples[level](x, current)
        mask32 = current.reshape(batch, curves, 32)
        connectivity = "leave_self_out" if self.cross_sequence_communication else "self_only"
        x = self.group_interaction32(x.transpose(1, 2).reshape(batch, curves, 32, self.channels[-1]), mask32, connectivity=connectivity).reshape(batch * curves, 32, self.channels[-1]).transpose(1, 2)
        x = self.decoder[0](x, skips[3], masks[3])
        mask64 = masks[3].reshape(batch, curves, 64)
        x = self.group_interaction64(x.transpose(1, 2).reshape(batch, curves, 64, self.channels[-2]), mask64, connectivity=connectivity).reshape(batch * curves, 64, self.channels[-2]).transpose(1, 2)
        for decoder_index in range(1, len(self.decoder)):
            x = self.decoder[decoder_index](x, skips[3 - decoder_index], masks[3 - decoder_index])
        embeddings = F.normalize(self.head(x).transpose(1, 2).float(), dim=-1).to(x.dtype).reshape(batch, curves, length, self.proj_dim).masked_fill(mask[..., None], 0.0)
        return {
            "emb": embeddings,
            "matching_features": embeddings,
            "matching_mode": "direct_cosine",
            "group_crf_alpha": F.softplus(self.group_crf_alpha_raw),
            "group_crf_beta": self.group_crf_beta,
        }
