"""END's frozen hierarchical model contract, including capacity scaling.

This implementation intentionally owns the END construction rather than
changing V3's historical model factory. At the Large settings it creates the
identical module names, tensor shapes, state-dict keys and forward semantics as
the frozen V3/V6 hierarchy. Tiny and Base vary only pre-registered width
fields, so old V3--V6 code and checkpoints remain untouched.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from ssia_alignment.model_parts.hierarchical_groupwise import (
    CoarseToFineGroupSlotGenerator,
    CosineNoNullAssignmentHead,
    GroupSetInteraction,
)
from ssia_alignment.model_parts.m3_lite import (
    Downsample1d,
    ResidualStage1d,
    UpsampleFuse1d,
    _group_count,
    _masked_zero,
    _model_section,
    _resize_mask,
    _value,
)


_END_CAPACITIES = {
    (32, 64, 96, 128, 160): "tiny",
    (48, 96, 144, 192, 240): "base",
    (64, 128, 192, 256, 320): "large",
}


class HierarchicalGroupwiseEndModel(nn.Module):
    """One END topology for Tiny, Base and the frozen Large checkpoint."""

    method_version = "end_hierarchical_groupwise"
    group_alignment_enabled = True
    group_alignment_no_null = True

    def __init__(self, cfg: Mapping[str, Any] | Any) -> None:
        super().__init__()
        section = _model_section(cfg)
        channels = tuple(int(value) for value in _value(section, "channels", (64, 128, 192, 256, 320)))
        blocks = tuple(int(value) for value in _value(section, "blocks_per_stage", (1, 1, 2, 2, 2)))
        if channels not in _END_CAPACITIES or blocks != (1, 1, 2, 2, 2):
            raise ValueError("END permits only Tiny/Base/Large widths with the fixed five-level residual depth")
        if int(_value(section, "decoder_blocks_per_stage", 1)) != 1:
            raise ValueError("END scaling fixes one residual block per decoder stage")
        if int(_value(section, "ffn_ratio", 4)) != 4:
            raise ValueError("END scaling fixes the interaction and slot FFN expansion ratio at 4")
        self.channels = channels
        groups = int(_value(section, "groupnorm_groups", 8))
        dropout = float(_value(section, "dropout", 0.0))
        self.assume_full_length = bool(_value(section, "assume_full_length", True))
        self.stem = nn.Conv1d(int(_value(section, "input_channels", 1)), channels[0], 7, padding=3)
        self.encoder_stages = nn.ModuleList(ResidualStage1d(width, depth, groups, dropout) for width, depth in zip(channels, blocks))
        self.downsamples = nn.ModuleList(Downsample1d(channels[index], channels[index + 1]) for index in range(4))

        heads = int(_value(section, "group_set_heads", 8))
        interaction_dropout = float(_value(section, "group_interaction_dropout", 0.1))
        self.group_interaction32 = GroupSetInteraction(channels[-1], heads, interaction_dropout, self.assume_full_length)
        self.group_interaction64 = GroupSetInteraction(channels[-2], heads, interaction_dropout, self.assume_full_length)
        self.decoder = nn.ModuleList(
            UpsampleFuse1d(channels[index], channels[index - 1], channels[index - 1], 1, groups, dropout)
            for index in range(4, 0, -1)
        )
        self.proj_dim = int(_value(section, "proj_dim", 192))
        self.head = nn.Sequential(
            nn.GroupNorm(_group_count(channels[0], groups), channels[0]),
            nn.SiLU(),
            nn.Conv1d(channels[0], self.proj_dim, 1),
        )
        self.register_buffer("emb_temp", torch.tensor(float(_value(section, "emb_temp", 0.07))))

        group = _value(section, "group_alignment", {}) or {}
        if bool(_value(group, "use_null", False)):
            raise ValueError("END is no-NULL; set model.group_alignment.use_null=false")
        slots = int(_value(group, "num_slots", 512))
        slot_dim = int(_value(group, "slot_dim", 192))
        self.group_slot_generator = CoarseToFineGroupSlotGenerator(
            channels[-1], channels[-2], slots, slot_dim,
            int(_value(group, "heads", 8)), float(_value(group, "dropout", 0.0)),
        )
        assignment_scale = float(_value(group, "assignment_scale", _value(group, "assignment_scale_init", 10.0)))
        self.group_assignment_head = CosineNoNullAssignmentHead(
            self.proj_dim, slot_dim, int(_value(group, "assignment_dim", 192)), assignment_scale,
            bool(_value(group, "assignment_scale_trainable", False)),
        )
        alpha = float(_value(group, "crf_scale_init", 1.0))
        if alpha <= 0.0:
            raise ValueError("crf_scale_init must be positive")
        target_length = int(_value(group, "crf_target_length", 512))
        self.group_crf_alpha_raw = nn.Parameter(torch.tensor(math.log(math.expm1(alpha))))
        self.group_crf_beta = nn.Parameter(torch.tensor(alpha * math.log(float(slots)) + math.log(float(target_length))))
        self.num_slots, self.crf_target_length = slots, target_length

    def forward(self, z: torch.Tensor, mask: torch.Tensor, edge_index: Any = None) -> dict[str, torch.Tensor]:
        del edge_index
        if z.ndim != 3 or z.shape != mask.shape:
            raise ValueError("expects [B,K,L]")
        batch, curves, length = z.shape
        if length != 512:
            raise ValueError("END fixes canonical L=512 for every capacity scale")
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
        f32 = self.group_interaction32(x.transpose(1, 2).reshape(batch, curves, 32, self.channels[-1]), mask32)
        context32 = f32.reshape(batch, curves * 32, self.channels[-1])
        context32mask = mask32.reshape(batch, curves * 32)
        x = f32.reshape(batch * curves, 32, self.channels[-1]).transpose(1, 2)
        x = self.decoder[0](x, skips[3], masks[3])

        mask64 = masks[3].reshape(batch, curves, 64)
        f64 = self.group_interaction64(x.transpose(1, 2).reshape(batch, curves, 64, self.channels[-2]), mask64)
        context64 = f64.reshape(batch, curves * 64, self.channels[-2])
        context64mask = mask64.reshape(batch, curves * 64)
        x = f64.reshape(batch * curves, 64, self.channels[-2]).transpose(1, 2)
        for decoder_index in range(1, len(self.decoder)):
            x = self.decoder[decoder_index](x, skips[3 - decoder_index], masks[3 - decoder_index])

        emb = F.normalize(self.head(x).transpose(1, 2).float(), dim=-1).to(x.dtype)
        emb = emb.reshape(batch, curves, length, self.proj_dim).masked_fill(mask[..., None], 0.0)
        slots = self.group_slot_generator(context32, context64, context32mask, context64mask)
        assignment = self.group_assignment_head(emb, slots, mask)
        alpha = F.softplus(self.group_crf_alpha_raw)
        uniform_emission = -alpha * math.log(float(self.num_slots)) + self.group_crf_beta - math.log(float(self.crf_target_length))
        return {
            "emb": emb, "emb_temp": self.emb_temp,
            "group_context32": f32, "group_context64": f64, "group_slots": slots,
            "group_crf_alpha": alpha, "group_crf_beta": self.group_crf_beta,
            "group_crf_uniform_emission": uniform_emission, **assignment,
        }


def build_end_model(cfg: Mapping[str, Any] | Any) -> HierarchicalGroupwiseEndModel:
    """Build the only permitted END topology at one pre-registered capacity."""
    section = cfg.get("model", cfg) if isinstance(cfg, Mapping) else getattr(cfg, "model", cfg)
    name = section.get("name", "hierarchical_groupwise_end") if isinstance(section, Mapping) else getattr(section, "name", "hierarchical_groupwise_end")
    if str(name).lower() == "interaction_direct_cosine_two_gsi_large_always_overlap_crf_r0":
        from .direct_cosine_model import DirectCosineTwoGSIEndModel
        return DirectCosineTwoGSIEndModel(cfg)
    if str(name).lower() not in {"hierarchical_groupwise_end", "hierarchical-groupwise-end", "groupwise_hierarchical_end"}:
        raise ValueError("END requires model.name=hierarchical_groupwise_end")
    return HierarchicalGroupwiseEndModel(cfg)
