"""Hierarchical no-NULL Groupwise model; Formal v1 M3-Lite remains untouched."""
from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse only generic Conv/UNet helpers.  The V3 factory never constructs the
# historical M3LiteModel or its NULL assignment path.
from .m3_lite import Downsample1d, ResidualStage1d, UpsampleFuse1d, _group_count, _masked_zero, _model_section, _resize_mask, _value


class GroupSetInteraction(nn.Module):
    """Permutation-equivariant leave-self-out cross-attention over a curve set."""
    def __init__(self, channels: int, heads: int, dropout: float, assume_full_length: bool) -> None:
        super().__init__()
        if channels % heads: raise ValueError("GroupSetInteraction channels must divide heads")
        self.channels, self.assume_full_length = int(channels), bool(assume_full_length)
        self.query_norm, self.context_norm = nn.LayerNorm(channels), nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, 4*channels), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*channels, channels), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, *, connectivity: str = "leave_self_out") -> torch.Tensor:
        batch, curves, length, channels = x.shape
        if channels != self.channels: raise ValueError("GroupSetInteraction channel mismatch")
        if connectivity not in {"leave_self_out", "self_only"}:
            raise ValueError(f"unsupported GroupSetInteraction connectivity: {connectivity}")
        if connectivity == "self_only":
            query=x.reshape(batch*curves,length,channels)
            key_padding=None if mask is None or self.assume_full_length else mask.reshape(batch*curves,length)
            attended,_=self.attention(self.query_norm(query),self.context_norm(query),self.context_norm(query),key_padding_mask=key_padding,need_weights=False)
            output=query+attended; output=output+self.ffn(self.ffn_norm(output)); output=output.reshape(batch,curves,length,channels)
            return output if mask is None else output.masked_fill(mask[...,None],0.)
        if curves < 2:
            output=x+self.ffn(self.ffn_norm(x)); return output if mask is None else output.masked_fill(mask[...,None],0.)
        outputs=[]
        for curve in range(curves):
            other=torch.cat((x[:,:curve],x[:,curve+1:]),dim=1).reshape(batch,(curves-1)*length,channels)
            key_padding=None
            if mask is not None and not self.assume_full_length:
                key_padding=torch.cat((mask[:,:curve],mask[:,curve+1:]),dim=1).reshape(batch,(curves-1)*length)
            query=self.query_norm(x[:,curve]); context=self.context_norm(other)
            attended,_=self.attention(query,context,context,key_padding_mask=key_padding,need_weights=False)
            updated=x[:,curve]+attended; outputs.append(updated+self.ffn(self.ffn_norm(updated)))
        output=torch.stack(outputs,dim=1)
        return output if mask is None else output.masked_fill(mask[...,None],0.)


class _SlotCrossBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__(); self.qnorm=nn.LayerNorm(dim); self.cnorm=nn.LayerNorm(dim)
        self.attn=nn.MultiheadAttention(dim,heads,dropout=dropout,batch_first=True); self.fnorm=nn.LayerNorm(dim)
        self.ffn=nn.Sequential(nn.Linear(dim,4*dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(4*dim,dim),nn.Dropout(dropout))
    def forward(self, slots, context, padding=None):
        result=slots+self.attn(self.qnorm(slots),self.cnorm(context),self.cnorm(context),key_padding_mask=padding,need_weights=False)[0]
        return result+self.ffn(self.fnorm(result))


class CoarseToFineGroupSlotGenerator(nn.Module):
    """Unordered queries -> L32 group context -> L64 refinement; no self-attention."""
    def __init__(self, context32_dim: int, context64_dim: int, num_slots: int, slot_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if slot_dim % heads: raise ValueError("slot_dim must divide attention heads")
        self.num_slots=int(num_slots); self.context32_projection=nn.Linear(context32_dim,slot_dim); self.context64_projection=nn.Linear(context64_dim,slot_dim)
        self.latent_queries=nn.Parameter(torch.empty(num_slots,slot_dim)); nn.init.normal_(self.latent_queries,std=slot_dim**-.5)
        self.coarse=_SlotCrossBlock(slot_dim,heads,dropout); self.refine=_SlotCrossBlock(slot_dim,heads,dropout); self.output_norm=nn.LayerNorm(slot_dim)
    def forward(self, context32, context64, mask32=None, mask64=None):
        slots=self.latent_queries[None].expand(context32.shape[0],-1,-1)
        slots=self.coarse(slots,self.context32_projection(context32),mask32)
        slots=self.refine(slots,self.context64_projection(context64),mask64)
        return self.output_norm(slots)


class CosineNoNullAssignmentHead(nn.Module):
    """Full-resolution membership with normalized cosine logits.

    V3 fixes the assignment scale by default.  A trainable scale remains an
    explicit opt-in ablation setting, rather than an accidental optimizer
    parameter in the main experiment.
    """
    def __init__(self, embedding_dim: int, slot_dim: int, assignment_dim: int,
                 scale_init: float = 10., scale_trainable: bool = False) -> None:
        super().__init__()
        if scale_init <= 0: raise ValueError("assignment_scale_init must be positive")
        self.embedding_projection=nn.Linear(embedding_dim,assignment_dim,bias=False); self.slot_projection=nn.Linear(slot_dim,assignment_dim,bias=False)
        value=torch.tensor(math.log(float(scale_init)))
        if scale_trainable:
            self.log_s=nn.Parameter(value)
        else:
            self.register_buffer("log_s", value)
        self.scale_trainable=bool(scale_trainable)
    def forward(self, embeddings, slots, mask=None):
        left=F.normalize(self.embedding_projection(embeddings),dim=-1); right=F.normalize(self.slot_projection(slots),dim=-1)
        logits=self.log_s.exp()*torch.einsum("bkld,btd->bklt",left,right); h=torch.softmax(logits,dim=-1)
        if mask is not None: logits=logits.masked_fill(mask[...,None],0.); h=h.masked_fill(mask[...,None],0.)
        return {"group_assignment_logits":logits,"group_assignments":h,"group_membership":h,"assignment_scale":self.log_s.exp()}


class HierarchicalGroupwiseModel(nn.Module):
    """Conv1D U-Net, L32+L64 Group Set Attention, and coarse-to-fine slots."""
    group_alignment_enabled=True
    group_alignment_no_null=True
    def __init__(self,cfg: Mapping[str,Any]|Any) -> None:
        super().__init__(); cfg=_model_section(cfg); channels=tuple(int(x) for x in _value(cfg,"channels",(64,128,192,256,320))); blocks=tuple(int(x) for x in _value(cfg,"blocks_per_stage",(1,1,2,2,2)))
        if channels!=(64,128,192,256,320) or blocks!=(1,1,2,2,2): raise ValueError("hierarchical model fixes requested channels/blocks")
        groups=int(_value(cfg,"groupnorm_groups",8)); dropout=float(_value(cfg,"dropout",0.)); decoder_blocks=int(_value(cfg,"decoder_blocks_per_stage",1)); self.assume_full_length=bool(_value(cfg,"assume_full_length",True))
        self.stem=nn.Conv1d(int(_value(cfg,"input_channels",1)),channels[0],7,padding=3)
        self.encoder_stages=nn.ModuleList(ResidualStage1d(c,n,groups,dropout) for c,n in zip(channels,blocks)); self.downsamples=nn.ModuleList(Downsample1d(channels[i],channels[i+1]) for i in range(4))
        heads=int(_value(cfg,"group_set_heads",8)); interaction_dropout=float(_value(cfg,"group_interaction_dropout",0.1))
        self.group_interaction32=GroupSetInteraction(320,heads,interaction_dropout,self.assume_full_length); self.group_interaction64=GroupSetInteraction(256,heads,interaction_dropout,self.assume_full_length)
        self.decoder=nn.ModuleList(UpsampleFuse1d(channels[i],channels[i-1],channels[i-1],decoder_blocks,groups,dropout) for i in range(4,0,-1))
        self.proj_dim=int(_value(cfg,"proj_dim",192)); self.head=nn.Sequential(nn.GroupNorm(_group_count(64,groups),64),nn.SiLU(),nn.Conv1d(64,self.proj_dim,1)); self.register_buffer("emb_temp",torch.tensor(float(_value(cfg,"emb_temp",.07))))
        group=_value(cfg,"group_alignment",{}) or {}
        if bool(_value(group, "use_null", False)):
            raise ValueError("V3 hierarchical_groupwise is no-NULL; set model.group_alignment.use_null=false")
        slots=int(_value(group,"num_slots",512)); slot_dim=int(_value(group,"slot_dim",192))
        self.group_slot_generator=CoarseToFineGroupSlotGenerator(320,256,slots,slot_dim,int(_value(group,"heads",8)),float(_value(group,"dropout",0.)))
        assignment_scale=float(_value(group,"assignment_scale",_value(group,"assignment_scale_init",10.)))
        self.group_assignment_head=CosineNoNullAssignmentHead(
            self.proj_dim,slot_dim,int(_value(group,"assignment_dim",192)),assignment_scale,
            bool(_value(group,"assignment_scale_trainable",False)),
        )
        alpha=float(_value(group,"crf_scale_init",1.))
        if alpha <= 0.0: raise ValueError("crf_scale_init must be positive")
        target_length=int(_value(group,"crf_target_length",512))
        self.group_crf_alpha_raw=nn.Parameter(torch.tensor(math.log(math.expm1(alpha))))
        # Uniform H gives C=1/T.  This makes alpha*log(C)+beta-log(Q)=0.
        self.group_crf_beta=nn.Parameter(torch.tensor(alpha*math.log(float(slots))+math.log(float(target_length))))
        self.num_slots,self.crf_target_length=slots,target_length

    def forward(self,z,mask,edge_index=None):
        if z.ndim!=3 or z.shape!=mask.shape: raise ValueError("expects [B,K,L]")
        batch,curves,length=z.shape
        if length!=512: raise ValueError("hierarchical model fixes canonical L=512")
        flat_mask=mask.reshape(batch*curves,length); x=_masked_zero(self.stem(z.reshape(batch*curves,1,length).masked_fill(flat_mask[:,None],0.)),flat_mask); skips=[]; masks=[]; current=flat_mask
        for level,stage in enumerate(self.encoder_stages):
            x=stage(x,current); skips.append(x); masks.append(current)
            if level<4: current=_resize_mask(current,x.shape[-1]//2); x=self.downsamples[level](x,current)
        mask32=current.reshape(batch,curves,32); f32=self.group_interaction32(x.transpose(1,2).reshape(batch,curves,32,320),mask32); context32=f32.reshape(batch,curves*32,320); context32mask=mask32.reshape(batch,curves*32)
        x=f32.reshape(batch*curves,32,320).transpose(1,2); x=self.decoder[0](x,skips[3],masks[3]); mask64=masks[3].reshape(batch,curves,64); f64=self.group_interaction64(x.transpose(1,2).reshape(batch,curves,64,256),mask64); context64=f64.reshape(batch,curves*64,256); context64mask=mask64.reshape(batch,curves*64)
        x=f64.reshape(batch*curves,64,256).transpose(1,2)
        for decoder_index in range(1,len(self.decoder)): x=self.decoder[decoder_index](x,skips[3-decoder_index],masks[3-decoder_index])
        emb=F.normalize(self.head(x).transpose(1,2).float(),dim=-1).to(x.dtype).reshape(batch,curves,length,self.proj_dim).masked_fill(mask[...,None],0.)
        slots=self.group_slot_generator(context32,context64,context32mask,context64mask); assignment=self.group_assignment_head(emb,slots,mask); alpha=F.softplus(self.group_crf_alpha_raw)
        uniform_emission=-alpha*math.log(float(self.num_slots))+self.group_crf_beta-math.log(float(self.crf_target_length))
        return {"emb":emb,"emb_temp":self.emb_temp,"group_context32":f32,"group_context64":f64,"group_slots":slots,"group_crf_alpha":alpha,"group_crf_beta":self.group_crf_beta,"group_crf_uniform_emission":uniform_emission,**assignment}
