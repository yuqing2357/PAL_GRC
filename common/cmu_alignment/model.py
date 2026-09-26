"""Variable-length 39-channel hierarchical END model in one file."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _groups(c:int,wanted:int)->int:
    g=min(c,wanted)
    while c%g:g-=1
    return g
def _down_mask(valid:torch.Tensor, output_length:int)->torch.Tensor:
    """Exact Conv1d(k=4,s=2,p=1) valid-prefix propagation: floor(T/2)."""
    count=torch.div(valid.long().sum(-1),2,rounding_mode='floor')
    return torch.arange(output_length,device=valid.device)<count[...,None]
def _zero(x:torch.Tensor,valid:torch.Tensor)->torch.Tensor:return x*valid[:,None].to(x.dtype)
class _Norm(nn.Module):
    def __init__(self,g,c):super().__init__();self.g=g;self.w=nn.Parameter(torch.ones(c));self.b=nn.Parameter(torch.zeros(c))
    def forward(self,x,v):
        B,C,L=x.shape;z=x.reshape(B,self.g,C//self.g,L);m=v[:,None,None].to(x.dtype);d=(m.sum(-1,keepdim=True)*(C//self.g)).clamp_min(1);mean=(z*m).sum((2,3),keepdim=True)/d;var=((z-mean).square()*m).sum((2,3),keepdim=True)/d;return ((z-mean)/torch.sqrt(var+1e-5)).reshape(B,C,L)*self.w[None,:,None]+self.b[None,:,None]
class _Res(nn.Module):
    def __init__(self,c,g,d):super().__init__();self.a=_Norm(_groups(c,g),c);self.b=_Norm(_groups(c,g),c);self.c1=nn.Conv1d(c,c,3,padding=1);self.c2=nn.Conv1d(c,c,3,padding=1);self.d=nn.Dropout(d)
    def forward(self,x,v):
        y=_zero(self.c1(_zero(F.silu(self.a(x,v)),v)),v)
        y=_zero(self.c2(self.d(_zero(F.silu(self.b(y,v)),v))),v)
        return _zero(x+y,v)
class _Stage(nn.Module):
    def __init__(self,c,n,g,d):super().__init__();self.x=nn.ModuleList(_Res(c,g,d) for _ in range(n))
    def forward(self,x,v):
        for f in self.x:x=f(x,v)
        return x
class _Up(nn.Module):
    def __init__(self,a,b,g,d):super().__init__();self.p=nn.Conv1d(a,b,3,padding=1);self.f=nn.Conv1d(2*b,b,1);self.s=_Stage(b,1,g,d)
    def forward(self,x,skip,target,source):
        y=torch.zeros(x.shape[0],x.shape[1],skip.shape[-1],dtype=x.dtype,device=x.device)
        # The former one-row-at-a-time interpolation is mathematically exact
        # but turns a length-bucketed batch into hundreds of tiny GPU kernels.
        # Grouping equal valid lengths preserves the same align_corners=False
        # interpolation while issuing one call per unique (source,target) pair.
        groups={}
        for n,(a,b) in enumerate(zip(source.sum(-1).tolist(),target.sum(-1).tolist(),strict=True)):
            groups.setdefault((a,b),[]).append(n)
        for (a,b), rows in groups.items():
            if a and b:
                selected=torch.as_tensor(rows,device=x.device,dtype=torch.long)
                values=F.interpolate(x.index_select(0,selected)[:,:,:a],size=b,mode='linear',align_corners=False)
                # Under BF16 autocast interpolation may deliberately execute
                # in FP32.  The former slice assignment cast it back to ``y``
                # implicitly; ``index_copy_`` requires that cast explicitly.
                y.index_copy_(0,selected,F.pad(values,(0,skip.shape[-1]-b)).to(dtype=y.dtype))
        return self.s(_zero(self.f(torch.cat((self.p(y),skip),1)),target),target)
class GroupSetInteraction(nn.Module):
    """Member-set attention with an explicit, auditable connectivity control."""
    def __init__(self,c,h,d):super().__init__();self.q=nn.LayerNorm(c);self.k=nn.LayerNorm(c);self.a=nn.MultiheadAttention(c,h,dropout=d,batch_first=True);self.n=nn.LayerNorm(c);self.f=nn.Sequential(nn.Linear(c,4*c),nn.GELU(),nn.Dropout(d),nn.Linear(4*c,c),nn.Dropout(d))
    def forward(self,x,v,*,connectivity="leave_self_out"):
        B,K,L,C=x.shape
        if connectivity not in {"leave_self_out","self_only"}:
            raise ValueError(f"unsupported GroupSetInteraction connectivity: {connectivity}")
        if connectivity=="self_only":
            query=x.reshape(B*K,L,C);query_valid=v.reshape(B*K,L)
            attention=self.a(self.q(query),self.k(query),self.k(query),key_padding_mask=~query_valid,need_weights=False)[0]
            output=query+attention;output=output+self.f(self.n(output))
            return output.reshape(B,K,L,C)*query_valid.reshape(B,K,L)[...,None].to(x.dtype)
        indices=torch.arange(K,device=x.device)
        other_indices=torch.stack([indices[indices!=i] for i in range(K)])
        query=x.reshape(B*K,L,C)
        context=x[:,other_indices].reshape(B*K,(K-1)*L,C)
        query_valid=v.reshape(B*K,L)
        context_valid=v[:,other_indices].reshape(B*K,(K-1)*L)
        attention=self.a(self.q(query),self.k(context),self.k(context),key_padding_mask=~context_valid,need_weights=False)[0]
        output=query+attention
        output=output+self.f(self.n(output))
        return output.reshape(B,K,L,C)*query_valid.reshape(B,K,L)[...,None].to(x.dtype)
class _SlotCrossBlock(nn.Module):
    """Frozen END cross-attention block; only context padding is CMU-specific."""
    def __init__(self,d,h,drop):
        super().__init__();self.qnorm=nn.LayerNorm(d);self.cnorm=nn.LayerNorm(d);self.attn=nn.MultiheadAttention(d,h,dropout=drop,batch_first=True);self.fnorm=nn.LayerNorm(d);self.ffn=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Dropout(drop),nn.Linear(4*d,d),nn.Dropout(drop))
    def forward(self,slots,context,padding=None):
        updated=slots+self.attn(self.qnorm(slots),self.cnorm(context),self.cnorm(context),key_padding_mask=padding,need_weights=False)[0]
        return updated+self.ffn(self.fnorm(updated))

class _Slots(nn.Module):
    """Frozen END coarse/refine slot generator with compact valid CMU contexts."""
    def __init__(self,a,b,n,d,h,drop):
        super().__init__()
        if d%h:raise ValueError('slot_dim must divide slot heads')
        self.q=nn.Parameter(torch.empty(n,d));nn.init.normal_(self.q,std=d**-.5)
        self.a=nn.Linear(a,d);self.b=nn.Linear(b,d);self.coarse=_SlotCrossBlock(d,h,drop);self.refine=_SlotCrossBlock(d,h,drop);self.n=nn.LayerNorm(d)
    def forward(self,a,b,av,bv):
        if (~av).all(dim=1).any() or (~bv).all(dim=1).any():
            raise ValueError('each K4 group needs valid coarse/refine context')
        slots=self.q[None].expand(a.shape[0],-1,-1)
        slots=self.coarse(slots,self.a(a),padding=~av)
        slots=self.refine(slots,self.b(b),padding=~bv)
        return self.n(slots)
class _Membership(nn.Module):
    def __init__(self,e,s,a,scale):super().__init__();self.e=nn.Linear(e,a,bias=False);self.s=nn.Linear(s,a,bias=False);self.register_buffer('log_scale',torch.tensor(math.log(scale)))
    def forward(self,e,s,v):
        logits=self.log_scale.exp()*torch.einsum('bkld,bsd->bkls',F.normalize(self.e(e),dim=-1),F.normalize(self.s(s),dim=-1));return torch.softmax(logits,-1)*v[...,None].to(logits.dtype)
@dataclass
class ENDModelOutput:
    # Direct cosine matching intentionally has no membership tensor.  Keeping
    # this explicit makes a silent slot/membership fallback impossible.
    memberships:torch.Tensor|None;sequence_valid_mask:torch.Tensor;lengths:torch.Tensor;group_crf_alpha:torch.Tensor|None;group_crf_beta:torch.Tensor|None;group_crf_temperature:torch.Tensor|None;group_crf_gamma:torch.Tensor|None;embeddings:torch.Tensor;scale_lengths:tuple[torch.Tensor,...];matching_features:torch.Tensor;matching_mode:str;projector_features:torch.Tensor
class HierarchicalGroupwiseEND(nn.Module):
    def __init__(self,cfg:dict):
        super().__init__();m=cfg.get('model',cfg)
        if int(m.get('K',4))!=4 or int(m.get('input_channels',39))!=39:raise ValueError('CMU END fixes K=4 and MFCC39')
        c=tuple(m.get('channels',(64,128,192,256,320)));b=tuple(m.get('blocks_per_stage',(1,1,2,2,2)))
        if c not in {(32,64,96,128,160),(48,96,144,192,240),(64,128,192,256,320)} or b!=(1,1,2,2,2):raise ValueError('END uses frozen Tiny/Base/Large capacity grids and five stages')
        self.groupwise_mode=str(m.get('groupwise_mode','end'))
        valid_modes={'end','interaction_only_fixed_slots','independent_pairwise','interaction_direct_cosine','independent_pairwise_direct_cosine'}
        if self.groupwise_mode not in valid_modes:raise ValueError(f'groupwise_mode must be one of {sorted(valid_modes)}')
        g=int(m.get('groupnorm_groups',8));d=float(m.get('dropout',0));self.c=c;self.stem=nn.Conv1d(39,c[0],7,padding=3);self.enc=nn.ModuleList(_Stage(x,n,g,d) for x,n in zip(c,b));self.down=nn.ModuleList(nn.Conv1d(c[i],c[i+1],4,stride=2,padding=1) for i in range(4));h=int(m.get('group_set_heads',8));self.dec=nn.ModuleList(_Up(c[i],c[i-1],g,d) for i in range(4,0,-1));self.dim=int(m.get('proj_dim',192));self.hn=_Norm(_groups(c[0],g),c[0]);self.head=nn.Conv1d(c[0],self.dim,1)
        direct=self.groupwise_mode in {'interaction_direct_cosine','independent_pairwise_direct_cosine'}
        if direct and 'group_alignment' in m:raise ValueError('direct-cosine modes must not declare group_alignment / slot configuration')
        self.crf_emission_mode=str(cfg.get('alignment',{}).get('crf_emission_mode','legacy'))
        directional_modes={'directional_rowmean_log','directional_rowcol_softmax'}
        directional_mode=self.crf_emission_mode in directional_modes
        if directional_mode and direct:
            raise ValueError('directional CRF emission modes are defined for the membership Interaction-only route')
        a=m.get('group_alignment',{});slots=int(a.get('num_slots',512));sd=int(a.get('slot_dim',192));
        if bool(a.get('assignment_scale_trainable',False)):raise ValueError('formal CMU END fixes assignment_scale_trainable=false')
        reference=(cfg.get('alignment',{}).get('crf_reference_length') if isinstance(cfg,dict) else None) or a.get('crf_reference_length')
        if reference is None:raise ValueError('alignment.crf_reference_length is required')
        # The one-block direct-cosine control retains only the coarse
        # leave-self-out interaction (``i1``), before the first decoder up
        # block.  It is intentionally restricted to the slots-free direct
        # route so END and fixed-slot semantics remain frozen at two blocks.
        self.group_interaction_blocks=int(m.get('group_interaction_blocks',2))
        self.cross_sequence_communication=bool(m.get('cross_sequence_communication',True))
        if self.groupwise_mode in {'end','interaction_only_fixed_slots'} and self.group_interaction_blocks != 2:
            raise ValueError('END and interaction_only_fixed_slots require exactly two GroupSetInteraction blocks')
        if self.groupwise_mode=='interaction_direct_cosine' and self.group_interaction_blocks not in {1,2}:
            raise ValueError('interaction_direct_cosine requires one or two GroupSetInteraction blocks')
        if self.groupwise_mode in {'end','interaction_only_fixed_slots','interaction_direct_cosine'}:
            self.i1=GroupSetInteraction(c[-1],h,float(m.get('group_interaction_dropout',.1)))
            if self.group_interaction_blocks == 2:
                self.i2=GroupSetInteraction(c[-2],h,float(m.get('group_interaction_dropout',.1)))
        if self.groupwise_mode=='end':
            self.slots=_Slots(c[-1],c[-2],slots,sd,int(a.get('heads',8)),float(a.get('dropout',0)))
        if self.groupwise_mode not in {'end','interaction_direct_cosine','independent_pairwise_direct_cosine'}:
            # This codebook is learned but input-independent.  In the
            # interaction-only mode, other members can enter only via i1/i2.
            self.fixed_slots=nn.Parameter(torch.empty(slots,sd));nn.init.normal_(self.fixed_slots,std=sd**-.5)
        if not direct:
            self.member=_Membership(self.dim,sd,int(a.get('assignment_dim',192)),float(a.get('assignment_scale',10)))
        if directional_mode:
            # This mode deliberately has no alpha parameter.  Temperature is
            # positive by construction and gamma remains the per-included-row
            # path bias.  The raw Gram affinity is normalized later, per
            # directed view, before either value enters the CRF.
            temperature_min=float(m.get('group_alignment',{}).get('crf_temperature_min',1e-3))
            temperature_init=float(m.get('group_alignment',{}).get('crf_temperature_init',1.0))
            if temperature_min <= 0 or temperature_init <= temperature_min:
                raise ValueError('crf_temperature_init must exceed positive crf_temperature_min')
            unconstrained=math.log(math.expm1(temperature_init-temperature_min))
            self.crf_temperature_raw=nn.Parameter(torch.tensor(unconstrained))
            self.register_buffer('crf_temperature_min',torch.tensor(temperature_min))
            self.gamma=nn.Parameter(torch.tensor(float(m.get('group_alignment',{}).get('crf_gamma_init',0.0))))
            self.alpha=None;self.beta=None
        elif not direct:
            initial_alpha=float(a.get('crf_scale_init',1));initial_beta=initial_alpha*math.log(slots)+math.log(float(reference))
        else:
            direct_cfg=m.get('direct_matching',{})
            # Original membership beta contains alpha*log(num_slots)+
            # log(L_ref).  Direct cosine removes only the slot term: keeping
            # log(L_ref) makes a cosine-zero emission neutral after the CRF's
            # -log(L_ref) calibration and lets nonempty paths compete with
            # the explicit no-overlap candidate.
            initial_alpha=float(direct_cfg.get('crf_scale_init',1));initial_beta=float(direct_cfg.get('crf_bias_init',0))
        if not directional_mode:
            self.alpha=nn.Parameter(torch.tensor(math.log(math.expm1(initial_alpha))));self.beta=nn.Parameter(torch.tensor(initial_beta))
    def forward(self,x,sequence_valid_mask,lengths):
        if x.ndim!=4 or x.shape[1]!=4 or x.shape[-1]!=39 or sequence_valid_mask.shape!=x.shape[:3] or lengths.shape!=x.shape[:2]:raise ValueError('expects [B,4,L,39], valid and lengths')
        # Every temporal operator below assumes a contiguous valid prefix.  Do
        # not silently accept a sparse mask: that would make length propagation
        # and the down-sampled masks disagree with the acoustic sequence.
        if not torch.equal(sequence_valid_mask.sum(-1).to(lengths.dtype),lengths):raise ValueError('sequence_valid_mask counts must equal lengths')
        prefix=torch.arange(x.shape[2],device=x.device)[None,None,:] < lengths[...,None]
        if not torch.equal(sequence_valid_mask,prefix):raise ValueError('sequence_valid_mask must be a contiguous valid prefix')
        B,K,L,_=x.shape;v=sequence_valid_mask;cur=v.reshape(B*K,L);y=_zero(self.stem(x.permute(0,1,3,2).reshape(B*K,39,L)),cur);skip=[];mask=[]
        for level,stage in enumerate(self.enc):
            y=stage(y,cur);skip.append(y);mask.append(cur)
            if level<4:
                y=self.down[level](y);cur=_down_mask(cur,y.shape[-1]);y=_zero(y,cur)
        if self.groupwise_mode in {'end','interaction_only_fixed_slots','interaction_direct_cosine'}:
            connectivity="leave_self_out" if self.cross_sequence_communication else "self_only"
            coarse=cur.reshape(B,K,-1);z=self.i1(y.transpose(1,2).reshape(B,K,-1,self.c[-1]),coarse,connectivity=connectivity);ctx1=z.reshape(B,-1,self.c[-1]);ctx1v=coarse.reshape(B,-1);y=self.dec[0](z.reshape(B*K,-1,self.c[-1]).transpose(1,2),skip[3],mask[3],cur)
            if self.group_interaction_blocks == 2:
                mid=mask[3].reshape(B,K,-1);z=self.i2(y.transpose(1,2).reshape(B,K,-1,self.c[-2]),mid,connectivity=connectivity);ctx2=z.reshape(B,-1,self.c[-2]);ctx2v=mid.reshape(B,-1);y=z.reshape(B*K,-1,self.c[-2]).transpose(1,2)
                if self.groupwise_mode=='end':slot_context=self.slots(ctx1,ctx2,ctx1v,ctx2v)
            if self.groupwise_mode=='interaction_only_fixed_slots':slot_context=self.fixed_slots[None].expand(B,-1,-1)
        elif self.groupwise_mode=='independent_pairwise_direct_cosine':
            # The slots-free pairwise control deliberately performs the same
            # per-utterance encoder/decoder computation with no K=4 mixing.
            y=self.dec[0](y,skip[3],mask[3],cur)
        else:
            y=self.dec[0](y,skip[3],mask[3],cur);slot_context=self.fixed_slots[None].expand(B,-1,-1)
        source=mask[3]
        for i in range(1,4):target=mask[3-i];y=self.dec[i](y,skip[3-i],target,source);source=target
        emb=F.normalize(self.head(F.silu(self.hn(y,mask[0]))).transpose(1,2).float(),dim=-1).reshape(B,K,L,self.dim)*v[...,None]
        if self.groupwise_mode in {'interaction_direct_cosine','independent_pairwise_direct_cosine'}:
            # This factor induces the nonnegative direct-cosine affinity
            # R=(1+E E^T)/2 on valid frames.  It is only consumed when a
            # direct-affinity Projector is enabled; matching remains raw
            # cosine and never passes through this representation.
            factor=torch.cat((emb,v[...,None].to(emb.dtype)),dim=-1)*math.sqrt(.5)
            return ENDModelOutput(None,v,lengths,F.softplus(self.alpha),self.beta,None,None,emb,tuple(m.sum(-1).reshape(B,K) for m in mask),emb,'direct_cosine',factor)
        H=self.member(emb,slot_context,v)
        if self.crf_emission_mode in {'directional_rowmean_log','directional_rowcol_softmax'}:
            temperature=self.crf_temperature_min+F.softplus(self.crf_temperature_raw)
            return ENDModelOutput(H,v,lengths,None,None,temperature,self.gamma,emb,tuple(m.sum(-1).reshape(B,K) for m in mask),H,'membership_log',H)
        return ENDModelOutput(H,v,lengths,F.softplus(self.alpha),self.beta,None,None,emb,tuple(m.sum(-1).reshape(B,K) for m in mask),H,'membership_log',H)
