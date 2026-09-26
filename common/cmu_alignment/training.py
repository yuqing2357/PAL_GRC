"""Deterministic full-manifest training, validation, and checkpoint state."""
from __future__ import annotations

import copy
import math
import os
import random
from collections import OrderedDict, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cmu_alignment.loss import exact_low_rank_projector


# D34's prefetcher transfers only the tensors consumed by its train step.
# CMU keeps evaluation/provenance fields in every collated record, but moving
# those fields during training only consumes PCIe bandwidth and pinned memory.
TRAIN_GPU_BATCH_KEYS = frozenset({
    "x", "sequence_valid_mask", "lengths", "q_of_p", "pair_valid",
    "sample_weight", "schedule_weight",
})


class CUDAPrefetcher:
    """Overlap pinned-host → GPU copies with the preceding END update."""
    def __init__(self, loader, device: torch.device):
        self.iterator, self.device = iter(loader), device
        self.stream = torch.cuda.Stream(device=device)
        self.next_batch = None
        self._preload()

    def _preload(self) -> None:
        try:
            cpu_batch = next(self.iterator)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = {
                key: value.to(self.device, non_blocking=True) if key in TRAIN_GPU_BATCH_KEYS and isinstance(value, torch.Tensor) else value
                for key, value in cpu_batch.items()
            }

    def next(self):
        if self.next_batch is None:
            return None
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self.next_batch
        for key in TRAIN_GPU_BATCH_KEYS:
            value=batch.get(key)
            if isinstance(value, torch.Tensor) and value.is_cuda:
                value.record_stream(torch.cuda.current_stream(self.device))
        self._preload()
        return batch


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = .9995):
        # EMA is a plain module on every rank, never a second DDP wrapper.
        # DDP keeps model parameters synchronized after each optimizer step.
        source=model.module if hasattr(model,"module") else model
        self.decay = decay; self.model = copy.deepcopy(source).eval()
        for parameter in self.model.parameters(): parameter.requires_grad_(False)
    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: float) -> None:
        source=model.module if hasattr(model,"module") else model
        # This is the original END/V3 bias-corrected warm start.  A fixed
        # 0.9995 decay is appropriate after thousands of updates, but would
        # leave a 900-step CMU EMA dominated by its random initialization.
        decay=min(self.decay,(1.0+float(step))/(10.0+float(step)))
        for old, new in zip(self.model.state_dict().values(), source.state_dict().values(), strict=True):
            old.lerp_(new.detach(), 1-decay) if old.is_floating_point() else old.copy_(new)
    def state_dict(self): return self.model.state_dict()
    def load_state_dict(self, value): self.model.load_state_dict(value)


def build_optimizer(model, cfg: dict[str, Any]):
    options = {
        "lr": float(cfg.get("lr", 2e-4)),
        "betas": tuple(cfg.get("betas", (.9, .999))),
        "weight_decay": float(cfg.get("weight_decay", 0.0)),
    }
    if bool(cfg.get("fused_optimizer", True)) and torch.cuda.is_available():
        options["fused"] = True
    try:
        return torch.optim.AdamW(model.parameters(), **options)
    except (TypeError, RuntimeError):
        # Older PyTorch builds do not expose fused AdamW.  This is only an
        # execution fallback; the optimizer family and all hyperparameters stay
        # identical.
        options.pop("fused", None)
        return torch.optim.AdamW(model.parameters(), **options)


def build_scheduler(optimizer, cfg: dict[str, Any]):
    total, warm, floor = max(1, int(cfg.get("max_steps", 1))), max(0, int(cfg.get("warmup_steps", 0))), float(cfg.get("min_lr_ratio", .1))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (step+1)/max(1,warm) if step<warm else floor+(1-floor)*.5*(1+math.cos(math.pi*(step-warm)/max(1,total-warm))))


def parent_global_mean_weight(rows: list[dict[str, Any]]) -> float:
    """Mean child weight over the immutable training manifest = parents/children."""
    if not rows: raise ValueError("empty training manifest")
    return len({row["parent_prompt_id"] for row in rows}) / len(rows)


def _projector_scale(progress: float, warm: float, ramp: float) -> float:
    return 0.0 if progress <= warm else min(1.0, (progress-warm)/ramp) if ramp else 1.0


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device: torch.device, amp: str = "off", grad_clip: float = 1.0, ema_decay: float | None = .9995, total_steps: int = 1, warmup_fraction: float = 0.0, ramp_fraction: float = 0.0, ema_step_scale: float = 1.0):
        if ema_step_scale <= 0: raise ValueError("ema_step_scale must be positive")
        self.model,self.criterion,self.opt,self.scheduler,self.device=model,criterion,optimizer,scheduler,device
        self.clip,self.ema,self.total,self.warm,self.ramp,self.step,self.amp,self.ema_step_scale=grad_clip,(ModelEMA(model,ema_decay) if ema_decay else None),max(1,total_steps),warmup_fraction,ramp_fraction,0,amp,float(ema_step_scale)
        self.consumed_batch_in_epoch=0
        self.scaler=torch.amp.GradScaler("cuda",enabled=amp=="fp16" and device.type=="cuda")
    def train_step(self, batch: dict[str, Any], *, materialize_log: bool = True) -> dict[str, float | int]:
        tensor = {
            key: value if value.device == self.device else value.to(self.device, non_blocking=self.device.type == "cuda")
            for key, value in batch.items() if key in TRAIN_GPU_BATCH_KEYS and isinstance(value, torch.Tensor)
        }
        self.model.train(); self.opt.zero_grad(set_to_none=True)
        scale=_projector_scale(self.step/self.total,self.warm,self.ramp); enabled=self.device.type=="cuda" and self.amp in {"bf16","fp16"}; dtype=torch.bfloat16 if self.amp=="bf16" else torch.float16
        with torch.autocast(self.device.type,enabled=enabled,dtype=dtype): output=self.model(tensor["x"],tensor["sequence_valid_mask"],tensor["lengths"])
        with torch.autocast(self.device.type,enabled=False): result=self.criterion(output,tensor,scale,distributed_training=hasattr(self.model,"module"),compute_diagnostics=materialize_log)
        self.scaler.scale(result["total_loss"]).backward(); self.scaler.unscale_(self.opt)
        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.clip); self.scaler.step(self.opt); self.scaler.update(); self.scheduler.step()
        if self.ema: self.ema.update(self.model,(self.step+1)*self.ema_step_scale)
        projector_count=len(batch["k4_group_id"]) if self.criterion.projector_batch_groups is None else min(len(batch["k4_group_id"]),int(self.criterion.projector_batch_groups))
        self.criterion.advance_projector_group_offset(projector_count)
        # Match the D34 continuation semantics: the rotating Group-PSD
        # subset advances only on epochs where that loss is actually active.
        if self.criterion.group_psd_is_active:
            group_psd_count=len(batch["k4_group_id"]) if self.criterion.group_psd_batch_groups is None else min(len(batch["k4_group_id"]),int(self.criterion.group_psd_batch_groups))
            self.criterion.advance_group_psd_group_offset(group_psd_count)
        self.step += 1; self.consumed_batch_in_epoch += 1
        metric_tensors={
            "weighted_total_loss": result["weighted_total_loss"].detach(),
            "weighted_alignment_loss": result["weighted_alignment_loss"].detach(),
            "weighted_projector_loss": result["weighted_projector_loss"].detach(),
            "weighted_projector_contribution": result["weighted_projector_contribution"].detach(),
            "weighted_group_score_loss": result["weighted_group_score_loss"].detach(),
            "weighted_group_score_contribution": result["weighted_group_score_contribution"].detach(),
            "weighted_group_psd_loss": result["weighted_group_psd_loss"].detach(),
            "weighted_group_psd_contribution": result["weighted_group_psd_contribution"].detach(),
            "weighted_similarity_loss": result["weighted_similarity_loss"].detach(),
            "weighted_similarity_contribution": result["weighted_similarity_contribution"].detach(),
        }
        if not materialize_log:
            # Keep normal CUDA steps entirely asynchronous.  Converting any
            # result tensor to a Python float here would synchronize the GPU.
            return {"global_step":self.step,"batch_group_count":len(batch["k4_group_id"]),"L_batch":int(tensor["x"].shape[2]),"_window_metrics":metric_tensors}
        return {"global_step":self.step,"weighted_total_loss":float(metric_tensors["weighted_total_loss"]),"weighted_alignment_loss":float(metric_tensors["weighted_alignment_loss"]),"weighted_projector_loss":float(metric_tensors["weighted_projector_loss"]),"weighted_projector_contribution":float(metric_tensors["weighted_projector_contribution"]),"weighted_group_score_loss":float(metric_tensors["weighted_group_score_loss"]),"weighted_group_score_contribution":float(metric_tensors["weighted_group_score_contribution"]),"weighted_group_psd_loss":float(metric_tensors["weighted_group_psd_loss"]),"weighted_group_psd_contribution":float(metric_tensors["weighted_group_psd_contribution"]),"weighted_similarity_loss":float(metric_tensors["weighted_similarity_loss"]),"weighted_similarity_contribution":float(metric_tensors["weighted_similarity_contribution"]),"unweighted_total_loss":float(result["unweighted_total_loss"].detach()),"unweighted_alignment_loss":float(result["unweighted_alignment_loss"].detach()),"unweighted_projector_loss":float(result["unweighted_projector_loss"].detach()),"unweighted_group_score_loss":float(result["unweighted_group_score_loss"].detach()),"unweighted_group_mean_loss":float(result["unweighted_group_mean_loss"].detach()),"unweighted_group_variance_loss":float(result["unweighted_group_variance_loss"].detach()),"unweighted_group_psd_loss":float(result["unweighted_group_psd_loss"].detach()),"unweighted_similarity_loss":float(result["unweighted_similarity_loss"].detach()),"projector_scale":scale,"projector_stride":int(result["diagnostics"]["projector_stride"]),"projector_groups_used":int(result["diagnostics"]["projector_groups_used"]),"projector_offset":int(result["diagnostics"]["projector_offset"]),"projector_nodes":int(result["diagnostics"]["projector_node_count"]),"group_score_points":int(result["diagnostics"]["group_score_points"]),"group_score_groups":int(result["diagnostics"]["group_score_groups"]),"similarity_points":int(result["diagnostics"]["similarity_points"]),"group_psd_active":int(result["diagnostics"]["group_psd_active"]),"group_psd_groups_used":int(result["diagnostics"]["group_psd_groups_used"]),"group_psd_offset":int(result["diagnostics"]["group_psd_offset"]),"group_psd_positions_used":int(result["diagnostics"]["group_psd_positions_used"]),"group_psd_min_eigenvalue":float(result["diagnostics"]["group_psd_min_eigenvalue"]),"group_psd_negative_eigenvalue_count":float(result["diagnostics"]["group_psd_negative_eigenvalue_count"]),"group_psd_negative_spectral_mass":float(result["diagnostics"]["group_psd_negative_spectral_mass"]),"group_psd_directional_disagreement":float(result["diagnostics"]["group_psd_directional_disagreement"]),"group_psd_offdiagonal_relation_mass":float(result["diagnostics"]["group_psd_offdiagonal_relation_mass"]),"group_psd_total_relation_mass":float(result["diagnostics"]["group_psd_total_relation_mass"]),"group_psd_posterior_entropy":float(result["diagnostics"]["group_psd_posterior_entropy"]),"group_psd_max_posterior":float(result["diagnostics"]["group_psd_max_posterior"]),"group_psd_expected_path_length":float(result["diagnostics"]["group_psd_expected_path_length"]),"group_psd_predicted_overlap_fraction":float(result["diagnostics"]["group_psd_predicted_overlap_fraction"]),"grad_norm":float(norm),"learning_rate":self.opt.param_groups[0]["lr"],"batch_group_count":len(batch["k4_group_id"]),"L_batch":int(tensor["x"].shape[2]),"pair_cell_count":float(result["diagnostics"]["pair_cell_count"]),"_window_metrics":metric_tensors}

    def validate(self, loader, device: torch.device | None = None, use_ema: bool = True) -> dict[str,float|int|str]:
        model=self.ema.model if use_ema and self.ema is not None else self.model
        return {**validate_loader(model,self.criterion,loader,device or self.device),"weights":"ema" if use_ema and self.ema is not None else "online"}


@torch.no_grad()
def validation_parent_rows(model, criterion, loader, device: torch.device) -> dict[str,list[dict[str,float]]]:
    """Return unreduced local observations so DDP can gather before averaging."""
    model.eval(); per_parent: dict[str, list[dict[str,float]]] = defaultdict(list)
    for batch in loader:
        tensor={key:value.to(device,non_blocking=device.type=="cuda") for key,value in batch.items() if isinstance(value,torch.Tensor)}
        output=model(tensor["x"],tensor["sequence_valid_mask"],tensor["lengths"])
        result=criterion(output,tensor,1.0)
        # Pairwise controls set lambda_projector=0.  Do not materialize the
        # K=4 relation diagnostic in that case: it is absent from their loss,
        # checkpoint-selection score, and prediction contract.
        if criterion.lambda_projector != 0.0:
            full_projector=exact_low_rank_projector(output.projector_features,output.sequence_valid_mask,stride=criterion.projector_stride,dtype=criterion.projector_dtype)
        else:
            full_projector=torch.zeros(len(batch["parent_prompt_id"]),device=device,dtype=output.matching_features.dtype)
        for index,parent in enumerate(batch["parent_prompt_id"]):
            # Synthetic schedule entries only exist in DDP training.  They have
            # zero objective weight and are never a validation observation.
            if float(tensor.get("schedule_weight",torch.ones(1,device=device))[index]) == 0.0: continue
            per_parent[parent].append({"alignment":float(result["per_sample_alignment_loss"][index]),"projector":float(full_projector[index]),"similarity":float(result["per_sample_similarity_loss"][index]),"group_psd":float(result["per_sample_group_psd_loss"][index]),"total":float(result["per_sample_alignment_loss"][index]+criterion.lambda_projector*full_projector[index]+criterion.lambda_similarity*result["per_sample_similarity_loss"][index]+criterion.lambda_group_psd*result["per_sample_group_psd_loss"][index])})
    return per_parent


def summarize_validation(per_parent: dict[str,list[dict[str,float]]]) -> dict[str,float|int]:
    """Parent average first, dataset average second; never rank-average metrics."""
    if not per_parent: raise ValueError("empty validation loader")
    parent={key:{metric:float(np.mean([row[metric] for row in values])) for metric in ("alignment","projector","similarity","group_psd","total")} for key,values in per_parent.items()}
    return {"parent_count":len(parent),"subgroup_count":sum(len(x) for x in per_parent.values()),"parent_balanced_alignment_loss":float(np.mean([x["alignment"] for x in parent.values()])),"parent_balanced_projector_loss":float(np.mean([x["projector"] for x in parent.values()])),"parent_balanced_similarity_loss":float(np.mean([x.get("similarity",0.0) for x in parent.values()])),"parent_balanced_group_psd_loss":float(np.mean([x.get("group_psd",0.0) for x in parent.values()])),"parent_balanced_total_loss":float(np.mean([x["total"] for x in parent.values()]))}


@torch.no_grad()
def validate_loader(model, criterion, loader, device: torch.device) -> dict[str, float | int]:
    """Single-process convenience wrapper around the DDP-safe raw rows API."""
    return summarize_validation(validation_parent_rows(model,criterion,loader,device))


def _rng_state() -> dict[str, Any]:
    state={"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state()}
    # One DDP rank owns one CUDA device.  Capturing only that generator avoids
    # having every rank later overwrite every other rank's GPU RNG state.
    if torch.cuda.is_available(): state["cuda_local"]=torch.cuda.get_rng_state(torch.cuda.current_device())
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    if "by_rank" in state:
        import torch.distributed as dist
        rank=dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        state=state["by_rank"][rank]
    # Checkpoints are normally loaded with map_location=cuda:<local_rank>.
    # RNG states are control metadata, not model tensors: torch's setters
    # require CPU ByteTensors even when model/optimizer state is on CUDA.
    cpu_state=state["torch"].detach().cpu() if isinstance(state["torch"],torch.Tensor) else state["torch"]
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(cpu_state)
    if torch.cuda.is_available():
        if "cuda_local" in state:
            cuda_state=state["cuda_local"]
        elif "cuda" in state:
            # Compatibility with checkpoints written before local-device RNG
            # capture.  Restore only this process's current device, never all.
            cuda_state=state["cuda"][torch.cuda.current_device()]
        else:
            cuda_state=None
        if cuda_state is not None:
            cuda_state=cuda_state.detach().cpu() if isinstance(cuda_state,torch.Tensor) else cuda_state
            torch.cuda.set_rng_state(cuda_state,device=torch.cuda.current_device())


def _checkpoint_state(model, trainer: Trainer, epoch: int, config: dict[str, Any], data_hashes: dict[str, str], sampler=None, best_metric: float | None = None) -> dict[str, Any] | None:
    """Collect an exact checkpoint state; only DDP rank zero returns it."""
    sampler_state=None if sampler is None else (sampler.state_dict() if hasattr(sampler,"state_dict") else {"epoch":getattr(sampler,"epoch",0)})
    rng=_rng_state()
    import torch.distributed as dist
    distributed=dist.is_available() and dist.is_initialized()
    if distributed:
        gathered=[None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered,rng)
        rng={"by_rank":gathered}
        if dist.get_rank()!=0:return None
    return {"model_state_dict":model.state_dict(),"ema_state_dict":None if trainer.ema is None else trainer.ema.state_dict(),"optimizer_state_dict":trainer.opt.state_dict(),"scheduler_state_dict":trainer.scheduler.state_dict(),"grad_scaler_state_dict":trainer.scaler.state_dict(),"epoch":epoch,"global_step":trainer.step,"consumed_batch_in_epoch":trainer.consumed_batch_in_epoch,"projector_offset":trainer.criterion.projector_offset_value,"group_psd_offset":trainer.criterion.group_psd_offset_value,"rng":rng,"sampler_state":sampler_state,"resolved_config":config,"data_hashes":data_hashes,"best_metric":best_metric}


def _cpu_clone(value: Any) -> Any:
    """Detach checkpoint content from live GPU/optimizer state before writing."""
    if isinstance(value,torch.Tensor):
        return value.detach().to(device="cpu",copy=True)
    if isinstance(value,np.ndarray):
        return value.copy()
    if isinstance(value,dict):
        return {key:_cpu_clone(item) for key,item in value.items()}
    if isinstance(value,list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value,tuple):
        return tuple(_cpu_clone(item) for item in value)
    return copy.deepcopy(value)


def snapshot_checkpoint(model, trainer: Trainer, epoch: int, config: dict[str, Any], data_hashes: dict[str, str], sampler=None, best_metric: float | None = None) -> dict[str, Any] | None:
    """Return an immutable CPU snapshot suitable for background serialization.

    Every DDP rank participates in RNG collection.  Rank zero then creates the
    CPU snapshot while other ranks may continue toward the next update; no
    background thread ever reads mutable model or optimizer tensors.
    """
    state=_checkpoint_state(model,trainer,epoch,config,data_hashes,sampler,best_metric)
    return None if state is None else _cpu_clone(state)


def _atomic_torch_save(path: str | Path, state: dict[str, Any]) -> None:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        torch.save(state,temporary)
        os.replace(temporary,path)
    finally:
        if temporary.exists(): temporary.unlink()


class AsyncCheckpointWriter:
    """One non-blocking ordered writer for immutable CPU checkpoint snapshots.

    A newer snapshot for the same destination supersedes a queued older one.
    This matters for ``best.pt`` under epoch-level validation: training never
    waits for disk I/O, while the final flush still writes the newest best
    state and every distinct periodic checkpoint destination.
    """
    def __init__(self) -> None:
        self._executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix="cmu-checkpoint")
        self._future: Future[None] | None = None
        self._path: Path | None = None
        self._pending: OrderedDict[Path,dict[str,Any]] = OrderedDict()

    def _start_next(self) -> None:
        if self._future is None and self._pending:
            self._path,state=self._pending.popitem(last=False)
            self._future=self._executor.submit(_atomic_torch_save,self._path,state)

    def _collect_finished(self) -> None:
        if self._future is not None and self._future.done():
            self._future.result()
            self._future,self._path=None,None
        self._start_next()

    def submit(self, path: str | Path, state: dict[str, Any]) -> None:
        """Queue a save without waiting for a previous disk operation."""
        path=Path(path)
        self._collect_finished()
        if self._future is None:
            self._path=path
            self._future=self._executor.submit(_atomic_torch_save,path,state)
        else:
            # Assignment replaces an older queued snapshot for ``path`` while
            # preserving its queue position relative to distinct destinations.
            self._pending[path]=state

    def poll(self) -> None:
        self._collect_finished()

    def flush(self) -> None:
        while self._future is not None or self._pending:
            if self._future is not None:
                self._future.result()
                self._future,self._path=None,None
            self._start_next()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._executor.shutdown(wait=True)


def save_checkpoint(path: str | Path, model, trainer: Trainer, epoch: int, config: dict[str, Any], data_hashes: dict[str, str], sampler=None, best_metric: float | None = None) -> None:
    """Synchronous compatibility API used by tests and explicit callers."""
    state=_checkpoint_state(model,trainer,epoch,config,data_hashes,sampler,best_metric)
    if state is not None:
        _atomic_torch_save(path,state)


def load_checkpoint(path: str | Path, model, trainer: Trainer, sampler=None, map_location="cpu") -> dict[str, Any]:
    state=torch.load(path,map_location=map_location,weights_only=False)
    model.load_state_dict(state["model_state_dict"]); trainer.opt.load_state_dict(state["optimizer_state_dict"]); trainer.scheduler.load_state_dict(state["scheduler_state_dict"]); trainer.scaler.load_state_dict(state["grad_scaler_state_dict"]); trainer.step=int(state["global_step"]);trainer.consumed_batch_in_epoch=int(state.get("consumed_batch_in_epoch",0));trainer.criterion.set_projector_group_offset(int(state.get("projector_offset",0)));trainer.criterion.set_group_psd_group_offset(int(state.get("group_psd_offset",0)))
    if trainer.ema is not None and state.get("ema_state_dict") is not None: trainer.ema.load_state_dict(state["ema_state_dict"])
    if sampler is not None and state.get("sampler_state") is not None:
        sampler.load_state_dict(state["sampler_state"]) if hasattr(sampler,"load_state_dict") else sampler.set_epoch(int(state["sampler_state"]["epoch"]))
        if hasattr(sampler,"cursor"): sampler.cursor=trainer.consumed_batch_in_epoch
    _restore_rng(state["rng"])
    return state
