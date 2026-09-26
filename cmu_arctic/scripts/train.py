#!/usr/bin/env python3
"""Formal fixed-K4 CMU trainer. DDP shards pre-generated K4 rows only."""
from __future__ import annotations
import argparse, hashlib, json, shutil, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

# A stale editable install from the predecessor project may expose the same
# ``cmu_alignment`` package name.  Formal entry points must always run this
# checkout's source tree, independently of the caller's PYTHONPATH.
_SRC = Path(__file__).resolve().parents[2] / "common"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cmu_alignment.core import all_gather_object, append_jsonl, configure_cuda_performance, distributed_cleanup, distributed_context, load_yaml, prepare_run, run_environment, save_yaml, seed_everything, worker_seed
from cmu_alignment.data import DistributedLengthBucketBatchSampler, K4FullUtteranceAlignmentDataset, LengthBucketBatchSampler, collate_k4_full_utterance, load_crf_reference_length
from cmu_alignment.loss import ENDCriterion
from cmu_alignment.model import HierarchicalGroupwiseEND
from cmu_alignment.training import AsyncCheckpointWriter, CUDAPrefetcher, Trainer, build_optimizer, build_scheduler, load_checkpoint, parent_global_mean_weight, save_checkpoint, snapshot_checkpoint, summarize_validation, validation_parent_rows
from cmu_alignment.evaluation import evaluate_batch_d34, summarize_d34_index_metrics

def checkpoint_sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1 << 20),b""): digest.update(block)
    return digest.hexdigest()

def model_state_sha256(model: torch.nn.Module) -> str:
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf8"));digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def normalized_model_config(config: dict) -> dict:
    """Return the architecture contract with explicit defaults materialized.

    Older production checkpoints omit ``cross_sequence_communication`` because
    the model default is ``True``.  A controlled E3 config may spell that same
    default out.  Those two encodings instantiate identical parameter graphs
    and must not be treated as different architectures.
    """
    model = dict(config.get("model", {}))
    model.setdefault("cross_sequence_communication", True)
    return model


def architecture_matches(source_config: dict, target_config: dict) -> bool:
    return (
        normalized_model_config(source_config) == normalized_model_config(target_config)
        and source_config.get("alignment") == target_config.get("alignment")
    )


def validate_group_psd_continuation(source: dict, target: dict, *, checkpoint: str) -> None:
    """Reject a continuation that silently changes the frozen CRF experiment."""
    source_cfg=source.get("resolved_config")
    if not isinstance(source_cfg,dict):
        raise RuntimeError("Group-PSD continuation requires a CMU checkpoint with resolved_config")
    if not architecture_matches(source_cfg, target):
        raise RuntimeError("Group-PSD continuation requires architecture- and CRF-identical source weights")
    source_loss, target_loss=source_cfg.get("loss",{}),target.get("loss",{})
    fixed=("lambda_projector","align_corridor_radius","allow_no_overlap_output","membership_null_slot","variable_pair_shape","target_length_calibration","segment_existence_bias","zero_radius_continuous_target_policy","lambda_similarity","lambda_group_score")
    if any(source_loss.get(key)!=target_loss.get(key) for key in fixed):
        raise RuntimeError("Group-PSD continuation may change only Group-PSD controls")
    if float(source_loss.get("group_psd_weight",0.0)) != 0.0 or float(target_loss.get("group_psd_weight",0.0)) <= 0.0:
        raise RuntimeError("Group-PSD continuation must start from CRF-only weights and enable a positive Group-PSD weight")
    source_runtime=source_cfg.get("runtime",{})
    for key in ("experiment_seed","max_steps","per_rank_batch_size","world_size"):
        if source_runtime.get(key)!=target.get("runtime",{}).get(key):
            raise RuntimeError(f"Group-PSD continuation must retain source runtime {key}: {checkpoint}")

def loader(dataset, sampler, workers, pin, prefetch_factor):
    args=dict(batch_sampler=sampler,num_workers=workers,pin_memory=pin,collate_fn=collate_k4_full_utterance,worker_init_fn=worker_seed)
    if workers: args.update(persistent_workers=True,prefetch_factor=prefetch_factor)
    return DataLoader(dataset,**args)

def criterion(cfg, rows):
    groups=cfg["loss"].get("projector_batch_groups","all")
    psd_groups=cfg["loss"].get("group_psd_batch_groups","all")
    return ENDCriterion(corridor_radius=cfg["loss"]["align_corridor_radius"],lambda_projector=cfg["loss"]["lambda_projector"],global_mean_sample_weight=parent_global_mean_weight(rows),crf_reference_length=cfg["alignment"]["crf_reference_length"],projector_stride=cfg["loss"].get("projector_stride",1),projector_batch_groups=None if groups in (None,"all") else int(groups),projector_dtype=torch.float32,allow_no_overlap_output=bool(cfg["loss"].get("allow_no_overlap_output",True)),emission_mode=str(cfg["alignment"].get("crf_emission_mode","legacy")),zero_radius_continuous_target_policy=str(cfg["loss"].get("zero_radius_continuous_target_policy","error")),lambda_group_score=float(cfg["loss"].get("lambda_group_score",0.0)),group_score_radius=int(cfg["loss"].get("group_score_radius",2)),lambda_group_variance=float(cfg["loss"].get("lambda_group_variance",1.0)),group_score_variance_epsilon=float(cfg["loss"].get("group_score_variance_epsilon",1e-4)),lambda_similarity=float(cfg["loss"].get("lambda_similarity",0.0)),similarity_target_radius=int(cfg["loss"].get("similarity_target_radius",0)),similarity_target_sigma=float(cfg["loss"].get("similarity_target_sigma",1.0)),similarity_eps=float(cfg["alignment"].get("similarity_eps",1e-6)),normalization_eps=float(cfg["alignment"].get("normalization_eps",1e-8)),lambda_group_psd=float(cfg["loss"].get("group_psd_weight",0.0)),group_psd_num_bins=int(cfg["loss"].get("group_psd_num_positions",cfg["loss"].get("group_psd_num_bins",32))),group_psd_position_fraction=cfg["loss"].get("group_psd_position_fraction",None),group_psd_batch_groups=None if psd_groups in (None,"all") else int(psd_groups),group_psd_enabled=cfg["loss"].get("group_psd_enabled",None),group_psd_start_epoch=int(cfg["loss"].get("group_psd_start_epoch",1)))

def distributed_validation(model, loss, loader_, device, ddp):
    merged={}
    for shard in all_gather_object(validation_parent_rows(model,loss,loader_,device),ddp):
        for parent, rows in shard.items():
            # K4 children of a single parent may legitimately be assigned to
            # distinct ranks.  Aggregate children first; parent balancing is
            # performed only after this global merge.
            merged.setdefault(parent,[]).extend(rows)
    return summarize_validation(merged)

def distributed_decoded_validation(model, loader_, device, reference, ddp, expected_group_count, *, allow_no_overlap_output=True, emission_mode="legacy", similarity_eps=1e-6, normalization_eps=1e-8):
    """Frozen validation-only decoding with the formal evaluator's metrics."""
    local_pairs=[];local_triples=[];local_errors=[]
    for batch in loader_:
        result=evaluate_batch_d34(model,batch,device,reference,allow_no_overlap_output=allow_no_overlap_output,emission_mode=emission_mode,similarity_eps=similarity_eps,normalization_eps=normalization_eps)
        # Retain scalar rows only: decoded paths are not needed for checkpoint
        # selection, whereas raw pointwise triple errors are needed for exact
        # D34-style pooled MAE/P95.
        local_pairs.extend({key:value for key,value in row.items() if key!="predicted_q"} for row in result.pair_rows)
        local_triples.extend(result.triple_rows)
        if len(result.transitive_errors): local_errors.append(result.transitive_errors)
    local={"pairs":local_pairs,"triples":local_triples,"errors":np.concatenate(local_errors) if local_errors else np.empty(0,dtype=np.float32)}
    shards=all_gather_object(local,ddp)
    rows=[row for shard in shards for row in shard["pairs"]]
    triples=[row for shard in shards for row in shard["triples"]]
    errors=np.concatenate([shard["errors"] for shard in shards if len(shard["errors"])]) if any(len(shard["errors"]) for shard in shards) else np.empty(0,dtype=np.float32)
    if len(rows) != expected_group_count * 12:
        raise RuntimeError(f"incomplete decoded validation pair coverage: {len(rows)} != {expected_group_count * 12}")
    if len(triples) != expected_group_count * 24:
        raise RuntimeError(f"incomplete decoded validation triple coverage: {len(triples)} != {expected_group_count * 24}")
    return {
        "decoded_validation_subgroup_pairs":len(rows),
        "decoded_validation_ordered_triples":len(triples),
        "decoded_validation_parent_count":len({row["parent_prompt_id"] for row in rows}),
        **summarize_d34_index_metrics(rows,triples,errors,groups=expected_group_count),
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-root",required=True);p.add_argument("--feature-root",required=True);p.add_argument("--model-config",default="configs/model/end_cmu.yaml");p.add_argument("--align-corridor-radius",type=float);p.add_argument("--zero-radius-continuous-target-policy",choices=("error","nearest_integer"),default=None);p.add_argument("--group-psd-weight",type=float,default=None)
    p.add_argument("--run-root",default=".");p.add_argument("--run-type",default="main",choices=("main","smoke","debug","benchmark"));p.add_argument("--run-id",required=True);p.add_argument("--overwrite",action="store_true");p.add_argument("--seed",type=int,default=20260916)
    p.add_argument("--max-steps",type=int,default=10);p.add_argument("--per-rank-batch-size",type=int,default=1);p.add_argument("--num-workers",type=int,default=4);p.add_argument("--prefetch-factor",type=int,default=2);p.add_argument("--preload-train-to-memory",action="store_true");p.add_argument("--preload-val-to-memory",action="store_true");p.add_argument("--decoded-validation",action="store_true");p.add_argument("--skip-validation",action="store_true");p.add_argument("--async-checkpoint",action="store_true");p.add_argument("--amp",choices=("off","bf16","fp16"),default="bf16");p.add_argument("--no-pin-memory",action="store_true");p.add_argument("--resume");p.add_argument("--resume-mode",choices=("exact","group_psd_continuation","controlled_continuation","staged_continuation"),default="exact");p.add_argument("--e1-condition",choices=("pal","pal_grc"),default=None);p.add_argument("--staged-condition",choices=("pal","pal_grc","pal_direction"),default=None);p.add_argument("--val-every",type=int,default=100);p.add_argument("--early-stop-patience",type=int,default=0);p.add_argument("--early-stop-min-delta",type=float,default=0.0);p.add_argument("--early-stop-min-epochs",type=int,default=0);p.add_argument("--no-latest-checkpoint-on-validation",action="store_true");p.add_argument("--checkpoint-every-epochs",type=int,default=0);p.add_argument("--progress-every",type=int,default=1);p.add_argument("--learning-rate",type=float,default=2e-4);p.add_argument("--warmup-steps",type=int,default=0);p.add_argument("--min-lr-ratio",type=float,default=.1);p.add_argument("--grad-clip-norm",type=float,default=1.0);p.add_argument("--ema-decay",type=float,default=.9995);p.add_argument("--ema-step-scale",type=float,default=1.0);p.add_argument("--projector-batch-groups",type=int,default=None);p.add_argument("--projector-warmup-steps",type=int,default=None);p.add_argument("--projector-ramp-steps",type=int,default=None);p.add_argument("--limit-train-groups",type=int);p.add_argument("--ddp",action="store_true")
    # Keep the continuation protocol identifier small and stable in manifests.
    p.add_argument("--controlled-experiment",choices=("E1","E2","E3"),default="E1")
    p.add_argument("--controlled-condition",default=None)
    p.add_argument("--stop-after-steps",type=int,default=None,help="E2 pretraining-only stop; keeps the full --max-steps LR schedule.")
    a=p.parse_args()
    if a.prefetch_factor < 1: raise ValueError("prefetch-factor must be positive")
    if a.val_every < 1: raise ValueError("val-every must be positive")
    if a.seed < 0: raise ValueError("seed must be non-negative")
    if a.early_stop_patience < 0: raise ValueError("early-stop-patience must be non-negative")
    if a.early_stop_min_delta < 0: raise ValueError("early-stop-min-delta must be non-negative")
    if a.early_stop_min_epochs < 0: raise ValueError("early-stop-min-epochs must be non-negative")
    if a.checkpoint_every_epochs < 0: raise ValueError("checkpoint-every-epochs must be non-negative")
    if a.ema_step_scale <= 0: raise ValueError("ema-step-scale must be positive")
    if a.projector_batch_groups is not None and a.projector_batch_groups < 1: raise ValueError("projector-batch-groups must be positive")
    if a.skip_validation and (a.preload_val_to_memory or a.decoded_validation): raise ValueError("validation options cannot be combined with --skip-validation")
    rank,world,_,device=distributed_context(a.ddp)
    checkpoint_writer=None
    try:
        configure_cuda_performance(main_threads=2)
        if rank==0: print("[CMU train] CUDA/DDP initialized; loading frozen manifests",flush=True)
        cfg=load_yaml(a.model_config);reference=load_crf_reference_length(a.data_root)
        if int(cfg["alignment"]["crf_reference_length"])!=reference: raise ValueError("config L_ref disagrees with frozen data metadata")
        if a.align_corridor_radius is not None:
            if a.align_corridor_radius < 0: raise ValueError("align-corridor-radius must be non-negative")
            cfg["loss"]["align_corridor_radius"]=float(a.align_corridor_radius)
        if a.zero_radius_continuous_target_policy is not None:
            cfg["loss"]["zero_radius_continuous_target_policy"]=a.zero_radius_continuous_target_policy
        if a.group_psd_weight is not None:
            if a.group_psd_weight < 0: raise ValueError("group-psd-weight must be non-negative")
            cfg["loss"]["group_psd_weight"] = float(a.group_psd_weight)
        if a.projector_batch_groups is not None: cfg["loss"]["projector_batch_groups"]=a.projector_batch_groups
        projector_warmup=int(cfg["loss"].get("projector_warmup_steps",0) if a.projector_warmup_steps is None else a.projector_warmup_steps)
        projector_ramp=int(cfg["loss"].get("projector_ramp_steps",0) if a.projector_ramp_steps is None else a.projector_ramp_steps)
        if projector_warmup < 0 or projector_ramp < 0: raise ValueError("projector warmup/ramp steps must be non-negative")
        cfg["loss"].update({"projector_warmup_steps":projector_warmup,"projector_ramp_steps":projector_ramp})
        cfg["runtime"]={"experiment_seed":a.seed,"train_sampler_seed":a.seed,"validation_sampler_seed":a.seed+1,"max_steps":a.max_steps,"per_rank_batch_size":a.per_rank_batch_size,"world_size":world,"num_workers":a.num_workers,"prefetch_factor":a.prefetch_factor,"preload_train_to_memory":a.preload_train_to_memory,"preload_val_to_memory":a.preload_val_to_memory,"pin_memory":not a.no_pin_memory,"amp":a.amp,"ddp":a.ddp,"skip_validation":a.skip_validation,"async_checkpoint":a.async_checkpoint,"curriculum":False,"full_fixed_manifest_from_epoch0":True,"fused_optimizer":True,"lr":a.learning_rate,"warmup_steps":a.warmup_steps,"min_lr_ratio":a.min_lr_ratio,"grad_clip_norm":a.grad_clip_norm,"ema_decay":a.ema_decay,"ema_step_scale":a.ema_step_scale,"ema_update_rule":"min(ema_decay, (1 + effective_step) / (10 + effective_step))","validation_every_steps":a.val_every,"early_stop_monitor":"parent_balanced_alignment_loss_ema","early_stop_patience_epochs":a.early_stop_patience,"early_stop_min_delta":a.early_stop_min_delta,"early_stop_min_epochs":a.early_stop_min_epochs,"latest_checkpoint_on_validation":not a.no_latest_checkpoint_on_validation,"projector_warmup_steps":projector_warmup,"projector_ramp_steps":projector_ramp,"resume_mode":a.resume_mode,"controlled_experiment":a.controlled_experiment,"controlled_condition":a.controlled_condition or a.e1_condition or a.staged_condition,"controlled_fixed_budget":bool(a.e1_condition)}
        if a.resume_mode == "group_psd_continuation":
            if not a.resume:
                raise ValueError("group_psd_continuation requires --resume epoch-10 checkpoint")
            validate_group_psd_continuation(torch.load(a.resume,map_location="cpu",weights_only=False),cfg,checkpoint=a.resume)
        if a.resume_mode == "controlled_continuation":
            if not a.resume or a.e1_condition is None:
                raise ValueError("controlled_continuation requires --resume and --e1-condition")
            source=torch.load(a.resume,map_location="cpu",weights_only=False)
            source_cfg=source.get("resolved_config")
            if not isinstance(source_cfg,dict) or not architecture_matches(source_cfg, cfg):
                raise RuntimeError("E1 controlled continuation requires an architecture-identical common fork checkpoint")
            # E1 deliberately permits only the Group-PSD controls to differ.
            source_loss,target_loss=source_cfg.get("loss",{}),cfg.get("loss",{})
            invariant=("lambda_projector","align_corridor_radius","allow_no_overlap_output","membership_null_slot","variable_pair_shape","target_length_calibration","segment_existence_bias","zero_radius_continuous_target_policy","lambda_similarity","lambda_group_score")
            if any(source_loss.get(key)!=target_loss.get(key) for key in invariant):
                raise RuntimeError("E1 branches may differ only in Group-PSD controls")
            if float(source_loss.get("group_psd_weight",0.0)) != 0.0:
                raise RuntimeError("E1 fork must precede Group-PSD updates")
            expected_positive=a.e1_condition == "pal_grc"
            actual_positive=float(target_loss.get("group_psd_weight",0.0)) > 0.0 and bool(target_loss.get("group_psd_enabled",False))
            if actual_positive != expected_positive:
                raise RuntimeError("E1 condition and Group-PSD enablement disagree")
        if a.resume_mode == "staged_continuation":
            if not a.resume or a.staged_condition is None:
                raise ValueError("staged_continuation requires --resume and --staged-condition")
            source=torch.load(a.resume,map_location="cpu",weights_only=False)
            source_cfg=source.get("resolved_config")
            if not isinstance(source_cfg,dict) or not architecture_matches(source_cfg, cfg):
                raise RuntimeError("staged continuation requires an architecture-identical common fork checkpoint")
            source_loss,target_loss=source_cfg.get("loss",{}),cfg.get("loss",{})
            invariant=("lambda_projector","align_corridor_radius","allow_no_overlap_output","membership_null_slot","variable_pair_shape","target_length_calibration","segment_existence_bias","zero_radius_continuous_target_policy","lambda_similarity","lambda_group_score")
            if any(source_loss.get(key)!=target_loss.get(key) for key in invariant):
                raise RuntimeError("staged branches may differ only in Group-PSD controls")
            if float(source_loss.get("group_psd_weight",0.0)) != 0.0:
                raise RuntimeError("staged fork must precede Group-PSD updates")
            expected_positive=a.staged_condition in ("pal_grc","pal_direction")
            actual_positive=float(target_loss.get("group_psd_weight",0.0)) > 0.0 and bool(target_loss.get("group_psd_enabled",False))
            if actual_positive != expected_positive:
                raise RuntimeError("staged condition and Group-PSD enablement disagree")
        out=prepare_run(a.run_root,a.run_type,a.run_id,a.overwrite) if rank==0 else Path(a.run_root)/"runs"/a.run_type/a.run_id
        if a.ddp:
            import torch.distributed as dist;dist.barrier()
        if rank==0:
            save_yaml(out/"resolved_config.yaml",cfg);(out/"command.txt").write_text(" ".join(sys.argv)+"\n");(out/"environment.json").write_text(json.dumps(run_environment(world,device),indent=2)+"\n")
        seed_everything(a.seed+rank)
        train=K4FullUtteranceAlignmentDataset(a.data_root,a.feature_root,"train")
        val=None if a.skip_validation else K4FullUtteranceAlignmentDataset(a.data_root,a.feature_root,"val")
        if a.limit_train_groups is not None:
            if not 1<=a.limit_train_groups<=len(train.rows):raise ValueError("limit-train-groups must be within the frozen train manifest")
            train.rows=train.rows[:a.limit_train_groups]
        def preload(dataset, enabled, label):
            if not enabled:return
            if rank==0:
                print(f"[CMU train] preloading immutable {label} features/GT into RAM",flush=True)
                dataset.preload_to_memory()
            if a.ddp:
                import torch.distributed as dist;dist.barrier()
            if rank!=0: dataset.preload_to_memory()
            summaries=all_gather_object(dataset.cache_summary(),a.ddp)
            if rank==0: print(f"[CMU train] {label} RAM cache ready: {summaries[0]}",flush=True)
            if a.ddp:
                import torch.distributed as dist;dist.barrier()
        preload(train,a.preload_train_to_memory,"train")
        if val is not None: preload(val,a.preload_val_to_memory,"validation")
        train_lengths=[max(row["native_lengths"]) for row in train.rows]
        train_sampler=DistributedLengthBucketBatchSampler(train_lengths,a.per_rank_batch_size,world,rank,a.seed,True) if a.ddp else LengthBucketBatchSampler(train_lengths,a.per_rank_batch_size,a.seed)
        val_lengths=[] if val is None else [max(row["native_lengths"]) for row in val.rows]
        val_sampler=None if val is None else (DistributedLengthBucketBatchSampler(val_lengths,a.per_rank_batch_size,world,rank,a.seed+1,False) if a.ddp else LengthBucketBatchSampler(val_lengths,a.per_rank_batch_size,a.seed+1))
        pin=device.type=="cuda" and not a.no_pin_memory;train_loader=loader(train,train_sampler,a.num_workers,pin,a.prefetch_factor);val_loader=None if val is None else loader(val,val_sampler,a.num_workers,pin,a.prefetch_factor)
        steps_per_epoch=len(train_sampler)
        checkpoint_every_steps=a.checkpoint_every_epochs*steps_per_epoch
        cfg["runtime"].update({"steps_per_epoch":steps_per_epoch,"checkpoint_every_epochs":a.checkpoint_every_epochs,"checkpoint_every_steps":checkpoint_every_steps})
        data_hashes=json.loads((Path(a.data_root)/"metadata/source_hashes.json").read_text())
        if rank==0:
            # Include sampler-derived epoch/checkpoint semantics in the saved
            # run contract rather than leaving them implicit in the launcher.
            save_yaml(out/"resolved_config.yaml",cfg)
        if rank==0: print("[CMU train] building frozen END graph and CUDA prefetch pipeline",flush=True)
        raw=HierarchicalGroupwiseEND(cfg).to(device)
        if rank==0 and not a.resume:
            (out/"initialization_manifest.json").write_text(json.dumps({"seed":a.seed,"initial_model_state_sha256":model_state_sha256(raw),"cross_sequence_communication":bool(cfg.get("model",{}).get("cross_sequence_communication",True)),"parameter_count":sum(p.numel() for p in raw.parameters()),"trainable_parameter_count":sum(p.numel() for p in raw.parameters() if p.requires_grad)},indent=2)+"\n")
        model=DDP(raw,device_ids=[device.index],output_device=device.index,broadcast_buffers=False,find_unused_parameters=False,gradient_as_bucket_view=True,static_graph=True) if a.ddp else raw
        loss=criterion(cfg,train.rows).to(device)
        optimizer=build_optimizer(model,cfg["runtime"]);scheduler=build_scheduler(optimizer,cfg["runtime"])
        trainer=Trainer(model,loss,optimizer,scheduler,device,amp=a.amp,grad_clip=a.grad_clip_norm,ema_decay=a.ema_decay,total_steps=a.max_steps,warmup_fraction=projector_warmup/max(1,a.max_steps),ramp_fraction=projector_ramp/max(1,a.max_steps),ema_step_scale=a.ema_step_scale)
        checkpoint_writer=AsyncCheckpointWriter() if a.async_checkpoint and rank==0 else None
        def persist_checkpoint(path):
            if a.async_checkpoint:
                snapshot=snapshot_checkpoint(raw,trainer,epoch,cfg,data_hashes,train_sampler,best)
                if rank==0:
                    checkpoint_writer.submit(path,snapshot)
                    print(f"[CMU checkpoint] queued: {path}",flush=True)
            else:
                save_checkpoint(path,raw,trainer,epoch,cfg,data_hashes,train_sampler,best)
                if rank==0: print(f"[CMU checkpoint] saved: {path}",flush=True)
        if rank==0:
            groups="all" if loss.projector_batch_groups is None else str(loss.projector_batch_groups)
            print(f"[CMU objective] L = L_CRF + projector_scale * {loss.lambda_projector:g} * L_Projector; Projector groups/rank={groups}, stride={loss.projector_stride}, warmup={projector_warmup}, ramp={projector_ramp}",flush=True)
            psd_groups="all" if loss.group_psd_batch_groups is None else str(loss.group_psd_batch_groups)
            if loss.lambda_group_psd != 0.0:
                position_spec = f"{loss.group_psd_position_fraction:.0%} per-curve native length" if loss.group_psd_position_fraction is not None else str(loss.group_psd_num_bins)
                print(f"[CMU objective] + {loss.lambda_group_psd:g} * L_GroupPSD from epoch {loss.group_psd_start_epoch}; prediction-only directed native-position Always-Overlap occupancy; no reciprocal fusion/pooling/degree normalization; requested positions={position_spec}; rotating groups/rank={psd_groups}; alpha/beta detached in auxiliary branch.", flush=True)
        # ``best`` chooses the checkpoint by the strict minimum validation
        # score.  ``early_stop_anchor`` is deliberately separate: min_delta
        # controls patience only and must never discard a numerically better
        # checkpoint from the final test.
        epoch=0;best=float("inf");best_step=None;early_stop_anchor=float("inf");non_improving_validations=0;stopped_early=False;initial_checkpoint=None
        if a.resume:
            state=load_checkpoint(a.resume,raw,trainer,train_sampler,device);epoch=int(state["epoch"]);best=float(state.get("best_metric",best));best_step=trainer.step;initial_checkpoint=str(Path(a.resume).resolve())
            # A continuation must treat its source validation winner as an
            # actual candidate.  Otherwise a run that never beats the source
            # score produces no ``best.pt`` although its initial model is the
            # correct validation-selected result.  Copy the immutable source
            # snapshot once; a later strict improvement overwrites it through
            # the normal checkpoint path.
            if best < float("inf"):
                early_stop_anchor=best
                if rank==0:
                    destination=out/"checkpoints"/"best.pt"
                    shutil.copy2(a.resume,destination)
                    print(f"[CMU checkpoint] carried forward resume winner: {destination}",flush=True)
                if a.ddp:
                    import torch.distributed as dist;dist.barrier()
        if a.e1_condition and rank==0:
            digest=checkpoint_sha256(Path(a.resume))
            control={"experiment":f"{a.controlled_experiment} controlled continuation","condition":a.controlled_condition or a.e1_condition,"fork_checkpoint":str(Path(a.resume).resolve()),"fork_checkpoint_sha256":digest,"resume_mode":a.resume_mode,"state_restore":["online_model","optimizer","scheduler","EMA","GradScaler","per_rank_RNG","sampler_state"],"start_global_update":trainer.step,"budget_end_global_update":a.max_steps,"post_fork_optimizer_updates":a.max_steps-trainer.step,"microbatch_per_rank":a.per_rank_batch_size,"world_size":world,"effective_global_batch":a.per_rank_batch_size*world,"validation_every_updates":a.val_every,"validation_candidates_including_fork":1+(a.max_steps-trainer.step)//a.val_every,"selection_metric":"EMA parent_balanced_alignment_loss","early_stopping":"disabled; fixed optimizer-update budget","data_order":"deterministic LengthBucketBatchSampler(seed=experiment_seed, epoch) restored from common sampler state","lr_control":"same resumed scheduler state and identical max_steps/LambdaLR contract"}
            (out/f"{a.controlled_experiment.lower()}_control_manifest.json").write_text(json.dumps(control,indent=2)+"\n")
        if a.staged_condition and rank==0:
            digest=checkpoint_sha256(Path(a.resume))
            staged={"experiment":a.controlled_experiment,"condition":a.staged_condition,"fork_checkpoint":str(Path(a.resume).resolve()),"fork_checkpoint_sha256":digest,"resume_mode":a.resume_mode,"state_restore":["online_model","optimizer","scheduler","EMA","GradScaler","per_rank_RNG","sampler_state"],"start_global_update":trainer.step,"maximum_global_update":a.max_steps,"maximum_post_fork_optimizer_updates":a.max_steps-trainer.step,"microbatch_per_rank":a.per_rank_batch_size,"world_size":world,"effective_global_batch":a.per_rank_batch_size*world,"validation_every_updates":a.val_every,"selection_metric":"EMA parent_balanced_alignment_loss","early_stopping":{"enabled":bool(a.early_stop_patience),"patience":a.early_stop_patience,"min_delta":a.early_stop_min_delta,"min_epochs":a.early_stop_min_epochs},"data_order":"LengthBucketBatchSampler restored from source sampler state","lr_control":"resumed scheduler state with production absolute max-step horizon"}
            (out/"staged_continuation_manifest.json").write_text(json.dumps(staged,indent=2)+"\n")
        iterator=CUDAPrefetcher(train_loader,device) if device.type=="cuda" else iter(train_loader)
        started=time.perf_counter();window_started=started;window_groups=0;window_steps=0;window_metric_sums=None
        train_stop=a.max_steps if a.stop_after_steps is None else int(a.stop_after_steps)
        if not 1 <= train_stop <= a.max_steps: raise ValueError("stop-after-steps must be in [1,max-steps]")
        while trainer.step<train_stop:
            if rank==0 and checkpoint_writer is not None: checkpoint_writer.poll()
            try:
                batch=iterator.next() if device.type=="cuda" else next(iterator)
                if batch is None: raise StopIteration
            except StopIteration:
                epoch+=1;trainer.consumed_batch_in_epoch=0;train_sampler.set_epoch(epoch);iterator=CUDAPrefetcher(train_loader,device) if device.type=="cuda" else iter(train_loader);batch=iterator.next() if device.type=="cuda" else next(iterator)
                if batch is None: raise RuntimeError("empty frozen training manifest")
            # Epoch numbering is one-based for the loss contract.  On an
            # epoch-10 continuation the first resumed update is therefore
            # epoch 11, exactly matching the validated D34 hand-off.
            loss.set_training_epoch(trainer.step//steps_per_epoch + 1)
            materialize=((trainer.step+1)%a.progress_every==0 or trainer.step+1==a.max_steps)
            log=trainer.train_step(batch,materialize_log=materialize)
            metric_tensors=log.pop("_window_metrics")
            if window_metric_sums is None:
                window_metric_sums={key:value.clone() for key,value in metric_tensors.items()}
            else:
                for key,value in metric_tensors.items(): window_metric_sums[key].add_(value)
            window_groups+=world*int(log["batch_group_count"]);window_steps+=1
            if materialize:
                if device.type=="cuda": torch.cuda.synchronize(device)
                metric_names=tuple(window_metric_sums)
                metric_vector=torch.stack([window_metric_sums[key] for key in metric_names])
                if a.ddp:
                    import torch.distributed as dist
                    dist.all_reduce(metric_vector)
                metric_vector.div_(world*window_steps)
                for key,value in zip(metric_names,metric_vector.tolist(),strict=True): log[key]=float(value)
                # These old fields were a *last length bucket* observation.
                # Preserve them only under an explicit name; the primary loss
                # fields above are exact four-GPU window means.
                for key in ("unweighted_total_loss","unweighted_alignment_loss","unweighted_projector_loss","unweighted_group_score_loss","unweighted_group_mean_loss","unweighted_group_variance_loss","unweighted_group_psd_loss","unweighted_similarity_loss"):
                    log[f"last_batch_{key}"]=log.pop(key)
                for key in ("grad_norm","learning_rate","projector_scale","projector_groups_used","projector_nodes","projector_offset","group_score_points","group_score_groups","similarity_points","group_psd_active","group_psd_groups_used","group_psd_offset","group_psd_positions_used","group_psd_min_eigenvalue","group_psd_negative_eigenvalue_count","group_psd_negative_spectral_mass","group_psd_directional_disagreement","group_psd_posterior_entropy","group_psd_max_posterior","group_psd_expected_path_length","group_psd_predicted_overlap_fraction","pair_cell_count","L_batch","batch_group_count"):
                    log[f"last_batch_{key}"]=log.pop(key)
                log["window_steps"]=window_steps;log["global_window_groups"]=window_groups
                window_seconds=time.perf_counter()-window_started
                log["update_seconds"]=window_seconds/max(1,window_steps)
                # Directional CRF modes have no alpha.  Materialize their two
                # score parameters only at the existing progress boundary so
                # ordinary GPU updates retain their asynchronous hot path.
                if hasattr(raw,"crf_temperature_raw"):
                    log["crf_temperature"]=float((raw.crf_temperature_min+torch.nn.functional.softplus(raw.crf_temperature_raw)).detach().cpu())
                    log["crf_gamma"]=float(raw.gamma.detach().cpu())
                if rank==0:append_jsonl(out/"logs/train.jsonl",log)
            if rank==0 and materialize:
                rate=window_groups/max(1e-9,window_seconds)
                memory=torch.cuda.max_memory_allocated(device)/(1024**3) if device.type=="cuda" else 0.0
                elapsed=time.perf_counter()-started
                remaining=max(0,a.max_steps-trainer.step)*window_seconds/max(1,window_steps)
                calibration="" if "crf_temperature" not in log else f" tau={log['crf_temperature']:.6f} gamma={log['crf_gamma']:.6f}"
                print(f"[CMU train] step={trainer.step}/{a.max_steps} window={log['window_steps']} loss={log['weighted_total_loss']:.6f} crf={log['weighted_alignment_loss']:.6f} sim={log['weighted_similarity_loss']:.6f} sim_term={log['weighted_similarity_contribution']:.6f} group={log['weighted_group_score_loss']:.6f} group_term={log['weighted_group_score_contribution']:.6f} psd={log['weighted_group_psd_loss']:.6f} psd_term={log['weighted_group_psd_contribution']:.6f} psd_active={log['last_batch_group_psd_active']} psd_pos={log['last_batch_group_psd_positions_used']} dir_energy={log['last_batch_group_psd_directional_disagreement']:.3e} neg_energy={log['last_batch_group_psd_negative_spectral_mass']:.3e} proj_est={log['weighted_projector_loss']:.6f} proj_term={log['weighted_projector_contribution']:.6f} pscale_last={log['last_batch_projector_scale']:.3f} grad_last={log['last_batch_grad_norm']:.4f}{calibration} update={log['update_seconds']:.3f}s rate={rate:.2f} groups/s gpu_alloc={memory:.2f}GiB elapsed={elapsed:.0f}s ETA={remaining:.0f}s",flush=True)
            if materialize:
                # Every DDP rank contributes to the next all-reduced log
                # window.  Resetting only rank 0 would make ranks 1..N retain
                # earlier windows and report an artificial, monotonically
                # increasing loss, although their optimization is correct.
                window_started=time.perf_counter();window_groups=0;window_steps=0;window_metric_sums=None
            checkpoint_due=(checkpoint_every_steps and trainer.step%checkpoint_every_steps==0) or trainer.step==train_stop
            validation_due=not a.skip_validation and (trainer.step%a.val_every==0 or trainer.step==train_stop)
            # A numbered epoch checkpoint is a continuation source.  When it
            # coincides with validation, persist it *after* validation so it
            # includes that epoch's strict-best score and early-stop state.
            if checkpoint_due and not validation_due:
                completed_epochs=trainer.step//steps_per_epoch
                checkpoint=out/f"checkpoints/final.pt" if trainer.step==train_stop else out/f"checkpoints/epoch_{completed_epochs:04d}.pt"
                # ``epoch`` remains the sampler's current zero-based epoch
                # until its exhausted iterator is rewound on the next loop.
                # Persist that state for an exact mid-boundary resume; the
                # filename records the human-facing completed epoch count.
                persist_checkpoint(checkpoint)
            if validation_due:
                if device.type=="cuda": torch.cuda.synchronize(device)
                # Validation samplers intentionally retain a cursor for resume
                # support.  Every evaluation must rewind to cover the complete
                # frozen validation manifest, rather than silently reusing an
                # exhausted iterator.
                val_sampler.cursor=0
                metric=distributed_validation(trainer.ema.model,loss,val_loader,device,a.ddp)
                score=metric["parent_balanced_alignment_loss"]
                checkpoint_improved=score < best
                if checkpoint_improved: best=score;best_step=trainer.step
                patience_improved=score < early_stop_anchor-a.early_stop_min_delta
                if patience_improved:
                    early_stop_anchor=score;non_improving_validations=0
                else:
                    non_improving_validations+=1
                if rank==0:
                    if hasattr(trainer.ema.model,"crf_temperature_raw"):
                        metric["crf_temperature"]=float((trainer.ema.model.crf_temperature_min+torch.nn.functional.softplus(trainer.ema.model.crf_temperature_raw)).detach().cpu())
                        metric["crf_gamma"]=float(trainer.ema.model.gamma.detach().cpu())
                    metric={"global_step":trainer.step,"validation_weights":"ema","checkpoint_improved":checkpoint_improved,"patience_improved":patience_improved,"non_improving_validations":non_improving_validations,"best_parent_balanced_alignment_loss":best,"best_global_step":best_step,**metric};append_jsonl(out/"logs/validation.jsonl",metric)
                    calibration="" if "crf_temperature" not in metric else f" tau={metric['crf_temperature']:.6f} gamma={metric['crf_gamma']:.6f}"
                    print(f"[CMU valid] step={trainer.step} alignment={metric['parent_balanced_alignment_loss']:.6f} similarity={metric['parent_balanced_similarity_loss']:.6f} psd={metric['parent_balanced_group_psd_loss']:.6f} projector={metric['parent_balanced_projector_loss']:.6f} total={metric['parent_balanced_total_loss']:.6f}{calibration}",flush=True)
                if a.decoded_validation:
                    val_sampler.cursor=0
                    decoded=distributed_decoded_validation(trainer.ema.model,val_loader,device,reference,a.ddp,len(val),allow_no_overlap_output=bool(cfg["loss"].get("allow_no_overlap_output",True)),emission_mode=str(cfg["alignment"].get("crf_emission_mode","legacy")),similarity_eps=float(cfg["alignment"].get("similarity_eps",1e-6)),normalization_eps=float(cfg["alignment"].get("normalization_eps",1e-8)))
                    if rank==0:
                        decoded={"global_step":trainer.step,"validation_weights":"ema",**decoded}
                        append_jsonl(out/"logs/decoded_validation.jsonl",decoded)
                        print(f"[CMU valid decoded] PathMAE={decoded['path_mae_index']:.3f} F1@2={decoded['f1_tolerance_0_20']['2']:.3f} PromptBoundary={decoded['prompt_boundary_mae_index']:.3f} PromptIoU={decoded['interval_iou']:.3f} TransMAE={decoded['transitive_mae_index']:.3f} TransP95={decoded['transitive_p95_index']:.3f}",flush=True)
                if not a.no_latest_checkpoint_on_validation: persist_checkpoint(out/"checkpoints/latest.pt")
                if checkpoint_improved:persist_checkpoint(out/"checkpoints/best.pt")
                if checkpoint_due:
                    completed_epochs=trainer.step//steps_per_epoch
                    checkpoint=out/f"checkpoints/final.pt" if trainer.step==train_stop else out/f"checkpoints/epoch_{completed_epochs:04d}.pt"
                    persist_checkpoint(checkpoint)
                if a.ddp:
                    import torch.distributed as dist;dist.barrier()
                completed_epochs=trainer.step//steps_per_epoch
                if a.early_stop_patience and completed_epochs>=a.early_stop_min_epochs and non_improving_validations>=a.early_stop_patience:
                    stopped_early=True
                    persist_checkpoint(out/"checkpoints/early_stop.pt")
                    if rank==0: print(f"[CMU early-stop] step={trainer.step} no_material_improvement={non_improving_validations}/{a.early_stop_patience} best_step={best_step} best_alignment={best:.6f}",flush=True)
                    if a.ddp:
                        import torch.distributed as dist;dist.barrier()
                    break
        if rank==0 and checkpoint_writer is not None:
            checkpoint_writer.flush()
            print("[CMU checkpoint] background writer finished",flush=True)
        if a.ddp:
            import torch.distributed as dist;dist.barrier()
        if rank==0:
            if not a.skip_validation:
                selection={"selection_protocol":"strict minimum EMA parent_balanced_alignment_loss during online validation; test is not read","selected_checkpoint":str((out/"checkpoints/best.pt").resolve()),"selected_global_step":best_step,"selected_parent_balanced_alignment_loss":None if best==float("inf") else best,"early_stop_patience_validation_events":a.early_stop_patience,"early_stop_min_delta":a.early_stop_min_delta,"early_stop_min_epochs":a.early_stop_min_epochs,"stopped_early":stopped_early}
                (out/"best_checkpoint.json").write_text(json.dumps(selection,indent=2)+"\n")
            metadata={"run_id":a.run_id,"run_type":a.run_type,"training_manifest_records":len(train),"training_parent_prompts":len({r["parent_prompt_id"] for r in train.rows}),"global_mean_sample_weight":parent_global_mean_weight(train.rows),"K_construction_at_training":False,"full_utterance":True,"world_size":world,"global_step":trainer.step,"crf_reference_length":reference,"initial_checkpoint":initial_checkpoint,"checkpoint_selection":"none_validation_disabled" if a.skip_validation else "parent_balanced_validation_alignment_loss_ema","early_stop":{"enabled":bool(a.early_stop_patience),"patience_validation_events":a.early_stop_patience,"min_delta":a.early_stop_min_delta,"stopped_early":stopped_early,"best_global_step":best_step,"best_parent_balanced_alignment_loss":None if best==float("inf") else best,"non_improving_validations_at_stop":non_improving_validations}};(out/"run_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    finally:
        if checkpoint_writer is not None: checkpoint_writer.close()
        distributed_cleanup(a.ddp)

if __name__=="__main__":main()
