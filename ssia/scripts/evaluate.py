#!/usr/bin/env python3
"""DDP-safe, inference-only V4 evaluator for a frozen D3--D4 END checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from ssia_alignment.data.dataset import D34BridgeDataset, D34PreparedImpRGTDataset, FixedIndexBatchSampler, build_loader
from ssia_alignment.end.model import build_end_model
from ssia_alignment.end.partial_overlap_crf import viterbi_decode_batch
from ssia_alignment.evaluation import EVALUATION_PROTOCOL, draw_group_alignment, evaluate_predictions, summarize
from ssia_alignment.runtime import CUDAPrefetcher, barrier, ddp_setup


def _write_json(path: Path,payload: dict[str,Any]) -> None:
    def safe(value):
        if isinstance(value,dict): return {str(k):safe(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)): return [safe(v) for v in value]
        if isinstance(value,(float,np.floating)) and not np.isfinite(value): return None
        if isinstance(value,np.integer): return int(value)
        return value
    temporary=path.with_suffix(path.suffix+".tmp"); temporary.write_text(json.dumps(safe(payload),ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8"); temporary.replace(path)


def _configure_cuda(config: dict[str,Any]) -> tuple[bool,torch.dtype]:
    train=config.get("train",{}); tf32=bool(train.get("tf32",True))
    torch.backends.cuda.matmul.allow_tf32=tf32; torch.backends.cudnn.allow_tf32=tf32; torch.backends.cudnn.benchmark=bool(train.get("cudnn_benchmark",True))
    if hasattr(torch,"set_float32_matmul_precision"): torch.set_float32_matmul_precision("high")
    amp=str(train.get("amp","bf16")).lower()
    return amp in {"bf16","fp16"},torch.bfloat16 if amp=="bf16" else torch.float16


def _resolve_terminal_checkpoint(run_dir: Path,requested: Path|None) -> Path:
    """Prefer last.pt, with a train-log fallback for copied run folders.

    Evaluation is intentionally terminal-checkpoint evaluation: the completed
    final EMA state, not the minimum-validation-loss state.  Copies can lose
    symlinks, so train_log.jsonl identifies the same final optimizer step.
    """
    if requested is not None:
        path=requested.resolve()
        if not path.is_file(): raise FileNotFoundError(path)
        return path
    linked=run_dir/"checkpoints"/"last.pt"
    if linked.is_file(): return linked.resolve()
    log=run_dir/"train_log.jsonl"
    if not log.is_file(): raise FileNotFoundError(f"missing last.pt and training log: {run_dir}")
    selected=None
    for line in log.read_text(encoding="utf8").splitlines():
        item=json.loads(line)
        if "step" in item: selected=int(item["step"])
    if selected is None: raise RuntimeError(f"no terminal training step in {log}")
    path=run_dir/"checkpoints"/f"step_{selected:06d}.pt"
    if not path.is_file(): raise FileNotFoundError(f"terminal step {selected}, but checkpoint is missing: {path}")
    return path.resolve()


def _decode_paths(output: dict[str,torch.Tensor]) -> np.ndarray:
    """One raw-score partial-overlap CRF Viterbi decode for every directed pair."""
    if output.get("matching_mode")=="direct_cosine":
        features=output["matching_features"].float(); batch,curves,length,dimension=features.shape
        ids=torch.arange(curves,device=features.device); source,target=ids[:,None].expand(curves,curves),ids[None,:].expand(curves,curves); select=source!=target; source,target=source[select],target[select]
        forward=source<target; left_id,right_id=source[forward],target[forward]; canonical_left,canonical_right=torch.minimum(source,target),torch.maximum(source,target); route=((canonical_left[:,None]==left_id[None,:])&(canonical_right[:,None]==right_id[None,:])).to(torch.long).argmax(dim=1)
        left=features.index_select(1,left_id).reshape(batch*len(left_id),length,dimension); right=features.index_select(1,right_id).reshape(batch*len(right_id),length,dimension)
        unordered=torch.bmm(left,right.transpose(1,2)).reshape(batch,len(left_id),length,length); similarity=unordered.index_select(1,route); similarity=torch.where(forward[None,:,None,None],similarity,similarity.transpose(-1,-2))
        score=output["group_crf_alpha"].float()*similarity.clamp(-1.,1.)+output["group_crf_beta"].float()
        decoded=viterbi_decode_batch(score.reshape(batch*len(source),length,length),match_bias=0.,segment_bias=0.,allow_empty=False).q_of_p.numpy().reshape(batch,len(source),length)
        paths=np.full((batch,curves,curves,length),-1.,dtype=np.float32); paths[:,source.detach().cpu().numpy(),target.detach().cpu().numpy()]=decoded
        return paths
    membership=output["group_membership"].float(); batch,curves,length,_=membership.shape
    ids=torch.arange(curves,device=membership.device); source,target=ids[:,None].expand(curves,curves),ids[None,:].expand(curves,curves); select=source!=target; source,target=source[select],target[select]
    left=membership.index_select(1,source).reshape(batch*len(source),length,-1); right=membership.index_select(1,target).reshape(batch*len(target),length,-1)
    correspondence=torch.bmm(left,right.transpose(1,2))
    score=output["group_crf_alpha"].float()*correspondence.clamp_min(1.e-6).log()+output["group_crf_beta"].float()
    decoded=viterbi_decode_batch(score,match_bias=0.,segment_bias=None).q_of_p.numpy().reshape(batch,len(source),length)
    paths=np.full((batch,curves,curves,length),-1.,dtype=np.float32); paths[:,source.detach().cpu().numpy(),target.detach().cpu().numpy()]=decoded
    return paths


def _gather_rows(rows: list[dict[str,Any]],world_size:int) -> list[dict[str,Any]]|None:
    if world_size==1: return rows
    gathered:[Any]=[None]*world_size; dist.all_gather_object(gathered,rows)
    return [row for shard in gathered for row in shard] if dist.get_rank()==0 else None


def _distributed_transitive_stats(errors: np.ndarray,device: torch.device,world_size:int) -> dict[str,Any]:
    """Exact pooled mean/P95 with scalar-only DDP communication."""
    values=np.asarray(errors,np.float32).reshape(-1); values=values[np.isfinite(values)]
    if world_size==1:
        return {"transitive_error_points":int(len(values)),"transitive_mae_index":float(values.mean()) if len(values) else float("nan"),"transitive_p95_index":float(np.percentile(values,95)) if len(values) else float("nan")}
    values=np.sort(values); totals=torch.tensor([float(values.sum(dtype=np.float64)),float(len(values))],device=device,dtype=torch.float64); dist.all_reduce(totals,op=dist.ReduceOp.SUM); count=int(totals[1].item())
    if count==0: return {"transitive_error_points":0,"transitive_mae_index":float("nan"),"transitive_p95_index":float("nan")}
    bits=values.view(np.uint32); lower=torch.tensor(int(bits.min()) if len(bits) else 2**32-1,device=device,dtype=torch.int64); upper=torch.tensor(int(bits.max()) if len(bits) else 0,device=device,dtype=torch.int64); dist.all_reduce(lower,op=dist.ReduceOp.MIN); dist.all_reduce(upper,op=dist.ReduceOp.MAX)
    def kth(rank:int) -> float:
        lo,hi=int(lower.item()),int(upper.item())
        while lo<hi:
            middle=(lo+hi)//2; value=np.array([middle],dtype=np.uint32).view(np.float32)[0]
            seen=torch.tensor(int(np.searchsorted(values,value,side="right")),device=device,dtype=torch.int64); dist.all_reduce(seen,op=dist.ReduceOp.SUM)
            if int(seen.item())>rank: hi=middle
            else: lo=middle+1
        return float(np.array([lo],dtype=np.uint32).view(np.float32)[0])
    position=.95*(count-1); left=int(math.floor(position)); right=int(math.ceil(position)); left_value,right_value=kth(left),kth(right)
    return {"transitive_error_points":count,"transitive_mae_index":float(totals[0].item()/count),"transitive_p95_index":left_value+(position-left)*(right_value-left_value)}


def _write_csv(path: Path,rows: list[dict[str,Any]]) -> None:
    if not rows: raise ValueError(f"cannot write no rows: {path}")
    with path.open("w",newline="",encoding="utf8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _per_group_rows(pair_rows: list[dict[str,Any]],triple_rows: list[dict[str,Any]]) -> list[dict[str,Any]]:
    pairs:dict[tuple[int,int],list[dict[str,Any]]]={}; triples:dict[tuple[int,int],list[dict[str,Any]]]={}
    for row in pair_rows: pairs.setdefault((int(row["volume_id"]),int(row["coordinate_group_id"])),[]).append(row)
    for row in triple_rows: triples.setdefault((int(row["volume_id"]),int(row["coordinate_group_id"])),[]).append(row)
    output=[]
    for key,items in sorted(pairs.items()):
        triple=triples[key]; output.append({"volume_id":key[0],"coordinate_group_id":key[1],"quality_tier":items[0]["quality_tier"],"storage_index":items[0]["storage_index"],"pairwise_path_mae_index":float(np.nanmean([row["path_mae_index"] for row in items])),"pairwise_f1_tol_02_index":float(np.mean([row["f1_tol_02_index"] for row in items])),"pairwise_iou":float(np.mean([row["iou"] for row in items])),"transitive_mae_index":float(np.nanmean([row["transitive_mae_index"] for row in triple])),"transitive_p95_index":float(np.nanmean([row["transitive_p95_index"] for row in triple])),"transitive_valid_points":int(sum(row["transitive_valid_points"] for row in triple))})
    return output


def _plot_f1(summary: dict[str,Any],output: Path,label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    curve=summary["f1_tolerance_0_20"]; x=np.arange(21); y=np.asarray([curve[str(item)] for item in x])
    figure,axis=plt.subplots(figsize=(6.2,4.1),constrained_layout=True); axis.plot(x,y,color="#2B6CB0",marker="o",markersize=3.4,linewidth=1.8,label=label); axis.set(xlim=(0,20),ylim=(0,1.02),xlabel="Tolerance (index points)",ylabel="F1",title="F1–Tolerance curve"); axis.grid(color="#E5E5E5",linewidth=.6); axis.legend(frameon=False); figure.savefig(output,dpi=300,facecolor="white"); plt.close(figure)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True); parser.add_argument("--split",choices=("train","validation","test"),required=True); parser.add_argument("--output-dir",type=Path,required=True); parser.add_argument("--checkpoint",type=Path,default=None)
    parser.add_argument("--data-root",type=Path,default=None,help="optional read-only dataset root override; model/checkpoint configuration is unchanged")
    parser.add_argument("--weights",choices=("ema","model"),default="ema",help="checkpoint ema_model or direct model state")
    parser.add_argument("--model-config",type=Path,default=None,help="optional external END config; its model is evaluated on this D34 run's data")
    parser.add_argument("--batch-size",type=int,default=8); parser.add_argument("--num-workers",type=int,default=4); parser.add_argument("--prefetch-factor",type=int,default=2); parser.add_argument("--worker-threads",type=int,default=1); parser.add_argument("--qualitative-count",type=int,default=10); parser.add_argument("--progress-every",type=int,default=5)
    parser.add_argument("--export-predictions",action="store_true",help="write all directed test paths in the shared qualitative-archive JSONL schema")
    args=parser.parse_args()
    if args.batch_size<1 or args.qualitative_count<0: raise ValueError("batch size must be positive and qualitative count non-negative")
    rank,world_size,local_rank,distributed=ddp_setup(); device=torch.device(f"cuda:{local_rank}"); run_dir,out_dir=args.run_dir.resolve(),args.output_dir.resolve(); evaluation_config=json.loads((run_dir/"config_resolved.json").read_text(encoding="utf8")); checkpoint=_resolve_terminal_checkpoint(run_dir,args.checkpoint)
    if rank==0:
        if out_dir.exists() and any(out_dir.iterdir()): raise FileExistsError(f"refusing to overwrite non-empty output: {out_dir}")
        (out_dir/"metrics").mkdir(parents=True,exist_ok=True); (out_dir/"qualitative"/"group_alignment"/args.split).mkdir(parents=True,exist_ok=True)
        if args.export_predictions: (out_dir/"predictions").mkdir(parents=True,exist_ok=True)
    barrier(distributed)
    payload=torch.load(checkpoint,map_location=device,weights_only=False)
    raw_model_config=json.loads(args.model_config.resolve().read_text(encoding="utf8")) if args.model_config is not None else payload.get("config",evaluation_config)
    model_config=raw_model_config.get("training_config",raw_model_config)
    expected=str(model_config["experiment"]["method_version"])
    if payload.get("method_version")!=expected: raise RuntimeError(f"checkpoint/model identity mismatch: {payload.get('method_version')!r} != {expected!r}")
    checkpoint_config=payload.get("config")
    checkpoint_model_config=checkpoint_config.get("training_config",checkpoint_config) if checkpoint_config is not None else None
    if checkpoint_model_config is not None and json.dumps(checkpoint_model_config,sort_keys=True)!=json.dumps(model_config,sort_keys=True): raise RuntimeError("checkpoint config differs from the selected model config")
    state=payload.get("ema_model") if args.weights=="ema" else payload.get("model")
    if state is None: raise RuntimeError(f"checkpoint lacks requested {args.weights} weights")
    model=build_end_model(model_config).to(device); model.load_state_dict(state,strict=True); model.eval(); amp_enabled,amp_dtype=_configure_cuda(model_config)
    data=dict(evaluation_config["data"])
    if args.data_root is not None:
        data["root"]=str(args.data_root.resolve())
    if str(data.get("format", "")).lower()=="prepared_imp_rgt_v1":
        dataset=D34PreparedImpRGTDataset(data["root"],args.split,min_overlap_fraction=float(data["min_overlap_fraction"]),coordinate_tolerance=float(data["coordinate_tolerance"]))
    else:
        dataset=D34BridgeDataset(data["root"],args.split,min_overlap_fraction=float(data["min_overlap_fraction"]),coordinate_tolerance=float(data["coordinate_tolerance"]))
    sampler=FixedIndexBatchSampler(len(dataset),rank=rank,world_size=world_size,batch_size=args.batch_size); loader=build_loader(dataset,sampler,rank=rank,workers=args.num_workers,prefetch_factor=args.prefetch_factor,worker_threads=args.worker_threads,validation=True)
    started=time.perf_counter(); pair_rows=[]; triple_rows=[]; error_chunks=[]; completed=0
    prediction_stream=(out_dir/"predictions"/f"predictions_rank{rank:03d}.jsonl").open("w",encoding="utf8") if args.export_predictions else None
    prefetcher=CUDAPrefetcher(loader,device)
    with torch.inference_mode():
        while True:
            try: batch=prefetcher.next()
            except StopIteration: break
            with torch.autocast("cuda",dtype=amp_dtype,enabled=amp_enabled): output=model(batch["z"],batch["mask"])
            paths=_decode_paths(output); gt=batch["q_of_p"].detach().cpu().numpy(); valid=batch["valid_pair"].detach().cpu().numpy()
            metadata=[{"volume_id":int(batch["volume_id"][index]),"coordinate_group_id":int(batch["coordinate_group_id"][index]),"quality_tier":int(batch["quality_tier"][index]),"storage_index":int(batch["storage_index"][index])} for index in range(len(paths))]
            if prediction_stream is not None:
                for index, item in enumerate(metadata):
                    group_id=f"v{item['volume_id']:05d}_g{item['coordinate_group_id']:03d}"
                    record={"group_id":group_id,"paths":{f"{source}->{target}":paths[index,source,target].tolist() for source in range(8) for target in range(8) if source!=target}}
                    prediction_stream.write(json.dumps(record,separators=(",",":"),allow_nan=False)+"\n")
            local_pairs,local_triples,errors=evaluate_predictions(paths,gt,valid,metadata); pair_rows.extend(local_pairs); triple_rows.extend(local_triples)
            if len(errors): error_chunks.append(errors)
            completed+=len(paths)
            if completed==len(paths) or completed%max(1,args.progress_every*args.batch_size)==0 or completed==len(sampler.indices): print(f"[D34 {args.split}] rank {rank}/{world_size}: {completed}/{len(sampler.indices)} local groups, elapsed {(time.perf_counter()-started)/60:.1f} min",flush=True)
    if prediction_stream is not None: prediction_stream.close()
    errors=np.concatenate(error_chunks) if error_chunks else np.empty(0,dtype=np.float32); transitive_stats=_distributed_transitive_stats(errors,device,world_size); all_pairs=_gather_rows(pair_rows,world_size); all_triples=_gather_rows(triple_rows,world_size); barrier(distributed)
    if rank==0:
        assert all_pairs is not None and all_triples is not None
        source_config="checkpoint_embedded_config" if args.model_config is None and checkpoint_config is not None else (None if args.model_config is None else str(args.model_config.resolve()))
        policy="best validation CRF checkpoint" if checkpoint.name=="best.pt" else "terminal completed-training checkpoint"
        summary=summarize(all_pairs,all_triples,transitive_stats,groups=len(dataset)); summary.update({"method_version":expected,"evaluation_data_method_version":str(evaluation_config["experiment"]["method_version"]),"source_model_config":source_config,"split":args.split,"checkpoint":str(checkpoint),"checkpoint_step":int(payload.get("global_step",-1)),"weights":"ema_model" if args.weights=="ema" else "model","checkpoint_policy":{"path":checkpoint.name,"policy":policy},"world_size":world_size,"batch_size_per_rank":args.batch_size,"runtime_seconds":time.perf_counter()-started})
        metrics_dir=out_dir/"metrics"; _write_json(metrics_dir/"summary.json",summary); _write_csv(metrics_dir/"pairwise_metrics.csv",all_pairs); _write_csv(metrics_dir/"transitive_metrics.csv",all_triples); _write_csv(metrics_dir/"per_group_metrics.csv",_per_group_rows(all_pairs,all_triples)); _write_json(metrics_dir/"f1_tolerance_0_20.json",{"coordinate_unit":"index points","f1_tolerance_0_20":summary["f1_tolerance_0_20"]}); _plot_f1(summary,metrics_dir/"f1_tolerance_0_20.png",f"{model_config['experiment'].get('model_scale','END')}-{args.split}")
        selected=[]
        if args.qualitative_count:
            positions=np.linspace(0,len(dataset)-1,min(args.qualitative_count,len(dataset)),dtype=np.int64)
            for ordinal,position in enumerate(positions,start=1):
                item=dataset[int(position)]; z=item["z"].unsqueeze(0).to(device); mask=item["mask"].unsqueeze(0).to(device)
                with torch.inference_mode(),torch.autocast("cuda",dtype=amp_dtype,enabled=amp_enabled): paths=_decode_paths(model(z,mask))[0]
                volume,group=int(item["volume_id"]),int(item["coordinate_group_id"]); filename=f"{args.split}_{ordinal:02d}_volume_{volume}_group_{group}_storage_{int(item['storage_index'])}.png"; draw_group_alignment(out_dir/"qualitative"/"group_alignment"/args.split/filename,item["z"].numpy(),np.asarray(dataset.rgt[int(position)],dtype=np.float32),paths,f"{args.split} group {ordinal:02d}: GT versus END prediction")
                selected.append({"ordinal":ordinal,"storage_index":int(item["storage_index"]),"volume_id":volume,"coordinate_group_id":group,"quality_tier":int(item["quality_tier"]),"file":str(Path("group_alignment")/args.split/filename),"selection":"evenly spaced within split index order; prediction-independent"}); print(f"[D34 qualitative] {args.split} {ordinal}/{len(positions)} storage={int(item['storage_index'])}",flush=True)
        _write_json(out_dir/"qualitative"/"selected_samples.json",{"split":args.split,"samples":selected}); _write_json(out_dir/"qualitative"/"visualization_summary.json",{"figure_contract":{"panels":["original depth","ground-truth aligned","prediction-only groupwise aligned"],"prediction":"all directed raw-score partial-overlap CRF Viterbi paths; GT/RGT is rendered only","style_source":"groupwise_alignment_end/scripts/visualize_end_protocol_v1_alignment.py"},"split":args.split,"count":len(selected)})
        _write_json(out_dir/"evaluation_metadata.json",{"command":[sys.executable,*sys.argv],"started_utc":datetime.now(timezone.utc).isoformat(),"evaluation_protocol":EVALUATION_PROTOCOL,"dataset_root":str(Path(data['root']).resolve()),"source_model_config":source_config,"groups":len(dataset)})
        print(json.dumps({"output_dir":str(out_dir),"PathMAE":summary["path_mae_index"],"F1@2":summary["f1_tolerance_0_20"]["2"],"IoU":summary["iou"],"TransitiveMAE":summary["transitive_mae_index"],"TransitiveP95":summary["transitive_p95_index"]},indent=2),flush=True)
    barrier(distributed)
    if distributed: dist.destroy_process_group()


if __name__=="__main__": main()
