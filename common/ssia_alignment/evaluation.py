"""Frozen D3--D4 path metrics and END qualitative rendering.

This module intentionally mirrors the mature END formal-test contracts: raw
CRF-score Viterbi paths are decoded once, ground truth is used only afterwards
for metrics/rendering, and qualitative samples are prediction-independent.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


F1_TOLERANCES = tuple(range(21))
EVALUATION_PROTOCOL = "COMMON_RGT_CORRESPONDENCE_METRICS_V1_INDEX_POINTS"


def _bounds(path: np.ndarray) -> tuple[float,float]:
    active=np.flatnonzero(np.asarray(path)>=0)
    return (-1.,-1.) if len(active)==0 else (float(active[0]),float(active[-1]))


def _target_bounds(path: np.ndarray, active: np.ndarray) -> tuple[float,float]:
    return (-1.,-1.) if not active.any() else (float(np.min(path[active])),float(np.max(path[active])))


def _iou(left: tuple[float,float],right: tuple[float,float]) -> float:
    if min(*left,*right)<0: return 0.0
    return float(max(0.,min(left[1],right[1])-max(left[0],right[0])+1.)/max(1.,max(left[1],right[1])-min(left[0],right[0])+1.))


def _compose_continuously(qij: np.ndarray,qjk: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Evaluate q_jk(q_ij(p)) without integer casting or gap bridging."""
    qij=np.asarray(qij,np.float32); qjk=np.asarray(qjk,np.float32)
    value=np.full_like(qij,-1.); inside=(qij>=0)&(qij<=len(qjk)-1)
    lo=np.floor(np.clip(qij,0,len(qjk)-1)).astype(np.int64); hi=np.ceil(np.clip(qij,0,len(qjk)-1)).astype(np.int64)
    usable=inside&(qjk[lo]>=0)&(qjk[hi]>=0)
    weight=qij-lo
    value[usable]=(1.-weight[usable])*qjk[lo[usable]]+weight[usable]*qjk[hi[usable]]
    return value,usable


def _common_label_domain(valid: np.ndarray) -> np.ndarray:
    """One shared GT-correspondence domain for every metric in a K-curve group.

    A source position is evaluable precisely when its ground-truth label has a
    real correspondence on every other curve in the group.  This constructs
    the all-K common RGT/label region directly from the same GT labels used by
    PathMAE, F1, IoU, and transitive evaluation.
    """
    valid=np.asarray(valid,bool)
    if valid.ndim!=3 or valid.shape[0]!=valid.shape[1]:
        raise ValueError(f"valid labels must be [K,K,L], got {valid.shape}")
    curves=valid.shape[0]; off_diagonal=~np.eye(curves,dtype=bool)
    return np.all(valid[off_diagonal].reshape(curves,curves-1,valid.shape[-1]),axis=1)


def _pair_metric(pred: np.ndarray,gt: np.ndarray,common_source: np.ndarray) -> dict[str,Any]:
    """One directed pair score on the sole all-K GT-correspondence domain."""
    pred=np.asarray(pred,np.float32); gt=np.asarray(gt,np.float32); common_source=np.asarray(common_source,bool)
    if pred.shape!=gt.shape or pred.shape!=common_source.shape:
        raise ValueError("prediction, GT, and common-label mask must share [L] shape")
    active=(pred>=0)&common_source; matched=active
    row:dict[str,Any]={"path_mae_index":float(np.abs(pred[matched]-gt[matched]).mean()) if matched.any() else float("nan")}
    for tolerance in F1_TOLERANCES:
        correct=matched&(np.abs(pred-gt)<=tolerance); tp=int(correct.sum()); fp=int((active&~correct).sum())
        fn=int((common_source&~correct).sum())
        row[f"f1_tol_{tolerance:02d}_index"]=2.*tp/max(1,2*tp+fp+fn)
    # IoU is the target interval reached by mapping this same common GT domain
    # from source to target, compared with that domain's GT target interval.
    # It is not an IoU between the two full raw curves' independent extents.
    target_pred,target_gt=_target_bounds(pred,active),_target_bounds(gt,common_source)
    row["iou"]=_iou(target_pred,target_gt)
    return row


def _transitive_metric(qij: np.ndarray,qjk: np.ndarray,gtik: np.ndarray,common_source: np.ndarray) -> tuple[dict[str,Any],np.ndarray]:
    """Propagation i→j→k compared directly with ground-truth i→k."""
    composed,composed_valid=_compose_continuously(qij,qjk)
    gtik=np.asarray(gtik,np.float32); domain=composed_valid&np.asarray(common_source,bool); errors=np.abs(composed[domain]-gtik[domain]).astype(np.float32,copy=False)
    return {"transitive_mae_index":float(errors.mean()) if len(errors) else float("nan"),"transitive_p95_index":float(np.percentile(errors,95)) if len(errors) else float("nan"),"transitive_valid_points":int(len(errors))},errors


def evaluate_predictions(
    paths: np.ndarray,gt: np.ndarray,valid: np.ndarray,metadata: list[dict[str,Any]],
) -> tuple[list[dict[str,Any]],list[dict[str,Any]],np.ndarray]:
    """Return common-domain per-pair/per-triple rows and the error pool."""
    paths=np.asarray(paths,np.float32); gt=np.asarray(gt,np.float32); valid=np.asarray(valid,bool)
    if paths.ndim!=4 or paths.shape[1:3]!=(8,8) or paths.shape!=gt.shape or valid.shape!=gt.shape:
        raise ValueError(f"expected matching [B,8,8,L] paths/GT/valid arrays, got {paths.shape}, {gt.shape}, {valid.shape}")
    if len(metadata)!=len(paths): raise ValueError("metadata/path batch mismatch")
    pair_rows:list[dict[str,Any]]=[]; triple_rows:list[dict[str,Any]]=[]; chunks:list[np.ndarray]=[]
    for item,(pred,truth,truth_valid,common) in enumerate(zip(paths,gt,valid,metadata,strict=True)):
        del item
        common_source=_common_label_domain(truth_valid)
        for source in range(8):
            for target in range(8):
                if source==target: continue
                row=_pair_metric(pred[source,target],truth[source,target],common_source[source]); row.update(common|{"source":source,"target":target}); pair_rows.append(row)
        for source in range(8):
            for via in range(8):
                for target in range(8):
                    if len({source,via,target})<3: continue
                    row,errors=_transitive_metric(pred[source,via],pred[via,target],truth[source,target],common_source[source])
                    row.update(common|{"source":source,"via":via,"target":target}); triple_rows.append(row)
                    if len(errors): chunks.append(errors)
    return pair_rows,triple_rows,np.concatenate(chunks) if chunks else np.empty(0,dtype=np.float32)


def _hierarchical_mean(rows:list[dict[str,Any]],key:str) -> float:
    grouped:dict[tuple[str,str],list[float]]=defaultdict(list)
    for row in rows:
        value=float(row[key])
        if np.isfinite(value): grouped[(str(row["volume_id"]),str(row["coordinate_group_id"]))].append(value)
    parents:dict[str,list[float]]=defaultdict(list)
    for (parent,_),values in grouped.items(): parents[parent].append(float(np.mean(values)))
    return float(np.mean([np.mean(values) for values in parents.values()])) if parents else float("nan")


def summarize(pair_rows:list[dict[str,Any]],triple_rows:list[dict[str,Any]],transitive_stats:dict[str,Any],*,groups:int) -> dict[str,Any]:
    if not pair_rows: raise ValueError("cannot summarize zero evaluated groups")
    f1={str(tolerance):_hierarchical_mean(pair_rows,f"f1_tol_{tolerance:02d}_index") for tolerance in F1_TOLERANCES}
    return {"evaluation_protocol":EVALUATION_PROTOCOL,"coordinate_unit":"index points","evaluation_domain":"source positions with GT correspondence to every other curve in the K=8 group; all metrics use this identical domain","groups":groups,"directed_pairs":len(pair_rows),"ordered_triples":len(triple_rows),"pairwise_aggregation":"common-label position→pair→group→volume→dataset (equal pair weight)","groupwise_aggregation":"all common-label pointwise transitive errors pooled across ordered triples; MAE and P95 use the identical pool","path_mae_index":_hierarchical_mean(pair_rows,"path_mae_index"),"f1_tolerance_0_20":f1,"iou":_hierarchical_mean(pair_rows,"iou"),"transitive_error_points":int(transitive_stats["transitive_error_points"]),"transitive_mae_index":float(transitive_stats["transitive_mae_index"]),"transitive_p95_index":float(transitive_stats["transitive_p95_index"])}


# The functions below are copied with only naming/data adaptation from
# visualize_end_protocol_v1_alignment.py in the prior END project.
def _shared_gt_coordinate(rgt: np.ndarray) -> np.ndarray:
    coordinate=np.maximum.accumulate(np.asarray(rgt,dtype=np.float64),axis=1); lower,upper=float(coordinate.min()),float(coordinate.max())
    return (coordinate-lower)*(511./max(upper-lower,1.e-8))


def _prediction_coordinate(paths: np.ndarray) -> np.ndarray:
    paths=np.asarray(paths,np.float64); curves,length=paths.shape[0],paths.shape[-1]
    path_copy=paths.copy(); path_copy[np.arange(curves),np.arange(curves)]=np.arange(length,dtype=np.float64)
    path_copy[path_copy<0]=np.nan
    return np.maximum.accumulate(np.nanmean(path_copy,axis=1),axis=1)


def _compress_mapping(coordinate: np.ndarray,positions: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    order=np.argsort(coordinate,kind="stable"); coordinate,positions=coordinate[order],positions[order]
    unique,inverse=np.unique(coordinate,return_inverse=True)
    return unique,np.asarray([positions[inverse==index].mean() for index in range(len(unique))],dtype=np.float64)


def _inverse_warp(curve: np.ndarray,coordinate: np.ndarray,lattice: np.ndarray) -> np.ndarray:
    valid=np.isfinite(coordinate); output=np.full_like(lattice,np.nan,dtype=np.float64)
    if valid.sum()<2: return output
    knots,positions=_compress_mapping(coordinate[valid],np.flatnonzero(valid).astype(np.float64)); inside=(lattice>=knots[0])&(lattice<=knots[-1])
    source=np.interp(lattice[inside],knots,positions); output[inside]=np.interp(source,np.arange(len(curve),dtype=np.float64),curve)
    return output


def _horizons(gt_coordinate: np.ndarray,count:int=20) -> tuple[np.ndarray,np.ndarray]:
    lower=max(float(np.nanmin(row)) for row in gt_coordinate); upper=min(float(np.nanmax(row)) for row in gt_coordinate)
    values=np.linspace(lower,upper,count); positions=np.arange(gt_coordinate.shape[1],dtype=np.float64)
    return values,np.stack([np.interp(values,row,positions) for row in gt_coordinate],axis=1)


def _draw_tracks(axis,curves: np.ndarray,vertical: np.ndarray,horizons: np.ndarray,title: str,ylabel: str|None=None) -> None:
    palette=plt.get_cmap("tab10").colors; accent=set(np.linspace(0,len(horizons)-1,min(10,len(horizons)),dtype=int).tolist())
    for curve_index,curve in enumerate(curves):
        centered=curve-np.nanmedian(curve); scale=max(float(np.nanpercentile(np.abs(centered),98)),1.e-8)
        axis.plot(curve_index+.36*centered/scale,vertical,color="#252525",linewidth=.62,zorder=2)
        for horizon_index,y in enumerate(horizons[:,curve_index]):
            axis.plot([curve_index-.38,curve_index+.38],[y,y],color=palette[horizon_index%len(palette)] if horizon_index in accent else "#B7B7B7",alpha=.90 if horizon_index in accent else .48,linewidth=.72 if horizon_index in accent else .42,zorder=1)
    axis.set_xlim(-.65,curves.shape[0]-.35); axis.set_ylim(511,0); axis.set_xticks(range(curves.shape[0]),[f"C{index+1}" for index in range(curves.shape[0])]); axis.xaxis.tick_top(); axis.set_title(title,fontsize=9,pad=18)
    if ylabel: axis.set_ylabel(ylabel)
    axis.grid(axis="y",color="#E5E5E5",linewidth=.45)


def draw_group_alignment(output,curves: np.ndarray,rgt: np.ndarray,paths: np.ndarray,title: str) -> None:
    """Render the legacy three-panel qualitative figure for one K=8 group."""
    curves=np.asarray(curves,np.float64); gt_coordinate=_shared_gt_coordinate(rgt); predicted_coordinate=_prediction_coordinate(paths); lattice=np.arange(curves.shape[1],dtype=np.float64)
    horizons,original_markers=_horizons(gt_coordinate); gt_markers=np.repeat(horizons[:,None],curves.shape[0],axis=1)
    predicted_markers=np.asarray([[np.interp(position,lattice,predicted_coordinate[curve]) for curve,position in enumerate(row)] for row in original_markers],dtype=np.float64)
    gt_curves=np.asarray([_inverse_warp(curve,gt_coordinate[index],lattice) for index,curve in enumerate(curves)])
    predicted_curves=np.asarray([_inverse_warp(curve,predicted_coordinate[index],lattice) for index,curve in enumerate(curves)])
    figure,axes=plt.subplots(1,3,figsize=(11.,5.2),sharey=True,constrained_layout=True)
    _draw_tracks(axes[0],curves,lattice,original_markers,"(a) Original depth domain","Relative depth index")
    _draw_tracks(axes[1],gt_curves,lattice,gt_markers,"(b) Ground-truth aligned domain")
    _draw_tracks(axes[2],predicted_curves,lattice,predicted_markers,"(c) Predicted groupwise-aligned domain")
    figure.suptitle(title,fontsize=9.5,y=1.02); output.parent.mkdir(parents=True,exist_ok=True); figure.savefig(output,dpi=350,bbox_inches="tight",facecolor="white"); plt.close(figure)
