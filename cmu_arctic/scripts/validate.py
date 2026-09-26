#!/usr/bin/env python3
"""Distributed EMA validation over disjoint materialized K4 rows."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Keep this entry point runnable via ``torchrun scripts/validate.py`` from any
# working directory.  In particular, do not rely on a predecessor project's
# editable install or on a launcher's PYTHONPATH export.
_SRC=Path(__file__).resolve().parents[2]/'common'
if str(_SRC) not in sys.path:sys.path.insert(0,str(_SRC))

from cmu_alignment.core import all_gather_object, distributed_cleanup, distributed_context, load_yaml, prepare_run, run_environment, save_yaml, worker_seed
from cmu_alignment.data import DistributedLengthBucketBatchSampler,K4FullUtteranceAlignmentDataset,LengthBucketBatchSampler,collate_k4_full_utterance,load_crf_reference_length
from cmu_alignment.loss import ENDCriterion
from cmu_alignment.model import HierarchicalGroupwiseEND
from cmu_alignment.training import parent_global_mean_weight,summarize_validation,validation_parent_rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--feature-root',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--model-config',default='configs/model/end_cmu.yaml');p.add_argument('--run-root',default='.');p.add_argument('--run-id',required=True);p.add_argument('--batch-size',type=int,default=1);p.add_argument('--num-workers',type=int,default=4);p.add_argument('--ddp',action='store_true');a=p.parse_args();rank,world,_,device=distributed_context(a.ddp)
 try:
  cfg=load_yaml(a.model_config);ref=load_crf_reference_length(a.root)
  if cfg['alignment']['crf_reference_length']!=ref:raise ValueError('L_ref mismatch')
  out=prepare_run(a.run_root,'validation',a.run_id) if rank==0 else Path(a.run_root)/'runs/validation'/a.run_id
  if a.ddp:
   import torch.distributed as dist;dist.barrier()
  ds=K4FullUtteranceAlignmentDataset(a.root,a.feature_root,'val');lengths=[max(x['native_lengths']) for x in ds.rows]
  sampler=DistributedLengthBucketBatchSampler(lengths,a.batch_size,world,rank,20260904,False) if a.ddp else LengthBucketBatchSampler(lengths,a.batch_size,20260904)
  opts=dict(batch_sampler=sampler,num_workers=a.num_workers,pin_memory=device.type=='cuda',collate_fn=collate_k4_full_utterance,worker_init_fn=worker_seed)
  if a.num_workers:opts.update(persistent_workers=True,prefetch_factor=2)
  model=HierarchicalGroupwiseEND(cfg).to(device);state=torch.load(a.checkpoint,map_location=device,weights_only=False);model.load_state_dict(state.get('ema_state_dict') or state['model_state_dict'])
  psd_groups=cfg['loss'].get('group_psd_batch_groups','all')
  loss=ENDCriterion(corridor_radius=cfg['loss']['align_corridor_radius'],lambda_projector=cfg['loss']['lambda_projector'],global_mean_sample_weight=parent_global_mean_weight(ds.rows),crf_reference_length=ref,projector_stride=cfg['loss'].get('projector_stride',1),projector_dtype=torch.float32,allow_no_overlap_output=bool(cfg['loss'].get('allow_no_overlap_output',True)),emission_mode=str(cfg['alignment'].get('crf_emission_mode','legacy')),zero_radius_continuous_target_policy=str(cfg['loss'].get('zero_radius_continuous_target_policy','error')),lambda_group_score=float(cfg['loss'].get('lambda_group_score',0.0)),group_score_radius=int(cfg['loss'].get('group_score_radius',2)),lambda_group_variance=float(cfg['loss'].get('lambda_group_variance',1.0)),group_score_variance_epsilon=float(cfg['loss'].get('group_score_variance_epsilon',1e-4)),lambda_similarity=float(cfg['loss'].get('lambda_similarity',0.0)),similarity_target_radius=int(cfg['loss'].get('similarity_target_radius',0)),similarity_target_sigma=float(cfg['loss'].get('similarity_target_sigma',1.0)),lambda_group_psd=float(cfg['loss'].get('group_psd_weight',0.0)),group_psd_num_bins=int(cfg['loss'].get('group_psd_num_bins',32)),group_psd_batch_groups=None if psd_groups in (None,'all') else int(psd_groups))
  merged={}
  for shard in all_gather_object(validation_parent_rows(model,loss,DataLoader(ds,**opts),device),a.ddp):
   for parent,rows in shard.items():
    merged.setdefault(parent,[]).extend(rows)
  if rank==0:
   result={'validation_weights':'ema',**summarize_validation(merged)};save_yaml(out/'resolved_config.yaml',cfg);(out/'command.txt').write_text(' '.join(sys.argv)+'\n');(out/'environment.json').write_text(json.dumps(run_environment(world,device),indent=2)+'\n');(out/'metrics'/'validation_summary.json').write_text(json.dumps(result,indent=2)+'\n');(out/'run_metadata.json').write_text(json.dumps({'run_id':a.run_id,'run_type':'validation','world_size':world,'unique_parent_prompts':len(merged),'subgroups':sum(map(len,merged.values()))},indent=2)+'\n');print(json.dumps(result,indent=2))
 finally:distributed_cleanup(a.ddp)
if __name__=='__main__':main()
