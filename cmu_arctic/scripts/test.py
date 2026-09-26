#!/usr/bin/env python3
"""Distributed frozen-EMA evaluation on either validation or test rows."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

# Do not accidentally evaluate with a predecessor project's editable install.
_SRC=Path(__file__).resolve().parents[2]/'common'
if str(_SRC) not in sys.path:sys.path.insert(0,str(_SRC))

from cmu_alignment.core import distributed_cleanup,distributed_context,load_yaml,prepare_run,run_environment,save_yaml,worker_seed
from cmu_alignment.data import DistributedLengthBucketBatchSampler,K4FullUtteranceAlignmentDataset,LengthBucketBatchSampler,collate_k4_full_utterance,load_crf_reference_length
from cmu_alignment.evaluation import evaluate_batch_d34,summarize_d34_index_metrics
from cmu_alignment.model import HierarchicalGroupwiseEND

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--feature-root',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--model-config',default='configs/model/end_cmu.yaml');p.add_argument('--run-root',default='.');p.add_argument('--run-id',required=True);p.add_argument('--split',choices=('val','test'),default='test');p.add_argument('--weights',choices=('ema','online'),default='ema');p.add_argument('--batch-size',type=int,default=1);p.add_argument('--num-workers',type=int,default=4);p.add_argument('--ddp',action='store_true');a=p.parse_args();rank,world,_,device=distributed_context(a.ddp)
 try:
  cfg=load_yaml(a.model_config);ref=load_crf_reference_length(a.root)
  if cfg['alignment']['crf_reference_length']!=ref:raise ValueError('L_ref mismatch')
  run_type='validation' if a.split=='val' else 'test'
  out=prepare_run(a.run_root,run_type,a.run_id) if rank==0 else Path(a.run_root)/'runs'/run_type/a.run_id
  if a.ddp:
   import torch.distributed as dist;dist.barrier()
  ds=K4FullUtteranceAlignmentDataset(a.root,a.feature_root,a.split);lengths=[max(x['native_lengths']) for x in ds.rows]
  sampler=DistributedLengthBucketBatchSampler(lengths,a.batch_size,world,rank,20260905,False) if a.ddp else LengthBucketBatchSampler(lengths,a.batch_size,20260905)
  opts=dict(batch_sampler=sampler,num_workers=a.num_workers,pin_memory=device.type=='cuda',collate_fn=collate_k4_full_utterance,worker_init_fn=worker_seed)
  if a.num_workers:opts.update(persistent_workers=True,prefetch_factor=2)
  model=HierarchicalGroupwiseEND(cfg).to(device);state=torch.load(a.checkpoint,map_location=device,weights_only=False)
  if a.weights=='ema':
   if state.get('ema_state_dict') is None: raise RuntimeError('checkpoint has no EMA state')
   model.load_state_dict(state['ema_state_dict'])
  else:model.load_state_dict(state['model_state_dict'])
  rows=[];triples=[];error_chunks=[]
  for batch in DataLoader(ds,**opts):
   result=evaluate_batch_d34(model,batch,device,ref,allow_no_overlap_output=bool(cfg['loss'].get('allow_no_overlap_output',True)),emission_mode=str(cfg['alignment'].get('crf_emission_mode','legacy')),similarity_eps=float(cfg['alignment'].get('similarity_eps',1e-6)),normalization_eps=float(cfg['alignment'].get('normalization_eps',1e-8)))
   rows.extend(result.pair_rows);triples.extend(result.triple_rows)
   if len(result.transitive_errors):error_chunks.append(result.transitive_errors)
  shard=out/'predictions'/f'predictions_rank{rank:03d}.jsonl';shard.write_text(''.join(json.dumps(row)+'\n' for row in rows))
  triple_shard=out/'predictions'/f'transitive_metrics_rank{rank:03d}.jsonl';triple_shard.write_text(''.join(json.dumps(row)+'\n' for row in triples))
  np.save(out/'predictions'/f'transitive_errors_rank{rank:03d}.npy',np.concatenate(error_chunks).astype(np.float32,copy=False) if error_chunks else np.empty(0,dtype=np.float32))
  if a.ddp:
   import torch.distributed as dist;dist.barrier()
  if rank==0:
   all_rows=[]
   for shard in sorted((out/'predictions').glob('predictions_rank*.jsonl')):
    all_rows.extend(json.loads(line) for line in shard.read_text().splitlines() if line)
   all_triples=[]
   for shard in sorted((out/'predictions').glob('transitive_metrics_rank*.jsonl')):
    all_triples.extend(json.loads(line) for line in shard.read_text().splitlines() if line)
   error_shards=sorted((out/'predictions').glob('transitive_errors_rank*.npy'))
   errors=np.concatenate([np.load(shard) for shard in error_shards]) if error_shards else np.empty(0,dtype=np.float32)
   keys=[(r['k4_group_id'],r['source'],r['target']) for r in all_rows]
   if len(keys)!=len(set(keys)):raise RuntimeError('duplicate test subgroup/pair across rank shards')
   if len(all_rows)!=len(ds)*12:raise RuntimeError(f'incomplete {a.split} coverage: {len(all_rows)} != {len(ds)*12}')
   triple_keys=[(r['k4_group_id'],r['source'],r['via'],r['target']) for r in all_triples]
   if len(triple_keys)!=len(set(triple_keys)):raise RuntimeError('duplicate test subgroup/triple across rank shards')
   if len(all_triples)!=len(ds)*24:raise RuntimeError(f'incomplete {a.split} triple coverage: {len(all_triples)} != {len(ds)*24}')
   (out/'predictions'/'viterbi_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in all_rows))
   (out/'predictions'/'transitive_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in all_triples))
   scalar=[{k:v for k,v in r.items() if k!='predicted_q'} for r in all_rows]
   with (out/'metrics'/'subgroup_metrics.csv').open('w',newline='') as h:
    writer=csv.DictWriter(h,fieldnames=scalar[0]);writer.writeheader();writer.writerows(scalar)
   with (out/'metrics'/'pairwise_metrics.csv').open('w',newline='') as h:
    writer=csv.DictWriter(h,fieldnames=scalar[0]);writer.writeheader();writer.writerows(scalar)
   with (out/'metrics'/'transitive_metrics.csv').open('w',newline='') as h:
    writer=csv.DictWriter(h,fieldnames=all_triples[0]);writer.writeheader();writer.writerows(all_triples)
   summary=summarize_d34_index_metrics(all_rows,all_triples,errors,groups=len(ds))
   summary.update({'subgroup_pairs':len(all_rows),'parent_count':len({r['parent_prompt_id'] for r in all_rows})})
   (out/'metrics'/'f1_tolerance_0_20.json').write_text(json.dumps({'coordinate_unit':'native MFCC frame index','f1_tolerance_0_20':summary['f1_tolerance_0_20']},indent=2)+'\n')
   metadata={'run_id':a.run_id,'run_type':run_type,'split':a.split,'weights':a.weights,'world_size':world,'subgroups':len(ds),'parent_count':len({row['parent_prompt_id'] for row in all_rows}),'checkpoint':str(a.checkpoint),'evaluation_protocol':summary['evaluation_protocol'],'coordinate_unit':summary['coordinate_unit']}
   save_yaml(out/'resolved_config.yaml',cfg);(out/'command.txt').write_text(' '.join(sys.argv)+'\n');(out/'environment.json').write_text(json.dumps(run_environment(world,device),indent=2)+'\n');(out/'metrics'/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'run_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(summary,indent=2))
 finally:distributed_cleanup(a.ddp)
if __name__=='__main__':main()
