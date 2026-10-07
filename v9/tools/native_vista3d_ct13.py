"""Native VISTA3D fit128/FPS4 on the identical CT13 full validation scope."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import native_vista3d_pilot as pilot,native_ct13_eval as native,native_sammed3d_ct13 as shared
DEFAULT='output/native_vista3d_ct13_full6094_fit_fps4_20261007'
PROTOCOL=dict(pilot.PROTOCOL,arm='fit_fps4',point_cap_per_plane_role=4,tensor_order='canonical_XYZ')
validate_scope=shared.validate_scope;make_jobs=shared.make_jobs;choose_gpus=shared.choose_gpus
progress=shared.progress

def validate_audit(row):
    a=row['prompt_audit']
    if not a['background_cache_hit'] or a['fg_in_gt_fraction']!=1 or a['bg_in_gt_count']!=0:
        raise ValueError('prompt cache/alignment audit')
    if not row['stages']:raise ValueError('missing prediction stage')
    for s in row['stages']:
        if (s['point_axis_order']!='canonical_XYZ' or not math.isfinite(s['roundtrip_voxels'])
            or s['roundtrip_voxels']>1e-8 or not 0<s['foreground_points']<=8
            or not 0<s['background_points']<=8 or s['gt_used_for_roi'] or not s['encoded_all_selected_points']):
            raise ValueError('physical geometry/point budget audit')
        encoded=set()
        for w in s['patches']:
            if w['forward_calls']!=1 or w['class_vector'] is not None or w['prompt_class'] is not None or w['previous_mask'] is not None or w['gt_used']:
                raise ValueError('non-native or feedback patch input')
            encoded.update(w['encoded_point_indices'])
        if encoded!=set(range(len(s['point_labels']))):raise ValueError('incomplete point encoding')

def prepare(out):
    tasks=native.read(ROOT/'output/native_sam2_medsam2_ct13_full6094_20261007_gridfix/manifest.json')['tasks']
    validate_scope(tasks,native.read(ROOT/'configs/ct13_v10_2_split_20260928.json'))
    for t in tasks:
        if not (native.CACHE/'metadata/shared_adaptive_prompt_v1'/t['source_name']/t['prompt_cache_name']).is_file():
            raise ValueError('missing existing foreground prompt')
    manifest=dict(tasks=tasks,expected=6094,per_source_counts=native.COUNTS,train_overlap=0,protocol=PROTOCOL)
    if (out/'manifest.json').exists():
        if native.read(out/'manifest.json')!=manifest:raise ValueError('changed manifest')
        if not (out/'jobs.json').is_file():raise ValueError('incomplete prepare requires inspection')
        return
    native.write(out/'manifest.json',manifest);metas=[]
    for j in make_jobs(tasks):
        j['output']=str(out/'jobs'/j['name']);path=out/'job_specs'/(j['name']+'.json')
        native.write(path,j);metas.append(dict(name=j['name'],source=j['source'],path=str(path),tasks=len(j['tasks'])))
    native.write(out/'jobs.json',dict(jobs=metas,total_tasks=6094));native.write(out/'protocol.json',PROTOCOL)
    print('PREPARED',len(metas),'whole-case shards;6094 tasks',flush=True)

def signature(job):
    payload=dict(pilot=pilot.signature(job['tasks']),entry=native.sha(Path(__file__)),
                 shared=native.sha(ROOT/'tools/native_sammed3d_ct13.py'),protocol=PROTOCOL,tasks=job['tasks'])
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()

def worker(job_path,uuid):
    import torch
    if str(torch.cuda.get_device_properties(0).uuid).replace('GPU-','')!=uuid.replace('GPU-',''):
        raise ValueError('CUDA UUID mismatch')
    job=native.read(job_path);out=Path(job['output']);sig=signature(job)
    protocol=dict(signature=sig,protocol=PROTOCOL,weight_sha256=pilot.WEIGHT_SHA)
    if (out/'protocol.json').exists():
        if native.read(out/'protocol.json')!=protocol:raise ValueError('changed-code resume refused')
    else:native.write(out/'protocol.json',protocol)
    rows=native.read(out/'all_metrics_live.json')['per_case_class'] if (out/'all_metrics_live.json').exists() else []
    done={native.key(r) for r in rows};native.validate_rows(rows,[t for t in job['tasks'] if native.key(t) in done])
    for r in rows:validate_audit(r)
    if len(rows)<len(job['tasks']):
        model=pilot.load_model();base=native.base_module();loader=native.CaseLoader(base);scorer=base.metric_module()
        split=native.read(ROOT/'configs/ct13_v10_2_split_20260928.json');case=None;normalized=None
        for t in job['tasks']:
            if native.key(t) in done:continue
            start=time.time();ct,gt,fg,bg,spacing,audit=loader(t,split,'scribble',str(pilot.shared.BG))
            case_key=(t['source_name'],t['case_id'])
            if case!=case_key:normalized=pilot.normalize(ct);case=case_key
            pred,stages=pilot.predict(model,normalized,fg,bg,spacing,audit['canonical_prompt_slices'],'fit_fps4')
            metrics=scorer.binary_prompt_metrics_from_masks(torch.from_numpy(pred),torch.from_numpy(gt),tuple(spacing))
            row=dict(t,**{m:float(metrics[m]) for m in native.METRICS},seconds=time.time()-start,
                prompt_audit=audit,stages=stages,gpu_uuid=uuid)
            validate_audit(row);rows.append(row);done.add(native.key(t))
            native.validate_rows(rows,[t for t in job['tasks'] if native.key(t) in done])
            native.write(out/'all_metrics_live.json',dict(per_case_class=rows,signature=sig))
            print('DONE',job['name'],len(rows),'/',len(job['tasks']),t['case_id'],t['class_name'],'Dice',row['dice'],'seconds',row['seconds'],flush=True)
    native.validate_rows(rows,job['tasks'])
    native.write(out/'complete_metrics.json',dict(per_case_class=rows,signature=sig,
        summary={m:float(np.mean([r[m] for r in rows])) for m in native.METRICS}))

def merge(out):
    rows=[]
    for meta in native.read(out/'jobs.json')['jobs']:
        j=native.read(meta['path']);d=native.read(Path(j['output'])/'complete_metrics.json')
        if d['signature']!=signature(j):raise ValueError('changed merge signature')
        native.validate_rows(d['per_case_class'],j['tasks']);rows+=d['per_case_class']
    native.validate_rows(rows,native.read(out/'manifest.json')['tasks'])
    if dict(Counter(r['source_name'] for r in rows))!=native.COUNTS:raise ValueError('source coverage')
    for r in rows:validate_audit(r)
    native.write(out/'merged_metrics.json',dict(per_case_class=rows,protocol=PROTOCOL,weight_sha256=pilot.WEIGHT_SHA,
        summary={m:float(np.mean([r[m] for r in rows])) for m in native.METRICS},
        per_source={s:dict(count=native.COUNTS[s],**{m:float(np.mean([r[m] for r in rows if r['source_name']==s])) for m in native.METRICS}) for s in native.COUNTS}))

def schedule(out,gpus):
    import fcntl
    lock=(out/'scheduler.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    metas=native.read(out/'jobs.json')['jobs'];active={};failures=Counter()
    while True:
        done={m['name'] for m in metas if (out/'jobs'/m['name']/'complete_metrics.json').exists()}
        for name,item in list(active.items()):
            rc=item['p'].poll()
            if rc is None:continue
            item['log'].close();del active[name]
            if rc:
                failures[name]+=1;oom='out of memory' in item['path'].read_text().lower()
                print('WORKER_FAILED',name,rc,'OOM',oom,flush=True)
                if not oom or failures[name]>=3:
                    native.write(out/'failure.json',dict(job=name,returncode=rc,oom=oom,log=str(item['path'])))
                    for v in active.values():v['p'].wait();v['log'].close()
                    raise RuntimeError('VISTA worker failed: '+name)
        pending=[m for m in metas if m['name'] not in done and m['name'] not in active]
        if not pending and not active:
            merge(out);native.write(out/'status.json',dict(state='complete',**progress(out)));return
        inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
        for index,uuid in choose_gpus(inventory,gpus,{v['uuid'] for v in active.values()}):
            if not pending:break
            j=pending.pop(0);path=out/'logs'/(j['name']+'.{}.try{}.log'.format(time.strftime('%Y%m%d_%H%M%S'),failures[j['name']]))
            path.parent.mkdir(exist_ok=True);log=path.open('x')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='2')
            p=subprocess.Popen([sys.executable,str(Path(__file__)),'--worker','--job',j['path'],'--uuid',uuid],cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT)
            active[j['name']]=dict(p=p,log=log,path=path,uuid=uuid)
            print('START',j['name'],'GPU',index,uuid,'PID',p.pid,'tasks',j['tasks'],flush=True)
        native.write(out/'status.json',dict(state='running',time=time.strftime('%Y-%m-%d %H:%M:%S'),scheduler_pid=os.getpid(),
            active={k:dict(pid=v['p'].pid,uuid=v['uuid']) for k,v in active.items()},**progress(out)))
        time.sleep(20)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prepare','worker','schedule','merge','status'):p.add_argument('--'+name,action='store_true')
    p.add_argument('--output',default=DEFAULT);p.add_argument('--job');p.add_argument('--uuid');p.add_argument('--gpus',default='0,2,3,4,5,6')
    a=p.parse_args();out=ROOT/a.output
    if a.prepare:prepare(out)
    elif a.worker:worker(a.job,a.uuid)
    elif a.schedule:schedule(out,{int(i) for i in a.gpus.split(',')})
    elif a.merge:merge(out)
    elif a.status:print(json.dumps(progress(out),indent=2))
    else:p.error('select prepare/worker/schedule/merge/status')

if __name__=='__main__':main()
