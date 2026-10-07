"""Isolated, resumable SAM-Med3D Turbo full CT13 validation, physical FPS4."""
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import native_sammed3d_pilot as pilot,native_ct13_eval as native
DEFAULT='output/native_sammed3d_turbo_ct13_full6094_fps4_20261007'
PROTOCOL=dict(model='SAM-Med3D Turbo',arm='fps4',image_size=128,threshold=.5,
    tensor_order='canonical_XYZ',point_cap_per_plane_role=4,point_spacing_mm=3.,
    promptgen=False,semantic=False,points_once=True,previous_logits='zero',
    roi='prompt_8mm_20pct_connected_expand50_cap100',metrics=list(native.METRICS))

def validate_scope(tasks,split,counts=None):
    counts=native.COUNTS if counts is None else counts
    if dict(Counter(t['source_name'] for t in tasks))!=counts:raise ValueError('source count mismatch')
    if len({native.key(t) for t in tasks})!=len(tasks):raise ValueError('duplicate task')
    for source in counts:
        dataset=split['datasets'][source]
        val={r['case_id'] for r in dataset['val']};train={r['case_id'] for r in dataset['train']}
        if any(t['case_id'] not in val or t['case_id'] in train for t in tasks if t['source_name']==source):
            raise ValueError('validation membership/training overlap')

def make_jobs(tasks):
    jobs=[]
    for source in sorted({t['source_name'] for t in tasks}):
        rows=[t for t in tasks if t['source_name']==source]
        for i,part in enumerate(native.case_parts(rows,64 if source=='totalseg' else (24 if source=='amos' else 4))):
            jobs.append(dict(name='{}_{:03d}'.format(source,i),source=source,tasks=part))
    return sorted(jobs,key=lambda j:(j['source']=='totalseg',j['source']=='amos',len(j['tasks']),j['name']))

def validate_audit(row):
    a=row['prompt_audit']
    if not a['background_cache_hit'] or a['fg_in_gt_fraction']!=1. or a['bg_in_gt_count']!=0:
        raise ValueError('prompt cache/alignment audit')
    if not row['stages']:raise ValueError('missing prediction stages')
    for s in row['stages']:
        if (s['native_tensor_axis_order']!='canonical_XYZ' or not math.isfinite(s['roundtrip_voxels'])
            or s['roundtrip_voxels']>1e-8 or not 0<s['foreground_points']<=8
            or not 0<s['background_points']<=8 or s['gt_used_for_roi'] or s['decoder_calls']!=1):
            raise ValueError('native XYZ/point budget/ROI audit')

def choose_gpus(inventory,allowed,busy):
    chosen=[]
    for line in inventory.strip().splitlines():
        index,uuid,free=[s.strip() for s in line.split(',')]
        if int(index) in allowed and int(index) not in (1,7) and uuid not in busy and int(free)>=24000:
            chosen.append((int(index),uuid))
    return chosen

def prepare(out):
    tasks=native.read(ROOT/'output/native_sam2_medsam2_ct13_full6094_20261007_gridfix/manifest.json')['tasks']
    split=native.read(ROOT/'configs/ct13_v10_2_split_20260928.json');validate_scope(tasks,split)
    for t in tasks:
        if not (native.CACHE/'metadata/shared_adaptive_prompt_v1'/t['source_name']/t['prompt_cache_name']).is_file():
            raise ValueError('missing existing foreground prompt')
    manifest=dict(tasks=tasks,expected=6094,per_source_counts=native.COUNTS,train_overlap=0,protocol=PROTOCOL)
    if (out/'manifest.json').exists():
        if native.read(out/'manifest.json')!=manifest:raise ValueError('changed manifest')
        return
    native.write(out/'manifest.json',manifest);metas=[]
    for j in make_jobs(tasks):
        j['output']=str(out/'jobs'/j['name']);path=out/'job_specs'/(j['name']+'.json')
        native.write(path,j);metas.append(dict(name=j['name'],source=j['source'],path=str(path),tasks=len(j['tasks'])))
    native.write(out/'jobs.json',dict(jobs=metas,total_tasks=6094));native.write(out/'protocol.json',PROTOCOL)
    print('PREPARED',len(metas),'whole-case shards, 6094 tasks',flush=True)

def signature(job):
    files=[Path(__file__),ROOT/'tools/native_sammed3d_pilot.py',ROOT/'infer/native_sammed3d_prompts.py',
        ROOT/'tools/native_ct13_eval.py',ROOT/'tools/native_sam2_physical_eval.py',
        ROOT/'infer/native_sam2_physical_prompts.py',ROOT/'utils/metrics.py',ROOT/'configs/ct13_v10_2_split_20260928.json']
    files+=sorted((pilot.CODE/'segment_anything').rglob('*.py'))
    code=hashlib.sha256(''.join(native.sha(p) for p in files).encode()).hexdigest()
    prompts=[native.sha(native.CACHE/'metadata/shared_adaptive_prompt_v1'/t['source_name']/t['prompt_cache_name']) for t in job['tasks']]
    return hashlib.sha256(json.dumps(dict(protocol=PROTOCOL,tasks=job['tasks'],code=code,
        prompts=prompts,weight=pilot.WEIGHT_SHA),sort_keys=True).encode()).hexdigest()

def worker(job_path,uuid):
    import torch
    if str(torch.cuda.get_device_properties(0).uuid).replace('GPU-','')!=uuid.replace('GPU-',''):
        raise ValueError('GPU UUID/CUDA mapping mismatch')
    job=native.read(job_path);out=Path(job['output']);sig=signature(job)
    protocol=dict(signature=sig,weight_sha256=pilot.WEIGHT_SHA,protocol=PROTOCOL)
    if (out/'protocol.json').exists():
        if native.read(out/'protocol.json')!=protocol:raise ValueError('changed-code/protocol resume refused')
    else:native.write(out/'protocol.json',protocol)
    rows=native.read(out/'all_metrics_live.json')['per_case_class'] if (out/'all_metrics_live.json').exists() else []
    done={native.key(r) for r in rows};expected=[t for t in job['tasks'] if native.key(t) in done]
    native.validate_rows(rows,expected)
    for r in rows:validate_audit(r)
    if len(rows)<len(job['tasks']):
        model=pilot.load_model();base=native.base_module();loader=native.CaseLoader(base);scorer=base.metric_module()
        split=native.read(ROOT/'configs/ct13_v10_2_split_20260928.json')
        for t in job['tasks']:
            if native.key(t) in done:continue
            start=time.time();ct,gt,fg,bg,spacing,audit=loader(t,split,'scribble',str(pilot.BG))
            if not audit['background_cache_hit']:raise ValueError('expected existing background cache')
            pred,stages=pilot.predict(model,ct,fg,bg,spacing,audit['canonical_prompt_slices'],'fps4')
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

def progress(out):
    total=0;complete=0;sources=Counter()
    for meta in native.read(out/'jobs.json')['jobs']:
        path=out/'jobs'/meta['name'];file=path/'complete_metrics.json'
        if file.exists():complete+=1
        else:file=path/'all_metrics_live.json'
        if file.exists():
            rows=native.read(file)['per_case_class'];total+=len(rows);sources.update(r['source_name'] for r in rows)
    return dict(completed_jobs=complete,completed_tasks=total,total_tasks=6094,per_source=dict(sources))

def merge(out):
    rows=[]
    for meta in native.read(out/'jobs.json')['jobs']:
        j=native.read(meta['path']);d=native.read(Path(j['output'])/'complete_metrics.json')
        if d['signature']!=signature(j):raise ValueError('merge signature changed')
        native.validate_rows(d['per_case_class'],j['tasks']);rows+=d['per_case_class']
    native.validate_rows(rows,native.read(out/'manifest.json')['tasks'])
    if dict(Counter(r['source_name'] for r in rows))!=native.COUNTS:raise ValueError('merge source coverage')
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
                failures[name]+=1;log=item['path'].read_text();oom='out of memory' in log.lower()
                print('WORKER_FAILED',name,rc,'OOM',oom,flush=True)
                if not oom or failures[name]>=3:
                    native.write(out/'failure.json',dict(job=name,returncode=rc,oom=oom,log=str(item['path'])))
                    for v in active.values():v['p'].wait();v['log'].close()
                    raise RuntimeError('Turbo worker failed: '+name)
        pending=[m for m in metas if m['name'] not in done and m['name'] not in active]
        if not pending and not active:
            merge(out);native.write(out/'status.json',dict(state='complete',**progress(out)));return
        inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
        for index,uuid in choose_gpus(inventory,gpus,{v['uuid'] for v in active.values()}):
            if not pending:break
            j=pending.pop(0);path=out/'logs'/(j['name']+'.try{}.log'.format(failures[j['name']]))
            path.parent.mkdir(exist_ok=True);log=path.open('w')
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
