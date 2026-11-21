"""Matched axial background point-budget comparison on a frozen MAGIC subset."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

ENTRY = Path(__file__).resolve()
ROOT = ENTRY.parents[1]
sys.path.insert(0, str(ROOT))
from tools import amos_prompt_matrix_native as native_predict
from tools import magic_baseline_prompt_matrix as magic

m, n = magic.matrix, magic.native
OUT = ROOT / 'output/magic_baseline_bg_budget25_20261121'
ARMS = dict.fromkeys(('sam2', 'medsam2'), ('fg2_bg2', 'fg2_bg4'))
OLD_PREDICTOR = native_predict.NativePredictor
ORIGINAL_CODE_BINDING = m.code_binding
ORIGINAL_GPU_INVENTORY = m.gpu_inventory
MIN_MM, RADIUS_MM = 3., 12.


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def separated(a, b, spacing, mm=MIN_MM):
    return sum(((u-v)*s)**2 for u,v,s in zip(a,b,spacing,strict=True)) >= mm*mm-1e-9


def rank_frame(mask, anchors, fg_xy, planes, yx_bounds, spacing):
    """Grow one pool on the registered planes; return the ordered prefix of <=4."""
    (y0,y1),(x0,x1) = yx_bounds
    coronal, sagittal = planes
    pitch_y,pitch_x = spacing
    dy,dx = int(RADIUS_MM/pitch_y),int(RADIUS_MM/pitch_x)
    fg_yx = [(y,x) for x,y in fg_xy]
    candidates = set()
    for y,x in anchors:
        if x == coronal:
            candidates.update((t,x) for t in range(max(y0,y-dy),min(y1,y+dy+1)))
        if y == sagittal:
            candidates.update((y,t) for t in range(max(x0,x-dx),min(x1,x+dx+1)))
    pool = sorted(p for p in candidates if y0<=p[0]<y1 and x0<=p[1]<x1 and not mask[p]
                  and all(separated(p,q,spacing) for q in fg_yx))
    ranked = []
    for p in anchors:
        if p in pool and all(separated(p,q,spacing) for q in ranked):
            ranked.append(p)
    while len(ranked)<4:
        remaining = [p for p in pool if all(separated(p,q,spacing) for q in ranked)]
        if not remaining:
            break
        def spacing_score(p):
            return min(sum(((u-v)*s)**2 for u,v,s in zip(p,q,spacing,strict=True))
                       for q in (ranked or anchors))
        ranked.append(max(remaining,key=spacing_score))
    return ranked,pool


def signed_inputs():
    parent = magic.OUT
    code = ORIGINAL_CODE_BINDING()
    for f in ('tools/amos_prompt_matrix.py','tools/magic_baseline_prompt_matrix.py',
              'tools/magic_bg_budget25.py'):
        code[f] = n.sha(ROOT/f)
    verification = n.read(parent/'vista3d_edge_retry_result.json')
    total_sha = n.sha(parent/'merged_metrics.json')
    if verification['merged_sha256']!=total_sha or verification['total']!=1295:
        raise ValueError('archived MAGIC results changed')
    return dict(code=code, parent_protocol=n.sha(parent/'protocol.json'), parent_merged=total_sha)


def audit_signed(p):
    if p['inputs']!=signed_inputs() or p['methods']!={k:list(v) for k,v in ARMS.items()}:
        raise ValueError('signed inputs or comparison conditions changed')
    if len(p['cases'])!=25 or len(p['tasks'])!=25 or sorted({t['case_id'] for t in p['tasks']})!=p['cases']:
        raise ValueError('25 distinct case-class keys required')


def stage_points(stage):
    _,(y0,_),(x0,_) = stage['bounds_dhw']
    return [[r['z'], [[x+x0,y+y0] for x,y in r['fg_sampled_xy']],
             [[y+y0,x+x0] for y,x in r['bg_original_yx']]]
            for r in stage['point_provenance']]


def freeze_prompt(t, prompts, stages, cert):
    bounds = stages[0]['bounds_dhw']
    frames = stage_points(stages[0]); expected_fg = {z:fg for z,fg,_ in frames if fg}
    if any({z:fg for z,fg,_ in stage_points(stage) if fg}!=expected_fg for stage in stages):
        raise ValueError('archived foreground coordinates varied across stages')
    bg_union = prompts.background[0] | prompts.background[1]
    pool_audit, plan, counts = [], [], Counter()
    for z,fg_xy,bg_yx in frames:
        anchors = [tuple(x) for x in bg_yx]
        if any(not bg_union[z,y,x] for y,x in anchors):
            raise ValueError('archived background point outside original support')
        if anchors:
            estimate,pool = rank_frame(prompts.target[z],anchors,fg_xy,prompts.planes,bounds[1:],prompts.spacing[1:])
            if not estimate:
                raise ValueError(f'no valid background points on a frozen frame {z}')
            counts[str(len(estimate))]+=1
            pool_audit.append([z,pool])
        else:
            estimate=[]
        plan.append([z,fg_xy,[list(x) for x in estimate]])
    if not counts['4']:
        raise ValueError(f"BG4 could not send four points anywhere in {t['case_id']}")
    return dict(frames=plan, counts={str(i):counts[str(i)] for i in range(1,5)},
                initial_bounds=bounds, pool_sha256=digest(pool_audit), archived_record_sha256=cert)


def prepare(cases):
    file=OUT/'protocol.json'
    if file.exists() or len(cases)!=25 or len(set(cases))!=25:
        raise ValueError('prepare requires 25 explicit cases and a new protocol')
    frozen=signed_inputs(); tasks=[]
    with magic.runner():
        parent=m.protocol(); magic.verify_inputs(parent)
        loader=magic.CaseLoader()
        for t in (t for t in parent['tasks'] if t['case_id'] in cases):
            prompts=loader(t)
            if m.prompt_identity(prompts,loader.files)!=t['prompt_identity']:
                raise ValueError('archived case input differs from loaded input')
            pairs={}; hashes={}
            for method in ARMS:
                old=m.read_record(method,t,0,frozen['parent_protocol'])
                cc=n.read(m.certificate_file(method,t['case_id'],0))
                if cc['keys']!=[list(n.key(t))] or cc['uuid']!=old['gpu_uuid'] or cc['protocol_sha256']!=frozen['parent_protocol']:
                    raise ValueError('archived record has no matching certificate')
                pairs[method]=next(x['audit']['stages'] for x in old['rows'] if x['arm']=='fg2_bg2')
                hashes[method]=n.sha(m.task_file(method,t,0))
            if (stage_points(pairs['sam2'][0])!=stage_points(pairs['medsam2'][0])
                    or pairs['sam2'][0]['bounds_dhw']!=pairs['medsam2'][0]['bounds_dhw']):
                raise ValueError('SAM2 and MedSAM2 archives do not share initial prompts')
            budget=freeze_prompt(t,prompts,pairs['sam2'],hashes)
            fg=lambda stages: {z:p for z,p,_ in stage_points(stages[0]) if p}
            ref=fg(pairs['sam2'])
            if any({z:p for z,p,_ in stage_points(st) if p}!=ref for st in pairs['medsam2']):
                raise ValueError('MedSAM2 archived FG changed between stages')
            tasks.append(dict(t,bg_budget=budget))
            print('MAGIC_BG_PREFLIGHT',t['case_id'],budget['counts'],flush=True)
    data=dict(inputs=frozen,cases=sorted(cases),tasks=tasks,methods={k:list(v) for k,v in ARMS.items()})
    audit_signed(data)
    n.write(file,data)
    print('MAGIC_BG_PREPARED',len(tasks),100,flush=True)


def expected_points(plan, cap):
    return {z:(fg,bg[:cap]) for z,fg,bg in plan}


def audit_stage(stage, plan, cap, spacing):
    want=expected_points(plan,cap)
    got={z:(fg,bg) for z,fg,bg in stage_points(stage)}
    if got!=want or stage['background_points_encoded']!=sum(len(bg) for _,bg in want.values()):
        raise ValueError('native encoder submission differs from signed FG/BG coordinates')
    for fg,bg in got.values():
        for i,p in enumerate(bg):
            if not all(separated(p,q,spacing) for q in (*bg[i+1:],*((y,x) for x,y in fg))):
                raise ValueError('actual points violate 3 mm spacing')


class BudgetPredictor(OLD_PREDICTOR):
    def evaluate(self, task, prompts):
        import torch

        foreground=prompts.foreground[0] | prompts.foreground[1]
        budget=task['bg_budget']
        initial=tuple(tuple(pair) for pair in budget['initial_bounds'])
        for arm,cap in zip(ARMS[self.method],(2,4),strict=True):
            bg=np.asarray([(z,y,x) for z,_,q in budget['frames'] for y,x in q[:cap]],dtype=int)
            if prompts.target[tuple(bg.T)].any():
                raise ValueError('frozen candidate is not a true background voxel')
            args=SimpleNamespace(max_points=2,max_background_points=0,point_spacing_mm=MIN_MM,
                                 dense_prior='none',sigma_mm=2.,amplitude=4.)
            stages=[]; bounds=initial
            with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
                for _ in range(8):
                    depth,(y0,y1),(x0,x1)=bounds
                    roi,st=self.adapter.predict_roi(self.model,prompts.ct[:,y0:y1,x0:x1],
                        foreground[:,y0:y1,x0:x1],bg-np.array([0,y0,x0]),prompts.spacing,depth,args)
                    stage=dict(st,bounds_dhw=bounds)
                    audit_stage(stage,budget['frames'],cap,prompts.spacing[1:])
                    stages.append(stage)
                    grown=self.adapter.expand_bounds_3d(roi,foreground,bounds,initial)
                    if grown==bounds:
                        break
                    bounds=grown
                else:
                    raise RuntimeError('AutoExpand did not converge')
            pred=np.zeros(prompts.ct.shape,dtype=bool)
            pred[tuple(slice(a,b) for a,b in bounds)]=roi
            yield arm,pred,dict(stages=stages, pool_sha256=budget['pool_sha256'], bg_cap=cap,
                   native_resampling=False, originals=budget['archived_record_sha256'],
                   gt_used_only_for_background_simulation=True,assets=self.assets)


def summarize(p):
    audit_signed(p); sha=n.sha(OUT/'protocol.json')
    stats={method:{a:[] for a in arms} for method,arms in ARMS.items()}
    counts=Counter(); cases_four=Counter()
    for t in p['tasks']:
        subset=t['bg_budget']
        for method,arms in ARMS.items():
            cert=n.read(m.certificate_file(method,t['case_id'],0))
            record=m.read_record(method,t,0,sha)
            if (cert['keys']!=[list(n.key(t))] or cert['protocol_sha256']!=sha or cert['uuid']!=record['gpu_uuid']
                    or cert['group']!=0 or cert['method']!=method):
                raise ValueError('missing or mismatched result certificate')
            checked={}
            for row in record['rows']:
                arm=row['arm']; cap=int(arm[-1]); d=row['audit']
                if d['pool_sha256']!=subset['pool_sha256'] or d['bg_cap']!=cap:
                    raise ValueError('candidate pool or requested cap differed')
                for stage in d['stages']:
                    audit_stage(stage,subset['frames'],cap,t['prompt_identity']['spacing'][1:])
                checked[arm]={z:len(bg) for z,_,bg in stage_points(d['stages'][-1]) if bg}
                stats[method][arm].append(row['metrics'])
            a,b=(checked[arm] for arm in arms)
            if set(a)!=set(b) or not any(k==4 and a[z]==2 for z,k in b.items()):
                raise ValueError('BG4 did not send more points on this certified case')
            cases_four[method]+=1
            counts[method,'bg_frames']+=len(a)
            counts[method,'extra_frames']+=sum(b[z]>v for z,v in a.items())
            counts[method,'bg4_four_frames']+=sum(v==4 for v in b.values())
            counts[method,'bg2_points']+=sum(a.values())
            counts[method,'bg4_points']+=sum(b.values())
    data={}
    for method,arms in stats.items():
        if cases_four[method]!=25 or any(len(q)!=25 for q in arms.values()):
            raise ValueError('comparison is incomplete')
        a,b=(arms[key] for key in ARMS[method])
        data[method]=dict(averages={arm:{k:math.fsum(row[k] for row in v)/25 for k in n.METRICS}
                                    for arm,v in arms.items()},
             delta_dice=math.fsum(y['dice']-x['dice'] for x,y in zip(a,b,strict=True))/25,
             exposure={key:val for (mm,key),val in counts.items() if mm==method})
    return dict(protocol_sha256=sha,cases=p['cases'],rows=100,methods=data)


def merge(p, existing_only=False):
    path=OUT/'merged_metrics.json'
    result=summarize(p)
    if path.exists() or existing_only:
        if n.read(path)!=result:
            raise ValueError('saved summary differs from certified results')
    else:
        n.write(path,result)
    print('MAGIC_BG_VERIFY_PASS' if existing_only else 'MAGIC_BG_COMPLETE',100,flush=True)


def idle_inventory():
    available=ORIGINAL_GPU_INVENTORY()
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                 '--format=csv,noheader,nounits'],text=True)
    busy={line.split(',')[0].strip() for line in raw.splitlines() if ',' in line}
    return {i:(uuid,free) for i,(uuid,free) in available.items() if uuid not in busy}


@contextmanager
def runner(dispatching=False):
    values=dict(OUT=OUT,ARMS=ARMS,PARALLEL_ARMS=set(),__file__=str(ENTRY),
                __doc__=__doc__,CaseLoader=magic.CaseLoader,verify_inputs=audit_signed,merge=merge)
    if dispatching:
        values['gpu_inventory']=idle_inventory
    old={k:getattr(m,k) for k in values}
    vars(m).update(values)
    native_predict.NativePredictor=BudgetPredictor
    try:
        yield
    finally:
        vars(m).update(old)
        native_predict.NativePredictor=OLD_PREDICTOR


def main():
    action=sys.argv[1:2]
    if action==['prepare']:
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument('cases',nargs=25)
        prepare(parser.parse_args(sys.argv[2:]).cases)
    else:
        with runner(dispatching=action==['run']):
            if action==['verify']:
                merge(m.protocol(),existing_only=True)
            else:
                m.main()


if __name__=='__main__':
    main()
