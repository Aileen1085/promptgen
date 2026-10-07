"""Native SegVol official FPS4 on CT13 full held-out validation, isolated scheduler."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import native_segvol_pilot as pilot,native_sammed3d_ct13 as framework
native=pilot.native
DEFAULT='output/native_segvol_ct13_full6094_fps4_20261007'
PROTOCOL=dict(model='SegVol native',arm='official_fps4',image_size=[32,256,256],threshold=.5,
    tensor_order='Z_Y_X',point_convention='official',point_cap_per_plane_role=4,point_spacing_mm=3.,
    promptgen=False,semantic=False,masks=None,points_once=True,zoom_in_reflection=False,
    roi='prompt_8mm_20pct_connected_expand50_cap100',metrics=list(native.METRICS))
FRAMEWORK_FILE=Path(framework.__file__)

def validate_audit(row):
    a=row['prompt_audit']
    if not a['background_cache_hit'] or a['fg_in_gt_fraction']!=1. or a['bg_in_gt_count']!=0:
        raise ValueError('SegVol prompt cache/alignment audit')
    if not row['stages']:raise ValueError('missing SegVol prediction stages')
    for s in row['stages']:
        if (s['tensor_axis_order']!='Z_Y_X' or s['point_convention']!='official'
            or not math.isfinite(s['roundtrip_voxels']) or s['roundtrip_voxels']>1e-8
            or not 0<s['foreground_points']<=8 or not 0<s['background_points']<=8
            or s['gt_used_for_roi'] or s['decoder_calls']!=1):
            raise ValueError('SegVol point/axis/ROI audit')

def signature(job):
    code=pilot.signature()+native.sha(Path(__file__))+native.sha(FRAMEWORK_FILE)
    code+=native.sha(ROOT/'tools/native_sam2_physical_eval.py')+native.sha(ROOT/'configs/ct13_v10_2_split_20260928.json')
    prompts=[native.sha(native.CACHE/'metadata/shared_adaptive_prompt_v1'/t['source_name']/t['prompt_cache_name']) for t in job['tasks']]
    return hashlib.sha256(json.dumps(dict(protocol=PROTOCOL,tasks=job['tasks'],code=code,prompts=prompts,
        weight=pilot.WEIGHT_SHA),sort_keys=True).encode()).hexdigest()

def worker(job_path,uuid):
    import torch
    torch.set_num_threads(2)
    if str(torch.cuda.get_device_properties(0).uuid).replace('GPU-','')!=uuid.replace('GPU-',''):
        raise ValueError('GPU UUID/CUDA mapping mismatch')
    job=native.read(job_path);out=Path(job['output']);sig=signature(job)
    protocol=dict(signature=sig,weight_sha256=pilot.WEIGHT_SHA,protocol=PROTOCOL)
    if (out/'protocol.json').exists():
        if native.read(out/'protocol.json')!=protocol:raise ValueError('changed-code/protocol resume refused')
    else:native.write(out/'protocol.json',protocol)
    rows=native.read(out/'all_metrics_live.json')['per_case_class'] if (out/'all_metrics_live.json').exists() else []
    done={native.key(r) for r in rows};native.validate_rows(rows,[t for t in job['tasks'] if native.key(t) in done])
    for r in rows:validate_audit(r)
    if len(rows)<len(job['tasks']):
        model=pilot.load_model();base=native.base_module();loader=native.CaseLoader(base);scorer=base.metric_module()
        split=native.read(ROOT/'configs/ct13_v10_2_split_20260928.json');case=None;ctnorm=None
        for t in job['tasks']:
            if native.key(t) in done:continue
            start=time.time();ct,gt,fg,bg,spacing,audit=loader(t,split,'scribble',str(pilot.shared.BG))
            if not audit['background_cache_hit'] or audit['fg_in_gt_fraction']!=1. or audit['bg_in_gt_count']!=0:
                raise ValueError('prompt audit failed before prediction')
            current=(t['source_name'],t['case_id'])
            if case!=current:ctnorm=pilot.normalize(ct);case=current
            pred,stages=pilot.predict(model,ctnorm,fg,bg,spacing,audit['canonical_prompt_slices'],'official_fps4')
            metrics=scorer.binary_prompt_metrics_from_masks(torch.from_numpy(pred),torch.from_numpy(gt),tuple(spacing))
            row=dict(t,**{m:float(metrics[m]) for m in native.METRICS},seconds=time.time()-start,
                prompt_audit=audit,stages=stages,gpu_uuid=uuid)
            validate_audit(row);rows.append(row);done.add(native.key(t))
            native.validate_rows(rows,[t for t in job['tasks'] if native.key(t) in done])
            native.write(out/'all_metrics_live.json',dict(per_case_class=rows,signature=sig))
            print('DONE',job['name'],len(rows),'/',len(job['tasks']),t['case_id'],t['class_name'],
                'Dice',row['dice'],'seconds',row['seconds'],flush=True)
    native.validate_rows(rows,job['tasks'])
    native.write(out/'complete_metrics.json',dict(per_case_class=rows,signature=sig,
        summary={m:float(np.mean([r[m] for r in rows])) for m in native.METRICS}))

def main():
    # Only this process's framework globals are bound; its source and other running models are untouched.
    framework.DEFAULT=DEFAULT;framework.PROTOCOL=PROTOCOL;framework.pilot=pilot
    framework.signature=signature;framework.worker=worker;framework.validate_audit=validate_audit
    framework.__file__=__file__;framework.__doc__=__doc__
    framework.main()

if __name__=='__main__':main()
