"""Fixed-logit calibration and memory-commit diagnostics; not GT-oracle tuning."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import amos_module_sensitivity as amos
from tools import magic_module_sensitivity as magic

OUT = 'output/v102_inference_calibration_amos_magic_e490_20261023'
FIXED = (.35, .45, .50, .55, .60, .65, .70)


def policy_names():
    return ('baseline',) + tuple('fixed_%.2f' % t for t in FIXED) + tuple(
        '%s_r%d' % (kind, radius) for kind in ('logit', 'probability') for radius in range(3))


def select_tasks(rows, split, amos_cases=8, magic_per_class=2):
    b = amos.base_module()
    selected = []
    for source in ('amos', 'magic'):
        available = [r for r in rows if r['source_name'] == source]
        ds = split['datasets'][source]
        val = {r['case_id'] for r in ds['val']}
        train = {r['case_id'] for r in ds['train']}
        if {r['case_id'] for r in available} - val or val & train:
            raise ValueError('invalid held-out manifest or train overlap')
        if source == 'amos':
            cases = set(sorted({r['case_id'] for r in available})[:amos_cases])
            selected.extend(r for r in available if r['case_id'] in cases)
        else:
            for class_id in sorted({int(r['global_class_id']) for r in available}):
                selected.extend(sorted((r for r in available if int(r['global_class_id']) == class_id),
                                       key=b.key)[:magic_per_class])
    if len({b.key(r) for r in selected}) != len(selected):
        raise ValueError('duplicate selected tasks')
    return sorted(selected, key=b.key)


def candidate_predictions(logits, foreground, background, component_fg, component_bg):
    import numpy as np
    import torch
    from scipy import ndimage
    from infer.prompt_roi_postprocess import (dynamic_mask_threshold_from_prompt_logits,
        dynamic_probability_threshold_from_prompt_logits, prompt_constrained_components)
    probabilities = torch.sigmoid(logits.detach().float()).squeeze(1).cpu().numpy()
    masks = {}
    for radius in range(3):
        fg = foreground.copy(); bg = background.copy()
        if radius:
            structure = ndimage.generate_binary_structure(3, 1)
            fg = ndimage.binary_dilation(fg, structure=structure, iterations=radius)
            bg = ndimage.binary_dilation(bg, structure=structure, iterations=radius)
        masks[radius] = (fg, np.logical_and(bg, ~fg))
    baseline = dynamic_mask_threshold_from_prompt_logits(logits, *masks[2])
    definitions = [('baseline', baseline)]
    definitions += [('fixed_%.2f' % t, dict(threshold=t, source='fixed_global')) for t in FIXED]
    for kind, function in (('logit', dynamic_mask_threshold_from_prompt_logits),
                           ('probability', dynamic_probability_threshold_from_prompt_logits)):
        for radius in range(3):
            definitions.append(('%s_r%d' % (kind, radius), function(logits, *masks[radius])))
    for name, audit in definitions:
        prediction = prompt_constrained_components(probabilities >= audit['threshold'], component_fg, component_bg)
        yield name, prediction, dict(audit, policy=name, fixed_final_roi=True)


def scan_metrics(logits, fg, bg, component_fg, component_bg, score):
    result = {}
    for name, prediction, audit in candidate_predictions(logits, fg, bg, component_fg, component_bg):
        result[name] = dict(audit, metrics=score(prediction, audit))
    return result


def check_policy_coverage(by_policy, expected):
    b = amos.base_module()
    if set(by_policy) != set(policy_names()):
        raise ValueError('missing or unknown calibration policy')
    for rows in by_policy.values():
        b.validate_rows(rows, expected)


def signature(out):
    b = amos.base_module()
    files=[Path(__file__),ROOT/'tools/amos_module_sensitivity.py',
           ROOT/'tools/magic_module_sensitivity.py',ROOT/'tools/ct13_full_validation_eval.py',ROOT/'utils/metrics.py']
    files+=sorted(ROOT.glob('*.py'))
    for folder in ('model','infer','v10','v10/model','v10/infer'):
        files+=sorted((ROOT/folder).glob('*.py'))
    result=dict(code={str(p.relative_to(ROOT)):amos.sha(p) for p in files},
        weights={'v10_2':dict(path=b.WEIGHTS['v10_2'],sha256=amos.sha(ROOT/b.WEIGHTS['v10_2']))},
        split=amos.sha(ROOT/b.SPLIT),manifest=amos.sha(out/'manifest.json'),refs=amos.sha(out/'refs.json'))
    refs = b.read(out/'refs.json')['per_case_class']
    result['prompts'] = {r['source_name']+'/'+r['prompt_cache_name']:
        amos.sha(ROOT/b.CACHE/'metadata/shared_adaptive_prompt_v1'/r['source_name']/r['prompt_cache_name']) for r in refs}
    result['code']['tools/magic_module_sensitivity.py'] = amos.sha(ROOT/'tools/magic_module_sensitivity.py')
    result['protocol'] = ('E490 live; deterministic AMOS8CT/MAGIC2perclass; memory groups1/4; '
        'original AutoExpand then same-final-logit fourteen-policy scan; component radius2 constant; '
        'fullmask10; original exact FG/BG; no logits/mask persistence; exploratory not unbiased test')
    return result


def verify(out):
    if signature(out) != amos.base_module().read(out/'protocol.json'):
        raise ValueError('changed code/weights/prompts/split; independent recovery required')


def prepare(out):
    b = amos.base_module()
    if out.exists(): raise FileExistsError(out)
    rows = select_tasks(b.read(ROOT/amos.HISTORY/'manifest.json')['tasks'], b.read(ROOT/b.SPLIT))
    refs = {}
    for source, count in (('amos',12),('magic',7)):
        for p in sorted((ROOT/amos.HISTORY/'prepared').glob(source+'_s*of%d/refs.json' % count)):
            for r in b.read(p)['per_case_class']:
                if b.key(r) in refs: raise ValueError('duplicate historical prompt reference')
                refs[b.key(r)] = r
    selected_refs = [dict(refs[b.key(r)], source_name=r['source_name']) for r in rows]
    # Validate caches before creating output. Never generate replacement prompts.
    for r in selected_refs:
        amos.sha(ROOT/b.CACHE/'metadata/shared_adaptive_prompt_v1'/r['source_name']/r['prompt_cache_name'])
    b.write(out/'manifest.json',b.manifest_payload(rows))
    b.write(out/'refs.json',dict(per_case_class=selected_refs))
    jobs=[]
    for group in (1,4):
        for source in ('amos','magic'):
            source_rows=[r for r in rows if r['source_name']==source]
            parts=[source_rows[:1]]+[b.case_shard(source_rows[1:],4,i) for i in range(4)]
            for i,part in enumerate(parts):
                if not part: continue
                name='%s__memory%d__%s' % (source,group,'smoke' if i==0 else 's%02d'%i)
                prepared=out/'prepared'/name
                b.write(prepared/'manifest.json',b.manifest_payload(part))
                b.write(prepared/'refs.json',dict(per_case_class=[refs[b.key(r)] for r in part]))
                jobs.append(dict(name=name,model='v10_2',variant='baseline',source=source,
                    memory_group=group,prepared=str(prepared),task_count=len(part),
                    frame_batch=16,attempt=0,smoke=i==0))
    b.write(out/'jobs.json',jobs)
    b.write(out/'protocol.json',signature(out))
    print('PREPARED',len(jobs),'jobs',len(rows),'tasks/group; 14 fixed-final-ROI policies',flush=True)


def run(job):
    b=amos.base_module(); out=Path(job['run']).parents[1]; verify(out)
    os.environ['CUDA_VISIBLE_DEVICES']=str(job['gpu'])
    sys.path.insert(0,str(ROOT/'v10'))
    sys.argv=[__file__,'--gpu',str(job['gpu'])]
    import numpy as np
    from scipy import ndimage
    import v10.infer.infer_amos_sam2_promptgen_v10 as scorer
    import v10.infer.infer_eight_source_adaptive_v10_2 as ev
    import v10_sam2_memory_roi as memory
    if job['memory_group'] not in (1,4): raise ValueError('invalid memory group')
    native_memory=memory.memory_sliding_window_decode
    memory_calls=[]
    def memory_decode(adapter,video,bundle,tokens,anchor,args,device,training):
        if training or int(args.eval_memory_group_size)!=job['memory_group']:
            raise ValueError('unexpected actual memory commit configuration')
        memory_calls.append(dict(group=int(args.eval_memory_group_size),frames=int(video.shape[0])))
        return native_memory(adapter,video,bundle,tokens,anchor,args,device,training)
    memory.memory_sliding_window_decode=memory_decode
    native_decode=ev._dynamic_adaptive_decode
    native_score=scorer.restore_and_score_full_volume
    tasks=sorted(b.read(Path(job['prepared'])/'manifest.json')['tasks'],key=b.key)
    collected={name:[] for name in policy_names()}; context={}; calls=0
    def decode(adapter,prompt,video,bundle,class_id,args,device,initial_bounds):
        nonlocal calls
        if context: raise ValueError('previous final logits not scored')
        args.eval_memory_group_size=int(job['memory_group'])
        result=native_decode(adapter,prompt,video,bundle,class_id,args,device,initial_bounds)
        if result[5] is not None: raise ValueError('unexpected patch geometry')
        start,end=result[2]; region=dict(bundle); region['frame_coords']=bundle['frame_coords'][start:end]
        shape=(int(result[0].shape[0]),int(result[0].shape[2]),int(result[0].shape[3]))
        fg=scorer._foreground_prompt_seed_mask(region,shape)
        bg=scorer._background_prompt_seed_mask(region,shape)
        structure=ndimage.generate_binary_structure(3,1)
        component_fg=ndimage.binary_dilation(fg,structure=structure,iterations=2)
        component_bg=ndimage.binary_dilation(bg,structure=structure,iterations=2)&~component_fg
        context.update(fg=fg,bg=bg,component_fg=component_fg,component_bg=component_bg,
                       class_id=int(class_id),baseline_prediction=result[1],threshold=result[4]['threshold'])
        calls+=1
        return result
    def score(logits,**kwargs):
        if not context: raise ValueError('missing final ROI context')
        task=tasks[len(collected['baseline'])]
        if int(task['global_class_id'])!=context['class_id']: raise ValueError('task/class mismatch')
        baseline=native_score(logits,**kwargs)
        def measure(prediction,audit):
            if audit['policy']=='baseline':
                if (abs(audit['threshold']-context['threshold'])>1e-7 or
                        not np.array_equal(prediction,context['baseline_prediction'])):
                    raise ValueError('scan baseline differs from native final prediction')
                values=baseline[0]
            else:
                arguments=dict(kwargs,prediction=prediction,threshold=audit['threshold'])
                values=native_score(logits,**arguments)[0]
            return {m:float(values[m]) for m in b.METRICS}
        scan=scan_metrics(logits,context['fg'],context['bg'],context['component_fg'],context['component_bg'],measure)
        for name,value in scan.items():
            collected[name].append(dict(task,**value['metrics'],threshold=value['threshold'],
                threshold_audit={k:v for k,v in value.items() if k!='metrics'}))
        context.clear()
        b.write(Path(job['run'])/'calibration_live.json',collected)
        print('CALIBRATION_SCAN',b.key(task),'policies14 memory',job['memory_group'],flush=True)
        return baseline
    ev._dynamic_adaptive_decode=decode
    scorer.restore_and_score_full_volume=score
    amos.verify=verify
    amos.run(job)
    check_policy_coverage(collected,tasks)
    if calls!=len(tasks) or context or not memory_calls: raise ValueError('prediction/score/memory count mismatch')
    b.write(Path(job['run'])/'calibration_metrics.json',collected)
    b.write(Path(job['run'])/'calibration_audit.json',dict(memory_group=job['memory_group'],
        native_final_decode_calls=calls,threshold_policy_count=14,extra_decode_calls_for_thresholds=0,
        actual_memory_decode_stages=memory_calls,fixed_final_roi=True,
        all_ten_metrics_finite=True,no_array_persistence=True))


def merge(out):
    b=amos.base_module(); verify(out); jobs=b.read(out/'jobs.json')
    expected=b.read(out/'manifest.json')['tasks']; result={}
    for group in (1,4):
        policies={name:[] for name in policy_names()}
        for j in jobs:
            if j['memory_group']!=group: continue
            run_dir=out/'jobs'/j['name']
            audit=b.read(run_dir/'calibration_audit.json')
            if audit['memory_group']!=group or audit['extra_decode_calls_for_thresholds']!=0:
                raise ValueError('invalid memory/calibration audit')
            if not b.read(run_dir/'intervention_audit.json')['guards_passed']: raise ValueError('invalid parameter audit')
            for name,rows in b.read(run_dir/'calibration_metrics.json').items(): policies[name]+=rows
        check_policy_coverage(policies,expected)
        result[str(group)]={}
        for source in ('amos','magic'):
            baseline=[r for r in policies['baseline'] if r['source_name']==source]
            result[str(group)][source]={name:magic.summarize([r for r in rows if r['source_name']==source],baseline)
                                       for name,rows in policies.items()}
    b.write(out/'merged_metrics.json',result)
    print('MERGED diagnostic policies; no GT-oracle candidate selection',flush=True)


def launch(out,gpus):
    import fcntl
    b=amos.base_module(); verify(out)
    lock=(out/'diagnostic_schedule.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try: args=p.read_bytes().decode(errors='replace').split('\0')
        except OSError: continue
        if 'worker' in args and any(str(out/'job_specs') in a for a in args): raise ValueError('existing worker alive')
    jobs=b.read(out/'jobs.json'); native_read=b.read; b.__file__=__file__; b.merge=lambda _out:None
    try:
        for phase in ('smoke','memory1','memory4'):
            selected=[j for j in jobs if (j['smoke'] if phase=='smoke' else
                      not j['smoke'] and j['memory_group']==int(phase[-1]))]
            for j in selected:
                run_dir=out/'jobs'/j['name']
                if (run_dir/'complete_metrics.json').exists() and not (run_dir/'calibration_audit.json').exists():
                    raise ValueError('incomplete audit retained; investigate before resuming')
            b.read=lambda p:native_read(p) if Path(p)!=out/'jobs.json' else selected
            b.write(out/'queue_phase.json',dict(phase=phase)); b.launch(out,gpus)
            for j in selected:
                audit=native_read(out/'jobs'/j['name']/'calibration_audit.json')
                if audit['native_final_decode_calls']!=j['task_count']: raise ValueError('smoke/worker coverage')
            print('PHASE_PASSED',phase,flush=True)
    finally: b.read=native_read
    merge(out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('prepare','worker','launch','status','merge'))
    p.add_argument('--out',default=OUT); p.add_argument('--job'); p.add_argument('--gpus',default='0,2,3,4,5,6')
    args=p.parse_args(); os.chdir(ROOT); out=(ROOT/args.out).resolve(); b=amos.base_module()
    if args.mode=='prepare': prepare(out)
    elif args.mode=='worker': run(b.read(args.job))
    elif args.mode=='merge': merge(out)
    elif args.mode=='launch':
        gpus=[int(x) for x in args.gpus.split(',')]
        if len(set(gpus))!=len(gpus) or not set(gpus)<={0,2,3,4,5,6}: raise ValueError('invalid GPU selection')
        launch(out,gpus)
    else:
        jobs=b.read(out/'jobs.json')
        print(json.dumps(dict(jobs=len(jobs),audited=sum((out/'jobs'/j['name']/'calibration_audit.json').exists() for j in jobs),
            phase=b.read(out/'queue_phase.json') if (out/'queue_phase.json').exists() else None,
            scheduler=b.read(out/'status.json') if (out/'status.json').exists() else None),indent=2))


if __name__=='__main__': main()
