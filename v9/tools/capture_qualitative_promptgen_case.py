"""Observe genuine PromptGen final full-volume masks for five frozen tasks.

One CLI invocation, one source, one native model process. No CT/GT array export.
Manifest schema five_source_promptgen_capture_v1: export_authorized=true; entries
exactly one each amos, magic, kits23, msd_task07, msd_task10. Entry fields: task,
prompt_reference (native reference including prompt_cache_name), reference (the
original PromptGen row), runner, shape_dhw, input_sha256 (project-relative files),
job (native frozen fields including gpu/gpu_uuid), checkpoint {path,epoch,sha256},
and paths for AMOS. Every
source/checkpoint/prompt/shared CT/protocol file must be inventoried upstream.
CLI requires the independently frozen manifest SHA. Upstream must establish
canonical geometry/axes; this observer checks shape and full scorer parity only.

AMOS retains signed owner initialize/evaluate/finish, including its original
GPU2/4/5/6 restriction; no guard is bypassed. MAGIC uses original floor60 runner
and a fresh signed one-task scope, not ct13.run_model directly. S1 uses original
ct13 runner, MSD10 S7 original encoderfix runner. No alternate model is inferred.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).absolute().parents[1]
METRICS = ('dice','mean_slice_dice','iou','precision','recall','specificity',
           'pred_gt_ratio','mean_slice_nsd_1mm','mean_slice_nsd_2mm','mean_slice_nsd_3mm')
RUNNERS = dict(amos='amos_s21',magic='magic_s16',kits23='s1_e490',
               msd_task07='s1_e490',msd_task10='msd10_s7')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')


def key(task):
    # Signed native scorer calls the GLOBAL ID class_id; it is never local_class_id.
    global_id=task.get('global_class_id',task.get('class_id'))
    if global_id is None:raise ValueError('global class identifier required')
    if 'global_class_id' in task and 'class_id' in task and int(task['global_class_id'])!=int(task['class_id']):
        raise ValueError('conflicting global class identifiers')
    return (task['source_name'], task['case_id'], int(global_id))


def require_key(row, task):
    if key(row) != key(task): raise ValueError('exact source/case/global-class mismatch')


def authorized_entry(manifest, source):
    if (manifest.get('schema')!='five_source_promptgen_capture_v1'
            or manifest.get('export_authorized') is not True):
        raise ValueError('frozen five-task export authorization required')
    entries=manifest['entries']
    if len(entries)!=5 or {e['task']['source_name'] for e in entries}!=set(RUNNERS):
        raise ValueError('exact five distinct sources required')
    if len({key(e['task']) for e in entries})!=5: raise ValueError('duplicate authorization')
    for entry in entries:
        if entry['runner']!=RUNNERS[entry['task']['source_name']]:
            raise ValueError('source-specific original runner required')
    return next(entry for entry in entries if entry['task']['source_name']==source)


def _array(value):
    import numpy as np
    if hasattr(value,'detach'): value=value.detach().cpu().numpy()
    return np.asarray(value)


@contextmanager
def observe_final_mask(namespace, shape):
    """Copy only scorer result[1]; never recompute prediction from GT or logits."""
    import numpy as np
    name='restore_and_score_full_volume'
    mapping=isinstance(namespace,dict)
    original=namespace[name] if mapping else getattr(namespace,name)
    captured={}
    def scored(*args,**kwargs):
        if captured: raise ValueError('multiple scorer calls for one frozen task')
        result=original(*args,**kwargs)
        prediction=_array(result[1])
        if (prediction.shape!=tuple(shape) or not np.isfinite(prediction).all()
                or not np.isin(prediction,[0,1]).all()):
            raise ValueError('native final full-volume binary mask/shape mismatch')
        indices=np.asarray(kwargs.get('frame_indices'))
        if (indices.ndim!=1 or indices.dtype.kind not in 'iu' or not len(indices)
                or (indices<0).any() or (indices>=shape[0]).any()
                or (np.diff(indices)!=1).any()):
            raise ValueError('missing or incomplete final contiguous ROI frame mapping')
        captured.update(prediction_dhw=prediction.astype(np.uint8,copy=True),
                        metrics=dict(result[0]),frame_indices=indices.astype(np.int64,copy=True))
        return result
    if mapping: namespace[name]=scored
    else:setattr(namespace,name,scored)
    try:
        yield captured
        if not captured:raise ValueError('native scorer not observed')
    finally:
        if mapping:namespace[name]=original
        else:setattr(namespace,name,original)


def compare_metrics(row, reference, tolerance=1e-6):
    differences={}
    for name in METRICS:
        if name not in row or name not in reference:raise ValueError('missing metric '+name)
        a,b=float(row[name]),float(reference[name])
        difference=abs(a-b)
        if not math.isfinite(a) or not math.isfinite(b) or difference>tolerance:
            raise ValueError('reference metric mismatch '+name)
        differences[name]=difference
    for name in ('mask_threshold_used','roi_initial_frames','roi_reserved_frames',
                 'roi_final_frames','roi_auto_expand_iterations'):
        if name in reference and row.get(name)!=reference[name]:
            raise ValueError('reference protocol mismatch '+name)
    return differences


def verify_inputs(inventory):
    if not inventory:raise ValueError('frozen input/source/checkpoint inventory required')
    result={}
    # Original protocols name sibling data/code via ../. These are frozen,
    # individually hashed read-only inputs; relative names are never write paths.
    for relative,expected in inventory.items():
        path=ROOT/relative
        if Path(relative).is_absolute():
            raise ValueError('inventory must use project-relative paths')
        actual=sha(path)
        if actual!=expected:raise ValueError('frozen input SHA mismatch '+relative)
        result[relative]=actual
    return result


def check_checkpoint(entry, actual_path, actual_epoch):
    expected=entry['checkpoint'];relative=expected['path']
    if entry['input_sha256'].get(relative)!=expected['sha256']:
        raise ValueError('checkpoint not bound to signed input inventory')
    actual=Path(actual_path)
    if not actual.is_absolute():actual=ROOT/actual
    # absolute() avoids filesystem mutation and is sufficient for frozen native paths.
    if actual.absolute()!=(ROOT/relative).absolute() or int(actual_epoch)!=int(expected['epoch']):
        raise ValueError('actual original checkpoint path/epoch mismatch')


def resource_snapshot(text, gpu, expected_uuid, runner):
    rows={int(parts[0]):parts for parts in
          ([part.strip() for part in line.split(',')] for line in text.splitlines() if line.strip())}
    if runner=='amos_s21' and gpu not in (2,4,5,6):
        raise ValueError('signed AMOS runtime requires original allowed physical GPU2/4/5/6')
    if gpu not in rows or rows[gpu][1]!=expected_uuid or int(rows[gpu][2])<30720:
        raise ValueError('physical GPU UUID/free memory requirement failed')
    return dict(gpu=gpu,gpu_uuid=expected_uuid,free_mib=int(rows[gpu][2]),minimum_free_mib=30720)


def bind_resources(entry):
    job=entry['job']
    query=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.free',
                          '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
    proof=resource_snapshot(query.stdout,job['gpu'],job['gpu_uuid'],entry['runner'])
    os.environ['CUDA_VISIBLE_DEVICES']=str(job['gpu'])
    import torch
    actual='GPU-'+str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
    if actual!=job['gpu_uuid'] or torch.ones(1,device='cuda:0').item()!=1:
        raise ValueError('native CUDA UUID/allocation probe mismatch')
    torch.set_num_threads(2)
    return dict(proof,cuda_probe=True,pid=os.getpid())


def save_mask(out, prediction, metadata):
    import numpy as np
    out=Path(out)
    if (out/'capture.json').exists() or (out/'prediction.npz').exists():
        raise FileExistsError('capture exists; no overwrite/resume')
    with (out/'prediction.npz').open('xb') as stream:
        np.savez_compressed(stream,prediction_dhw=prediction)
    write_new(out/'capture.json',dict(metadata,prediction_sha256=sha(out/'prediction.npz')))


def prepare(entry, out):
    from tools import ct13_full_validation_eval as base
    scope=out/'native'; prepared=scope/'prepared/capture';run=scope/'jobs/capture'
    manifest=base.manifest_payload([entry['task']])
    ref=entry['prompt_reference'];require_key(ref,entry['task'])
    write_new(prepared/'manifest.json',manifest)
    write_new(prepared/'refs.json',dict(per_case_class=[ref]))
    job=dict(entry['job'],run=str(run),prepared=str(prepared),task_count=1,
             source=entry['task']['source_name'],model='v10_2')
    if entry['runner']=='magic_s16':
        job.update(name='capture',variant='baseline')
        from tools import v102_floor60_confirmation as floor
        write_new(scope/'manifest.json',manifest)
        write_new(scope/'refs.json',dict(per_case_class=[ref]))
        write_new(scope/'jobs.json',[job])
        write_new(scope/'protocol.json',floor.signature(scope))
    return job


def run_amos(entry, job):
    from tools import amos_prompt_reuse_owner_runtime as runtime
    paths=dict(entry['paths'],manifest=str(Path(job['prepared'])/'manifest.json'),
               refs=str(Path(job['prepared'])/'refs.json'),run_dir=job['run'])
    job=dict(job,case_id=entry['task']['case_id'],pending=[entry['task']],control=False)
    session=runtime.initialize(job,paths)
    with observe_final_mask(session.namespace,entry['shape_dhw']) as observed:
        record=runtime.evaluate(session,entry['task'],'fg_scribble_bg_scribble')
    final=session.finish()
    if (final.get('epoch'),final.get('prompt_parameters_exact'),final.get('decoder_parameters_exact'),
            final.get('successful_native_completion'))!=(490,418,212,True):
        raise ValueError('original normal E490 finish proof missing')
    check_checkpoint(entry,session.namespace['custom'].weights,final['epoch'])
    require_key(record['row'],entry['task'])
    return observed,record['row'],dict(normal_finish=final,record=record,input_audit=session.input_audit)


def run_original(entry, job):
    from tools import ct13_full_validation_eval as base
    sys.path.insert(0,str(ROOT/'v10'))
    # Preserve CT13's import-order contract before importing the scorer.
    import ct13_training_entry
    import v10.infer.infer_amos_sam2_promptgen_v10 as scorer
    runner=entry['runner']
    if runner=='s1_e490':invoke=base.run_model
    elif runner=='msd10_s7':
        from tools import v102_source_best_full as source_best
        invoke=source_best.run_model
    elif runner=='magic_s16':
        from tools import v102_floor60_confirmation as floor
        invoke=floor.run
    else:raise ValueError('unsupported original runner')
    with observe_final_mask(scorer,entry['shape_dhw']) as observed:
        invoke(job)
    rows=read(Path(job['run'])/'complete_metrics.json')['per_case_class']
    if len(rows)!=1:raise ValueError('one native completed row required')
    require_key(rows[0],entry['task'])
    audit=read(Path(job['run'])/'exact_prompt_audit.json')
    if ([tuple(k) for k in audit['loaded_task_keys']] != [key(entry['task'])]
            or audit['loaded_task_count']!=1 or audit['dynamic_threshold_summary']['count']!=1):
        raise ValueError('native completed prompt/task audit mismatch')
    proofs=dict(native_returned_normally=True,exact_prompt_audit=audit)
    completed=read(Path(job['run'])/'validation_only_metrics.json')
    check_checkpoint(entry,completed['checkpoint'],completed['checkpoint_epoch'])
    proofs['checkpoint']=dict(path=completed['checkpoint'],epoch=completed['checkpoint_epoch'])
    if runner=='msd10_s7':
        encoder=read(Path(job['run'])/'encoder_restore_audit.json')
        if encoder['epoch']!=15 or encoder['all_tensors_exact'] is not True or encoder['restored_tensor_count']<=0:
            raise ValueError('S7 encoderfix restoration proof missing')
        proofs['encoder_restore']=encoder
    if runner=='magic_s16':
        intervention=read(Path(job['run'])/'intervention_audit.json')
        confirmation=read(Path(job['run'])/'confirmation_audit.json')
        if (intervention['guards_passed'] is not True or intervention['checkpoint_epoch']!=490
                or intervention['prompt_parameters_exact']!=418 or intervention['decoder_parameters_exact']!=212
                or confirmation['dynamic_floor']!=.60 or confirmation['memory_group']!=1):
            raise ValueError('S16 native restoration/completion proof missing')
        proofs.update(intervention=intervention,confirmation=confirmation)
    return observed,rows[0],proofs


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--source',required=True,choices=tuple(RUNNERS))
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args(argv)
    if sha(a.manifest)!=a.manifest_sha256:raise ValueError('frozen manifest SHA mismatch')
    entry=authorized_entry(read(a.manifest),a.source)
    if a.output.exists():raise FileExistsError('output exists; no overwrite or partial resume')
    require_key(entry['reference'],entry['task'])
    before=verify_inputs(entry['input_sha256'])
    os.environ['V9_SHARED_CACHE_WRITE_POLICY']='read_only'
    os.environ['CUDA_VISIBLE_DEVICES']=str(entry['job']['gpu'])
    sys.path.insert(0,str(ROOT));os.chdir(ROOT)
    resources=bind_resources(entry)
    a.output.mkdir(parents=True,exist_ok=False)
    write_new(a.output/'frozen_entry.json',entry)
    job=prepare(entry,a.output)
    observed,row,proofs=(run_amos if entry['runner']=='amos_s21' else run_original)(entry,job)
    differences=compare_metrics(row,entry['reference'])
    compare_metrics(observed['metrics'],{name:entry['reference'][name] for name in METRICS})
    if verify_inputs(entry['input_sha256'])!=before:raise ValueError('inputs changed during inference')
    metadata=dict(schema='five_source_promptgen_mask_v1',method='PromptGen',task=entry['task'],
        runner=entry['runner'],shape_dhw=entry['shape_dhw'],axes='ZXY',metric_scope='full_volume',
        metrics={name:row[name] for name in METRICS},reference_differences=differences,
        frame_indices=observed['frame_indices'].tolist(),proofs=proofs,
        input_sha256=before,resources=resources,manifest_sha256=a.manifest_sha256,capture_source_sha256=sha(__file__))
    save_mask(a.output,observed['prediction_dhw'],metadata)
    print('CAPTURE_COMPLETE',a.source,flush=True)


if __name__=='__main__':main()
