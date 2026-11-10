"""One-shot controls -> independent gate -> full -> independently audited merge."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).absolute().parents[1]
sys.path.insert(0,str(ROOT))
from tools.five_source_prompt_type_runtime import METRICS,VARIANTS,key


def summarize(records,tasks):
    expected={key(t) for t in tasks};index={}
    for r in records:
        ident=(key(r['row']),r['variant'])
        if ident in index:raise ValueError('duplicate raw record')
        index[ident]=r
    if set(index)!={(k,v) for k in expected for v in VARIANTS}:raise ValueError('raw paired key coverage mismatch')
    for k in expected:
        if len({index[k,v]['actual_frame_batch'] for v in VARIANTS})!=1:
            raise ValueError('paired prompt conditions must share actual frame batch')
    result={}
    for source in sorted({k[0] for k in expected}):
        keys=sorted(k for k in expected if k[0]==source);result[source]={}
        for v in VARIANTS:
            selected=[index[k,v] for k in keys];groups=defaultdict(list)
            for r in selected:groups[key(r['row'])[2]].append(r)
            for r in selected:
                if not all(math.isfinite(float(r['row'][m])) for m in METRICS):raise ValueError('nonfinite raw metric')
            result[source][v]=dict(tasks=len(keys),
                mean={m:math.fsum(float(r['row'][m]) for r in selected)/len(keys) for m in METRICS},
                class_macro={m:math.fsum(math.fsum(float(r['row'][m]) for r in group)/len(group) for group in groups.values())/len(groups) for m in METRICS},
                paired_delta_vs_baseline={m:math.fsum(float(index[k,v]['row'][m])-float(index[k,'baseline']['row'][m]) for k in keys)/len(keys) for m in METRICS},
                fallback_tasks=sum(str(r['row'].get('dynamic_threshold_source','')).startswith('fallback') for r in selected))
    return result


def read(path):return json.loads(path.read_text())


def numeric_error(actual,expected):
    actual,expected=float(actual),float(expected)
    if not math.isfinite(actual) or not math.isfinite(expected):raise ValueError('nonfinite merged numeric evidence')
    return abs(actual-expected)


def write_new(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def audit_used_batch(out,e,protocol,digest,source,batch):
    gate=e.require_batch_gate(out,source,batch,digest)
    tasks=[t for t in protocol['inventory']['tasks'] if key(t)[:2]==(source,gate['case_id'])]
    if not tasks or gate['records']!=4*len(tasks):raise ValueError('batch control scope mismatch')
    errors=[]
    for task in tasks:
        actual={}
        for variant in ('native_control',*VARIANTS):
            record=read(e.record_path(out/('control_batch_'+str(batch)),task,variant,True))
            if record.get('control') is not True or record['protocol_sha256']!=digest:raise ValueError('batch control binding mismatch')
            e.runtime.validate_record(record,task,variant,batch,control=True)
            if not record['normal_finish']['successful_native_completion']:raise ValueError('batch control normal finish missing')
            actual[variant]=record
        n,b=actual['native_control'],actual['baseline']
        error=max(abs(float(n['row'][m])-float(b['row'][m])) for m in METRICS)
        if error>1e-6 or any(n['roi'][k]!=b['roi'][k] for k in ('initial','reserved','final','history')) or n['stages']!=b['stages']:
            raise ValueError('used source/batch independent parity failure')
        errors.append(error)
    return dict(source=source,actual_batch=batch,control_records=gate['records'],max_ten_metric_error=max(errors))


def controls(out):
    from tools import five_source_prompt_type_eval_v2 as e
    protocol,digest=e.protocol_at(out)
    state=read(out/'queue_controls.json')
    if state['phase']!='complete' or state['failed'] or state['running'] or state['pending']:
        raise ValueError('controls did not naturally drain')
    counts={};comparisons=[]
    for source in e.SOURCES:
        gate=read(out/'controls'/source/'control_pass.json');batch=gate['actual_frame_batch']
        if not gate['passed'] or gate['protocol_sha256']!=digest:raise ValueError('actual source gate mismatch')
        e.require_batch_gate(out,source,batch,digest)
        tasks=[t for t in protocol['inventory']['tasks'] if key(t)[:2]==(source,gate['case_id'])]
        count=0
        for task in tasks:
            records={}
            for variant in ('native_control',*VARIANTS):
                record=read(e.record_path(out/('control_batch_'+str(batch)),task,variant,True))
                if record['protocol_sha256']!=digest or record.get('control') is not True:raise ValueError('control record binding mismatch')
                e.runtime.validate_record(record,task,variant,batch,control=True)
                if not record['normal_finish']['successful_native_completion']:raise ValueError('control normal finish missing')
                records[variant]=record;count+=1
            n,b=records['native_control'],records['baseline']
            delta=max(abs(float(n['row'][m])-float(b['row'][m])) for m in METRICS)
            geometry_equal=all(n['roi'][k]==b['roi'][k] for k in ('initial','reserved','final','history'))
            if delta>1e-6 or not geometry_equal or n['stages']!=b['stages']:raise ValueError('independent native parity failed')
            comparisons.append(dict(task_key=list(key(task)),actual_batch=batch,max_ten_metric_error=delta))
        if not tasks or count!=gate['records']:raise ValueError('complete control case record count mismatch')
        counts[source]=count
    write_new(out/'independent_control_gate.json',dict(passed=True,protocol_sha256=digest,control_records=counts,comparisons=comparisons))


def final(out):
    from tools import five_source_prompt_type_eval_v2 as e
    protocol,digest=e.protocol_at(out);tasks=protocol['inventory']['tasks']
    state=read(out/'queue_full.json')
    if state['phase']!='complete' or state['running'] or state['pending'] or state['failed']:raise ValueError('full queue not drained')
    if e.status(out)['completed']!=519:raise ValueError('519 actual validated records required')
    records=[read(e.record_path(out,t,v)) for t in tasks for v in VARIANTS]
    batch_controls=[audit_used_batch(out,e,protocol,digest,s,b) for s,b in sorted({(key(r['row'])[0],r['actual_frame_batch']) for r in records})]
    summary=summarize(records,tasks);merged=read(out/'merged_metrics.json')
    if (merged.get('schema'),merged.get('protocol_sha256'),merged.get('task_count'),merged.get('controls_excluded'))!=('five_source_prompt_type_merged_v1',digest,519,True):
        raise ValueError('merged scope/protocol binding mismatch')
    saved_rows=merged['per_case_class'];saved_index={(key(r),r['variant']):r for r in saved_rows}
    if len(saved_index)!=519 or len(saved_rows)!=519 or set(saved_index)!={(key(r['row']),r['variant']) for r in records}:
        raise ValueError('merged row identity mismatch')
    error=0.
    for r in records:
        saved=saved_index[key(r['row']),r['variant']]
        if saved['actual_frame_batch']!=r['actual_frame_batch']:raise ValueError('merged actual batch mismatch')
        for m in METRICS:error=max(error,numeric_error(saved[m],r['row'][m]))
    for source,variants in summary.items():
        for variant,value in variants.items():
            saved=merged['per_source'][source][variant]
            if saved['tasks']!=value['tasks'] or saved['fallback_tasks']!=value['fallback_tasks']:raise ValueError('count mismatch')
            for field in ('mean','class_macro','paired_delta_vs_baseline'):
                for metric,expected in value[field].items():error=max(error,numeric_error(saved[field][metric],expected))
    if error>1e-12:raise ValueError('independent raw fsum aggregation mismatch')
    write_new(out/'independent_final_audit.json',dict(passed=True,protocol_sha256=digest,
        unique_production_rows=519,controls_excluded=True,paired_same_actual_batch=True,
        maximum_recompute_error=error,per_source=summary,
        full_input_rehash_files=len(protocol['inventory']['input_sha256']),
        independently_verified_used_batch_controls=batch_controls,
        merged_sha256=hashlib.sha256((out/'merged_metrics.json').read_bytes()).hexdigest()))
    print('FIVE_SOURCE_PROMPT_TYPES_FINAL_PASS',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();out=a.out.absolute()
    if (out/'queue_controls.json').exists():raise FileExistsError('one-shot pipeline already dispatched')
    queue=str(ROOT/'tools/five_source_prompt_type_queue_v2.py');entry=str(ROOT/'tools/five_source_prompt_type_eval_v2.py')
    subprocess.run([sys.executable,'-B',queue,'controls','--out',str(out),'--gpus','0,3'],cwd=ROOT,check=True)
    controls(out)
    subprocess.run([sys.executable,'-B',queue,'full','--out',str(out),'--gpus','0,3'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-B',entry,'merge','--out',str(out)],cwd=ROOT,check=True)
    final(out)


if __name__=='__main__':main()
