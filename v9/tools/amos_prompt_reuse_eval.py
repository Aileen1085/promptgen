"""Exact AMOS prompt reuse with immutable old-class evidence and natural drain.

The signed native controller is compiled into fresh globals. Its only extra
AST edit changes evaluation order, keeping canonical protocol/output variants.
Only the owned scheduler is stopped; captured workers always finish naturally.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import types

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
PARENT=importlib.import_module('tools.amos_prompt_ablation_native_manifest_eval')
PARENT_SHA256='25f234af32e55780bafd4a53954dcdaaeadc80ef6fbf8f7bc6032441299b1ce8'
RUNTIME_SHA256='bd3f4c1274088edc8fea1cfc0c908d9a4e1b436471a85d2e2400be9f56d9725a'
PARENT_PROTOCOL_SHA256='a652c2c2639265bb4e74a560bc917660a8b545785824bbece0182bffe9d435a5'
PARENT_GATE_SHA256='1c9f2945f7699b4daf3b6c61d04fa6d45e3e507d6e228901297fd46b35787fb3'
OUT='output/amos_v102_e490_floor070_prompt_ablations_20261030_reusefast'
TMUX='amos_v102_floor070_prompt_ablations_reusefast'
PIPELINE='output/amos_prompt_ablations_reusefast_pipeline_20261030.log'
HANDOFF='output/amos_prompt_reuse_handoff_20261030.json'
STOPPED='output/amos_prompt_reuse_stopped_20261030.json'
DRAINED='output/amos_prompt_reuse_drained_20261030.json'
EVALUATION_ORDER=('baseline','fg_scribble_bg_scribble','fg_box_bg_point','fg_box_bg_scribble','sagittal_only','coronal_only')
ENTRIES=frozenset(('amos_prompt_ablation_eval.py','amos_prompt_ablation_restore_eval.py',
                   'amos_prompt_ablation_native_manifest_eval.py','amos_prompt_reuse_eval.py'))
_ENGINE=None


def sha(path):return PARENT._sha(path)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def parent_engine():return PARENT.engine()
def parent_out():return ROOT/PARENT.OUT


def source_guard():
    for name,expected in (('tools/amos_prompt_ablation_native_manifest_eval.py',PARENT_SHA256),
                          ('tools/amos_prompt_reuse_runtime.py',RUNTIME_SHA256)):
        if sha(ROOT/name)!=expected:raise ValueError('reuse signed source SHA changed: '+name)
    PARENT._source_guard(ROOT)


def reverse_worker_transform(tree):
    tree=copy.deepcopy(tree)
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name=='worker':
            for stmt in ast.walk(node):
                if isinstance(stmt,ast.comprehension) and isinstance(stmt.iter,ast.Name) and stmt.iter.id=='EVALUATION_ORDER':stmt.iter.id='VARIANTS'
    return tree


def transform_controller(source,filename,expected_sha256):
    tree,proof=PARENT.transform_controller(source,filename,expected_sha256)
    before=copy.deepcopy(tree)
    workers=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker']
    if len(workers)!=1:raise ValueError('unique worker iteration anchor required')
    matches=[n for n in ast.walk(workers[0]) if isinstance(n,ast.Assign) and ast.unparse(n)==
             'rows = [runtime().evaluate(session, task, v) for v in VARIANTS]']
    if len(matches)!=1:raise ValueError('exact worker iteration anchor changed')
    matches[0].value.generators[0].iter.id='EVALUATION_ORDER'
    if ast.dump(reverse_worker_transform(tree),include_attributes=False)!=ast.dump(before,include_attributes=False):
        raise ValueError('unexpected extra worker AST edit')
    proof=dict(proof,worker_iteration_replacements=1,worker_order=list(EVALUATION_ORDER),
               reuse_ast_sha256=PARENT._ast_sha(tree))
    return ast.fix_missing_locations(tree),proof


def inventory(paths,root=None):return parent_engine()['file_inventory'](paths,root or ROOT)
def verify_inventory(files,root=None):return parent_engine()['verify_inventory'](files,root or ROOT)


def union_records(old,new):
    if set(old)&set(new):raise ValueError('duplicate reused/new committed class')
    return {**old,**new}


def advance_history(canonical,previous,evidence):
    """Never infer OOM from a live worker, warning, or intentional stop."""
    if previous is None:return copy.deepcopy(canonical)
    if (not isinstance(evidence,dict) or type(evidence.get('returncode')) is not int
            or 'log_text' not in evidence or 'observed_at' not in evidence):
        raise ValueError('latest parent attempt lacks positive completion/resource evidence')
    code=evidence['returncode']
    if code==0:
        value=copy.deepcopy(previous);value['attempt']+=1
    else:
        value=parent_engine()['retry'](previous,code,evidence['log_text'],evidence['observed_at'])
        if value is None:raise ValueError('non-resource parent worker error blocks reuse')
    for name in ('run','spec','log','gpu','gpu_uuid','actual_frame_batch','control_gate_sha256'):
        value.pop(name,None)
    # Original full task list, variant order and new prepared path stay canonical.
    result=copy.deepcopy(canonical)
    for name in ('attempt','frame_batch','minimum_free_mib','oom_retries','not_before','avoid_gpus'):result[name]=value[name]
    return result


def _stat(text):
    end=text.rfind(')');prefix=text[:text.find(' ')];fields=text[end+2:].split()
    if end<0 or len(fields)<50:raise ValueError('torn proc stat')
    return dict(pid=int(prefix),state=fields[0],ppid=int(fields[1]),starttime=int(fields[19]),raw_exit_status=int(fields[49]))


def read_process(pid,proc_root=None):
    """Read twice; a disappearing, reused or torn process is never signalable."""
    folder=Path(proc_root or '/proc')/str(pid)
    try:
        first=_stat((folder/'stat').read_text());status=(folder/'status').read_text()
        uid_lines=[line for line in status.splitlines() if line.startswith('Uid:')]
        if len(uid_lines)!=1:raise ValueError('missing proc UID')
        uid=int(uid_lines[0].split()[1]);argv=(folder/'cmdline').read_bytes().split(b'\0')
        argv=[a.decode(errors='strict') for a in argv if a]
        cwd=None if first['state']=='Z' else os.readlink(folder/'cwd')
        wchan=(folder/'wchan').read_text().strip()
        second=_stat((folder/'stat').read_text())
        again=[a.decode(errors='strict') for a in (folder/'cmdline').read_bytes().split(b'\0') if a]
    except FileNotFoundError:return None
    if (first['pid']!=int(pid) or any(first[k]!=second[k] for k in ('pid','starttime','ppid')) or argv!=again
            or first['state']=='Z' and second['state']!='Z'):
        raise ValueError('torn proc identity')
    return dict(second,uid=uid,argv=argv,cwd=cwd,wchan=wchan)


def same_identity(expected,actual,zombie=False):
    fields=('pid','uid','ppid','starttime') if zombie else ('pid','uid','ppid','starttime','cwd','argv')
    return actual is not None and all(expected.get(k)==actual.get(k) for k in fields)


def signal_exact(identity,sig):
    if not same_identity(identity,read_process(identity['pid'])):raise ValueError('process identity changed; signal refused')
    os.kill(identity['pid'],sig)


def observed_exit(identity,current):
    if (not same_identity(identity,current,zombie=True) or current.get('state')!='Z'
            or type(current.get('raw_exit_status')) is not int):
        raise ValueError('unknown/reused/torn worker exit cannot certify completion')
    return os.waitstatus_to_exitcode(current['raw_exit_status'])


def related(proc_root=None):
    proc=Path(proc_root or '/proc');rows=[]
    if not proc.exists():return rows
    for folder in proc.iterdir():
        if not folder.name.isdigit() or int(folder.name)==os.getpid():continue
        try:args=[a.decode(errors='replace') for a in (folder/'cmdline').read_bytes().split(b'\0') if a]
        except FileNotFoundError:continue
        if any(Path(a).name in ENTRIES for a in args) and any(a in ('prepare','run','worker','wait-run','freeze-parent') for a in args):
            rows.append(dict(pid=int(folder.name),args=args))
    return rows


def require_quiescence():
    checks=[]
    for i in range(2):
        if i:time.sleep(1)
        checks.append(related())
        if checks[-1]:raise ValueError('related process quiescence failed: '+repr(checks[-1]))
    return checks


def verify_parent(*,rehash=False):
    source_guard();p=parent_engine();out=parent_out()
    if sha(out/'protocol.json')!=PARENT_PROTOCOL_SHA256:raise ValueError('parent protocol SHA changed')
    if sha(out/'control_verification.json')!=PARENT_GATE_SHA256:raise ValueError('parent independent gate SHA changed')
    p['verify'](out,rehash=rehash);p['check_control_gate'](out)
    status=read(out/'status.json')
    if status.get('phase')!='full' or status.get('failed'):raise ValueError('parent must be healthy full phase')
    return p['class_records'](out)


def _scheduler(identity):
    args=identity.get('argv',[]) if identity else []
    flags=args[1:-2] if len(args)>=3 else []
    script=Path(args[-2]) if len(args)>=3 else Path('.')
    if not script.is_absolute():script=ROOT/script
    if (not identity or identity['uid']!=os.getuid() or identity['cwd']!=str(ROOT.resolve())
            or len(args)<3 or args[-1]!='run' or len(flags)!=len(set(flags)) or set(flags)-{'-B','-u'}
            or script.resolve()!=(ROOT/'tools/amos_prompt_ablation_native_manifest_eval.py').resolve()
            or identity['state'] in ('Z','X')):raise ValueError('not exact owned parent scheduler')
    return identity


def _bind_worker(identity,scheduler):
    p=parent_engine();args=identity.get('argv',[])
    if (identity['uid']!=scheduler['uid'] or identity['ppid']!=scheduler['pid'] or identity['cwd']!=str(ROOT.resolve())
            or len(args)!=5 or args[2:4]!=['worker','--job'] or identity['state']=='Z'
            or Path(args[1]).resolve()!=(ROOT/'tools/amos_prompt_ablation_native_manifest_eval.py').resolve()):
        raise ValueError('unbound or unknown child worker')
    path=Path(args[4]);spec=read(path);out=parent_out()
    canonical=next((j for j in read(out/'jobs.json')['cases'] if j['name']==spec['name']),None)
    if canonical is None:raise ValueError('unknown parent job')
    p['validate_attempt'](spec,canonical,out)
    if spec['spec']!=str(path) or spec['control']:raise ValueError('worker signed spec mismatch')
    return dict(identity=identity,spec=spec,spec_sha256=sha(path))


def children(scheduler):
    result=[]
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit():continue
        try:stat=_stat((folder/'stat').read_text())
        except FileNotFoundError:continue
        if stat['ppid']!=scheduler['pid']:continue
        value=read_process(int(folder.name))
        if value is not None and value['ppid']==scheduler['pid']:result.append(value)
    return result


def validate_status_workers(queue,workers):
    expected={item['identity']['pid']:dict(pid=item['identity']['pid'],
             **{f:item['spec'][f] for f in ('name','spec','log')}) for item in workers}
    actual=list(queue.get('running',{}).values())
    if (queue.get('phase')!='full' or queue.get('failed') or len(expected)!=len(workers)
            or len(actual)!=len(expected) or any(expected.get(row.get('pid'))!=row for row in actual)):
        raise ValueError('atomic scheduler status differs from signed captured workers')


def freeze_parent(scheduler_pid=None):
    """Exclusive handoff; freeze at scheduler sleep, never signal workers."""
    if any((ROOT/name).exists() for name in (HANDOFF,STOPPED,DRAINED)):
        raise FileExistsError('handoff evidence exists; repeated freeze refused')
    verify_parent();out=parent_out()
    if scheduler_pid is None:
        candidates=[]
        for row in related():
            try:candidates.append(_scheduler(read_process(row['pid'])))
            except ValueError:continue
        if len(candidates)!=1:raise ValueError('unique owned scheduler required')
        scheduler=candidates[0]
    else:scheduler=_scheduler(read_process(scheduler_pid))
    # A bounded wait for the sleep boundary is separate from unbounded model drain.
    deadline=time.monotonic()+120
    while scheduler['wchan'] not in ('hrtimer_nanosleep','do_nanosleep'):
        if time.monotonic()>deadline:raise ValueError('scheduler sleep boundary unavailable; no signal sent')
        time.sleep(.1);current=_scheduler(read_process(scheduler['pid']))
        if not same_identity(scheduler,current):raise ValueError('scheduler identity changed')
        scheduler=current
    queue_before=read(out/'status.json')
    before=[_bind_worker(value,scheduler) for value in children(scheduler)]
    validate_status_workers(queue_before,before)
    frozen=False;published=False
    try:
        signal_exact(scheduler,signal.SIGSTOP);frozen=True
        current=read_process(scheduler['pid'])
        deadline=time.monotonic()+5
        while current and current['state'] not in ('T','t') and time.monotonic()<deadline:
            time.sleep(.05);current=read_process(scheduler['pid'])
        if not same_identity(scheduler,current) or current['state'] not in ('T','t'):raise ValueError('parent did not freeze with stable identity')
        after=[_bind_worker(value,scheduler) for value in children(scheduler)]
        if {r['identity']['pid'] for r in before}!={r['identity']['pid'] for r in after}:
            raise ValueError('worker set changed across scheduler freeze')
        for a,b in zip(sorted(before,key=lambda r:r['identity']['pid']),sorted(after,key=lambda r:r['identity']['pid'])):
            if not same_identity(a['identity'],b['identity']) or a['spec_sha256']!=b['spec_sha256']:raise ValueError('worker identity changed across freeze')
        verify_parent();queue=read(out/'status.json')
        validate_status_workers(queue,after)
        if queue!=queue_before:raise ValueError('atomic scheduler status changed across freeze')
        evidence=dict(kind='amos_exact_parent_natural_drain_v1',scheduler=current,workers=after,
                      captured_before=before,status_before=queue_before,status=queue,created_at=time.time(),
                      parent_protocol_sha256=sha(out/'protocol.json'),parent_gate_sha256=sha(out/'control_verification.json'),
                      parent_controller_sha256=sha(ROOT/'tools/amos_prompt_ablation_native_manifest_eval.py'))
        parent_engine()['atomic_new'](ROOT/HANDOFF,evidence)
        parent_engine()['atomic_new'](ROOT/STOPPED,dict(handoff_sha256=sha(ROOT/HANDOFF),scheduler=current,workers=after))
        published=True;return evidence
    finally:
        if frozen and not published:signal_exact(scheduler,signal.SIGCONT)


def verify_handoff():
    source_guard();h=read(ROOT/HANDOFF);stopped=read(ROOT/STOPPED)
    if (h.get('kind')!='amos_exact_parent_natural_drain_v1' or stopped!=dict(handoff_sha256=sha(ROOT/HANDOFF),scheduler=h['scheduler'],workers=h['workers'])
            or h['parent_protocol_sha256']!=PARENT_PROTOCOL_SHA256 or h['parent_gate_sha256']!=PARENT_GATE_SHA256
            or h['parent_controller_sha256']!=PARENT_SHA256):raise ValueError('handoff chain changed')
    _scheduler(h['scheduler'])
    validate_status_workers(h['status'],h['workers'])
    validate_status_workers(h['status_before'],h['captured_before'])
    if h['status']!=h['status_before']:raise ValueError('handoff status changed during freeze')
    for item in h['workers']:
        if sha(item['spec']['spec'])!=item['spec_sha256']:raise ValueError('captured worker spec changed')
        if _bind_worker(item['identity'],h['scheduler'])['spec']!=item['spec']:raise ValueError('captured worker binding changed')
    return h


def validate_finished_worker(item,current):
    """Actual zero exit and native final restoration are both required."""
    code=observed_exit(item['identity'],current);spec=item['spec'];log=Path(spec['log'])
    text=log.read_text(encoding='utf-8',errors='replace')
    result=dict(returncode=code,raw_exit_status=current['raw_exit_status'],identity=current,
                spec=spec['spec'],spec_sha256=sha(spec['spec']),log=spec['log'],log_sha256=sha(log),
                log_text=text,observed_at=time.time(),log_mtime_ns=log.stat().st_mtime_ns)
    if code==0:
        p=parent_engine();p['verify_worker_complete'](spec);found=p['class_records'](parent_out())
        if not {p['key'](t) for t in spec['pending']}<=set(found):raise ValueError('normal worker exited without entire signed pending coverage')
    elif parent_engine()['retry'](spec,code,text,result['observed_at']) is None:
        raise ValueError('non-resource actual worker exit blocks handoff: '+str(code))
    return result


def finish_parent():
    if (ROOT/DRAINED).exists():return verify_drained()
    h=verify_handoff();scheduler=h['scheduler'];completed={}
    while len(completed)<len(h['workers']):
        actual=read_process(scheduler['pid'])
        if not same_identity(scheduler,actual) or actual['state'] not in ('T','t'):
            raise ValueError('captured parent is no longer the same frozen scheduler')
        for item in h['workers']:
            pid=item['identity']['pid']
            if str(pid) in completed:continue
            current=read_process(pid)
            if not same_identity(item['identity'],current,zombie=current is not None and current['state']=='Z'):
                raise ValueError('captured worker disappeared/reused; no positive exit status')
            if current['state']=='Z':completed[str(pid)]=validate_finished_worker(item,current)
        if len(completed)<len(h['workers']):time.sleep(10)
    # Freeze remains in effect until every exit and final proof is captured.
    for item in h['workers']:
        if observed_exit(item['identity'],read_process(item['identity']['pid']))!=completed[str(item['identity']['pid'])]['returncode']:
            raise ValueError('worker exit changed before scheduler retirement')
    actual=read_process(scheduler['pid'])
    if not same_identity(scheduler,actual) or actual['state'] not in ('T','t'):
        raise ValueError('captured parent is no longer the same frozen scheduler')
    signal_exact(scheduler,signal.SIGTERM);signal_exact(scheduler,signal.SIGCONT)
    deadline=time.monotonic()+30
    while True:
        current=read_process(scheduler['pid'])
        if current is None or same_identity(scheduler,current,zombie=True) and current['state']=='Z':break
        if not same_identity(scheduler,current):raise ValueError('scheduler PID reused during retirement')
        if time.monotonic()>deadline:raise ValueError('owned scheduler retirement did not complete')
        time.sleep(.1)
    checks=require_quiescence()
    # Logs/status are hashed only after their writers have gone.
    files=inventory([ROOT/HANDOFF,ROOT/STOPPED,parent_out()/'status.json',ROOT/PARENT.PIPELINE])
    result=dict(kind='amos_natural_drain_complete_v1',handoff_sha256=sha(ROOT/HANDOFF),
                stopped_sha256=sha(ROOT/STOPPED),workers=completed,related_checks=checks,files=files,retired_scheduler=scheduler)
    parent_engine()['atomic_new'](ROOT/DRAINED,result);return result


def verify_drained():
    h=verify_handoff();d=read(ROOT/DRAINED)
    if (d.get('kind')!='amos_natural_drain_complete_v1' or d.get('handoff_sha256')!=sha(ROOT/HANDOFF)
            or d.get('stopped_sha256')!=sha(ROOT/STOPPED) or d.get('related_checks')!=[[],[]]
            or d.get('retired_scheduler')!=h['scheduler']
            or set(d.get('workers',{}))!={str(r['identity']['pid']) for r in h['workers']}):
        raise ValueError('drained handoff chain changed')
    verify_inventory(d['files'])
    for item in h['workers']:
        row=d['workers'][str(item['identity']['pid'])]
        spec=item['spec'];log=Path(spec['log'])
        if (observed_exit(item['identity'],row['identity'])!=row['returncode']
                or row.get('raw_exit_status')!=row['identity']['raw_exit_status']
                or row.get('spec')!=spec['spec'] or row.get('log')!=spec['log']
                or row['spec_sha256']!=sha(spec['spec']) or row['log_sha256']!=sha(log)
                or row.get('log_mtime_ns')!=log.stat().st_mtime_ns
                or row.get('log_text')!=log.read_text(encoding='utf-8',errors='replace')):
            raise ValueError('actual captured worker exit evidence changed')
        if row['returncode']==0:parent_engine()['verify_worker_complete'](spec)
        elif parent_engine()['retry'](spec,row['returncode'],row['log_text'],row['observed_at']) is None:
            raise ValueError('non-resource captured worker failure')
    return d


def _runtime():return importlib.import_module('tools.amos_prompt_reuse_runtime')


def parent_history(found=None):
    """Recover latest actual attempt policy; unknown last attempts fail closed."""
    p=parent_engine();out=parent_out();drain=verify_drained();found=verify_parent() if found is None else found
    captured={row['spec']:row for row in drain['workers'].values()};result={}
    canonical={j['name']:j for j in read(out/'jobs.json')['cases']}
    by_name={name:[] for name in canonical}
    for path in sorted((out/'job_specs').glob('*.json')):
        spec=read(path)
        if spec.get('control'):continue
        if spec.get('name') not in canonical:raise ValueError('unknown parent attempt history')
        p['validate_attempt'](spec,canonical[spec['name']],out)
        by_name[spec['name']].append(spec)
    for name,attempts in by_name.items():
        if not attempts:result[name]=dict(previous=None,evidence=None);continue
        numbers=[j['attempt'] for j in attempts]
        if len(numbers)!=len(set(numbers)) or set(numbers)!=set(range(max(numbers)+1)):
            raise ValueError('parent attempt history gaps or duplicate attempt')
        latest=max(attempts,key=lambda j:j['attempt']);evidence=captured.get(latest['spec'])
        success=out/'case_success'/(name+'.json')
        if evidence is None and success.exists():
            proof=p['verify_case_success'](out,canonical[name],found)
            if proof['final_spec']!=latest['spec']:raise ValueError('latest parent attempt lacks final success evidence')
            log=Path(latest['log'])
            evidence=dict(returncode=0,log_text=log.read_text(encoding='utf-8',errors='replace'),
                          observed_at=log.stat().st_mtime,case_success=str(success),case_success_sha256=sha(success))
        advance_history(canonical[name],latest,evidence)  # reject ambiguous history now
        result[name]=dict(previous=latest,evidence=evidence)
    return result


def initial_case_jobs(jobs):
    history=parent_history()
    if set(history)!={j['name'] for j in jobs}:raise ValueError('canonical100 job history changed')
    return [advance_history(j,**history[j['name']]) for j in jobs]


def prepare_guard():
    source_guard();verify_drained();require_quiescence();verify_parent(rehash=True)


def _record_index(found):
    return [dict(key=list(k),marker=v['marker'],marker_sha256=v['marker_sha256'],
                 spec=v['spec']['spec'],spec_sha256=sha(v['spec']['spec'])) for k,v in sorted(found.items())]


def _old_case_proof(canonical,found,drain):
    p=parent_engine();out=parent_out();chosen={k:v for k,v in found.items() if v['spec']['name']==canonical['name']}
    if set(chosen)!={p['key'](t) for t in canonical['pending']}:raise ValueError('reused case canonical coverage incomplete')
    marker=out/'case_success'/(canonical['name']+'.json')
    if marker.exists():
        p['verify_case_success'](out,canonical,found)
        return dict(kind='parent_case_success',path=str(marker),sha256=sha(marker))
    # The stopped scheduler could not publish case_success for its last workers.
    # Only a positively observed normal exit plus the untouched native finish
    # guard may replace that publication; all canonical classes must be present.
    matches=[row for row in drain['workers'].values() if read(row['spec'])['name']==canonical['name']]
    if len(matches)!=1 or matches[0]['returncode']!=0:raise ValueError('fully reused case missing normal completion evidence')
    row=matches[0];spec=read(row['spec']);p['verify_worker_complete'](spec)
    return dict(kind='captured_normal_worker_completion',spec=spec['spec'],spec_sha256=sha(spec['spec']),
                log=spec['log'],log_sha256=sha(spec['log']),captured_exit=row,
                native_worker_complete_sha256=sha(Path(spec['run'])/'native_worker_complete.json'),
                restoration_final_sha256=sha(Path(spec['run'])/'restoration_final.json'),
                markers={v['marker']:v['marker_sha256'] for v in chosen.values()},
                contributing_attempts=p['contributing_attempts'](chosen))


def build_reuse_audit():
    p=parent_engine();out=parent_out();drain=verify_drained();found=verify_parent(rehash=True);history=parent_history(found)
    paths=[ROOT/HANDOFF,ROOT/STOPPED,ROOT/DRAINED,ROOT/PARENT.PIPELINE]
    for name in ('protocol.json','protocol_signature.json','native_bootstrap_audit.json','jobs.json','manifest.json','refs.json',
                 'control_verification.json','status.json'):paths.append(out/name)
    protocol=read(out/'protocol.json')
    for field in ('sources','prepared'):paths.extend(ROOT/name for name in protocol[field])
    for item in found.values():
        paths.append(Path(item['marker']));paths.extend(Path(name) for name in read(item['marker'])['files'])
        paths.extend((Path(item['spec']['spec']),Path(item['spec']['log'])))
    # Freeze all old control evidence independently; it never substitutes the
    # new native + six condition controls or the new independent gate.
    controls=p['class_records'](out,True)
    for item in controls.values():
        paths.extend((Path(item['marker']),Path(item['spec']['spec']),Path(item['spec']['log']),Path(item['spec']['run'])/'native_control.json'))
        paths.extend(Path(name) for name in read(item['marker'])['files'])
        paths.extend(Path(item['spec']['run'])/n for n in ('native_worker_complete.json','restoration_final.json'))
    cases={}
    for canonical in read(out/'jobs.json')['cases']:
        selected={k:v for k,v in found.items() if v['spec']['name']==canonical['name']}
        if set(selected)=={p['key'](t) for t in canonical['pending']}:
            cases[canonical['name']]=_old_case_proof(canonical,found,drain)
    for path in (out/'case_success').glob('*.json'):paths.append(path)
    for path in (out/'job_specs').glob('*.json'):
        spec=read(path);paths.append(path)
        if Path(spec['log']).exists():paths.append(Path(spec['log']))
        for name in ('native_worker_complete.json','restoration_start.json','restoration_final.json'):
            if (Path(spec['run'])/name).exists():paths.append(Path(spec['run'])/name)
    return dict(kind='amos_complete_six_class_reuse_v1',parent_output=PARENT.OUT,
                parent_protocol_sha256=PARENT_PROTOCOL_SHA256,parent_gate_sha256=PARENT_GATE_SHA256,
                handoff_sha256=sha(ROOT/HANDOFF),drained_sha256=sha(ROOT/DRAINED),
                records=_record_index(found),histories=history,cases=cases,files=inventory(paths))


def verify_reuse(out):
    out=Path(out);audit=read(out/'reuse_audit.json');bootstrap=read(out/'native_bootstrap_audit.json')
    if (audit.get('kind')!='amos_complete_six_class_reuse_v1' or audit.get('parent_output')!=PARENT.OUT
            or audit.get('parent_protocol_sha256')!=PARENT_PROTOCOL_SHA256 or audit.get('parent_gate_sha256')!=PARENT_GATE_SHA256
            or bootstrap.get('reuse_audit_sha256')!=sha(out/'reuse_audit.json')
            or audit.get('handoff_sha256')!=sha(ROOT/HANDOFF) or audit.get('drained_sha256')!=sha(ROOT/DRAINED)):
        raise ValueError('immutable reuse audit chain changed')
    verify_inventory(audit['files']);verify_drained();return audit


def frozen_records(out):
    audit=verify_reuse(out);found=verify_parent()
    if _record_index(found)!=audit['records']:raise ValueError('frozen parent complete class set changed')
    return found


def verify_reused_case(out,canonical,found):
    audit=verify_reuse(out);old=frozen_records(out);p=parent_engine()
    selected={k:v for k,v in found.items() if v['spec']['name']==canonical['name']}
    expected={p['key'](t) for t in canonical['pending']}
    if set(selected)!=expected or not expected<=set(old):raise ValueError('case without new success marker is not fully reused')
    old_canonical=next(j for j in read(parent_out()/'jobs.json')['cases'] if j['name']==canonical['name'])
    if old_canonical['pending']!=canonical['pending']:raise ValueError('reused canonical task list changed')
    proof=_old_case_proof(old_canonical,old,verify_drained())
    if audit['cases'].get(canonical['name'])!=proof:raise ValueError('frozen reused normal case proof changed')
    return proof


def bootstrap(out,jobs,controls):
    audit=build_reuse_audit();out=Path(out);ns=engine()
    ns['atomic_new'](out/'reuse_audit.json',audit)
    native=PARENT.run_native_bootstrap(root=ns['ROOT'],out=out,jobs=jobs,controls=controls)
    native.update(reuse_audit_sha256=sha(out/'reuse_audit.json'),reuse_files=audit['files'],
                  reuse_controller_transform=ns['TRANSFORM_PROOF'])
    path=out/'native_bootstrap_audit.json';ns['atomic_new'](path,native);return path


def engine():
    global _ENGINE
    if _ENGINE is not None:return _ENGINE
    source_guard();p=parent_engine();source=ROOT/'tools/amos_prompt_ablation_eval.py'
    tree,proof=transform_controller(source.read_bytes(),str(source),PARENT.NATIVE_CONTROLLER_SHA256)
    private=types.ModuleType('_amos_prompt_reuse_private');ns=private.__dict__
    ns['__file__']=str(Path(__file__).resolve());exec(compile(tree,str(source),'exec'),ns)
    ns.update(OUT=OUT,TMUX=TMUX,PIPELINE=PIPELINE,EVALUATION_ORDER=EVALUATION_ORDER,related=related,runtime=_runtime,
              _native_manifest_payload=PARENT.native_manifest_payload,TRANSFORM_PROOF=proof)
    ns['SOURCE_SEEDS']=tuple(dict.fromkeys((*p['SOURCE_SEEDS'],'tools/amos_prompt_reuse_eval.py','tools/amos_prompt_reuse_runtime.py')))
    # Wrappers are compiled in the child globals too. No parent function is
    # installed as a child controller function, and the parent dict is untouched.
    ns.update(_frozen_records=frozen_records,_reuse_prepare_guard=prepare_guard,_reuse_bootstrap=bootstrap,
              _reuse_verify=verify_reuse,_reuse_case_success=verify_reused_case,_reuse_case_jobs=initial_case_jobs,
              _source_guard=source_guard,_finish_parent=finish_parent)
    exec(compile('''
native_class_records=class_records
native_verify=verify
native_prepare=prepare
native_pipeline=run_pipeline
native_verify_case_success=verify_case_success
native_case_jobs=case_jobs
native_schedule=schedule
def class_records(out,control=False):
    if control:return native_class_records(out,True)
    return _union_records(_frozen_records(out),native_class_records(out,False))
def verify(out,**kwargs):
    _source_guard();_reuse_verify(out)
    return native_verify(out,**kwargs)
def prepare(out,**kwargs):
    _reuse_prepare_guard()
    return native_prepare(out,**kwargs)
def case_jobs(tasks,strict=True):
    jobs=native_case_jobs(tasks,strict)
    return _reuse_case_jobs(jobs) if strict else jobs
def verify_case_success(out,canonical,found):
    if canonical['control'] or (Path(out)/'case_success'/(canonical['name']+'.json')).exists():
        return native_verify_case_success(out,canonical,found)
    return _reuse_case_success(out,canonical,found)
def _native_bootstrap_preflight(out,jobs,controls):
    return _reuse_bootstrap(out,jobs,controls)
def run_pipeline(out):
    _reuse_prepare_guard()
    return native_pipeline(out)
def schedule(out,control):
    if control:return native_schedule(out,True)
    verify(out,rehash=False);check_control_gate(out)
    committed=class_records(out)
    jobs=read(Path(out)/'jobs.json')['cases']
    pending=[job for job in jobs if any(key(task) not in committed for task in job['pending'])]
    if pending:return schedule_queue(pending,False,Backend(out,False))
    state=dict(time=time.time(),phase='full',pending=0,running={},failed=[],committed_classes=len(committed))
    write_status(Path(out)/'status.json',state)
    return state
''',str(Path(__file__).resolve()),'exec'),ns)
    ns['_union_records']=union_records
    _ENGINE=ns;return ns


def __getattr__(name):
    if name.startswith('__'):raise AttributeError(name)
    ns=engine()
    if name not in ns:raise AttributeError(name)
    return ns[name]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('freeze-parent','wait-run','prepare','run','worker','status','merge'))
    parser.add_argument('--scheduler-pid',type=int);parser.add_argument('--job')
    args=parser.parse_args()
    if args.mode=='freeze-parent':return freeze_parent(args.scheduler_pid)
    if args.mode=='wait-run':finish_parent();return engine()['run_pipeline'](ROOT/OUT)
    # Use the original child dispatch for every inherited mode.
    return engine()['main']()


if __name__=='__main__':main()
