"""Additive recovery of a captured AMOS worker's transient Linux proc snapshot.

Only incomplete process reads are retried. Stable identity changes and missing
exit evidence still fail closed; all native completion and control guards remain.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
CONTROLLER='tools/amos_prompt_reuse_eval.py'
CONTROLLER_SHA256='3ac648c6acd801be5e3ac8d9497621df730314d0ea89ca414e8a0a39cb7f5349'
RUNTIME='tools/amos_prompt_reuse_runtime.py'
RUNTIME_SHA256='bd3f4c1274088edc8fea1cfc0c908d9a4e1b436471a85d2e2400be9f56d9725a'
FAILURE_EVIDENCE='output/amos_prompt_reuse_proc_transition_failure_20261030.json'
FAILURE_EVIDENCE_SHA256='e849aaeb6e1cd72ffa2a5661f88266acb3f5386c8bddc880fd7df14f0733d4d59'
OUT='output/amos_v102_e490_floor070_prompt_ablations_20261030_reusefast_procfix'
TMUX='amos_v102_floor070_prompt_ablations_reusefast_procfix'
PIPELINE='output/amos_prompt_ablations_reusefast_procfix_pipeline_20261030.log'
DRAINED='output/amos_prompt_reuse_drained_procfix_20261030.json'
SNAPSHOT_ATTEMPTS=12
SNAPSHOT_DELAY_SECONDS=.05
_CONTEXT=None


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def read_captured_process(pid,captured,reader,proc_root=None,*,allow_retired_missing=False):
    """Resample only an incomplete read of an already captured process.

    The signed reader still does its double-stat and cmdline check. A stable
    mismatching identity is never retried. A stable zombie must carry its real
    kernel exit status before the unchanged drain code can accept it.
    """
    if pid not in captured:raise ValueError('PID is not an original captured process')
    expected=captured[pid]
    for attempt in range(SNAPSHOT_ATTEMPTS):
        missing=False
        try:
            value=reader(pid,proc_root) if proc_root is not None else reader(pid)
            missing=value is None
        except ValueError as error:
            if str(error) not in ('torn proc identity','torn proc stat'):raise
            value=None
        if value is not None:
            if any(value.get(k)!=expected.get(k) for k in ('pid','uid','ppid','starttime')):
                raise ValueError('captured process identity changed; recovery refused')
            if value.get('state')=='Z':
                if type(value.get('raw_exit_status')) is int:return value
            else:
                if (value.get('argv') and value['argv']!=expected['argv']
                        or value.get('cwd') is not None and value['cwd']!=expected['cwd']):
                    raise ValueError('captured process identity changed; recovery refused')
                if value.get('argv') and value.get('cwd') is not None:return value
        if attempt+1<SNAPSHOT_ATTEMPTS:time.sleep(SNAPSHOT_DELAY_SECONDS)
    if allow_retired_missing and missing:return None
    raise ValueError('no stable captured process snapshot; recovery refused')


def verify_failure(handoff):
    path=ROOT/FAILURE_EVIDENCE
    if sha(path)!=FAILURE_EVIDENCE_SHA256:raise ValueError('transition failure evidence SHA changed')
    value=read(path)
    if (value.get('kind')!='amos_reuse_proc_transition_failure_v1' or value.get('source_sha256')!=CONTROLLER_SHA256
            or value.get('new_out_exists') is not False
            or value.get('handoff_sha256')!=sha(ROOT/'output/amos_prompt_reuse_handoff_20261030.json')
            or value.get('stopped_sha256')!=sha(ROOT/'output/amos_prompt_reuse_stopped_20261030.json')):
        raise ValueError('transition failure handoff/source chain changed')
    scheduler=value.get('scheduler',{});expected=handoff['scheduler']
    if scheduler.get('state') not in ('T','t') or any(scheduler.get(k)!=expected.get(k) for k in ('pid','uid','ppid','starttime','cwd','argv')):
        raise ValueError('failure evidence scheduler identity changed')
    workers={row['captured']['pid']:row for row in value['workers']}
    if len(workers)!=len(value['workers']) or set(workers)!={row['identity']['pid'] for row in handoff['workers']}:
        raise ValueError('failure evidence captured worker set changed')
    for item in handoff['workers']:
        row=workers[item['identity']['pid']]
        if row['captured']!=item['identity'] or row['spec_sha256']!=item['spec_sha256']:
            raise ValueError('failure evidence captured worker/spec identity changed')
    log=Path(value['helper_log'])
    if not log.is_absolute():log=ROOT/log
    if not log.resolve().is_relative_to(ROOT.resolve()) or log.suffix!='.log':raise ValueError('failure helper log path is not within project')
    if sha(log)!=value.get('helper_log_sha256') or log.read_text(encoding='utf-8')!=value.get('helper_log_text'):
        raise ValueError('failure helper log changed')
    return value


def compile_context(*,source=None):
    """Compile frozen controller bytes; no parent module/global is modified."""
    payload=(ROOT/CONTROLLER).read_bytes() if source is None else source
    if hashlib.sha256(payload).hexdigest()!=CONTROLLER_SHA256:raise ValueError('signed reuse controller SHA changed')
    module=types.ModuleType('_amos_prompt_reuse_procfix_private');c=module.__dict__
    c['__file__']=str(Path(__file__).resolve())
    exec(compile(payload,str(ROOT/CONTROLLER),'exec'),c)
    original_guard=c['source_guard'];original_reader=c['read_process'];original_audit=c['build_reuse_audit'];original_signal=c['signal_exact']
    original_verify=c['verify_reuse'];original_engine=c['engine'];original_drained=c['DRAINED'];original_out=c['OUT'];original_related=c['related']
    c.update(OUT=OUT,TMUX=TMUX,PIPELINE=PIPELINE,DRAINED=DRAINED,
             ENTRIES=c['ENTRIES']|{'amos_prompt_reuse_resume_eval.py'},
             _procfix_original_reader=original_reader,_procfix_original_audit=original_audit,_procfix_original_signal=original_signal)
    retirement=dict(term=False,continued=False)
    def guard():
        if sha(c['ROOT']/CONTROLLER)!=CONTROLLER_SHA256 or sha(c['ROOT']/RUNTIME)!=RUNTIME_SHA256:
            raise ValueError('procfix signed controller/runtime SHA changed')
        original_guard()
    def captured_read(pid,proc_root=None):
        handoff=c['verify_handoff']()
        captured={row['identity']['pid']:row['identity'] for row in handoff['workers']}
        captured[handoff['scheduler']['pid']]=handoff['scheduler']
        retired_scheduler=pid==handoff['scheduler']['pid'] and retirement['continued']
        return read_captured_process(pid,captured,c['_procfix_original_reader'],proc_root,allow_retired_missing=retired_scheduler)
    def signal_exact(identity,sig):
        scheduler=c['verify_handoff']()['scheduler']
        if not c['same_identity'](scheduler,identity):raise ValueError('only captured scheduler retirement is allowed; worker signals refused')
        if sig==15:
            current=captured_read(scheduler['pid'])
            if retirement['term'] or current['state'] not in ('T','t'):raise ValueError('scheduler is not the original frozen retirement target')
            c['_procfix_original_signal'](identity,sig);retirement['term']=True
        elif sig==18 and retirement['term'] and not retirement['continued']:
            c['_procfix_original_signal'](identity,sig);retirement['continued']=True
        else:raise ValueError('only the guarded scheduler TERM/CONT retirement pair is allowed')
    def verify(out):
        verify_failure(c['verify_handoff']());return original_verify(out)
    def audit():
        evidence=verify_failure(c['verify_handoff']());value=c['_procfix_original_audit']()
        extra=[c['ROOT']/FAILURE_EVIDENCE,Path(evidence['helper_log']),c['ROOT']/c['HANDOFF'],c['ROOT']/c['STOPPED']]
        value['files'].update(c['inventory'](extra))
        value['proc_transition_recovery']=dict(evidence_sha256=FAILURE_EVIDENCE_SHA256,
             helper_log=evidence['helper_log'],helper_log_sha256=evidence['helper_log_sha256'],
             original_controller_sha256=CONTROLLER_SHA256,runtime_sha256=RUNTIME_SHA256)
        return value
    def engine():
        ns=original_engine()
        seeds=['tools/amos_prompt_reuse_resume_eval.py',FAILURE_EVIDENCE,c['HANDOFF'],c['STOPPED']]
        if (c['ROOT']/FAILURE_EVIDENCE).exists():
            evidence=read(c['ROOT']/FAILURE_EVIDENCE)
            path=Path(evidence['helper_log'])
            if not path.is_absolute():path=c['ROOT']/path
            seeds.append(c['os'].path.relpath(path,c['ROOT']).replace('\\','/'))
        ns['SOURCE_SEEDS']=tuple(dict.fromkeys((*ns['SOURCE_SEEDS'],*seeds)))
        return ns
    def related(proc_root=None):
        rows=original_related(proc_root);seen={row['pid'] for row in rows};proc=Path(proc_root or '/proc')
        if not proc.exists():return rows
        for folder in proc.iterdir():
            if not folder.name.isdigit() or int(folder.name) in seen or int(folder.name)==os.getpid():continue
            try:args=[arg.decode(errors='replace') for arg in (folder/'cmdline').read_bytes().split(b'\0') if arg]
            except FileNotFoundError:continue
            if 'resume-run' in args and any(Path(arg).name=='amos_prompt_reuse_resume_eval.py' for arg in args):
                rows.append(dict(pid=int(folder.name),args=args))
        return rows
    c.update(source_guard=guard,read_process=captured_read,signal_exact=signal_exact,verify_reuse=verify,build_reuse_audit=audit,engine=engine,
             related=related,ORIGINAL_DRAINED=original_drained,ORIGINAL_OUT=original_out)
    return module


def startup_guard(context):
    c=context.__dict__;c['source_guard']();handoff=c['verify_handoff']();verify_failure(handoff)
    for name in (c['OUT'],c['PIPELINE'],c['ORIGINAL_OUT'],c['ORIGINAL_DRAINED']):
        if (c['ROOT']/name).exists():raise FileExistsError('existing output/log/original drain prevents resume: '+name)
    if (c['ROOT']/c['DRAINED']).exists():
        c['verify_drained']();return handoff
    current=c['read_process'](handoff['scheduler']['pid'])
    if not c['same_identity'](handoff['scheduler'],current) or current['state'] not in ('T','t'):
        raise ValueError('original parent is no longer the same frozen scheduler')
    for row in c['related']():
        args=row['args']
        old=any(Path(arg).name=='amos_prompt_reuse_eval.py' for arg in args) and any(mode in args for mode in ('wait-run','freeze-parent','run','prepare','worker'))
        new=any(Path(arg).name=='amos_prompt_reuse_resume_eval.py' for arg in args) and 'resume-run' in args
        if old or new:
            raise ValueError('original failed helper or reuse worker is still active')
    return handoff


def context():
    global _CONTEXT
    if _CONTEXT is None:_CONTEXT=compile_context()
    return _CONTEXT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('resume-run','worker','status','merge'));p.add_argument('--job')
    args=p.parse_args();c=context()
    if args.mode=='resume-run':
        startup_guard(c);c.finish_parent();return c.engine()['run_pipeline'](c.ROOT/c.OUT)
    return c.engine()['main']()


if __name__=='__main__':main()
