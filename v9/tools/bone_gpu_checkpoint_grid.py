"""Cooperative complete-class handoff into the reviewed Bone GPU pipeline."""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools import bone_gpu_pipeline_grid as base

BASE_SHA='6027539967241c718440c1df9e082bad50f4a12c5d6b47963b5416205378c2b4'
RUNTIME_SHA='f5deaec94d0148c0bfff294636baba6e8e0a1ef385ec54da6c333abdd6356b84'
ENTRY='bone_gpu_checkpoint_grid.py'
OUT='output/bone_v92e360_v102e490_floor_grid_20261028_cpugpu_checkpoint'
TMUX='bone_v92_v102_cpugpu_checkpoint'
PIPELINE='output/bone_cpugpu_checkpoint_pipeline_20261028.log'
CHUNK_CLASSES=8
PROCESS_NAMES=(*base.PROCESS_NAMES,ENTRY)


def reviewed_sources():
    for path,expected in [('tools/bone_gpu_pipeline_grid.py',BASE_SHA),
                          ('tools/bone_gpu_pipeline_runtime.py',RUNTIME_SHA)]:
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=expected:
            raise ValueError('reviewed pipeline source changed: '+path)


def matching_processes(h):
    result=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            r=h.process_record(proc.name)
            if (r and r['state']!='Z' and len(r['args'])>=3
                    and Path(r['args'][1]).name in PROCESS_NAMES
                    and r['args'][2] in ('worker','wait-launch','launch','controls')):result.append(r)
        except (OSError,ValueError):continue
    return result


def handoff_helpers():
    reviewed_sources();h=base.handoff_helpers()
    h.ENTRY=ENTRY
    h.HANDOFF='output/bone_cpugpu_checkpoint_handoff_20261028.json'
    h.STOPPED='output/bone_cpugpu_checkpoint_handoff_stopped_20261028.json'
    h.DRAINED='output/bone_cpugpu_checkpoint_handoff_drained_20261028.json'
    h.matching_processes=lambda:matching_processes(h)
    return h


def chunk_case_jobs(native):
    def jobs(tasks,completed,history):
        result=[]
        for job in native(tasks,completed,history):
            for offset in range(0,len(job['pending']),CHUNK_CLASSES):
                pending=job['pending'][offset:offset+CHUNK_CLASSES]
                result.append(dict(job,name=job['name']+'_chunk'+format(offset//CHUNK_CLASSES,'02d'),
                                   pending=pending,task_count=len(pending)))
        return result
    return jobs


class CheckpointHandoff:
    """Signal only checked own workers; let their native scheduler reap and exit."""
    def __init__(self,value,helpers=None,parent=None,uid=None,kill=None):
        self.value=value;self.h=helpers or handoff_helpers();self.s=parent or base.prior_engine()
        self.root=self.h.ROOT;self.q=self.s.ROOT/self.s.OUT
        self.uid=os.getuid() if uid is None else uid
        self.kill=os.kill if kill is None else kill
        self.workers={};self.intents={};self.exits={};self.resolved={};self.scheduler=None
        self.acknowledged=False;self.status=None

    def status_read(self):
        state=self.s.b.read(self.q/'status.json')
        if (state.get('phase')!='full' or type(state.get('pending')) is not int
                or state['pending']<0 or not isinstance(state.get('running'),dict)
                or not isinstance(state.get('failed'),list)):
            raise ValueError('missing native full scheduler status evidence')
        return state

    def validate_worker(self,r):
        args=r['args']
        if (r['uid']!=self.uid or Path(r['cwd'])!=self.root or not r.get('starttime')
                or r['state'] in ('T','Z') or len(args)!=5 or args[2:4]!=['worker','--job']
                or (self.root/args[1]).resolve()!=(self.root/'tools'/self.h.CURRENT_SCRIPT).resolve()):
            raise ValueError('unexpected own worker identity')
        path=Path(args[4])
        if path.parent!=self.q/'job_specs':raise ValueError('worker spec outside signed parent')
        spec=self.s.b.read(path)
        if not isinstance(spec,dict) or 'name' not in spec:raise ValueError('captured worker spec changed')
        if spec.get('spec')!=str(path):raise ValueError('worker argv differs from canonical signed spec path')
        canonical=next((j for j in self.s.b.read(self.q/'jobs.json')['cases'] if j['name']==spec['name']),None)
        if canonical is None:raise ValueError('worker canonical job missing')
        self.s.validate_attempt(spec,canonical,self.q)
        pending=spec['pending'];keys=[self.s.p.key(x['task']) for x in pending]
        if (not pending or len(keys)!=len(set(keys)) or spec['task_count']!=len(pending)
                or any(x not in canonical['pending'] for x in pending)):
            raise ValueError('worker is not a canonical signed class subset')
        if not Path(spec['log']).is_file():raise ValueError('worker log evidence missing')
        return spec

    def discover(self,processes):
        for r in processes:
            if r['pid']==self.scheduler['pid']:
                if not self.h._same_process(r,self.scheduler):raise ValueError('scheduler identity changed')
                continue
            if len(r['args'])<3 or r['args'][2]!='worker':raise ValueError('unexpected or duplicate related scheduler')
            spec=self.validate_worker(r);path=spec['spec']
            if path in self.workers:
                if not self.h._same_process(r,self.workers[path]):raise ValueError('worker identity changed')
                continue
            if any(old['pid']==r['pid'] for old in self.workers.values()):raise ValueError('captured worker PID reused')
            old=dict(r,spec=path,spec_sha=self.s.sha(path),
                     baseline_markers={str(f):self.s.sha(f) for f in (Path(spec['run'])/'classes').glob('*.done.json')})
            self.workers[path]=old

    def capture(self):
        for name in (self.h.HANDOFF,self.h.STOPPED,self.h.DRAINED):
            if (self.root/name).exists():raise FileExistsError('checkpoint handoff already exists; do not recapture')
        self.s.verify(self.q);self.s.check_controls(self.q)
        processes=self.h.matching_processes()
        schedulers=[r for r in processes if r['args'][2]=='wait-launch']
        if len(schedulers)!=1:raise ValueError('expected exactly one current metricfast scheduler')
        self.scheduler=schedulers[0]
        self.h.validate_scheduler(self.scheduler,self.root,self.uid,frozen=False)
        self.protocol_sha=self.s.sha(self.q/'protocol.json')
        self.catalog_sha=self.s.sha(self.q/'jobs.json')
        self.sources=dict(self.s.b.read(self.q/'protocol.json')['sources'])
        self.status=self.status_read()
        if self.status['failed']:raise ValueError('parent already has an unintended non-resource failure')
        self.discover(processes)
        prior=self.root/'output/bone_cpugpu_handoff_20261028.json'
        evidence={str(prior):self.s.sha(prior)} if prior.exists() else {}
        self.failed_freeze=evidence
        self.s.atomic_new(self.root/self.h.HANDOFF,dict(time=time.strftime('%F %T'),scheduler=self.scheduler,
            workers=list(self.workers.values()),parent_protocol_sha=self.protocol_sha,canonical_catalog_sha=self.catalog_sha,parent_state=self.status,
            prior_failed_freeze=evidence,method='native complete-class checkpoint SIGTERM; no scheduler signal'))
        print('CHECKPOINT_PARENT_CAPTURED',self.scheduler['pid'],len(self.workers),flush=True)

    def immutable_worker(self,old):
        if self.s.sha(old['spec'])!=old['spec_sha']:raise ValueError('captured worker spec changed')
        for path,expected in old['baseline_markers'].items():
            if not Path(path).is_file() or self.s.sha(path)!=expected:raise ValueError('baseline class marker changed')
        now=self.h.process_record(old['pid'])
        if now is not None and not self.h._same_process(now,old):raise ValueError('worker identity changed')
        return now

    def validate_intent(self,old,ref):
        if self.s.sha(ref['path'])!=ref['sha']:raise ValueError('checkpoint intent changed')
        data=self.s.b.read(ref['path']);spec=self.s.b.read(old['spec'])
        if (data['worker']!=old or data['spec_sha']!=old['spec_sha'] or data['log']!=spec['log']
                or data['parent_protocol_sha']!=self.protocol_sha or data['canonical_catalog_sha']!=self.catalog_sha or data['sources']!=self.sources
                or self.s.sha(data['marker'])!=data['marker_sha']):
            raise ValueError('checkpoint intent identity/marker binding changed')
        marker=self.s.b.read(data['marker'])
        if (marker['files']!=data['files'] or len(data['files'])!=7
                or data['files'].get(old['spec'])!=old['spec_sha']):raise ValueError('checkpoint seven-proof binding changed')
        for path,expected in data['files'].items():
            if self.s.sha(path)!=expected:raise ValueError('checkpoint proof changed')
        with Path(spec['log']).open('rb') as handle:prefix=handle.read(data['log_bytes'])
        if len(prefix)!=data['log_bytes'] or hashlib.sha256(prefix).hexdigest()!=data['log_prefix_sha']:
            raise ValueError('checkpoint log evidence changed')
        return data

    def audit_status(self,state):
        seen=set()
        for failure in state['failed']:
            matches=[path for path in self.intents if self.s.b.read(path)['name']==failure.get('name')
                     and self.s.b.read(path)['log']==failure.get('log')]
            if len(matches)!=1 or type(failure.get('returncode')) is not int or failure['returncode']!=-signal.SIGTERM:
                raise ValueError('unintended native non-resource failure lacks exact checkpoint intent')
            path=matches[0]
            if path in seen:raise ValueError('duplicate native failure acknowledgement')
            seen.add(path);old=self.workers[path];ref=self.intents[path]
            self.validate_intent(old,ref)
            self.exits[path]=dict(worker=old,wait_status=signal.SIGTERM,intent=ref,native_failure=dict(failure))
        if not set(self.exits)<=seen:raise ValueError('native failure acknowledgement disappeared')
        if state['failed']:self.acknowledged=True

    def retire(self,old,now,records):
        path=old['spec']
        if now is None or now['state']=='Z' or path in self.intents:return
        spec=self.s.b.read(path);self.validate_worker(now)
        fresh=[p for p in (Path(spec['run'])/'classes').glob('*.done.json') if str(p) not in old['baseline_markers']]
        if not fresh:return
        allowed={item['marker']:item for item in records.values()}
        marker=next((p for p in sorted(fresh) if str(p) in allowed),None)
        if marker is None:
            # A class may commit after tick's initial snapshot. Validate again
            # after observing the fresh immutable marker, before judging it.
            allowed={item['marker']:item for item in self.s.class_records(self.q).values()}
            marker=next((p for p in sorted(fresh) if str(p) in allowed),None)
        if marker is None:raise ValueError('new complete-class marker failed native validation')
        item=allowed[str(marker)];info=self.s.b.read(marker)
        if item['marker_sha']!=self.s.sha(marker) or info['job_name']!=spec['name']:
            raise ValueError('validated boundary marker identity changed')
        log=Path(spec['log']).read_bytes()
        data=dict(worker=old,spec_sha=old['spec_sha'],log=spec['log'],log_bytes=len(log),
            log_prefix_sha=hashlib.sha256(log).hexdigest(),marker=str(marker),marker_sha=self.s.sha(marker),
            files=info['files'],parent_protocol_sha=self.protocol_sha,canonical_catalog_sha=self.catalog_sha,
            sources=self.sources,time=time.strftime('%F %T'))
        intent=self.root/'output'/('bone_cpugpu_checkpoint_stop_'+str(old['pid'])+'_'+old['starttime']+'_20261028.json')
        self.s.atomic_new(intent,data)
        ref=dict(path=str(intent),sha=self.s.sha(intent));self.intents[path]=ref
        self.validate_intent(old,ref)
        now=self.immutable_worker(old)
        if now is None or now['state']=='Z':return
        # Record intent once; never resend if a signal or natural exit wins this race.
        try:self.kill(old['pid'],signal.SIGTERM)
        except ProcessLookupError:pass
        print('CHECKPOINT_OWN_WORKER_SIGTERM',old['pid'],str(marker),flush=True)

    def resource_evidence(self,old,spec,text):
        launcher=(self.root/'output/bone_metricfast_pipeline_20261028.log').read_text(errors='replace')
        # Bound historical requeues to the actual captured PID, before any next
        # attempt of the same canonical job starts. Later retries stay distinct.
        started=False;resource=False;segment=[]
        for line in launcher.splitlines():
            words=line.split()
            if len(words)>3 and words[0]=='SHARED_START' and words[3]==spec['name']:
                if started:break
                started=words[2]==str(old['pid'])
            elif started and len(words)>1 and words[0]=='SHARED_RESOURCE_REQUEUE' and words[1]==spec['name']:
                resource=True
            if started:segment.append(line)
        if not resource:raise ValueError('departed worker lacks positive native completion/resource evidence')
        return self.h.recorded_resource_exit(spec['name'],text,'\n'.join(segment))

    def departed(self,old,now,state,records):
        path=old['spec']
        if now is not None and now['state']!='Z':return
        spec=self.s.b.read(path);text=Path(spec['log']).read_text(errors='replace')[-14000:]
        if now is not None:
            status=now['exit_status']
            if status!=signal.SIGTERM or path not in self.intents:
                self.h.inherited_history(spec,text,Path(spec['log']).stat().st_mtime,status)
        if any(r.get('spec')==path for r in state['running'].values()):return
        if path in self.exits:
            self.exits[path]['final_log_sha']=self.s.sha(spec['log'])
            self.resolved[path]='native_checkpoint_ack';return
        if now is not None and now['exit_status']==signal.SIGTERM:
            status=self.resource_evidence(old,spec,text)
            self.h.inherited_history(spec,text,Path(spec['log']).stat().st_mtime,status)
            self.resolved[path]='native_resource_requeue';return
        complete=all((spec['model'],self.s.p.key(item['task'])) in records for item in spec['pending'])
        if complete:self.resolved[path]='complete_class_records';return
        status=self.resource_evidence(old,spec,text)
        self.h.inherited_history(spec,text,Path(spec['log']).stat().st_mtime,status)
        self.resolved[path]='native_resource_requeue'

    def tick(self):
        if self.scheduler is None:raise ValueError('capture required before checkpoint polling')
        if self.s.sha(self.q/'protocol.json')!=self.protocol_sha:raise ValueError('signed parent protocol changed')
        if self.s.sha(self.q/'jobs.json')!=self.catalog_sha:raise ValueError('signed canonical catalog changed')
        for path,expected in self.sources.items():
            if self.s.sha(self.root/path)!=expected:raise ValueError('signed parent source changed')
        for path,expected in self.failed_freeze.items():
            if self.s.sha(path)!=expected:raise ValueError('original failed-freeze evidence changed')
        state=self.status_read();now=self.h.process_record(self.scheduler['pid'])
        if now is not None:
            if not self.h._same_process(now,self.scheduler):raise ValueError('scheduler identity changed')
            if now['state']!='Z':self.h.validate_scheduler(now,self.root,self.uid,frozen=False)
        elif state['running']:raise ValueError('parent disappeared before native drain acknowledgement')
        self.discover(self.h.matching_processes());self.audit_status(state)
        records=self.s.class_records(self.q)
        for old in self.workers.values():
            worker=self.immutable_worker(old)
            self.retire(old,worker,records)
            self.departed(old,self.immutable_worker(old),state,records)
        self.status=state
        scheduler_gone=now is None or now['state']=='Z'
        if scheduler_gone and not self.acknowledged and (state['running'] or state['pending']):
            raise ValueError('parent exited without checkpoint or successful completion acknowledgement')
        return scheduler_gone and not state['running'] and len(self.resolved)==len(self.workers)

    def finish(self):
        if not self.tick():raise ValueError('native scheduler/workers are not yet drained')
        self.value.require_handoff([dict(verified=True)],self.value.related())
        time.sleep(3)
        self.value.require_handoff([dict(verified=True)],self.value.related())
        # Only persist final drained evidence after both independent quiescence checks.
        self.s.atomic_new(self.root/self.h.STOPPED,dict(scheduler=self.scheduler,workers=list(self.workers.values()),
            handoff_sha=self.s.sha(self.root/self.h.HANDOFF),native_status=self.status,
            native_status_sha=self.s.sha(self.q/'status.json'),dispatch_stop_acknowledged=self.acknowledged,
            natural_full_completion=not self.status['failed'] and not self.status['pending']))
        self.s.atomic_new(self.root/self.h.DRAINED,dict(exits=self.exits,resolved=self.resolved,
            stopped_sha=self.s.sha(self.root/self.h.STOPPED),all_workers_exited=True))
        return dict(verified=True,checkpoint_handoff=True,evidence={str(self.root/n):self.s.sha(self.root/n)
                    for n in (self.h.HANDOFF,self.h.STOPPED,self.h.DRAINED)})


def finish_parent(value,helpers):
    handoff=CheckpointHandoff(value,helpers);handoff.capture()
    while not handoff.tick():time.sleep(2)
    return handoff.finish()


def engine():
    h=handoff_helpers();value=base.engine()
    value.OUT=OUT;value.__file__=__file__;value.SOURCES+=('tools/bone_gpu_checkpoint_grid.py',)
    value.related=lambda:[str(r['pid']) for r in h.matching_processes()]
    value.finish_parent=lambda:finish_parent(value,h)
    value.collect_parent=h.collect_parent
    value.case_jobs=chunk_case_jobs(value.case_jobs)
    native_signature=value.signature
    def signature(out):
        result=native_signature(out)
        result['checkpoint_handoff_policy']=dict(class_chunk_limit=CHUNK_CLASSES,
            native_scheduler_acknowledgement=True,entry=ENTRY,output=OUT,tmux=TMUX,pipeline=PIPELINE)
        return result
    value.signature=signature
    return value


if __name__=='__main__':engine().main()
