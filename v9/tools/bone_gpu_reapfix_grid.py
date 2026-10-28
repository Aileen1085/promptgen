"""Resume the immutable Bone checkpoint handoff across native worker reaping."""
import hashlib
import os
from pathlib import Path
import sys
import time
import types

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools import bone_gpu_checkpoint_grid as checkpoint

CHECKPOINT_SHA='6bd6cb7f797bd9bafab5bec1913b9801e50e7f93a0c06125d56baadc0b6b11e0'
CHECKPOINT_TEST_SHA='0fdc98cd2738e93cc482af0cd8a125ca9ed6e2b9bf457f86c342c70ed6fca56f'
FAILED_LOG_SHA='e798f03e35da7a6d1d580d3650a2ce9b2aca3fea0eab677a58e81fbd4e354057'
HANDOFF_SHA='6fab51ad4bd2834a1a3c28450bf764cb107f9d68af904175486653077ad0bd63'
PRIMARY_INTENT_NAME='bone_cpugpu_checkpoint_stop_392719_17824298_20261028.json'
PRIMARY_INTENT_SHA='52f0b6b63f75c386ab1da3173eb2234c8b88030a5459985e015aff13b3183a34'
ENTRY='bone_gpu_reapfix_grid.py'
OUT=checkpoint.OUT+'_reapfix'
TMUX=checkpoint.TMUX+'_reapfix'
PIPELINE='output/bone_cpugpu_checkpoint_reapfix_pipeline_20261028.log'
RESUME='output/bone_cpugpu_checkpoint_reapfix_resume_20261028.json'
CHUNK_CLASSES=checkpoint.CHUNK_CLASSES
IDENTITY_WAIT_SECONDS=60
PROCESS_NAMES=(*checkpoint.PROCESS_NAMES,ENTRY)


def reviewed_sources():
    checkpoint.reviewed_sources()
    for path,expected in [('tools/bone_gpu_checkpoint_grid.py',CHECKPOINT_SHA),
                          ('tests/test_bone_gpu_checkpoint_grid.py',CHECKPOINT_TEST_SHA)]:
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=expected:
            raise ValueError('frozen checkpoint source/test changed: '+path)


def handoff_helpers():
    reviewed_sources();h=checkpoint.handoff_helpers();h.ENTRY=ENTRY
    # Preserve the original private collector and its existing evidence paths.
    namespace=dict(checkpoint.matching_processes.__globals__,PROCESS_NAMES=PROCESS_NAMES)
    discover=types.FunctionType(checkpoint.matching_processes.__code__,namespace)
    h.matching_processes=lambda:discover(h)
    return h


class IdentityPending(Exception):
    """An uncertain /proc snapshot grants no permission to send a signal."""


class CheckpointHandoff(checkpoint.CheckpointHandoff):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.clock=time.monotonic;self.identity_deadlines={};self.resume_sha=None

    def immutable_files(self,old):
        if self.s.sha(old['spec'])!=old['spec_sha']:raise ValueError('captured worker spec changed')
        for path,expected in old['baseline_markers'].items():
            if not Path(path).is_file() or self.s.sha(path)!=expected:raise ValueError('baseline class marker changed')

    def restore(self):
        """Load the exact failed waiter evidence without recapturing any baseline."""
        reviewed_sources()
        for name in (self.h.STOPPED,self.h.DRAINED,checkpoint.OUT,OUT,RESUME):
            if (self.root/name).exists():raise FileExistsError('checkpoint resume/output already exists: '+name)
        failed=self.root/checkpoint.PIPELINE;handoff=self.root/self.h.HANDOFF
        if self.s.sha(failed)!=FAILED_LOG_SHA:raise ValueError('failed waiter log SHA mismatch')
        if failed.read_text(errors='replace').splitlines()[-2:]!=['ValueError: worker identity changed','CHECKPOINT_PIPELINE_RC=1']:
            raise ValueError('failed waiter log does not prove the known identity-read failure and RC1')
        if self.s.sha(handoff)!=HANDOFF_SHA:raise ValueError('original checkpoint handoff SHA mismatch')
        self.s.verify(self.q);self.s.check_controls(self.q)
        evidence=self.s.b.read(handoff)
        self.protocol_sha=evidence['parent_protocol_sha'];self.catalog_sha=evidence['canonical_catalog_sha']
        if (self.s.sha(self.q/'protocol.json')!=self.protocol_sha
                or self.s.sha(self.q/'jobs.json')!=self.catalog_sha):raise ValueError('signed parent protocol/catalog changed')
        self.sources=dict(self.s.b.read(self.q/'protocol.json')['sources'])
        for path,expected in self.sources.items():
            if self.s.sha(self.root/path)!=expected:raise ValueError('signed parent source changed')
        self.failed_freeze=dict(evidence['prior_failed_freeze']);self.scheduler=evidence['scheduler']
        self.h.validate_scheduler(self.scheduler,self.root,self.uid,frozen=False)
        for old in evidence['workers']:
            spec=self.validate_worker(old);self.immutable_files(old)
            if spec['spec'] in self.workers:raise ValueError('duplicate captured worker spec')
            self.workers[spec['spec']]=old
        primary=self.root/'output'/PRIMARY_INTENT_NAME
        if self.s.sha(primary)!=PRIMARY_INTENT_SHA:raise ValueError('original checkpoint intent SHA mismatch')
        records=self.s.class_records(self.q);allowed={item['marker']:item for item in records.values()}
        for path in sorted((self.root/'output').glob('bone_cpugpu_checkpoint_stop_*_20261028.json')):
            intent=self.s.b.read(path);old=intent['worker'];spec=self.validate_worker(old)
            self.immutable_files(old)
            if old['spec'] in self.workers and self.workers[old['spec']]!=old:
                raise ValueError('intent worker differs from immutable original capture')
            self.workers[old['spec']]=old
            ref=dict(path=str(path),sha=self.s.sha(path))
            self.validate_intent(old,ref)
            marker=allowed.get(intent['marker'])
            if marker is None or marker['marker_sha']!=intent['marker_sha']:
                raise ValueError('restored intent marker failed native class validation')
            if old['spec'] in self.intents:raise ValueError('duplicate original worker stop intents')
            self.intents[spec['spec']]=ref
        if not self.intents:raise ValueError('original checkpoint intent missing')
        # Native acknowledgement has priority over any new observation of the PID.
        # Once Popen.poll() reported -15, that original process has been reaped.
        self.status=self.status_read();self.audit_status(self.status)
        if not self.acknowledged:raise ValueError('resume requires actual native checkpoint stop acknowledgement')
        for path,expected in self.failed_freeze.items():
            if self.s.sha(path)!=expected:raise ValueError('original failed-freeze evidence changed')
        data=dict(time=time.strftime('%F %T'),handoff=dict(path=str(handoff),sha=HANDOFF_SHA),
            failed_waiter_log=dict(path=str(failed),sha=FAILED_LOG_SHA,returncode=1),
            frozen_checkpoint_sources={'tools/bone_gpu_checkpoint_grid.py':CHECKPOINT_SHA,
                'tests/test_bone_gpu_checkpoint_grid.py':CHECKPOINT_TEST_SHA},
            intents=self.intents,workers=list(self.workers.values()),scheduler=self.scheduler,
            parent_protocol_sha=self.protocol_sha,canonical_catalog_sha=self.catalog_sha,
            native_status=self.status,native_status_sha=self.s.sha(self.q/'status.json'))
        self.s.atomic_new(self.root/RESUME,data);self.resume_sha=self.s.sha(self.root/RESUME)
        self.failed_log_sha=FAILED_LOG_SHA
        print('CHECKPOINT_REAPFIX_RESUMED',len(self.workers),len(self.exits),flush=True)
        return dict(resumed=True,evidence=dict(path=str(self.root/RESUME),sha=self.resume_sha))

    def defer_identity(self,old):
        deadline=self.identity_deadlines.setdefault(old['spec'],self.clock()+IDENTITY_WAIT_SECONDS)
        if self.clock()>=deadline:raise ValueError('unresolved worker identity after fixed 60 seconds: '+str(old['pid']))
        raise IdentityPending(old['spec'])

    def positive_departure(self,old):
        state=self.status_read();self.audit_status(state)
        if old['spec'] in self.exits:return True
        if any(r.get('spec')==old['spec'] for r in state['running'].values()):return False
        records=self.s.class_records(self.q)
        try:super().departed(old,None,state,records)
        except ValueError as error:
            if str(error)!='departed worker lacks positive native completion/resource evidence':raise
            return False
        return old['spec'] in self.resolved

    def immutable_worker(self,old):
        self.immutable_files(old)
        if old['spec'] in self.exits or old['spec'] in self.resolved:return None
        try:now=self.h.process_record(old['pid'])
        except OSError as error:
            if error.errno not in (2,3):raise
            now=None
        if now is not None and self.h._same_process(now,old):return now
        # Disappearance, reuse, and torn argv/cwd are all uncertain until native
        # status or complete-class/resource evidence identifies the old exit.
        if self.positive_departure(old):return None
        self.defer_identity(old)

    def discover(self,processes):
        captured={old['pid']:old for old in self.workers.values()}
        for record in processes:
            old=captured.get(record['pid'])
            if old is not None:
                self.immutable_worker(old)
                continue
            super().discover([record])

    def tick(self):
        try:
            if self.resume_sha is not None:
                if self.s.sha(self.root/RESUME)!=self.resume_sha:raise ValueError('immutable resume evidence changed')
                if self.s.sha(self.root/checkpoint.PIPELINE)!=self.failed_log_sha:raise ValueError('failed waiter log changed')
            self.audit_status(self.status_read())
            return super().tick()
        except IdentityPending:
            return False

    def finish(self):
        result=super().finish()
        if self.resume_sha is not None:result['evidence'][str(self.root/RESUME)]=self.resume_sha
        return result


def finish_parent(value,helpers):
    handoff=CheckpointHandoff(value,helpers);handoff.restore()
    while not handoff.tick():time.sleep(2)
    return handoff.finish()


def engine():
    h=handoff_helpers();value=checkpoint.engine()
    value.OUT=OUT;value.__file__=__file__;value.SOURCES+=('tools/bone_gpu_reapfix_grid.py',)
    value.related=lambda:[str(r['pid']) for r in h.matching_processes()]
    value.finish_parent=lambda:finish_parent(value,h)
    def collect_parent():
        completed,history,specs,files,recovered=h.collect_parent()
        for path,expected in [(ROOT/RESUME,None),(ROOT/checkpoint.PIPELINE,FAILED_LOG_SHA)]:
            actual=value.sha(path)
            if expected is not None and actual!=expected:raise ValueError('failed waiter evidence changed')
            files[str(path)]=actual
        return completed,history,specs,files,recovered
    value.collect_parent=collect_parent
    native_signature=value.signature
    def signature(out):
        result=native_signature(out)
        result['reapfix_handoff_policy']=dict(entry=ENTRY,output=OUT,tmux=TMUX,pipeline=PIPELINE,
            resume_sha=value.sha(ROOT/RESUME),failed_waiter_log_sha=FAILED_LOG_SHA,
            checkpoint_source_sha=CHECKPOINT_SHA,checkpoint_test_sha=CHECKPOINT_TEST_SHA,
            fixed_identity_wait_seconds=IDENTITY_WAIT_SECONDS)
        return result
    value.signature=signature
    return value


if __name__=='__main__':engine().main()
