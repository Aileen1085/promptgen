"""Offline finalization of the four natively acknowledged Bone checkpoints."""
import hashlib
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools import bone_gpu_reapfix_grid as reapfix

REAPFIX_SHA='232f31f00c601280b12965bd0d0bc207e78581bbe4516f00816247d9f44d501a'
REAPFIX_TEST_SHA='1215007f563ef21e9f940fd1c6419bf94de7aa322cb220d692572df5f161cdc6'
OLD_RESUME_SHA='a61c2f291d03812d5e9d56c5d7ae7577c69e6c3419b7ee131910a3349188adff'
FAILED_REAPFIX_LOG_SHA='3421bc46cf430a4ccd29802b36c6e7827d38ef2c89f8d518e8464e026fac1fd0'
FINAL_STATUS_SHA='f32656d071ddc8e180dd562dcc57b57b7b799b7610496417a6ade2e58525a547'
ENTRY='bone_gpu_postdrain_grid.py'
OUT='output/bone_v92e360_v102e490_floor_grid_20261028_cpugpu_postdrain'
TMUX='bone_v92_v102_cpugpu_postdrain'
PIPELINE='output/bone_cpugpu_postdrain_pipeline_20261028.log'
RESUME='output/bone_cpugpu_postdrain_resume_20261028.json'
PROCESS_NAMES=(*reapfix.PROCESS_NAMES,ENTRY)


def reviewed_sources():
    reapfix.reviewed_sources()
    for path,expected in [('tools/bone_gpu_reapfix_grid.py',REAPFIX_SHA),
                          ('tests/test_bone_gpu_reapfix_grid.py',REAPFIX_TEST_SHA)]:
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=expected:
            raise ValueError('frozen reapfix source/test changed: '+path)


def handoff_helpers():
    reviewed_sources();h=reapfix.handoff_helpers();h.ENTRY=ENTRY
    native=reapfix.checkpoint.matching_processes
    discover=types.FunctionType(native.__code__,dict(native.__globals__,PROCESS_NAMES=PROCESS_NAMES))
    h.matching_processes=lambda:discover(h)
    return h


def reject_signal(*args):raise ValueError('offline postdrain finalizer cannot send signals')


class CheckpointHandoff(reapfix.CheckpointHandoff):
    def __init__(self,*args,**kwargs):
        kwargs['kill']=reject_signal;super().__init__(*args,**kwargs)

    def capture(self):raise ValueError('offline postdrain finalizer cannot recapture workers')

    def closed_status(self):
        if self.s.sha(self.q/'status.json')!=self.final_status_sha:raise ValueError('final native status changed')
        state=self.status_read()
        workers=list(self.workers.values()) if self.workers else self.s.b.read(self.root/self.h.HANDOFF)['workers']
        expected={(self.s.b.read(w['spec'])['name'],self.s.b.read(w['spec'])['log'],-15) for w in workers}
        actual={(f.get('name'),f.get('log'),f.get('returncode')) for f in state['failed']}
        if state['running'] or len(workers)!=4 or len(expected)!=4 or len(state['failed'])!=4 or actual!=expected:
            raise ValueError('offline handoff requires exactly four native intent-bound -15 acknowledgements and no running workers')
        return state

    def restore(self):
        reviewed_sources()
        if (self.root/reapfix.OUT).exists():raise FileExistsError('old reapfix output already exists')
        self.offline_refs={str(self.root/reapfix.RESUME):OLD_RESUME_SHA,
                           str(self.root/reapfix.PIPELINE):FAILED_REAPFIX_LOG_SHA}
        for path,expected in self.offline_refs.items():
            if self.s.sha(path)!=expected:raise ValueError('frozen failed reapfix evidence changed')
        if (self.root/reapfix.PIPELINE).read_text(errors='replace').splitlines()[-2:]!=[
                'ValueError: unresolved worker identity after fixed 60 seconds: 425925','REAPFIX_PIPELINE_RC=1']:
            raise ValueError('failed reapfix log does not prove the known timeout and RC1')
        if self.s.sha(self.root/self.h.HANDOFF)!=reapfix.HANDOFF_SHA:raise ValueError('original checkpoint handoff SHA mismatch')
        self.final_status_sha=FINAL_STATUS_SHA;self.closed_status()
        for _ in range(2):self.value.require_handoff([dict(verified=True)],self.value.related())
        native=reapfix.CheckpointHandoff.restore
        if native.__closure__:raise ValueError('frozen restore unexpectedly acquired a closure')
        # Reuse every original proof guard; only the new audit/output destinations
        # and the additive source guard differ in this private function namespace.
        namespace=dict(native.__globals__,RESUME=RESUME,OUT=OUT,reviewed_sources=reviewed_sources)
        result=types.FunctionType(native.__code__,namespace)(self)
        if len(self.exits)!=4 or len(self.workers)!=4:raise ValueError('offline native acknowledgement count changed')
        self.checkpoint_log_sha=self.s.sha(self.root/reapfix.checkpoint.PIPELINE)
        return result

    def immutable_worker(self,old):
        self.immutable_files(old)
        if old['spec'] not in self.exits:raise ValueError('offline worker lacks native acknowledgement')
        return None

    def retire(self,old,now,records):
        if now is not None or old['spec'] not in self.exits:raise ValueError('offline finalizer cannot retire a live worker')

    def tick(self):
        if self.s.sha(self.root/RESUME)!=self.resume_sha:raise ValueError('postdrain resume evidence changed')
        for path,expected in self.offline_refs.items():
            if self.s.sha(path)!=expected:raise ValueError('frozen failed reapfix evidence changed')
        if self.s.sha(self.root/reapfix.checkpoint.PIPELINE)!=self.checkpoint_log_sha:raise ValueError('failed checkpoint log changed')
        if self.s.sha(self.q/'protocol.json')!=self.protocol_sha or self.s.sha(self.q/'jobs.json')!=self.catalog_sha:
            raise ValueError('signed parent protocol/catalog changed')
        for path,expected in {**{str(self.root/p):v for p,v in self.sources.items()},**self.failed_freeze}.items():
            if self.s.sha(path)!=expected:raise ValueError('signed parent source/evidence changed')
        self.status=self.closed_status();self.audit_status(self.status)
        for old in self.workers.values():
            self.immutable_worker(old);self.departed(old,None,self.status,{})
        return True

    def finish(self):
        result=reapfix.checkpoint.CheckpointHandoff.finish(self)
        result['evidence'].update({str(self.root/RESUME):self.resume_sha,**self.offline_refs})
        return result


def finish_parent(value,helpers):
    handoff=CheckpointHandoff(value,helpers);handoff.restore();return handoff.finish()


def engine():
    h=handoff_helpers();value=reapfix.engine()
    value.OUT=OUT;value.__file__=__file__;value.SOURCES+=('tools/bone_gpu_postdrain_grid.py',)
    value.related=lambda:[str(r['pid']) for r in h.matching_processes()]
    value.finish_parent=lambda:finish_parent(value,h)
    prior_collect=value.collect_parent
    def collect_parent():
        completed,history,specs,files,recovered=prior_collect()
        for path,expected in [(ROOT/RESUME,None),(ROOT/reapfix.RESUME,OLD_RESUME_SHA),(ROOT/reapfix.PIPELINE,FAILED_REAPFIX_LOG_SHA)]:
            actual=value.sha(path)
            if expected is not None and actual!=expected:raise ValueError('frozen postdrain evidence changed')
            files[str(path)]=actual
        return completed,history,specs,files,recovered
    value.collect_parent=collect_parent;native_signature=value.signature
    def signature(out):
        result=native_signature(out)
        result['postdrain_handoff_policy']=dict(entry=ENTRY,output=OUT,tmux=TMUX,pipeline=PIPELINE,
            resume_sha=value.sha(ROOT/RESUME),old_resume_sha=OLD_RESUME_SHA,
            failed_reapfix_log_sha=FAILED_REAPFIX_LOG_SHA,final_native_status_sha=FINAL_STATUS_SHA,
            reapfix_source_sha=REAPFIX_SHA,reapfix_test_sha=REAPFIX_TEST_SHA,offline_native_acknowledgements=4)
        return result
    value.signature=signature;return value


if __name__=='__main__':engine().main()
