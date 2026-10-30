"""Controller dispatch/provenance regressions for the additive import repair."""
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT=Path(__file__).resolve().parents[1]


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RestoreControllerTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT/'tools/amos_prompt_ablation_restore_eval.py').exists(),'add the additive restored controller')
        self.m=importlib.import_module('tools.amos_prompt_ablation_restore_eval')
        self.m._ENGINE=None

    def fixture(self):
        m=self.m;d=Path(tempfile.mkdtemp());log=d/m.OLD_OUTPUT/'logs/amos_native_control_a0.log'
        queue=dict(time=1,phase='controls',pending=0,running={},failed=[dict(name='amos_native_control',returncode=1,log=str(log))],committed_classes=0)
        for name in m.FAILURE_FILES:
            p=d/name;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text(json.dumps(queue) if name.endswith('/status.json') else 'immutable bootstrap failure fixture\n')
        value=dict(time='fixture',failure='ct13 import order',model_loaded=False,controls_passed=False,full_started=False,
                   committed_classes=0,queue=queue,related_after_independent_second_check=[],
                   files={name:file_sha(d/name) for name in m.FAILURE_FILES})
        path=d/m.BOOTSTRAP_EVIDENCE;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
        return d,path,value

    def test_signed_old_source_and_one_private_globals_preserve_scientific_protocol(self):
        m=self.m;ns=m.engine();old=importlib.import_module('tools.amos_prompt_ablation_eval')
        self.assertEqual(file_sha(ROOT/'tools/amos_prompt_ablation_eval.py'),m.NATIVE_CONTROLLER_SHA256)
        self.assertEqual(ns['OUT'],'output/amos_v102_e490_floor070_prompt_ablations_20261030_restorefix')
        self.assertEqual(ns['TMUX'],'amos_v102_floor070_prompt_ablations_restorefix')
        self.assertEqual(ns['PIPELINE'],'output/amos_prompt_ablations_restorefix_pipeline_20261030.log')
        for name in ('VARIANTS','ALLOWED','METRICS','TRAINING_SHA','WEIGHT_SHA','OLD_SHA'):
            self.assertEqual(ns[name],getattr(old,name))
        for name in ('worker','schedule_queue','class_records','merge','main'):
            self.assertIs(ns[name].__globals__,ns)
        self.assertIs(ns['Backend'].spawn.__globals__,ns)
        self.assertTrue(set(m.FAILURE_FILES)<=set(ns['SOURCE_SEEDS']))
        self.assertIn(m.BOOTSTRAP_EVIDENCE,ns['SOURCE_SEEDS'])
        self.assertIs(ns['runtime'](),importlib.import_module('tools.amos_prompt_ablation_restore_runtime'))

    def test_actual_backend_dispatches_new_worker_entry(self):
        m=self.m;ns=m.engine();d=Path(tempfile.mkdtemp());j=dict(log=str(d/'attempt.log'),gpu=2,spec=str(d/'spec.json'))
        backend=ns['Backend'](d,False)
        with patch.object(ns['subprocess'],'Popen',return_value=types.SimpleNamespace(pid=123)) as spawn:
            process,handle=backend.spawn(j);handle.close()
        command=spawn.call_args.args[0]
        self.assertEqual(Path(command[1]),ROOT/'tools/amos_prompt_ablation_restore_eval.py')
        self.assertEqual(command[2:],['worker','--job',j['spec']]);self.assertEqual(process.pid,123)

    def test_source_guard_refuses_changed_old_controller_or_new_runtime(self):
        m=self.m
        with patch.object(m,'NATIVE_CONTROLLER_SHA256','0'*64):
            with self.assertRaisesRegex(ValueError,'controller'):m.engine()
        m._ENGINE=None
        with patch.object(m,'RESTORE_RUNTIME_SHA256','0'*64):
            with self.assertRaisesRegex(ValueError,'runtime'):m.engine()

    def test_bootstrap_evidence_and_each_bound_file_must_remain_unchanged(self):
        m=self.m;d,path,value=self.fixture()
        with patch.object(m,'BOOTSTRAP_SHA256',file_sha(path)):
            self.assertEqual(m.verify_bootstrap(d),value)
            chosen=d/m.FAILURE_FILES[-1];chosen.write_text('changed historical pipeline')
            with self.assertRaisesRegex(ValueError,'bootstrap|failure'):m.verify_bootstrap(d)
        with patch.object(m,'BOOTSTRAP_SHA256','0'*64):
            with self.assertRaisesRegex(ValueError,'bootstrap'):m.verify_bootstrap(d)

    def test_evidence_must_show_no_model_no_controls_no_full_and_no_commits(self):
        m=self.m
        for changes in (dict(model_loaded=True),dict(controls_passed=True),dict(full_started=True),dict(committed_classes=1),dict(related_after_independent_second_check=[1])):
            d,path,value=self.fixture();value.update(changes);path.write_text(json.dumps(value))
            with patch.object(m,'BOOTSTRAP_SHA256',file_sha(path)):
                with self.assertRaises(ValueError):m.verify_bootstrap(d)

    def test_related_finds_old_and_new_workers_before_fresh_prepare(self):
        m=self.m;d=Path(tempfile.mkdtemp())
        for pid,entry,mode in ((111,'amos_prompt_ablation_eval.py','worker'),(222,'amos_prompt_ablation_restore_eval.py','run'),(333,'other.py','worker')):
            folder=d/str(pid);folder.mkdir();(folder/'cmdline').write_bytes(('python\0'+entry+'\0'+mode+'\0').encode())
        rows=m.related(proc_root=d)
        self.assertEqual({r['pid'] for r in rows},{111,222})
        ns=m.engine()
        with patch.dict(ns,related=lambda:rows):
            with self.assertRaisesRegex(ValueError,'live|duplicate'):ns['preflight'](d/'fresh',tmux_exists=False)

    def test_operational_modes_guard_evidence_before_mutations_and_help_remains_usable(self):
        m=self.m;ns=m.engine();d=Path(tempfile.mkdtemp());pipeline=d/ns['PIPELINE']
        with patch.dict(ns,ROOT=d),patch.object(m,'verify_bootstrap',side_effect=ValueError('changed bootstrap evidence')):
            with self.assertRaisesRegex(ValueError,'bootstrap'):ns['run_pipeline'](d/ns['OUT'])
            with self.assertRaisesRegex(ValueError,'bootstrap'):ns['prepare'](d/ns['OUT'])
            with self.assertRaisesRegex(ValueError,'bootstrap'):ns['verify'](d/ns['OUT'])
        self.assertFalse(pipeline.exists());self.assertFalse((d/ns['OUT']).exists())
        self.assertEqual(set(ns['parser']()._actions[1].choices),{'prepare','run','worker','status','merge'})

    def test_independent_gpu_gate_is_still_required(self):
        m=self.m;ns=m.engine();d=Path(tempfile.mkdtemp());ns['atomic_new'](d/'protocol.json',{'frozen':True})
        with patch.dict(ns,check_controls=lambda out:dict(digest='actual GPU control digest')):
            with self.assertRaises(FileNotFoundError):ns['check_control_gate'](d)
            gate=dict(passed=True,real_gpu=True,reviewer='independent-review',protocol_sha256=ns['sha'](d/'protocol.json'),controls_digest='actual GPU control digest')
            ns['atomic_new'](d/'control_verification.json',gate);self.assertEqual(ns['check_control_gate'](d),gate)


if __name__=='__main__':unittest.main()
