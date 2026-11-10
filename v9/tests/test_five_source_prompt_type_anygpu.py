import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class AnyGpuTests(unittest.TestCase):
    def module(self):
        try:
            module=importlib.import_module('tools.five_source_prompt_type_anygpu')
            self.addCleanup(setattr,module,'_installed_out',module._installed_out)
            self.addCleanup(setattr,module,'_installed_digest',module._installed_digest)
            module._installed_out=None;module._installed_digest=None
            return module
        except ModuleNotFoundError:
            self.fail('additive any-GPU resource overlay missing')

    def policy(self, m, out):
        (out/'protocol.json').write_text('{}')
        digest=m.e.sha(out/'protocol.json')
        with patch.object(m,'EXPECTED_PROTOCOL_SHA',digest):
            return m.freeze_policy(out,'a'*40,'User explicitly authorizes any currently free GPU')

    def test_policy_freezes_source_and_protocol_without_overwriting(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);self.policy(m,out)
            with patch.object(m,'EXPECTED_PROTOCOL_SHA',m.e.sha(out/'protocol.json')):
                p=m.read_policy(out)
                self.assertEqual(p['allowed_physical_gpus'],list(range(8)))
                self.assertEqual(p['max_workers'],4)
                self.assertEqual(p['min_free_mib'],30720)
                with self.assertRaises(FileExistsError):m.freeze_policy(out,'a'*40,'authorization')
                path=out/m.POLICY_NAME
                value=json.loads(path.read_text());value['max_workers']=8
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):m.read_policy(out)

    def test_bind_checks_twice_cuda_identity_waits_and_preserves_45000(self):
        m=self.module()
        torch=SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda i:SimpleNamespace(uuid='abc')),
                              ones=lambda *a,**k:SimpleNamespace(item=lambda:1),set_num_threads=lambda n:None)
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);self.policy(m,out)
            with patch.object(m,'EXPECTED_PROTOCOL_SHA',m.e.sha(out/'protocol.json')), \
                 patch.object(m.e.shutil,'disk_usage',return_value=SimpleNamespace(free=10*1024**3)), \
                 patch.dict('sys.modules',torch=torch):
                job=dict(gpu=7,gpu_uuid='GPU-abc',minimum_free_mib=45000)
                with patch.object(m.e.subprocess,'check_output',side_effect=['7, GPU-abc, 50000','7, GPU-abc, 49000']):
                    binding=m.bind_resources(job,out)
                self.assertEqual(binding['gpu'],7)
                self.assertEqual(binding['minimum_free_mib'],45000)
                self.assertTrue(binding['cuda_probe'])
                m.validate_binding(binding,out)
                for snapshots in (['7, GPU-abc, 44000'],['7, GPU-abc, 50000','7, GPU-abc, 44000']):
                    with patch.object(m.e.subprocess,'check_output',side_effect=snapshots):
                        with self.assertRaises(SystemExit) as err:m.bind_resources(job,out)
                        self.assertEqual(err.exception.code,75)
                with patch.object(m.e.subprocess,'check_output',return_value='7, GPU-wrong, 1'):
                    with self.assertRaises(ValueError):m.bind_resources(job,out)
                with patch.object(m.e.shutil,'disk_usage',return_value=SimpleNamespace(free=1)):
                    with self.assertRaises(SystemExit) as err:m.bind_resources(job,out)
                    self.assertEqual(err.exception.code,75)

    def test_historical_gpu03_valid_new_cards_require_policy_binding(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);self.policy(m,out)
            with patch.object(m,'EXPECTED_PROTOCOL_SHA',m.e.sha(out/'protocol.json')):
                for gpu in (0,3):
                    m.validate_binding(dict(gpu=gpu),out)
                for gpu in (1,2,4,5,6,7):
                    with self.assertRaises(ValueError):m.validate_binding(dict(gpu=gpu),out)
                with self.assertRaises(ValueError):m.validate_binding(dict(gpu=8),out)

    def test_install_only_changes_two_resource_functions(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);self.policy(m,out)
            with patch.object(m,'EXPECTED_PROTOCOL_SHA',m.e.sha(out/'protocol.json')):
                evaluate=m.runtime.initialize;validate=m.runtime.validate_record;worker=m.e.worker
                with patch.object(m.runtime,'resource_snapshot'),patch.object(m.e,'bind_resources'):
                    m.install(out)
                    self.assertIs(m.e.bind_resources,m.bind_resources)
                    self.assertIs(m.runtime.resource_snapshot,m.resource_snapshot)
                    self.assertIs(m.runtime.initialize,evaluate)
                    self.assertIs(m.runtime.validate_record,validate)
                    self.assertIs(m.e.worker,worker)
                    self.assertEqual(m.runtime.resource_snapshot('5, GPU-abc, 50000',5,'GPU-abc')['gpu'],5)
                    with self.assertRaises(ValueError):m.runtime.resource_snapshot('5, GPU-wrong, 50000',5,'GPU-abc')

    def test_policy_rehashes_dispatcher_and_rejects_live_policy_change(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);(out/'protocol.json').write_text('{}')
            with patch.object(m,'EXPECTED_PROTOCOL_SHA',m.e.sha(out/'protocol.json')):
                p=m.freeze_policy(out,'a'*40,'user authorization', ['tests/test_five_source_prompt_type_anygpu.py'])
                self.assertEqual(len(p['extra_source_sha256']),1)
                original=m.e.sha
                with patch.object(m.e,'sha',side_effect=lambda path:'wrong' if str(path).endswith('test_five_source_prompt_type_anygpu.py') else original(path)):
                    with self.assertRaisesRegex(ValueError,'dispatcher'):m.read_policy(out)
                with patch.object(m.runtime,'resource_snapshot'),patch.object(m.e,'bind_resources'):
                    m.install(out)
                    p['user_authorization']='changed';p['policy_sha256']=m.policy_digest(p)
                    (out/m.POLICY_NAME).write_text(json.dumps(p))
                    with self.assertRaisesRegex(ValueError,'changed'):m.read_policy(out)

    def test_records_audit_catches_unbound_new_gpu_and_snapshot_tampering(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);p=self.policy(m,out)
            with patch.object(m,'EXPECTED_PROTOCOL_SHA',m.e.sha(out/'protocol.json')):
                snapshot=dict(gpu=6,gpu_uuid='GPU-abc',free_mib=50000,minimum_free_mib=30720)
                binding=dict(gpu=6,gpu_uuid='GPU-abc',cuda_probe=True,minimum_free_mib=45000,
                    first_snapshot=dict(snapshot),second_snapshot=dict(snapshot),free_mib_before_load=50000,
                    resource_policy_sha256=p['policy_sha256'],resource_policy_path=m.POLICY_NAME,
                    resource_policy_source_sha256=p['source_sha256'])
                path=out/'records/source/case/baseline/1.json';path.parent.mkdir(parents=True)
                path.write_text(json.dumps(dict(resource_binding=binding)))
                self.assertEqual(m.verify_records(out)['records']['resource_policy'],1)
                binding['first_snapshot']['gpu_uuid']='GPU-other'
                path.write_text(json.dumps(dict(resource_binding=binding)))
                with self.assertRaises(ValueError):m.verify_records(out)
                path.write_text(json.dumps(dict(resource_binding=dict(gpu=6))))
                with self.assertRaises(ValueError):m.verify_records(out)
                path.write_text(json.dumps(dict(resource_binding=dict(gpu=0))))
                self.assertEqual(m.verify_records(out)['records']['historical_protocol'],1)

    def test_worker_dispatch_rehashes_policy_without_scanning_all_records(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);self.policy(m,out)
            with patch.object(m,'EXPECTED_PROTOCOL_SHA',m.e.sha(out/'protocol.json')), \
                 patch.object(m.runtime,'resource_snapshot'),patch.object(m.e,'bind_resources'), \
                 patch.object(m,'verify_records',side_effect=AssertionError('full record scan in worker')), \
                 patch.object(m,'read_policy',wraps=m.read_policy) as rehash,patch.object(m.e,'main') as dispatch:
                for mode in ('worker','control'):
                    before=rehash.call_count
                    m.main([mode,'--out',str(out),'--job','job.json'])
                    self.assertGreaterEqual(rehash.call_count-before,2)
                    dispatch.assert_called_with([mode,'--out',str(out.absolute()),'--job','job.json'])

    def test_policy_binding_enforces_45000_on_both_batch_one_snapshots(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);p=self.policy(m,out)
            with patch.object(m,'EXPECTED_PROTOCOL_SHA',m.e.sha(out/'protocol.json')):
                binding=dict(gpu=7,gpu_uuid='GPU-abc',cuda_probe=True,minimum_free_mib=45000,
                    first_snapshot=dict(gpu=7,gpu_uuid='GPU-abc',free_mib=50000),
                    second_snapshot=dict(gpu=7,gpu_uuid='GPU-abc',free_mib=50000),free_mib_before_load=50000,
                    resource_policy_sha256=p['policy_sha256'],resource_policy_path=m.POLICY_NAME,
                    resource_policy_source_sha256=p['source_sha256'])
                m.validate_binding(binding,out)
                for name in ('first_snapshot','second_snapshot'):
                    binding[name]['free_mib']=44000
                    with self.assertRaises(ValueError):m.validate_binding(binding,out)
                    binding[name]['free_mib']=50000


if __name__=='__main__':unittest.main()
