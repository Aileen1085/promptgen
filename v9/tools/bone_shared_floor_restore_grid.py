"""Independent restoration-order recovery of the frozen shared-stage Bone grid."""
from pathlib import Path
import hashlib
import types


def recovery_source(text):
    old = "('bone_floor_memory_wait_recovery.py','bone_shared_floor_grid.py')"
    if text.count(old) != 1:raise ValueError('shared process guard anchor changed')
    text = text.replace(old, "('bone_floor_memory_wait_recovery.py','bone_shared_floor_grid.py','bone_shared_floor_restore_grid.py')")
    old = 'from tools.bone_shared_floor_runtime import GridSession,install_v92,install_v10,install_restoration,tag_carrier'
    if text.count(old) != 1:raise ValueError('shared runtime import anchor changed')
    text = text.replace(old, 'from tools.bone_shared_floor_restore_runtime import GridSession,install_v92,install_v10,install_restoration,tag_carrier')
    old = "job.get('frame_batch') not in (1,2)"
    if text.count(old) != 1:raise ValueError('attempt batch guard anchor changed')
    text = text.replace(old, "job.get('frame_batch') not in ((1,2,4) if canonical['control'] else (1,2))")
    # Batch4 is the last native canary configuration independently matched to
    # the original smokes. Production keeps its inherited smaller batch policy.
    old = 'task_count=1,frame_batch=2,minimum_free_mib=30720'
    if text.count(old) != 1:raise ValueError('fresh control batch anchor changed')
    text = text.replace(old, 'task_count=1,frame_batch=4,minimum_free_mib=30720')
    old = 'session=GridSession(job[\'model\'],expected,json.dumps(b.read(out/\'protocol.json\'),sort_keys=True),committed)'
    if text.count(old) != 1:raise ValueError('session control binding anchor changed')
    return text.replace(old, old+"\n    session.shared_control = bool(job['control'])")


def engine():
    root = Path(__file__).resolve().parents[1]
    base = root/'tools/bone_shared_floor_grid.py'
    source = base.read_bytes()
    if hashlib.sha256(source).hexdigest() != 'b1cedcb405e26df9c07f451d1eac01c9cccd2dd53f5f87fb3fa70c674eca12d2':
        raise ValueError('frozen shared-grid source changed')
    text = recovery_source(source.decode('utf-8'))
    value = types.ModuleType('bone_shared_floor_restore_engine')
    value.__file__ = __file__
    exec(compile(text, str(base), 'exec'), value.__dict__)
    prior = root/value.OUT
    value.OUT += '_restorefix'
    value.SOURCES += ('tools/bone_shared_floor_restore_runtime.py', 'tools/bone_shared_floor_restore_grid.py')
    native_signature = value.signature
    def signature(out):
        # This failed control-only attempt is immutable evidence, never final rows.
        if not prior.exists():raise ValueError('missing original shared control evidence')
        from tools import bone_shared_floor_grid as original
        original.verify(prior)
        result = native_signature(out)
        result['prior_shared_control_files'] = {
            str(f): value.sha(f) for f in sorted(prior.rglob('*')) if f.is_file()
        }
        return result
    value.signature = signature
    native_check_controls = value.check_controls
    def check_controls(out):
        records = native_check_controls(out)
        for (model, _key), item in records.items():
            if model != 'v10_2':continue
            probes = item['record'].get('native_batch_probe', {})
            if set(map(int,probes)) != {1,2,4}:raise ValueError('missing actual native batch twin controls')
            for probe in probes.values():
                if (not probe.get('exact_same_batch') or probe['raw_sha'] != probe['native_raw_sha']
                        or not probe['encoder_shas'] or probe['encoder_shas'] != probe['native_encoder_shas']):
                    raise ValueError('native batch twin proof failed')
        return records
    value.check_controls = check_controls
    return value


if __name__ == '__main__':engine().main()
