"""Explicit readonly result reuse; changed code never resumes old signatures."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import native_ct13_eval as native

OLD_ENTRY_SHA = '42b1fb40b7cbc62b3f8c9e348c73a73d776afca3113fae4f10a0b479f9f99965'
OLD = native.ROOT / 'output/native_sam2_medsam2_ct13_full6094_20261007'
OUTPUT = native.ROOT / 'output/native_sam2_medsam2_ct13_full6094_20261007_gridfix'


def pending_tasks(tasks, rows, audits):
    keys = [native.key(r) for r in rows]; audit_keys = [native.key(a) for a in audits]
    expected = {native.key(t) for t in tasks}
    if (len(keys) != len(set(keys)) or len(audit_keys) != len(set(audit_keys)) or
            set(keys) != set(audit_keys) or not set(keys).issubset(expected)):
        raise ValueError('unsafe partial reuse: missing/duplicate/out-of-scope audit')
    if any(k not in r or not math.isfinite(float(r[k])) for r in rows for k in native.METRICS):
        raise ValueError('unsafe partial reuse: missing/nonfinite metric')
    return [t for t in tasks if native.key(t) not in set(keys)]


def validate_audits(audits):
    for a in audits:
        assert a['complete_frame_prediction'] and not a['promptgen_loaded'] and not a['semantic_input']
        assert a['fg_in_gt_fraction'] == 1 and a['bg_in_gt_count'] == 0
        assert a['physical_roundtrip_error_mm'] < .001
        assert a['dense_prior'] == 'none' and a['memory_mode'] == 'on'
        assert a['native_mask_threshold_logits'] == 0
        for s in a['stages']:
            assert s['foreground_supported_frames'] == s['foreground_point_frames']
            assert s['background_support_frames'] == s['background_point_frames']
            assert s['effective_memory_slots'] == 7
            for points in s['point_provenance']:
                assert len(points['fg_sampled_xy']) <= 2 and len(points['bg_original_yx']) <= 2


def old_code_signature(model):
    assets = native.asset_spec(model); code = native.ROOT / assets['code_root']
    files = [native.ROOT/'tools/native_sam2_physical_eval.py',
             native.ROOT/'infer/native_sam2_physical_prompts.py', native.ROOT/'utils/metrics.py',
             code/'sam2/build_sam.py', code/'sam2/sam2_video_predictor.py', code/'sam2'/assets['config']]
    signature = hashlib.sha256((OLD_ENTRY_SHA + ''.join(native.sha(p) for p in files)).encode()).hexdigest()
    return assets, native.sha(native.ROOT/assets['checkpoint']), signature


def prepare():
    assert not OUTPUT.exists(), 'recovery directory exists; inspect before creating again'
    manifest = native.read(OLD/'manifest.json'); tasks = manifest['tasks']
    assert len(tasks) == 6094 and len({native.key(t) for t in tasks}) == 6094
    assert manifest['train_overlap'] == 0 and manifest['per_source_counts'] == native.COUNTS
    split = native.read(native.ROOT/'configs/ct13_v10_2_split_20260928.json')
    for t in tasks:
        d = split['datasets'][t['source_name']]
        assert t['case_id'] in {r['case_id'] for r in d['val']}
        assert t['case_id'] not in {r['case_id'] for r in d['train']}
    signatures = {model:old_code_signature(model) for model in ('sam2', 'medsam2')}
    jobs = []; refs = []; seen = defaultdict(set); completed = 0
    for meta in native.read(OLD/'jobs.json')['jobs']:
        job = native.read(meta['path']); out = Path(job['output'])
        file = out/'complete_metrics.json'
        rows, audits = ((native.read(file)['per_case_class'], native.read(out/'exact_prompt_audit.json')['per_case_class'])
                        if file.exists() else native.collect_attempts(job))
        pending = pending_tasks(job['tasks'], rows, audits); validate_audits(audits)
        if rows:
            assets, weight, code = signatures[job['model']]
            protocol = native.read(out/'protocol.json')
            expected = native.protocol_signature(job, weight, code)
            assert protocol['assets'] == assets and protocol['weight_sha256'] == weight
            assert protocol['code_sha256'] == code and protocol['signature'] == expected
            assert protocol['protocol'] == native.PROTOCOL
            if file.exists():
                assert native.read(file)['signature'] == expected
                assert native.read(out/'exact_prompt_audit.json')['signature'] == expected
            files = [Path(meta['path']), out/'protocol.json']
            if file.exists(): files += [file, out/'exact_prompt_audit.json']
            else:
                files += [p for a in sorted(out.glob('attempt_*')) if a.is_dir()
                          for p in (a/'all_metrics_live.json', a/'exact_prompt_audit.json') if p.exists()]
            refs.append(dict(job_path=meta['path'], model=job['model'], count=len(rows),
                             complete=file.exists(), files={str(p):native.sha(p) for p in files}))
            keys = {native.key(r) for r in rows}
            assert not keys & seen[job['model']]; seen[job['model']].update(keys)
        if not pending: completed += 1
        else:
            new_job = dict(job, tasks=pending, output=str(OUTPUT/'jobs'/job['name']))
            path = OUTPUT/'job_specs'/(job['name']+'.json')
            jobs.append(dict(meta, path=str(path), tasks=len(pending)))
            # Written only after every old result signature has been verified.
    OUTPUT.mkdir(parents=True)
    for meta in jobs:
        old_job = native.read(next(r['path'] for r in native.read(OLD/'jobs.json')['jobs'] if r['name']==meta['name']))
        remaining = [t for t in old_job['tasks'] if native.key(t) not in seen[old_job['model']]]
        native.write(meta['path'], dict(old_job, tasks=remaining, output=str(OUTPUT/'jobs'/old_job['name'])))
    native.write(OUTPUT/'manifest.json', manifest)
    native.write(OUTPUT/'jobs.json', dict(jobs=jobs, total_tasks=sum(j['tasks'] for j in jobs)))
    native.write(OUTPUT/'reuse_audit.json', dict(old_root=str(OLD), old_entry_sha256=OLD_ENTRY_SHA,
                 new_entry_sha256=native.sha(Path(native.__file__)), recovery_entry_sha256=native.sha(Path(__file__)),
                 old_completed_jobs=completed, original_jobs=208, refs=refs,
                 reused_tasks={k:len(v) for k,v in seen.items()}, reason='s0808 sform rounding, identical qform; no tensor/prompt/model change'))
    print('RECOVERY_PREPARED', len(jobs), 'jobs;', {k:len(v) for k,v in seen.items()}, 'readonly reused', flush=True)


def reused_rows():
    reuse = native.read(OUTPUT/'reuse_audit.json'); result = defaultdict(list)
    assert reuse['new_entry_sha256'] == native.sha(Path(native.__file__))
    assert reuse['recovery_entry_sha256'] == native.sha(Path(__file__))
    for ref in reuse['refs']:
        for file, checksum in ref['files'].items(): assert native.sha(file) == checksum, 'old result changed'
        job = native.read(ref['job_path']); out = Path(job['output'])
        rows = native.read(out/'complete_metrics.json')['per_case_class'] if ref['complete'] else native.collect_attempts(job)[0]
        assert len(rows) == ref['count']; result[ref['model']] += rows
    return result


ORIGINAL_PROGRESS = native.progress


def progress(output):
    result = ORIGINAL_PROGRESS(output); old = reused_rows()
    result['completed_jobs'] += native.read(OUTPUT/'reuse_audit.json')['old_completed_jobs']
    for model, rows in old.items():
        result['model_tasks'][model] = result['model_tasks'].get(model, 0) + len(rows)
        source = Counter(result['per_source'].get(model, {})); source.update(r['source_name'] for r in rows)
        result['per_source'][model] = dict(source)
    return result


def merge(output):
    import numpy as np
    result = {}; reused = reused_rows(); manifest = native.read(OUTPUT/'manifest.json')['tasks']
    for model in ('sam2', 'medsam2'):
        rows = list(reused[model])
        assets, weight, _ = old_code_signature(model)
        for meta in native.read(OUTPUT/'jobs.json')['jobs']:
            if meta['model'] == model:
                job = native.read(meta['path']); out = Path(job['output'])
                data = native.read(out/'complete_metrics.json'); protocol = native.read(out/'protocol.json')
                files = [Path(native.__file__),native.ROOT/'tools/native_sam2_physical_eval.py',
                         native.ROOT/'infer/native_sam2_physical_prompts.py',native.ROOT/'utils/metrics.py',
                         native.ROOT/assets['code_root']/'sam2/build_sam.py',
                         native.ROOT/assets['code_root']/'sam2/sam2_video_predictor.py',
                         native.ROOT/assets['code_root']/'sam2'/assets['config']]
                code = hashlib.sha256(''.join(native.sha(p) for p in files).encode()).hexdigest()
                expected = native.protocol_signature(job, weight, code)
                assert protocol['signature'] == data['signature'] == expected
                audits = native.read(out/'exact_prompt_audit.json'); assert audits['signature'] == expected
                validate_audits(audits['per_case_class']); native.validate_rows(data['per_case_class'], job['tasks'])
                rows += data['per_case_class']
        native.validate_rows(rows, manifest); assert dict(Counter(r['source_name'] for r in rows)) == native.COUNTS
        result[model] = dict(per_case_class=rows, summary={k:float(np.mean([r[k] for r in rows])) for k in native.METRICS},
                            per_source={s:dict(count=n, **{k:float(np.mean([r[k] for r in rows if r['source_name']==s]))
                                                         for k in native.METRICS}) for s,n in native.COUNTS.items()})
    native.write(OUTPUT/'merged_metrics.json', dict(methods=result, protocol=native.PROTOCOL, task_count_per_model=6094,
                 readonly_reuse_audit='reuse_audit.json', previous_output=str(OLD)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true'); parser.add_argument('--schedule', action='store_true')
    parser.add_argument('--status', action='store_true'); parser.add_argument('--merge', action='store_true')
    args = parser.parse_args()
    if args.prepare: prepare()
    elif args.status: print(progress(OUTPUT))
    elif args.merge: merge(OUTPUT)
    elif args.schedule:
        reused_rows(); native.progress = progress; native.merge = merge
        native.schedule(OUTPUT, {0,2,3,4,5,6})
    else: parser.error('choose an action')
