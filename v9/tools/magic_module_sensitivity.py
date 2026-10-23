"""MAGIC held-out fixed-weight sensitivity; reuse audited AMOS interventions."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import amos_module_sensitivity as amos

OUT = 'output/magic_module_sensitivity_v92e360_v102e490_20261023'
HISTORY = amos.HISTORY
variants = amos.variants
apply_intervention = amos.apply_intervention
base_module = amos.base_module
sha = amos.sha


def select_tasks(manifest, split):
    b = base_module()
    rows = sorted((r for r in manifest['tasks'] if r['source_name'] == 'magic'), key=b.key)
    dataset = split['datasets']['magic']
    cases = {r['case_id'] for r in rows}
    val = [r['case_id'] for r in dataset['val']]
    if (len(rows) != 35 or len(cases) != 35 or len({b.key(r) for r in rows}) != 35
            or len(val) != 35 or len(set(val)) != 35 or cases != set(val)
            or cases & {r['case_id'] for r in dataset['train']}):
        raise ValueError('requires all 35 unique MAGIC val cases/tasks; train overlap zero')
    return rows


def validate_refs(rows, refs):
    b = base_module()
    index = {b.key(r): r for r in refs}
    if len(index) != len(refs) or set(index) != {b.key(r) for r in rows}:
        raise ValueError('readonly MAGIC prompt reference coverage mismatch')
    return index


def task_parts(rows):
    b = base_module()
    return [rows[:1]] + [b.case_shard(rows[1:], 4, i) for i in range(4)]


def signature(out):
    b = base_module()
    files = [Path(__file__), ROOT/'tools/amos_module_sensitivity.py',
             ROOT/'tools/ct13_full_validation_eval.py', ROOT/'utils/metrics.py']
    files += sorted(ROOT.glob('*.py'))
    for folder in ('model', 'infer', 'v10', 'v10/model', 'v10/infer'):
        files += sorted((ROOT/folder).glob('*.py'))
    refs = b.read(out/'refs.json')['per_case_class']
    return dict(code={str(p.relative_to(ROOT)): sha(p) for p in files},
                weights={m: dict(path=p, sha256=sha(ROOT/p)) for m, p in b.WEIGHTS.items()},
                split=sha(ROOT/b.SPLIT), manifest=sha(out/'manifest.json'), refs=sha(out/'refs.json'),
                prompts={r['prompt_cache_name']: sha(ROOT/b.CACHE/'metadata/shared_adaptive_prompt_v1/magic'/r['prompt_cache_name']) for r in refs},
                protocol='MAGIC35 fixed-live sensitivity; expanded semantics except no_semantic; '
                         'original single raw-NRRD/model-NIfTI alignment; exact FG scribbles/BG points; '
                         'fullmask10 dynamic .35-.75 fallback .60; readonly CT/prompts')


def verify(out):
    if signature(out) != base_module().read(out/'protocol.json'):
        raise ValueError('changed code/weights/split/prompts; independent output required')


def prepare(out):
    b = base_module()
    if out.exists():
        raise FileExistsError(out)
    rows = select_tasks(b.read(ROOT/HISTORY/'manifest.json'), b.read(ROOT/b.SPLIT))
    refs = []
    for p in sorted((ROOT/HISTORY/'prepared').glob('magic_s*of7/refs.json')):
        refs.extend(b.read(p)['per_case_class'])
    index = validate_refs(rows, refs)
    # All prompts must exist before creating an output; never generate replacements.
    for ref in refs:
        sha(ROOT/b.CACHE/'metadata/shared_adaptive_prompt_v1/magic'/ref['prompt_cache_name'])
    b.write(out/'manifest.json', b.manifest_payload(rows))
    b.write(out/'refs.json', dict(per_case_class=[index[b.key(r)] for r in rows]))
    jobs = []
    for model in ('v9_2', 'v10_2'):
        for variant in variants(model):
            for i, part in enumerate(task_parts(rows)):
                name = model+'__'+variant+('__smoke' if i == 0 else '__s%02d' % i)
                prepared = out/'prepared'/name
                b.write(prepared/'manifest.json', b.manifest_payload(part))
                b.write(prepared/'refs.json', dict(per_case_class=[index[b.key(r)] for r in part]))
                jobs.append(dict(name=name, model=model, variant=variant, source='magic',
                                 prepared=str(prepared), task_count=len(part), frame_batch=16,
                                 attempt=0, smoke=i == 0))
    jobs.sort(key=lambda j: (not j['smoke'], j['name']))
    b.write(out/'jobs.json', jobs)
    b.write(out/'not_applicable.json', dict(v9_2=dict(no_memory_adapter='N/A: original architecture absent',
        no_both_adapters='equivalent to no_3d_adapter; no duplicate prediction')))
    b.write(out/'protocol.json', signature(out))
    print('PREPARED 50 jobs /10 configurations x35 tasks; readonly exact MAGIC prompts', flush=True)


def configure():
    # Only process-local dispatch changes. Neither signed historical source is edited.
    amos.verify = verify
    amos.merge = merge
    amos.__file__ = __file__  # Existing scheduler spawns this independent worker CLI.


def verify_magic_audit(job):
    b = base_module()
    run = Path(job['run'])
    audit = b.read(run/'exact_prompt_audit.json')
    field, expected = ('magic_prompt_alignment', 'raw_nrrd_to_model_nifti_swap_flip_xy_v1') if job['model'] == 'v9_2' else (
        'prompt_coordinate_transform', 'raw_magic_nrrd_to_preprocessed_nifti_swap_flip_xy_v1')
    keys = [tuple(k) for k in audit['loaded_task_keys']]
    tasks = b.read(Path(job['prepared'])/'manifest.json')['tasks']
    if (audit[field] != expected or audit['background_mode'] != 'point' or audit['sources'] != ['magic']
            or len(keys) != len(set(keys)) or set(keys) != {b.key(r) for r in tasks}
            or audit['dynamic_threshold']['minimum'] != .35 or audit['dynamic_threshold']['maximum'] != .75
            or audit['dynamic_threshold']['fallback'] != .60):
        raise ValueError('MAGIC alignment/prompt/threshold audit mismatch')
    return dict(alignment=expected, semantics='expanded_metadata_except_final_zero_no_semantic',
                task_count=len(keys), readonly_prompt_sha_verified=True)


def run(job):
    configure()
    amos.run(job)
    b = base_module()
    audit = b.read(Path(job['run'])/'intervention_audit.json')
    audit['magic_protocol'] = verify_magic_audit(job)
    b.write(Path(job['run'])/'intervention_audit.json', audit)
    print('MAGIC_PROTOCOL_AUDIT_OK', job['name'], flush=True)


def summarize(rows, baseline):
    b = base_module()
    b.validate_rows(baseline, baseline)
    b.validate_rows(rows, baseline)
    paired = {b.key(r): r for r in baseline}
    return dict(tasks=len(rows),
                mean={m: math.fsum(float(r[m]) for r in rows)/len(rows) for m in b.METRICS},
                delta_from_baseline={m: math.fsum(float(r[m])-float(paired[b.key(r)][m]) for r in rows)/len(rows)
                                     for m in b.METRICS}, per_case_class=rows)


def merge(out):
    b = base_module()
    verify(out)
    expected = select_tasks(b.read(out/'manifest.json'), b.read(ROOT/b.SPLIT))
    result = {}
    for model in ('v9_2', 'v10_2'):
        by_variant = {}
        for variant in variants(model):
            rows = []
            for j in b.read(out/'jobs.json'):
                if (j['model'], j['variant']) != (model, variant):
                    continue
                run_dir = out/'jobs'/j['name']
                audit = b.read(run_dir/'intervention_audit.json')
                spec = b.read(out/'job_specs'/(j['name']+'.json'))
                if (not audit['guards_passed'] or audit['variant'] != variant
                        or audit['magic_protocol'] != verify_magic_audit(spec)):
                    raise ValueError('incomplete MAGIC intervention/protocol audit')
                rows += b.read(run_dir/'complete_metrics.json')['per_case_class']
            b.validate_rows(rows, expected)
            by_variant[variant] = rows
        result[model] = {v: summarize(rows, by_variant['baseline']) for v, rows in by_variant.items()}
    b.write(out/'merged_metrics.json', result)
    print('MERGED 10 configurations /350 complete MAGIC tasks', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'worker', 'launch', 'status', 'merge'))
    parser.add_argument('--out', default=OUT)
    parser.add_argument('--job')
    parser.add_argument('--gpus', default='0,2,3,4,5,6')
    args = parser.parse_args()
    os.chdir(ROOT)
    out = (ROOT/args.out).resolve()
    b = base_module()
    if args.mode == 'prepare':
        prepare(out)
    elif args.mode == 'worker':
        run(b.read(args.job))
    elif args.mode == 'merge':
        merge(out)
    elif args.mode == 'launch':
        gpus = [int(x) for x in args.gpus.split(',')]
        if len(set(gpus)) != len(gpus) or not set(gpus) <= {0, 2, 3, 4, 5, 6}:
            raise ValueError('allowed GPUs only 0/2/3/4/5/6 without duplicates')
        configure()
        amos.launch(out, gpus)
    else:
        jobs = b.read(out/'jobs.json')
        print(json.dumps(dict(total_jobs=len(jobs),
            audited_jobs=sum((out/'jobs'/j['name']/'intervention_audit.json').exists() for j in jobs),
            scheduler=b.read(out/'status.json') if (out/'status.json').exists() else None), indent=2))


if __name__ == '__main__':
    main()
