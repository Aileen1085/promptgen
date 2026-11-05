"""Report the actual mask/metrics of one frozen nnInteractive qualitative rerun.

Use only with manifest reference_policy='report_actual_rerun', capture_attempt=1.
All strict native manifest, task, input/source, physical GPU, full-mask and normal
completion guards remain in force. Historical ten-metric parity is reported,
never optimized or required. Figures must use actual newly computed Dice.
This entry neither selects cases nor retries to match historical metrics.
The original strict capture module and evaluators remain untouched.

Manifest fields otherwise follow capture_qualitative_nninteractive_case.py. Mandatory
source inventory includes both that immutable helper and this separate entry.
Run once per fresh process/output, with an original nnInteractive single-task
job and existing shared CT/prompt references. No CT or GT arrays are exported.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
from tools import capture_qualitative_nninteractive_case as strict


def validate_policy(spec):
    strict.validate_scope(spec)
    if (spec.get('method') != 'nnInteractive'
            or spec.get('reference_policy') != 'report_actual_rerun'
            or type(spec.get('capture_attempt')) is not int or spec['capture_attempt'] != 1):
        raise ValueError('explicit nnInteractive report_actual_rerun first-attempt policy required')


def metric_report(actual, reference):
    if strict.key(actual) != strict.key(reference):
        raise ValueError('actual/historical task key mismatch')
    values = []
    for row in (actual, reference):
        metrics = {}
        for name in strict.METRICS:
            value = row.get(name)
            if (type(value) not in (int, float) or not math.isfinite(value) or value < 0
                    or (name != 'pred_gt_ratio' and value > 1)):
                raise ValueError('missing/nonfinite/out-of-range actual/reference metric: ' + name)
            metrics[name] = float(value)
        values.append(metrics)
    deltas = {name: values[0][name] - values[1][name] for name in strict.METRICS}
    parity = {name: abs(deltas[name]) <= 1e-6 for name in strict.METRICS}
    return dict(actual_metrics=values[0], historical_metrics=values[1], actual_minus_historical=deltas,
                per_metric_match_atol_1e_6=parity, all_ten_match_atol_1e_6=all(parity.values()),
                max_absolute_metric_delta=max(abs(v) for v in deltas.values()),
                interpretation='Actual frozen-case rerun; differences are disclosed, not attributed to a proven cause')


def runtime_context():
    """Observe existing backend settings; do not alter numerical execution."""
    torch = sys.modules.get('torch')
    if torch is None:
        return dict(torch_loaded=False)
    return dict(torch_loaded=True, torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                cudnn_version=torch.backends.cudnn.version(), torch_threads=torch.get_num_threads(),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)


def capture(manifest_path):
    manifest_path = Path(manifest_path).absolute()
    spec = strict.read(manifest_path)
    validate_policy(spec)
    base = manifest_path.parent
    job_path, output = base / spec['job_path'], base / spec['output_dir']
    job = strict.read(job_path)
    run, prepared = ROOT / job['run'], ROOT / job['prepared']
    if output.exists() or run.exists():
        raise FileExistsError('actual-rerun capture and original worker outputs must both be new')
    if job.get('model') != 'nninteractive':
        raise ValueError('nnInteractive original job required')
    tasks = strict.read(prepared / 'manifest.json')['tasks']
    refs = strict.read(prepared / 'refs.json')['per_case_class']
    if (len(tasks) != 1 or len(refs) != 1 or strict.key(tasks[0]) != strict.key(spec['task'])
            or strict.key(refs[0]) != strict.key(spec['task'])):
        raise ValueError('prepared job and prompt refs must contain exactly the frozen task')
    task = dict(tasks[0], prompt_cache_name=refs[0]['prompt_cache_name'])
    strict.validate_task_details(spec['task'], task)
    metric_report(spec['reference'], dict(spec['task'], **{m:spec['reference'].get(m) for m in strict.METRICS}))
    gpu_binding = strict.validate_nn_gpu_index(job.get('gpu'), spec['gpu_uuid'])
    sources, inputs = strict.required_bindings(spec, job, job_path)
    sources.append(Path(__file__).absolute())
    inventory = strict.verify_inventory(spec['source_sha256'], sources, base)
    inventory.update(strict.verify_inventory(spec['input_sha256'], inputs, base))
    before_job, before_manifest = strict.sha(job_path), strict.sha(manifest_path)
    minimum = spec.get('minimum_free_mib', 30720)
    resource = strict.resource_check(spec['gpu_uuid'], minimum)
    observer = strict.MaskObserver(spec['shape_dhw'])
    previous_argv, previous_cwd = list(sys.argv), Path.cwd()
    try:
        os.chdir(ROOT)
        free = strict.gpu_free_memory(spec['gpu_uuid'])
        if free < minimum:
            raise ValueError('insufficient free memory before original nnInteractive delegate')
        if strict.validate_nn_gpu_index(job['gpu'], spec['gpu_uuid']) != gpu_binding:
            raise ValueError('nnInteractive physical GPU mapping changed')
        resource.update(free_before_delegate_mib=free, nninteractive_gpu_binding=gpu_binding)
        strict.run_delegate('nnInteractive', job_path, job, spec['gpu_uuid'], observer)
    finally:
        sys.argv = previous_argv
        os.chdir(previous_cwd)
    rows = strict.read(run / 'complete_metrics.json')['per_case_class']
    if len(rows) != 1 or observer.mask is None:
        raise ValueError('normal original completion and observed full-volume mask required')
    comparison = metric_report(rows[0], spec['reference'])
    strict.compare_reference(dict(spec['task'], **observer.metrics), rows[0])
    if strict.sha(job_path) != before_job or strict.sha(manifest_path) != before_manifest:
        raise ValueError('actual-rerun manifest/job changed')
    for name, digest in inventory.items():
        if strict.sha(name) != digest:
            raise ValueError('signed source/input changed: ' + name)
    strict.save_capture(output, observer, dict(schema_version=1, method='nnInteractive', task=spec['task'],
        reference_policy='report_actual_rerun', capture_attempt=1, normal_completion=True,
        reference=spec['reference'], actual_row=rows[0], historical_comparison=comparison,
        figure_metric_policy='Use actual mask Dice, never substitute historical Dice',
        case_selection_policy='Frozen before rerun; no winner selection or retries to match metrics',
        grid_verification=spec['grid_verification'], resource=resource, runtime_context=runtime_context(),
        source_and_input_sha256=inventory, manifest_sha256=before_manifest, job_sha256=before_job,
        observer_sha256=strict.sha(__file__), strict_helper_sha256=strict.sha(strict.__file__),
        worker_complete_metrics_sha256=strict.sha(run / 'complete_metrics.json'),
        checkpoint_restoration='Original initialize_from_trained_model_folder and normal evaluation completed; checkpoint hashes bound'))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    args = parser.parse_args(argv)
    print('NNINTERACTIVE_ACTUAL_CAPTURE_COMPLETE', capture(args.manifest))


if __name__ == '__main__':
    main()
