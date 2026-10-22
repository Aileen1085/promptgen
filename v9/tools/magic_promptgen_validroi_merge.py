"""Merge completed MAGIC valid-ROI predictions, auditing slots versus voxels."""
import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def validate_counts(points, foreground_count, stage):
    if len(points) != 4 or any(len(p) != 3 for p in points):
        raise ValueError('exactly four original background slots required')
    unique_count = len({tuple(p) for p in points})
    a = stage['calibration']; t = stage['threshold']
    if (stage['background_points_encoded'] != 4
            or not stage['all_prompts_before_forward'] or stage['gt_used_for_roi']
            or a['foreground_outside_predicted_roi'] or a['background_outside_predicted_roi']
            or foreground_count <= 0 or a['valid_foreground_count'] != foreground_count
            or a['valid_background_count'] != unique_count
            or t['foreground_count'] != foreground_count or t['background_count'] != unique_count):
        raise ValueError('calibration count/coverage does not match cached unique voxels')
    return unique_count


def merge(out):
    import numpy as np
    from tools import magic_promptgen_validroi_eval as fix
    fix.install(); e = fix.e; base = e.base
    e.verify(out)  # Retain all original worker/code/weight/prompt signature checks.
    expected = base.read(out/'manifest.json')['tasks']
    sources = {}; audit_rows = []; cross_model = {}; counts = Counter()
    for job in base.read(out/'jobs.json'):
        run = out/'jobs'/job['name']
        for name in ('complete_metrics.json', 'model_audit.json', 'exact_prompt_audit.json'):
            p = run/name; sources[str(p.relative_to(ROOT))] = e.sha(p)
        part = base.read(run/'complete_metrics.json')['per_case_class']
        base.validate_rows(part, base.read(Path(job['prepared'])/'manifest.json')['tasks'])
        if not base.read(run/'model_audit.json')['strict_effective_modules_exact']:
            raise ValueError('model restoration mismatch')
        exact = base.read(run/'exact_prompt_audit.json')['per_case_class']
        if exact != [dict(task_key=list(base.key(r)),prompt_audit=r['prompt_audit'],stages=r['stages']) for r in part]:
            raise ValueError('exact prompt audit differs from completed rows')
        for row in part:
            a = row['prompt_audit']
            path = ROOT/base.CACHE/'metadata/shared_adaptive_prompt_v1/magic'/row['prompt_cache_name']
            if e.sha(path) != a['prompt_cache_sha256']:
                raise ValueError('raw prompt cache signature mismatch')
            with np.load(path, allow_pickle=False) as z:
                shape = tuple(map(int,z['shape_dhw']))
                fg = np.unpackbits(z['coronal_bits'],count=int(np.prod(shape)),bitorder='little')
                fg |= np.unpackbits(z['sagittal_bits'],count=int(np.prod(shape)),bitorder='little')
                raw_fg = np.argwhere(fg.reshape(shape))
                raw_bg = z['background_points_dhw'].copy()
                planes, _ = e.mapped_plane_metadata(z['slices'],a['source_affine_dhw_ras'],a['target_affine_dhw_ras'])
            mapped_bg = e.map_background(raw_bg,a['source_affine_dhw_ras'],a['target_affine_dhw_ras'])
            mapped_fg = e.map_background(raw_fg,a['source_affine_dhw_ras'],a['target_affine_dhw_ras'])
            if (not np.array_equal(mapped_bg,np.asarray(a['bg_points_dhw']))
                    or list(planes) != a['plane_slices'] or len(mapped_fg) != a['fg_voxels']
                    or {tuple(p) for p in mapped_fg} & {tuple(p) for p in mapped_bg}):
                raise ValueError('cached prompt mapping/foreground mismatch')
            prompt_identity = (a['prompt_cache_sha256'],a['bg_points_dhw'],a['plane_slices'],row['semantic_text'])
            key = base.key(row)
            if key in cross_model and cross_model[key] != prompt_identity:
                raise ValueError('models used different prompts or semantics')
            cross_model[key] = prompt_identity
            if not row['stages'] or row['epoch'] != e.ASSETS[job['model']][1]:
                raise ValueError('missing stages or wrong epoch')
            for stage in row['stages']:
                unique = validate_counts(mapped_bg.tolist(),len(mapped_fg),stage)
                bounds = stage.get('bounds',[(stage['z_bounds'] if 'z_bounds' in stage else [0,0]),
                    [0,a['target_shape_dhw'][1]],[0,a['target_shape_dhw'][2]]])
                bounds = np.asarray(bounds,int)
                points = np.concatenate((mapped_fg,mapped_bg))
                if (bounds.shape != (3,2) or (bounds[:,0]<0).any()
                        or (bounds[:,1]>a['target_shape_dhw']).any() or (bounds[:,0]>=bounds[:,1]).any()
                        or (points<bounds[:,0]).any() or (points>=bounds[:,1]).any()
                        or int(np.prod(bounds[:,1]-bounds[:,0])) != stage['calibration']['actual_predicted_voxels']
                        or not .35 <= stage['threshold']['threshold'] <= .75):
                    raise ValueError('explicit stage bounds do not cover all cached prompts')
                if stage.get('bg_roundtrip_voxels',0) > 1e-4:
                    raise ValueError('background roundtrip error')
            counts[job['model']] += 1
            audit_rows.append(dict(model=job['model'],task_key=list(key),background_slots=4,
                unique_background_voxels=unique,foreground_voxels=len(mapped_fg),stages=len(row['stages'])))
    if counts != Counter({m:35 for m in e.ASSETS}) or set(cross_model) != {base.key(t) for t in expected}:
        raise ValueError('requires exact 3x35 task scope')
    # Only aggregation is reused, after the stricter independent raw-cache audit.
    # Signed prediction entry/protocol/results are never changed or bypassed.
    fix.ORIGINAL_MERGE(out)
    if any(e.sha(ROOT/p) != value for p,value in sources.items()):
        raise ValueError('completed source changed during merge')
    audit = dict(entry_sha256=e.sha(__file__),signed_worker_sha256=e.sha(fix.__file__),
        policy='four original slots; exact unique voxel counts derived from readonly raw cache; no inference rerun',
        model_counts=dict(counts),source_files=sources,per_case_class=audit_rows,
        merged_sha256=e.sha(out/'merged_metrics.json'))
    base.write(out/'merge_validroi_audit.json',audit)
    print('AUDITED',dict(counts),'duplicate-slot tasks',sum(r['unique_background_voxels']<4 for r in audit_rows))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='output/magic_promptgen_three_backbones35_20261022_validroi')
    args=p.parse_args();merge(ROOT/args.out)


if __name__=='__main__':main()
