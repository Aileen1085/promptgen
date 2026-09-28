"""CT13 validation metadata and aggregation; never performs inference."""
from collections import Counter, defaultdict
from copy import deepcopy
import math

NEW_SOURCES = {'lndb', 'msd_task06', 'covid19_20', 'kits23', 'lnq2023_lite'}


def class_count_weights(split):
    counts = Counter(r['dataset'] for r in split['class_catalog']
                     if not r.get('not_applicable', False))
    if any(counts[s] <= 0 for s in split['source_order']):
        raise ValueError('source has no nonempty validation classes')
    total = sum(counts.values())
    return {s: counts[s] / total for s in split['source_order']}


def validation_two_per_class(split, validation):
    result = deepcopy(validation)
    groups = defaultdict(list)
    for task in result['tasks']:
        groups[(task['dataset'], int(task['global_class_id']))].append(task)
    chosen = set()
    for row in split['class_catalog']:
        source = row['dataset']
        if row.get('not_applicable', False) or source not in NEW_SOURCES | {'topcow2024_cta'}:
            continue
        key = (source, int(row['global_class_id']))
        unique = []
        seen = set()
        for task in groups[key]:
            if task['case_id'] not in seen:
                unique.append(task)
                seen.add(task['case_id'])
        if len(unique) < 2:
            raise ValueError(f'{key}: requires two distinct nonempty heldout cases')
        if source == 'topcow2024_cta':
            chosen.update((source, key[1], t['case_id']) for t in unique[:2])
    result['tasks'] = [t for t in result['tasks'] if t['dataset'] != 'topcow2024_cta'
                       or (t['dataset'], int(t['global_class_id']), t['case_id']) in chosen]
    result['selection_weight_policy'] = 'nonempty_class_count'
    result['task_macro_policy'] = 'reuse_existing_case_class_metrics'
    result.pop('sha256', None)
    result.pop('split_sha256', None)
    result['protocol'] = str(validation.get('protocol', 'ct13')) + '_topcow2_classweight_v2'
    return result


def case_class_macro(source_metrics):
    sums, counts = defaultdict(float), Counter()
    tasks = 0
    for metrics in source_metrics.values():
        for row in metrics['per_prompt_mode'].values():
            tasks += int(row['count'])
            metric_counts = row.get('metric_counts', {})
            for name, value in row.items():
                if name in ('count', 'metric_counts'):
                    continue
                count = int(metric_counts.get(name, row['count']))
                if count > 0 and math.isfinite(float(value)):
                    sums[name] += float(value) * count
                    counts[name] += count
    return {'count': tasks, 'mean': {n: sums[n]/counts[n] for n in sums},
            'metric_counts': dict(counts)}
