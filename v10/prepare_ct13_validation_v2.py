"""Publish a versioned CT13 validation list without altering training splits."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ct13_validation_policy import class_count_weights, validation_two_per_class
from v9_2_extended_protocol import protocol_sha256
from tools.build_v9_2_extended_multidataset_manifest import write_json_with_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    split = json.loads(args.split.read_text())
    old = json.loads(args.validation.read_text())
    validation = validation_two_per_class(split, old)
    for task in validation['tasks']:
        source = task['dataset']
        data = split['datasets'][source]
        heldout = {r['case_id']: r for r in data['val']}
        training = {r['case_id'] for r in data['train']}
        case = task['case_id']
        if case in training or case not in heldout or task['global_class_id'] not in heldout[case]['classes']:
            raise ValueError(f'not a nonempty heldout task: {task}')
    validation['selection_weights'] = class_count_weights(split)
    validation['split_sha256'] = split['sha256']
    validation['sha256'] = protocol_sha256(validation)
    digest = write_json_with_sha256(args.output, validation)
    print(json.dumps({'path': str(args.output), 'sha256': digest,
                     'tasks': len(validation['tasks']),
                     'source_tasks': dict(Counter(t['dataset'] for t in validation['tasks'])),
                     'selection_weights': validation['selection_weights']}, indent=2))


if __name__ == '__main__':
    main()
