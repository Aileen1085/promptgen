"""Independent result-audit fixtures; temporary files are deliberately retained."""
import copy
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


METRICS = ('dice', 'mean_slice_dice', 'iou', 'precision', 'recall', 'specificity',
           'pred_gt_ratio', 'mean_slice_nsd_1mm', 'mean_slice_nsd_2mm', 'mean_slice_nsd_3mm')
FLOORS = {'v9_2': .35, 'v10_2': .75}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


class Fixture:
    """Two unequal class populations distinguish task mean from class macro."""
    def __init__(self):
        self.root = Path(tempfile.mkdtemp(prefix='bone_independent_audit_')).resolve()
        self.out = self.root / 'fresh'
        self.old = self.root / 'old.json'
        self.nn = self.root / 'nn.json'
        self.tasks = [dict(source_name='bone', case_id=c, global_class_id=g,
                           local_class_id=g-10000, class_name='bone_'+str(g))
                      for c, g in [('a', 10001), ('b', 10001), ('a', 10002)]]
        dump(self.out/'manifest.json', dict(tasks=self.tasks, expected_task_count=3))
        dump(self.out/'refs.json', dict(per_case_class=self.tasks))
        dump(self.out/'protocol.json', {'fixture': 'immutable protocol'})
        dump(self.out/'status.json', dict(phase='full', pending=0, running={}, failed=[]))
        jobs, controls = [], []
        merged_models, old_models = {}, {}
        self.markers = []
        self.records = []
        self.rows = {}
        for model, floor in FLOORS.items():
            all_rows = []
            for number, (task, value) in enumerate(zip(self.tasks, (.1, .9, .8))):
                row = task | {name: value+i/100 for i, name in enumerate(METRICS)}
                row['pred_gt_ratio'] = 2+value
                fallback = number % 2 == 0
                row.update(label_id=task['local_class_id'], mask_threshold_used=.6 if fallback else floor,
                           dynamic_threshold_source='fallback_missing_prompt_logits' if fallback else 'prompt_logits')
                all_rows.append(row)
            self.rows[model] = all_rows
            for case in ('a', 'b'):
                selected = [t for t in self.tasks if t['case_id'] == case]
                job = self.job('fresh_'+model+'_'+case, model, selected, False)
                jobs.append(job)
                for task in selected:
                    self.commit(job, task, next(r for r in all_rows if r['case_id']==case and r['global_class_id']==task['global_class_id']))
            job = self.job('control_'+model, model, self.tasks[:1], True)
            controls.append(job)
            self.commit(job, self.tasks[0], all_rows[0])
            means = {name: .6+i/100 for i, name in enumerate(METRICS)}
            means['pred_gt_ratio'] = 2.6
            deltas = {name: .05 for name in METRICS}; deltas['pred_gt_ratio'] = .5
            nn_deltas = {name: .1 for name in METRICS}; nn_deltas['pred_gt_ratio'] = 1.
            merged_models[model] = dict(task_count=3, floor=floor, mean=means, class_macro_dice=.65,
                fallback_count=2, actual_frame_batches={'1':0,'2':3}, per_case_class=list(reversed(all_rows)),
                delta_from_old_same_floor=deltas, delta_from_nninteractive=nn_deltas)
            old_rows = [r | {n:r[n]-(.5 if n=='pred_gt_ratio' else .05) for n in METRICS} for r in reversed(all_rows)]
            old_models[model] = dict(floors={format(floor,'.2f'):dict(per_case_class=old_rows),
                                           '.60':dict(per_case_class=[])})
        dump(self.out/'jobs.json', dict(cases=jobs, controls=controls))
        dump(self.old, dict(models=old_models))
        nn_rows = [r | {n:r[n]-(1. if n=='pred_gt_ratio' else .1) for n in METRICS} for r in self.rows['v9_2']]
        dump(self.nn, dict(per_case_class=[nn_rows[1], nn_rows[2], nn_rows[0]]))
        dump(self.out/'merged_metrics.json', dict(models=merged_models, rows=6, predictions=6,
            task_count_per_model=3, controls_in_final=False, protocol_sha=digest(self.out/'protocol.json'),
            nninteractive_reference_sha256=digest(self.nn)))

    def job(self, name, model, tasks, control):
        return dict(name=name, model=model, case_id=tasks[0]['case_id'], control=control,
                    pending=[dict(task=t, floors=[FLOORS[model]]) for t in tasks], task_count=len(tasks),
                    prepared=str(self.out/'prepared'/name), source='bone', variant='baseline',
                    frame_batch=2, minimum_free_mib=30720, parent_history=[])

    def commit(self, job, task, row):
        run = self.out/'runs'/(job['name']+'_a0')
        spec_path = self.out/'job_specs'/(run.name+'.json')
        spec = job | dict(attempt=0, oom_retries=0, gpu=2, gpu_uuid='GPU-fixture', run=str(run),
                          spec=str(spec_path), log=str(self.out/'logs'/(run.name+'.log')))
        dump(spec_path, spec)
        restoration = dict(guards_passed=True, model=job['model'])
        dump(run/'restoration_start.json', restoration)
        resource = {k:spec[k] for k in ('gpu','gpu_uuid','frame_batch','minimum_free_mib','parent_history')}
        dump(run/'resource_policy_audit.json', resource | dict(actual_start_free_mib=60000,new_oom_retries=0))
        dump(run/'prepared/manifest.json', dict(tasks=[x['task'] for x in spec['pending']], expected_task_count=job['task_count']))
        dump(run/'prepared/refs.json', dict(per_case_class=[x['task'] for x in spec['pending']]))
        dump(run/'prepared_input_audit.json', dict(canonical_prepared=job['prepared'],
             files={str(run/'prepared'/n):digest(run/'prepared'/n) for n in ('manifest.json','refs.json')}))
        path = run/'classes'/(task['case_id']+'__'+str(task['global_class_id'])+'.json')
        record = dict(model=job['model'], task=task, restoration=restoration,
            semantic_execution={'control':job['control']}, results=[dict(floor=FLOORS[job['model']], row=row,
                stages=[dict(key=['bone',task['case_id'],task['global_class_id']], floor=FLOORS[job['model']],
                             threshold=row['mask_threshold_used'], source=row['dynamic_threshold_source'])])])
        dump(path, record)
        marker = path.with_suffix('.done.json')
        files = [path,run/'restoration_start.json',run/'resource_policy_audit.json',spec_path,
                 run/'prepared/manifest.json',run/'prepared/refs.json',run/'prepared_input_audit.json',self.out/'protocol.json']
        dump(marker, dict(job_name=job['name'], record=str(path), protocol_sha=digest(self.out/'protocol.json'),
                          files={str(p):digest(p) for p in files}))
        self.markers.append(marker); self.records.append(path)

    def change_record(self, index, mutate):
        path = self.records[index]; row = load(path); mutate(row); dump(path,row)
        marker = self.markers[index]; info = load(marker); info['files'][str(path)] = digest(path); dump(marker,info)


class ResultsAuditTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('tools.audit_bone_training_semantics_results'),
                             'independent standard-library auditor is missing')
        self.m = importlib.import_module('tools.audit_bone_training_semantics_results')
        self.f = Fixture()
        for name, value in [('EXPECTED_SCOPE',(2,2,3)),('OLD_SHA',digest(self.f.old)),('NN_SHA',digest(self.f.nn))]:
            handle=patch.object(self.m,name,value);handle.start();self.addCleanup(handle.stop)

    def audit(self):
        return self.m.audit_results(self.f.out,self.f.old,self.f.nn)

    def test_recomputes_rows_task_means_macro_fallback_and_key_paired_ten_deltas(self):
        report=self.audit()
        self.assertEqual(report['fresh_rows'],6);self.assertEqual(report['excluded_control_records'],2)
        for model in FLOORS:
            summary=report['models'][model]
            self.assertAlmostEqual(summary['mean']['dice'],.6)
            self.assertAlmostEqual(summary['class_macro_dice'],.65)
            self.assertEqual(summary['fallback_count'],2)
            for name in METRICS:
                self.assertAlmostEqual(summary['delta_from_old_same_floor'][name],.5 if name=='pred_gt_ratio' else .05)
                self.assertAlmostEqual(summary['delta_from_nninteractive'][name],1 if name=='pred_gt_ratio' else .1)
        self.assertEqual(len(report['class_markers']),8)

    def test_production_scope_cannot_accept_a_small_fixture(self):
        with patch.object(self.m,'EXPECTED_SCOPE',(20,47,770)),self.assertRaises(ValueError):self.audit()

    def test_incomplete_or_duplicate_task_inventory_is_rejected(self):
        path=self.f.out/'manifest.json';original=load(path);value=copy.deepcopy(original);value['tasks'].pop();dump(path,value)
        with self.assertRaises(ValueError):self.audit()
        value=copy.deepcopy(original);value['tasks'][-1]=copy.deepcopy(value['tasks'][0]);dump(path,value)
        with self.assertRaises(ValueError):self.audit()

    def test_missing_fresh_marker_or_extra_duplicate_marker_is_rejected(self):
        marker=self.f.markers[0];saved=marker.read_text();held=marker.with_name('held.marker')
        marker.rename(held)
        with self.assertRaises(ValueError):self.audit()
        held.rename(marker);marker.write_text('{}')
        with self.assertRaises(ValueError):self.audit()
        marker.write_text(saved)
        duplicate=marker.with_name('unexpected.done.json');duplicate.write_text(saved)
        with self.assertRaises(ValueError):self.audit()

    def test_control_leakage_or_missing_control_cannot_pass(self):
        self.f.change_record(0,lambda r:r['semantic_execution'].update(control=True))
        with self.assertRaises(ValueError):self.audit()
        self.f.change_record(0,lambda r:r['semantic_execution'].update(control=False))
        controls=load(self.f.out/'jobs.json');controls['controls'].pop();dump(self.f.out/'jobs.json',controls)
        with self.assertRaises(ValueError):self.audit()

    def test_final_merged_rows_cannot_smuggle_controls_or_hide_class_rows(self):
        path=self.f.out/'merged_metrics.json';merged=load(path)
        merged['models']['v9_2']['per_case_class'].append(copy.deepcopy(merged['models']['v9_2']['per_case_class'][0]));dump(path,merged)
        with self.assertRaises(ValueError):self.audit()

    def test_wrong_label_identity_and_wrong_spec_floor_are_rejected(self):
        self.f.change_record(0,lambda r:r['results'][0]['row'].update(label_id=2))
        with self.assertRaises(ValueError):self.audit()
        self.f.change_record(0,lambda r:r['results'][0]['row'].update(label_id=1))
        self.f.change_record(0,lambda r:r['results'][0].update(floor=.75))
        with self.assertRaises(ValueError):self.audit()

    def test_numeric_equality_cannot_hide_a_changed_label_type(self):
        self.f.change_record(0,lambda r:r['results'][0]['row'].update(local_class_id=1.0))
        with self.assertRaises(ValueError):self.audit()

    def test_resigned_worker_spec_cannot_change_canonical_pending_tasks(self):
        marker=self.f.markers[0];info=load(marker)
        path=next(Path(p) for p in info['files'] if Path(p).parent.name=='job_specs')
        spec=load(path);spec['pending'][0]['floors']=[.75];dump(path,spec)
        for marker in self.f.markers:
            info=load(marker)
            if str(path) in info['files']:
                info['files'][str(path)]=digest(path);dump(marker,info)
        with self.assertRaises(ValueError):self.audit()

    def test_nonfinite_boolean_unit_bounds_and_negative_ratio_are_rejected(self):
        original=load(self.f.records[0])['results'][0]['row']
        for name,value in [('dice',float('nan')),('iou',1.01),('precision',True),('recall',-.01),('pred_gt_ratio',-1.),('pred_gt_ratio',float('inf'))]:
            self.f.change_record(0,lambda r:r['results'][0]['row'].update(original))
            self.f.change_record(0,lambda r:r['results'][0]['row'].update({name:value}))
            with self.subTest(name=name,value=value),self.assertRaises(ValueError):self.audit()

    def test_final_fallback_comes_from_last_native_stage(self):
        self.f.change_record(0,lambda r:r['results'][0]['stages'][-1].update(source='prompt_logits'))
        with self.assertRaises(ValueError):self.audit()

    def test_each_of_eight_referenced_files_must_have_current_sha(self):
        for path in load(self.f.markers[0])['files']:
            path=Path(path);original=path.read_bytes();path.write_bytes(original+b' ')
            with self.subTest(path=path),self.assertRaises(ValueError):self.audit()
            path.write_bytes(original)
        info=load(self.f.markers[0]);info['files'].pop(next(iter(info['files'])));dump(self.f.markers[0],info)
        with self.assertRaises(ValueError):self.audit()

    def test_reference_hash_and_exact_paired_key_sets_are_required(self):
        original=self.f.nn.read_bytes();self.f.nn.write_bytes(original+b' ')
        with self.assertRaises(ValueError):self.audit()
        self.f.nn.write_bytes(original)
        value=load(self.f.nn);value['per_case_class'][0]['case_id']='another';dump(self.f.nn,value)
        with patch.object(self.m,'NN_SHA',digest(self.f.nn)),self.assertRaises(ValueError):self.audit()

    def test_merged_mean_macro_deltas_and_actual_rows_are_independently_checked(self):
        path=self.f.out/'merged_metrics.json';original=load(path)
        for mutate in [lambda s:s['mean'].update(dice=.65),lambda s:s.update(class_macro_dice=.6),
                       lambda s:s.update(fallback_count=1),lambda s:s['delta_from_old_same_floor'].update(iou=0),
                       lambda s:s['delta_from_nninteractive'].update(recall=0),
                       lambda s:s['per_case_class'][0].update(dice=.7)]:
            value=copy.deepcopy(original);mutate(value['models']['v9_2']);dump(path,value)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):self.audit()

    def test_cli_requires_drained_state_and_creates_one_output_without_overwriting(self):
        output=self.f.out/'audit.json';args=['--out',str(self.f.out),'--old-reference',str(self.f.old),
            '--nn-reference',str(self.f.nn),'--audit-output',str(output)]
        dump(self.f.out/'status.json',dict(phase='full',pending=0,running={'2':{'pid':7}},failed=[]))
        with self.assertRaises(ValueError):self.m.main(args)
        self.assertFalse(output.exists())
        dump(self.f.out/'status.json',dict(phase='full',pending=0,running={},failed=[]))
        self.m.main(args);saved=output.read_bytes()
        with self.assertRaises(FileExistsError):self.m.main(args)
        self.assertEqual(output.read_bytes(),saved)

    def test_import_uses_only_standard_library(self):
        module=Path(self.m.__file__)
        script="""import builtins,runpy,sys
native=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in ('torch','numpy','model','v10','tools'):
        raise AssertionError('non-standard-library import: '+name)
    return native(name,*args,**kwargs)
builtins.__import__=guarded
runpy.run_path(sys.argv[1],run_name='independent_audit_import_probe')
"""
        result=subprocess.run([sys.executable,'-B','-X','utf8','-c',script,str(module)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
