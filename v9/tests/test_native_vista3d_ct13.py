import importlib.util
from pathlib import Path
import pytest

def api():
    p=Path(__file__).resolve().parents[1]/'tools/native_vista3d_ct13.py'
    assert p.is_file(),'VISTA full entry missing'
    spec=importlib.util.spec_from_file_location('vf',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def row():
    return {'prompt_audit':{'background_cache_hit':True,'fg_in_gt_fraction':1.,'bg_in_gt_count':0},
            'stages':[{'point_axis_order':'canonical_XYZ','roundtrip_voxels':0.,
                'foreground_points':8,'background_points':8,'gt_used_for_roi':False,
                'encoded_all_selected_points':True,'point_labels':[1]*8+[0]*8,
                'patches':[{'encoded_point_indices':list(range(16)),'forward_calls':1,
                    'class_vector':None,'prompt_class':None,'previous_mask':None,'gt_used':False}]}]}

def test_best_pilot_protocol_and_audit():
    m=api();assert m.PROTOCOL['arm']=='fit_fps4' and m.PROTOCOL['threshold']==.5
    assert not m.PROTOCOL['semantic'] and not m.PROTOCOL['promptgen']
    m.validate_audit(row())
    bad=row();bad['stages'][0]['patches'][0]['previous_mask']='feedback'
    with pytest.raises(ValueError):m.validate_audit(bad)
    bad=row();bad['stages'][0]['patches'][0]['encoded_point_indices']=[0]
    with pytest.raises(ValueError):m.validate_audit(bad)

def test_scope_partition_and_gpu_safety():
    m=api();tasks=[{'source_name':'amos','case_id':c,'global_class_id':i} for c in ['a','b','c'] for i in range(3)]
    jobs=m.make_jobs(tasks);seen={}
    for j in jobs:
        for t in j['tasks']:
            seen.setdefault(t['case_id'],set()).add(j['name'])
    assert sum(len(j['tasks']) for j in jobs)==9 and all(len(v)==1 for v in seen.values())
    assert m.choose_gpus('1, blocked, 80000\n7, blocked7, 80000\n2, allowed, 30000',{1,2,7},set())==[(2,'allowed')]
