import importlib.util
from pathlib import Path
import pytest

def api():
    path=Path(__file__).resolve().parents[1]/'tools/native_sammed3d_ct13.py'
    assert path.is_file(), 'full Turbo entry missing'
    spec=importlib.util.spec_from_file_location('turbo_full',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def tasks():
    return [dict(source_name=s,case_id=c,global_class_id=i) for s in ('totalseg','amos')
            for c in ('a','b','c') for i in range(3)]

def test_shards_cover_unique_tasks_without_splitting_cases():
    m=api();rows=tasks();jobs=m.make_jobs(rows)
    assert sum(len(j['tasks']) for j in jobs)==18
    assert len({m.native.key(t) for j in jobs for t in j['tasks']})==18
    for s in ('totalseg','amos'):
        for c in ('a','b','c'):
            assert sum(any(t['source_name']==s and t['case_id']==c for t in j['tasks']) for j in jobs)==1

def test_scope_rejects_training_overlap_and_duplicate():
    m=api();rows=tasks();split={'datasets':{s:{'val':[{'case_id':c} for c in ('a','b','c')],'train':[]} for s in ('totalseg','amos')}}
    m.validate_scope(rows,split,{'totalseg':9,'amos':9})
    with pytest.raises(ValueError):m.validate_scope(rows+[rows[0]],split,{'totalseg':10,'amos':9})
    split['datasets']['amos']['train']=[{'case_id':'a'}]
    with pytest.raises(ValueError):m.validate_scope(rows,split,{'totalseg':9,'amos':9})

def good_row():
    return {'prompt_audit':{'background_cache_hit':True,'fg_in_gt_fraction':1.,'bg_in_gt_count':0},
            'stages':[{'native_tensor_axis_order':'canonical_XYZ','roundtrip_voxels':0.,
                       'foreground_points':8,'background_points':8,'gt_used_for_roi':False,'decoder_calls':1}]}

def test_audit_rejects_wrong_axis_or_excessive_points():
    m=api();r=good_row();m.validate_audit(r)
    r['stages'][0]['native_tensor_axis_order']='ZXY'
    with pytest.raises(ValueError):m.validate_audit(r)
    r=good_row();r['stages'][0]['background_points']=9
    with pytest.raises(ValueError):m.validate_audit(r)

def test_gpu_selection_uses_free_space_and_uuid():
    m=api();inventory='0, GPU-a, 80000\n1, GPU-b, 80000\n2, GPU-c, 12000\n3, GPU-d, 40000'
    assert m.choose_gpus(inventory,{0,2,3},{'GPU-a'})==[(3,'GPU-d')]
