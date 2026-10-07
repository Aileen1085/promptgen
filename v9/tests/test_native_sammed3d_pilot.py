import importlib.util
from pathlib import Path
import numpy as np

PATH=Path(__file__).resolve().parents[1]/'tools/native_sammed3d_pilot.py'
def api():
    assert PATH.is_file(), 'Turbo pilot entry is missing'
    spec=importlib.util.spec_from_file_location('native_sammed3d_pilot',PATH)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_constant_image_roundtrip_preserves_center():
    m=api();bounds=((0,12),(0,16),(0,20))
    g=m.cube_geometry(bounds,[1.,1.,1.],128,'physical')
    cube=m.resample_cube(np.ones((12,16,20)),bounds,g)
    assert cube[64,64,64]==1.
    local=m.restore_logits(np.ones((128,128,128)),g)
    assert local.shape==(12,16,20) and np.all(local==1.)

def test_resample_and_points_agree_on_anisotropic_center():
    m=api();ct=np.zeros((11,15,21));ct[5,7,10]=1
    g=m.cube_geometry(((0,11),(0,15),(0,21)),[3.,2.,1.],128,'physical')
    cube=m.resample_cube(ct,((0,11),(0,15),(0,21)),g)
    p=m.to_cube([[5,7,10]],g)[0]
    peak=np.array(np.unravel_index(cube.argmax(),cube.shape))
    assert np.linalg.norm(peak-p)<1.8

def test_pilot_selection_two_cases_six_categories_only():
    m=api();cats=('liver','spleen','kidney_left','pancreas','aorta','adrenal_gland_left')
    tasks=[dict(source_name='amos',case_id=c,global_class_id=i,class_name=n)
           for c in ('amos_0001','amos_0002','amos_0003') for i,n in enumerate(cats)]
    selected=m.select_tasks(tasks)
    assert len(selected)==12 and set(t['case_id'] for t in selected)=={'amos_0001','amos_0002'}

def test_missing_category_does_not_silently_change_scope():
    import pytest
    m=api()
    with pytest.raises(ValueError):m.select_tasks([dict(source_name='amos',case_id='a',class_name='liver')])

def test_amos_official_space_names_supported_without_changing_class_ids():
    m=api();cats=('liver','spleen','left kidney','pancreas','aorta','left adrenal gland')
    tasks=[dict(source_name='amos',case_id=c,global_class_id=i,class_name=n)
           for c in ('amos_0008','amos_0013') for i,n in enumerate(cats)]
    selected=m.select_tasks(tasks)
    assert len(selected)==12 and selected==sorted(tasks,key=lambda t:(t['case_id'],t['class_name']))
