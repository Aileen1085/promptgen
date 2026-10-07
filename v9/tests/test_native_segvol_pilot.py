import importlib.util
from pathlib import Path
import numpy as np

def api():
    p=Path(__file__).resolve().parents[1]/'tools/native_segvol_pilot.py'
    assert p.is_file(),'SegVol pilot missing'
    spec=importlib.util.spec_from_file_location('segpilot',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_arms_have_fixed_nonlearned_protocol():
    m=api();assert len(m.ARMS)==5
    assert m.ARMS['official_fps2']==(2,'fps','stretch','official')
    assert m.ARMS['physical_fps4'][2]=='physical'
    assert m.ARMS['official_fps4_pe_aligned'][3]=='pe_aligned'

def test_normalization_is_finite_and_preserves_shape():
    m=api();ct=np.arange(80,dtype=np.float32).reshape(4,4,5)-40
    a=m.normalize(ct);assert a.shape==ct.shape and np.isfinite(a).all()
    assert a.min()==0 and a.max()==1

def test_geometry_affine_restoration_has_no_axis_flip():
    m=api();ct=np.indices((6,7,9))[2].astype(np.float32)
    g=m.geometry(((0,6),(0,7),(0,9)),(1,1,1),'stretch')
    restored=m.restore(m.resample(ct,g),g)
    np.testing.assert_allclose(restored[1:-1,1:-1,1:-1],ct[1:-1,1:-1,1:-1],atol=1e-5)
