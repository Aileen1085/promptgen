import importlib.util
from pathlib import Path
import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / 'infer/native_sammed3d_prompts.py'

def api():
    assert PATH.is_file(), '3D prompt geometry implementation is missing'
    spec = importlib.util.spec_from_file_location('native_sammed3d_prompts', PATH)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

@pytest.mark.parametrize('method', ['fps', 'axis'])
def test_samples_only_existing_scribble_deterministically(method):
    m = api(); p = np.c_[np.arange(30), np.full(30, 4), np.full(30, 8)]
    q = m.sample_points(p, [2., 1., 1.], 4, 3., method)
    assert len(q) == 4 and np.array_equal(q, m.sample_points(p, [2., 1., 1.], 4, 3., method))
    assert all(tuple(x) in set(map(tuple,p)) for x in q)
    d = np.linalg.norm((q[:,None]-q[None,:])*[2.,1.,1.],axis=-1)
    assert d[np.triu_indices(4,1)].min() >= 3.

def test_fps_uses_real_spacing_not_voxel_distance():
    m=api();p=np.array([[0,0,0],[0,10,0],[3,0,0]])
    q=m.sample_points(p,[10.,1.,1.],2,0.,'fps')
    assert any(np.array_equal(x,[3,0,0]) for x in q)

def test_plane_budgets_preserve_both_orthogonal_planes():
    m=api(); fg=np.zeros((20,12,13),bool)
    fg[2:18,4,2:10]=True;fg[2:18,2:10,6]=True
    bg=np.array([[z,0,6] for z in range(20)]+[[z,4,0] for z in range(20)])
    p,l=m.scribble_points(fg,bg,[2.,1.,1.],(6,4),2,'fps')
    assert 1 in l and 0 in l and len(p)<=8
    assert np.all(fg[tuple(p[l==1].astype(int).T)])
    assert all(tuple(x) in set(map(tuple,bg)) for x in p[l==0])

def test_physical_cube_center_roundtrip():
    m=api();g=m.cube_geometry(((10,30),(5,35),(2,42)),[3.,2.,1.],128,'physical')
    points=np.array([[10.,5.,2.],[29.,34.,41.],[18.,20.,25.]])
    q=m.to_cube(points,g);back=m.from_cube(q,g)
    assert np.allclose(points,back,atol=1e-10)
    delta=(q[1]-q[0])/((points[1]-points[0])*[3.,2.,1.])
    assert np.allclose(delta,delta[0]) and np.all(q>=-.5) and np.all(q<127.5)

def test_stretch_is_explicit_and_center_aligned():
    m=api();g=m.cube_geometry(((0,20),(0,40),(0,60)),[1.,1.,1.],128,'stretch')
    assert np.allclose(m.to_cube([[0,0,0]],g),[[2.7,1.1,.5666666666666667]])

def test_invalid_geometry_rejected():
    m=api()
    with pytest.raises(ValueError):m.cube_geometry(((0,20),)*3,[1.,0.,1.],128,'physical')
    with pytest.raises(ValueError):m.sample_points([[0,0,0]],[1.,1.,1.],0,3.,'fps')
