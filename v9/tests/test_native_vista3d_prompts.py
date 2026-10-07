import importlib.util
from pathlib import Path
import numpy as np

def api():
    p=Path(__file__).resolve().parents[1]/'infer/native_vista3d_prompts.py'
    assert p.is_file(),'native VISTA physical geometry missing'
    spec=importlib.util.spec_from_file_location('vista_geometry',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_physical_geometry_roundtrip_and_fixed_spacing():
    m=api();b=((3,95),(10,220),(8,130));s=(2.5,.8,.9)
    p=np.array([[3,10,8],[94,219,129],[50,88,70]],float)
    for mode in ('fixed','fit'):
        g=m.geometry(b,s,mode);q=m.forward_points(p,g)
        np.testing.assert_allclose(m.inverse_points(q,g),p,atol=1e-10)
        assert np.ptp(g['pitch'])==0
    assert m.geometry(b,s,'fixed')['pitch']==[1.5]*3

def test_native_xyz_axes_roundtrip():
    m=api();x=np.arange(3*4*5).reshape(3,4,5);p=np.array([[1,2,3]])
    v,q=m.native_view(x,p)
    assert v.shape==(4,5,3) and v[2,3,1]==x[1,2,3]
    np.testing.assert_array_equal(q,[[2,3,1]])
    np.testing.assert_array_equal(m.canonical_view(v),x)

def test_windows_deduplicate_cover_points_and_pad_small_grid():
    m=api();points=np.array([[0,0,0],[0,0,0],[190,145,110]],float)
    w,shape,pad=m.windows(points,(191,146,111),128)
    assert len(w)==len(set(w))==2 and shape[2]>=128
    shifted=points+pad
    for p in shifted:
        assert any(all(a<=c<b for c,(a,b) in zip(p,v)) for v in w)

def test_fusion_counts_overlaps_and_uncovered_is_negative():
    m=api();w=[((0,2),(0,2),(0,2)),((1,3),(0,2),(0,2))]
    r=m.fuse([np.full((2,2,2),2.),np.full((2,2,2),-4.)],w,(4,2,2))
    np.testing.assert_array_equal(r[:,0,0],[2.,-1.,-4.,-9999.])
