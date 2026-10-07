import importlib.util
from pathlib import Path
import numpy as np
import pytest

def api():
    p=Path(__file__).resolve().parents[1]/'infer/native_segvol_prompts.py'
    assert p.is_file(),'SegVol physical prompt module missing'
    spec=importlib.util.spec_from_file_location('g',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_geometry_roundtrip_and_spacing():
    m=api();bounds=((4,44),(10,170),(20,100));spacing=(2.,.7,.8)
    points=np.array([[4,10,20],[43,169,99],[20,80,40]],float)
    for mode in ('physical','stretch'):
        g=m.geometry(bounds,spacing,mode);q=m.forward_points(points,g)
        np.testing.assert_allclose(m.inverse_points(q,g),points,atol=1e-10)
        assert (q>=-.5).all() and (q<np.array(g['target'])-.5+1e-8).all()
    g=m.geometry(bounds,spacing,'physical');assert np.ptp(g['pitch'])==0
    assert np.ptp(m.geometry(bounds,spacing,'stretch')['pitch'])>0

def test_array_and_native_encoder_point_axes():
    m=api();volume=np.arange(3*5*7).reshape(3,5,7);p=np.array([[1,2,3]],float)
    image,q=m.native_view(volume,p,'official')
    assert image.shape==(3,7,5) and image[1,3,2]==volume[1,2,3]
    np.testing.assert_array_equal(q,[[1,3,2]])
    _,q=m.native_view(volume,p,'pe_aligned');np.testing.assert_array_equal(q,[[3,1,2]])
    np.testing.assert_array_equal(m.canonical_view(image),volume)

def test_geometry_rejects_invalid_spacing():
    m=api()
    with pytest.raises(ValueError):m.geometry(((0,2),)*3,(1,0,1),'physical')
    with pytest.raises(ValueError):m.geometry(((0,2),)*3,(1,1,1),'unknown')
