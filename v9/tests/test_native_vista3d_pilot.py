import importlib.util
from pathlib import Path
import numpy as np

def api():
    p=Path(__file__).resolve().parents[1]/'tools/native_vista3d_pilot.py'
    assert p.is_file(),'VISTA isolated pilot missing'
    spec=importlib.util.spec_from_file_location('vp',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_native_only_protocol_and_shared_scope():
    m=api()
    assert set(m.ARMS)=={'fixed_fps2','fixed_fps4','fixed_axis4','fit_fps4'}
    assert m.WEIGHT_SHA=='889042ab37dbb9f9b2467e4a91654fb3b56482b74b98d315fc89026ab1570af8'
    assert m.PROTOCOL['semantic'] is False and m.PROTOCOL['promptgen'] is False
    assert m.PROTOCOL['previous_mask'] is None and m.PROTOCOL['threshold']==.5

def test_constant_resample_restore_has_no_axis_drift():
    m=api();x=np.full((30,40,50),.75,np.float32);b=((2,25),(3,35),(5,40))
    for mode in ('fixed','fit'):
        g=m.geometry(b,(2.,.8,.9),mode);y=m.resample(x,g)
        z=m.restore(y,g)
        np.testing.assert_allclose(z,.75,atol=1e-6)
        assert z.shape==(23,32,35)
