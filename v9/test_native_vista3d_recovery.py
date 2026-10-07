import importlib.util
from pathlib import Path
import numpy as np


def test_valid_first_voxel_cell_point_is_encoded():
    path=Path(__file__).resolve().parents[1]/'tools/native_vista3d_recovery.py'
    assert path.is_file(), 'half-voxel recovery entry missing'
    spec=importlib.util.spec_from_file_location('vr',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    points=np.array([[47.377450980392155,47.02696078431372,-.08169934640522876],
                     [67.,78.,92.],[0,0,-.51],[0,0,127.49],[0,0,127.5]])
    np.testing.assert_array_equal(m.cell_inside(points,np.zeros(3),128),[True,True,False,True,False])
