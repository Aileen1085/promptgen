from pathlib import Path
import importlib.util
import subprocess


def test_inventory_probe_uses_native_popen(monkeypatch):
    path=Path(__file__).resolve().parents[1]/'tools/native_vista3d_resume_schedule.py'
    assert path.is_file(),'isolated scheduler bootstrap missing'
    spec=importlib.util.spec_from_file_location('vs',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    custom=lambda *a,**k:None
    monkeypatch.setattr(subprocess,'Popen',custom)
    assert m.inventory_output(['cmd','/c','echo','probe'],text=True).strip()=='probe'
    assert subprocess.Popen is custom
