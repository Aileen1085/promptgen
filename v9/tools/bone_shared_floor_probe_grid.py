"""Independent checked-decoder binding recovery; preserve frozen predecessors."""
import ast
import os
import sys
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.bone_shared_floor_restore_grid import engine as prior_engine, recovery_source


def engine():
    value = prior_engine()
    prior = value.ROOT/value.OUT
    value.OUT += '_probebinding'
    value.__file__ = __file__
    value.SOURCES += ('tools/bone_shared_floor_probe_runtime.py', 'tools/bone_shared_floor_probe_grid.py')
    source = (value.ROOT/'tools/bone_shared_floor_grid.py').read_bytes()
    if hashlib.sha256(source).hexdigest() != 'b1cedcb405e26df9c07f451d1eac01c9cccd2dd53f5f87fb3fa70c674eca12d2':
        raise ValueError('frozen shared runner changed')
    tree = ast.parse(recovery_source(source.decode('utf-8')))
    run = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    imports = [n for n in ast.walk(run) if isinstance(n,ast.ImportFrom) and n.module=='tools.bone_shared_floor_restore_runtime']
    if len(imports)!=1:raise ValueError('checked runtime import anchor changed')
    imports[0].module = 'tools.bone_shared_floor_probe_runtime'
    exec(compile(ast.fix_missing_locations(ast.Module(body=[run],type_ignores=[])),__file__,'exec'),value.__dict__)
    def related():
        found=[]
        names=('bone_floor_memory_wait_recovery.py','bone_shared_floor_grid.py',
               'bone_shared_floor_restore_grid.py','bone_shared_floor_probe_grid.py')
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
            try:args=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
            except OSError:continue
            if any(Path(x).name in names for x in args) and any(x in ('worker','wait-launch','launch','controls') for x in args):found.append(proc.name)
        return found
    value.related=related
    native_signature=value.signature
    def signature(out):
        prior_engine().verify(prior)
        result=native_signature(out)
        result['prior_restore_control_files']={str(f):value.sha(f) for f in sorted(prior.rglob('*')) if f.is_file()}
        return result
    value.signature=signature
    return value


if __name__=='__main__':engine().main()
