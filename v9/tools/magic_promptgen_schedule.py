"""Environment-only launcher for the signed MAGIC PromptGen workers."""
import argparse
from pathlib import Path
import os
import subprocess
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
ENTRY=ROOT/'tools/magic_promptgen_three_backbones.py'
PYTHONS=dict(sammed3d='../conda-envs/al/bin/python',
             vista3d='vista3d/.venv-vista/bin/python',segvol='segvol-3d/.venv/bin/python')


def worker_command(command, model):
    result=list(command)
    if len(result)>2 and Path(result[1]).resolve()==ENTRY.resolve() and result[2]=='worker':
        result[0]=str(ROOT/PYTHONS[model]) if model in PYTHONS else ''
        if not result[0]:raise ValueError('unknown backbone environment')
    return result


def subprocess_proxy(spawn):
    return SimpleNamespace(Popen=spawn,check_output=subprocess.check_output,STDOUT=subprocess.STDOUT)


def main():
    sys.path.insert(0,str(ROOT))
    from tools import magic_promptgen_three_backbones as e
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default=e.OUT);p.add_argument('--gpus',default='0,2,3,4,5,6')
    a=p.parse_args();os.chdir(ROOT);out=ROOT/a.out;e.verify(out)
    for model in e.ASSETS:
        if not (out/'jobs'/(model+'_smoke')/'complete_metrics.json').is_file():
            raise ValueError('all three signed real-task smoke checks required')
    environments={}
    for model,relative in PYTHONS.items():
        python=ROOT/relative
        info=subprocess.check_output([str(python),'-c',
            'import sys,torch;print(sys.executable,sys.version,torch.__version__)'],text=True)
        environments[model]=dict(python=str(python),version=info)
    e.base.write(out/'environment_launcher_audit.json',dict(launcher_sha256=e.sha(__file__),
        worker_sha256=e.sha(ENTRY),environments=environments,protocol_unchanged=True))
    original=subprocess.Popen
    def spawn(command,**kwargs):
        job=e.base.read(command[command.index('--job')+1])
        return original(worker_command(command,job['model']),**kwargs)
    # Isolate only this base module's subprocess reference; GPU queries retain
    # their real check_output and cannot be misinterpreted as model workers.
    e.base.subprocess=subprocess_proxy(spawn)
    e.base.__file__=str(ENTRY);e.base.merge=e.merge
    e.base.launch(out,list(map(int,a.gpus.split(','))))


if __name__=='__main__':main()
