"""Additive fresh controller for the CT13 bootstrap import-order repair.

The signed predecessor supplies the complete scheduling, inference collection,
controls, and arithmetic in one private globals namespace. Only the entry path,
isolated output, runtime lookup, and bootstrap evidence guards change.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import types


ROOT=Path(__file__).resolve().parents[1]
NATIVE_CONTROLLER_SHA256='59963323d0a8cdc86537f1063b04b225966380889218257b577fa16f0affdcc5'
RESTORE_RUNTIME_SHA256='ad1c19d82449c84186160b876adfba14ea40f0faf41fc491a5a2f637dce929e3'
BOOTSTRAP_SHA256='b9ddcea34e89caff2a3184baef6a09b332588454e81bf7f74ef0c33dac6b4dbf'
BOOTSTRAP_EVIDENCE='output/amos_prompt_ablation_bootstrap_failure_20261030.json'
OLD_OUTPUT='output/amos_v102_e490_floor070_prompt_ablations_20261030'
FAILURE_FILES=(OLD_OUTPUT+'/protocol.json',OLD_OUTPUT+'/protocol_signature.json',
               OLD_OUTPUT+'/status.json',OLD_OUTPUT+'/job_specs/amos_native_control_a0.json',
               OLD_OUTPUT+'/logs/amos_native_control_a0.log','output/amos_prompt_ablations_pipeline_20261030.log')
OUT='output/amos_v102_e490_floor070_prompt_ablations_20261030_restorefix'
TMUX='amos_v102_floor070_prompt_ablations_restorefix'
PIPELINE='output/amos_prompt_ablations_restorefix_pipeline_20261030.log'
_ENGINE=None


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def _source_guard(root):
    root=Path(root)
    if _sha(root/'tools/amos_prompt_ablation_eval.py')!=NATIVE_CONTROLLER_SHA256:
        raise ValueError('signed predecessor controller SHA mismatch')
    if _sha(root/'tools/amos_prompt_ablation_restore_runtime.py')!=RESTORE_RUNTIME_SHA256:
        raise ValueError('signed restored runtime SHA mismatch')


def verify_bootstrap(root=None):
    """Authenticate the failed, fully drained bootstrap and its six retained files."""
    root=Path(root or ROOT);path=root/BOOTSTRAP_EVIDENCE
    if _sha(path)!=BOOTSTRAP_SHA256:raise ValueError('immutable bootstrap failure evidence SHA changed')
    evidence=json.loads(path.read_text(encoding='utf-8'))
    if (evidence.get('model_loaded') is not False or evidence.get('controls_passed') is not False
            or evidence.get('full_started') is not False or evidence.get('committed_classes')!=0
            or evidence.get('related_after_independent_second_check')!=[]):
        raise ValueError('bootstrap evidence does not prove an empty pre-model failure')
    files=evidence.get('files',{})
    if set(files)!=set(FAILURE_FILES):raise ValueError('bootstrap failure file inventory changed')
    for name,expected in files.items():
        if _sha(root/name)!=expected:raise ValueError('retained bootstrap failure file changed: '+name)
    queue=evidence.get('queue',{});failed=queue.get('failed',[])
    if (queue.get('phase')!='controls' or queue.get('pending')!=0 or queue.get('running')!={}
            or queue.get('committed_classes')!=0 or len(failed)!=1
            or failed[0].get('name')!='amos_native_control' or failed[0].get('returncode')!=1
            or Path(failed[0].get('log','')).resolve()!=(root/FAILURE_FILES[4]).resolve()
            or json.loads((root/FAILURE_FILES[2]).read_text(encoding='utf-8'))!=queue):
        raise ValueError('bootstrap failure queue is not the signed naturally drained control failure')
    if any((root/OLD_OUTPUT/'runs').glob('*/classes/*.done.json')) or any((root/OLD_OUTPUT/'case_success').glob('*.json')):
        raise ValueError('bootstrap failure unexpectedly contains committed predictions')
    return evidence


def related(proc_root=None):
    """Read process state for both entries; never signal or restart a process."""
    proc_root=Path('/proc') if proc_root is None else Path(proc_root)
    if not proc_root.exists():return []
    entries={'amos_prompt_ablation_eval.py','amos_prompt_ablation_restore_eval.py'};result=[]
    for folder in proc_root.iterdir():
        if not folder.name.isdigit() or int(folder.name)==os.getpid():continue
        try:args=(folder/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except OSError:continue
        if any(Path(arg).name in entries for arg in args) and any(arg in ('prepare','run','worker') for arg in args):
            result.append(dict(pid=int(folder.name),args=args))
    return result


def _runtime():
    return importlib.import_module('tools.amos_prompt_ablation_restore_runtime')


def engine():
    """Execute the complete signed source exactly once; retain its original globals."""
    global _ENGINE
    if _ENGINE is not None:return _ENGINE
    _source_guard(ROOT)
    old_path=ROOT/'tools/amos_prompt_ablation_eval.py'
    private=types.ModuleType('_amos_prompt_ablation_restore_private')
    namespace=private.__dict__
    # The new __file__ is set before execution so all spawned workers use this
    # additive entry. The private __name__ prevents native main dispatch here.
    namespace['__file__']=str(Path(__file__).resolve())
    exec(compile(old_path.read_bytes(),str(old_path),'exec'),namespace)
    namespace.update(OUT=OUT,TMUX=TMUX,PIPELINE=PIPELINE,related=related,runtime=_runtime)
    namespace['SOURCE_SEEDS']=namespace['SOURCE_SEEDS']+(
        'tools/amos_prompt_ablation_restore_eval.py','tools/amos_prompt_ablation_restore_runtime.py',
        BOOTSTRAP_EVIDENCE,*FAILURE_FILES)
    original_prepare=namespace['prepare'];original_verify=namespace['verify'];original_pipeline=namespace['run_pipeline']
    def guard():
        verify_bootstrap(namespace['ROOT']);_source_guard(namespace['ROOT'])
    def prepare(*args,**kwargs):
        guard();return original_prepare(*args,**kwargs)
    def verify(*args,**kwargs):
        guard();return original_verify(*args,**kwargs)
    def run_pipeline(*args,**kwargs):
        guard();return original_pipeline(*args,**kwargs)
    namespace.update(prepare=prepare,verify=verify,run_pipeline=run_pipeline)
    _ENGINE=namespace
    return namespace


def __getattr__(name):
    if name.startswith('__'):raise AttributeError(name)
    namespace=engine()
    if name not in namespace:raise AttributeError(name)
    return namespace[name]


def main():
    # --help parses without operational server artifacts. Every actual mode
    # reaches guarded prepare/verify/run_pipeline before creating an output.
    return engine()['main']()


if __name__=='__main__':main()
