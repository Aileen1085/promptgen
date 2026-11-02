"""Fresh semantic-only controller after authenticated AMOS raw-data restoration.

The failed original run is immutable. This wrapper creates a separate private
controller and freezes restoration provenance, without changing its scientific
runtime, independent GPU control gate, queue, or atomic publication protocol.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ORIGINAL_ENTRY = 'tools/amos_semantic_only_eval.py'
ORIGINAL_ENTRY_SHA256 = 'd457210306b20ecabb0e48acd66188b571a3e95583c1fc859aeacab87b1f8add'
ORIGINAL_RUNTIME = 'tools/amos_semantic_only_runtime.py'
ORIGINAL_RUNTIME_SHA256 = '1061b03cbbc22a684cb6a896b0046c5bfe4ce3e5af61e85ccd808883c8f3661c'
OUT = 'output/amos_v102_e490_floor070_semantic_only_restored_20261102'
TMUX = 'amos_v102_floor070_semantic_only_restored_20261102'
PIPELINE = 'output/amos_semantic_only_restored_pipeline_20261102.log'
ORIGINAL_OUT = 'output/amos_v102_e490_floor070_semantic_only_20261102'
ORIGINAL_TMUX = 'amos_v102_floor070_semantic_only_20261102'
DATA_READY = 'output/amos22_official_download_20261102/DATA_READY.json'
EVIDENCE_SHA256 = {
    DATA_READY: '9b207d3ca87bd2393e71d3bb9eb0aefd47e189e6f3dfbb48b94e4f283cb51718',
    'output/amos22_official_download_20261102/restore_audit.json':
        'c0f6cea611e6d33f83e5e67493306b66fd3b83575d7a0b73bbc1c460fbf52044',
    'output/amos22_official_download_20261102/independent_root_data_review_1793621843849010630.json':
        'ef801085b064dcb9483906bfac0c8a0f16d837da143997f6353ae46734718ac9',
    'output/amos_semantic_only_pipeline_20261102.log':
        'e309e8c714311201022b36e7cceb4682674f32d3a5bf1bf35ddcf6ac3181cb56',
    'output/amos_semantic_prepare_cachepath_failure_20261102.json':
        '77c5326d889746d0a20dccfce9c1095ed797795d65912c484b704852c9c33f35',
    'output/amos_semantic_raw_data_availability_20261102.json':
        'babe1edbb0f79240e171db2f6e485f0c0501375a47b7065d55bf6c7b8375a69f',
}
READINESS_REQUIRED = dict(
    complete=True, archive_bytes=24234336519,
    archive_md5='67717b2a483ac0744c89c3016b7aaef7',
    archive_sha256='d2fbf2c31abba9824d183f05741ce187b17905b8cca64d1078eabf1ba96775c2',
    restored_files=600, train_cases=200, val_cases=100,
    historical_sha_and_mtime_verified_files=200, cache_path_identity_cases=100,
    existing_case_ct_caches=100, new_ct_cache_copies=0, model_launched=False,
)
RECOVERY_SOURCES = (
    'tools/amos_semantic_only_restored_eval.py',
    'tests/test_amos_semantic_only_restored_eval.py',
)
_CONTEXT = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_guard(*, production=False):
    """Authenticate the original code/profile and the exact restoration evidence."""
    for name, expected in ((ORIGINAL_ENTRY, ORIGINAL_ENTRY_SHA256),
                           (ORIGINAL_RUNTIME, ORIGINAL_RUNTIME_SHA256)):
        if sha(ROOT/name) != expected:
            raise ValueError('signed recovery dependency SHA changed: '+name)
    original = importlib.import_module('tools.amos_semantic_only_eval')
    proof = original.source_guard(production=production)
    if not production:
        return proof
    for name, expected in EVIDENCE_SHA256.items():
        if sha(ROOT/name) != expected:
            raise ValueError('restoration evidence SHA changed: '+name)
    ready = json.loads((ROOT/DATA_READY).read_text(encoding='utf-8'))
    if not isinstance(ready, dict) or any(
        type(ready.get(name)) is not type(value) or ready[name] != value
        for name, value in READINESS_REQUIRED.items()
    ):
        raise ValueError('restoration readiness contract changed')
    return dict(proof, recovery_evidence=dict(EVIDENCE_SHA256))


def related(proc_root=None):
    """Include both generations by exact argv basename; never signal processes."""
    proc_root = Path('/proc') if proc_root is None else Path(proc_root)
    if not proc_root.exists():
        return []
    names = {'amos_semantic_only_eval.py', 'amos_semantic_only_restored_eval.py'}
    rows = []
    for folder in proc_root.iterdir():
        if not folder.name.isdigit() or int(folder.name) == os.getpid():
            continue
        try:
            args = (folder/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except OSError:
            continue
        if any(Path(arg).name in names for arg in args) and any(
            arg in ('prepare', 'run', 'worker') for arg in args
        ):
            rows.append(dict(pid=int(folder.name), args=args))
    return rows


def preflight(out, *, processes=None, tmux_exists=None, pipeline=None):
    """Apply recovery-only quiescence checks at initial start, never in workers."""
    if Path(out) != ROOT/OUT:
        raise ValueError('fresh dedicated recovery output required')
    if (ROOT/ORIGINAL_OUT).exists():
        raise FileExistsError('original semantic output exists; recovery refused')
    _native_preflight(out, processes=processes, tmux_exists=tmux_exists, pipeline=pipeline)
    old = subprocess.run(['tmux', 'has-session', '-t', ORIGINAL_TMUX], capture_output=True, text=True)
    if old.returncode not in (0, 1):
        raise RuntimeError('original tmux preflight failed: '+old.stderr)
    if old.returncode == 0:
        raise ValueError('original semantic tmux session exists; recovery refused')


def dependency_sources():
    # Provenance is frozen in protocol.sources as evidence, never as predictions.
    # Keep JSON/log paths outside SOURCE_SEEDS: the native import walker parses
    # Python seeds, while the inventory can authenticate any regular file.
    return list(dict.fromkeys([*_native_dependency_sources(), *(ROOT/name for name in EVIDENCE_SHA256)]))


def compile_context(*, source=None):
    source_guard()
    original = importlib.import_module('tools.amos_semantic_only_eval')
    private = original.compile_context(source=source)
    ns = private.__dict__
    ns.update(__name__='_amos_semantic_restored_controller_private',
              __file__=str(Path(__file__).resolve()), __doc__=__doc__,
              OUT=OUT, TMUX=TMUX, PIPELINE=PIPELINE,
              ORIGINAL_ENTRY=ORIGINAL_ENTRY, ORIGINAL_ENTRY_SHA256=ORIGINAL_ENTRY_SHA256,
              ORIGINAL_RUNTIME=ORIGINAL_RUNTIME, ORIGINAL_RUNTIME_SHA256=ORIGINAL_RUNTIME_SHA256,
              ORIGINAL_OUT=ORIGINAL_OUT, ORIGINAL_TMUX=ORIGINAL_TMUX,
              DATA_READY=DATA_READY, EVIDENCE_SHA256=dict(EVIDENCE_SHA256),
              READINESS_REQUIRED=dict(READINESS_REQUIRED),
              _native_preflight=ns['preflight'], _native_dependency_sources=ns['dependency_sources'])
    ns['SOURCE_SEEDS'] = tuple(dict.fromkeys((*ns['SOURCE_SEEDS'], *RECOVERY_SOURCES)))
    for fn in (source_guard, related, preflight, dependency_sources):
        bound = types.FunctionType(fn.__code__, ns, fn.__name__, fn.__defaults__, fn.__closure__)
        bound.__kwdefaults__ = fn.__kwdefaults__
        ns[fn.__name__] = bound
    ns['_source_guard'] = ns['source_guard']
    # The inherited bootstrap and every worker/scheduler function resolve their
    # engine/runtime through this same private namespace, including child argv.
    exec("def engine(): return globals()\ndef runtime(): return importlib.import_module('tools.amos_semantic_only_runtime')\n", ns)
    return private


def context():
    global _CONTEXT
    if _CONTEXT is None:
        _CONTEXT = compile_context()
    return _CONTEXT


def main():
    return context().engine()['main']()


if __name__ == '__main__':
    main()
