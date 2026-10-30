"""Additive native-manifest repair with the real pre-CUDA startup contract.

Exactly four manifest constructors and one preparation audit insertion differ
from the signed controller. Native inference, controls, scheduling and arithmetic
remain in a single private globals namespace.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
NATIVE_CONTROLLER_SHA256='59963323d0a8cdc86537f1063b04b225966380889218257b577fa16f0affdcc5'
RESTORE_CONTROLLER_SHA256='d6576a410f1a9d2050695021fe95252c192379db2a7674780bbab48a4c44f132'
RESTORE_RUNTIME_SHA256='ad1c19d82449c84186160b876adfba14ea40f0faf41fc491a5a2f637dce929e3'
SECOND_EVIDENCE='output/amos_prompt_ablation_manifest_failure_20261030.json'
SECOND_SHA256='56b53681dfb5494fd87d3d1542cb23d73b2dfa153ab8c23e983359e20f082920'
SECOND_OUTPUT='output/amos_v102_e490_floor070_prompt_ablations_20261030_restorefix'
SECOND_FILES=(SECOND_OUTPUT+'/protocol.json',SECOND_OUTPUT+'/protocol_signature.json',
              SECOND_OUTPUT+'/status.json',SECOND_OUTPUT+'/job_specs/amos_native_control_a0.json',
              SECOND_OUTPUT+'/logs/amos_native_control_a0.log','output/amos_prompt_ablations_restorefix_pipeline_20261030.log')
OUT='output/amos_v102_e490_floor070_prompt_ablations_20261030_nativefix'
TMUX='amos_v102_floor070_prompt_ablations_nativefix'
PIPELINE='output/amos_prompt_ablations_nativefix_pipeline_20261030.log'
_ENGINE=None


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def _ast_sha(node):return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()


def native_manifest_payload(*,tasks,expected_task_count):
    from tools.ct13_full_validation_eval import manifest_payload
    if expected_task_count!=len(tasks):raise ValueError('native manifest expected task count differs')
    return manifest_payload(tasks)


def reverse_transform(tree):
    """Audit helper: reverse the five narrow edits to recover the signed AST."""
    tree=copy.deepcopy(tree)
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='_native_manifest_payload':node.func.id='dict'
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name=='prepare':
            node.body=[stmt for stmt in node.body if not (
                isinstance(stmt,ast.Expr) and isinstance(stmt.value,ast.Call)
                and ast.unparse(stmt.value)=='immutable.append(_native_bootstrap_preflight(out, jobs, controls))')]
    return tree


def transform_controller(source,filename,expected_sha256):
    payload=source if isinstance(source,bytes) else source.encode()
    if hashlib.sha256(payload).hexdigest()!=expected_sha256:raise ValueError('signed native controller SHA mismatch')
    original=ast.parse(payload,filename=str(filename));tree=copy.deepcopy(original);matches=[]
    for node in ast.walk(tree):
        if (isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='dict' and not node.args
                and {kw.arg for kw in node.keywords}=={'tasks','expected_task_count'}):matches.append(node)
    if len(matches)!=4:raise ValueError('exactly four native manifest constructors required')
    for node in matches:node.func.id='_native_manifest_payload'
    prepares=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare']
    if len(prepares)!=1:raise ValueError('unique native prepare required')
    body=prepares[0].body
    anchors=[i for i,n in enumerate(body) if isinstance(n,ast.Assign) and len(n.targets)==1
             and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='protocol']
    if len(anchors)!=1 or not any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='immutable' for t in n.targets) for n in body[:anchors[0]]):
        raise ValueError('native immutable/protocol insertion anchors changed')
    body.insert(anchors[0],ast.parse('immutable.append(_native_bootstrap_preflight(out,jobs,controls))').body[0])
    if ast.dump(reverse_transform(tree),include_attributes=False)!=ast.dump(original,include_attributes=False):raise ValueError('unexpected controller AST change')
    proof=dict(original_source_sha256=expected_sha256,original_ast_sha256=_ast_sha(original),transformed_ast_sha256=_ast_sha(tree),
               manifest_replacements=4,audit_insertions=1)
    return ast.fix_missing_locations(tree),proof


def compile_bootstrap(source,filename,expected_sha256):
    """Compile the unchanged native main prefix, ending before device mutation."""
    runtime=importlib.import_module('tools.amos_prompt_ablation_runtime')
    program=runtime.compile_native(source,filename,expected_sha256)
    payload=source if isinstance(source,bytes) else source.encode();tree=ast.parse(payload,filename=str(filename))
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    anchors=[]
    for i,node in enumerate(main.body):
        if isinstance(node,ast.Assign) and any(ast.unparse(t)=="os.environ['CUDA_VISIBLE_DEVICES']" for t in node.targets):anchors.append(i)
    if len(anchors)!=1:raise ValueError('unique native CUDA startup boundary required')
    prefix=main.body[:anchors[0]]
    last=ast.unparse(prefix[-1]) if prefix else ''
    if last!='split, validation = build_selected_protocol(base_split, tasks)':raise ValueError('native pre-CUDA metadata contract endpoint changed')
    fragment=ast.fix_missing_locations(ast.Module(body=copy.deepcopy(prefix),type_ignores=[]))
    return dict(namespace=program.namespace(),code=compile(fragment,str(filename),'exec'),prefix_ast_sha256=_ast_sha(fragment),
                prefix_source=ast.unparse(fragment),last_statement=last,native_source_sha256=expected_sha256)


def run_native_bootstrap(*,root,out,jobs,controls,strict=True):
    """Run all real native metadata setup paths without opening weights or CTs."""
    root=Path(root);out=Path(out);context=engine();split_path=root/context['TRAINING']
    split=json.loads(split_path.read_text(encoding='utf-8'))
    restored=importlib.import_module('tools.amos_prompt_ablation_restore_runtime')
    preload=restored._preload_ct13(split)
    if strict:
        if len(jobs)!=100 or len(controls)!=1 or sum(j['task_count'] for j in jobs)!=1485:raise ValueError('native bootstrap requires100 cases and1 separate control')
        context['validate_scope']([t for j in jobs for t in j['pending']],split)
        if controls[0]['task_count']!=1 or controls[0]['pending']!=[sorted([t for j in jobs for t in j['pending']],key=context['key'])[0]]:raise ValueError('native bootstrap control key changed')
    native_path=root/context['NATIVE'];native_sha=_sha(native_path)
    before_cuda=os.environ.get('CUDA_VISIBLE_DEVICES');torch_before='torch' in sys.modules
    program=compile_bootstrap(native_path.read_bytes(),str(native_path),native_sha);ns=program['namespace'];audits=[]
    for job in jobs+controls:
        prepared=Path(job['prepared']);expected=sorted(job['pending'],key=context['key'])
        argv=[str(native_path),'--independent-task-manifest',str(prepared/'manifest.json'),
              '--nninteractive-metrics',str(prepared/'refs.json'),'--shared-adaptive-prompt-root',str(root/context['CACHE']),
              '--weights',str(root/context['WEIGHTS']),'--source','amos','--background-mode','point',
              '--eval-out-dir',str(out),'--run-name',job['name'],'--gpu','0','--sam2-frame-batch-size',str(job['frame_batch']),
              '--task-shard-count','1','--task-shard-index','0','--dynamic-threshold-min','0.70',
              '--dynamic-threshold-max','0.75','--dynamic-threshold-fallback','0.60',
              '--prompt-logit-neighborhood-radius','2','--base-split-json',str(split_path),'--runtime-version','v10.2']
        saved_argv=sys.argv
        try:
            sys.argv=argv;exec(program['code'],ns)
        finally:sys.argv=saved_argv
        keys=[context['key'](t) for t in expected]
        if (ns['tasks']!=expected or ns['manifest_tasks']!=expected or set(ns['prompt_index'])!=set(keys)
                or len({k[:2] for k in keys})!=1 or ns['custom'].sam2_frame_batch_size!=job['frame_batch']):
            raise ValueError('actual native bootstrap task/prompt/case/batch contract mismatch')
        validation_keys=[context['key'](t) for t in ns['validation']['tasks']]
        if set(validation_keys)!=set(keys) or len(validation_keys)!=len(keys):raise ValueError('actual native selected validation task keys changed')
        if tuple(ns['split']['source_order'])!=tuple(split['source_order']):raise ValueError('actual native selected source order changed')
        audits.append(dict(name=job['name'],control=job['control'],case_id=job['case_id'],task_count=len(keys),
                           actual_task_keys=[list(k) for k in keys],actual_prompt_keys=[list(k) for k in sorted(ns['prompt_index'])],
                           actual_validation_keys=[list(k) for k in sorted(validation_keys)],
                           manifest_sha256=_sha(prepared/'manifest.json'),refs_sha256=_sha(prepared/'refs.json')))
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=before_cuda or (not torch_before and 'torch' in sys.modules):raise ValueError('native metadata bootstrap crossed model/CUDA boundary')
    if _sha(native_path)!=native_sha:raise ValueError('native source changed during bootstrap')
    return dict(protocol='actual_native_main_pre_cuda_metadata_contract_v1',native_source_sha256=native_sha,
                prefix_ast_sha256=program['prefix_ast_sha256'],ct13_preload=preload,job_count=len(audits),
                full_tasks=sum(j['task_count'] for j in jobs),control_tasks=sum(j['task_count'] for j in controls),jobs=audits,
                weights_opened=False,model_loaded=False,cuda_initialized=False,ct_read=False,
                controller_transform=context['TRANSFORM_PROOF'])


def _source_guard(root):
    root=Path(root)
    for name,expected in (('tools/amos_prompt_ablation_eval.py',NATIVE_CONTROLLER_SHA256),
                          ('tools/amos_prompt_ablation_restore_eval.py',RESTORE_CONTROLLER_SHA256),
                          ('tools/amos_prompt_ablation_restore_runtime.py',RESTORE_RUNTIME_SHA256)):
        if _sha(root/name)!=expected:raise ValueError('signed source SHA changed: '+name)


def verify_failures(root=None):
    root=Path(root or ROOT);_source_guard(root)
    prior=importlib.import_module('tools.amos_prompt_ablation_restore_eval');first=prior.verify_bootstrap(root)
    if _sha(root/SECOND_EVIDENCE)!=SECOND_SHA256:raise ValueError('second immutable bootstrap failure SHA changed')
    second=json.loads((root/SECOND_EVIDENCE).read_text(encoding='utf-8'))
    if (second.get('ct13_import_order_passed') is not True or second.get('model_loaded') is not False
            or second.get('controls_passed') is not False or second.get('full_started') is not False
            or second.get('committed_classes')!=0 or second.get('related_checks')!=[[],[]]
            or set(second.get('files',{}))!=set(SECOND_FILES)):raise ValueError('second bootstrap evidence is not an empty pre-model manifest failure')
    for name,expected in second['files'].items():
        if _sha(root/name)!=expected:raise ValueError('second retained bootstrap file changed: '+name)
    queue=second.get('queue',{});failed=queue.get('failed',[])
    if (queue.get('phase')!='controls' or queue.get('pending')!=0 or queue.get('running')!={} or queue.get('committed_classes')!=0
            or len(failed)!=1 or failed[0].get('name')!='amos_native_control' or failed[0].get('returncode')!=1
            or Path(failed[0].get('log','')).resolve()!=(root/SECOND_FILES[4]).resolve()
            or json.loads((root/SECOND_FILES[2]).read_text(encoding='utf-8'))!=queue
            or any((root/SECOND_OUTPUT/'runs').glob('*/classes/*.done.json'))
            or any((root/SECOND_OUTPUT/'case_success').glob('*.json'))):raise ValueError('second bootstrap failure queue/proofs changed')
    # The predecessor protocol files are themselves immutable evidence. Verify
    # their explicitly frozen source inventories too, including native helpers;
    # a fresh prepare must not silently accept modified predecessor code.
    for protocol_path in (root/prior.FAILURE_FILES[0],root/SECOND_FILES[0]):
        sources=json.loads(protocol_path.read_text(encoding='utf-8')).get('sources')
        if not isinstance(sources,dict) or not sources:raise ValueError('prior protocol source inventory missing')
        for name,expected in sources.items():
            if _sha(root/name)!=expected:raise ValueError('prior protocol frozen source changed: '+name)
    return first,second


def related(proc_root=None):
    proc_root=Path('/proc') if proc_root is None else Path(proc_root)
    if not proc_root.exists():return []
    entries={'amos_prompt_ablation_eval.py','amos_prompt_ablation_restore_eval.py','amos_prompt_ablation_native_manifest_eval.py'};rows=[]
    for folder in proc_root.iterdir():
        if not folder.name.isdigit() or int(folder.name)==os.getpid():continue
        try:args=(folder/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except OSError:continue
        if any(Path(a).name in entries for a in args) and any(a in ('prepare','run','worker') for a in args):rows.append(dict(pid=int(folder.name),args=args))
    return rows


def _runtime():return importlib.import_module('tools.amos_prompt_ablation_restore_runtime')


def engine():
    global _ENGINE
    if _ENGINE is not None:return _ENGINE
    _source_guard(ROOT);old=ROOT/'tools/amos_prompt_ablation_eval.py'
    tree,proof=transform_controller(old.read_bytes(),str(old),NATIVE_CONTROLLER_SHA256)
    private=types.ModuleType('_amos_prompt_ablation_native_manifest_private');ns=private.__dict__;ns['__file__']=str(Path(__file__).resolve())
    exec(compile(tree,str(old),'exec'),ns)
    ns.update(OUT=OUT,TMUX=TMUX,PIPELINE=PIPELINE,related=related,runtime=_runtime,
              _native_manifest_payload=native_manifest_payload,TRANSFORM_PROOF=proof)
    prior=importlib.import_module('tools.amos_prompt_ablation_restore_eval')
    ns['SOURCE_SEEDS']=ns['SOURCE_SEEDS']+('tools/amos_prompt_ablation_restore_eval.py','tools/amos_prompt_ablation_restore_runtime.py',
        'tools/amos_prompt_ablation_native_manifest_eval.py',prior.BOOTSTRAP_EVIDENCE,*prior.FAILURE_FILES,SECOND_EVIDENCE,*SECOND_FILES)
    def bootstrap(out,jobs,controls):
        audit=run_native_bootstrap(root=ns['ROOT'],out=out,jobs=jobs,controls=controls)
        path=Path(out)/'native_bootstrap_audit.json';ns['atomic_new'](path,audit);return path
    ns['_native_bootstrap_preflight']=bootstrap
    original_prepare=ns['prepare'];original_verify=ns['verify'];original_pipeline=ns['run_pipeline']
    def prepare(*args,**kwargs):verify_failures(ns['ROOT']);return original_prepare(*args,**kwargs)
    def verify(*args,**kwargs):verify_failures(ns['ROOT']);return original_verify(*args,**kwargs)
    def run_pipeline(*args,**kwargs):verify_failures(ns['ROOT']);return original_pipeline(*args,**kwargs)
    ns.update(prepare=prepare,verify=verify,run_pipeline=run_pipeline)
    _ENGINE=ns;return ns


def __getattr__(name):
    if name.startswith('__'):raise AttributeError(name)
    ns=engine()
    if name not in ns:raise AttributeError(name)
    return ns[name]


def main():return engine()['main']()


if __name__=='__main__':main()
