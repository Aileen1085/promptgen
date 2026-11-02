"""Fresh full-CT semantic-only AMOS evaluation; independent real GPU gate.

Production writes one fresh row per original key. A separate selected-key
control executes native BGscribble, replay BGscribble, then semantic-only.
The native resource queue, atomic commits and normal completion remain intact.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.amos_semantic_only_runtime import VARIANTS, CONTROL_VARIANTS, PROMPT_VARIANT

ANCESTOR='tools/amos_prompt_ablation_eval.py'
ANCESTOR_SHA256='59963323d0a8cdc86537f1063b04b225966380889218257b577fa16f0affdcc5'
MANIFEST='tools/amos_prompt_ablation_native_manifest_eval.py'
MANIFEST_SHA256='25f234af32e55780bafd4a53954dcdaaeadc80ef6fbf8f7bc6032441299b1ce8'
SELECTED_HELPERS='tools/amos_bg_scribble_module_eval.py'
SELECTED_HELPERS_SHA256='cb8b115b61817525b318e687933d99d6c858f27c571c6435e240915d16e597d4'
SELECTED_OUTPUT='output/amos_v102_e490_floor070_prompt_ablations_20261031_reusefast_ownerfix'
SELECTED_MERGED=SELECTED_OUTPUT+'/merged_metrics.json'
SELECTED_SHA256='5e2bdbf2fa340c69b9d35bd079ed3d04332d95f9b1eea5871111c3ee55cf8777'
SELECTED_AUDIT=SELECTED_OUTPUT+'/independent_results_audit_20261031.json'
SELECTED_AUDIT_SHA256='ff2c1ae5dee728ed2bed22db9d23f72b372bb69fdffd663deb44fa37d7ce5134'
OUT='output/amos_v102_e490_floor070_semantic_only_20261102'
TMUX='amos_v102_floor070_semantic_only_20261102'
PIPELINE='output/amos_semantic_only_pipeline_20261102.log'
PRODUCTION_ROWS=1485
_CONTEXT=None


def source_guard():
    for name,expected in ((ANCESTOR,ANCESTOR_SHA256),(MANIFEST,MANIFEST_SHA256),(SELECTED_HELPERS,SELECTED_HELPERS_SHA256)):
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise ValueError('signed semantic controller SHA changed: '+name)
    importlib.import_module('tools.amos_semantic_only_runtime').source_guard()


_EXTENSIONS=r'''
def expected_variants(control):return CONTROL_VARIANTS if control else VARIANTS


def policy_extension():
    return dict(fresh_rows=1485,reused_rows=0,spatial_prompt_policy='none',full_original_CT=True,
        semantic_prompt_count=1,spatial_prompt_count=0,feedback_interactions=0,
        controls_variants=list(CONTROL_VARIANTS),geometry_source='actual_readonly_case_CT_shape',
        component_policy='retain_threshold_mask_only_when_verified_semantic_and_both_seeds_empty',
        prompt_cache_inference=False,selected_plane_inference=False,gt_geometry_inference=False,
        zero_weighted_pool_unchanged=True,zero_strength_anchor_unchanged=True)


def control_jobs(tasks):
    task=selected_control()['task']
    if task not in tasks:raise ValueError('selected control outside original1485 keys')
    j=case_jobs([task],strict=False)[0]
    return [dict(j,name='amos_native_control',control=True,variants=list(CONTROL_VARIANTS))]


def validate_class(records,task,control=False):
    expected=expected_variants(control)
    if len(records)!=len(expected) or {r.get('variant') for r in records}!=set(expected):
        raise ValueError('all expected variants must complete atomically')
    batches={r.get('actual_frame_batch') for r in records}
    if len(batches)!=1 or next(iter(batches)) not in (1,2,4):raise ValueError('atomic variants require same actual frame batch')
    batch=next(iter(batches))
    for record in records:validate_record(record,task,record['variant'],batch,control=control)
    return batch


def _native_bootstrap_preflight(out,jobs,controls):
    audit=_bootstrap_function(root=ROOT,out=out,jobs=jobs,controls=controls)
    path=Path(out)/'native_bootstrap_audit.json';atomic_new(path,audit);return path


def verify(out,**kwargs):
    _source_guard();protocol=_native_verify(out,**kwargs);p=protocol['policy']
    if p.get('rows')!=1485 or p.get('interactions')!=1 or any(p.get(k)!=v for k,v in policy_extension().items()):
        raise ValueError('semantic full CT one-row policy changed')
    jobs=read(Path(out)/'jobs.json')
    if any(j.get('variants')!=list(expected_variants(j['control'])) for group in jobs.values() for j in group):raise ValueError('canonical variant policy changed')
    return protocol


def prepare(out,**kwargs):
    _source_guard();return _native_prepare(out,**kwargs)


def related(proc_root=None):
    proc_root=Path('/proc') if proc_root is None else Path(proc_root)
    if not proc_root.exists():return []
    rows=[]
    for folder in proc_root.iterdir():
        if not folder.name.isdigit() or int(folder.name)==os.getpid():continue
        try:args=(folder/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except OSError:continue
        if any(Path(a).name=='amos_semantic_only_eval.py' for a in args) and any(a in ('prepare','run','worker') for a in args):
            rows.append(dict(pid=int(folder.name),args=args))
    return rows


def build_merged(out):
    out=Path(out)
    require_merge_state(read(out/'status.json'),[p for p in related() if 'worker' in p['args']])
    verify(out,rehash=True);check_control_gate(out)
    tasks=read(out/'manifest.json')['tasks'];validate_scope(tasks)
    found=class_records(out);jobs=read(out/'jobs.json')['cases']
    if len(found)!=1485 or set(found)!={key(t) for t in tasks} or len(jobs)!=100:raise ValueError('exact1485 fresh keys and100 case proofs required')
    for j in jobs:verify_case_success(out,j,found)
    rows=[]
    for t in sorted(tasks,key=key):
        records=found[key(t)]['record']['records']
        if len(records)!=1 or records[0]['variant']!='semantic_only':raise ValueError('production contains baseline/control rows')
        rows.append(records[0]['row'])
    validate_rows(rows,tasks);previous=native_reference();summary=summarize(rows);index={key(r):r for r in previous}
    summary.update(delta_from_selected_prompt_baseline=paired_delta(rows,previous),
        paired_deltas=[dict(key=list(key(r)),selected_prompt_baseline={m:float(r[m])-float(index[key(r)][m]) for m in METRICS}) for r in sorted(rows,key=key)],
        actual_frame_batches={str(b):sum(v['spec']['frame_batch']==b for v in found.values()) for b in (1,2,4)})
    return dict(variants={'semantic_only':summary},rows=1485,task_count_per_variant=1485,cases=100,classes=15,
        fresh_rows=1485,reused_rows=0,controls_in_final=False,spatial_prompt_policy='none',full_original_CT=True,
        floor=.70,fallback=.60,weights='E490live',winner_selection=False,
        protocol_sha256=sha(out/'protocol.json'),control_verification_sha256=sha(out/'control_verification.json'),
        selected_prompt_reference_sha256=OLD_SHA,
        comparison='Selected S21 BGscribble baseline uses its original prompt ROI; semantic-only uses full original CT.')


def merge(out):
    out=Path(out);value=build_merged(out)
    if (out/'merged_metrics.json').exists():
        if read(out/'merged_metrics.json')!=value:raise ValueError('existing once-only merged result differs from complete proofs')
        return value
    verify(out);require_merge_state(read(out/'status.json'),[p for p in related() if 'worker' in p['args']])
    atomic_new(out/'merged_metrics.json',value);return value
'''


def compile_context(*,source=None):
    source_guard();source=(ROOT/ANCESTOR).read_bytes() if source is None else source
    if hashlib.sha256(source).hexdigest()!=ANCESTOR_SHA256:raise ValueError('signed native controller SHA mismatch')
    manifest=importlib.import_module('tools.amos_prompt_ablation_native_manifest_eval')
    tree,proof=manifest.transform_controller(source,str(ROOT/ANCESTOR),ANCESTOR_SHA256)
    original=copy.deepcopy(tree);changes=[];counts={}
    def replace(p,f,value,label):
        changes.append((p,f,getattr(p,f)));setattr(p,f,value);counts[label]=counts.get(label,0)+1
    for fn in tree.body:
        if not isinstance(fn,ast.FunctionDef):continue
        if fn.name=='worker':
            for n in ast.walk(fn):
                if isinstance(n,ast.comprehension) and ast.unparse(n.iter)=='VARIANTS':
                    replace(n,'iter',ast.parse("expected_variants(job['control'])",mode='eval').body,'worker_variants')
                if isinstance(n,ast.keyword) and n.arg=='variants' and isinstance(n.value,ast.Constant) and n.value.value==6:
                    replace(n,'value',ast.parse("len(expected_variants(job['control']))",mode='eval').body,'worker_count')
        if fn.name=='verify_worker_complete':
            for n in ast.walk(fn):
                if isinstance(n,ast.Compare) and ast.unparse(n)=="value.get('variants') != 6":
                    replace(n,'comparators',[ast.parse("len(expected_variants(job['control']))",mode='eval').body],'complete_count')
        if fn.name=='prepare':
            for n in ast.walk(fn):
                if isinstance(n,ast.keyword) and n.arg=='policy':
                    keywords=n.value.keywords
                    for k in keywords:
                        if k.arg=='rows':replace(k,'value',ast.Constant(value=1485),'policy_rows')
                    replace(n.value,'keywords',[*keywords,ast.keyword(arg=None,value=ast.Call(func=ast.Name(id='policy_extension',ctx=ast.Load()),args=[],keywords=[]))],'policy_extension')
                if isinstance(n,ast.keyword) and n.arg=='reference' and ast.unparse(n.value)=="file_inventory([ROOT / OLD, ROOT / HISTORY / 'manifest.json'] + ref_paths)":
                    replace(n,'value',ast.parse("file_inventory([ROOT/OLD,ROOT/SELECTED_AUDIT,ROOT/HISTORY/'manifest.json']+ref_paths+list(map(Path,selected_control()['files'])))",mode='eval').body,'reference')
    expected=dict(worker_variants=1,worker_count=1,complete_count=1,policy_rows=1,policy_extension=1,reference=1)
    if counts!=expected:raise ValueError('semantic controller anchors changed: '+repr(counts))
    transformed=copy.deepcopy(tree)
    for p,f,v in reversed(changes):setattr(p,f,v)
    if ast.dump(tree,include_attributes=False)!=ast.dump(original,include_attributes=False):raise ValueError('semantic controller reversal failed')
    private=types.ModuleType('_amos_semantic_controller_private');ns=private.__dict__;ns['__file__']=str(Path(__file__).resolve())
    exec(compile(ast.fix_missing_locations(transformed),str(ROOT/ANCESTOR),'exec'),ns)
    ns.update(OUT=OUT,TMUX=TMUX,PIPELINE=PIPELINE,VARIANTS=VARIANTS,CONTROL_VARIANTS=CONTROL_VARIANTS,
        PROMPT_VARIANT=PROMPT_VARIANT,PRODUCTION_ROWS=PRODUCTION_ROWS,
        OLD=SELECTED_MERGED,OLD_SHA=SELECTED_SHA256,SELECTED_OUTPUT=SELECTED_OUTPUT,
        SELECTED_AUDIT=SELECTED_AUDIT,SELECTED_AUDIT_SHA256=SELECTED_AUDIT_SHA256,
        TRANSFORM_PROOF=dict(proof,controller_reverse_ast_equal=True,semantic_counts=counts),
        runtime=lambda:importlib.import_module('tools.amos_semantic_only_runtime'),_source_guard=source_guard,
        _native_manifest_payload=manifest.native_manifest_payload,
        _native_check_controls=ns['check_controls'],_native_verify=ns['verify'],_native_prepare=ns['prepare'])
    ns['SOURCE_SEEDS']=tuple(dict.fromkeys((*ns['SOURCE_SEEDS'],MANIFEST,SELECTED_HELPERS,
        'tools/amos_semantic_only_runtime.py','tools/amos_semantic_only_eval.py','tools/amos_semantic_only_prompts.py',
        'tests/test_amos_semantic_only_runtime.py','tests/test_amos_semantic_only_eval.py','tests/test_amos_semantic_only_prompts.py',
        'tools/amos_prompt_reuse_owner_runtime.py','tools/amos_prompt_reuse_runtime.py','tools/amos_prompt_ablation_restore_runtime.py')))
    # Reuse only the signed source-audited selected-reference functions. Their
    # globals point to this controller, never to the predecessor's namespace.
    helper_tree=ast.parse((ROOT/SELECTED_HELPERS).read_bytes())
    text=next(ast.literal_eval(n.value) for n in helper_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_EXTENSIONS' for t in n.targets))
    names={'native_reference','selected_control','compare_selected_reference','check_controls','publish_control_gate'}
    helpers=[n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in helpers}!=names:raise ValueError('selected reference helper anchors changed')
    exec(compile(ast.fix_missing_locations(ast.Module(body=helpers,type_ignores=[])),str(ROOT/SELECTED_HELPERS),'exec'),ns)
    # The native pre-CUDA metadata path remains real, with the selected fresh
    # control key and BGscribble compatibility tag as in the selected baseline.
    fn=next(n for n in ast.parse((ROOT/MANIFEST).read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='run_native_bootstrap')
    original_fn=copy.deepcopy(fn);undo=[];edits=0
    for n in ast.walk(fn):
        if isinstance(n,ast.Constant) and n.value=='point':undo.append((n,'value',n.value));n.value='scribble';edits+=1
        if isinstance(n,ast.Compare) and ast.unparse(n)=="controls[0]['pending'] != [sorted([t for j in jobs for t in j['pending']], key=context['key'])[0]]":
            undo.append((n,'comparators',n.comparators));n.comparators=[ast.parse("[selected_control()['task']]",mode='eval').body];edits+=1
    if edits!=2:raise ValueError('native bootstrap anchors changed')
    transformed_fn=copy.deepcopy(fn)
    for n,f,v in reversed(undo):setattr(n,f,v)
    if ast.dump(fn,include_attributes=False)!=ast.dump(original_fn,include_attributes=False):raise ValueError('bootstrap reversal failed')
    ns.update(engine=lambda:ns,_sha=manifest._sha,compile_bootstrap=manifest.compile_bootstrap)
    fragment=ast.fix_missing_locations(ast.Module(body=[transformed_fn],type_ignores=[]));ns['BOOTSTRAP_SOURCE']=ast.unparse(fragment)
    exec(compile(fragment,str(ROOT/MANIFEST),'exec'),ns);ns['_bootstrap_function']=ns.pop('run_native_bootstrap')
    ns['TRANSFORM_PROOF'].update(bootstrap_reverse_ast_equal=True,bootstrap_anchor_replacements=2)
    exec(compile(_EXTENSIONS,str(Path(__file__).resolve()),'exec'),ns)
    ns['__doc__']=__doc__;private.engine=lambda:ns
    return private


def context():
    global _CONTEXT
    if _CONTEXT is None:_CONTEXT=compile_context()
    return _CONTEXT


def main():return context().engine()['main']()


if __name__=='__main__':main()
