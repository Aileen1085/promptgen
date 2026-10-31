"""Fresh AMOS E490/live module sensitivity under actual FG/BG scribbles.

No old class metrics enter this controller. Frozen native scheduling, atomic
class commits and exact metrics execute in one current private namespace.
Full production waits for a separate once-only real-GPU review artifact.
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
from tools.amos_bg_scribble_module_runtime import MODULE_VARIANTS, PROMPT_VARIANT

ANCESTOR='tools/amos_prompt_ablation_eval.py'
ANCESTOR_SHA256='59963323d0a8cdc86537f1063b04b225966380889218257b577fa16f0affdcc5'
MANIFEST='tools/amos_prompt_ablation_native_manifest_eval.py'
MANIFEST_SHA256='25f234af32e55780bafd4a53954dcdaaeadc80ef6fbf8f7bc6032441299b1ce8'
SELECTED_OUTPUT='output/amos_v102_e490_floor070_prompt_ablations_20261031_reusefast_ownerfix'
SELECTED_MERGED=SELECTED_OUTPUT+'/merged_metrics.json'
SELECTED_SHA256='5e2bdbf2fa340c69b9d35bd079ed3d04332d95f9b1eea5871111c3ee55cf8777'
SELECTED_AUDIT=SELECTED_OUTPUT+'/independent_results_audit_20261031.json'
SELECTED_AUDIT_SHA256='ff2c1ae5dee728ed2bed22db9d23f72b372bb69fdffd663deb44fa37d7ce5134'
OUT='output/amos_v102_e490_floor070_bg_scribble_modules_20261031'
TMUX='amos_v102_floor070_bg_scribble_modules_20261031'
PIPELINE='output/amos_bg_scribble_modules_pipeline_20261031.log'
_CONTEXT=None


def source_guard():
    for name,expected in ((ANCESTOR,ANCESTOR_SHA256),(MANIFEST,MANIFEST_SHA256)):
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise ValueError('signed module controller SHA changed: '+name)
    importlib.import_module('tools.amos_bg_scribble_module_runtime').source_guard()


# Executed in the native controller dictionary. These helpers must never keep
# references to old controller globals, labels, output, or runtime modules.
_EXTENSIONS=r'''
def native_reference():
    if sha(ROOT/OLD)!=OLD_SHA or sha(ROOT/SELECTED_AUDIT)!=SELECTED_AUDIT_SHA256:
        raise ValueError('completed selected BGscribble reference SHA changed')
    audit=read(ROOT/SELECTED_AUDIT)
    if not all(audit.get(f) is True for f in ('complete','data_rehash','signature_and_real_controls_pass','scheduler_workers_absent')):
        raise ValueError('completed BGscribble independent audit missing')
    value=read(ROOT/OLD)
    rows=value['variants'][PROMPT_VARIANT]['per_case_class']
    validate_scope(rows,require_labels=False);validate_rows(rows,rows)
    if value.get('rows')!=8910 or value.get('floor')!=.70 or value.get('weights')!='E490live':
        raise ValueError('selected E490 floor070 merged scope changed')
    return rows


def selected_control():
    # Choose a fully proven *fresh* record in the completed prompt output. Its
    # earlier reused680 classes are not imported into this evaluator at all.
    folder=ROOT/SELECTED_OUTPUT
    for marker in sorted((folder/'runs').glob('*/classes/*.done.json')):
        info=read(marker);record_path=Path(info['record']);value=read(record_path)
        if value.get('control') or value.get('actual_frame_batch')!=4:continue
        matches=[r for r in value['records'] if r.get('variant')==PROMPT_VARIANT]
        if len(matches)!=1:raise ValueError('selected BGscribble control record ambiguous')
        for name,expected in info['files'].items():
            if sha(name)!=expected:raise ValueError('selected completed class proof changed')
        if info['files'].get(str(record_path))!=sha(record_path):raise ValueError('selected marker omitted record')
        success_path=folder/'case_success'/(info['job_name']+'.json');success=read(success_path)
        if success['markers'].get(str(marker))!=sha(marker):raise ValueError('selected normal completion omitted class')
        files=dict(info['files']);files[str(marker)]=sha(marker);files[str(success_path)]=sha(success_path)
        for path,field in ((success['native_worker_complete'],'native_worker_complete_sha256'),
                           (success['final_spec'],'final_spec_sha256'),(success['final_log'],'final_log_sha256')):
            if sha(path)!=success[field]:raise ValueError('selected successful completion proof changed')
            files[path]=success[field]
        worker=read(success['native_worker_complete']);run=Path(success['native_worker_complete']).parent
        final=read(run/'restoration_final.json')
        if (worker.get('native_worker_complete') is not True
                or worker.get('restoration_final_sha256')!=sha(run/'restoration_final.json')
                or final.get('successful_native_completion') is not True
                or final.get('full_tensor_comparison_phase')!='normal_worker_completion'
                or tuple(final.get(f) for f in ('model','epoch','weight_mode','prompt_parameters_exact','decoder_parameters_exact','guards_passed','model_load_count'))!=('v10_2',490,'live',418,212,True,1)):
            raise ValueError('selected normal finish restoration missing')
        files[str(run/'restoration_final.json')]=sha(run/'restoration_final.json')
        record=matches[0]
        if record.get('actual_frame_batch')!=4:raise ValueError('selected record actual batch differs')
        runtime().native_engine()['validate_record'](record,value['task'],PROMPT_VARIANT,4,control=False)
        baseline={key(r):r for r in native_reference()}
        if any(float(record['row'][m])!=float(baseline[key(value['task'])][m]) for m in METRICS):
            raise ValueError('selected completed class differs from signed merged rows')
        return dict(task=value['task'],record=record,files=files)
    raise ValueError('no completed same-key batch4 BGscribble reference class available')


def control_jobs(tasks):
    reference=selected_control();task=reference['task']
    if task not in tasks:raise ValueError('selected control outside original1485 keys')
    j=case_jobs([task],strict=False)[0]
    return [dict(j,name='amos_native_control',control=True)]


def compare_selected_reference(native,previous):
    if native.get('actual_frame_batch')!=4 or previous.get('actual_frame_batch')!=4:
        raise ValueError('selected controls require actual batch4')
    if key(native['row'])!=key(previous['row']):raise ValueError('selected control key differs')
    error=max(abs(float(native['row'][m])-float(previous['row'][m])) for m in METRICS)
    if error>1e-6:raise ValueError('selected BGscribble metrics differ')
    geometry=lambda r:{k:r['roi'][k] for k in ('initial','reserved','final','history')}
    if geometry(native)!=geometry(previous) or native['stages']!=previous['stages']:
        raise ValueError('selected BGscribble ROI/stages differ')
    return dict(max_metric_error=error,roi_equal=True,stages_equal=True,actual_frame_batch=4)


def check_controls(out):
    result=_native_check_controls(out)
    reference=selected_control()
    if key(result['task'])!=key(reference['task']):raise ValueError('fresh controls use wrong selected key')
    spec=result['spec'];control=read(Path(spec['run'])/'native_control.json')
    comparison=compare_selected_reference(control['native'],reference['record'])
    result['digest']=digest(dict(native_control_digest=result['digest'],selected_files=file_inventory(list(map(Path,reference['files']))),
        selected_merged_sha256=OLD_SHA,selected_audit_sha256=SELECTED_AUDIT_SHA256,comparison=comparison))
    result['selected_reference']=comparison
    return result


def _native_bootstrap_preflight(out,jobs,controls):
    audit=_bootstrap_function(root=ROOT,out=out,jobs=jobs,controls=controls)
    path=Path(out)/'native_bootstrap_audit.json';atomic_new(path,audit);return path


def verify(out,**kwargs):
    _source_guard()
    protocol=_native_verify(out,**kwargs)
    policy=protocol['policy']
    if policy.get('prompt_variant')!=PROMPT_VARIANT or policy.get('fresh_module_rows')!=8910:
        raise ValueError('constant module BGscribble protocol changed')
    return protocol


def prepare(out,**kwargs):
    _source_guard()
    return _native_prepare(out,**kwargs)


def related(proc_root=None):
    proc_root=Path('/proc') if proc_root is None else Path(proc_root)
    if not proc_root.exists():return []
    rows=[]
    for folder in proc_root.iterdir():
        if not folder.name.isdigit() or int(folder.name)==os.getpid():continue
        try:args=(folder/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except OSError:continue
        if any(Path(a).name=='amos_bg_scribble_module_eval.py' for a in args) and any(a in ('prepare','run','worker') for a in args):
            rows.append(dict(pid=int(folder.name),args=args))
    return rows


def publish_control_gate(out,reviewer):
    if not reviewer or reviewer in ('controller','runtime','cpu_tests'):raise ValueError('independent reviewer required')
    out=Path(out)
    if (out/'control_verification.json').exists():return check_control_gate(out)
    verify(out,rehash=True);control=check_controls(out)
    gate=dict(passed=True,real_gpu=True,reviewer=reviewer,protocol_sha256=sha(out/'protocol.json'),controls_digest=control['digest'])
    atomic_new(out/'control_verification.json',gate)
    return check_control_gate(out)


def merge(out):
    out=Path(out)
    if (out/'merged_metrics.json').exists():
        verify(out,rehash=True);check_control_gate(out)
        value=read(out/'merged_metrics.json')
        expected=build_merged(out)
        if value!=expected:raise ValueError('existing once-only merged result differs from complete proofs')
        return value
    value=build_merged(out)
    verify(out);require_merge_state(read(out/'status.json'),[p for p in related() if 'worker' in p['args']])
    atomic_new(out/'merged_metrics.json',value)
    return value


def build_merged(out):
    out=Path(out);require_merge_state(read(out/'status.json'),[p for p in related() if 'worker' in p['args']])
    verify(out,rehash=True);check_control_gate(out)
    tasks=read(out/'manifest.json')['tasks'];found=class_records(out)
    if len(found)!=1485 or set(found)!={key(t) for t in tasks}:raise ValueError('exact1485 fresh class commits required')
    for j in read(out/'jobs.json')['cases']:verify_case_success(out,j,found)
    by_variant={v:[] for v in VARIANTS}
    for t in sorted(tasks,key=key):
        for r in found[key(t)]['record']['records']:by_variant[r['variant']].append(r['row'])
    fresh=by_variant['baseline'];previous=native_reference();index={key(r):r for r in fresh};result={}
    for variant,rows in by_variant.items():
        validate_rows(rows,tasks);summary=summarize(rows)
        summary.update(delta_from_fresh_baseline=paired_delta(rows,fresh),delta_from_selected_prompt_baseline=paired_delta(rows,previous),
                       floor=.70,prompt_variant=PROMPT_VARIANT,module_variant=variant,
                       paired_deltas=[dict(key=list(key(r)),fresh_baseline={m:float(r[m])-float(index[key(r)][m]) for m in METRICS}) for r in sorted(rows,key=key)],
                       actual_frame_batches={str(b):sum(v['spec']['frame_batch']==b for v in found.values()) for b in (1,2,4)})
        result[variant]=summary
    return dict(variants=result,rows=8910,task_count_per_variant=1485,controls_in_final=False,fresh_rows=8910,reused_rows=0,
        prompt_variant=PROMPT_VARIANT,floor=.70,weights='E490live',winner_selection=False,
        protocol_sha256=sha(out/'protocol.json'),control_verification_sha256=sha(out/'control_verification.json'),
        selected_prompt_reference_sha256=OLD_SHA)
'''


def compile_context(*,source=None):
    source_guard();source=(ROOT/ANCESTOR).read_bytes() if source is None else source
    if hashlib.sha256(source).hexdigest()!=ANCESTOR_SHA256:raise ValueError('signed native controller SHA mismatch')
    manifest=importlib.import_module('tools.amos_prompt_ablation_native_manifest_eval')
    tree,proof=manifest.transform_controller(source,str(ROOT/ANCESTOR),ANCESTOR_SHA256)
    # Add two explicit policy fields to the native prepare dict, retaining its
    # immutable publication/signature ordering and all original data checks.
    policies=[n for n in ast.walk(tree) if isinstance(n,ast.keyword) and n.arg=='policy' and isinstance(n.value,ast.Call)]
    if len(policies)!=1:raise ValueError('unique native policy anchor required')
    policies[0].value.keywords.extend([ast.keyword(arg='prompt_variant',value=ast.Name(id='PROMPT_VARIANT',ctx=ast.Load())),
        ast.keyword(arg='fresh_module_rows',value=ast.Constant(value=8910))])
    # The completed selected class proof joins the frozen reference inventory.
    refs=[n for n in ast.walk(tree) if isinstance(n,ast.keyword) and n.arg=='reference' and ast.unparse(n.value)=="file_inventory([ROOT / OLD, ROOT / HISTORY / 'manifest.json'] + ref_paths)"]
    if len(refs)!=1:raise ValueError('unique native reference inventory anchor required')
    refs[0].value=ast.parse("file_inventory([ROOT/OLD,ROOT/SELECTED_AUDIT,ROOT/HISTORY/'manifest.json']+ref_paths+list(map(Path,selected_control()['files'])))",mode='eval').body
    # Undo both additions, then use the signed manifest inverse to certify that
    # every other native statement is unchanged.
    reverse=copy.deepcopy(tree)
    for n in ast.walk(reverse):
        if isinstance(n,ast.keyword) and n.arg=='policy' and isinstance(n.value,ast.Call):n.value.keywords=n.value.keywords[:-2]
        if isinstance(n,ast.keyword) and n.arg=='reference':n.value=ast.parse("file_inventory([ROOT/OLD,ROOT/HISTORY/'manifest.json']+ref_paths)",mode='eval').body
    if ast.dump(manifest.reverse_transform(reverse),include_attributes=False)!=ast.dump(ast.parse(source),include_attributes=False):raise ValueError('unexpected module controller AST change')
    private=types.ModuleType('_amos_bg_scribble_controller_private');ns=private.__dict__;ns['__file__']=str(Path(__file__).resolve())
    exec(compile(ast.fix_missing_locations(tree),str(ROOT/ANCESTOR),'exec'),ns)
    ns.update(OUT=OUT,TMUX=TMUX,PIPELINE=PIPELINE,VARIANTS=MODULE_VARIANTS,MODULE_VARIANTS=MODULE_VARIANTS,PROMPT_VARIANT=PROMPT_VARIANT,
        OLD=SELECTED_MERGED,OLD_SHA=SELECTED_SHA256,SELECTED_OUTPUT=SELECTED_OUTPUT,SELECTED_AUDIT=SELECTED_AUDIT,SELECTED_AUDIT_SHA256=SELECTED_AUDIT_SHA256,
        TRANSFORM_PROOF=dict(proof,controller_reverse_ast_equal=True,policy_additions=2,reference_replacements=1),
        runtime=lambda:importlib.import_module('tools.amos_bg_scribble_module_runtime'),_source_guard=source_guard,
        _native_manifest_payload=manifest.native_manifest_payload,
        _native_check_controls=ns['check_controls'],_native_verify=ns['verify'],_native_prepare=ns['prepare'])
    ns['SOURCE_SEEDS']=tuple(dict.fromkeys((*ns['SOURCE_SEEDS'],ANCESTOR,MANIFEST,'tools/amos_module_sensitivity.py',
        'tools/amos_bg_scribble_module_runtime.py','tools/amos_bg_scribble_module_eval.py','tools/amos_prompt_reuse_owner_runtime.py',
        'tools/amos_prompt_reuse_runtime.py','tools/amos_prompt_ablation_restore_runtime.py')))
    # Compile the validated native metadata bootstrap with current globals.
    # Its only changes are the real BGscribble argv and explicitly chosen key.
    bootstrap_source=ast.parse((ROOT/MANIFEST).read_bytes())
    fn=next(n for n in bootstrap_source.body if isinstance(n,ast.FunctionDef) and n.name=='run_native_bootstrap')
    original_bootstrap=copy.deepcopy(fn);changes=[];edits=0
    for n in ast.walk(fn):
        if isinstance(n,ast.Constant) and n.value=='point':
            changes.append((n,'value',n.value));n.value='scribble';edits+=1
        if isinstance(n,ast.Compare) and ast.unparse(n)=="controls[0]['pending'] != [sorted([t for j in jobs for t in j['pending']], key=context['key'])[0]]":
            changes.append((n,'comparators',n.comparators));n.comparators=[ast.parse("[selected_control()['task']]",mode='eval').body];edits+=1
    if edits!=2:raise ValueError('native bootstrap prompt/key anchors changed')
    transformed_bootstrap=copy.deepcopy(fn)
    for node,field,value in reversed(changes):setattr(node,field,value)
    if ast.dump(fn,include_attributes=False)!=ast.dump(original_bootstrap,include_attributes=False):raise ValueError('unexpected bootstrap AST change')
    fn=transformed_bootstrap
    ns['TRANSFORM_PROOF'].update(bootstrap_reverse_ast_equal=True,bootstrap_anchor_replacements=2)
    ns.update(engine=lambda:ns,_sha=manifest._sha,compile_bootstrap=manifest.compile_bootstrap)
    fragment=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
    ns['BOOTSTRAP_SOURCE']=ast.unparse(fragment)
    exec(compile(fragment,str(ROOT/MANIFEST),'exec'),ns)
    ns['_bootstrap_function']=ns.pop('run_native_bootstrap')
    exec(compile(_EXTENSIONS,str(Path(__file__).resolve()),'exec'),ns)
    ns['__doc__']=__doc__
    private.engine=lambda:ns
    return private


def context():
    global _CONTEXT
    if _CONTEXT is None:_CONTEXT=compile_context()
    return _CONTEXT


def main():return context().engine()['main']()


if __name__=='__main__':main()
