"""Fresh offline AMOS controls with the actual forward_image owner guard.

Reuse the verified natural-drain certificate and complete native classes. No
capture, scheduler retirement, worker signal or failed control result is reused.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
ANCESTOR='tools/amos_prompt_reuse_resume_eval.py'
ANCESTOR_SHA256='a5205f0ab23f501cf2e21060289e019cc944eb6608728cd1eabd26aa0bccd6e4'
RUNTIME_ANCESTOR='tools/amos_prompt_reuse_runtime.py'
RUNTIME_ANCESTOR_SHA256='bd3f4c1274088edc8fea1cfc0c908d9a4e1b436471a85d2e2400be9f56d9725a'
OWNER_RUNTIME='tools/amos_prompt_reuse_owner_runtime.py'
OWNER_RUNTIME_SHA256='dcf768949161021469138e02b5ba2894b09e30f98ae59caf928d27ec4cb89c67'
FAILURE='output/amos_reuse_encoder_owner_failure_20261031.json'
FAILURE_SHA256='71d7a30a2cfba98aac4a3d0291ea4fc172c9cf36660574924cc795de65a44b6c'
PROBE='output/amos_encoder_owner_probe_20261031/owner_probe.json'
PROBE_SHA256='f10b24055fe3920c06d2a2c62d6df09a3010d848ff5e88c898a2ec64ccfc3f92'
PROBE_INVENTORY='output/amos_encoder_owner_probe_inventory_20261031.json'
PROBE_INVENTORY_SHA256='704b09121d08d263645afb199d3e16cd03b26a8b7855f3e57e7f45005abbcc5e'
FAILED_PROTOCOL_SHA256='d0b4f41610cea5c567e36e7d593567048a95f3acd5da9330e71bb84116806454'
FAILED_OUT='output/amos_v102_e490_floor070_prompt_ablations_20261030_reusefast_procfix'
OUT='output/amos_v102_e490_floor070_prompt_ablations_20261031_reusefast_ownerfix'
TMUX='amos_v102_floor070_prompt_ablations_reusefast_ownerfix'
PIPELINE='output/amos_prompt_ablations_reusefast_ownerfix_pipeline_20261031.log'
_CONTEXT=None


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def source_guard():
    for name,expected in ((ANCESTOR,ANCESTOR_SHA256),(RUNTIME_ANCESTOR,RUNTIME_ANCESTOR_SHA256),(OWNER_RUNTIME,OWNER_RUNTIME_SHA256)):
        if sha(ROOT/name)!=expected:raise ValueError('owner-fix signed source SHA changed: '+name)


def verify_failure(context):
    """The failed controls contributed zero new class or native-control rows."""
    if sha(ROOT/FAILURE)!=FAILURE_SHA256:raise ValueError('owner failure evidence SHA changed')
    value=read(ROOT/FAILURE);c=context.__dict__;out=ROOT/FAILED_OUT
    if (value.get('kind')!='amos_reuse_encoder_owner_failure_v1' or value.get('output')!=FAILED_OUT
            or value.get('source_sha256')!=ANCESTOR_SHA256 or value.get('runtime_sha256')!=RUNTIME_ANCESTOR_SHA256
            or value.get('related_checks')!=[[],[]] or value.get('audited_reused_classes')!=680
            or value.get('audited_reused_rows')!=4080 or value.get('new_classes')!=0 or value.get('new_rows')!=0
            or value.get('controls_passed') is not False or value.get('independent_gate_exists') is not False
            or value.get('merged_exists') is not False):raise ValueError('owner failure evidence scope/source changed')
    files=value.get('files',{})
    required={FAILED_OUT+'/'+name for name in ('protocol.json','protocol_signature.json','reuse_audit.json','jobs.json','status.json','native_bootstrap_audit.json')}
    if not isinstance(files,dict) or not required<=set(files):raise ValueError('owner failure omitted mandatory file inventory')
    c['verify_inventory'](files)
    status=read(out/'status.json')
    if (status!=value['status'] or status.get('phase')!='controls' or status.get('pending')!=0 or status.get('running')
            or len(status.get('failed',[]))!=1 or status['failed'][0].get('name')!='amos_native_control'
            or status['failed'][0].get('returncode')!=1):raise ValueError('owner failure is not the single empty control failure')
    log=Path(value['failed_log'])
    if not log.is_absolute():log=ROOT/log
    rel=c['os'].path.relpath(log,ROOT).replace('\\','/')
    if rel not in files or read_text(log)!=value['failed_log_text']:raise ValueError('owner failed control log changed')
    if (out/'control_verification.json').exists() or (out/'merged_metrics.json').exists() or any((out/'case_success').glob('*.json')):
        raise ValueError('failed owner controls unexpectedly acquired success/gate/merge')
    ns=c['engine']();ns['native_verify'](out)
    c['_owner_ancestor_verify_reuse'](out)
    if ns['native_class_records'](out,False) or ns['native_class_records'](out,True):
        raise ValueError('failed owner controls unexpectedly committed new classes')
    found=c['verify_parent']()
    if len(found)!=680 or c['_record_index'](found)!=read(out/'reuse_audit.json')['records']:
        raise ValueError('frozen680 complete native classes changed')
    verify_probe(context)
    return value


def read_text(path):return Path(path).read_text(encoding='utf-8',errors='replace')


def verify_probe(context):
    if sha(ROOT/PROBE)!=PROBE_SHA256 or sha(ROOT/PROBE_INVENTORY)!=PROBE_INVENTORY_SHA256:
        raise ValueError('actual owner diagnostic probe SHA changed')
    inventory=read(ROOT/PROBE_INVENTORY);probe=read(ROOT/PROBE);folder=str(Path(PROBE).parent).replace('\\','/')
    expected={folder+'/'+name for name in ('spec.json','prepared/manifest.json','prepared/refs.json','owner_probe.json')}
    if (inventory.get('kind')!='amos_encoder_owner_probe_inventory_v1' or inventory.get('diagnostic_only') is not True
            or inventory.get('predictions')!=0 or set(inventory.get('files',{}))!=expected):
        raise ValueError('diagnostic probe inventory scope changed')
    context.verify_inventory(inventory['files'])
    if (probe.get('kind')!='amos_actual_encoder_owner_probe_v1' or probe.get('predictions')!=0
            or probe.get('outer_class')!='peft.peft_model.PeftModelForFeatureExtraction'
            or probe.get('owner_class')!='sam2.modeling.sam2_base.SAM2Base' or probe.get('same_object') is not False
            or probe.get('forward_qualname')!='SAM2Base.forward_image'
            or probe.get('outer_decoder')!='SAM2MaskDecoderWith3DPromptAdapterV10' or probe.get('owner_decoder')!='MaskDecoder'
            or probe.get('outer_highres') is not True or probe.get('owner_highres') is not True
            or probe.get('owner_convs')!={'conv_s0':True,'conv_s1':True} or probe.get('outer_convs')!={'conv_s0':False,'conv_s1':False}
            or probe.get('protocol_sha256')!=FAILED_PROTOCOL_SHA256 or sha(ROOT/FAILED_OUT/'protocol.json')!=FAILED_PROTOCOL_SHA256):
        raise ValueError('actual forward_image owner diagnostic changed')
    fields=('model','epoch','weight_mode','prompt_parameters_exact','decoder_parameters_exact','guards_passed','model_load_count')
    target=('v10_2',490,'live',418,212,True,1)
    if (tuple(probe.get('restoration',{}).get(f) for f in fields)!=target or tuple(probe.get('finish',{}).get(f) for f in fields)!=target
            or probe['finish'].get('successful_native_completion') is not True
            or probe['finish'].get('full_tensor_comparison_phase')!='normal_worker_completion'):
        raise ValueError('owner diagnostic normal exact restoration missing')
    resource=probe.get('resource',{})
    if (resource.get('gpu')!=2 or resource.get('visible_cuda_ordinal')!=0 or not str(resource.get('gpu_uuid','')).startswith('GPU-')
            or resource.get('minimum_free_mib')!=30720 or resource.get('free_mib_before_load',0)<30720):
        raise ValueError('owner diagnostic physical GPU resource binding changed')
    source=ROOT/'sam2/sam2/modeling/sam2_base.py'
    if Path(probe.get('forward_source','')).resolve()!=source.resolve():raise ValueError('owner diagnostic forward source path changed')
    protocol=read(ROOT/FAILED_OUT/'protocol.json')
    if protocol['sources'].get('sam2/sam2/modeling/sam2_base.py')!=sha(source):raise ValueError('owner diagnostic forward source SHA changed')
    return inventory


def compile_context(*,source=None):
    payload=(ROOT/ANCESTOR).read_bytes() if source is None else source
    if hashlib.sha256(payload).hexdigest()!=ANCESTOR_SHA256:raise ValueError('signed procfix controller SHA changed')
    ancestor=types.ModuleType('_amos_prompt_reuse_owner_ancestor');a=ancestor.__dict__;a['__file__']=str(Path(__file__).resolve())
    exec(compile(payload,str(ROOT/ANCESTOR),'exec'),a)
    context=a['compile_context']();c=context.__dict__
    original_guard=c['source_guard'];original_verify=c['verify_reuse'];original_audit=c['build_reuse_audit']
    original_prepare=c['prepare_guard'];original_engine=c['engine']
    c.update(OUT=OUT,TMUX=TMUX,PIPELINE=PIPELINE,ENTRIES=c['ENTRIES']|{'amos_prompt_reuse_owner_eval.py'},
             _runtime=lambda:importlib.import_module('tools.amos_prompt_reuse_owner_runtime'),
             _owner_ancestor_verify_reuse=original_verify)
    def guard():source_guard();original_guard()
    def offline_only(*args,**kwargs):raise ValueError('offline owner-fix cannot capture, signal or drain processes')
    def prepare_guard():verify_failure(context);return original_prepare()
    def verify(out):verify_failure(context);return original_verify(out)
    def audit():
        failure=verify_failure(context);probe=verify_probe(context);value=original_audit()
        value['files'].update(failure['files']);value['files'].update(c['inventory']([ROOT/FAILURE]))
        value['files'].update(probe['files']);value['files'].update(c['inventory']([ROOT/PROBE_INVENTORY]))
        value['encoder_owner_recovery']=dict(evidence_sha256=FAILURE_SHA256,failed_output=FAILED_OUT,
             failed_protocol_sha256=sha(ROOT/FAILED_OUT/'protocol.json'),failed_control_rows_reused=0,
             audited_reused_classes=680,audited_reused_rows=4080,diagnostic_probe_sha256=PROBE_SHA256,diagnostic_predictions=0)
        return value
    def engine():
        ns=original_engine();seeds=['tools/amos_prompt_reuse_owner_eval.py',OWNER_RUNTIME,ANCESTOR,FAILURE,PROBE,PROBE_INVENTORY]
        if (ROOT/FAILURE).exists():seeds.extend(read(ROOT/FAILURE)['files'])
        if (ROOT/PROBE_INVENTORY).exists():seeds.extend(read(ROOT/PROBE_INVENTORY)['files'])
        ns['SOURCE_SEEDS']=tuple(dict.fromkeys((*ns['SOURCE_SEEDS'],*seeds)))
        return ns
    c.update(source_guard=guard,prepare_guard=prepare_guard,verify_reuse=verify,build_reuse_audit=audit,engine=engine,
             signal_exact=offline_only,freeze_parent=offline_only,finish_parent=offline_only)
    return context


def context():
    global _CONTEXT
    if _CONTEXT is None:_CONTEXT=compile_context()
    return _CONTEXT


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('run','worker','status','merge'));p.add_argument('--job')
    args=p.parse_args();c=context()
    if args.mode=='run':return c.engine()['run_pipeline'](ROOT/OUT)
    return c.engine()['main']()


if __name__=='__main__':main()
