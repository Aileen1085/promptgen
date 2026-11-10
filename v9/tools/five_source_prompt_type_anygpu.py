"""Operational all-GPU overlay; the signed v2 scientific protocol stays intact."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools import five_source_prompt_type_eval_v2 as e
from tools import five_source_prompt_type_runtime as runtime

POLICY_NAME='resource_policy_anygpu_20261110.json'
EXPECTED_PROTOCOL_SHA='8e6f9e238eea2b410245cb3607988bd35b052bc52ee41273b7c220698303377e'
_installed_out=None
_installed_digest=None


def policy_digest(value):
    value={k:v for k,v in value.items() if k!='policy_sha256'}
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')).hexdigest()


def freeze_policy(out,source_commit,authorization,extra_sources=()):
    out=Path(out)
    if e.sha(out/'protocol.json')!=EXPECTED_PROTOCOL_SHA:raise ValueError('original signed protocol SHA mismatch')
    if len(source_commit)!=40 or any(c not in '0123456789abcdef' for c in source_commit):raise ValueError('full published source commit required')
    if not authorization.strip():raise ValueError('explicit user resource authorization required')
    extra={}
    for relative in extra_sources:
        path=Path(relative)
        if path.is_absolute() or '..' in path.parts:raise ValueError('relative project source required')
        extra[path.as_posix()]=e.sha(ROOT/path)
    policy=dict(schema='five_source_prompt_type_resource_policy_anygpu_v1',protocol_sha256=EXPECTED_PROTOCOL_SHA,
        allowed_physical_gpus=list(range(8)),min_free_mib=30720,max_workers=4,
        source_path='tools/five_source_prompt_type_anygpu.py',source_sha256=e.sha(Path(__file__)),
        source_commit=source_commit,user_authorization=authorization,extra_source_sha256=extra)
    policy['policy_sha256']=policy_digest(policy)
    e.atomic_new(out/POLICY_NAME,policy)
    return policy


def read_policy(out):
    out=Path(out);policy=e.read(out/POLICY_NAME)
    if (policy.get('schema'),policy.get('protocol_sha256'),policy.get('allowed_physical_gpus'),
        policy.get('min_free_mib'),policy.get('max_workers'),policy.get('source_path'))!=(
        'five_source_prompt_type_resource_policy_anygpu_v1',EXPECTED_PROTOCOL_SHA,list(range(8)),30720,4,
        'tools/five_source_prompt_type_anygpu.py'):
        raise ValueError('resource policy scope mismatch')
    if e.sha(out/'protocol.json')!=EXPECTED_PROTOCOL_SHA:raise ValueError('original signed protocol SHA mismatch')
    if policy.get('policy_sha256')!=policy_digest(policy):raise ValueError('resource policy digest mismatch')
    if policy.get('source_sha256')!=e.sha(Path(__file__)):raise ValueError('resource overlay source SHA mismatch')
    commit=policy.get('source_commit','')
    if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit) or not policy.get('user_authorization','').strip():
        raise ValueError('resource authorization/commit missing')
    for relative,digest in policy.get('extra_source_sha256',{}).items():
        path=Path(relative)
        if path.is_absolute() or '..' in path.parts or e.sha(ROOT/path)!=digest:raise ValueError('resource dispatcher source SHA mismatch')
    if _installed_out==out.absolute() and _installed_digest!=policy['policy_sha256']:
        raise ValueError('installed resource policy changed')
    return policy


def resource_snapshot(text,gpu,expected_uuid):
    if _installed_out is not None:read_policy(_installed_out)
    rows={int(p[0]):p for p in ([v.strip() for v in line.split(',')] for line in text.splitlines() if line.strip())}
    if gpu not in range(8) or gpu not in rows or rows[gpu][1]!=expected_uuid:
        raise ValueError('physical GPU identity/UUID mismatch')
    if int(rows[gpu][2])<30720:raise ValueError('record GPU free-memory requirement failed')
    return dict(gpu=gpu,gpu_uuid=expected_uuid,free_mib=int(rows[gpu][2]),minimum_free_mib=30720)


def bind_resources(job,out):
    policy=read_policy(out)
    if e.shutil.disk_usage(out).free<3*1024**3:raise SystemExit(75)
    gpu=int(job['gpu']);minimum=max(policy['min_free_mib'],int(job.get('minimum_free_mib',30720)))
    def query():
        text=e.subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
        rows={int(p[0]):p for p in ([v.strip() for v in line.split(',')] for line in text.splitlines() if line.strip())}
        if gpu not in policy['allowed_physical_gpus'] or gpu not in rows or rows[gpu][1]!=job['gpu_uuid']:
            raise ValueError('physical GPU identity/UUID mismatch')
        if int(rows[gpu][2])<minimum:raise SystemExit(75)
        return resource_snapshot(text,gpu,job['gpu_uuid'])
    first=query()
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu);os.environ['V9_SHARED_CACHE_WRITE_POLICY']='read_only'
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
    import torch
    raw_uuid=str(torch.cuda.get_device_properties(0).uuid)
    visible=raw_uuid if raw_uuid.startswith('GPU-') else 'GPU-'+raw_uuid
    if visible!=job['gpu_uuid'] or torch.ones(1,device='cuda:0').item()!=1:raise ValueError('CUDA UUID/probe mismatch')
    torch.set_num_threads(2);second=query()
    read_policy(out)
    return dict(first_snapshot=first,second_snapshot=second,gpu=gpu,gpu_uuid=visible,
        free_mib_before_load=second['free_mib'],minimum_free_mib=minimum,cuda_probe=True,pid=os.getpid(),
        resource_policy_sha256=policy['policy_sha256'],resource_policy_path=POLICY_NAME,
        resource_policy_source_sha256=policy['source_sha256'])


def install(out):
    global _installed_out,_installed_digest
    policy=read_policy(out)
    _installed_out=Path(out).absolute();_installed_digest=policy['policy_sha256']
    runtime.resource_snapshot=resource_snapshot
    e.bind_resources=bind_resources
    return policy


def validate_binding(binding,out):
    policy=read_policy(out);gpu=binding.get('gpu')
    if gpu not in policy['allowed_physical_gpus']:raise ValueError('record GPU outside resource policy')
    if 'resource_policy_sha256' not in binding:
        if gpu not in (0,3):raise ValueError('new GPU requires resource policy binding')
        return 'historical_protocol'
    if (binding.get('resource_policy_sha256'),binding.get('resource_policy_path'),binding.get('resource_policy_source_sha256'))!=(
        policy['policy_sha256'],POLICY_NAME,policy['source_sha256']):raise ValueError('record resource policy binding mismatch')
    if binding.get('cuda_probe') is not True:raise ValueError('CUDA probe missing')
    minimum=max(30720,int(binding['minimum_free_mib']))
    for name in ('first_snapshot','second_snapshot'):
        row=binding[name]
        if row['gpu']!=gpu or row['gpu_uuid']!=binding['gpu_uuid'] or row['free_mib']<minimum:
            raise ValueError('resource policy snapshot/identity/minimum mismatch')
    if binding['free_mib_before_load']!=binding['second_snapshot']['free_mib']:raise ValueError('load memory snapshot mismatch')
    return 'resource_policy'


def verify_records(out):
    out=Path(out);policy=read_policy(out);counts={'historical_protocol':0,'resource_policy':0}
    roots=[out/'records']+[path/'control_records' for path in out.glob('control_batch_*') if path.is_dir()]
    for root in roots:
        for path in sorted(root.rglob('*.json')):
            record=e.read(path)
            counts[validate_binding(record['resource_binding'],out)]+=1
    read_policy(out)
    return dict(passed=True,protocol_sha256=policy['protocol_sha256'],resource_policy_sha256=policy['policy_sha256'],
                source_sha256=policy['source_sha256'],extra_source_sha256=policy['extra_source_sha256'],records=counts)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('freeze-policy','control','worker','status','merge','audit-controls','audit-final'))
    parser.add_argument('--out',required=True,type=Path);parser.add_argument('--job',type=Path)
    parser.add_argument('--source-commit');parser.add_argument('--authorization');parser.add_argument('--extra-source',action='append',default=[])
    args=parser.parse_args(argv);out=args.out.absolute()
    if args.mode=='freeze-policy':
        if not args.source_commit or not args.authorization:parser.error('freeze-policy requires --source-commit and --authorization')
        print(json.dumps(freeze_policy(out,args.source_commit,args.authorization,args.extra_source)));return
    install(out)
    if args.mode not in ('control','worker'):verify_records(out)
    if args.mode.startswith('audit-'):
        from tools import five_source_prompt_type_pipeline_v2 as pipeline
        (pipeline.controls if args.mode=='audit-controls' else pipeline.final)(out)
        proof=verify_records(out)
        e.atomic_new(out/('independent_anygpu_'+args.mode[6:]+'_policy_audit.json'),proof)
    else:
        arguments=[args.mode,'--out',str(out)]
        if args.job is not None:arguments+=['--job',str(args.job)]
        e.main(arguments)
        if args.mode in ('control','worker'):read_policy(out)
        else:verify_records(out)


if __name__=='__main__':main()
