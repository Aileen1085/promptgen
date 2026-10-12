"""Freeze Graz/PENGWIN/Adrenal test20 cases; remainder train, FUMPE excluded."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
SOURCES=('fumpe','graz','pengwin','adrenal')
TRAIN_SOURCES=('graz','pengwin','adrenal')
DIRS={'fumpe':'FUMPE','graz':'Aortic-Dissection-Graz/dataset','pengwin':'PENGWIN2026/data','adrenal':'Adrenal-ACC-Ki67-Seg'}
OUT='configs/graz_pengwin_adrenal_test20_train_rest_v1_20261012.json'
PREVIOUS='output/four_source_v102e490_nninteractive_30case_20261011/selected_cases.json'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def signature_identity(rows):
    if rows is None:return None
    return sorted((str((ROOT/Path(r['path'])).resolve()),int(r['size_bytes']),int(r['mtime_ns'])) for r in rows)


def build_split(records,previous,test_count=20,seed=20261012):
    if int(test_count)<=0:raise ValueError('positive test count required')
    result=dict(schema='three_external_sources_patient_train_test_v1',seed=int(seed),test_count_per_train_source=int(test_count),
        selection='sha256_rank_within_previous30_no_metric_selection',split_unit='native_case_id_all_slices_fragments_together',
        validation_policy='unassigned; derive validation from train only, never select checkpoints on test',
        test_previously_evaluated=True,datasets={})
    for source in SOURCES:
        rows=records[source];old=previous[source]
        index={str(r['case_id']):r for r in rows};prior={str(r['case_id']):r for r in old}
        if len(index)!=len(rows) or len(prior)!=len(old):raise ValueError('duplicate case IDs: '+source)
        if len(prior)!=30:raise ValueError('previous test must have30 cases: '+source)
        if not set(prior).issubset(index):raise ValueError('missing prior case: '+source)
        if any(r['source_name']!=source for r in rows+old):raise ValueError('source mismatch')
        for case_id,r in prior.items():
            if 'source_signatures' in r and signature_identity(r['source_signatures'])!=signature_identity(index[case_id].get('source_signatures')):
                raise ValueError('prior source files changed: '+source+'/'+case_id)
        all_ids=set(index)
        if source in TRAIN_SOURCES:
            if len(prior)<test_count or len(all_ids)<=test_count:raise ValueError('insufficient paired cases')
            ranked=sorted(prior,key=lambda c:hashlib.sha256(f'{seed}:{source}:{c}'.encode()).hexdigest())
            test=set(ranked[:test_count]);train=all_ids-test;unused=set()
        else:test=set(prior);train=set();unused=all_ids-test
        if train&test or train|test|unused!=all_ids:raise ValueError('invalid split coverage')
        result['datasets'][source]=dict(relative_dataset_dir=DIRS[source],total_cases=len(all_ids),
            training_enabled=source in TRAIN_SOURCES,train=sorted(train),val=[],test=sorted(test),unused=sorted(unused),
            previous_test=sorted(prior),previous_test_moved_to_train=sorted(train&set(prior)),
            train_test_overlap_count=0,inventory_sha256=digest(sorted(rows,key=lambda r:str(r['case_id']))))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',default='../data')
    p.add_argument('--previous-selection',default=PREVIOUS);p.add_argument('--out',default=OUT)
    p.add_argument('--test-count',type=int,default=20);p.add_argument('--seed',type=int,default=20261012)
    a=p.parse_args();os.chdir(ROOT)
    for folder in (ROOT/'tools',ROOT/'third_party/four_source_eval_deps'):
        sys.path.insert(0,str(folder))
    from four_source_data import discover
    source=Path(a.previous_selection).resolve();data=Path(a.data_root).resolve();out=Path(a.out).resolve()
    if ROOT not in out.parents:raise ValueError('split output must stay under project root')
    old=json.loads(source.read_text(encoding='utf-8'));records=discover(data)
    value=build_split(records,old,a.test_count,a.seed)
    value.update(path_base='project_root',data_root=Path(os.path.relpath(data,ROOT)).as_posix(),
        previous_selection=Path(os.path.relpath(source,ROOT)).as_posix(),previous_selection_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        split_script='tools/build_three_external_source_split.py',split_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        native_reader='tools/four_source_data.py',native_reader_sha256=hashlib.sha256((ROOT/'tools/four_source_data.py').read_bytes()).hexdigest(),
        shared_case_ct_cache='v10/.cache/totalseg_sam2_promptgen_v10_prompt_roi_full_v1',cache_policy='reuse_existing_no_CT_export')
    if out.exists():
        if json.loads(out.read_text(encoding='utf-8'))!=value:raise ValueError('different frozen split exists; use a new filename')
    else:
        out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('x',encoding='utf-8') as handle:handle.write(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print(str(out),flush=True)
    for source,ds in value['datasets'].items():
        print(source,'total',ds['total_cases'],'train',len(ds['train']),'test',len(ds['test']),'unused',len(ds['unused']),flush=True)


if __name__=='__main__':main()
