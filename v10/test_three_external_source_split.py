import importlib.util
from pathlib import Path
import pytest


def module():
    path=Path(__file__).resolve().parents[1]/'tools/build_three_external_source_split.py'
    assert path.is_file(),'split entry missing'
    spec=importlib.util.spec_from_file_location('split_test',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def fixture():
    counts={'fumpe':35,'graz':40,'pengwin':340,'adrenal':53}
    records={s:[dict(source_name=s,case_id=f'{i:03}') for i in range(n)] for s,n in counts.items()}
    return records,{s:v[:30] for s,v in records.items()}


def test_exact_counts_complete_disjoint_prior_subset_and_fumpe_exclusion():
    m=module();records,old=fixture();d=m.build_split(records,old)
    for s,n in [('graz',20),('pengwin',320),('adrenal',33)]:
        ds=d['datasets'][s];train=set(ds['train']);test=set(ds['test'])
        assert len(train)==n and len(test)==20 and not train&test
        assert train|test=={r['case_id'] for r in records[s]}
        assert test<={r['case_id'] for r in old[s]} and ds['val']==[]
    assert d['datasets']['fumpe']['train']==[]
    assert len(d['datasets']['fumpe']['test'])==30 and len(d['datasets']['fumpe']['unused'])==5


def test_order_independent_and_deterministic():
    m=module();records,old=fixture()
    assert m.build_split(records,old)==m.build_split({s:v[::-1] for s,v in records.items()},{s:v[::-1] for s,v in old.items()})


def test_duplicate_or_missing_old_case_rejected():
    m=module();records,old=fixture();records['graz'].append(records['graz'][0])
    with pytest.raises(ValueError,match='duplicate'):m.build_split(records,old)
    records,old=fixture();old['graz'][0]={**old['graz'][0],'case_id':'not_found'}
    with pytest.raises(ValueError,match='missing'):m.build_split(records,old)


def test_signature_path_alias_allowed_but_size_or_time_change_rejected(tmp_path):
    m=module();records,old=fixture()
    actual=tmp_path/'data'/'ct.nii.gz';alias=tmp_path/'data'/'folder'/'..'/'ct.nii.gz'
    records['fumpe'][0]['source_signatures']=[dict(path=str(actual),size_bytes=123,mtime_ns=456)]
    old['fumpe'][0]={**old['fumpe'][0],'source_signatures':[dict(path=str(alias),size_bytes=123,mtime_ns=456)]}
    m.build_split(records,old)
    old['fumpe'][0]['source_signatures'][0]['mtime_ns']=457
    with pytest.raises(ValueError,match='changed'):m.build_split(records,old)
