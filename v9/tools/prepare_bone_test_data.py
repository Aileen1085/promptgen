"""Safely extract bone PNG labels; audit real classes without inventing CT geometry."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import zipfile
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')


def parse_mapping(text):
    result={};colors=set()
    for line in text.splitlines():
        m=re.fullmatch(r'\s*(\d+)\s*\|\s*(.+?)\s*\|\s*\((\d+),(\d+),(\d+)\)\s*',line)
        if not m:
            if re.match(r'\s*\d+\s*\|',line):raise ValueError('invalid mapping row')
            continue
        key=int(m[1]);rgb=tuple(map(int,m.group(3,4,5)))
        if key>255 or key in result or rgb in colors or any(x>255 for x in rgb):raise ValueError('duplicate/invalid ID or RGB')
        result[key]=dict(label_id=key,name=m[2],rgb=list(rgb));colors.add(rgb)
    if not result or 0 not in result or result[0]['rgb']!=[0,0,0]:raise ValueError('missing black background')
    return result


def safe_relative(name):
    name=name.replace('\\','/');p=PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or ':' in name or '\x00' in name:raise ValueError('unsafe ZIP path')
    if p.parts[0]!='bone':raise ValueError('unexpected archive root')
    return Path(*p.parts[1:])


def extract_archive(archive,dest):
    archive=Path(archive).resolve();dest=Path(dest).absolute()
    if dest.is_symlink():raise ValueError('destination symlink')
    source_sha=sha(archive);audit_path=dest/'archive_audit.json'
    if dest.exists() and any(dest.iterdir()) and not audit_path.is_file():raise ValueError('unowned nonempty output')
    with zipfile.ZipFile(archive) as z:
        entries=[];seen=set()
        for i in z.infolist():
            rel=safe_relative(i.filename);key=rel.as_posix()
            if key in seen:raise ValueError('duplicate normalized archive member')
            if stat.S_ISLNK(i.external_attr>>16):raise ValueError('archive symlink')
            seen.add(key);entries.append((i,rel))
        if shutil.disk_usage(dest.parent).free<sum(i.file_size for i,_ in entries)+2*1024**3:raise ValueError('insufficient extraction headroom')
        dest.mkdir(parents=True,exist_ok=True);records={};resolved=dest.resolve()
        directories={rel if i.is_dir() else rel.parent for i,rel in entries}
        directories|={p for d in list(directories) for p in d.parents}
        for rel in sorted(directories,key=lambda p:(len(p.parts),str(p))):
            path=dest/rel
            path.mkdir(parents=True,exist_ok=True)
            if path.is_symlink() or not path.resolve().is_relative_to(resolved):raise ValueError('unsafe extraction directory')
        for index,(i,rel) in enumerate(entries):
            path=dest/rel
            if i.is_dir():continue
            content=z.read(i);digest=hashlib.sha256(content).hexdigest()
            try:
                with path.open('xb') as f:f.write(content)
            except FileExistsError:
                if path.is_symlink() or not path.is_file() or sha(path)!=digest:raise ValueError('existing extracted file changed: '+str(rel))
            records[rel.as_posix()]=dict(sha256=digest,size=i.file_size,crc32=i.CRC)
            if index%5000==0:print('EXTRACT',index,'/',len(entries),flush=True)
    audit=dict(archive_sha256=source_sha,archive_size=archive.stat().st_size,files=records)
    if audit_path.exists():
        if json.loads(audit_path.read_text())!=audit:raise ValueError('archive audit changed')
    else:write_json(audit_path,audit)
    return audit


def restore_slice(ids):
    return np.flipud(ids).T


def deduplicate(root,signatures):
    groups={};aliases=[]
    for f in sorted(root.rglob('index.json')):
        d=json.loads(f.read_text(encoding='utf-8-sig'));key=(d['case'],d['phase'])
        if not re.fullmatch(r'\d+',str(key[0])) or not re.fullmatch(r'phase_\d+',key[1]):raise ValueError('unsafe case identity')
        groups.setdefault(key,[]).append((f.parent,d))
    selected=[]
    for key,parts in sorted(groups.items()):
        parts.sort(key=lambda x:(x[0].name!=str(key[0]),str(x[0])))
        def identity(part):
            folder,d=part
            files=sorted((kind,p.name,signatures[p.relative_to(root).as_posix()]['sha256'])
                         for kind in ('label_id_png','color_png') for p in (folder/kind).glob('*.png'))
            return d['canonical_shape'],d['source_sha256'],files
        base=identity(parts[0])
        for part in parts[1:]:
            if identity(part)!=base:raise ValueError('conflicting duplicate case '+str(key))
            aliases.append(dict(case_id='_'.join(key),alias=part[0].relative_to(root).as_posix(),
                                canonical=parts[0][0].relative_to(root).as_posix(),all_png_sha256_equal=True))
        selected.append(parts[0])
    return selected,aliases


def scan_case(folder,meta,mapping):
    shape=tuple(meta['canonical_shape']);depth=int(meta['slice_count'])
    if len(shape)!=3 or shape[2]!=depth or meta['canonical_axcodes']!=['R','A','S'] or meta['slice_axis']!=2:raise ValueError('unsupported axes/shape')
    if 'transpose; vertical flip' not in meta['png_orientation']:raise ValueError('unknown PNG export transform')
    expected={f'{i:04d}.png' for i in range(depth)}
    for kind in ('label_id_png','color_png'):
        if {p.name for p in (folder/kind).glob('*.png')}!=expected:raise ValueError('missing/extra slices')
    lut=np.zeros((256,3),np.uint8);allowed=np.zeros(256,bool)
    for k,row in mapping.items():lut[k]=row['rgb'];allowed[k]=True
    volume=np.empty((depth,shape[0],shape[1]),np.uint8);counts=np.zeros(256,np.int64)
    for z in range(depth):
        name=f'{z:04d}.png'
        with Image.open(folder/'label_id_png'/name) as f:ids=np.asarray(f).copy()
        with Image.open(folder/'color_png'/name) as f:rgb=np.asarray(f).copy()
        if ids.dtype!=np.uint8 or ids.shape!=(shape[1],shape[0]) or rgb.shape!=ids.shape+(3,):raise ValueError('PNG dtype/shape')
        if not allowed[ids].all() or not np.array_equal(rgb,lut[ids]):raise ValueError('unknown ID or RGB/ID pixel mismatch')
        counts+=np.bincount(ids.ravel(),minlength=256)
        volume[z]=restore_slice(ids)
    actual={str(k):int(v) for k,v in enumerate(counts) if v}
    if actual!={str(k):int(v) for k,v in meta['label_counts'].items() if int(v)} or sorted(map(int,actual))!=sorted(meta['ids']):raise ValueError('actual voxels/index metadata mismatch')
    return volume,counts


def prepare(archive,output):
    output=Path(output).resolve();ready=output/'test_ready_v1'
    if ready.exists():raise ValueError('derived output exists; do not overwrite')
    audit=extract_archive(archive,output);mapping=parse_mapping((output/'mapping.txt').read_text(encoding='utf-8-sig'))
    selected,aliases=deduplicate(output,audit['files']);cases=[];tasks=[];total=np.zeros(256,np.int64);presence=np.zeros(256,np.int64)
    if not selected:raise ValueError('no case indices')
    for folder,meta in selected:
        key=meta['case']+'_'+meta['phase'];volume,counts=scan_case(folder,meta,mapping)
        target=ready/'labels'/(key+'.npz');target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:np.savez_compressed(f,labels_dhw=volume)
        total+=counts;presence+=(counts>0);ids=[k for k in sorted(mapping) if k and counts[k]]
        row=dict(case_id=key,source_index=folder.relative_to(output).as_posix()+'/index.json',
                 label_npz=target.relative_to(output).as_posix(),label_npz_sha256=sha(target),label_key='labels_dhw',
                 shape_dhw=list(volume.shape),canonical_axes_xyz=['R','A','S'],stored_axes='ZXY',
                 image=None,affine_ras_xyz=None,spacing_mm=None,test_ready=False,
                 blockers=['missing_corresponding_CT','missing_physical_affine_and_spacing'],present_foreground_ids=ids,
                 label_counts={str(k):int(v) for k,v in enumerate(counts) if v})
        cases.append(row)
        tasks.extend(dict(case_id=key,label_id=k,class_name=mapping[k]['name'],image=None,label_npz=row['label_npz'],test_ready=False) for k in ids)
        print('CASE',key,'shape',list(volume.shape),'present_FG',len(ids),flush=True)
        del volume
    classes=[dict(v,voxel_count=int(total[k]),case_count=int(presence[k])) for k,v in sorted(mapping.items())]
    result=dict(schema='bone_index_space_labels_v1',test_ready=False,archive_sha256=audit['archive_sha256'],
                mapping_sha256=sha(output/'mapping.txt'),classes=classes,cases=cases,duplicate_aliases=aliases,
                present_foreground_ids=[k for k in sorted(mapping) if k and total[k]],
                absent_foreground_ids=[k for k in sorted(mapping) if k and not total[k]],
                task_count=len(tasks),split_policy='all_unique_cases_inventory; no_training_split_created',
                geometry_policy='invert_documented_png_flip_transpose; no_fake_affine_or_1mm_spacing',
                preparation_source_sha256=sha(Path(__file__)))
    write_json(ready/'manifest.json',result);write_json(ready/'tasks.json',dict(tasks=tasks))
    write_json(ready/'class_inventory.json',dict(classes=classes,present_foreground_ids=result['present_foreground_ids'],absent_foreground_ids=result['absent_foreground_ids']))
    with (ready/'classes.tsv').open('x',encoding='utf-8') as f:
        f.write('label_id\tname\trgb\tvoxel_count\tcase_count\n')
        for row in classes:f.write('\t'.join(str(row[k]) for k in ('label_id','name','rgb','voxel_count','case_count'))+'\n')
    with (ready/'README.md').open('x',encoding='utf-8') as f:
        f.write('# Bone label preparation\n\n'+str(len(cases))+' unique labeled cases; '+str(len(result['present_foreground_ids']))+' actual foreground classes.\n\n')
        f.write('Labels are uint8 labels_dhw in compressed NPZ, canonical index order Z,X,Y. Original IDs/names/RGB are retained in classes.tsv. PNG inverse is flipud(png).T, not a guessed image flip.\n\n')
        f.write('NOT READY FOR MODEL TESTING: archive contains no CT image or physical affine/spacing. image/affine/spacing are null in manifest; do not substitute RGB masks, fake CT, identity affine or 1mm voxels. Supply the matching CT and trusted physical geometry before NIfTI pairing and NSD testing. No training split or model inference was run.\n')
    print('PREPARED_LABELS',len(cases),'cases',len(result['present_foreground_ids']),'actual_FG','CT_MISSING',flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',default='../data/bone.zip');p.add_argument('--output',default='../data/bone')
    a=p.parse_args();prepare(ROOT/a.archive,ROOT/a.output)


if __name__=='__main__':main()
