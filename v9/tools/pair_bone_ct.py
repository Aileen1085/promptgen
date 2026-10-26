"""Pair audited canonical bone labels with source CT NIfTI geometry, without CT copies."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import zipfile
import nibabel as nib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.prepare_bone_test_data import sha,write_json


def extract_ct(archive,dest):
    archive=Path(archive);dest=Path(dest)
    if dest.is_symlink():raise ValueError('CT destination symlink')
    audit_path=dest/'archive_audit.json';source_sha=sha(archive)
    if dest.exists() and any(dest.iterdir()) and not audit_path.is_file():raise ValueError('unowned CT directory')
    with zipfile.ZipFile(archive) as z:
        entries=[];seen=set()
        for info in z.infolist():
            p=PurePosixPath(info.filename)
            if info.filename=='ct/' and info.is_dir():continue
            if len(p.parts)!=2 or p.parts[0]!='ct' or not re.fullmatch(r'\d{4}\.nii\.gz',p.parts[1]):raise ValueError('unexpected CT archive member')
            if p.name in seen or stat.S_ISLNK(info.external_attr>>16):raise ValueError('duplicate/symlink CT member')
            seen.add(p.name);entries.append((info,p.name))
        if shutil.disk_usage(dest.parent).free<sum(i.file_size for i,_ in entries)+2*1024**3:raise ValueError('CT disk headroom')
        dest.mkdir(parents=True,exist_ok=True);records={}
        for info,name in entries:
            path=dest/name;digest=hashlib.sha256()
            if path.exists() or path.is_symlink():
                if path.is_symlink() or not path.is_file():raise ValueError('unsafe CT target')
                with z.open(info) as src:
                    for b in iter(lambda:src.read(4*1024**2),b''):digest.update(b)
                if sha(path)!=digest.hexdigest():raise ValueError('existing CT changed')
            else:
                with z.open(info) as src,path.open('xb') as out:
                    for b in iter(lambda:src.read(4*1024**2),b''):out.write(b);digest.update(b)
            records[name]=dict(sha256=digest.hexdigest(),size=info.file_size,crc32=info.CRC)
            print('CT_EXTRACT',name,info.file_size,flush=True)
    audit=dict(archive_sha256=source_sha,files=records)
    if audit_path.exists():
        if json.loads(audit_path.read_text())!=audit:raise ValueError('CT audit changed')
    else:write_json(audit_path,audit)
    return audit


def geometry(affine):
    affine=np.asarray(affine)
    if affine.shape!=(4,4) or not np.isfinite(affine).all() or not np.allclose(affine[3],[0,0,0,1]) or abs(np.linalg.det(affine[:3,:3]))<1e-9:raise ValueError('invalid physical affine')
    spacing=nib.affines.voxel_sizes(affine)
    if np.any(spacing<=0):raise ValueError('invalid spacing')
    return spacing


def native_labels(labels_dhw,image):
    geometry(image.affine)
    if len(image.shape)!=3:raise ValueError('CT is not three dimensional')
    native=nib.orientations.io_orientation(image.affine);ras=nib.orientations.axcodes2ornt(('R','A','S'))
    to_ras=nib.orientations.ornt_transform(native,ras)
    shape=tuple(np.asarray(image.shape)[np.argsort(to_ras[:,0])])
    xyz=labels_dhw.transpose(1,2,0)
    if xyz.shape!=shape:raise ValueError('CT canonical shape differs from label export')
    return nib.orientations.apply_orientation(xyz,nib.orientations.ornt_transform(ras,native))


def intensity_audit(data,labels):
    samples=[]
    for z in range(0,data.shape[2],max(1,data.shape[2]//32)):
        frame=np.asanyarray(data[:,:,z]);mask=labels[:,:,z]>0
        if not np.isfinite(frame).all():raise ValueError('nonfinite CT intensities')
        values=frame[mask]
        if len(values):samples.append(values[::max(1,len(values)//2048)])
    if not samples:raise ValueError('empty labeled CT audit samples')
    values=np.concatenate(samples)
    return dict(sample_count=int(len(values)),foreground_hu_median=float(np.median(values)),
                foreground_above150_fraction=float(np.mean(values>150)),
                warning='review_CT_label_pairing' if np.median(values)<0 else None)


def pair(root,ct_root,output):
    root=Path(root).resolve();ct_root=Path(ct_root).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('paired output exists; preserve previous results')
    source=root/'test_ready_v1/manifest.json';labels=json.loads(source.read_text(encoding='utf-8'))
    output.mkdir(parents=True);(output/'labels').mkdir();cases=[];tasks=[];missing=[]
    for row in labels['cases']:
        key=row['case_id']
        if not re.fullmatch(r'\d{4}_phase_\d+',key):raise ValueError('unsafe case key')
        image_path=ct_root/(key.split('_')[0]+'.nii.gz');c=dict(row)
        if not image_path.is_file():
            missing.append(key);c.update(test_ready=False,image=None,label_nifti=None,blockers=['missing_CT'])
            cases.append(c);continue
        image=nib.load(image_path);spacing=geometry(image.affine)
        if image.header.get_xyzt_units()[0] not in ('mm','unknown'):raise ValueError('CT spatial units are not mm')
        q,qcode=image.get_qform(coded=True);s,scode=image.get_sform(coded=True)
        if qcode and scode and not np.allclose(q,s,atol=1e-3,rtol=0):raise ValueError('qform/sform conflict')
        with np.load(root/row['label_npz'],allow_pickle=False) as f:dhw=f['labels_dhw']
        native=native_labels(dhw,image);data=np.asanyarray(image.dataobj)
        hu=intensity_audit(data,native);del data
        target=output/'labels'/(key+'.nii.gz');header=image.header.copy();header.set_data_dtype(np.uint8)
        header.set_slope_inter(1,0);header.set_xyzt_units('mm')
        gt=nib.Nifti1Image(native,image.affine,header);gt.set_sform(image.affine,code=int(scode) or 1)
        if qcode:gt.set_qform(q,code=int(qcode))
        nib.save(gt,target)
        saved=nib.load(target);canonical=nib.as_closest_canonical(saved)
        if not np.array_equal(np.asanyarray(canonical.dataobj).transpose(2,0,1),dhw):raise ValueError('saved GT canonical roundtrip failed')
        if saved.shape!=image.shape or not np.allclose(saved.affine,image.affine,atol=1e-6):raise ValueError('saved CT/GT geometry mismatch')
        ready=hu['warning'] is None
        c.update(image=Path(__import__('os').path.relpath(image_path,root)).as_posix(),image_sha256=sha(image_path),
                 label_nifti=target.relative_to(root).as_posix(),label_nifti_sha256=sha(target),
                 affine_ras_xyz=image.affine.tolist(),spacing_mm=spacing.tolist(),native_shape=list(image.shape),
                 native_axcodes=list(nib.aff2axcodes(image.affine)),test_ready=ready,
                 blockers=[] if ready else [hu['warning']],intensity_audit=hu,geometry_roundtrip_exact=True,
                 pairing_basis='same explicit case ID, canonical RAS shape, documented PNG inverse; CT physical affine retained',
                 phase_provenance='CT filename omits phase; paired to supplied label phase by case ID and matching canonical grid')
        cases.append(c)
        tasks.extend(dict(case_id=key,label_id=k,class_name=next(x['name'] for x in labels['classes'] if x['label_id']==k),
                          image=c['image'],label_nifti=c['label_nifti'],test_ready=ready) for k in row['present_foreground_ids'])
        print('PAIR',key,'ready',ready,'HU',round(hu['foreground_hu_median'],2),'spacing',spacing.tolist(),flush=True)
        del native,dhw,gt,canonical,saved
    result=dict(schema='bone_paired_nifti_v1',test_ready=all(x['test_ready'] for x in cases),cases=cases,
                ready_case_count=sum(x['test_ready'] for x in cases),missing_ct_cases=missing,
                ready_task_count=sum(x['test_ready'] for x in tasks),classes=labels['classes'],
                label_manifest_sha256=sha(source),label_archive_sha256=labels['archive_sha256'],
                mapping_sha256=labels['mapping_sha256'],pairing_source_sha256=sha(Path(__file__)),
                ct_storage='one extracted source NIfTI per case; no CT resampling or class-level copies',
                split_policy='inventory only; no training split or inference performed')
    write_json(output/'manifest.json',result);write_json(output/'tasks.json',dict(tasks=tasks))
    write_json(output/'dataset.json',dict(channel_names={'0':'CT'},labels={x['name']:x['label_id'] for x in labels['classes']},
              file_ending='.nii.gz',test=[dict(case_id=c['case_id'],image=c['image'],label=c['label_nifti']) for c in cases if c['test_ready']]))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',default='../data/ct.zip')
    p.add_argument('--root',default='../data/bone');p.add_argument('--output',default='paired_v1')
    a=p.parse_args();root=(ROOT/a.root).resolve();raw=root/'ct_raw'
    extract_ct(ROOT/a.archive,raw);r=pair(root,raw,root/a.output)
    print('PAIRED',r['ready_case_count'],'cases',r['ready_task_count'],'tasks','MISSING',r['missing_ct_cases'],flush=True)


if __name__=='__main__':main()
