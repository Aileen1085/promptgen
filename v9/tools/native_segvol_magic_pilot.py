"""SegVol MAGIC held-out two-per-class physical prompt pilot; reuses AMOS runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def select_magic(tasks):
    groups={}
    for t in tasks:
        if t['source_name']=='magic':
            groups.setdefault(t['global_class_id'],{})[t['case_id']]=t
    if not groups or any(len(g)<2 for g in groups.values()):
        raise ValueError('MAGIC requires two distinct held-out cases per class')
    return [groups[k][c] for k in sorted(groups) for c in sorted(groups[k])[:2]]

def main():
    from tools import native_segvol_pilot as seg
    n=seg.native
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--arm',choices=tuple(seg.ARMS));p.add_argument('--uuid');p.add_argument('--limit',type=int)
    a=p.parse_args()
    seg.OUTPUT=ROOT/'output/native_segvol_magic_physical_pilot_20261007'
    manifest=seg.OUTPUT/'manifest.json'
    if a.prepare:
        tasks=select_magic(n.read(ROOT/'output/native_sam2_medsam2_ct13_full6094_20261007_gridfix/manifest.json')['tasks'])
        split=n.read(ROOT/'configs/ct13_v10_2_split_20260928.json')['datasets']['magic']
        val={r['case_id'] for r in split['val']};train={r['case_id'] for r in split['train']}
        if any(t['case_id'] not in val or t['case_id'] in train for t in tasks):raise ValueError('val membership')
        data=dict(tasks=tasks,arms=seg.ARMS,train_overlap=0,weight_sha256=seg.WEIGHT_SHA,
                  magic_alignment='existing CaseLoader applies raw-to-model transform once')
        if manifest.exists():
            if n.read(manifest)!=json.loads(json.dumps(data)):raise ValueError('changed manifest')
        else:n.write(manifest,data)
        print('PREPARED',len(tasks),'held-out MAGIC tasks x 5 arms',flush=True)
        print([(t['class_name'],t['case_id']) for t in tasks],flush=True)
    elif a.arm and a.uuid:
        original=seg.signature
        seg.signature=lambda:hashlib.sha256((original()+n.sha(Path(__file__))+n.sha(manifest)).encode()).hexdigest()
        # Existing runtime's progress label says /12; manifest and completion checks use true length (14).
        seg.run(a.arm,a.uuid,a.limit)
    else:p.error('use --prepare or --arm/--uuid')

if __name__=='__main__':main()
