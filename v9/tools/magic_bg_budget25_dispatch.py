"""Resource-only dispatch for the already signed paired MAGIC background retest."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import magic_bg_budget25 as test


def eligible(cards, active):
    return {i:(uuid,free) for i,(uuid,free,util) in cards.items()
            if i not in active and free>=45000 and util<=50}


def resources():
    amos=ROOT/'output/amos_prompt_matrix_random10_20261121/status.json'
    state=test.n.read(amos)
    active={int(gpu) for gpu,info in state['active'].items()
            if (Path('/proc')/str(info['pid'])).exists()}
    smi=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu',
                                 '--format=csv,noheader,nounits'],text=True)
    cards={}
    for line in smi.splitlines():
        index,uuid,free,util=(x.strip() for x in line.split(','))
        if free.isdigit() and util.isdigit():
            cards[int(index)]=uuid,int(free),int(util)
    return eligible(cards,active)


def main():
    with test.runner():
        original=test.m.gpu_inventory
        test.m.gpu_inventory=resources
        try:
            test.m.run()
        finally:
            test.m.gpu_inventory=original


if __name__=='__main__':
    main()
