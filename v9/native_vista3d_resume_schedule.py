"""Isolate GPU inventory subprocess from the signed recovery worker launcher."""
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import native_vista3d_recovery as recovery
NATIVE_POPEN=subprocess.Popen
NATIVE_CHECK_OUTPUT=subprocess.check_output


def inventory_output(*args,**kwargs):
    # The scheduler is single-threaded. Only worker launches use its dispatcher.
    dispatcher=subprocess.Popen
    try:
        subprocess.Popen=NATIVE_POPEN
        return NATIVE_CHECK_OUTPUT(*args,**kwargs)
    finally:
        subprocess.Popen=dispatcher


if __name__=='__main__':
    recovery.reused_rows()
    recovery.native.write(recovery.OUTPUT/'scheduler_bootstrap_audit.json',dict(
        bootstrap_sha256=recovery.native.sha(Path(__file__)),
        recovery_sha256=recovery.native.sha(Path(recovery.__file__))))
    subprocess.check_output=inventory_output
    sys.argv=[str(Path(recovery.__file__)),'--schedule']
    recovery.main()
