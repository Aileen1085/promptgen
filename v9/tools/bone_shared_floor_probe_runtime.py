"""Bind the checked raw decoder into the compiled shared main namespace."""
from tools.bone_shared_floor_restore_runtime import (
    GridSession, install_v92, install_restoration, tag_carrier,
    install_v10 as prior_install_v10,
)


def bind_current_decode(evaluator):
    evaluator.main.__globals__['_dynamic_adaptive_decode'] = evaluator._dynamic_adaptive_decode


def install_v10(session):
    prior_install_v10(session)
    import v10.infer.infer_eight_source_adaptive_v10_2 as evaluator
    bind_current_decode(evaluator)
