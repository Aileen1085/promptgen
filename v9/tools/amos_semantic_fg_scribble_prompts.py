"""Cached native foreground scribbles with every background input removed.

No CT or target geometry is accepted. Native validation, selected coordinates,
square padding, pixel-preserving rasterization, and ROI arithmetic are reused.
"""
from __future__ import annotations
import copy
import numpy as np
from tools import amos_prompt_ablation_prompts as native

TARGET='semantic_fg_scribble'
VARIANTS=(TARGET,)


def variant_modes(variant):
    if variant!=TARGET:return native.variant_modes(variant)
    return dict(active_planes=('coronal','sagittal'),plane_ids=(1,0),
        foreground_mode='scribble',foreground_mode_id=0,foreground_type_id=0,
        background_mode='none',background_mode_id=2,background_type_id=None,
        background_active=False)


def build_variant(package,native_groups,variant,prompt_plane_size):
    if variant!=TARGET:return native.build_variant(package,native_groups,variant,prompt_plane_size)
    # Baseline reconstructs cached FG rather than newly simulated plane FG.
    # Erasing only BG after native channel-independent resize is bit-exact to
    # erasing raw BG before that resize. Audit both representations explicitly.
    converted,audit=native.build_variant(package,native_groups,'fg_scribble_bg_scribble',prompt_plane_size)
    shape,slices,foreground,groups,_,_=native._validated_inputs(package,native_groups)
    audit=copy.deepcopy(audit);converted=copy.deepcopy(converted)
    converted['groups'][:,1:]=0
    converted['background_mode']='none'
    converted['background_points_per_plane']={p:0 for p in ('coronal','sagittal')}
    audit.update(variant=variant,**{k:list(v) if isinstance(v,tuple) else v for k,v in variant_modes(variant).items()},
        raw_background_points_per_plane={'coronal':0,'sagittal':0},
        raster_background_pixels_per_plane={'coronal':0,'sagittal':0},
        groups_sha256=native._sha(converted['groups']))
    for index,plane in enumerate(('coronal','sagittal')):
        raw=np.zeros_like(groups[plane]);raw[0]=foreground[plane]
        raster=converted['groups'][index]
        a=audit['planes'][plane];a.update(background_type_id=None,background_active=False)
        for prefix,array in (('raw',raw),('raster',raster)):
            a[prefix+'_sha256']=native._sha(array)
            a[prefix+'_channel_sha256']=[native._sha(c) for c in array]
            a[prefix+'_channel_nonzero']=[int(np.count_nonzero(c)) for c in array]
    return converted,audit


roi_indices=native.roi_indices
