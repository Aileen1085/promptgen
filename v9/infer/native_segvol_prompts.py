"""Invertible physical transforms for native SegVol, no learned conversion."""
import numpy as np

def geometry(bounds,spacing,mode):
    starts=np.array([a for a,b in bounds],float);shape=np.array([b-a for a,b in bounds],float)
    spacing=np.asarray(spacing,float);target=np.array((32,256,256),float)
    if spacing.shape!=(3,) or not np.all(np.isfinite(spacing)&(spacing>0)) or not np.all(shape>0):
        raise ValueError('invalid geometry')
    if mode=='stretch':pitch=shape*spacing/target
    elif mode=='physical':pitch=np.repeat(np.max(shape*spacing/target),3)
    else:raise ValueError('unknown geometry')
    return dict(starts=starts.tolist(),shape=shape.astype(int).tolist(),spacing=spacing.tolist(),
                pitch=pitch.tolist(),pad=((target-shape*spacing/pitch)/2).tolist(),target=target.astype(int).tolist(),mode=mode)

def forward_points(points,g):
    return (np.asarray(points)-g['starts']+.5)*g['spacing']/g['pitch']+g['pad']-.5

def inverse_points(points,g):
    return (np.asarray(points)+.5-g['pad'])*g['pitch']/g['spacing']+g['starts']-.5

def native_view(volume,points,convention):
    image=np.ascontiguousarray(np.asarray(volume).transpose(0,2,1))
    if convention=='official':q=np.asarray(points)[:,[0,2,1]]
    elif convention=='pe_aligned':q=np.asarray(points)[:,[2,0,1]]
    else:raise ValueError('unknown point convention')
    return image,q

def canonical_view(volume):return np.asarray(volume).transpose(0,2,1)
