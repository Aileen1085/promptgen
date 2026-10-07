"""Non-learned 3D sampling and invertible physical cube geometry.

Arrays and point triples use the SAME axis order. No axial prompt expansion.
"""
import numpy as np


def sample_points(points, spacing, cap, minimum_mm=3., method='fps'):
    p = np.unique(np.asarray(points).reshape(-1, 3), axis=0)
    spacing = np.asarray(spacing, float)
    if cap < 1 or spacing.shape != (3,) or not np.all(np.isfinite(spacing) & (spacing > 0)):
        raise ValueError('invalid sampling budget/spacing')
    if not len(p): return np.zeros((0, 3), float)
    physical = p * spacing
    center = physical.mean(0)
    if method == 'fps':
        selected = [int(np.argmin(((physical-center)**2).sum(1)))]
        distance = np.full(len(p), np.inf)
        while len(selected) < min(cap, len(p)):
            distance = np.minimum(distance, ((physical-physical[selected[-1]])**2).sum(1))
            distance[selected] = -1
            candidate = int(distance.argmax())
            if distance[candidate] < minimum_mm**2: break
            selected.append(candidate)
    elif method == 'axis':
        axis = int(np.ptp(physical, axis=0).argmax())
        targets = np.linspace(physical[:,axis].min(),physical[:,axis].max(),min(cap,len(p)))
        selected = []
        for value in targets:
            candidates = np.argsort(abs(physical[:,axis]-value),kind='stable')
            for candidate in candidates:
                if candidate in selected: continue
                if selected and np.linalg.norm(physical[selected]-physical[candidate],axis=1).min() < minimum_mm:
                    continue
                selected.append(int(candidate)); break
    else: raise ValueError('unknown 3D sampler')
    return p[selected].astype(float)


def scribble_points(foreground, background, spacing, plane_slices, cap, method='fps'):
    fg = np.argwhere(foreground); bg = np.asarray(background).reshape(-1,3)
    coronal, sagittal = plane_slices
    points = []; labels = []
    for role, positions in ((1, fg),(0,bg)):
        sampled = []
        for axis,index in ((2,coronal),(1,sagittal)):
            selected = sample_points(positions[positions[:,axis]==index], spacing, cap, 3., method)
            sampled.extend(selected.tolist())
        if not sampled: raise ValueError('missing foreground/background plane support')
        q = np.unique(np.asarray(sampled),axis=0)
        points.extend(q.tolist()); labels.extend([role]*len(q))
    return np.asarray(points,float),np.asarray(labels,np.int64)


def cube_geometry(bounds, spacing, size=128, mode='physical'):
    starts = np.asarray([a for a,b in bounds],float)
    shape = np.asarray([b-a for a,b in bounds],float)
    spacing = np.asarray(spacing,float)
    if (spacing.shape != (3,) or not np.all(np.isfinite(spacing)&(spacing>0)) or
        not np.all(shape>0) or size<2): raise ValueError('invalid cube geometry')
    if mode=='physical':
        pitch = np.repeat(max(shape*spacing)/size,3)
        pad = (size-shape*spacing/pitch)/2
    elif mode=='stretch':
        pitch=shape*spacing/size;pad=np.zeros(3)
    else: raise ValueError('unknown cube geometry')
    return dict(starts=starts.tolist(),shape=shape.tolist(),spacing=spacing.tolist(),
                pitch=pitch.tolist(),pad=pad.tolist(),size=int(size),mode=mode)


def to_cube(points, geometry):
    g=geometry
    return ((np.asarray(points)-g['starts']+.5)*g['spacing']/g['pitch']+g['pad']-.5)


def from_cube(points, geometry):
    g=geometry
    return ((np.asarray(points)+.5-g['pad'])*g['pitch']/g['spacing']+g['starts']-.5)
