"""Non-learned VISTA XYZ geometry and count-normalized local-window fusion."""
import numpy as np

def geometry(bounds,spacing,mode):
    starts=np.array([a for a,b in bounds],float);shape=np.array([b-a for a,b in bounds],int)
    spacing=np.asarray(spacing,float)
    if spacing.shape!=(3,) or not np.all(np.isfinite(spacing)&(spacing>0)) or not np.all(shape>0):
        raise ValueError('invalid geometry')
    if mode=='fixed':
        pitch=np.repeat(1.5,3);target=np.maximum(1,np.rint((shape-1)*spacing/pitch).astype(int)+1)
        offset=np.zeros(3)
    elif mode=='fit':
        pitch=np.repeat(max(shape*spacing)/128.,3);target=np.repeat(128,3)
        pad=(target-shape*spacing/pitch)/2
        offset=(.5-pad)*pitch/spacing-.5
    else:raise ValueError('unknown physical geometry')
    return dict(starts=starts.tolist(),shape=shape.tolist(),spacing=spacing.tolist(),pitch=pitch.tolist(),
                target=target.tolist(),offset=offset.tolist(),mode=mode)

def forward_points(points,g):
    return (np.asarray(points)-g['starts']-g['offset'])*g['spacing']/g['pitch']

def inverse_points(points,g):
    return np.asarray(points)*g['pitch']/g['spacing']+g['offset']+g['starts']

def native_view(image,points):
    return np.ascontiguousarray(np.asarray(image).transpose(1,2,0)),np.asarray(points)[:,[1,2,0]]

def canonical_view(image):return np.asarray(image).transpose(2,0,1)

def windows(points,shape,size=128):
    shape=np.asarray(shape,int);p=np.asarray(points,float).reshape(-1,3)
    if size<2 or not len(p) or np.any(shape<1) or not np.isfinite(p).all():raise ValueError('invalid windows')
    padded=np.maximum(shape,size);pad=(padded-shape)//2;p=p+pad
    selected=[]
    for point in p:
        start=np.clip(np.floor(point-size/2).astype(int),0,padded-size)
        window=tuple((int(a),int(a+size)) for a in start)
        if window not in selected:selected.append(window)
    return selected,tuple(padded.tolist()),pad

def fuse(logits,windows,shape):
    if len(logits)!=len(windows):raise ValueError('window/logit mismatch')
    sums=np.zeros(shape,np.float32);counts=np.zeros(shape,np.uint16)
    for output,bounds in zip(logits,windows):
        sl=tuple(slice(a,b) for a,b in bounds)
        if np.asarray(output).shape!=sums[sl].shape or not np.isfinite(output).all():raise ValueError('invalid output')
        sums[sl]+=output;counts[sl]+=1
    result=np.full(shape,-9999.,np.float32);covered=counts>0
    result[covered]=sums[covered]/counts[covered];return result
