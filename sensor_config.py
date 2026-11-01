"""Single source of truth shared by training, inference, simulation and HUD."""
import math
import numpy as np

RANGE=4.0
RAY_ORIGIN=.43
COARSE_DEG=(90,45,0,-45,-90)
DENSE_DEG=(90,75,60,45,30,15,0,-15,-30,-45,-60,-75,-90)
FRONT_MASK=np.array([abs(a)<=45 for a in COARSE_DEG])
FRONT_INDEX=COARSE_DEG.index(0)
CONFIDENCE_THRESHOLD=.80


def features(distances,error,angles):
    distances=np.asarray(distances,dtype=float)
    if distances.shape!=(len(angles),) or not np.all(np.isfinite(distances)):
        raise ValueError('Sensor contract mismatch')
    return np.array([*(np.clip(distances,0,RANGE)/RANGE),math.sin(error),math.cos(error)])


def analytic_ranges(boxes,pose,angles):
    x,y,yaw=pose;result=[]
    for a in angles:
        theta=math.radians(a)+yaw
        direction=np.array([math.cos(theta),math.sin(theta)])
        origin=np.array([x,y])+RAY_ORIGIN*direction
        nearest=RANGE
        for cx,cy,hx,hy in boxes:
            low=np.array([cx-hx,cy-hy]);high=np.array([cx+hx,cy+hy])
            if np.all(origin>=low) and np.all(origin<=high):
                nearest=0.;continue
            safe=np.where(abs(direction)<1e-12,np.copysign(1e-12,direction+1e-20),direction)
            lo=(low-origin)/safe;hi=(high-origin)/safe
            entry=np.max(np.minimum(lo,hi));exit=np.min(np.maximum(lo,hi))
            if exit>=max(entry,0):nearest=min(nearest,max(entry,0))
        result.append(nearest)
    return np.array(result)


def teacher(dense,error):
    # Geometric imitation target, not a learned success/optimality label.
    left=np.array([dense[i] for i,a in enumerate(DENSE_DEG) if 15<=a<=75])
    right=np.array([dense[i] for i,a in enumerate(DENSE_DEG) if -75<=a<=-15])
    score=.65*(np.min(left)-np.min(right))+.30*(np.mean(left)-np.mean(right))+.10*error
    return 1 if score>=0 else -1
