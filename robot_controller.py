import math
import numpy as np
from sensor_config import FRONT_MASK,CONFIDENCE_THRESHOLD


def wrap(angle):return (angle+math.pi)%(2*math.pi)-math.pi


def track(yaw,target,speed):
    e=wrap(target-yaw)
    return (speed*max(0.,math.cos(e)) if abs(e)<.8 else 0.),float(np.clip(2.5*e,-1.5,1.5))


class Controller:
    def __init__(self,model,method='adaptive'):
        self.model=model;self.method=method;self.mode='FAST';self.phase='GOAL'
        self.direction=0;self.target_heading=0.;self.entry_distance=0.;self.clear_time=0.;self.slow_time=0.
        self.events=[];self.confidence=None;self.reobservations=0

    def apply(self,result,yaw,t,pose,sensors):
        self.direction=result['direction'];self.confidence=result.get('confidence')
        self.target_heading=wrap(yaw+self.direction*math.radians(70))
        self.mode,self.phase='SLOW_AVOID','TURN'
        self.events.append(dict(event='APPLY',t=t,pose=list(pose),sensors=list(sensors),**result))

    def step(self,pose,coarse,dense,goal,corridor_clear,dt,t):
        x,y,yaw=pose;distance=math.hypot(goal[0]-x,goal[1]-y)
        bearing=math.atan2(goal[1]-y,goal[0]-x);error=wrap(bearing-yaw)
        if distance<.35:
            self.mode,self.phase='GOAL','STOP';return 0.,0.
        if self.mode=='FAST':
            if not corridor_clear and min(coarse[FRONT_MASK])<1.2 and abs(error)<.65:
                self.entry_distance=distance;self.clear_time=0.;self.slow_time=0.
                self.mode,self.phase='SLOW_DECIDE','DECIDE'
                self.events.append(dict(event='ENTER',t=t,pose=list(pose),sensors=list(coarse)))
                return 0.,0.
            return track(yaw,bearing,.9)
        self.slow_time+=dt
        if self.slow_time>25:
            self.mode,self.phase='FAILED','TIMEOUT';return 0.,0.
        if self.mode=='SLOW_DECIDE':
            if self.method in ('fixed_left','fixed_right'):
                result={'direction':1 if self.method=='fixed_left' else -1,'kind':'fixed'}
            else:
                kind='dense' if self.method=='always_dense' else 'coarse'
                result=self.model.decide(dense if kind=='dense' else coarse,error,kind)
                self.events.append(dict(event='INFER',t=t,pose=list(pose),sensors=list(dense if kind=='dense' else coarse),**result))
                self.confidence=result['confidence']
                if self.method=='adaptive' and result['confidence']<CONFIDENCE_THRESHOLD:
                    self.mode,self.phase='SLOW_REOBSERVE','DENSE_SCAN';self.reobservations+=1
                    self.events.append(dict(event='REOBSERVE',t=t,reason='uncalibrated max probability below threshold',threshold=CONFIDENCE_THRESHOLD))
                    return 0.,0.
            self.apply(result,yaw,t,pose,coarse);return 0.,0.
        if self.mode=='SLOW_REOBSERVE':
            if dense is None:raise ValueError('Dense observation required')
            result=self.model.decide(dense,error,'dense')
            self.events.append(dict(event='INFER',t=t,pose=list(pose),sensors=list(dense),**result))
            self.apply(result,yaw,t,pose,dense);return 0.,0.
        self.clear_time=self.clear_time+dt if corridor_clear and distance<self.entry_distance-.1 else 0.
        if self.phase=='TURN':
            if abs(wrap(self.target_heading-yaw))<.06:self.phase='PASS'
            return track(yaw,self.target_heading,0.)
        if self.clear_time>.45:
            self.mode,self.phase='FAST','GOAL';return track(yaw,bearing,.9)
        if min(coarse[FRONT_MASK])<.55:
            self.target_heading=wrap(self.target_heading+self.direction*math.radians(35));self.phase='TURN'
            return 0.,float(self.direction)
        return track(yaw,self.target_heading,.65)
