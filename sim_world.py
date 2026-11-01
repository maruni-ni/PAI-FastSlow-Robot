"""PyBullet distance sensing/collision queries; planar kinematic robot motion."""
from pathlib import Path
import argparse
import json
import math
import time
import numpy as np
import pybullet as p
from ai_model import AvoidanceModel
from robot_controller import Controller,wrap
from sensor_config import COARSE_DEG,RANGE,RAY_ORIGIN

ROOT = Path(__file__).resolve().parent
DT = 1/30
RADIUS = .39
ANGLES = np.radians([90,45,0,-45,-90])


def scenario(name, seed=0):
    if name == 'clear':
        boxes=[]
    elif name == 'single':
        boxes=[(4.3,0,.65,.7)]
    elif name == 'offset':
        boxes=[(4.3,.35,.65,.7)]
    elif name == 'double':
        boxes=[(3.5,0,.55,.65),(7.0,-.65,.6,.7)]
    elif name == 'blocked':
        boxes=[(2.3,0,.6,8)]
    elif name == 'random':
        rng=np.random.default_rng(seed)
        boxes=[(rng.uniform(3.8,4.8),rng.uniform(-.4,.4),rng.uniform(.4,.75),rng.uniform(.45,.85))]
    else:
        raise ValueError(name)
    return dict(name=name,seed=seed,start=[0,0,0],goal=[10,0],boxes=boxes)


def body(shape, position, color, **kwargs):
    collision = p.createCollisionShape(shape,**kwargs)
    visual_kwargs = dict(kwargs)
    if 'height' in visual_kwargs:
        visual_kwargs['length'] = visual_kwargs.pop('height')
    visual = p.createVisualShape(shape,rgbaColor=color,**visual_kwargs)
    return p.createMultiBody(baseMass=0,baseCollisionShapeIndex=collision,baseVisualShapeIndex=visual,basePosition=position)


class World:
    def __init__(self, config, gui=False):
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        self.config = config
        self.total_rays=0
        self.coarse_scans=0
        self.dense_scans=0
        self.corridor_queries=0
        p.setGravity(0,0,0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        body(p.GEOM_BOX,[5,0,-.09],[.12,.17,.23,1],halfExtents=[6.5,4,.08])
        self.obstacles = [body(p.GEOM_BOX,[cx,cy,.5],[.95,.39,.19,1],halfExtents=[hx,hy,.5]) for cx,cy,hx,hy in config['boxes']]
        # Cylinder is the conservative circular collision footprint.
        self.robot = body(p.GEOM_CYLINDER,[0,0,.22],[.1,.72,.9,1],radius=RADIUS,height=.34)
        self.nose = p.createMultiBody(baseMass=0,baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX,halfExtents=[.12,.1,.07],rgbaColor=[1,1,1,1]))
        self.wheels=[]
        for side in [-1,1]:
            vis=p.createVisualShape(p.GEOM_CYLINDER,radius=.16,length=.09,visualFrameOrientation=p.getQuaternionFromEuler([math.pi/2,0,0]),rgbaColor=[.035,.045,.06,1])
            self.wheels.append((side,p.createMultiBody(baseMass=0,baseVisualShapeIndex=vis)))
        goal=config['goal']
        p.createMultiBody(baseMass=0,baseVisualShapeIndex=p.createVisualShape(p.GEOM_CYLINDER,radius=.48,length=.025,rgbaColor=[.2,.92,.57,1]),basePosition=[*goal,.015])
        self.pose=np.asarray(config['start'],dtype=float)
        self.place(self.pose)
        p.resetDebugVisualizerCamera(11,0,-65,[5,0,0])

    def place(self, pose):
        x,y,yaw=pose
        quat=p.getQuaternionFromEuler([0,0,yaw])
        p.resetBasePositionAndOrientation(self.robot,[x,y,.22],quat)
        p.resetBasePositionAndOrientation(self.nose,[x+.20*math.cos(yaw),y+.20*math.sin(yaw),.43],quat)
        for side,wheel in self.wheels:
            p.resetBasePositionAndOrientation(wheel,[x-side*.30*math.sin(yaw),y+side*.30*math.cos(yaw),.17],quat)
        p.performCollisionDetection()

    def sense(self,angles=COARSE_DEG):
        x,y,yaw=self.pose
        dirs=np.array([[math.cos(math.radians(a)+yaw),math.sin(math.radians(a)+yaw),0] for a in angles])
        center=np.array([x,y,.23])
        starts=center+RAY_ORIGIN*dirs
        ends=starts+RANGE*dirs
        hits=p.rayTestBatch(starts.tolist(),ends.tolist())
        self.total_rays+=len(angles)
        if len(angles)==5:self.coarse_scans+=1
        else:self.dense_scans+=1
        return np.array([hit[2]*RANGE for hit in hits])

    def corridor_clear(self):
        self.total_rays+=3
        self.corridor_queries+=1
        x,y,_=self.pose
        goal=np.array(self.config['goal'])
        delta=goal-np.array([x,y]); length=np.linalg.norm(delta)
        direction=delta/max(length,1e-9); normal=np.array([-direction[1],direction[0]])
        starts=[]; ends=[]
        # Three parallel, footprint-expanded goal rays, not a global planner.
        for offset in [-.46,0,.46]:
            start=np.array([x,y])+normal*offset+direction*.44
            starts.append([*start,.23]); ends.append([*(goal+normal*offset),.23])
        return all(hit[0] == -1 for hit in p.rayTestBatch(starts,ends))

    def clearance(self):
        return min([point[8] for obstacle in self.obstacles for point in p.getClosestPoints(self.robot,obstacle,10)] or [10.0])

    def advance(self,v,w):
        # Four substeps bound translation to 7.5 mm at the maximum speed.
        minimum=10.0
        for _ in range(4):
            x,y,yaw=self.pose
            self.pose=np.array([x+v*math.cos(yaw)*DT/4,y+v*math.sin(yaw)*DT/4,wrap(yaw+w*DT/4)])
            self.place(self.pose)
            minimum=min(minimum,self.clearance())
            if minimum<=0:
                break
        return minimum

    def close(self):
        p.disconnect(self.client)

