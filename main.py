from pathlib import Path
import argparse,json,math,time
import numpy as np
from sim_world import World,DT
from ai_model import AvoidanceModel
from robot_controller import Controller
from sensor_config import COARSE_DEG,DENSE_DEG

ROOT=Path(__file__).resolve().parent
METHODS=('fixed_left','fixed_right','event','adaptive','always_dense')
SCENARIOS=('clear','single','offset','double','left_open','right_open','ambiguous_left','ambiguous_right','blocked')


def scenario(name,seed=0):
    rng=np.random.default_rng(seed)
    cx=4.3;hx=.65;hy=.7
    if seed:
        cx=rng.uniform(4.1,4.5);hx=rng.uniform(.55,.7);hy=rng.uniform(.65,.8)
    boxes=[(cx,0,hx,hy)]
    if name=='clear':boxes=[]
    elif name=='offset':boxes=[(cx,.35,hx,hy)]
    elif name=='double':boxes=[(3.5,0,.55,.65),(7,-.65,.6,.7)]
    elif name in ('left_open','right_open'):
        sign=-1 if name=='left_open' else 1
        boxes.append((5,sign*(1.5 if not seed else rng.uniform(1.4,1.65)),5.5,.18))
    elif name in ('ambiguous_left','ambiguous_right'):
        sign=-1 if name=='ambiguous_left' else 1
        boxes.append((cx-.9,sign*.72,.18,.13))
    elif name=='blocked':boxes=[(2.3,0,.6,8)]
    elif name!='single':raise ValueError(name)
    return {'name':name,'seed':seed,'start':[0.,0.,0.],'goal':[10.,0.],'boxes':boxes}


def run(config,method='adaptive',gui=False,realtime=False,output=None,quiet=False):
    world=World(config,gui);model=AvoidanceModel();controller=Controller(model,method)
    rows=[];minimum=10.;collision=False;path_length=0.;start=time.perf_counter()
    try:
        for step in range(1800):
            t=step*DT
            coarse=world.sense(COARSE_DEG)
            dense=None
            if method=='always_dense' or controller.mode=='SLOW_REOBSERVE':dense=world.sense(DENSE_DEG)
            corridor=world.corridor_clear()
            clearance=world.clearance();minimum=min(minimum,clearance)
            v,w=controller.step(world.pose,coarse,dense,config['goal'],corridor,DT,t)
            if clearance<.06 and controller.mode!='GOAL':
                controller.mode,controller.phase='FAILED','SAFETY_STOP';v,w=0.,0.
            rows.append({'t':t,'pose':world.pose.tolist(),'mode':controller.mode,'phase':controller.phase,
                         'direction':controller.direction,'sensors':coarse.tolist(),'dense_sensors':None if dense is None else dense.tolist(),
                         'calls':model.calls,'confidence':controller.confidence,'dense_scans':world.dense_scans,
                         'total_rays':world.total_rays,'v':v,'w':w})
            if controller.mode in ('GOAL','FAILED'):break
            before=world.pose.copy();clearance=world.advance(v,w);minimum=min(minimum,clearance)
            path_length+=float(np.linalg.norm(world.pose[:2]-before[:2]))
            if clearance<=0:
                collision=True;controller.mode,controller.phase='FAILED','CONTACT';break
            if realtime:time.sleep(DT)
        else:controller.mode,controller.phase='FAILED','MAX_STEPS'
        elapsed=(time.perf_counter()-start)*1000
        summary={'scenario':config['name'],'seed':config['seed'],'method':method,
                 'success':controller.mode=='GOAL' and not collision,'collision':collision,
                 'final_mode':controller.mode,'failure_reason':None if controller.mode=='GOAL' else controller.phase,
                 'duration_s':rows[-1]['t'],'steps':len(rows),'path_length_m':path_length,'minimum_clearance_m':minimum,
                 'final_distance_m':math.dist(world.pose[:2],config['goal']),'inference_calls':model.calls,
                 'inference_ms':model.elapsed_ms,'loop_wall_ms':elapsed,'dense_scans':world.dense_scans,
                 'coarse_scans':world.coarse_scans,'corridor_queries':world.corridor_queries,'total_rays':world.total_rays,
                 'reobservations':controller.reobservations}
        record={'config':config,'summary':summary,'events':controller.events,'frames':rows}
        if output:
            path=Path(output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(record,indent=2),encoding='utf-8')
        if not quiet:print(json.dumps(summary))
        return record
    finally:world.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--scenario',default='left_open',choices=SCENARIOS)
    parser.add_argument('--method',default='adaptive',choices=METHODS)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--gui',action='store_true');parser.add_argument('--realtime',action='store_true')
    parser.add_argument('--output',default=str(ROOT/'output'/'run.json'))
    args=parser.parse_args();result=run(scenario(args.scenario,args.seed),args.method,args.gui,args.realtime,args.output)
    raise SystemExit(0 if result['summary']['success'] else 1)
