"""Behavior checks on development cases before freezing the final benchmark."""
import json
import numpy as np
from main import ROOT,scenario
from sim_world import World
from sensor_config import COARSE_DEG,DENSE_DEG,analytic_ranges


def verify():
    folder=ROOT/'output'/'dev'/'runs'
    def read(name,method='adaptive'):
        return json.loads(next(folder.glob(f'*_{name}_{method}.json')).read_text(encoding='utf-8'))
    def first(record,event):return next(e for e in record['events'] if e['event']==event)
    left,right=read('left_open'),read('right_open')
    a,b=read('ambiguous_left'),read('ambiguous_right')
    assert first(left,'APPLY')['direction']==1 and first(right,'APPLY')['direction']==-1
    assert left['summary']['success'] and right['summary']['success']
    assert left['summary']['dense_scans']==right['summary']['dense_scans']==0
    assert not read('left_open','fixed_right')['summary']['success']
    assert not read('right_open','fixed_left')['summary']['success']
    assert np.allclose(first(a,'INFER')['sensors'],first(b,'INFER')['sensors'],atol=1e-9)
    assert first(a,'APPLY')['direction']==1 and first(b,'APPLY')['direction']==-1
    assert a['summary']['dense_scans']>0 and b['summary']['dense_scans']>0
    b_infers=[e for e in b['events'] if e['event']=='INFER']
    assert b_infers[0]['direction']!=b_infers[1]['direction']
    assert any(f['mode']=='SLOW_REOBSERVE' and f['v']==0 and f['w']==0 for f in b['frames'])
    assert not read('clear')['events'] and read('clear')['summary']['inference_calls']==0
    # Compare actual PyBullet sensor contract to synthetic training geometry.
    largest_error=0.
    for name in ['single','left_open','right_open','ambiguous_left','ambiguous_right']:
        config=scenario(name);world=World(config)
        try:
            for pose in [[0,0,0],[2.04,0,0],[1.8,.05,.04]]:
                world.pose=np.array(pose,dtype=float);world.place(world.pose)
                for angles in [COARSE_DEG,DENSE_DEG]:
                    error=float(np.max(abs(world.sense(angles)-analytic_ranges(config['boxes'],pose,angles))))
                    largest_error=max(largest_error,error)
                    assert error<.006,(name,pose,error)
        finally:world.close()
    report={'left_right_choice':True,'fixed_wrong_side_fails':True,'coarse_alias_pair':True,
            'detailed_observation_changes_choice':True,'confident_no_extra_scan':True,
            'reobserve_stationary':True,'no_ai_on_clear_course':True,
            'maximum_training_vs_pybullet_range_error_m':largest_error}
    (ROOT/'output'/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':verify()
