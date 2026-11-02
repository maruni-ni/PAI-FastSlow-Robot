from pathlib import Path
import hashlib,json,math
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score
from sensor_config import COARSE_DEG,DENSE_DEG,RANGE,features,analytic_ranges,teacher

ROOT=Path(__file__).resolve().parent


def train():
    rng=np.random.default_rng(42);xs={'coarse':[],'dense':[]};ys=[];groups=[]
    for scene in range(3000):
        cx=rng.uniform(3.6,4.8);hx=rng.uniform(.45,.75);hy=rng.uniform(.55,.85)
        boxes=[(cx,0,hx,hy)]
        family=scene%3
        if family==0:
            boxes.append((5,-rng.uniform(1.25,1.9),5.5,rng.uniform(.12,.25)))
        elif family==1:
            # 粗いセンサの間に入る細い障害物。
            boxes.append((cx-rng.uniform(.7,1.1),-rng.uniform(.55,.85),rng.uniform(.12,.23),rng.uniform(.10,.18)))
        else:
            boxes += [(rng.uniform(2.7,5.5),rng.uniform(-2,2),rng.uniform(.15,.6),rng.uniform(.15,.5))]
        pose=(cx-hx-.43-rng.uniform(1.0,1.3),rng.uniform(-.06,.06),rng.uniform(-.06,.06))
        error=rng.uniform(-.15,.15)
        # 元配置と左右反転を同じグループにまとめる。
        for sign in [1,-1]:
            mirrored=[(x,sign*y,w,h) for x,y,w,h in boxes]
            actual_pose=(pose[0],sign*pose[1],sign*pose[2]);e=sign*error
            coarse=analytic_ranges(mirrored,actual_pose,COARSE_DEG)
            dense=analytic_ranges(mirrored,actual_pose,DENSE_DEG)
            xs['coarse'].append(features(coarse,e,COARSE_DEG));xs['dense'].append(features(dense,e,DENSE_DEG))
            ys.append(teacher(dense,e));groups.append(scene)
    ys=np.array(ys)
    tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=17).split(xs['coarse'],ys,groups))
    out=ROOT/'models';out.mkdir(exist_ok=True)
    metrics={'seed':42,'scenes':3000,'samples':len(ys),'training_samples':len(tr),'validation_samples':len(te),
             'split':'scene groups including their mirrors, 2250 train / 750 validation',
             'target':'same dense geometric clearance heuristic for both models','models':{}}
    for name,angles in [('coarse',COARSE_DEG),('dense',DENSE_DEG)]:
        x=np.array(xs[name]);model=RandomForestClassifier(n_estimators=80,max_depth=14,min_samples_leaf=5,random_state=42,n_jobs=1)
        model.fit(x[tr],ys[tr]);pred=model.predict(x[te])
        artifact={'model':model,'angles_deg':list(angles),'range_m':RANGE,'feature_count':len(angles)+2,'schema':2}
        path=out/f'{name}.joblib';joblib.dump(artifact,path,compress=3)
        metrics['models'][name]={'teacher_agreement':float(accuracy_score(ys[te],pred)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (out/'training_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':train()
