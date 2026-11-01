from pathlib import Path
import time
import joblib
import numpy as np
from sensor_config import COARSE_DEG,DENSE_DEG,RANGE,features


class AvoidanceModel:
    def __init__(self):
        root=Path(__file__).resolve().parent
        self.models={}
        for name,angles in [('coarse',COARSE_DEG),('dense',DENSE_DEG)]:
            artifact=joblib.load(root/'models'/f'{name}.joblib')
            if artifact['angles_deg']!=list(angles) or artifact['range_m']!=RANGE or artifact['feature_count']!=len(angles)+2:
                raise ValueError('Model/sensor schema mismatch')
            self.models[name]=artifact['model']
        self.calls=0;self.elapsed_ms=0.

    def decide(self,distances,error,kind='coarse'):
        angles=COARSE_DEG if kind=='coarse' else DENSE_DEG
        x=features(distances,error,angles)
        start=time.perf_counter();model=self.models[kind]
        probabilities=model.predict_proba([x])[0]
        duration=(time.perf_counter()-start)*1000
        index=int(np.argmax(probabilities));self.calls+=1;self.elapsed_ms+=duration
        return {'direction':int(model.classes_[index]),'confidence':float(probabilities[index]),
                'p_left':float(probabilities[list(model.classes_).index(1)]),'latency_ms':duration,'kind':kind}
