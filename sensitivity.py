"""Offline input replacement sensitivity, excluded from control cost counts."""
import json
import numpy as np
from main import ROOT
from ai_model import AvoidanceModel
from sensor_config import DENSE_DEG,RANGE,features


def analyze():
    source=ROOT/'output'/'final'/'runs'/'03_ambiguous_right_adaptive.json'
    record=json.loads(source.read_text(encoding='utf-8'))
    observation=next(e for e in record['events'] if e['event']=='INFER' and e['kind']=='dense')
    # Goal error is zero here because stationary reobservation precedes turning.
    error=0.;model=AvoidanceModel().models['dense'];left_index=list(model.classes_).index(1)
    distances=np.array(observation['sensors'])
    baseline=float(model.predict_proba([features(distances,error,DENSE_DEG)])[0,left_index])
    rows=[]
    for index,angle in enumerate(DENSE_DEG):
        altered=distances.copy();altered[index]=RANGE
        value=float(model.predict_proba([features(altered,error,DENSE_DEG)])[0,left_index])
        rows.append({'angle_deg':angle,'original_m':float(distances[index]),'replacement_m':RANGE,'delta_p_left':value-baseline})
    result={'source':str(source.relative_to(ROOT)),'baseline_p_left':baseline,'offline_inference_calls':14,
            'interpretation':'one-feature replacement with maximum range; local model sensitivity, not causal or safety proof; possibly off-manifold inputs',
            'sensors':rows}
    (ROOT/'output'/'sensitivity.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(sorted(rows,key=lambda r:abs(r['delta_p_left']),reverse=True)[:3],indent=2))


if __name__=='__main__':analyze()
