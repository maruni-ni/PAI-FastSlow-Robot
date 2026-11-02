"""同じ配置で回避方式を比較する。"""
import argparse,hashlib,json,time
from collections import defaultdict
import numpy as np
from main import ROOT,METHODS,scenario,run
from sensor_config import COARSE_DEG,DENSE_DEG,analytic_ranges,CONFIDENCE_THRESHOLD


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(split='dev'):
    out=ROOT/'output'/split
    if split=='final' and out.exists():raise RuntimeError('Final evaluation exists; preserve it and choose a new explicit run directory before rerunning.')
    out.mkdir(parents=True,exist_ok=True)
    if split=='dev':
        configs=[scenario(name) for name in ['clear','single','double','left_open','right_open','ambiguous_left','ambiguous_right','blocked']]
    else:
        # 比較用の配置は固定する。
        configs=[scenario(name,90000+i) for i in range(6) for name in ['left_open','right_open','ambiguous_left','ambiguous_right']]
    freeze={'split':split,'confidence_threshold':CONFIDENCE_THRESHOLD,'configs':configs,
            'source_hashes':{name:digest(ROOT/name) for name in ['main.py','sim_world.py','robot_controller.py','sensor_config.py','ai_model.py','train_model.py','evaluate.py']},
            'model_hashes':{name:digest(ROOT/'models'/f'{name}.joblib') for name in ['coarse','dense']},
            'timing_note':'single measured rollout per method/scene, randomized method order; excludes world/model initialization; not energy or latency guarantee'}
    (out/'freeze.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8')
    summaries=[];records={};rng=np.random.default_rng(722)
    for index,config in enumerate(configs):
        for method in rng.permutation(METHODS):
            name=f"{index:02d}_{config['name']}_{method}"
            record=run(config,str(method),output=out/'runs'/f'{name}.json',quiet=True)
            summary=dict(record['summary'],run_file=f'runs/{name}.json',scene_index=index)
            summaries.append(summary);records[(index,str(method))]=record
        print(f"{split} scene {index+1}/{len(configs)} {config['name']}: "+' '.join(f"{m}={'OK' if records[(index,m)]['summary']['success'] else records[(index,m)]['summary']['failure_reason']}" for m in METHODS),flush=True)
    aggregate={}
    for method in METHODS:
        rows=[s for s in summaries if s['method']==method]
        aggregate[method]={'runs':len(rows),'successes':sum(r['success'] for r in rows),
                           'collisions':sum(r['collision'] for r in rows),'safe_stops_or_timeouts':sum(not r['success'] and not r['collision'] for r in rows)}
        for key in ['path_length_m','duration_s','inference_calls','inference_ms','loop_wall_ms','dense_scans','total_rays','reobservations']:
            aggregate[method]['mean_'+key]=float(np.mean([r[key] for r in rows]))
        aggregate[method]['minimum_clearance_m']=min(r['minimum_clearance_m'] for r in rows)
        successes=[r for r in rows if r['success']]
        aggregate[method]['mean_success_duration_s']=float(np.mean([r['duration_s'] for r in successes])) if successes else None
    schema_ok=True;counts_ok=True;stopped_scan=True;confident_no_scan=0;uncertain_scan=0
    for record in records.values():
        s=record['summary'];frames=record['frames'];events=record['events']
        counts_ok &= s['total_rays']==5*s['coarse_scans']+13*s['dense_scans']+3*s['corridor_queries']
        counts_ok &= s['inference_calls']==sum(e['event']=='INFER' for e in events)
        for frame in frames:
            schema_ok &= len(frame['sensors'])==len(COARSE_DEG)
            if frame['dense_sensors'] is not None:schema_ok &= len(frame['dense_sensors'])==len(DENSE_DEG)
            if frame['mode']=='SLOW_REOBSERVE':stopped_scan &= frame['v']==0 and frame['w']==0
        if s['method']=='adaptive' and s['inference_calls']:
            confident_no_scan+=s['reobservations']==0
            uncertain_scan+=s['reobservations']>0
    checks={'sensor_schema':bool(schema_ok),'ray_and_inference_accounting':bool(counts_ok),
            'reobserve_stops_motion':bool(stopped_scan),'no_collisions':not any(s['collision'] for s in summaries),
            'adaptive_confident_runs_without_reobserve':int(confident_no_scan),
            'adaptive_uncertain_runs_with_reobserve':int(uncertain_scan)}
    report={'split':split,'scenes':len(configs),'methods':list(METHODS),'aggregate':aggregate,'checks':checks,'summaries':summaries}
    (out/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'aggregate':aggregate,'checks':checks},indent=2))
    assert schema_ok and counts_ok and stopped_scan
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['dev','final'],default='dev')
    evaluate(parser.parse_args().split)
