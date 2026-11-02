"""保存した走行ログを再生する。"""
import json,time
import pybullet as p
from main import ROOT
from sim_world import World

record=json.loads((ROOT/'output'/'final'/'runs'/'03_ambiguous_right_adaptive.json').read_text(encoding='utf-8'))
world=World(record['config'],gui=True);text=-1
try:
    while p.isConnected():
        for row in record['frames']:
            if not p.isConnected():break
            world.place(row['pose'])
            text=p.addUserDebugText(f"Replay | {row['mode']} | calls {row['calls']} | scans {row['dense_scans']}",[0,2.5,0],
                textColorRGB=[1,1,1],textSize=1.3,replaceItemUniqueId=text)
            if ord('q') in p.getKeyboardEvents():raise KeyboardInterrupt
            time.sleep(2 if row['mode']=='SLOW_REOBSERVE' or row['dense_sensors'] is not None else 1/30)
        time.sleep(2)
except KeyboardInterrupt:pass
finally:
    if p.isConnected():world.close()
