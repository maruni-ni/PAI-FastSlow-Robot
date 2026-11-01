"""Use only recorded benchmark states; no control reruns while rendering."""
import json,math
from functools import lru_cache
import numpy as np
import pybullet as p
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from main import ROOT
from sim_world import World
from sensor_config import COARSE_DEG,DENSE_DEG,RAY_ORIGIN

OUT=ROOT/'output';FPS=15;BG=(12,19,30);WHITE=(232,241,248);CYAN=(35,209,224);ORANGE=(255,162,76);GREEN=(72,227,157);MUTED=(158,181,200)


@lru_cache(maxsize=40)
def font(size):return ImageFont.truetype('C:/Windows/Fonts/YuGothM.ttc',size)


def label(draw,xy,text,size=23,fill=WHITE):draw.text(xy,text,font=font(size),fill=fill)


def card(title,subtitle,lines):
    frame=Image.new('RGB',(1280,720),BG);draw=ImageDraw.Draw(frame)
    draw.rectangle((55,65,65,655),fill=CYAN)
    label(draw,(95,65),'PHYSICAL AI / ADAPTIVE SENSING',23,CYAN)
    label(draw,(95,133),title,41)
    label(draw,(95,215),subtitle,25,MUTED)
    for i,line in enumerate(lines):label(draw,(100,305+61*i),line,25)
    label(draw,(95,665),'Kinematic simulation / Stationary extra scan / Uncalibrated model probability',17,MUTED)
    return frame


def frame(world,record,row,trail,caption,pause=False):
    result=Image.new('RGB',(1280,720),BG);draw=ImageDraw.Draw(result)
    label(draw,(25,15),'FAST / SLOW + Adaptive Sensing',30,CYAN)
    label(draw,(25,58),record['config']['name']+' / '+record['summary']['method'],19,MUTED)
    mode=row['mode'];color=ORANGE if mode.startswith('SLOW') else GREEN if mode=='GOAL' else CYAN
    label(draw,(975,20),mode,20,color)
    view=p.computeViewMatrix([5,-7.3,8.9],[5,0,0],[0,0,1]);proj=p.computeProjectionMatrixFOV(52,940/505,.1,35)
    pixels=p.getCameraImage(940,505,viewMatrix=view,projectionMatrix=proj,renderer=p.ER_TINY_RENDERER,shadow=1,lightDirection=[-3,-4,8])[2]
    scene=Image.fromarray(np.asarray(pixels,dtype=np.uint8).reshape(505,940,4)[:,:,:3]);sd=ImageDraw.Draw(scene)
    matrix=np.array(proj).reshape(4,4,order='F')@np.array(view).reshape(4,4,order='F')
    def project(x,y,z):
        v=matrix@np.array([x,y,z,1]);v=v[:3]/v[3]
        return ((v[0]+1)*470,(1-v[1])*252.5)
    if len(trail)>1:sd.line([project(x,y,.03) for x,y in trail],fill=(34,225,210),width=3)
    x,y,yaw=row['pose'];dense=row['dense_sensors']
    angles=DENSE_DEG if dense is not None else COARSE_DEG
    for angle,distance in zip(angles,dense if dense is not None else row['sensors']):
        a=math.radians(angle)+yaw
        sensitivity_angle=record.get('sensitivity_angle')
        highlighted=dense is not None and sensitivity_angle==angle
        sd.line([project(x+RAY_ORIGIN*math.cos(a),y+RAY_ORIGIN*math.sin(a),.24),
                 project(x+(RAY_ORIGIN+distance)*math.cos(a),y+(RAY_ORIGIN+distance)*math.sin(a),.24)],fill=(255,76,112) if highlighted else (49,190,245) if dense is not None else (245,194,76),width=4 if highlighted else 2 if dense is not None else 1)
    result.paste(scene,(20,90))
    label(draw,(990,104),'PHASE',17,MUTED);label(draw,(990,135),row['phase'],24,color)
    direction='LEFT' if row['direction']>0 else 'RIGHT' if row['direction']<0 else '-'
    label(draw,(990,191),'Applied direction',18,MUTED);label(draw,(990,220),direction,31,ORANGE)
    label(draw,(990,282),f"AI calls: {row['calls']}",23)
    label(draw,(990,325),f"Dense scans: {row['dense_scans']}",21)
    confidence='-' if row['confidence'] is None else f"{row['confidence']:.3f}"
    label(draw,(990,367),'Model max P',18,MUTED);label(draw,(990,397),confidence,27)
    label(draw,(990,447),f"Rays: {row['total_rays']}",21)
    label(draw,(990,488),f"Sim: {row['t']:.2f}s",21)
    label(draw,(990,538),'EXPLANATORY HOLD' if pause else '1x LOG REPLAY',15,ORANGE if pause else MUTED)
    if dense is not None and record.get('sensitivity_angle') is not None:
        label(draw,(990,568),f"Sensitivity: +{record['sensitivity_angle']} deg",16,(255,76,112))
    label(draw,(28,615),caption,24)
    label(draw,(28,668),'平面運動学モデル / 未校正確率 / 視点を動かさない適応的観測',19,MUTED)
    return result


def make():
    comparison=json.loads((OUT/'final'/'comparison.json').read_text(encoding='utf-8'))
    agg=comparison['aggregate'];adaptive=agg['adaptive'];dense=agg['always_dense'];event=agg['event']
    title=card('曖昧なときだけ、観測を増やす','5本センサ → AI判断 → 不明瞭なら停止して13本スキャン',[
        '明瞭な場面：追加スキャンなしで左右を選択。',
        '曖昧な場面：詳細観測によりモデルの判断を更新。',
        '固定左・固定右・粗観測・適応観測・常時詳細を比較。'])
    title.save(OUT/'title_v2.png')
    path=OUT/'PAI_FastSlow_v2_Demo.mp4';total=0
    with imageio.get_writer(path,fps=FPS,codec='libx264',quality=8,macro_block_size=16,ffmpeg_log_level='error') as writer:
        def append(im,count=1):
            nonlocal total
            data=np.asarray(im)
            for _ in range(count):writer.append_data(data);total+=1
        append(title,7*FPS)
        specs=[('00_left_open_adaptive','右側に壁：AIは左へ。明瞭なので追加スキャン0回。'),
               ('01_right_open_adaptive','鏡映したコース：AIは右へ。固定左では時間切れ。'),
               ('03_ambiguous_right_adaptive','粗観測では左予測。詳細スキャン後は右へ判断を変更。')]
        for name,caption in specs:
            record=json.loads((OUT/'final'/'runs'/f'{name}.json').read_text(encoding='utf-8'))
            if name=='03_ambiguous_right_adaptive':
                sensitivity=json.loads((OUT/'sensitivity.json').read_text(encoding='utf-8'))
                record['sensitivity_angle']=max(sensitivity['sensors'],key=lambda r:abs(r['delta_p_left']))['angle_deg']
            world=World(record['config']);trail=[];held=False;dense_held=False
            try:
                for i,row in enumerate(record['frames']):
                    trail.append(row['pose'][:2])
                    important=row['mode']=='SLOW_REOBSERVE' or row['dense_sensors'] is not None
                    if i%2 and not important:continue
                    world.place(row['pose'])
                    image=frame(world,record,row,trail,caption)
                    if i%2==0:append(image)
                    if row['mode']=='SLOW_REOBSERVE' and not held:
                        append(frame(world,record,row,trail,'判断が曖昧：停止し追加観測を予約（説明のため2秒静止）',True),2*FPS)
                        held=True
                    if row['dense_sensors'] is not None and not dense_held:
                        detail=frame(world,record,row,trail,'右へ更新。赤は入力置換に敏感なセンサ（説明のため2秒静止）',True)
                        append(detail,2*FPS);detail.save(OUT/'adaptive_evidence.png');dense_held=True
                last=record['frames'][-1];world.place(last['pose'])
                image=frame(world,record,last,trail,'GOAL：接触なし。保存した比較実験ログを再生しています。')
                append(image,FPS)
                if name.startswith('00'):image.save(OUT/'left_evidence.png')
                if name.startswith('01'):image.save(OUT/'right_evidence.png')
            finally:world.close()
        reduction=(1-adaptive['mean_total_rays']/dense['mean_total_rays'])*100
        comparison_card=card('比較結果：成功率は変わらなかった','開発後に凍結した24シーン × 5方式。全120試行を保存。',[
            '固定左 17/24、固定右 17/24。接触は全方式0回。',
            f"粗観測 {event['successes']}/24、適応観測 {adaptive['successes']}/24、常時詳細 {dense['successes']}/24。",
            f"適応観測の総レイ数は常時詳細より {reduction:.1f}% 少ない。",
            '2シーンは全方式で時間切れ。失敗も除外していません。'])
        comparison_card.save(OUT/'comparison_card.png');append(comparison_card,12*FPS)
        limitations=card('何を示し、何が残ったか','Adaptive Sensing：視点変更を伴うActive Perceptionではない',[
            '追加観測で左右判断が変わることを実証。',
            '最小間隔は改善したが、到達率の向上は確認できず。',
            '教師は幾何学規則。実機・動力学・迷路は未検証。',
            '動画は等速ログ再生＋説明用静止。音声なし。'])
        append(limitations,max(9*FPS,80*FPS-total))
    info={'file':path.name,'duration_s':total/FPS,'frames':total,'fps':FPS,'resolution':[1280,720],
          'sources':[f'final/runs/{name}.json' for name,_ in specs],
          'playback':'1x replay with two explicitly labelled 2-second event holds and explanatory cards; silent Japanese captions'}
    (OUT/'video_metadata.json').write_text(json.dumps(info,indent=2),encoding='utf-8');print(json.dumps(info))


if __name__=='__main__':make()
