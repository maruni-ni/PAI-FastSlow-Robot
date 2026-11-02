"""走行ログから字幕付き動画を作る。"""
from pathlib import Path
from functools import lru_cache
import json,math,sys
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT
DATA=PROJECT/'output';OUT=ROOT/'output'
import pybullet as p
from sim_world import World
from sensor_config import COARSE_DEG,DENSE_DEG,RAY_ORIGIN

FPS=15;INK=(30,30,30);GRAY=(100,100,100)
MODE={'FAST':'通常走行','SLOW_DECIDE':'左右を判断','SLOW_REOBSERVE':'停止して再観測','SLOW_AVOID':'回避中','GOAL':'ゴール到達','FAILED':'停止'}


@lru_cache(maxsize=20)
def font(size):return ImageFont.truetype('C:/Windows/Fonts/YuGothM.ttc',size)


def write(draw,xy,text,size=24,color=INK):draw.text(xy,text,font=font(size),fill=color)


def simple_world(config):
    world=World(config)
    p.changeVisualShape(p.getBodyUniqueId(0),-1,rgbaColor=[.93,.93,.93,1])
    for obstacle in world.obstacles:p.changeVisualShape(obstacle,-1,rgbaColor=[.52,.52,.52,1])
    p.changeVisualShape(world.robot,-1,rgbaColor=[.27,.44,.62,1])
    # 最後に作成した表示用オブジェクトがゴール。
    p.changeVisualShape(p.getBodyUniqueId(p.getNumBodies()-1),-1,rgbaColor=[.40,.61,.40,1])
    return world


def scene(world,row,trail):
    width,height=1280,490
    view=p.computeViewMatrix([5,-3.5,12],[5,0,0],[0,0,1])
    projection=p.computeProjectionMatrixFOV(27,width/height,.1,40)
    image=p.getCameraImage(width,height,viewMatrix=view,projectionMatrix=projection,renderer=p.ER_TINY_RENDERER,
                          shadow=0,lightDirection=[-3,-4,8])[2]
    image=Image.fromarray(np.array(image,dtype=np.uint8).reshape(height,width,4)[:,:,:3])
    draw=ImageDraw.Draw(image)
    matrix=np.array(projection).reshape(4,4,order='F')@np.array(view).reshape(4,4,order='F')
    def project(x,y,z=.035):
        vector=matrix@np.array([x,y,z,1]);ndc=vector[:3]/vector[3]
        return ((ndc[0]+1)*width/2,(1-ndc[1])*height/2)
    if len(trail)>1:draw.line([project(x,y) for x,y in trail],fill=(80,80,80),width=2)
    if row['dense_sensors'] is not None:
        x,y,yaw=row['pose']
        for angle,distance in zip(DENSE_DEG,row['dense_sensors']):
            theta=math.radians(angle)+yaw
            draw.line([project(x+RAY_ORIGIN*math.cos(theta),y+RAY_ORIGIN*math.sin(theta),.24),
                       project(x+(RAY_ORIGIN+distance)*math.cos(theta),y+(RAY_ORIGIN+distance)*math.sin(theta),.24)],fill=(130,130,130),width=1)
    gx,gy=project(*world.config['goal'],0)
    write(draw,(gx-30,gy+58),'ゴール',18)
    sx,sy=project(0,0,0);write(draw,(sx-25,sy+25),'開始',18)
    return image


def frame(world,record,row,trail,title,caption,hold=False):
    image=Image.new('RGB',(1280,720),'white');draw=ImageDraw.Draw(image)
    write(draw,(35,13),title,27)
    write(draw,(1000,19),'PAI最終課題  nini1723',18,GRAY)
    draw.line((35,65,1245,65),fill=(175,175,175),width=1)
    image.paste(scene(world,row,trail),(0,78))
    state=MODE[row['mode']]
    direction='左' if row['direction']>0 else '右' if row['direction']<0 else '未定'
    write(draw,(35,587),f"状態：{state}    回避方向：{direction}    推論 {row['calls']}回    追加観測 {row['dense_scans']}回",22)
    write(draw,(35,630),caption,24)
    note='説明のため2秒静止' if hold else '走行ログを等速再生'
    write(draw,(35,682),f"{row['t']:.2f}秒   {note}",17,GRAY)
    return image


def intro():
    image=Image.new('RGB',(1280,720),'white');draw=ImageDraw.Draw(image)
    write(draw,(85,145),'距離センサを用いた障害物回避',34)
    write(draw,(85,225),'通常時はゴールを追従し、障害物の近くで回避方向を判断する。',26)
    write(draw,(85,275),'判断が曖昧な場合にだけ、距離センサを5本から13本に増やす。',26)
    write(draw,(85,395),'PyBulletによる平面運動学シミュレーション',23)
    write(draw,(85,440),'実機や車輪の動力学は扱っていない。',23)
    write(draw,(85,595),'PAI最終課題   nini1723',22,GRAY)
    return image


def result_page(comparison):
    image=Image.new('RGB',(1280,720),'white');draw=ImageDraw.Draw(image)
    write(draw,(75,50),'24シーンでの比較結果',32)
    names={'fixed_left':'左に固定','fixed_right':'右に固定','event':'粗観測のみ','adaptive':'曖昧なときだけ追加観測','always_dense':'常に詳細観測'}
    columns=[75,535,720,955]
    for x,label in zip(columns,['方式','到達','最小間隔','平均総レイ数']):write(draw,(x,122),label,24)
    draw.line((75,166,1205,166),fill=(70,70,70),width=1)
    for i,method in enumerate(comparison['methods']):
        y=188+i*56;r=comparison['aggregate'][method]
        for x,text in zip(columns,[names[method],f"{r['successes']}/24",f"{r['minimum_clearance_m']:.3f}m",f"{r['mean_total_rays']:.0f}"]):write(draw,(x,y),text,24)
    draw.line((75,477,1205,477),fill=(130,130,130),width=1)
    write(draw,(75,505),'追加観測で左右の判断は変わったが、到達数は粗観測のみと同じだった。',25)
    write(draw,(75,552),'全方式で接触は0回。2シーンは全方式が時間切れになった。',25)
    write(draw,(75,625),'レイ数には通常観測とゴールへの通路確認も含む。失敗した試行も集計した。',20,GRAY)
    return image


def main(preview=False):
    OUT.mkdir(exist_ok=True)
    comparison=json.loads((DATA/'final'/'comparison.json').read_text(encoding='utf-8'))
    title=intro();title.save(OUT/'title.png')
    specs=[('00_left_open_adaptive','1. 右側に壁がある場合','右側の壁を検知し、左へ回避する。追加観測は行わない。'),
           ('01_right_open_adaptive','2. 左側に壁がある場合','壁の位置を反転すると、右へ回避する。'),
           ('03_ambiguous_right_adaptive','3. 粗い観測では判断しにくい場合','5本では左と予測したが、13本で観測し直すと右に変わった。')]
    if preview:
        record=json.loads((DATA/'final'/'runs'/f'{specs[2][0]}.json').read_text(encoding='utf-8'))
        world=simple_world(record['config'])
        try:
            row=next(f for f in record['frames'] if f['dense_sensors'] is not None);world.place(row['pose'])
            frame(world,record,row,[f['pose'][:2] for f in record['frames'] if f['t']<=row['t']],specs[2][1],specs[2][2],True).save(OUT/'preview.png')
        finally:world.close()
        result_page(comparison).save(OUT/'results.png');return
    video=OUT/'PAI_FastSlow_Demo.mp4';total=0
    with imageio.get_writer(video,fps=FPS,codec='libx264',quality=8,macro_block_size=16,ffmpeg_log_level='error') as writer:
        def append(image,count=1):
            nonlocal total
            array=np.asarray(image)
            for _ in range(count):writer.append_data(array);total+=1
        append(title,6*FPS)
        for name,heading,caption in specs:
            record=json.loads((DATA/'final'/'runs'/f'{name}.json').read_text(encoding='utf-8'))
            world=simple_world(record['config']);trail=[];stopped=False;scanned=False
            try:
                for i,row in enumerate(record['frames']):
                    trail.append(row['pose'][:2]);important=row['mode']=='SLOW_REOBSERVE' or row['dense_sensors'] is not None
                    if i%2 and not important:continue
                    world.place(row['pose']);image=frame(world,record,row,trail,heading,caption)
                    if i%2==0:append(image)
                    if row['mode']=='SLOW_REOBSERVE' and not stopped:
                        append(frame(world,record,row,trail,heading,'ここで停止し、13方向を観測する。',True),2*FPS);stopped=True
                    if row['dense_sensors'] is not None and not scanned:
                        image=frame(world,record,row,trail,heading,'追加観測後、右へ回避する判断に変わった。',True)
                        append(image,2*FPS);image.save(OUT/'observation.png');scanned=True
                last=record['frames'][-1];world.place(last['pose'])
                append(frame(world,record,last,trail,heading,'障害物に接触せず、ゴールに到達した。'),FPS)
            finally:world.close()
        results=result_page(comparison);results.save(OUT/'results.png');append(results,15*FPS)
        ending=Image.new('RGB',(1280,720),'white');draw=ImageDraw.Draw(ending)
        write(draw,(85,150),'残った課題',30)
        write(draw,(85,230),'狭い配置では、回避方向が合っていても時間切れになる。',25)
        write(draw,(85,280),'今回の検証は静止した障害物に限り、実機では試していない。',25)
        write(draw,(85,385),'コードと全試行の記録',23)
        write(draw,(85,430),'github.com/maruni-ni/PAI-FastSlow-Robot',23)
        write(draw,(85,570),'音声なし。保存した走行ログを等速再生し、追加観測の前後だけ2秒静止した。',20,GRAY)
        append(ending,8*FPS)
    metadata={'video':video.name,'fps':FPS,'frames':total,'duration_s':total/FPS,'sources':[name for name,_,_ in specs],
              'data_changed':False,'playback':'1x, two labelled 2-second holds, silent captions'}
    (OUT/'video_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8');print(json.dumps(metadata))


if __name__=='__main__':main('--preview' in sys.argv)
