"""走行結果を2ページのPDFにまとめる。"""
from pathlib import Path
import json
import fitz
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import black,Color
from make_video import ROOT,DATA,OUT,simple_world,scene

pdfmetrics.registerFont(TTFont('Mincho','C:/Windows/Fonts/msmincho.ttc',subfontIndex=0))
pdfmetrics.registerFont(TTFont('Gothic','C:/Windows/Fonts/msgothic.ttc',subfontIndex=0))
W,H=595.276,841.89


def make():
    comparison=json.loads((DATA/'final'/'comparison.json').read_text(encoding='utf-8'));a=comparison['aggregate']
    training=json.loads((ROOT/'models'/'training_metrics.json').read_text(encoding='utf-8'))
    record=json.loads((DATA/'final'/'runs'/'03_ambiguous_right_adaptive.json').read_text(encoding='utf-8'))
    world=simple_world(record['config'])
    try:
        row=next(f for f in record['frames'] if f['dense_sensors'] is not None);world.place(row['pose'])
        scene(world,row,[f['pose'][:2] for f in record['frames'] if f['t']<=row['t']]).save(OUT/'report_figure.png')
    finally:world.close()
    target=OUT/'説明資料_PAI最終課題_nini1723.pdf'
    c=canvas.Canvas(str(target),pagesize=(W,H));c.setTitle('距離センサを用いた障害物回避');c.setAuthor('nini1723')
    y=0
    def line(text,size=11,font='Mincho',x=48):
        nonlocal y
        c.setFillColor(black);c.setFont(font,size);c.drawString(x,y,text);y-=18
    def para(text,size=11):
        nonlocal y
        current=''
        for char in text:
            if pdfmetrics.stringWidth(current+char,'Mincho',size)>499:
                if char in '。、，．！？：；）］】」』%':
                    current+=char;line(current,size);current='';continue
                line(current,size);current=''
            current+=char
        if current:line(current,size)
        y-=5
    def heading(text):
        nonlocal y
        y-=8;line(text,12,'Gothic');y-=4
    def page(number):
        nonlocal y
        y=H-49
        c.setFillColor(Color(.4,.4,.4));c.setFont('Gothic',8)
        c.drawRightString(W-48,25,str(number))
    def table(headers,rows,widths):
        nonlocal y
        c.setStrokeColor(Color(.35,.35,.35));c.setLineWidth(.45);c.line(48,y+7,547,y+7)
        for index,values in enumerate([headers]+rows):
            x=52
            for value,width in zip(values,widths):
                c.setFont('Gothic' if index==0 else 'Mincho',10);c.setFillColor(black);c.drawString(x,y-8,str(value));x+=width
            y-=24
            if index==0:c.line(48,y+7,547,y+7)
        c.line(48,y+7,547,y+7);y-=10
    page(1)
    line('距離センサを用いた障害物回避',18,'Gothic');y-=12
    line('PAI最終課題　nini1723　　2026年11月2日',10,'Gothic');y-=10
    heading('1. 目的')
    para('普段は軽い制御でゴールへ進み、障害物の近くでだけ学習済みモデルに左右の回避方向を判断させる。判断しにくい場面では観測を増やすことで、回避方向や計算量がどう変わるかを調べた。')
    heading('2. 方法')
    para('PythonとPyBulletで移動ロボットを作成した。位置と姿勢を速度・旋回速度から更新する平面運動学モデルで、実機や車輪の摩擦・トルクは扱っていない。ゴール座標は既知とし、復帰判定にはゴールまで届くレイも使っている。')
    para('通常時（Fast）はゴール方位に向かうフィードバック制御を行う。前方の障害物が1.2m未満になると停止し、5方向の距離とゴール方位をRandom Forestへ入力する。モデルの左右出力を、そのまま回避する旋回方向に使う。')
    para('粗い観測での最大出力確率が0.80未満なら、姿勢を変えずに13方向を追加観測し、詳細モデルで判断し直す。追加観測は1イベントにつき1回までとした。これは観測密度を変えるAdaptive Sensingであり、視点を動かす方式ではない。')
    para('左右へ70度旋回した後は低速で通過する。機体幅を考慮したゴール向き3本のレイが0.45秒以上空き、ゴールへ近づいたら通常走行へ戻る。形状間隔が0.06m未満、回避が25秒超過、全体が60秒超過の場合は失敗停止とした。')
    image_h=185
    c.drawImage(str(OUT/'report_figure.png'),48,y-image_h,width=499,height=image_h,preserveAspectRatio=True,anchor='c')
    y-=image_h+17
    line('図1　追加観測を行った場面（青：ロボット、灰：障害物、緑：ゴール）',9,'Gothic')
    para('この場面では、5本の観測では左と予測したが、13本で観測し直すと右に変わった。左右を反転した配置では、追加観測後の回避方向も反転した。',10)
    assert y>35,('page1 overflow',y)
    c.showPage();page(2)
    heading('3. 学習と比較条件')
    para('合成した3000配置とその左右反転から6000観測を生成した。詳細センサの左右の空き具合を評価する幾何学規則を教師とし、粗・詳細の両モデルに同じラベルを与えた。配置とその反転を同じグループにまとめ、4500観測で学習、別の1500観測で検証した。')
    coarse=training['models']['coarse']['teacher_agreement']*100;dense=training['models']['dense']['teacher_agreement']*100
    para(f'教師規則との一致率は粗モデル{coarse:.1f}%、詳細モデル{dense:.1f}%だった。これは規則の再現度であり、走行の成功率ではない。実行中は教師規則を呼ばず、保存したモデルだけで方向を決める。')
    para('開発用8シーンとは別に、左右の壁と細い障害物を含む24シーンを用意し、5方式で計120試行を行った。最終試行の前にコード・モデル・配置を固定した。結果を見て配置や制御を調整せず、失敗も含めて保存した。')
    heading('4. 結果')
    names={'fixed_left':'左に固定','fixed_right':'右に固定','event':'粗観測のみ','adaptive':'必要時に追加観測','always_dense':'常に詳細観測'}
    rows=[]
    for method in comparison['methods']:
        r=a[method];rows.append([names[method],f"{r['successes']}/24",f"{r['minimum_clearance_m']:.3f}",f"{r['mean_dense_scans']:.2f}",f"{r['mean_total_rays']:.0f}"])
    table(['方式','到達','最小間隔m','平均追加観測','平均総レイ数'],rows,[155,62,87,94,101])
    para('接触は全120試行で0回だった。seed 90004の左右壁2シーンは全方式が時間切れになり、未到達として集計した。追加観測で到達数は増えなかったが、全試行の最小間隔は粗観測の0.333mから0.442mへ増えた。')
    para('必要時に追加観測する方式では、12シーンが追加0回、残り12シーンが各1回だった。通常の5本、追加の13本、ゴール通路確認の3本をすべて数えると、総レイ数は常時詳細観測より61.8%少なかった。粗観測のみと比べると、追加の観測と推論の分だけ処理は増える。')
    heading('5. 考察と限界')
    para('粗いセンサでは同じ値になる配置でも、追加観測によって左右を選び分けられた。一方、狭い配置では方向が合っていても時間切れになった。到達率を上げるには、方向の選択だけでなく、回避中の経路の修正が必要だと考える。')
    para('最大出力確率は未校正で、安全確率としては解釈していない。今回の結果は限定した静止障害物でのもので、実機、迷路、動く障害物、消費電力は未検証である。',10)
    y-=3;line('コード・モデル・全試行ログ',10,'Gothic')
    line('https://github.com/maruni-ni/PAI-FastSlow-Robot',10,'Gothic')
    para('再現手順と処理時間などの詳細は、READMEとoutput/final/comparison.jsonを参照。',9)
    assert y>35,('page2 overflow',y)
    c.save();doc=fitz.open(target);assert len(doc)==2
    for i,page_obj in enumerate(doc):page_obj.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(OUT/f'pdf-page-{i+1}.png')
    print(target)


if __name__=='__main__':make()
