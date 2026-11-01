"""Create and rasterize a three-page report from the frozen comparison."""
import json
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
import fitz
from main import ROOT

OUT=ROOT/'output';WIDTH,HEIGHT=595.276,841.89
pdfmetrics.registerFont(TTFont('JP','C:/Windows/Fonts/msgothic.ttc',subfontIndex=0))
INK=HexColor('#122238');ACCENT=HexColor('#087e96');GRAY=HexColor('#50647a')
NAMES={'fixed_left':'固定左','fixed_right':'固定右','event':'粗観測1回','adaptive':'適応的観測','always_dense':'常時詳細観測'}


def make():
    report=json.loads((OUT/'final'/'comparison.json').read_text(encoding='utf-8'));a=report['aggregate']
    training=json.loads((ROOT/'models'/'training_metrics.json').read_text(encoding='utf-8'))
    video=json.loads((OUT/'video_metadata.json').read_text(encoding='utf-8'))
    sensitivity=json.loads((OUT/'sensitivity.json').read_text(encoding='utf-8'))
    path=OUT/'説明資料_PAI最終課題_nini1723.pdf'
    c=canvas.Canvas(str(path),pagesize=(WIDTH,HEIGHT));c.setTitle('Fast/Slow制御とAdaptive Sensing');c.setAuthor('')
    y=0
    def text(line,size=10,color=INK,x=43):
        nonlocal y
        c.setFillColor(color);c.setFont('JP',size);c.drawString(x,y,line);y-=size*1.65
    def paragraph(content,size=10):
        nonlocal y
        line=''
        for ch in content:
            if ch=='\n' or pdfmetrics.stringWidth(line+ch,'JP',size)>509:
                text(line,size);line=''
                if ch=='\n':continue
            line+=ch
        if line:text(line,size)
        y-=6
    def section(title):
        nonlocal y
        y-=9;text(title,13,ACCENT);y-=3
    def page(number):
        nonlocal y
        c.setFillColor(ACCENT);c.rect(0,HEIGHT-12,WIDTH,12,fill=1,stroke=0)
        y=HEIGHT-43;text('PHYSICAL AI / FINAL PROJECT / nini1723',10,ACCENT);y-=6
        c.setFillColor(GRAY);c.setFont('JP',8)
        c.drawString(43,26,'2026-11-02 | PyBullet / Kinematic simulation / Adaptive Sensing')
        c.drawRightString(WIDTH-43,26,f'{number} / 3')
    def table(headers,rows,widths):
        nonlocal y
        x0=43;rowheight=26
        for i,row in enumerate([headers]+rows):
            c.setFillColor(HexColor('#e5f4f6') if i==0 else HexColor('#f3f6f9') if i%2 else HexColor('#ffffff'))
            c.rect(x0,y-rowheight+5,sum(widths),rowheight,fill=1,stroke=0)
            x=x0+5
            for value,width in zip(row,widths):
                c.setFillColor(INK);c.setFont('JP',9);c.drawString(x,y-10,str(value));x+=width
            y-=rowheight
        y-=10
    page(1)
    text('曖昧なときだけ、観測を増やす',21)
    text('イベント駆動Fast/Slow制御とAdaptive Sensing',13,GRAY)
    section('1. 目的：左右判断と追加観測の役割を示す')
    paragraph('通常は軽量なゴール追従制御で移動する。障害物に接近したら学習済みモデルで左右を判断し、不明瞭な場合だけ停止して観測密度を5本から13本に増やす。モデルの出力を実際の旋回方向に反映する。')
    paragraph('FAST → SLOW_DECIDE →（曖昧なら SLOW_REOBSERVE）→ SLOW_AVOID → FAST → GOAL',9)
    section('2. 追加観測による判断の更新')
    image_y=y-275
    c.drawImage(str(OUT/'adaptive_evidence.png'),43,image_y,width=509,height=286.3,preserveAspectRatio=True,mask='auto')
    y=image_y-15
    text('図1：詳細スキャン後、右回避へ更新。赤はオフライン入力置換に敏感なレイ。',8,GRAY)
    paragraph('左右を鏡映した細い障害物は、接近時の5本センサでは同じ観測値になる。追加した15度・30度などのレイが差を検知し、左右の配置に応じて異なる判断を得た。明瞭な壁の配置では追加観測なしで左右を選択した。')
    section('3. シミュレーションと安全処理')
    paragraph('Windows / Python 3.11 / PyBullet。速度・旋回速度で位置と姿勢を直接更新する平面運動学モデルで、車輪トルクや摩擦を再現する動力学ではない。30Hz制御、移動を4分割し形状間隔を監視。間隔0.06m未満・回避25秒超過・全体60秒で失敗停止する。')
    c.showPage();page(2);text('同一シーンでの5方式比較',20)
    paragraph('開発用8シーンとは異なるseed 90000-90005で24シーンを生成した。左右に壁がある鏡映対12、細い障害物の鏡映対12を全方式で走行。最初の実行前にコード・モデルのハッシュと配置を凍結し、120試行すべてを保存した。最終結果を見て配置・制御を調整していない。')
    section('4. 到達・接触・経路の結果')
    rows=[]
    for method in report['methods']:
        r=a[method];rows.append([NAMES[method],f"{r['successes']}/24",r['collisions'],r['safe_stops_or_timeouts'],f"{r['minimum_clearance_m']:.3f}",f"{r['mean_path_length_m']:.2f}",f"{r['mean_duration_s']:.2f}"])
    table(['方式','到達','接触','停止等','最小m','平均経路m','平均秒'],rows,[100,48,43,49,67,103,99])
    paragraph('停止等＝接触せず安全停止または時間切れ。平均経路・平均秒は失敗も含むため、成功率の異なる方式間の速さを直接示す指標ではない。適応的観測は粗観測と同じ22/24で到達率の改善なし。一方、全試行の最小間隔は0.333mから0.442mへ増えた。',9)
    section('5. 観測と推論の実測コスト（1試行あたり平均）')
    rows=[]
    for method in report['methods']:
        r=a[method];rows.append([NAMES[method],f"{r['mean_inference_calls']:.2f}",f"{r['mean_dense_scans']:.2f}",f"{r['mean_total_rays']:.1f}",f"{r['mean_inference_ms']:.2f}",f"{r['mean_loop_wall_ms']:.2f}"])
    table(['方式','推論回数','詳細回数','総レイ','推論ms','ループms'],rows,[100,76,76,100,76,81])
    reduction=100*(1-a['adaptive']['mean_total_rays']/a['always_dense']['mean_total_rays'])
    paragraph(f"総レイは毎制御の粗観測5本、条件付き詳細13本、ゴール通路確認3本を全て集計。適応的観測では24試行中12試行で追加観測0回、12試行で各1回。常時詳細観測に比べ総レイは{reduction:.1f}%少なかった。粗観測方式に比べると追加観測・追加推論の費用が増えている。",9)
    paragraph('常時詳細観測は毎ステップ13本を追加するが、推論は接近イベント時のみ。粗観測1回はv2で再学習した粗モデルを使う対照であり、旧版モデルの完全な再現ではない。',9)
    paragraph('時間はperf_counterで1回ずつ実測し、方式の実行順をシーンごとに乱択した。ループ時間はセンサ・衝突照会・推論・制御・メモリ内ログを含み、ワールド初期化・モデル読込・ファイル保存・描画は除く。単一PCの短い試行で、厳密な速度優位・消費電力削減を保証しない。',9)
    section('6. 失敗も保持する')
    paragraph('seed 90004の左右壁2シーンでは、全方式が回避時間上限に達した。接触なしでも未到達は失敗であり、比較の分母から除外していない。単純な旋回と直進を組み合わせた回避制御の限界として残した。',9)
    c.showPage();page(3);text('学習方法・検証・限界',20)
    section('7. 二つのモデルと同じ教師')
    paragraph('粗観測は[90,45,0,-45,-90]度の5本、詳細観測は90度から-90度まで15度刻みの13本。最大距離4mで正規化し、ゴール方位誤差のsin/cosを加える。特徴数は粗7・詳細15。角度、順序、原点オフセット、正規化を共通定義し、保存モデルのスキーマを読込時に検査する。')
    paragraph('3000個の合成配置とその左右鏡映から6000観測を生成した。詳細観測の左右15-75度の距離の最小値・平均値とゴール方位から教師ラベルを作り、粗・詳細の両モデルに同じ教師を与えた。Random Forest各80本、最大深さ14、最小葉5。鏡映対を同一グループとし、2250配置/4500観測で学習、別750配置/1500観測で分類評価した。')
    coarse_acc=training['models']['coarse']['teacher_agreement']*100;dense_acc=training['models']['dense']['teacher_agreement']*100
    paragraph(f"未学習配置での教師規則一致率は粗{coarse_acc:.1f}%、詳細{dense_acc:.1f}%。これは幾何学規則の再現度であり、実環境での成功率ではない。実行中は教師規則を呼ばず、保存モデルのpredict_probaで左右を決める。")
    section('8. 適応的観測の条件と検証')
    paragraph('粗モデルの最大出力確率が0.80未満なら、機体を停止したまま13本の追加観測を行い、詳細モデルで1回だけ再推論する。0.80は開発時に固定した暫定閾値であり、校正された事故確率ではない。詳細モデルも曖昧な場合は、その最大クラスを使うため安全保証はない。追加観測はイベントにつき最大1回で、際限なく再試行しない。')
    paragraph('機体の姿勢を変えないため、方式名はAdaptive Sensing（適応的観測）とした。視点変更を伴うActive Perceptionとは呼ばない。粗観測の一致・詳細観測後の判断変化・明瞭時の追加観測なし・停止中観測・レイ数の集計を自動確認した。学習用の解析レイとPyBulletレイの数値も照合した。')
    section('9. 判断の局所的な感度（補助解析）')
    top=max(sensitivity['sensors'],key=lambda r:abs(r['delta_p_left']))
    paragraph(f"右回避を選んだ一例で、各詳細センサを1本ずつ最大距離4mに置換してP(LEFT)の変化を求めた。{top['angle_deg']}度の距離を置換した際の変化は{top['delta_p_left']:+.3f}だった。これは置換方法に依存する局所入力感度で、因果説明や安全根拠ではない。オフライン14回の推論で、走行コストには含めていない。")
    section('10. 再現と残る課題')
    paragraph('READMEに新規環境向けの導入を先頭に記載し、学習済みモデル・凍結設定・全最終走行ログ・評価JSONを同梱する。既存版のZIP/MP4/PDFは変更していない。GitHub公開・YouTubeアップロード・OmniCampus提出はユーザーが担当する。')
    paragraph(f"動画は{video['duration_s']:.1f}秒。保存ログを等速再生し、追加観測前後に明示した2秒静止と説明カードを加えた。機能は静止した限定配置でのPoCであり、迷路、動く障害物、動力学、実機、VLA、電力評価は未検証。ROS等の利用可否を含む正式な採点要件は本資料では独立確認していない。",9)
    c.save()
    doc=fitz.open(path);assert len(doc)==3
    for index,page_obj in enumerate(doc):
        page_obj.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(OUT/f'report-page-{index+1}.png')
        blocks=page_obj.get_text('blocks');assert all(b[3]<HEIGHT+1 for b in blocks)
    print(path)


if __name__=='__main__':make()
