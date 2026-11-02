# 距離センサを用いた障害物回避

PAI最終課題で作成した、PyBullet上の移動ロボット。
通常はゴールへ向かって進み、障害物の近くで学習済みモデルに左右の回避方向を判断させる。
5方向の観測で判断しにくい場合は、停止したまま13方向を観測し直す。

## 実行

確認環境はWindows x64、Python 3.11.16。
PyBullet 3.2.7はソースからビルドしたため、WindowsではC++ Build Toolsが必要になる場合がある。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py --scenario left_open --method adaptive --gui --realtime
```

学習済みモデルは `models/` に入っている。
画面を表示しない場合は `--gui --realtime` を外す。

```powershell
.\.venv\Scripts\python.exe main.py --scenario right_open --method adaptive
.\.venv\Scripts\python.exe main.py --scenario ambiguous_right --method adaptive
.\.venv\Scripts\python.exe view_demo.py
```

`view_demo.py` は保存済みの走行ログを再生する。qキーで終了。

## 制御と学習

通常時（Fast）は、ゴールの方位誤差から前進速度と旋回速度を決める。
障害物が近づくと停止し、距離センサとゴール方位をRandom Forestへ入力する。
最大出力確率が0.80未満なら詳細モデルで判断し直し、選んだ方向へ70度旋回して通過する。
ゴールへの通路が空いたら通常走行へ戻る。

センサは粗観測が5本、詳細観測が13本。距離の上限は4m。
角度と特徴量は [sensor_config.py](sensor_config.py) にまとめている。
ゴール位置は既知で、復帰判定にはゴールまで届く3本のレイも使う。

3000配置とその左右反転から6000観測を生成した。
詳細センサの左右の空き具合を評価する規則で教師ラベルを作り、粗・詳細モデルに同じラベルを与えた。
反転した配置が学習側と検証側に分かれないようにし、4500観測で学習、1500観測で検証した。
教師規則との一致率は粗75.7%、詳細97.8%だった。走行成功率とは別の値である。
実行中は教師規則を呼ばず、保存したモデルの出力で回避方向を決める。

## 結果

開発用8シーンとは別の24シーンで、5方式を比較した。
最終試行前にコード・モデル・配置を固定し、全120試行を保存した。

| 方式 | 到達 | 最小間隔m | 平均総レイ数 |
|---|---:|---:|---:|
| 左に固定 | 17/24 | 0.333 | 4722.0 |
| 右に固定 | 17/24 | 0.333 | 4722.0 |
| 粗観測のみ | 22/24 | 0.333 | 4291.3 |
| 必要時に追加観測 | 22/24 | 0.442 | 4301.8 |
| 常に詳細観測 | 22/24 | 0.442 | 11264.8 |

接触は全方式で0回。seed 90004の左右壁2シーンは全方式が時間切れになった。
失敗した試行も表に含めている。

追加観測で到達数は増えなかったが、最小間隔は広がった。
12シーンでは追加観測なし、残り12シーンでは各1回だった。
総レイ数は常時詳細観測より61.8%少なかった。
総レイ数には、通常の5本、追加の13本、ゴール通路確認の3本を含む。

粗観測のみの比較には、今回の教師で学習した粗モデルを使った。
常時詳細観測は毎ステップ観測するが、推論は接近時だけ行う。
各試行の時間、経路長、推論回数、処理時間は [comparison.json](output/final/comparison.json) に記録している。
処理時間は1台のPCで各1回測った値で、厳密な速度比較や電力評価は行っていない。

## 評価の再実行

```powershell
.\.venv\Scripts\python.exe evaluate.py --split dev
.\.venv\Scripts\python.exe verify.py
```

`--method` は `fixed_left`、`fixed_right`、`event`、`adaptive`、`always_dense` から選ぶ。
それぞれ左固定、右固定、粗観測のみ、必要時の追加観測、常時詳細観測に対応する。

最終比較の配置とハッシュは [freeze.json](output/final/freeze.json)、走行ログは [output/final/runs/](output/final/runs/) にある。
最終比較を再実行する場合は、別の作業コピーで `output/final` を別名に保存してから `evaluate.py --split final` を実行する。既存の結果は上書きしない。

再学習は `train_model.py`、入力を置き換えた場合のモデル感度の確認は `sensitivity.py` で行う。
同梱モデルで結果を確認する場合、再学習は不要。

## 制約

位置と姿勢を直接更新する平面運動学モデルで、車輪の摩擦やトルクは再現していない。
狭い配置では、回避方向が合っていても時間切れになる。
モデルの出力確率は未校正で、安全確率としては扱っていない。
実機、動く障害物、迷路では検証していない。

[PyBullet](https://github.com/bulletphysics/bullet3)

