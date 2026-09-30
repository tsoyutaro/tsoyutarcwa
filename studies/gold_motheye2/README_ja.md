# gold_motheye2: 測定金データによる Fourier 次数掃引

## 推奨: 7波長平均の勾配を使うAdam

`optimize_adam_fullband.py` は400–700 nmの7波長の反射率を台形則で平均し、
**各波長の重み付き勾配をすべて足してから**Adamを1回更新する。
7波長は1つずつRCWA計算と逆伝播を行うので、7波長分の計算時間は必要だが、
7つの計算グラフを同時にGPUメモリへ保持しない。途中の勾配も波長ごとに保存して再開できる。
形状・測定金CSV・100層・M=8学習/M=16比較は下記の逐次更新版と共通。

```bash
python3 studies/gold_motheye2/optimize_adam_fullband.py --prepare-only
python3 studies/gold_motheye2/optimize_adam_fullband.py --device cuda
```

`torch.autograd.grad` が `does not require grad` で停止した場合は、
`optimize_adam.py` と `optimize_adam_fullband.py` の修正版を両方配置し、
同じ出力先・同じ計算条件で上記の計算コマンドを再実行する。
円形 matched-ASR は固定形状の評価中も座標変換の微分を必要とするため、
修正版はその評価中に `torch.enable_grad()` を使う。
既存の `checkpoint.json` は削除せず、その続きから再開できる。

既定の28更新は、**7波長それぞれを28回**使う（計196回の学習用波長計算）。
4更新ごとに7点平均を評価して最良形状を選び、最後に円錐と最良形状をM=16で比較する。
平均が終盤も低下するなら `--steps 56` で同じチェックポイントから延長する。
出力先は `results/adam_fullband_Nz100_M8`。`band_mean_history.svg` で経過を確認できる。
M=16での比較は最適化形状の次数収束を証明しない。必要なら同じチェックポイントで
`--verify-order 18` または20を指定して再評価する。

7波長だけで連続した400–700 nm全体の低反射は保証できない。
学習に用いなかった中間波長を含む10 nm刻みの独立確認は、最適化完了後に
同じコマンドへ次を追加して再実行する。学習は再実行されず、M=16で円錐と
最良形状の両方を計算し、`dense_validation_*.json` と `.svg` に保存する。

```bash
python3 studies/gold_motheye2/optimize_adam_fullband.py --device cuda --validation-wavelengths 400:700:10
```

## 軽量な1波長逐次更新（比較用）

`optimize_adam.py` は Optuna を使わず、既存の金モスアイモデルで全反射率の
400–700 nm 帯域平均を下げる。まず計算前に形状を確認できる。

```bash
python3 studies/gold_motheye2/optimize_adam.py --prepare-only
```

`results/adam_variable_endpoints_Nz100_M8/cone_profile.svg` は軸方向断面、
`cone_profile.csv` は上から下への100層の半径を示す。初期形状は直線円錐。
旧 `results/adam_Nz100_M8` は端点固定版の準備結果として残し、
端点可変版は別の出力先に保存する。
周期200 nm、構造高さ500 nm、三角格子、金柱と半無限金基板、
空気中からの正入射x偏光を固定する。先端直径10 nm、底面直径190 nmは
**初期形状**だけの値で、両方をAdamで最適化する。
金の複素誘電率は `data/au_measured_nk.csv` から読む。
先端、底面、中間の半径分布を合わせて最適化する。
既定の直径探索範囲は `0.1 < D_tip < D_base < 199.9 nm`。
`--diameter-margin-nm` で両端の数値上の余裕を変えられる。
厳密な直径0および周期と一致する直径200 nmは、現行の円形matched-ASR計算では
扱えないため探索範囲に含めない。
形状は上から下へ**厳密に半径が増加**する8区間の折れ線で、
softmaxによる正の区間増分に小さな下限を付ける。
実際のRCWA計算はその曲線を高さ方向100層の円柱で近似する。
`cone_geometry.json` と `best_geometry.json` に実際の端点直径と単調性判定を保存する。

CUDA版PyTorchを使えるLinux環境で、プロジェクトルートから実行する。

```bash
python3 studies/gold_motheye2/optimize_adam.py --device cuda
```

既定では7波長（400, 450, ..., 700 nm）を順に1波長ずつAdamで140回更新する
（20巡）。7点全体の評価は2巡ごと（14更新ごと）に行う。
学習率は0.05固定、Adamの β1=0.9、β2=0.999、ε=1e-8、勾配ノルム上限1。
平均は波長に関する台形則で、数値は0–1の分率。
140回も収束を保証する回数ではないため、`evaluations.csv` の帯域平均が終盤も
改善し続ける場合は更新を増やす。
学習次数は M=8、高さ分割 Nz=100、ASR grid=256、complex128。
最後に最良形状と直線円錐の両方を M=16 で同じ7点について比較する。
学習中の最良形状は M=8 の全7点平均で選ぶ。
中断時は同じコマンドで `checkpoint.json` から再開する。
短い動作確認だけなら `--steps 28 --verify-order 0` を指定し、その後に
既定の140回まで継続できる。さらに更新するには `--steps 280` などと指定する。
設定を変える場合は別の `--output-dir` を指定する。

`summary.json` に両次数での平均反射率と、高次数での差
`cone_mean_M_verify - best_mean_M_verify` が保存される。
正なら最適化形状の反射率が低い。`comparison.svg` は各波長の比較、
`band_mean_history.svg` は更新回数に対する帯域平均と最良値の推移、
`best_profile.svg` と `.csv` は最良形状、`history.csv` と `evaluations.csv` は
途中経過。最良形状が円錐のままの場合は、試した範囲で改善が見つからなかったことを示す。
`summary.json` には最良形状の先端・底面直径も保存する。

検証を別日に行う場合は最初に `--verify-order 0` を指定し、完了後、
`--steps 140 --verify-order 16` で同じ出力先を再実行する。
さらに M=18 や20でも同じ候補と円錐を比較できる。高次数での改善が
見られても、その新形状の Fourier 次数収束までは保証されない。
また7点の平均は連続帯域平均の近似なので、必要なら波長刻みを10 nmにした
別の出力先で再試験する。高次数や細かい波長刻みは時間がかかる。
100層の逆伝播でGPUメモリが足りない場合は、たとえば
`--order 6 --output-dir studies/gold_motheye2/results/adam_variable_endpoints_Nz100_M6`
を付けた別実験として試す。この場合も最終比較はM=16で行い、学習次数の結果だけで
改善を確定しない。

この設定は**金柱が金基板に連続するモデル**であり、PMMAモスアイ上の
30 nm蒸着金膜とは別の物理構造である。

## 金の CSV を置く場所

TSUBAME 上でプロジェクトルート（`rcwa_solver_auto.py` があるディレクトリ）から見た
`studies/gold_motheye2/data/au_measured_nk.csv` に置く。
このファイルは計算で読む整形済みCSVで、列は `wavelength_nm,n,k`。
添付された元の2ブロックCSVは `data/au_measured_nk_original.csv` に保存した。

元CSVの `wl` は数値範囲 0.1879～1.9370 から **µm と仮定**し、nmへ1000倍した。
49点の `n` と `k` は各波長で一致し、整形済み範囲は187.9～1937.0 nm。
既定の400、550、700 nmを含む。計算時には `(n+i k)^2` を金の複素誘電率とし、
表の点間は誘電率を線形補間する。別のCSVを使う場合は
`--gold-csv /path/to/wavelength_nm_n_k.csv` を指定する。

## 固定する数値条件と図

`converge.py` は `grid=256` と1回の実行中の高さ分割数を固定し、
Fourier次数 `M=4,6,8,10,12,14,16,18,20` だけを変える。層数の既定値は `Nz=100`。
波長は既定で400、550、700 nm。形状・入射条件は `studies/gold_motheye/converge.py` の
既定値（周期200 nm、高さ500 nm、三角配列、半無限金基板、正入射x偏光）を使う。

出力図は次の2枚。どちらも標準ライブラリだけで作るSVGで、ブラウザで開ける。

- `results/order_sweep/reflectance_vs_order.svg`: 各波長の R 対 M。
  波長ごとに縦軸を調整し、最高計算次数のRを中心に±0.5 percentage pointの帯を示す。
- `results/order_sweep/runtime_vs_order.svg`: 各波長の1ケースの実行時間と、
  全波長がそろった次数における合計時間の対M。

CSV・JSON・checkpointは同じ結果ディレクトリに保存する。各ケース終了後に保存するため、
途中停止後に同じ条件で再実行すると計算済みケースを再利用できる。
データ、物理モデル、次数・波長、solverソースが変わった場合は別の `--output-dir` を指定する。

## TSUBAME 4.0 で実行

プロジェクトルートから以下を実行する。PyTorchのCUDA版を使える環境が必要。

```bash
qsub -g YOUR_TSUBAME_GROUP studies/gold_motheye2/tsubame4_order.sh
qstat
```

ジョブは `gpu_1=1`、最大12時間を指定する。直接実行するなら次の通り。

```bash
python3 studies/gold_motheye2/converge.py --device cuda
```

15層に固定して次数ごとの反射率と計算時間を調べる場合:

```bash
python3 studies/gold_motheye2/converge.py --device cuda --slices 15
```

既定の次数・波長を使い、結果は `results/order_sweep_Nz15/` に保存する。
反射率は `reflectance_vs_order.svg`、各波長および3波長合計の時間は
`runtime_vs_order.svg` で確認できる。同条件の再実行ではcheckpointから再開する。
15層での次数収束が確認できても、140層など最終層数における次数収束を保証するものではない。

波長・次数を変更する例:

```bash
python3 studies/gold_motheye2/converge.py --device cuda \
  --orders 4,6,8,10,12 --wavelengths 400,550,700 \
  --output-dir studies/gold_motheye2/results/order_sweep_short
```

既存CSVから図だけを再生成する場合は、同じ結果ディレクトリで:

```bash
python3 studies/gold_motheye2/converge.py --plot-only \
  --output-dir studies/gold_motheye2/results/order_sweep
```

収束判定は、**各波長について最高次数側の連続2ステップの反射率変化がともに0.005以下**
（0.5 percentage point以下）の場合に限る。これは試した次数・3波長での判定であり、
全波長のスペクトルや層数・gridの収束を保証しない。時間は1ケースのsolver実行時間で、
ジョブ待ち時間や図作成時間は含まない。

## 15層の高次側と変換行列の条件数

15層の既存次数掃引は700 nmでM=18から20への反射率差が約12.14 percentage points
あり、`not_converged`。追加次数を最終結果とみなす前に、次の診断を行う。

```bash
python3 studies/gold_motheye2/diagnose_transform.py --device cuda \
  --slices 15 --orders 16,18,20 --wavelength-nm 700
```

別ディレクトリ `results/transform_condition_Nz15_700nm/` の
`transform_condition_numbers.csv` に各層の変換行列の2ノルム条件数を記録する。
`transform_condition_summary.csv` は次数ごとの最大条件数と、その層番号、
再計算した反射率・計算時間をまとめる。
これはD6計算で逆行列を取る三角格子star上の横方向ASR変換行列であり、
各次数の最大値・診断に追加したSVD時間も標準出力に表示する。
条件数の大きさだけでは、反射率の収束・非収束や原因を確定できない。
この診断は計算済みの `order_sweep_Nz15` を変更せず、対象のケースを再計算する。
条件数と反射率を見た後で高次側を調べる場合は、既存27ケースを再計算せずに
次数22と24を追加できる:

```bash
python3 studies/gold_motheye2/converge.py --device cuda --slices 15 \
  --append-orders 22,24
```

追加後も700 nmの反射率が大きく変わるなら、15層での次数収束は未確認のままとする。

M=32でGPUメモリ不足になり、M=30までの全波長が保存済みの場合は、
計画から未計算のM=32だけを外してCSV・checkpoint・図を確定できる。
GPUやPyTorchは不要で、M=30までの計算は繰り返さない。

```bash
python3 studies/gold_motheye2/converge.py --slices 15 \
  --finalize-through-order 30
```

実行前にcheckpointの署名と、M=30までの全ケースがそろい、それより高い次数の
計算済みケースがないことを確認する。元のcheckpoint・metadata・CSVは
`results/order_sweep_Nz15/backup_before_finalize_M32/` に保存される。
M=30までで反射率が安定しなければ判定は `not_converged` のまま。

TSUBAMEの `gpu_1`、`qsub` の使い方は
[公式のジョブ手引き](https://www.t4.cii.isct.ac.jp/docs/handbook.ja/jobs/)を参照。

## 400～700 nm の101点反射スペクトル

次数収束用の `converge.py` は複数次数で各波長を比較するためのスクリプトであり、
波長を横軸にしたスペクトル図は作らない。単一次数のスペクトルには `spectrum.py` を使う。
既定は `M=16`、`Nz=100`、`grid=256`、400～700 nm の101点（3 nm刻み）。
TSUBAMEのプロジェクトルートから以下を投入する。

```bash
qsub -g YOUR_TSUBAME_GROUP studies/gold_motheye2/tsubame4_spectrum.sh
```

検証用に `M=20` のスペクトルを別ジョブで計算する場合は次の通り。

```bash
qsub -g YOUR_TSUBAME_GROUP -v SPECTRUM_ORDER=20 \
  studies/gold_motheye2/tsubame4_spectrum.sh
```

出力は次数別ディレクトリの `reflectance_spectrum.csv` と
`reflectance_spectrum.svg`。M=16なら
`studies/gold_motheye2/results/spectrum_M16/` に保存される。
各波長の終了時にcheckpoint・CSV・部分SVGを更新し、同条件で再実行すれば続きから計算する。
既存CSVから図だけ再生成するには以下を実行する。

```bash
python3 studies/gold_motheye2/spectrum.py --order 16 --plot-only
```

対話的なGPU割当がある場合は、直接
`python3 studies/gold_motheye2/spectrum.py --device cuda --order 16`
でも実行できる。M=16の既存計測値は1波長あたり約103秒なので、101点の
前進計算だけで概算約2時間55分を見込む。計算環境や波長により変動する。
この図は固定次数のスペクトルであり、最適化した形状の次数・層数・gridの収束を
自動的に保証するものではない。

## 層数 Nz に対する反射率と計算時間

`converge_slices.py` は測定金データ、形状、高さ500 nm、周期200 nm、
ASR grid=256を固定し、層数だけを
`10,15,20,30,40,50,60,70,80,90,100` に変える。
既定次数はM=16。波長は400、550、700 nmで、別の波長を使う場合は
`--wavelengths` にカンマ区切りで指定する。次数M=18も指定できる。

GPU計算ノードを確保済みなら、プロジェクトルートで次を実行する。

```bash
python3 studies/gold_motheye2/converge_slices.py --device cuda --order 16
python3 studies/gold_motheye2/converge_slices.py --device cuda --order 18
```

2行をそのまま入力すると順番に実行する。TSUBAMEのバッチジョブを投入するなら
以下を利用できる。

```bash
qsub -g YOUR_TSUBAME_GROUP studies/gold_motheye2/tsubame4_slices.sh
qsub -g YOUR_TSUBAME_GROUP -v SWEEP_ORDER=18 \
  studies/gold_motheye2/tsubame4_slices.sh
```

各次数は独立して `results/slice_sweep_M16/` または
`results/slice_sweep_M18/` に次を出力する。

- `reflectance_vs_slices.svg`: 各波長の反射率対Nz。波長ごとに縦軸を調整する。
- `runtime_vs_slices.svg`: 各波長の計算時間と、3波長の合計時間の対Nz。
- `slice_sweep.csv`, `slice_sweep.json`, `slice_checkpoint.json`:
  生の結果、条件・末尾2ステップの判定、再開用checkpoint。

ケースごとにCSV・図・checkpointを更新するので、同条件で再実行すれば続きを計算できる。
条件や金データを変更する場合は `--output-dir` で別の場所を指定する。
図だけを再生成する場合は以下を実行する。

```bash
python3 studies/gold_motheye2/converge_slices.py --order 16 --plot-only
```

判定値は反射率の絶対変化0.005（0.5 percentage point）。Nz=80→90と90→100の
両方が全波長でこれ以下なら `converged_within_tested_slices` と記録する。
収束判定にかかわらず完了した図を出力する。次数・grid・全スペクトルの収束は
この層数掃引とは別に確認する。

### 計算済みの層数に110～140層を追加する

M=18の既存結果を残して上限を拡張するときは、更新版のスクリプトで
次のコマンドを実行する。古い10～100層を指定し直す必要はない。

```bash
python3 studies/gold_motheye2/converge_slices.py --device cuda --order 18 \
  --append-slices 110,120,130,140
```

同じ `results/slice_sweep_M18/` のcheckpointから、金データ、形状、次数、grid、
波長、solver、cascadeの設定が一致することを確認し、計算済みのケースを再利用する。
新しく追加したNzの3波長だけを解き、CSV・2枚のSVG・収束判定を更新する。
中断した場合は同じコマンドで再開できる。新しい層数は既存最大Nzより大きい値に限る。
最後の2段階の判定対象は、拡張後はNz=120→130と130→140に変わる。
