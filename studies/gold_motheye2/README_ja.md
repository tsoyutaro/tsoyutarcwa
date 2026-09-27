# gold_motheye2: 測定金データによる Fourier 次数掃引

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

`converge.py` は `grid=256`、高さ分割数 `Nz=100` を固定し、
Fourier次数 `M=4,6,8,10,12,14,16,18,20` だけを変える。
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
