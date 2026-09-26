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
