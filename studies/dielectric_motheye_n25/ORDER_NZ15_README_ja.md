# 15層固定の回折次数収束：正方／六方格子、NV／ASR、3波長

20260806フォルダの既存ソルバーで、無損失n=2.5のモスアイを計算する。
高さ分割を**Nz=15**に固定し、正方格子と六方対称（三角Bravais）格子、
NVとmatched-ASR、400／550／700 nmの**12組を独立に**掃引する。
勾配監査や61点スペクトルは実行しない。

## 実行

GPUを確保したTSUBAME等の環境で、更新したファイルをプロジェクトのルートへ反映する。
`rcwa_ext`を含めて現時点の20260806コードを使用する。
配布ZIPは必要な現行ソルバーと依存する研究用モジュールを含む。
既存プロジェクトへ展開しても、ZIPを別フォルダへ展開して実行してもよい。
外部依存はPyTorch（GPU実行時はCUDA版）、torcwa、NumPy。図にはMatplotlibが必要。

```bash
python run_n25_motheye_order_until_oom.py --device cuda
```

M=4,6,8,…を各組で増やす。実行順は次のとおり。

1. 正方NV：400 → 550 → 700 nm
2. 正方ASR：400 → 550 → 700 nm
3. 六方NV：400 → 550 → 700 nm
4. 六方ASR：400 → 550 → 700 nm

各次数を別プロセスで実行する。OOMが出た波長・格子・手法の組はそこで打ち切り、
子プロセスの終了でそのCUDAコンテキストとメモリを解放してから、次の組へ進む。
OOMが出ても残りの波長や手法を省略しない。成功した次数は毎回保存する。
収束判定が通っても、指定どおりOOMまたは明示した上限まで掃引する。

既定の出力先：

```text
studies/dielectric_motheye_n25/results/order_Nz15_Q512_until_oom/
```

設定確認だけ：

```bash
python run_n25_motheye_order_until_oom.py --device cuda --dry-run
```

先にM=12まで計算する場合：

```bash
python run_n25_motheye_order_until_oom.py --device cuda --max-order 12
```

同じ出力先で上限を外して再実行すると、完了したMを再計算せず続きを計算する。
同じコマンドは中断からの再開にも使用する。OOMで停止済みの組は既定で再試行しない。
GPUメモリを空けた後にOOM次数を再試行する場合は`--retry-oom`を加える。
同じ出力先への同時実行はしない。

```bash
python run_n25_motheye_order_until_oom.py --device cuda --retry-oom
```

格子・手法の一部だけも指定でき、同じ設定の別組は同じ出力先へ順に追加できる。

```bash
python run_n25_motheye_order_until_oom.py --device cuda --lattices square --methods asr
```

図だけ再生成：

```bash
python run_n25_motheye_order_until_oom.py --report-only
```

## 固定条件と次数の意味

| 条件 | 値 |
|---|---|
| モスアイと半無限基板 | n=2.5、k=0、ε=6.25 |
| 最近接周期P／高さH | 200 nm／500 nm |
| 先端半径／底部半径 | 5 nm／95 nm |
| 高さ分割 | 15層、各層の中点半径、層厚500/15 nm |
| 入射 | 空気から垂直入射、x偏光 |
| NV投影／ASR積分 | **Q=512固定**、512×512 |
| NV | 円の解析Fourier係数、法線投影、Lanczosフィルタなし |
| ASR | local-radial、G=0.03、一般化因数分解、固定公称円周 |
| 共通ソルバー | complex128、Redheffer接続、half S行列、場保存なし |
| 対称性 | 正方C2vのx入射セクター／六方D6の入射行 |
| 掃引 | M=4から2刻み、既定では次数の上限を指定しない |

Mは保持するFourier回折次数の打ち切りであり、伝搬している回折次数の数ではない。
既定のscalar harmonic数は正方(2M+1)²、六方3M(M+1)+1。
実際の固有値問題の縮約次元もCSVへ記録する。
R/Tは**全伝搬回折次数のフラックスの合計**で、ゼロ次だけではない。
400 nmでは基板側に複数の伝搬回折次数がある。

Qは途中で自動変更しない。ソルバーのサンプリング条件Q≥4M+4を守るため、
Q=512ではM≤127、既定の偶数列ではM=126までが上限となる。
先にこの限界へ達したら`stopped_grid_limit`と記録し、次の組へ進む。
これをOOMとして扱わない。Qを増やす場合は別出力先で全次数を計算し直す。

```bash
python run_n25_motheye_order_until_oom.py --device cuda --grid 768 \
  --output-root studies/dielectric_motheye_n25/results/order_Nz15_Q768_until_oom
```

## 出力と読み方

- `plan.json`、`geometry.svg`、`profile_layers.csv`：固定条件と15層の形状。
- `checkpoint.json`：再開用。形状・ソルバーソース・デバイス等の変更を検出する。
- `all_cases.csv`：成功した全組・全次数のR/T、時間、harmonic数、縮約次元、CUDAピーク割当／予約量。
- `report.json`：全組の停止理由、最大成功次数、最初のOOM次数、固定Nz/Qでの次数収束判定。
- `<格子>/<手法>/<波長>nm/cases.csv`、`report.json`：各組の数値と隣接次数の変化。
- `method_comparison.csv`：NV/ASRの最大共通成功次数でのR/T差。厳密解との誤差ではない。
- `figures/reflectance_vs_order.png/svg`、`transmittance_vs_order.png/svg`：2行（格子）×3列（波長）、NV/ASRを重ねた図。
- `figures/delta_R_abs_vs_order.png/svg`：隣接次数の反射率差（パーセントポイント）。
- `figures/runtime_seconds_vs_order.png/svg`：ソルバー時間。子プロセス起動やimport時間は含まない。
- `figures/energy_residual_vs_order.png/svg`：abs(R+T−1)。
- `workers/<格子>/<手法>/<波長>nm/M<次数>/`：要求、結果、ログ。失敗した次数にもログが残る。

OOMにはR/T値を割り当てない。図ではOOM次数を縦の点線と凡例で示す。
非物理的な計算値はCSVへ保持して図へ赤い×を付け、収束判定から除外する。
今回の材料は無損失なのでA=1−R−Tは物理的な吸収ではなく数値残差。

固定Nz/Qでの次数収束は、高次数側の連続する少なくとも2区間でR/T/Aの絶対変化が
0.00025以下（0.025パーセントポイント）、候補と全上位次数が同じ許容値内で一致し、
該当点が−1e−7≤R,T≤1+1e−7かつabs(R+T−1)≤1e−6の場合に限る。
`order_converged_at_fixed_Nz_Q`と`selected_order`に保存する。
**15層・固定Qでの次数収束であり、層数収束・積分格子収束・全帯域の収束ではない。**

CUDAピーク量はPyTorchの割当／予約量であり、ドライバや他プロセスの全GPU使用量ではない。
OOM次数はGPU容量・ソフトウェア・他のGPU使用状況に依存する。
CUDAを明示して利用できない場合は前段で停止し、CPUへ自動変更しない。
子プロセスが結果を書かずに殺された場合や数値エラーは`worker_error`とし、OOMと断定しない。
その組の掃引を停止して次へ進み、バッチの終了コードを1にする。
確認されたOOM／指定上限による終了は0、Qの限界は2、中断は130。
終了コード0自体は光学的収束を意味しない。

## ローカル検証

```bash
python -m unittest studies.dielectric_motheye_n25.test_order_until_oom -v
python run_n25_motheye_order_until_oom.py --device cpu --grid 64 \
  --start-order 2 --max-order 6 --cpu-threads 2 \
  --output-root studies/dielectric_motheye_n25/results/order_Nz15_cpu_smoke
```

単体テストはOOMを模擬して停止・次組への移行・再開を検査する。
CPU小規模計算は12組×M=2,4,6の動作確認用。GPUでのOOM実測やQ=512の収束結果ではない。

2026-10-14のローカル確認：停止・再開等の単体テスト10件が合格。
CPUで12組×3次数の36ケースを計算し、すべてエネルギー保存条件を満たした。
最大abs(R+T−1)は6.062e−14。配布用フォルダから六方ASRの3波長も別途実行できた。
PNG/SVGの5組を生成し、反射率図のレイアウトを確認した。
GPU計算と実際のCUDA OOMは手元で実測していない。
