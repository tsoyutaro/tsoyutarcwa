# Fe・Ni・Ti・W・Vの二次元モスアイ：収束調査の実行方法

2026-10-11に、既存Cr版と同じRCWA計算経路を使う実行ファイルを追加した。
材料にはユーザー添付のFe.csv、Ni.csv、Ti.csv、Vanadium.csv、Ordal(W).csvを使う。
元CSVをバイト単位で同一にコピーし、材料名・ハッシュ・補間方法を各計画と結果に保存する。
既存Crの計算ファイルと材料は保持している。共有の図作成コードには5材料の表示名を追加した。

## ファイルの配置

このプロジェクトのルートは次のフォルダ。

```text
C:\Users\tsoyu\Documents\Codex\2026-08-06\files-mentioned-by-the-user-extensions\outputs
```

- `run_metal_motheye_convergence.py`：金属モスアイ／同じ金属の基板。
- `run_pmma_metal_motheye_convergence.py`：PMMAモスアイへの金属被覆／PMMA基板。
- `run_transition_metals_convergence.py`：指定した材料と構造を順番に実行。
- `studies/metal_motheye/`：金属のみの収束調査・図作成。
- `studies/pmma_metal_motheye/`：PMMA被覆の形状・収束調査・図作成。
- `studies/shared/transition_metals_data/`：添付CSVのコピーとmanifest.json。
- `validate_transition_metals.py`：材料・小規模物理計算・再開処理の検証。

## 構造と計算条件

前回Cr版と同じ、円形断面を持つ二次元の六方配列（60°三角Bravais格子）を使う。

| 条件 | 金属のみ | PMMAへの金属被覆 |
|---|---|---|
| 最近接周期 | 200 nm | 200 nm |
| 突起／芯の高さ | 500 nm | PMMA芯500 nm |
| 先端半径 | 5 nm | PMMA芯5 nm |
| 底面半径 | 95 nm | PMMA芯65 nm、金属込み95 nm |
| 半径profile | 高さ方向に線形 | 高さ方向に線形 |
| 側面金属 | 突起全体が金属 | 半径方向に30 nmのシェル |
| 頂部金属 | ― | 厚さ30 nm、半径35 nmの円盤 |
| 谷底金属 | ― | 下端30 nmの領域、PMMA芯の外側を金属で充填 |
| 基板 | 半無限の同じ金属 | 損失なし・分散を持つ半無限PMMA |
| 入射 | 空気から垂直入射、x偏光 | 空気から垂直入射、x偏光 |
| 次数候補M | 4,6,8,10,12,14,16,18,20 | 8,12,16,18,20 |
| 層数候補Nz | 50,60,70,80,90,100 | 40,60,80,100 |
| ASR積分格子Q | 96,128,192,256 | 256,384,512 |
| 写像 | 円界面に適合するmatched-ASR | 外側金属／空気界面に適合するouter写像 |
| 共通手法 | Fourier因子分解、Redheffer接続、D6入射対称部分空間、complex128 | 同左 |
| ASRパラメータ | G=0.03 | G=0.03 |
| 許容変化 | 0.005＝0.5パーセントポイント | 同左 |

PMMAのNzは芯の500 nmを分割する層数で、頂部円盤がある場合は別に1層追加する。
側面の30 nmは半径方向の厚さであり、側面法線方向の一定厚さとは異なる。
実際の蒸着の代わりに同心シェル・頂部円盤・谷底充填で近似する。芯の下には金属膜を置かない。

材料は各点でε=(n+ik)²を作り、複素誘電率を波長に対して線形補間する。金属の外挿は拒否する。
Fe・Ni・Ti・VのCSVは188–1937 nmを含む。添付W/Ordalは667 nmからなので、
ユーザー指定に従いWは667–700 nmを調べる。400–700 nm全域のW結果はこのCSVでは作らない。

| 材料 | 収束判定波長 | 収束後のスペクトル（指定時） |
|---|---|---|
| Fe・Ni・Ti・V | 400,550,700 nm | 400–700 nm、5 nm刻み、61点 |
| W | 667,680,700 nm | 667,672,677,682,687,692,697,700 nmの8点 |

PMMAの既存Szczurowski.csvは下限404.7 nm。Cr版と同じく、既定では最初の2点から400 nmまで
nを線形外挿し、その使用を記録する。k=0。外挿を避ける場合は個別実行で
`--pmma-shortwave-policy error --anchor-wavelengths 405,550,700 --spectrum-wavelengths 405:700:5`を指定する。

## 実行コマンド

GPU計算機のプロジェクトルートに、今回の追加ファイルと共有ファイルを反映して実行する。
既存Cr版と同じPyTorch・torcwaを使う。PNG図にはMatplotlibが必要。

5材料×2構造の条件を生成する（電磁場の計算は行わない）：

```bash
python run_transition_metals_convergence.py --device cuda --dry-run --run-final-spectrum
```

5材料×2構造の収束調査を順番に実行する：

```bash
python -u run_transition_metals_convergence.py --device cuda --run-final-spectrum
```

GPUに同時に複数の計算を投入しない。各材料・構造ごとにM、Nz、Qを判定する。
収束した組合せだけ最終スペクトルを計算する。収束しない組合せも保存し、次の組合せへ進む。
最後に`studies/metal_motheye/results/batch_summary.json`へ全10通りの状態をまとめる。

構造を限定する：

```bash
python -u run_transition_metals_convergence.py --structures metal --device cuda --run-final-spectrum
python -u run_transition_metals_convergence.py --structures pmma --device cuda --run-final-spectrum
```

個別の材料を実行する（FeはNi、Ti、W、Vに置き換えられる）：

```bash
python -u run_metal_motheye_convergence.py --material Fe --device cuda --run-final-spectrum
python -u run_pmma_metal_motheye_convergence.py --material Fe --device cuda --run-final-spectrum
```

Wだけ両構造を実行する：

```bash
python -u run_transition_metals_convergence.py --materials W --device cuda --run-final-spectrum
```

WindowsでローカルCPUを使う場合：

```powershell
Set-Location 'C:\Users\tsoyu\Documents\Codex\2026-08-06\files-mentioned-by-the-user-extensions\outputs'
& 'C:\Users\tsoyu\AppData\Local\Programs\Python\Python313\python.exe' -u .\run_transition_metals_convergence.py --device cpu --run-final-spectrum
```

作成時のこのPCはCPU版PyTorchでCUDAを利用できない。`--device cuda`の本計算はGPU計算機で実行する。
CUDAを明示指定した場合、利用できなければ停止し、CPUへ自動的に切り替えない。

候補範囲が足りない場合の拡張例（まず全体結果を確認してから、未収束の材料を再実行する）：

```bash
python -u run_metal_motheye_convergence.py --material Fe --device cuda --orders 12,16,18,20,22,24 --slices 60,80,100,120 --grids 128,192,256,384 --max-cycles 4 --run-final-spectrum
python -u run_pmma_metal_motheye_convergence.py --material Fe --device cuda --orders 12,16,18,20,22,24 --slices 60,80,100,120 --grids 384,512,640 --max-cycles 4 --run-final-spectrum
```

同じ材料・形状・ソースならチェックポイントを再利用する。M/Nz/Q候補・許容値・サイクル数は変更できる。
材料CSV・形状・写像・判定波長・ソースを変更した場合は、別の`--output-root`を指定する。
違う署名のチェックポイントを混用すると、開始前に拒否する。

PMMAの膜厚を変更する例：

```bash
python -u run_pmma_metal_motheye_convergence.py --material Ni --device cuda --metal-side-thickness-nm 5 --metal-top-thickness-nm 30 --metal-valley-thickness-nm 30 --output-root studies/pmma_metal_motheye/results/ni_side5 --run-final-spectrum
```

PMMAの両金属界面に適合するdouble写像との比較例：

```bash
python -u run_pmma_metal_motheye_convergence.py --material Fe --device cuda --radial-mapping double --output-root studies/pmma_metal_motheye/results/fe_double --run-final-spectrum
```

図だけ再生成する：

```bash
python run_metal_motheye_convergence.py --material Fe --plot-only
python run_pmma_metal_motheye_convergence.py --material Fe --plot-only
```

## 収束判定と結果の場所

Nz→M→Qの順に検査し、各軸の上限側で連続2区間の変化が許容値以下かつ受動的であることを要求する。
採用したM/Nz/Qが次のサイクルでも変わらず、全軸が通った場合に`converged`とする。
単に高いMまたはNzを指定しただけでは収束済みと判定しない。

金属のみではR、基板へ入るP_sub、モスアイ領域の吸収A_mothを調べる。
半無限金属のためT_far=0、A_total=1−R、A_moth=1−R−P_sub。
PMMA被覆ではR、PMMAへ透過するT、金属の吸収A=1−R−Tを調べる。

結果はプロジェクトルートから以下へ保存する（`<metal>`はfe、ni、ti、w、v）。

```text
studies/metal_motheye/results/<metal>/
  metal_motheye_plan.json
  metal_motheye_checkpoint.json
  metal_motheye_convergence.json
  metal_motheye_all_cases.csv
  metal_motheye_anchor_spectrum.csv
  metal_motheye_spectrum.csv
  metal_motheye_figures/absolute_order.png, .svg
  metal_motheye_figures/absolute_slices.png, .svg
  metal_motheye_figures/absolute_grid.png, .svg
  metal_motheye_figures/convergence.svg
  metal_motheye_figures/spectrum.svg

studies/pmma_metal_motheye/results/<metal>/
  plan.json, geometry.svg, profile_layers.csv
  checkpoint.json, all_cases.csv
  pmma_metal_motheye_convergence.json
  anchor_spectrum.csv, spectrum.csv
  figures/convergence_absolute_order.png, .svg
  figures/convergence_absolute_slices.png, .svg
  figures/convergence_absolute_grid.png, .svg
  figures/convergence_delta_order.png, .svg
  figures/convergence_delta_slices.png, .svg
  figures/convergence_delta_grid.png, .svg
  figures/spectrum.png, .svg
```

CSVのパワーは0–1の割合。差分図のppはパーセントポイント。
次数・層数の図は、最新サイクルの候補全体を表示する。
`--output-root`を使う場合、バッチは指定先の`metal/<metal>`と`pmma/<metal>`へ保存する。

終了コードは収束0、候補不足／サイクル上限など2、計算失敗1。
各収束JSONの`status`と`recommendation`、最終スペクトルの`final_spectrum_complete`を確認する。
今回の既定収束判定は代表3波長で行う。全61点または全8点でM/Nz/Qを再判定したという意味ではない。

## 作成時の検証

```bash
python validate_transition_metals.py --integration --device cpu
```

5材料の元CSVハッシュ、単位、補間、範囲外拒否、PMMA形状、平坦界面のFresnel解、
金属モスアイのD6/Cs一致、PMMAの既存Cr計算経路との一致、再開と署名不一致拒否を検査する。
小規模なCPU実計算での検証であり、既定候補での本番収束調査は上のコマンドで別途実行する。
