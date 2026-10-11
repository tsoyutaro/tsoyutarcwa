# PMMAモスアイ＋Cr薄膜の収束確認

PMMAで形成したモスアイをCrで被覆し、半無限のPMMA基板上に置く。
既存の測定値を用いたPMMA＋Auモデルと同じ三領域ソルバー・形状分割を使う。
金属をCrのJohnson CSVに置き換え、回折次数・高さ分割数・ASR積分格子を順に検査する。
Crだけのモスアイのような金属基板はないため、透過率を省略しない。

## 計算条件

| 条件 | 初期値 |
|---|---|
| 格子 | 六方対称の三角格子、周期200 nm |
| PMMA芯 | 高さ500 nm、頂部半径5 nm、底部半径65 nm、半径は深さに比例 |
| Cr側面 | 半径方向に30 nm増やした同心シェル |
| Cr頂部 | 高さ30 nm、半径35 nmの円盤近似 |
| Cr谷底 | 下端30 nmの領域でPMMA芯の外側をCrで充填。芯の下にCrは置かない |
| 入射 | 空気から垂直入射、x偏光 |
| 基板 | 分散を持つ損失なしの半無限PMMA |
| Cr材料 | `data/Johnson.csv`、既存Crスタディと同じデータ、ε=(n+ik)²を波長に対して線形補間 |
| PMMA材料 | `data/Szczurowski.csv`、nを線形補間、k=0 |
| 計算法 | complex128、Redheffer接続、一般化Li則、D6入射対称部分空間 |
| ASR写像 | `outer`、外側Cr/空気界面に適合、G=0.03 |
| 回折次数M | 8,12,16,18,20 |
| PMMA形状分割Nz | 40,60,80,100。谷底境界に分割境界を合わせる。頂部円盤は別に1層 |
| ASR積分格子Q×Q | 256²,384²,512² |
| 収束判定波長 | 400,550,700 nm |
| 許容変化 | R/T/Aそれぞれ絶対値0.005＝0.5パーセントポイント |
| 最終スペクトル | 収束判定が通った場合のみ、400–700 nm、5 nm刻みの61点 |

側面膜厚は法線方向の一定厚さではなく**半径方向の厚さ**。頂部円盤も近似であり、
実際の蒸着に伴う陰影・酸化・粗さは含めない。膜厚は独立に指定できるので、
例えば薄い側面被覆を調べるには`--cr-side-thickness-nm 5 --cr-top-thickness-nm 30
--cr-valley-thickness-nm 30`を指定し、別の出力先を使う。

PMMAデータの下限は404.7 nm。既定では最初の404.7/411.5 nmの2点から400 nmまで
線形外挿する。外挿の方法・材料ハッシュ・各点の外挿の有無を記録し、元CSVは変更しない。
外挿を使わない場合は`--pmma-shortwave-policy error --anchor-wavelengths 405,550,700
--spectrum-wavelengths 405:700:5`を指定する。

## 実行

プロジェクトのルート（遠隔環境では`/home/7/uq06557/common2/20260813/tsoyutarcwa`）で実行。
このルートに`run_pmma_cr_motheye_convergence.py`と更新した`studies`を配置する。
共有ソルバーの`studies/pmma_gold_motheye/common.py`も更新が必要。
既存環境のPyTorch、torcwaを利用する。PNGにはMatplotlibが必要、SVGは標準ライブラリのみ。

設定・断面図だけ確認（行列計算なし）：

```text
python run_pmma_cr_motheye_convergence.py --device cuda --dry-run
```

GPUで収束検査し、通った場合に最終スペクトルを計算：

```text
python run_pmma_cr_motheye_convergence.py --device cuda --run-final-spectrum
```

`--device cuda`では行列演算・固有値計算・界面接続・フラックスをCUDAで計算。
幾何条件・材料補間・積分係数の準備・入出力はCPUで行う。
CUDAが使えない場合は開始前に停止し、CPUへ自動で切り替えない。
`--device auto`を明示すると利用可能なGPUを選び、なければCPUを選ぶ。
ASRの空間ヤコビアン生成には一時的に自動微分を使うが、固定形状では写像をdetachし、
各層の不要な補助配列を解放する。R/T/Aのみ保存し、電磁場は保存しない。

候補上限が足りない場合の拡張例（同じ出力先で完了ケースを再利用）：

```text
python run_pmma_cr_motheye_convergence.py --device cuda --orders 12,16,18,20,22,24 --slices 60,80,100,120 --grids 384,512,640 --max-cycles 4 --run-final-spectrum
```

全波長域をより密に確認する場合の例（別出力先）：

```text
python run_pmma_cr_motheye_convergence.py --device cuda --anchor-wavelengths 400:700:25 --tolerance 0.001 --run-final-spectrum --output-root studies/pmma_cr_motheye/results/dense_anchors
```

内外の金属界面へ適合する`double`写像で独立に検査する場合：

```text
python run_pmma_cr_motheye_convergence.py --device cuda --radial-mapping double --run-final-spectrum --output-root studies/pmma_cr_motheye/results/double_mapping
```

両写像は低次数では異なる値を出し得る。それぞれで三軸の収束を確認し、
最終値の整合も調べる。`outer`での収束判定だけで写像によらない精度は保証しない。
`--no-symmetry`で対称性縮約を外せるが、計算量とメモリ使用量が増える。

保存した結果から図だけ再生成：

```text
python run_pmma_cr_motheye_convergence.py --plot-only
```

異なる出力先で計算した場合は、その`--output-root`を再指定する。

## 判定と保存ファイル

Rは反射フラックス、Tは損失のないPMMA基板への透過フラックス、A=1−R−Tは
Cr被膜・谷底膜で吸収される割合。値は0–1の割合で、クリップしない。
薄膜試料なので1−Rを吸収率と解釈しない。
R+T+A=1は定義から成り立つため、体積吸収積分との独立した比較ではない。

各軸の**上限側の連続する2区間**で、全判定波長のR/T/Aの最大変化が許容値以下、
かつ受動性の範囲内であることを要求する。例えば16→18、18→20が通ればM=18を採用する。
途中の平坦部だけでは採用しない。Nz→M→Qの更新を繰り返し、
同じ三軸の組合せで全軸が通った場合にのみ`converged`とする。

- `converged`：指定した判定波長で三軸の条件を満たした。全スペクトルの誤差保証ではない。
- `candidate_range_insufficient`：いずれかの上限側の比較が不合格。候補範囲の拡張・図の確認が必要。
- `cycle_limit_reached`：軸の推奨値がまだ安定していない。
- `calculation_failed` / `interrupted`：途中停止。完了した各ケースは保存される。
- `nonphysical_final_spectrum`：最終スペクトルに受動性違反がある。平均値は採用しない。

終了コードは収束0、候補不足など2、計算失敗1。
同じコマンドを再実行すると`checkpoint.json`を読み、完了したケースを省略する。
M/Nz/Q候補・許容値・サイクル数は変更して再利用できる。
材料・形状・写像・判定波長・ソルバーの変更はハッシュで区別され、別の出力先が必要。
同じ出力先への同時実行は避ける。

出力は既定で`studies/pmma_cr_motheye/results/`：

- `plan.json`：形状・材料・判定条件・GPU情報・ソルバーのハッシュ。
- `geometry.svg` / `profile_layers.csv`：実際に渡す断面と層分割。
- `checkpoint.json` / `all_cases.csv`：全ての完了ケースのR/T/A・計算時間・写像診断。
- `pmma_cr_motheye_convergence.json`：ステータス・三軸の比較履歴・推奨条件・平均値。
- `anchor_spectrum.csv` / `spectrum.csv`：判定波長／最終スペクトル。
- `figures/convergence_absolute_order.*`：Mに対するR/T/Aそのもの（自動で縦軸範囲を調整）。
- `figures/convergence_absolute_slices.*` / `convergence_absolute_grid.*`：Nz/Qに対するR/T/A。
- `figures/convergence_delta_*.svg/png`：隣接候補の変化量。縦軸はパーセントポイント。
- `figures/spectrum.svg/png`：最終R/T/Aスペクトル。

平均は点の算術平均と、台形積分による波長平均を両方記録する。
太陽光や実際の光源のスペクトルでは重み付けしない。
再開時は未生成のスペクトルCSVをヘッダーのみとし、旧スペクトル図も再生成時に取り除く。
最新JSONの`status`、`final_spectrum_complete`と`final_spectrum`で完了を確認する。

## 小規模の検証

2026-10-11に保存済みGPU結果を確認した。M=18、Nz=80、Q=384で`converged`、
61波長完了。400–700 nmの台形積分平均はR=1.403412%、T=0.854296%、A=97.742292%。
指定の3判定波長と許容値0.005の範囲では妥当であり、モデルの修正は不要。
より厳しい評価には上記の密な判定波長を使う。
共有ソースの更新後に再計算する場合は新しい出力先を指定する。
詳しい確認は[n=2.5比較と併せた結果レビュー](../dielectric_motheye_n25/RESULTS_REVIEW_20261011.md)を参照。

```text
python -m unittest studies.pmma_cr_motheye.test_convergence
```

材料補間・短波長外挿・偽の早期収束の排除・受動性違反の排除・再開を検査する。
CPUの小規模実計算で、損失のない場合のフラックス保存と、既存ソルバーとの互換性も確認する。
これらは既定の高次数・GPU本計算の収束確認を代替しない。
