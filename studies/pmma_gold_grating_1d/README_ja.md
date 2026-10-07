# PMMA台形格子＋金蒸着膜：1次元RCWAの次数・層数探索

横方向はxの1方向だけ周期を持ち、y方向には無限に続く台形格子です。
PMMA円錐モスアイの断面を参考にした構造で、三角格子の円錐そのものではありません。
このフォルダを既存の `tsoyutarcwa/studies/pmma_gold_grating_1d/` に置いて使います。
既存の `studies/gold_grating_1d/solver.py` と `rcwa_ext/` を使用します。
共通RCWA本体や過去のstudyは変更しません。

## 初期構造

| 項目 | 設定 |
|---|---:|
| 周期 | 200 nm |
| PMMA台形の高さ | 500 nm |
| PMMA上部幅／底部幅 | 10 nm／130 nm |
| 側面の金：x方向の厚さ | 左右それぞれ30 nm |
| 上部の金だけの矩形 | 厚さ30 nm、幅70 nm |
| 谷底の金 | 厚さ30 nm |
| 基板 | 半無限PMMA、k=0 |
| 入射 | 空気側から垂直、TEとTM |

側面の金30 nmは、以前のPMMAモデルの半径方向厚さに対応する**横方向の厚さ**です。
側面に垂直な厚さ、蒸着装置が示す公称膜厚とは区別します。
これは比較用の理想化で、実蒸着の方向・遮蔽・粒状性を計算するモデルではありません。

谷底の金は、PMMA台形の底側30 nmの高さの範囲でPMMAコアの外側を埋めます。
上側はPMMA＋金＋空気、谷底側はPMMA＋金です。PMMAコアの下に金を挟みません。
`gold_valley_thickness_nm` を0にすれば谷底側もPMMA＋金＋空気になります。
側面・頂部・谷底の厚さは `config.json` で独立に変更できます。

`slices` / `Nz` は**PMMAの高さ500 nmを分割する数**です。
頂部金の矩形は別の1層なので、100分割なら有限層は101層です。
谷底金30 nmの境界に分割面を一致させ、PMMA分割数を保ったまま領域ごとの層厚を調整します。
入射側空気と半無限基板は有限層数に含めません。

## 最初に構造を確認する

```bash
python3 studies/pmma_gold_grating_1d/preview.py --slices 100
```

`results/preview/geometry.svg`、`geometry_layers.csv`、`settings.json` を出力します。
Torchを読み込まず、RCWAも計算しません。SVGはブラウザで確認できます。
図の階段形状と光学計算の層構成は同じ `geometry.build_layers()` を使用します。

## 次数と層数を探索する

```bash
python3 studies/pmma_gold_grating_1d/run_all.py --device cuda
```

初期探索は以下の条件です。

- 波長：400、450、500、550、600、650、700 nm。
- 次数：M=16,24,32,40,48,56。x方向の成分数は2M+1、y方向は1。
- PMMA分割数：Nz=100,140,180,220,260,300。
- complex128、Cartesian Liの逆則／直接則、Redheffer合成。
- Fourier係数は金・PMMA・空気の区間積分から解析的に計算するため、**grid探索は不要**。
- 許容値：反射率R、透過率T、吸収率Aの変化がそれぞれ0.5パーセントポイント以内。

最初は他方の軸を最大値に固定して2軸を調べます。
TE・TMの両偏光で、全波長・全3指標が最後の2つの隣接区間で許容値を満たすか判定します。
さらに、候補のMとNzを同時に設定した状態で2軸を再確認し、最大設定の結果とも比較します。
最初の2掃引は最大77波長ケースで、各ケースからTE・TMの両方が得られます。
候補での再確認が必要なら追加ケースを計算します。

結果から `recommended_numerics` が得られれば、その**試した波長・設定の範囲**での推奨値です。
最大設定が候補と同じ場合、基準との差が0になるのは同じ計算点のためです。
隣接設定間の変化を評価しており、絶対誤差や未計算波長での誤差を保証しません。
初期値だけで収束することを前提にしていません。

条件だけ作成する場合：

```bash
python3 studies/pmma_gold_grating_1d/run_all.py --prepare-only
```

## 保存・再開・探索範囲の追加

出力先は `results/analytic_search/`。各波長ケースを計算した直後にcheckpointを保存します。
同じコマンドで未完了ケースを再開でき、材料・形状・計算ソースが違う結果は混合しません。
範囲だけを広げる例：

```bash
python3 studies/pmma_gold_grating_1d/run_all.py --device cuda \
  --orders 16,24,32,40,48,56,64,72,80 \
  --slice-values 100,140,180,220,260,300,340,380,420
```

計算点はM・Nz・波長で照合します。片方の上限を変えると他方の掃引条件も変わるため、
その固定条件で未計算のケースがあれば新たに計算します。
再開時に引数を省略すれば、保存済みの探索範囲を使用します。
形状などを変更するときは別の `--output-dir` を指定してください。

`report.json` は探索の合否・必要な拡張軸・候補の再確認を記録します。
`checkpoint.json` は実値、`cases.csv` は全ケースのR/T/A・時間・ピークメモリです。
各掃引フォルダの `convergence.svg` と、Matplotlibがあれば `convergence.png` が図です。
`adjacent_changes.csv` は符号付き次数間／層数間の差をppで出力します。
`geometry.svg` と `geometry_layers.csv` は最大設定の事前構造図と層表です。

保存済み結果だけの再判定と作図：

```bash
python3 studies/pmma_gold_grating_1d/run_all.py --report-only
```

1軸だけ計算する例：

```bash
python3 studies/pmma_gold_grating_1d/converge.py --device cuda \
  --axis order --values 16,24,32,40,48,56 --slices 100
```

## R・T・Aと材料データ

PMMAはk=0としており、ここでのTは**半無限PMMA基板へ進む透過電力の割合**です。
PMMA基板内での吸収、裏面での反射、裏面から空気への出射はモデルに含めません。
`A=1-R-T` は有限構造の金部分で吸収された割合です。
以前の半無限金基板のように `T_far=0` とする定義は、このstudyには使用しません。
CSV内のR/T/Aは0〜1の割合、図は%、差のCSVはパーセントポイントです。

添付の `Szczurowski.csv` と `au_measured_nk.csv` をコピーしています。
波長はCSVのµmからnmに変換し、n・kを線形補間してからepsilon=(n+ik)^2を求めます。
PMMA表は404.7 nmからなので、既定の400 nmには最初の2点による短い線形外挿を使います。
元のCSVは変更せず、方法と値をplan・reportの材料情報に、使用したケースには
`pmma_index_source` に記録します。400 nm未満やその他の範囲外への外挿は許可しません。
`pmma_shortwave_extension` を `none` にする場合は、計算波長も404.7 nm以上にしてください。

## 小さいケースでの物理確認

```bash
python3 studies/pmma_gold_grating_1d/validate.py --device cuda
```

空気／PMMA界面のFresnel式、PMMA上の平坦な金30 nm膜の解析解、無損失の代替材料での
エネルギー保存・偏光分離、3材料の解析的Fourier係数と細かい数値積分、
金蒸着PMMA構造の受動性・固有値残差を確認します。
これらは低次数の物理確認であり、本番の次数・層数収束とは別の確認です。
作成環境ではCPUで5ケースすべてPASS。CUDAでの本番探索は未実行です。

## Vallius 1D ASRとの次数比較

```bash
python -m studies.asr_1d_comparison.compare --study pmma_gold_grating_1d --device auto
```

同じ形状・分散材料・PMMA基板・層数で、既存Li計算と1D ASRのR/T/Aを比較します。
結果は `results/li_vs_asr_1d/` に保存します。詳細な条件・GPUでの実行・内部倍率の変更・図の定義は
[比較用README](../asr_1d_comparison/README_ja.md)を参照してください。
