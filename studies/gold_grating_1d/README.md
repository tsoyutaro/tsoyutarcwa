# 金の一次元台形格子：次数・層数・grid の探索

このフォルダを既存の `tsoyutarcwa/studies/gold_grating_1d/` に置いて実行します。
既存リポジトリの `rcwa_ext` と `studies/shared/gold_dispersion.py` を利用します。
計算には既存と同じ PyTorch・torcwa が必要です。準備、JSON/CSV の読み出し、SVG 作図には追加パッケージは不要です。
Matplotlib が利用可能なら、標準的な科学作図で SVG と PNG を出力します。無い場合も内蔵の SVG 作図で動作します。

## 初期構造

| 項目 | 設定 |
| --- | --- |
| 材料 | 金／空気、半無限の金基板 |
| 周期 | x 方向 200 nm、y 方向に一様 |
| 高さ | 500 nm |
| 上部幅／底部幅 | 10 nm／190 nm |
| 幅の高さ依存性 | 上から下へ線形に増大する台形 |
| 入射 | 空気から垂直入射 |
| 偏光 | TE（E は y 方向）、TM（E は x 方向）の両方 |
| 波長 | 400–700 nm、50 nm 間隔の 7 点 |
| 金の光学定数 | `data/au_measured_nk.csv`、既存と同じ測定 CSV |
| 数値手法 | complex128、Redheffer 接続、一次元 Cartesian Li 因子分解 |

これは y 方向に無限に延びる台形の稜線です。二次元の円錐を持つ金モスアイとは異なる構造なので、反射率が同じになることは想定しません。
PMMA や頂部の別の 30 nm 金膜は含めません。

幅は `w(s) = top_width + (bottom_width - top_width) * s**profile_power`、`s` は頂部 0・底部 1 です。
各層の中点で幅を評価し、高さを等分します。例えば Nz=140 なら 1 層の高さは約 3.5714 nm。
幅・高さ・波長・探索範囲・許容差は `config.json` で変更できます。
幅は **0 より大きく周期未満**で、上部幅 ≤ 底部幅とします。

## まず構造と計画を確認する

リポジトリのルートで実行してください。絶対パスからの実行も可能です。

```bash
python3 studies/gold_grating_1d/run_all.py --prepare-only
```

`results/all_search/geometry.svg` と `plan.json` が出ます。構造図の細い矩形は初期設定の 140 層を示します。
`--prepare-only` では RCWA 計算を行いません。

## 次数・層数・grid を探索する

```bash
python3 studies/gold_grating_1d/run_all.py --device cuda
```

探索範囲の初期値：

- 次数 M：4, 8, 12, 16, 20, 24, 32, 40
- 層数 Nz：20, 40, 60, 80, 100, 140, 180, 220
- x 方向の grid：192, 256, 384, 512, 768, 1024

手順は次のとおりです。

1. 他の二つを探索範囲の最大値に固定して、grid → 次数 → 層数を比較する。
2. 各軸で、二回連続の差が許容差内になり、その後の全比較も許容差内となる候補を選ぶ。
3. 候補の **同じ組み合わせ**で各軸を再確認する。各軸の候補と、その直前の二設定を比較する。
4. 候補と、次数・層数が最大値で **解析的 Fourier 係数を用いた基準計算**との差も確認する。必要なら設定を上げて再確認する。

各波長・TE/TM の **R、P_sub、A_relief 全て**について、最後の二つの隣接比較がそれぞれ 0.005 以下（0.5 percentage points 以下）なら合格です。
探索の上限まで合格しなければ `not_converged` になります。その際、適切な設定を決めたことにはしません。
`config.json` の `search_values` を延長して、同じコマンドで再実行してください。
M を上げたときは、全 grid 値について `grid >= max(32, 4*M+4)` を満たす必要があります。
候補の再確認は初期値で最大 6 回。`--max-rounds 10` などで変更できます。

探索範囲の最大値を基準にするため、最初の比較は手動の初期値 M=24・Nz=140・grid=576 より重い計算です。
最初の三軸比較は最大 140 波長計算です。1 波長の同じ S 行列から TE/TM の両方を求めます。候補の確認が追加されます。
各計算の実測時間を出力するので、時間の見積もりにはその値を使ってください。

## 一つの軸だけを比較する

初回の三軸探索が未収束だった場合、矩形断面の解析的 Fourier 係数を使って横方向のサンプリング誤差を除く、追加の二軸探索を実行できます。

```bash
python3 studies/gold_grating_1d/run_analytic.py --device cuda
```

M=32,40,48,56、Nz=180,220,260,300 を比較し、候補の同じ組み合わせでも再確認します。grid は使いません。
出力先は `results/analytic_search/`。解析的係数の保存済み結果は再開・延長で再利用されます。
初回の結果の解釈と延長方法は [NEXT_ANALYTIC.md](NEXT_ANALYTIC.md) に記載しています。
M=48・700 nm で `K_y=0 only` が出る旧版の丸め誤差と、保存済み結果を照合して再開する修正は [FIX_ORTHOGONAL.md](FIX_ORTHOGONAL.md) に記載しています。

例えば、140 層・grid=576 で 700 nm の次数を比較する場合：

```bash
python3 studies/gold_grating_1d/converge.py \
  --axis order --values 4,8,12,16,20,24,32,40 \
  --slices 140 --grid 576 --wavelengths 700 --device cuda
```

層数を変える場合：

```bash
python3 studies/gold_grating_1d/converge.py \
  --axis slices --values 40,60,80,100,140,180,220 \
  --order 24 --grid 576 --device cuda
```

grid を変える場合：

```bash
python3 studies/gold_grating_1d/converge.py \
  --axis grid --values 192,256,384,512,768,1024 \
  --order 24 --slices 140 --device cuda
```

同じ出力先で範囲を延長すると、同じ M/Nz/grid/波長の保存済み計算は再利用されます。
任意の出力先は `--output-dir results/my_1d_sweep` で指定できます。
構造・材料・計算ソースの変更は保存時のハッシュで検出し、古い結果との混在を防ぎます。変更後は新しい出力先を指定してください。

## 出力と再作図

- `checkpoint.json`：1 波長の計算が終わるたびに保存。失敗しても完了済みの結果は残る。
- `report.json`：判定、各次数・層数・grid 間の差、最終候補の再確認。
- `cases.csv`：実値（0–1）、実行時間、GPU のピーク allocated/reserved メモリ、選択層の条件数。
- 各軸のフォルダに `convergence.svg` と `adjacent_changes.csv`：図は %、差の表は percentage points。
- `selected_geometry.svg`：最終確認に使った層数の断面図。

再計算せずに作図する例：

```bash
python3 studies/gold_grating_1d/plot.py \
  studies/gold_grating_1d/results/all_search/checkpoint.json
```

`plots/grid/`、`plots/order/`、`plots/slices/` に図と差の表が出ます。
全体の判定も保存済み結果だけで再確認する場合は、`run_all.py --report-only` を使えます。
Matplotlib が利用可能なら PNG も自動で出ます。Matplotlib が無い場合でも、CairoSVG が利用可能な環境では `plot.py` に `--png` を追加して PNG を生成できます。通常は SVG をそのまま表示・保存できます。

## 物理量と診断の意味

半無限の吸収性金基板なので、基板の先に出る遠方透過率 **T_far=0** です。
**P_sub** は台形構造から金基板へ入るパワーで、従来の金モスアイと同じ定義です。
**A_relief = 1 - R - P_sub** は台形領域の吸収、**A_total = 1 - R** は基板を含む全吸収です。
差から求めた A は、独立した吸収積分によるエネルギー検証ではありません。

次数は `[M, 0]`、x 方向の Fourier 成分数は `2*M+1`、y 方向は 1 です。
`solver.py` の一次元用アダプタが既存の固有値計算・ポート定義・S 行列接続を利用し、RCWA 本体は変更しません。
座標変換は使わず、法線方向の誘電率は `[1/epsilon]^-1`、接線方向は `[epsilon]` という Li の因子分解を用います。

grid 探索では各セルの中点で材料をサンプリングし、FFT で `epsilon` と `1/epsilon` の係数を求めます。
中点位置に対応する Fourier 位相も補正します。grid を増やすと境界のサンプリング誤差が変化するので、その影響を実際に比較できます。
基準計算では、幅比 f の中心配置された矩形の係数 `f*sinc(m*f)*exp(-i*pi*m)` を解析的に求めます。
こちらは grid に依存せず、基準計算の `coefficient_method` は `analytic`、`grid_used` は null です。
grid 比較が偶然小さい差になっただけで合格しないよう、候補とこの基準計算との差も判定します。

解析的係数だけを使う場合は設定ファイルの `solver.fourier_coefficients` を `analytic` にします。その場合、grid 探索は不要なので、`converge.py --axis order` または `--axis slices` を使ってください。
このフォルダでは、強い一次元 ASR 写像に伴う高次数での変換行列の悪条件化を避けるため、Cartesian Li を採用しています。

先端側・中間・底部側の三層について、逆誘電率の Toeplitz 行列・電場固有ベクトル行列・磁場固有ベクトル行列の条件数と固有値計算の残差を記録します。
全層でサンプリング幅比の最大誤差も記録します。条件数の値だけで反射率の精度を保証しません。
選択した三層での診断は全層の条件数の検査とは異なります。この計算には ASR の変換行列 T_star は存在しません。
GPU メモリは生きている PyTorch テンソルの最大値と予約領域の最大値を分けて保存します。GPU 全体の使用量とは一致しません。

## 小さな物理検証

```bash
python3 studies/gold_grating_1d/validate.py --device cuda
```

平坦な空気／金境界の Fresnel 解、均一金 30 nm 区間の伝搬、無損失一次元格子の R+T、偏光の分離、一次元因子分解と既存の一般式との一致、解析的係数と細かい grid の一致を確認します。
この検証に合格しても、500 nm 高さの金台形格子の離散化が収束したことにはなりません。
7 波長での合格は、その 7 点での判定です。広帯域スペクトルの精度確認には波長を追加し、同じ設定で再確認してください。
