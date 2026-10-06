# gold_motheye3: 更新後RCWAでの金モスアイ収束確認

`gold_motheye2` と同じ物理モデル・測定Auデータを使い、現在のRCWAで
Fourier次数 M、高さ分割数 Nz、ASR格子数をそれぞれ変える。
旧計算結果は引き継がない。計算ケースごとにcheckpointを保存する。

## 形状・物理条件

- 三角配列、周期200 nm、空気から正入射するx偏光。
- 金の円錐台を円柱スライスで近似。高さ500 nm、先端半径5 nm、底面半径95 nm、線形テーパー。
- 基板は半無限の金。PMMAはこのモデルに含まない。
- Auの `n,k` は `data/au_measured_nk.csv` を使用。`gold_motheye2` の整形済みCSVと同一内容。元データも `data/au_measured_nk_original.csv` に保存。
- 複素倍精度、matched-ASR、Li因子分解、D6のE1入射源行縮約、Redheffer結合。

半無限Au基板の奥に出る遠方透過率 `transmittance_far` は定義上0。
実際に計算する `power_into_substrate` は柱領域から基板へ入るパワーであり、
遠方透過率とは区別する。柱領域の吸収率は
`motheye_absorptance = 1 - reflectance - power_into_substrate`。

## 既定の3つの掃引

| 軸 | 変化させる値 | 固定条件 |
|---|---|---|
| M | 4,6,8,10,12,14,16,18,20 | Nz=100、grid=256 |
| Nz | 10,15,20,30,40,50,60,70,80,90,100 | M=16、grid=256 |
| ASR grid | 96,128,192,256,320 | M=16、Nz=100 |

各掃引で400、550、700 nmを計算する。合否は3波長すべてについて、
最高値側の連続2段階で `R`、基板へ入るパワー、柱領域の吸収率の
絶対変化が**それぞれ0.005以下**かどうかで決める（0.5 percentage point）。
各軸の収束は他の2軸の収束を保証しない。必要に応じて固定条件を上げて再確認する。

## Linuxでの実行

PyTorchとRCWAが使えるプロジェクトルートで、まず標準ライブラリだけの準備確認:

```bash
python3 studies/gold_motheye3/run_all.py --prepare-only
```

GPU計算を順に行う場合:

```bash
python3 studies/gold_motheye3/run_all.py --device cuda
```

計算時間やメモリを見ながら1軸ずつ進める場合:

```bash
python3 studies/gold_motheye3/converge.py --axis order --device cuda
python3 studies/gold_motheye3/converge.py --axis slices --device cuda
python3 studies/gold_motheye3/converge.py --axis grid --device cuda
```

例えばM=4,6,8の短い確認から始め、後に同じ出力場所で値を追加できる:

```bash
python3 studies/gold_motheye3/converge.py --axis order --values 4,6,8 --device cuda
python3 studies/gold_motheye3/converge.py --axis order --device cuda
```

後の呼び出しでは、前のリストを先頭に保ち、より大きな値だけ追加する。
例えば `--values 4,6,8,10,12,14,16,18,20,22` に拡張し、
M=22でメモリ不足になっても、M=20までの結果は残る。PyTorchなしで
保存済みの表・図を再生成するには、実行時と同じ値リストを指定する:

```bash
python3 studies/gold_motheye3/converge.py --axis order \
  --values 4,6,8,10,12,14,16,18,20,22 --report-only
```

`--values`、`--wavelengths`、`--order`、`--slices`、`--grid`、
`--gold-csv`、`--tolerance`、`--output-dir` で条件を変えられる。
固定条件・材料・RCWAソースが変わった場合、既存checkpointとの混用を拒否する。
別の `--output-dir` を指定して新しい掃引を開始する。

## 最初の結果で700 nmのgrid収束が未確認だった場合

既存の `grid_M16_Nz100/checkpoint.json` を残したまま、次のファイルを実行する。

```bash
python3 studies/gold_motheye3/run_next.py --device cuda
```

更新版はgrid=448までの21ケースを再利用し、grid=512と576の各3波長、
計6ケースだけを追加する。途中で停止しても同じコマンドで再開できる。
保存済みの実値と符号付き反射率差を表示するだけなら、以下を実行する。

```bash
python3 studies/gold_motheye3/run_next.py --report-only
```

`--add-grids 640,704` のように、さらに大きなgridを指定することもできる。
この判定はM=16でのgrid確認なので、最終的に採用するMでもgridを再確認する。

## grid=576でも700 nmの振動が減らない場合

gridを増やしても約1 percentage pointの反射率の振動が残る場合は、
ASRの座標変換と変換行列を先に診断する。

```bash
python3 studies/gold_motheye3/diagnose_grid_700.py --device cuda --transforms
```

既存のgrid checkpointから形状・M・Nz・材料CSVを読み取り、
grid=448,512,576の各々で1,50,100層目の実装中の座標変換を調べる。
座標変換のJacobianが正か、60度回転・鏡映に整合するか、金領域の面積の
積分が解析的な円の面積に近いか、材料ラベルに対称性のずれがあるかを出力する。
`--transforms` はD6計算が使用する `T_star = E^H T E` の特異値と
2ノルム条件数も測定する。条件数が大きい場合、行列の誤差が増幅されやすい。
これらの指標だけで反射率の誤差量や原因を断定することはできない。

100層の光学計算・R/Tの再計算は行わず、指定した3層×3gridを診断する。
SVDの時間はかかるため、座標変換のみを先に調べる場合は
`--transforms` を省く。準備確認には `--prepare-only`、全層を調べるには
`--layers all` を指定し、全層診断用に別の `--output-dir` を使う。

出力先は `results/grid_diagnostics_700_transform/`、座標変換のみの場合は
`results/grid_diagnostics_700_map/`。`diagnostics.json` に各診断値とエラー、
`summary.csv` に一覧、`plan.json` に条件と未完了ケースを保存する。
途中停止後は同じコマンドで未完了ケースを再開できる。
既存のgrid計算とRCWAソース・材料CSVが一致していることを確認し、
一致しない場合は別バージョンの結果を比較しないよう停止する。
診断ファイルはソルバー自体を変更しないので、既存checkpointは継続使用できる。

## 出力

各軸の結果は `results/order_Nz100_grid256/`、`results/slices_M16_grid256/`、
`results/grid_M16_Nz100/` に分けて保存する。各フォルダの
`cases.csv` は全観測量と実行秒数、`report.json` は隣接値間の差と判定、
`convergence.svg` は3観測量、`runtime.svg` は時間、`checkpoint.json` は再開用。
`plan.json` は形状、数値条件、測定CSVおよび全 `rcwa_ext/*.py` など
ソルバーソースのSHA-256を記録する。

本環境にはPyTorchがなく、数値計算自体は未実行。Linuxで実際の収束判定を確認する。

## 保存済みの実値と次数間の差を表示する

700 nm、Nz=100、grid=576、M=12,14,16,18,20を計算した後は、以下で表を表示する。

```bash
python3 studies/gold_motheye3/show_results.py
```

標準ライブラリだけで `results/order_700_Nz100_grid576/checkpoint.json` を読み取り、
各次数のR、基板へ入るパワー、柱領域の吸収率、実行秒数、隣接次数間の
符号付き変化を表示する。保存データ・ソルバーソースを変更せず、光学計算は行わない。
別の掃引のcheckpointは引数で指定できる。

```bash
python3 studies/gold_motheye3/show_results.py results/grid_M16_Nz100/checkpoint.json
```
