# PMMAモスアイ＋金30 nm：添付CSVを使う計算

`run_pmma_gold_30nm.py` は、このstudyのRCWA実装を呼び出す実行ファイルです。入力は `data/Szczurowski.csv`（PMMAの `wl,n`）と `data/au_measured_nk.csv`（金の `wl,n` ブロックと `wl,k` ブロック）です。`wl` はµmからnmへ変換し、各計算波長で `n` と `k` を線形補間します。表の範囲外には外挿しません。

R/T計算では、電磁場再構成に使わないASR補助テンソルを各層の後で解放する。
比較のため保持する場合は `--retain-auxiliary` と**別の出力先**を指定する。
この切替はFourier次数・ASR格子・材料値・倍精度設定を変えない。

## 既定の構造

| 項目 | 値 |
|---|---:|
| 三角格子周期、モスアイ高さ | 200 nm、500 nm |
| PMMA円錐の底面／頂部直径 | 130 nm／10 nm |
| 金の側面半径方向厚さ | 30 nm |
| 頂部の金円板 | 厚さ30 nm、直径70 nm |
| 平坦な谷部上の金 | 厚さ30 nm |
| 出力側 | 半無限PMMA基板 |
| 入射 | 空気側から垂直、x偏光 |

谷部上の金は底側30 nmの範囲で**PMMA円錐の外側**に置きます。PMMA円錐の下には金を挟みません。厚さ0 nmを指定すると、以前の「谷部は空気」の形状に戻ります。側面・谷部・頂部がすべて30 nmというのは比較用の理想化です。蒸着方向と遮蔽を考慮した実試料では厚さがそれぞれ異なる可能性があります。

## 実行

`outputs` ディレクトリで、`torch`・`torcwa`を利用できるPython環境から実行します。CSVと形状の事前確認はRCWAライブラリなしでできます。

```bash
python studies/pmma_gold_motheye/run_pmma_gold_30nm.py --check-only
```

**計算前に構造を図で確認**するには次を実行します。これは `torch` や `torcwa` を読み込まず、光学計算もしません。

```bash
python studies/pmma_gold_motheye/run_pmma_gold_30nm.py --preview-only
```

既定の `results/measured_30nm/` に `geometry_preview.svg`（断面図・底側と中間高さの上面図）、
`geometry_layers.csv`（RCWAへ渡す各層の高さ・半径・材料区分）、`settings.json` を保存します。
Pillowが使える場合は同じ図を `geometry_preview.png` にも保存します。
通常の光学計算でも、これらを**計算開始前**に同じ出力先へ保存します。谷部金なしの図は
`--valley-gold-nm 0 --preview-only --output-dir studies/pmma_gold_motheye/results/measured_30nm_no_valley`
で作成できます。谷部金の有無にかかわらず、PMMA形状32層＋頂部金円板1層の計33層です。既定の谷部金30 nmでは上側の殻30層と谷部2層に分け、層の厚さを調整して30 nmの境界に一致させます。

既定のM=14、PMMA形状32分割、ASR格子256×256、`outer` 写像、波長450/550/700 nmで計算します。

```bash
python studies/pmma_gold_motheye/run_pmma_gold_30nm.py --device cuda
```

谷部なしとの比較には、出力先も変えてください。

```bash
python studies/pmma_gold_motheye/run_pmma_gold_30nm.py \
  --valley-gold-nm 0 --output-dir studies/pmma_gold_motheye/results/measured_30nm_no_valley \
  --device cuda
```

次数収束を試す場合：

```bash
python studies/pmma_gold_motheye/run_pmma_gold_30nm.py \
  --orders 10,12,14,16,18,20,22 --device cuda \
  --output-dir studies/pmma_gold_motheye/results/measured_30nm_order
```

結果は指定先の `report.json`、`spectrum.csv`、再開用 `checkpoint.json` です。単一Mの結果は `single_order_unverified` で、収束済みを意味しません。次数収束が通っても形状分割数とASR格子の収束確認が別途必要です。PMMA表の開始波長は404.7 nmのため、従来の400 nm点はこの表だけでは計算できません。M=20、100層、400–700 nmの101点を計算する場合は、400 nm付近の短い外挿を明示的に行う `run_spectrum_M20_400_700_101.py` を使います。条件と所要時間は `README_spectrum_M20_101_ja.md` を参照してください。

## 高次数でOOMになった場合

`show_checkpoint_results.py` は保存済みの `checkpoint.json` だけを読み、指定した次数までのR・T・Aと次数間の差を表示します。再計算や `torch` の読み込みは行いません。出力パスが既定と異なる場合は、実際のチェックポイントを第1引数に指定してください。

```bash
python3 studies/pmma_gold_motheye/show_checkpoint_results.py \
  studies/pmma_gold_motheye/results/measured_30nm_Nz100_order/checkpoint.json \
  --max-order 20
```

判定欄は、保存済みの全波長でR・T・Aの変化がそれぞれ許容値以下かを示します。次数収束の候補は高次数側の連続する2段階以上で条件を満たしたときだけ表示します。不完全な次数は判定から除外します。

## 次数収束の図

`plot_order_convergence.py` は計算済みのチェックポイントから、波長別のR・T対Mと、隣接次数間の絶対変化を示す図を作ります。100層でM=22がOOMになった場合は、次のようにM=20までを指定します。

```bash
python3 studies/pmma_gold_motheye/plot_order_convergence.py \
  studies/pmma_gold_motheye/results/measured_30nm_Nz100_order/checkpoint.json \
  --max-order 20
```

チェックポイントと同じディレクトリに `order_convergence_M20_values.svg` と `order_convergence_M20_deltas.svg` を出力します。PillowがあればPNGも同時に作成します。SVGの作成には追加ライブラリは不要です。不完全な次数は図から除外します。

M=4,6,8を追加する場合、既存のチェックポイントと同じ出力先で `--orders` だけを変更すると署名不一致になります。低次数を別の出力先で計算し、図作成時に2つのチェックポイントを指定してください。以下は `studies/pmma_gold_motheye` を作業ディレクトリとした例です。

```bash
python3 run_pmma_gold_30nm.py --slices 100 --orders 4,6,8 --device cuda \
  --output-dir results/measured_30nm_Nz100_order_4_6_8
python3 plot_order_convergence.py \
  results/measured_30nm_Nz100_order/checkpoint.json \
  results/measured_30nm_Nz100_order_4_6_8/checkpoint.json \
  --max-order 20 --output-prefix results/order_convergence_M4_M20
```

図示スクリプトは両実行の設定が次数リスト以外で一致することを確認し、設定が異なる結果の混合を拒否します。元の `converge_order.py` は旧来の別モデル用です。
M=4,6,8だけの `report.json` の収束判定は全次数を含まないため、全範囲の判定には用いないでください。
全次数の数値表は次で表示できます。

```bash
python3 show_checkpoint_results.py \
  results/measured_30nm_Nz100_order/checkpoint.json \
  results/measured_30nm_Nz100_order_4_6_8/checkpoint.json --max-order 20
```
