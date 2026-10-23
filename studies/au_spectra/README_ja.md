# Au単体 / PMMA+Au のスペクトル

`run_au_spectra.py` を既存コードのルートに置く。既存のAu材料・形状・メモリ節約ワーカーと、Galerkin比較用の `run_ag_nv_factorization_check.py` を使う。
他の既存ファイルは変更しない。ZIPは作成しない。

```sh
python -u run_au_spectra.py \
  --device cuda:0 \
  --factorizations symmetric-nv,galerkin \
  --structures both \
  --wavelengths 400:700:5 \
  --order 36 --slices 210 --grid 512 \
  --reference-root studies/au_ag_nv_order_slices/results/orders_only_Nz210_M4-36_Q512_3wl \
  --output-root studies/au_spectra/results/M36_Nz210_Q512_400-700_61pt
```

NVだけの場合は `--factorizations symmetric-nv`、Galerkinだけなら `--factorizations galerkin`。
既定値はNVのみ。末尾に `--dry-run` を付けると計算予定を確認するだけで、ファイルを書かない。

## 条件

- 2D三角格子、周期200 nm、高さ500 nm、先端半径5 nm、垂直入射、x偏光。
- Au単体：底面半径95 nm、半無限Au基板。透過率は0で、構造＋基板の全吸収率は `A=1-R`。
- PMMA+Au：PMMAの底面半径65 nm、Auの側面・上部・谷部の膜厚各30 nm、PMMA基板。`A=1-R-T`。
- M36、プロファイル210分割。PMMA+Auでは上部キャップ層が加わる。
- Q512、complex128、D6、Redheffer、層ごとの不要ストレージ解放、各条件を独立プロセスで実行。
- Auは以前の `studies/gold_motheye_nv/data/au_measured_nk.csv` を使用。PMMAも以前の分散表・400 nm側への外挿方針を使用。
- Galerkinは以前のM14診断と同じ、横方向に通常のフーリエ畳み込みを用いる比較方式。NVの横方向補正を置き換える。

61波長×2構造で各手法122条件。両方なら244条件。
400/550/700 nmの既存NVチェックポイントについて、M・Nz・Q・材料・形状・ソルバーソースが一致する結果を再利用する。
参照元の3波長・両構造について、条件とソースが一致すればNV最大6条件を再利用できる。
一致しない場合はその波長も新規計算する。手元の確認ではAu分散ローダーのソース署名に差があり、新規予定はNV122＋Galerkin122条件。
元の結果は変更しない。Galerkinの値をNVの値として再利用しない。

## 出力と再開

- `all_spectra.csv`：手法・構造・波長ごとのR/T/A (%)、吸収の内訳、正常/非物理値/OOM/失敗の状態。
- `averages.csv`：各手法・構造の平均R/T/A等。主値は台形積分を帯域幅で割った平均。61点の単純平均も併記。
- `Au_spectra.png` / `.svg`：2構造×R/T/Aの図。NVは実線、Galerkinは破線。平均値を図内に表示。Xマークは使わない。
- `summary.json`：状態の件数、平均、完了状況。
- 各手法・構造・波長の `report.json`：結果、材料/ソース署名、参照元など。

同じコマンドで再開すると正常終了済みの条件を再利用する。失敗・非物理値も既定では再試行しない。
`--retry-failures` を追加すれば失敗した条件を再試行できる。
OOMや非物理値が出ても他の波長・構造・手法を続行し、生の非物理値を保存する。
全波長が既存のパワー検証に合格したスペクトルについてのみ全帯域平均を算出する。
図は非物理値も表示し、欠測部分は線を切る。失敗・非物理値が残る場合の終了コードは2。
M36/Nz210/Q512はこれまでの比較条件に合わせた設定であり、61波長全体の収束を保証する設定ではない。

```sh
python -m unittest validation.test_au_spectra
```
