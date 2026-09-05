# MOSMITE（PMMAモスアイ）へのAu 30 nm蒸着モデル

## なぜ従来の二重matched円だけでは計算しないか

既定形状は周期200 nm、PMMA底半径75 nm、Au公称膜厚30 nmです。底部の外半径は
105 nmとなり、最近接セル境界の100 nmを越えます。これは写像の不具合ではなく、隣接する
Au被覆が合体して連続膜へ移る幾何学的トポロジー変化です。重なる円を単一円matched写像へ
無理に入力してはいけません。

この実装は、各zスライスを次のように自動分類します。

- `r_core + 30 nm < (period - asr_min_gap)/2`: 内外界面に整合する単調C2二重matched-ASR。
- それ以外: 周期像を含む最近接格子点距離から、PMMAコア、合体Au、残留空気を作るラスタ層。

既定の `asr_min_gap_nm=2` は、外円が接触する直前の極端に圧縮されたASR領域を避けます。
32スライスなら概ね29層がASR、底部3層が合体ラスタになります。三角格子の被覆半径は
`period/sqrt(3)=115.47 nm`なので、外半径105 nmでは底部にも小さな空気孔が残り、底部を
単純な一様Au層へ置換してはいけません。

ラスタ層を完全D6のsource-row縮約と混在させる機能は未実装なので、このスタディは全基底を
使用します。そのため従来のD6短縮計算より重くなります。金属の合体部はmatched-ASRではないため、
回折次数、z層数、ラスタ格子をそれぞれ収束させる必要があります。

## 推奨する計算順序

最初に依存ライブラリ不要の形状・配線検証を実行します。

```bash
python studies/mosmite_gold_30nm/validation/validate.py
```

torch/torcwa環境では、ASR層と合体ラスタ層を同じS行列へ接続する最小計算も確認します。

```bash
python studies/mosmite_gold_30nm/validation/validate.py --integration --device cuda
```

次に400、550、700 nmで独立収束を確認します。全基底なので、まず小さい候補から開始します。

```bash
python studies/mosmite_gold_30nm/converge.py \
  --axis order --candidates 4,6,8,10 \
  --slices 48 --grid 384 --device cuda \
  --output-prefix studies/mosmite_gold_30nm/results/convergence/order

python studies/mosmite_gold_30nm/converge.py \
  --axis slices --candidates 24,32,48,64 \
  --order 8 --grid 384 --device cuda \
  --output-prefix studies/mosmite_gold_30nm/results/convergence/slices

python studies/mosmite_gold_30nm/converge.py \
  --axis grid --candidates 192,256,384,512 \
  --order 8 --slices 48 --device cuda \
  --output-prefix studies/mosmite_gold_30nm/results/convergence/grid
```

得られた推奨値を用いて400–700 nmを10 nm刻みで計算します。

```bash
python studies/mosmite_gold_30nm/run_spectrum.py \
  --wavelengths 400:700:10 \
  --order 8 --slices 48 --grid 384 --device cuda \
  --output-prefix studies/mosmite_gold_30nm/results/mosmite_au30
```

CSV、JSON、PNGを出力します。最終報告前には `--asr-min-gap-nm 2` と `5` も比較し、ASR／
ラスタ切替位置への依存が許容差以下であることを確認してください。

## 物理モデル上の注意

これは膜厚30 nmの等方的なコンフォーマル被覆を周期和として扱うモデルです。真空蒸着が
線源方向性を持つ場合、側壁膜厚は30 nmではありません。最終的には断面SEM/TEMまたは
蒸着角度・回転条件から `r_outer(z)` を更新してください。Au分散は既定でRakić bulkモデルです。
薄膜の粒界・粗さ・接着層を含める場合は、実測 `n,k` CSVを `--gold-model csv --gold-csv ...`
で指定します。
