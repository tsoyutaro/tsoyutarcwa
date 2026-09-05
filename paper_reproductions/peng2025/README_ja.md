# Peng–Zhang 2025 円形 aperture–particle 配列の再現計算

## 1. 対象

対象論文は S. Peng and X. Zhang, *Rigorous Coupled-Wave Analysis of
Multilayer Metal-Insulator Aperture-Particle Composite Periodic Arrays*,
IEEE Antennas and Wireless Propagation Letters 24, 1615–1619 (2025),
DOI: 10.1109/LAWP.2025.3543371 である。

今回の実行コードは、パラメータが最も明確な論文 Fig. 2 の MI 単層構造を扱う。

| 項目 | 論文記載値 |
|---|---:|
| 正方格子周期 | 62 µm |
| 外半径 R | 30 µm |
| 内半径 r | 14 µm |
| Ag パターン層厚 | 1 µm |
| PI 比誘電率 | 3.5 + 0.009i |
| Fig. 2 の PI 厚さ h2 | 本文に数値記載なし |
| 周波数 | 1–3 THz |
| 収束確認周波数 | 1.95 THz |
| 入射 | 空気側、垂直、TM |
| 論文 ASR–NV スペクトル次数 | Nx=Ny=23 |

本実装では、Ag の連続膜を背景とし、`r < rho < R` が空気の環状開口、
`rho < r` が Ag 中心粒子であるとする。PIはパターン層内の環状材料ではなく、
パターン層の下側にある出力基板である。したがって実装上は

```text
input = air
pattern = Ag background / air annulus / Ag core
output substrate = PI
```

である。

## 2. 論文だけから一意に決まらない条件

論文は Ag に Drude モデルを使うと述べるが、`epsilon_inf`、プラズマ角周波数、
衝突角周波数を掲載していない。また、Fig. 2 の MI 基板厚 `h2` の数値も本文にない。
このため既定値は次の明示的な仮定とした。

- Ag: `epsilon_inf=1`, `omega_p=1.37e16 rad/s`, `gamma=2.73e13 rad/s`
- 時間依存: `exp(-i omega t)`、受動媒質は `Im(epsilon)>=0`
- MI の PI 基板: 既定では半無限出力媒質。`--pi-thickness-um`を指定した場合は
  有限PI層、その下を空気とする

Drude 定数はすべてコマンドラインで変更でき、実行時の値は metadata JSON に保存する。
著者が使用した値が分かれば、その値を指定して再計算する必要がある。

## 3. フォルダ構成

```text
outputs/paper_reproductions/peng2025/
├── common.py
├── reproduce_square.py
├── compare_hex_supercell.py
├── README_ja.md
├── results/
│   ├── square/
│   └── hex_supercell/
└── validation/
    ├── validate.py
    └── results/
```

通常の論文再現結果と検証結果を分離し、旧smoke結果も
`validation/results/archive/`へ保存している。

## 4. 正方格子コード

`reproduce_square.py` は論文と同じ物理構造の収束解を独立に確認するため、次の
ソルバーを選択できる。

- `--solver nvm`（既定）: 内円・外円の誘電率Fourier係数をBessel関数で解析的に
  構成する同心コアシェルNVM。二つの円の法線は同じ半径方向なので、一つの周期的
  法線射影場で両界面へLiの逆則を適用する。hard rasterは使用しない。
- `--solver matched-asr`: 本プロジェクトのmatched-coordinate ASR。以下の
  `--radial-mapping`を選択する。

- `--radial-mapping outer`（既定）: 従来互換。外円 `R` だけへ整合し、内円 `r` は
  同じ変換座標上で求積する。
- `--radial-mapping double`: 計算空間の固定支持曲線 `h=H/3, 2H/3` をそれぞれ
  内円と外円へ写す、半径方向C2の二重matched写像を使う。内外半径の勾配も写像と
  Jacobianを通して保持する。

二重写像では、中心、内円、外円、周期セル境界を零曲率のquintic Hermite区間で接続し、
一般化Li normal-D/tangential-E factorizationを用いる。ただし高コントラストAgに対する
`double`の次数収束は未確立なので、論文結果の独立検算には既定の`nvm`を優先する。
どの方式も論文著者の段差型 separable ASR と補間NV場をbit-for-bitで複製するものではない。
今回の目的は同一の物理問題に対する収束解の比較である。

実行位置を混同しないこと。リポジトリルートからはモジュール形式を推奨する。

```powershell
python -m paper_reproductions.peng2025.reproduce_square --study smoke --device cpu
```

すでに`paper_reproductions/peng2025`へ移動済みなら、次のようにファイル名だけを指定する。

```powershell
python reproduce_square.py --study smoke --device cpu
```

移動後に`python paper_reproductions/peng2025/reproduce_square.py`と指定すると、添付ログのように
パスが二重になり、ファイルが見つからない。

高速 smoke test:

```powershell
python -m paper_reproductions.peng2025.reproduce_square --study smoke --device cpu
```

内外両円をmatchedする場合:

```powershell
python -m paper_reproductions.peng2025.reproduce_square --study smoke --solver matched-asr --radial-mapping double --device cpu
```

Fig. 2(d) と同じ 1–3 THz、報告次数 23:

```powershell
python -m paper_reproductions.peng2025.reproduce_square --study spectrum --solver nvm --device cuda
```

Fig. 2(c) と同じ 1.95 THz の次数収束:

```powershell
python -m paper_reproductions.peng2025.reproduce_square --study convergence --solver nvm --device cuda
```

論文にない対称性短縮を使う場合:

```powershell
python -m paper_reproductions.peng2025.reproduce_square --study spectrum --solver nvm --use-symmetry --device cuda
```

Fig. 2のPI厚`h2`は論文本文に数値がない。有限PI膜を仮定して感度を確認する場合は、例えば

```powershell
python -m paper_reproductions.peng2025.reproduce_square --study spectrum --solver nvm --pi-thickness-um 12 --device cuda
```

とする。これは論文値の断定ではなく仮定であり、metadata JSONへ記録される。

`--order 8`は論文Fig. 2(d)のASR-NV次数23より低い診断条件である。スペクトル計算で
受動性違反が一つでも生じた場合、既定ではCSVとmetadataだけを保存してエラー終了し、
誤ったスペクトル図を生成しない。次数収束調査では低次の非受動点も診断情報なので、
赤い`x`で区別して図を保存し、正常終了する。`--allow-nonpassive`は非受動スペクトルを
原因調査用に描く場合だけ使用する。

次数 23 では `(2N+1)^2=2209` harmonics、full vector modal dimension は 4418 である。
複素倍精度 eigensolve の作業領域は単一行列の約 312 MB より大幅に大きくなるため、
本計算は十分な RAM/VRAM を備えた環境で行う。

## 5. 三角格子と直交スーパーセル

最近接周期を `a=62 µm` とする三角 Bravais 格子は

\[
\mathbf a_1=(a,0),\qquad
\mathbf a_2=(a/2,\sqrt3a/2)
\]

で表される。これと同じ無限格子を生成する最小直交セルは

\[
\mathbf A_1=\mathbf a_1=(a,0),\qquad
\mathbf A_2=2\mathbf a_2-\mathbf a_1=(0,\sqrt3a)
\]

で、セル内に二つの同一サイトを持つ。真の正方形セルでは `sqrt(3)` が無理数なので
三角格子を有限サイズで厳密に周期化できない。したがって比較コードの「正方格子側」
は、Cartesian RCWA が扱える `a x sqrt(3)a` の直交長方形スーパーセルである。

スーパーセルには隠れた並進

\[
\mathbf t=(a/2,\sqrt3a/2)
\]

がある。直交セルの回折次数 `(m,n)` に対する構造因子から `m+n` が奇数の次数は
禁制になる。コードはこの folded-order power も出力し、スーパーセルが本来の
primitive 周期を壊していないか確認する。

高速比較:

```powershell
python paper_reproductions\peng2025\compare_hex_supercell.py --study smoke --device cpu
```

周波数スペクトル:

```powershell
python paper_reproductions\peng2025\compare_hex_supercell.py --study spectrum --order 8 --device cuda
```

D6 の x 入射セクターを native matched-ASR に使う場合:

```powershell
python paper_reproductions\peng2025\compare_hex_supercell.py --study spectrum --order 8 --use-d6 --device cuda
```

三角primitiveの内外両円をmatchedし、D6 x入射短縮も使う場合:

```powershell
python paper_reproductions\peng2025\compare_hex_supercell.py --study spectrum --order 8 --radial-mapping double --use-d6 --device cuda
```

比較は三系列を計算する。

1. 三角 primitive、matched-ASR-FR
2. 三角 primitive、standard raster
3. 直交二サイト・スーパーセル、standard raster

格子表現の同値性の合否は 2 と 3 で判定する。1 と 3 を直接比較すると、低次数では
matched-ASR と hard raster の因子分解誤差が格子差のように見えるためである。1 と 2
の差は method convergence として別に保存する。

## 6. 出力

正方格子コードは既定で `outputs/paper_reproductions/peng2025/results/square/` に次を生成する。

- `square_mi.csv`
- `square_mi.png`
- `square_mi_metadata.json`

三角格子比較コードは既定で `outputs/paper_reproductions/peng2025/results/hex_supercell/` に次を生成する。

- `hex_vs_supercell.csv`
- `hex_vs_supercell.png`
- `hex_vs_supercell_metadata.json`

CSV には R、T、A、Ag 誘電率、次数、格子、計算時間、対称性診断を保存する。
スーパーセル行には `m+n` 奇数次数の反射・透過 power も含む。

## 7. 検証

```powershell
python paper_reproductions\peng2025\validation\validate.py --device cpu
```

実施する確認は次の通りである。

- 論文記載幾何定数
- Drude モデルの受動符号
- 二サイト・スーパーセルの隠れた半セル並進対称性
- 解析的な空気環状開口の面積率と raster 面積率
- 正方、三角、D6、直交スーパーセルの有限な R/T/A
- 三角 raster primitive と直交 raster supercell の低次同値性
- `m+n` 奇数 folded orders の消失

CPU smoke validation では全項目が合格した。`N=1`、3 THz の raster 格子表現間の
最大 R/T/A 差は約 `8.29e-4`、禁制 folded-order power は約 `9.99e-37` だった。
D6は過小なstarでの見かけの負吸収を合格させないよう`N=4`で受動性を確認している。
これはコード経路と幾何同値性の検証であり、次数 23 の論文スペクトル収束を保証する
ものではない。

検証JSONとsmoke図は `outputs/paper_reproductions/peng2025/validation/results/` に保存する。
検証には、正方・三角格子の両界面半径一致、Jacobian正値、固定計算空間mask、
内外半径autogradと中心差分の一致、二重写像の受動性、D6短縮との併用を含む。

## 8. 定量的な論文一致に必要な追加情報

次のいずれかが得られれば、定量比較を強化できる。

- 論文で使用した Ag Drude 定数
- Fig. 2 の MI 基板厚または半無限条件の確認
- Fig. 2(c,d) の数値データ
- 著者実装における NV 場の補間格子と境界処理

これらがない限り、論文グラフから数値を創作せず、再現仮定と論文値を分離して報告する。
