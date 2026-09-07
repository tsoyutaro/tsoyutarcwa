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
├── diagnose_mapping.py
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

- `--solver nvm`（既定・baseline経路）: 内円・外円の誘電率Fourier係数をBessel関数で解析的に
  構成する同心コアシェルNVM。二つの円の法線は同じ半径方向なので、一つの周期的
  法線射影場で両界面へLiの逆則を適用する。hard rasterは使用しない。現在保存されている
  高次数計算では3経路のうち最も安定だが、N=31まで収束振動が残るため最終結果としては未確定である。
- `--solver matched-nvm`（実験経路）: matched-coordinate写像でASRを行った後、一般化Li
  normal-D/tangential-E因数分解を不連続な誘電率tensorへ適用する。透磁率tensorには
  NV補正を重ねず、座標変換用のWeiss対称因数分解だけを適用する。論文式(8)--(10)と同様に、
  NV補正を誘電率側へ限定した構成である。ただし保存済みdouble写像の収束試験は13点中
  9点が非受動で未収束のため、再現結果として使用しない。
- `--solver matched-asr`: 曲面NV因数分解を使わないmatched-coordinate ASR。
  `double`写像ではWeiss対称因数分解を使用する。Peng Fig. 2の高コントラストAgでは
  高次数側がまだ安定していないため、ASR単独との比較診断用である。

- `--radial-mapping auto`（既定）: この同心コアシェルでは単調性保証付き`double`を選ぶ。
- `--radial-mapping outer`: 従来互換の診断用。外円 `R` だけへ整合し、内円 `r` は
  同じ変換座標上で求積する。ただしPeng形状の `R/p=30/62` と `G=0.001` では
  `min(det J)`が約`1.2e-10`まで低下するため、安全閾値により固有値計算前に拒否される。
- `--radial-mapping double`: 計算空間の固定支持曲線 `h=H/3, 2H/3` をそれぞれ
  内円と外円へ写す、半径方向C2の二重matched写像を使う。内外半径の勾配も写像と
  Jacobianを通して保持する。

二重写像では、中心、内円、外円、周期セル境界を零曲率のquintic Hermite区間で接続する。
`matched-asr`はWeiss対称因数分解、`matched-nvm`は誘電率側の一般化Li
normal-D/tangential-E factorizationを用いる。Peng形状のouter-only写像へ
論文の`G=0.001`をそのまま使った保存結果は発散したため、まず既定の解析`nvm`で
次数収束を確認し、matched経路は独立な診断として比較する。論文の`G=0.001`は段差近似した分離ASRの設定であり、現在の
非分離円写像のJacobian下限を保証する値ではない。このため非分離写像のCLI既定値は
`G=0.03`とする。論文値を入力した感度試験は`--asr-g 0.001`と明示する。
どの方式も論文著者の段差型 separable ASR と補間NV場をbit-for-bitで複製するものではない。
今回の目的は同一の物理問題に対する収束解の比較である。

固有値計算を始める前に写像だけを診断する場合:

```powershell
python -m paper_reproductions.peng2025.diagnose_mapping --grid 256 --asr-g 0.03 --device cuda
```

`outer`と`double`について`minimum_jacobian`、Jacobianの点ごとの最大条件数、実効半径勾配を
JSONへ保存する。これは次数4以上の固有値問題を解かないため、最初に実行できる。
`usable=true`は写像が安全閾値を通ったことだけを意味し、RCWA解の受動性や次数収束を保証しない。

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

Fig. 2(d) と同じ 1–3 THz、次数23の基準計算:

```powershell
python -m paper_reproductions.peng2025.reproduce_square --study spectrum --solver nvm --device cuda
```

Fig. 2(c) と同じ 1.95 THz の次数収束:

```powershell
python -m paper_reproductions.peng2025.reproduce_square --study convergence --solver nvm --orders 4,6,8,10,12,14,16,18,20,22,23,24,26 --device cuda
```

実験的matched-NVM経路を診断する場合（結果を採用する前に受動性と末尾収束を必ず確認）:

```powershell
python -m paper_reproductions.peng2025.reproduce_square --study convergence --solver matched-nvm --radial-mapping auto --orders 4,6,8,10,12,14,16,18,20,22,23,24,26 --device cuda
```

論文にない対称性短縮を使う場合:

```powershell
python -m paper_reproductions.peng2025.reproduce_square --study spectrum --solver nvm --use-symmetry --device cuda
```

正入射x偏光のC2v sectorは完全行列を厳密にblock対角化する計算短縮である。N=8の回帰試験では
full計算との差はR/T/Aで最大`9e-12`だった。N=30を超える収束診断では、まず既知次数を一つ
含めてfull計算との一致を確認したうえで`--use-symmetry`を使用できる。

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

正方格子コードは既定で方式と試験を分離した
`outputs/paper_reproductions/peng2025/results/square_<solver>_<mapping>_<study>/`
（`nvm`ではmapping名なし）に次を生成する。`--output-dir`を指定すれば任意の場所へ変更できる。

- `square_mi.csv`
- `square_mi.png`
- `square_mi_metadata.json`

三角格子比較コードは既定で `outputs/paper_reproductions/peng2025/results/hex_supercell/` に次を生成する。

- `hex_vs_supercell.csv`
- `hex_vs_supercell.png`
- `hex_vs_supercell_metadata.json`

CSV には R、T、A、Ag 誘電率、次数、格子、計算時間、対称性診断を保存する。
スーパーセル行には `m+n` 奇数次数の反射・透過 power も含む。
収束試験のmetadataには、末尾3点のR/T/A変動幅、厳密な受動性、残留受動性誤差を分けた
`convergence_assessment`も保存する。`provisional_small_passivity_error`は収束確定ではない。

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

検証にはmatched-ASR後のpull-back法線が一般化Li経路を実際に選択したかも含める。
D6は過小なstarでの見かけの負吸収を合格させず、警告が出れば検証全体も失敗になる。
したがってsmoke/invariance検証の合格と、次数23の論文スペクトル収束は別々に判定する。

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
