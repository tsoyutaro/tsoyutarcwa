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
    ├── validate_paper_asr_nv.py
    └── results/
```

通常の論文再現結果と検証結果を分離し、旧smoke結果も
`validation/results/archive/`へ保存している。

## 4. 正方格子コードとASR修正（2026-09-14）

既定ソルバーを解析NVMに変更した。保存済みの1.95 THz、N=38--40では受動性と
末尾収束判定を満たしている。これは仮定したDrudeモデルの独立baselineであり、
論文のASR-NV実装そのものを再現できたという意味ではない。

| ソルバー | 計算方法と現在の扱い |
|---|---|
| `nvm`（既定） | 内外円のBessel Fourier係数と同心円NVM。基準計算用 |
| `paper-asr` | 階段境界、式(7)、Jacobianを含む厳密なstrip-Li積分。次数・階段格子収束の診断用 |
| `paper-asr-nv` | 座標変換した構成テンソルと法線に一般化Li則を適用。Agでは実験段階 |
| `matched-asr` | 滑らかなdouble写像とWeiss対称因数分解。高コントラストでは診断用 |
| `matched-nvm` | double写像と一般化Li-NV。高コントラストでは診断用 |

今回、次の不具合を修正した。

- 式(7)の逆写像を、強圧縮区間で振動するclipped Newtonから区間を保持した二分法へ変更。
- double写像の周期継ぎ目で、鏡映に対して奇関数となる交差Jacobian成分を左右平均の0に修正。
- 適応座標に材料を移すだけだったASRへ、`diag(g/f, f/g, f*g)`の構成テンソル変換を追加。
- ASR単独では、狭い`1/f`ピークのFFT求積を使わず、Jacobianと材料の区分ごとのFourier積分を厳密計算。
- C2v縮約ではHの再構成も縮約されたP/Qで行い、不要な他対称セクターへの丸め誤差を抑制。
- 一様ポートの接続を回折次数ごとの2×2 solveに、対角波数行列の積を行・列スケーリングに変更。

`paper-asr`の再現スクリプトは、既定の`--asr-interface-rule auto`で
`shared-adaptive`を選ぶ。外部媒質も同じ適応座標の一般化固有モードで表し、
切り詰めたCartesian Fourier基底への不安定な射影を避ける。
この専用経路の出力は総R/T/Aである。有限次数のポート係数は通常のCartesian回折振幅ではなく、
電場復元や任意の異なる写像を混ぜた多層構造には使用しない。
正入射のパターン1層と一様PI層の構成を対象とする。

汎用層APIの`flux-dual`接続は`TE^H C TH=C`を満たすが、PengのAg条件では高次にすると
射影行列が悪条件化する。`projected`は比較診断用である。`paper-asr-nv`では
`auto`は`flux-dual`を選び、`--nv-coordinate-rule covariant`を既定とする。
旧`paper-disclosed`等の選択肢は過去結果を調べるために残したが、適応座標の物理解として採用しない。
今回の構成テンソル・接続の補完は論文の未記載部分に対する実装上の選択であり、論文式の逐語再現とは区別する。

修正後の1.95 THz、階段格子64、G=0.001では、shared-adaptiveのTはN=24,28,32で
0.81260, 0.84483, 0.83707だった。高次の急落は解消したが、末尾幅は0.03223であり、
0.01の収束条件を満たさない。受動性だけで収束済みと判定しない。
Fourier次数と`--staircase-grid`は別々に検証する必要がある。

リポジトリルートから実行する。

```powershell
# 既定NVMの動作確認
python -m paper_reproductions.peng2025.reproduce_square --study smoke --device cpu

# 独立baselineの末尾収束確認
python -m paper_reproductions.peng2025.reproduce_square --study convergence --solver nvm --orders 36,38,40 --device cuda

# 修正したASRの収束診断
python -m paper_reproductions.peng2025.reproduce_square --study convergence --solver paper-asr --orders 8,12,16,20,24,28,32 --staircase-grid 64 --device cuda

# 接続方式の比較
python -m paper_reproductions.peng2025.reproduce_square --study convergence --solver paper-asr --asr-interface-rule flux-dual --orders 8,12,16,20,24,28,32 --device cuda

# 今回追加した物理・数値回帰試験
python -m paper_reproductions.peng2025.validation.validate_core_shell_asr
```

`--use-symmetry`はNVMとASR単独の正入射x偏光では自動で有効になる。
`--no-use-symmetry`で全行列と比較できる。有限PI膜では一様層の縮約が未対応のため、
自動縮約を無効にする。`--pi-thickness-um 12`等は論文値の断定ではなく明示的な仮定である。

`G=0.001`、`--uv-interval-allocation cube-root`をASRの既定とする。
階段格子数、u/v区間配分、IDW設定は論文から一意に決まらないのでmetadataに記録する。
`matched-*`では`--radial-mapping auto`が単調な`double`を選ぶ。
旧`outer`写像はPengの狭い周期ギャップでJacobianが安全閾値を下回るため拒否される。

スペクトル計算で受動性違反があればCSVとmetadataを残し、既定では図を生成せずエラー終了する。
収束調査の低次非受動点は診断情報として残す。収束JSONの`converged`判定は指定した末尾次数と
許容差についてだけ成立し、論文との一致や階段形状の収束を保証しない。

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

式(7)、式(8)--(9)、矩形Laurent積分、IDW単位法線、専用ソルバーdispatch、
低次数R/T/A smokeをまとめて確認するには次を実行する。

```powershell
python paper_reproductions\peng2025\validation\validate_paper_asr_nv.py --device cpu
```

これは実装回帰試験であり、次数23/35の収束を意味しない。

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
