# PMMAモスアイ＋金薄膜の収束計算

## 1. 計算モデル

空気から、金薄膜を被覆したPMMAモスアイへ正入射し、その下を半無限PMMA基板とします。

| 項目 | 既定値 |
|---|---:|
| 収束判定波長 | 400、550、700 nm |
| 格子／周期 | 60度三角格子／200 nm |
| PMMAモスアイ高さ | 500 nm |
| PMMA先端／底半径 | 5／75 nm |
| 金膜厚 | 20 nm |
| PMMA屈折率 | 1.49（無損失・波長非依存） |
| 入射 | 垂直、x偏光 |
| 金分散 | Rakić Lorentz–Drude、または測定CSV |
| S行列 | half、Redheffer |

PMMA形状を (N_z) 枚へ分け、空気側から数えたスライス (j) の中央半径を

\[
r_{c,j}=r_t+(r_b-r_t)
\left(\frac{j+1/2}{N_z}\right)^p
\]

とします。金の外半径は (r_{o,j}=r_{c,j}+t_{\rm Au}) です。各断面は

\[
\epsilon(x,y)=
\begin{cases}
\epsilon_{\rm PMMA},&\rho<r_c,\\
\epsilon_{\rm Au}(\lambda),&r_c\le\rho<r_o,\\
1,&r_o\le\rho
\end{cases}
\]

となります。PMMA先端より上には厚さ (t_{\rm Au})、半径
(r_t+t_{\rm Au}) の金円板を1層置き、上面キャップを近似します。
`--no-top-cap` で無効化できます。

これは「一定の半径方向膜厚＋上面キャップ」という階段近似です。蒸着方向、局所面法線、
シャドーイングで膜厚が変わる場合は、SEM/TEM断面に合わせた形状へ更新してください。

## 2. core–shell matched-ASR

`add_layer_circle_shell_asr` は、次の二つの周期写像を選択できます。

- `radial_mapping="outer"`（既定）: 外側の金–空気円境界だけに適合。
- `radial_mapping="double"`: 内側の金–PMMAと外側の金–空気の両円境界に適合し、
  level-set法線による一般化Li normal-D/tangential-E因数分解を適用。

いずれも

\[
(x,y)=\mathbf X(u,v),\qquad
J=\frac{\partial(x,y)}{\partial(u,v)},\qquad h=\det J
\]

を作ります。等方媒質を写像座標へ移した横成分は

\[
\boldsymbol\epsilon'_{tt}=\frac{\epsilon}{h}
\begin{bmatrix}
x_v^2+y_v^2&-(x_u x_v+y_u y_v)\\
-(x_u x_v+y_u y_v)&x_u^2+y_u^2
\end{bmatrix},
\quad \epsilon'_{zz}=\epsilon h,
\]

で、\(\boldsymbol\mu'\) も同形です。このFourier畳み込み行列から

\[
P Q W=W K_z^2,\qquad V=QWK_z^{-1}
\]

を解き、一般2次元変換 (T,T_z) を介して通常S行列へ接続します。

`double` では計算空間の固定支持曲線 `h=H/3,2H/3` を内外円へ写し、中心、両界面、
周期セル境界を零曲率quintic Hermite区間で接続します。半径方向にはC2で、物質maskは
計算空間に固定されるため、`core_radius` と `outer_radius` の両方をTensor設計変数にできます。
`outer` へtrainable `core_radius` を渡す場合は、従来どおり誤ったゼロ勾配を避けるため例外です。

二重放射支持曲線には既存の逐次u/v対称factorizationを流用せず、計算格子計量と
level-set法線から作る一般化Li因数分解を使います。法線Dには逆則、接線Eには直接則が
適用されます。`factorization_rules=False`を明示した場合だけ比較用の直接畳み込みへ戻ります。
どちらの方式でもFourier次数とASR格子の収束確認は必須です。
実行時は両収束スクリプトへ `--radial-mapping double` を追加します。

## 3. 反射率、透過率、吸収率

単位胞平均Poynting flux

\[
P_z=\frac12\operatorname{Re}\sum_g
(E_{x,g}H_{y,g}^*-E_{y,g}H_{x,g}^*)
\]

から

\[
R=-P_r/P_i,\qquad T=P_t/P_i,\qquad A=1-R-T
\]

を出力します。出力媒質は半無限PMMAなので、(T) はPMMA基板へ進む全伝搬powerです。
PMMAを実数屈折率とした既定モデルでは、(A) は金膜による吸収です。

### 完全D6＋特定偏光の既定短縮

三角格子、同心円、垂直x偏光という現在条件では、ゼロ次sourceはD6の (E_1) 表現に
属します。既定の `--symmetry-reduction d6-source` は12個のD6作用で演算子をReynolds平均し、
x偏光なら (E_1) matrix-unit row 0だけを固有値分解とS行列cascadeへ渡します。

\[
D_{E_1,x}=M(M+1)+1
\]

従来のCs鏡映sectorは (D_{Cs}=3M(M+1)+1) なので、次元は約3分の1、密行列主要項は
理論上約27分の1、二乗メモリは約9分の1です。比較検証用に旧経路へ戻す場合は
`--symmetry-reduction cs-source`、群論を完全に無効化する場合は `--no-symmetry` を指定します。
出力CSVの `symmetry_reduction` は既定で `D6-E1-source-row`、`symmetry_irrep` は `E1` です。

## 4. 独立した収束実行

以下は `outputs` ディレクトリ内で実行します。

回折次数 (M) だけを上げ、profile層数を32に固定:

```bash
python studies/pmma_gold_motheye/converge_order.py \
  --orders 10,12,14,16,18,20,22 --fixed-slices 32 --grid 256 \
  --output-prefix studies/pmma_gold_motheye/results/order/pmma_au_order
```

profile層数 (N_z) だけを上げ、次数を5に固定:

```bash
python studies/pmma_gold_motheye/converge_layers.py \
  --layers 8,12,16,24,32,48 --fixed-order 5 --grid 192 \
  --output-prefix studies/pmma_gold_motheye/results/layers/pmma_au_layers
```

`profile_slices` はPMMA形状の分割数です。上面キャップを使うと実際のpatterned layer総数は
`total_pattern_layers = profile_slices + 1` です。400–700 nmを25 nm刻みで判定するには
`--wavelengths 400:700:25` を追加します。

各候補間で全波長にわたる (R,T,A) の最大絶対変化を求めます。1組だけの偶然の一致を避け、
連続する2段階のrefinementがとも `--tolerance` 以下になった最小の中間候補を採用します。
既定値0.005は0.5 percentage pointです。

推奨手順は次です。

1. `fixed-slices=32` で次数収束を実行する。
2. 得られた次数を `--fixed-order` に設定して層数収束を実行する。
3. 得られた層数を `--fixed-slices` に戻し、次数収束を再実行する。
4. 最終候補で `--grid 192` と `--grid 256` を比較し、内側境界の求積もspot checkする。
5. 最終報告では許容差を0.001程度へ厳しくし、密な波長列で再確認する。

候補上限でも連続2回の合格がなければ、終了コード2と
`candidate_range_insufficient` を返します。候補範囲を上へ広げてください。

## 5. 出力と再開

`--output-prefix` を省略した場合も、次数と層数の結果はそれぞれ
`studies/pmma_gold_motheye/results/order/` と
`studies/pmma_gold_motheye/results/layers/` に分離されます。

- `*_checkpoint.json`: 1 solveごとの再開用データ
- `*_convergence.json`: 比較誤差、判定、推奨値、全仮定
- `*_all_cases.csv`: 全候補・全波長のR/T/A
- `*_selected_spectrum.csv`: 選択候補のR/T/A

異なる条件で同じprefixを使うと設定署名の不一致として停止します。別prefixを指定するか、
古いcheckpointを退避してください。

## 6. 分散、実験パラメータ、検証

既定のRakićモデルはbulk goldです。薄膜では粒径、粗さ、密度、接着層で損失が変わるため、
最終比較には実試料のellipsometryによる (n,k) CSVを推奨します。

```bash
python studies/pmma_gold_motheye/converge_order.py \
  --gold-model csv --gold-csv measured_gold.csv \
  --gold-thickness-nm 15 --pmma-index 1.49
```

CSVは `wavelength_nm,n,k` または
`wavelength_nm,epsilon_real,epsilon_imag` を受け付け、範囲外へ外挿しません。

依存ライブラリなしの構文・配線・分散チェック:

```bash
python studies/pmma_gold_motheye/validation/validate.py
```

torch、torcwaが利用できる環境での最小積層統合チェック:

```bash
python studies/pmma_gold_motheye/validation/validate.py --integration --device cpu
```

最終形状に必要な主条件は、金膜厚、PMMA先端・底半径、profile指数、格子種、実測PMMA分散、
接着層の有無です。既定値は実行可能な出発点であり、試料仕様の代替ではありません。

金モデルの出典は A. D. Rakić et al., *Applied Optics* 37, 5271–5283 (1998),
DOI 10.1364/AO.37.005271、および P. B. Johnson and R. W. Christy,
*Physical Review B* 6, 4370–4379 (1972), DOI 10.1103/PhysRevB.6.4370 です。
