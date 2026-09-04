# Wang et al. (2022) 図8：2D ASR-RCWA再現手順

対象論文は L. Wang, D. Fang, H. Jin, J. Li, “2D rigorous coupled wave analysis
with adaptive spatial resolution for a multilayer periodic structure,” *Optics
Express* **30**, 21295–21308 (2022), DOI
[10.1364/OE.459110](https://doi.org/10.1364/OE.459110) である。

再現コードは `paper_reproductions/wang2022_fig8/reproduce.py`、独立検証は
`paper_reproductions/wang2022_fig8/validation/validate.py` にある。
この実装は既存の `rcwa_ext/asr_maps.py`、`rcwa_ext/asr.py`、
`rcwa_ext/scattering.py` を使用し、貼付案にあった未存在の `rcwa_solver_2.py` は必要としない。

## 1. 再現対象と固定条件

| 項目 | 値 |
|---|---:|
| 格子 | 正方格子 |
| 周期 | \(\Lambda_x=\Lambda_y=30\ \mathrm{mm}\) |
| 金属パッチ | 中心配置の正方形 |
| fill factor | \(f_x=f_y=0.5\)、一辺15 mm |
| 厚さ | \(h=0.01\ \mathrm{mm}\) |
| 背景・上下半空間 | 空気、\(\epsilon_r=\mu_r=1\) |
| 論文の金属 | \(\epsilon_m=1-j10^6\) |
| 入射 | 空気側から垂直入射、x偏光 |
| 周波数 | 2–18 GHz |
| ASRパラメータ | \(G=0.001\) |
| ASR-FR | \(N=M=8\) |
| ASR（FRなし） | \(N=M=20\) |

論文は \(e^{+j\omega t}\) の符号規約である。一方、torcwaは
\(e^{-j\omega t}\) なので、受動媒質を保つためコードでは複素共役の

\[
\epsilon_m^{\mathrm{torcwa}}=1+j10^6
\]

を用いる。単位はmmに統一し、GHzは \(1/\mathrm{ns}\) なので

\[
f_{\mathrm{torcwa}}[\mathrm{mm}^{-1}]
=\frac{f[\mathrm{GHz}]}{299.792458}
\]

と変換する。既定のsampling grid 256×256と周波数33点（0.5 GHz間隔）はこの実装で
Fourier係数と曲線を十分滑らかに得るための選択であり、論文が開示した固有条件ではない。
必要なら `--grid` と `--points` を収束させる。

## 2. ASR写像

パッチ境界を含む各物理区間 \([x_{l-1},x_l]\) を変換区間
\([u_{l-1},u_l]\) へ写す。論文式(1)は

\[
x(u)=a_{1,l}+a_{2,l}u+\frac{a_{3,l}}{2\pi}
\sin\!\left(2\pi\frac{u-u_{l-1}}{\Delta u_l}\right),
\]

\[
a_{2,l}=\frac{\Delta x_l}{\Delta u_l},\qquad
a_{3,l}=G\Delta u_l-\Delta x_l.
\]

したがって境界でのJacobianは

\[
\left.\frac{dx}{du}\right|_{u=u_{l-1},u_l}=G,
\]

となり、不連続境界近傍に標本が集中する。変換区間幅は論文式(2)の周期正規化形

\[
\Delta u_l
=\Lambda_x\frac{(\Delta x_l)^{1/3}}
{\sum_k(\Delta x_k)^{1/3}},
\]

で決め、y方向も同様に \(y(v)\) を構成する。この計算が
`rcwa_ext/asr_maps.py::_piecewise_asr_map` である。

## 3. 変換媒質とFourier因数分解

\(f(u)=dx/du\)、\(g(v)=dy/dv\) とすると、斜交しない分離写像における変換媒質は

\[
\epsilon_{11}=\epsilon_{uv}\frac{g}{f},\qquad
\epsilon_{22}=\epsilon_{uv}\frac{f}{g},\qquad
\epsilon_{33}=\epsilon_{uv}fg,
\]

\[
\mu_{11}=\mu_{uv}\frac{g}{f},\qquad
\mu_{22}=\mu_{uv}\frac{f}{g},\qquad
\mu_{33}=\mu_{uv}fg.
\]

ASR-FRでは、不連続関数の積をすべて単純なLaurent則で畳み込まない。
`CustomRCWA_ASR_FR._factorized_bttb` は、\(\epsilon_{11}\) に対してu方向逆則・
v方向直接則、\(\epsilon_{22}\) に対してu方向逆則を適用後、v方向の相補的逆則を適用する。
これが `factorization_rules=True` である。比較対象のASRは同じ座標写像を使うが
`factorization_rules=False` とし、通常の2D block-Toeplitz畳み込みを用いる。

接線場ベクトルを \(\mathbf e_t=(E_u,E_v)^T\)、
\(\mathbf h_t=(H_u,H_v)^T\) と置くと、Fourier空間の一次系から

\[
\partial_{\tilde z}\mathbf e_t=P\mathbf h_t,\qquad
\partial_{\tilde z}\mathbf h_t=Q\mathbf e_t,
\]

\[
(PQ)W=W\Gamma^2,\qquad V=QW\Gamma^{-1}
\]

を解く。層内固有ベクトルは変換座標のまま他層へ接続せず、論文の変換関係

\[
W_{xy}=T W_{uv},\qquad V_{xy}=T V_{uv}
\]

でCartesian基底へ戻してから界面S行列を作る。実装箇所は
`_build_conversion_matrix_T` と `add_layer_rect_asr` である。このため、同じ通常の
Redheffer S行列cascadeをASR層にも使用できる。

## 4. 図8のpower

垂直入射かつ方位角0ではtorcwaのp入力がCartesian x入力になる。出力側のp/s成分を
両方含め、各回折次数 \((m,n)\) のpower-normalized振幅を足す。

\[
R_{\mathrm{tot}}=\sum_{m,n}(|r_{pp}^{mn}|^2+|r_{sp}^{mn}|^2),
\quad
T_{\mathrm{tot}}=\sum_{m,n}(|t_{pp}^{mn}|^2+|t_{sp}^{mn}|^2),
\]

\[
R_{00}=|r_{pp}^{00}|^2+|r_{sp}^{00}|^2,\qquad
T_{00}=|t_{pp}^{00}|^2+|t_{sp}^{00}|^2.
\]

図8(b)は

\[
\Delta R=R_{\mathrm{tot}}-R_{00},\qquad
\Delta T=T_{\mathrm{tot}}-T_{00}
\]

である。空気中・垂直入射の最初のRayleigh cutoffは

\[
f_c=\frac{c}{\Lambda}=9.993081933\ \mathrm{GHz}\simeq10\ \mathrm{GHz}.
\]

したがって、cutoff未満では数値誤差を除き \(\Delta R=\Delta T=0\)、cutoffより上では
grating lobeのpowerとして正になる。損失金属なので

\[
A=1-R_{\mathrm{tot}}-T_{\mathrm{tot}}\ge0
\]

も各点で保存する。

## 5. 実行方法

`outputs` ディレクトリで次を実行する。

```bash
python paper_reproductions/wang2022_fig8/reproduce.py --study fig8 --device cuda
```

出力先は既定で `paper_reproductions/wang2022_fig8/results/` であり、以下を生成する。

- `fig8_powers.csv`: 全power、差分、吸収、次数、計算時間
- `fig8.png`: 図8(a)(b)形式の2パネル図
- `fig8_metadata.json`: 論文条件、符号規約、数値条件、DOI

計算を中断した場合は、各周波数終了時のCSV checkpointから再開できる。

```bash
python paper_reproductions/wang2022_fig8/reproduce.py --study fig8 --device cuda --resume
```

CPUでも動作するが、論文のASR \(N=M=20\) は

\[
N_h=(2N+1)^2=1681,\qquad 2N_h=3362
\]

次元の密行列固有値問題になる。complex128の3362×3362行列1枚だけで約173 MiBであり、
実際には多数の行列と固有ベクトルが同時に存在するため数GiB以上を要する。
最初は次のsmoke testが適切である。

```bash
python paper_reproductions/wang2022_fig8/reproduce.py --study smoke --device cpu --no-plot
python paper_reproductions/wang2022_fig8/validation/validate.py
python paper_reproductions/wang2022_fig8/validation/validate.py --integration
```

Redhefferとの独立比較には次を使える。

```bash
python paper_reproductions/wang2022_fig8/reproduce.py --study smoke --cascade algo2a --output-dir smoke_algo2a
```

6 GHzで図9型の次数収束も実行できる。

```bash
python paper_reproductions/wang2022_fig8/reproduce.py --study convergence --max-order 24 --device cuda --no-plot
```

## 6. HFSSデータと「再現」の範囲

論文はunderlying dataを公開しておらず、著者への依頼で入手可能としている。そのため、
コード内にHFSS点を推定・捏造していない。著者提供値またはHFSSで再計算した値がある場合だけ、
次のCSVを用意する。

```csv
frequency_GHz,R00,T00
2.0,0.0,0.0
2.5,0.0,0.0
```

値を実データへ置き換え、次のように重ねる。

```bash
python paper_reproductions/wang2022_fig8/reproduce.py --study fig8 --resume --hfss-csv hfss_fig8.csv
```

ここで再現できるのは論文条件に基づくASR-FR/ASR曲線、cutoff挙動、図8(b)の差分である。
HFSSとのピクセル単位の完全一致は、元データがない限り検証対象に含めない。

## 7. 群論短縮について

この例は正方格子・正方形パッチでありD6ではなくC4v対称である。また論文の図8は
矩形Fourier打切りをそのまま用いたASR/ASR-FR比較である。今回の再現では結果を変えうる
未検証のC4v固有値セクター短縮を導入していない。計算出力はx入射に必要な前進
`Tf/Rf` のみとする `smatrix_size="half"` でS行列保存量を減らしているが、層固有値問題は
論文通りfull polarizationで解く。

## 8. 貼付案からの主な修正

- importを `from rcwa_ext import CustomRCWA_ASR_FR` に変更した。
- constructorの `freq/order/L` を、local mixinの契約通り位置引数にした。
- 空気のinput/output layerを明示した。
- `smatrix_size="half"` とし、x入射に不要な後進側公開Sブロックを計算しない。
- 条件数は `--condition-number` のときだけ計算し、未計算の `None` を参照しない。
- 図8(b)の \(\Delta R,\Delta T\) をCSVと図へ追加した。
- 周波数点ごとのatomic checkpointと `--resume` を追加した。
- 非公開HFSS値を「論文の全曲線」と誤称せず、外部CSV overlayに分離した。
- 6 GHz以下のzero/total一致、passivity、Redheffer/Li-2a一致を確認する検証を追加した。
