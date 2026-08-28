# 三角格子 matched-ASR・単一偏光短縮 実装／数式レポート

## 1. 達成内容

`rcwa_solver_auto.py` に次を実装した。

- 60°・等辺の三角 Bravais 格子における、中心円柱用の周期的 matched-coordinate ASR。
- 直交／三角格子の双方で使える一般 2D modal conversion (T,T_z)。
- 三角格子の共変波数 (K_1,K_2) を使う transformed-medium (P,Q)。
- D6 に閉じた hexagonal reciprocal-lattice star 上での畳み込み・逆行列。
- 正規入射 x または y 励振に必要な鏡映セクターだけを解く固有値短縮。
- 短縮セクターの Redheffer と Li algorithm 2a、half/quarter S 出力。
- 独立した写像検証と、torcwa を使う統合検証。

検証済みの主要 API は `Lattice.triangular(L)`、`method='matched-asr'`、`GroupTheoryOptions(polarization='x'|'y')` の組合せである。

## 2. 理論上の位置づけ

Weiss らは、曲面境界へ座標線を合わせ、共変 Maxwell 方程式・ASR・対称 Fourier factorization を組み合わせる一般 matched-coordinate FMM を示した。[Weiss et al., Optics Express 17, 8051 (2009)](https://pubmed.ncbi.nlm.nih.gov/19434137/) Essig–Busch は任意形状に対する adaptive coordinate generation を扱っており、非矩形系への展開を支える一般的な枠組みを与える。[Essig and Busch, Optics Express 18, 23258 (2010)](https://publikationen.bibliothek.kit.edu/1000035419) また、座標変換のモジュール化は後続研究でも整理されている。[Küchenmeister, Optics Express 22, 9404 (2014)](https://publikationen.bibliothek.kit.edu/1000042056)

今回の「六角 support function + Hermite radial map」という具体的写像そのものは、上記論文の式をそのまま写したものではなく、本実装で構成した D6-equivariant map である。根拠は次の三点である。

1. Maxwell 方程式の座標共変性に基づく transformed tensor を使う。
2. 計算座標の material interface を物理円へ厳密に写す。
3. セル境界で写像と一次微分を恒等写像へ戻し、周期接続を保つ。

したがって「1D ASR 行列を Cartesian NVM 行列へ後掛けする」経験的操作ではない。先に連続座標変換を定義し、その Jacobian から物性テンソルと modal conversion を導く。

## 3. 三角格子の座標と逆格子

ローカル座標で

\[
\mathbf a_1=(L,0),\qquad
\mathbf a_2=L(1/2,\sqrt3/2)
\]

とする。計算座標 (u,v\in[0,L)) から、未変形の物理座標は

\[
x=u+\frac12v,\qquad y=\frac{\sqrt3}{2}v
\]

である。入射 Cartesian 波数を ((k_{x0},k_{y0})) とすると、各 reciprocal order の共変成分は

\[
\bar k_1=\bar k_{x0}+\frac{2\pi m}{k_0L},
\]

\[
\bar k_2=\frac12\bar k_{x0}+\frac{\sqrt3}{2}\bar k_{y0}
+\frac{2\pi n}{k_0L}.
\]

Cartesian 成分へ戻す式は

\[
\bar k_x=\bar k_1,\qquad
\bar k_y=\frac{\bar k_2-\frac12\bar k_1}{\sqrt3/2}.
\]

`_build_circle_asr_pq` は微分演算子として (K_x,K_y) ではなく (K_1,K_2) を使う。物理面での放射条件と外部媒質の電力は Cartesian (K_x,K_y) を使う。

## 4. D6 matched-coordinate map

### 4.1 最短周期像

セル中心を (mathbf c=(L/2,L/2))（primitive coordinates）とし、各 ((u,v)) について

\[
\mathbf q=(q_1,q_2)=(u,v)-\mathbf c-(iL,jL)
\]

の物理距離

\[
\|\mathbf q\|_g^2=q_1^2+q_2^2+q_1q_2
\]

が最小になる (i,j\in\{-1,0,1\}) を選ぶ。これは中心円柱の最短周期像を選ぶ処理である。

### 4.2 正六角形の support function

\[
h(\mathbf q)=\max\left(
|q_1+q_2/2|,
|q_1/2+q_2|,
|q_1-q_2|/2
\right).
\]

(h=L/2) は三角格子の Wigner–Seitz 正六角形境界であり、(h) は D6 の 6 回回転・6 鏡映に不変である。計算座標上の matched interface を

\[
h(\mathbf q)=R
\]

と置く。ここで (R<L/2) は物理円半径である。

### 4.3 六角形から円への写像

\[
\rho=\frac{h(\mathbf q)}{R},\qquad
\mathbf q_0=\frac{\mathbf q}{\rho}quad(\rho>0)
\]

とすると、(mathbf q_0) は (h=R) 上にある。方向ごとの円合わせ倍率は

\[
s_c(\hat{\mathbf q})=rac{R}{\|\mathbf q_0\|_g}.
\]

これにより、(ho=1) で

\[
\|s_c\mathbf q_0\|_g=R
\]

が厳密に成り立つ。

### 4.4 Hermite ASR

区間 (t\in[0,1]) の cubic Hermite 補間を

\[
\mathcal H(t;y_0,m_0,y_1,m_1,\Delta)
=h_{00}y_0+h_{10}\Delta m_0+h_{01}y_1+h_{11}\Delta m_1
\]

\[
h_{00}=2t^3-3t^2+1,\quad
h_{10}=t^3-2t^2+t,
\]

\[
h_{01}=-2t^3+3t^2,\quad
h_{11}=t^3-t^2
\]

とする。Wigner–Seitz 境界の規格化半径は

\[
\rho_o=\frac{L}{2R}>1.
\]

radial function (F) を

\[
F(\rho)=
\begin{cases}
\mathcal H(\rho;0,1,s_c,G,1),&0\le\rho\le1,\\
\mathcal H\!\left(\frac{\rho-1}{\rho_o-1};s_c,G,\rho_o,1,\rho_o-1\right),
&1<\rho<\rho_o,\\
\rho,&\rho\ge\rho_o
\end{cases}
\]

とし、

\[
\mathbf p=F(\rho)\mathbf q_0
\]

へ写す。(G=	exttt{ASROptions.circle_G}) は interface の radial slope である。

この構成は

\[
F(0)=0,\ F'(0)=1,
\quad F(1)=s_c,\ F'(1)=G,
\]

\[
F(\rho_o)=\rho_o,\ F'(\rho_o)=1
\]

を満たす。したがって円近傍へサンプルを集めつつ、Wigner–Seitz 境界で値と一次微分が恒等写像へ戻る。最短像の隣接片はそこで周期的に接続する。

実装は PyTorch autograd で (x_u,x_v,y_u,y_v) を点ごとに求め、

\[
J=\frac{\partial(x,y)}{\partial(u,v)}
=\begin{bmatrix}x_u&x_v\\y_u&y_v\end{bmatrix},
\qquad h_J=\det J
\]

を作る。非有限値または (h_J\le0) を検出した場合は層追加を拒否する。

## 5. transformed material tensor

物理媒質が等方 (epsilon,\mu) でも、計算座標では異方的な tensor density になる。共変 Maxwell 形式では

\[
\epsilon'=h_JJ^{-1}\epsilon J^{-T},\qquad
\mu'=h_JJ^{-1}\mu J^{-T}.
\]

横成分は

\[
\epsilon'_{11}=\epsilon\frac{x_v^2+y_v^2}{h_J},
\]

\[
\epsilon'_{12}=\epsilon'_{21}
=-\epsilon\frac{x_ux_v+y_uy_v}{h_J},
\]

\[
\epsilon'_{22}=\epsilon\frac{x_u^2+y_u^2}{h_J},
\qquad
\epsilon'_{33}=\epsilon h_J,
\]

であり、(mu') も同形である。これは matched/adaptive coordinates の中心式である。座標変換後の誘電率が dielectric function と metric の積になることは、matched-coordinate FMM の後続応用でも明示されている。[Optics Express 26, 13746 (2018)](https://doi.org/10.1364/OE.26.013746)

## 6. 一般 2D modal conversion (T)

計算座標の共変横場係数から Cartesian 横場係数へ変換する。連続式は、物理セル面積 (A_0=L^2\sin\zeta) として

\[
T_{\alpha\beta}(\mathbf G,\mathbf G')
=\frac{1}{A_0}\int_{\rm cell}
\left[h_JJ^{-T}\right]_{\alpha\beta}
e^{-i(\mathbf k_0+\mathbf G)\cdot\mathbf r(u,v)}
e^{i(\boldsymbol\kappa_0+\mathbf G')\cdot(u,v)},du,dv.
\]

ここで (oldsymbol\kappa_0=(k_{1,0},k_{2,0})) は computational Bloch phase、(mathbf k_0=(k_{x0},k_{y0})) は Cartesian Bloch phase である。実装の重みは

\[
\frac{h_JJ^{-T}}{\sin\zeta}
=\frac{1}{\sin\zeta}
\begin{bmatrix}
y_v&-y_u\\-x_v&x_u
\end{bmatrix}.
\]

(z) 成分用には

\[
T_z\sim\frac{h_J}{\sin\zeta}
\]

を使う。FFT により、各 Cartesian 出力次数に対する全 computational 入力次数を一度に得る。未変形三角格子 (J=\begin{bmatrix}1&1/2\\0&\sqrt3/2\end{bmatrix}) では (T=J^{-T}) へ正しく還元される。

異なる座標系の層を modal conversion で接続する考え方は Wang らの 2D ASR 多層 RCWA に対応する。[Wang et al., Optics Express 30, 21295 (2022)](https://pubmed.ncbi.nlm.nih.gov/36224852/)

## 7. (P,Q) 行列

星印を付けたものを transformed tensor の Fourier 行列とし、(K_1,K_2) を共変微分行列とする。

\[
P_{11}=\mu'_{21}+K_1(\epsilon'_{33})^{-1}K_2,
\]

\[
P_{12}=\mu'_{22}-K_1(\epsilon'_{33})^{-1}K_1,
\]

\[
P_{21}=K_2(\epsilon'_{33})^{-1}K_2-\mu'_{11},
\]

\[
P_{22}=-\mu'_{12}-K_2(\epsilon'_{33})^{-1}K_1,
\]

\[
Q_{11}=-\epsilon'_{21}-K_1(\mu'_{33})^{-1}K_2,
\]

\[
Q_{12}=K_1(\mu'_{33})^{-1}K_1-\epsilon'_{22},
\]

\[
Q_{21}=\epsilon'_{11}-K_2(\mu'_{33})^{-1}K_2,
\]

\[
Q_{22}=\epsilon'_{12}+K_2(\mu'_{33})^{-1}K_1.
\]

固有値問題は

\[
PQW=W\Lambda^2,
\qquad V=P^{-1}W\Lambda.
\]

## 8. D6-closed Fourier star

rectangular truncation (|m|,|n|\le M) は三角格子の D6 回転・鏡映で閉じない。対称性短縮では

\[
\mathcal S_M={(m,n):\max(|m|,|n|,|m-n|)\le M\}
\]

を使う。その scalar dimension は

\[
N_\star=3M(M+1)+1
\]

である。

たとえば 60° 回転と local x 軸鏡映は reciprocal indices 上で

\[
C_6:(m,n)\mapsto(n,n-m),
\]

\[
\sigma_x:(m,n)\mapsto(m,m-n)
\]

となり、(mathcal S_M) は両方で閉じる。対称性を使う FMM では「打切り reciprocal lattice 自体も対称性を持つ」ことが必要であり、C2 の先行研究でもこれが明示されている。[Bai and Li, JOSA A 22, 654 (2005)](https://opg.optica.org/abstract.cfm?uri=josaa-22-4-654)

重要な実装上の修正は、rectangular box で ((\epsilon'_{33})^{-1}) を作ってから star に切り出さないことである。射影 (S_\star) を用いて

\[
E_{33,\star}=S_\star^\dagger E_{33}S_\star,
\qquad
E_{33,\star}^{-1}=(E_{33,\star})^{-1}
\]

の順に計算する。一般に

\[
S_\star^\dagger E^{-1}S_\star
\ne(S_\star^\dagger ES_\star)^{-1}
\]

であり、前者は star 外の rectangular corner modes を逆行列経由で混入させ、D6 対称性を壊す。検証時、この誤った順序は mirror residual を数％にしたが、star 上の直接構成では (10^{-15}) 以下になった。

`factorization_rules=True` の三角偏光短縮では、横 2×2 tensor の点ごとの逆を Fourier 化し、star 上の block convolution を逆に戻す coordinate-covariant block inverse rule を使う。

\[
\mathcal E_{t,\star}^{\rm eff}
=\left[S_{2\star}^\dagger[\![\epsilon_t'^{-1}]\!]S_{2\star}\right]^{-1}.
\]

Li の factorization rule は不連続 Fourier 積の基本根拠であり、NVM は interface normal/tangent を使って crossed grating の収束を改善する別経路である。[Li (1996)](https://doi.org/10.1364/JOSAA.13.001870) [Schuster et al., JOSA A 24, 2880 (2007)](https://opg.optica.org/josaa/abstract.cfm?uri=josaa-24-9-2880) [Götz et al., Optics Express 16, 17295 (2008)](https://doi.org/10.1364/OE.16.017295)

## 9. x/y 単一偏光セクター

### 9.1 「D6 の完全分解」と x/y の関係

正規入射・円柱・三角格子の点群は (C_{6v}\simeq D_6) である。Cartesian ((x,y)) は (C_{6v}) の 2 次元既約表現 (E_1) をなす。[C6v character table](https://www.quanty.org/physics_chemistry/point_groups/c6v)

したがって、D6 全体に関する「x だけ」「y だけ」は別々の 1 次元既約表現ではない。60° 回転が x と y を混合するからである。D6 の完全分解とは、全 12 演算の指標射影子を使って (A_1,A_2,B_1,B_2,E_1,E_2) の全 isotypic component へ分けることである。

本実装が単一偏光計算に使うのは、入射偏光を混ぜない local x 軸鏡映だけからなる部分群

\[
C_s=\{e,\sigma_x\}
\]

である。x 入射は mirror-even、y 入射は mirror-odd なので、必要な一方だけを厳密に選べる。これは「D6-closed star 上の (C_s) セクター分解」であり、「D6 の x/y 1D 完全分解」とは呼ばない。

### 9.2 鏡映作用

セル中心を通る local x 軸鏡映は primitive displacement に

\[
A=\begin{bmatrix}1&1\\0&-1\end{bmatrix}
\]

として作用する。中心移動による Fourier 位相を含め、scalar coefficient の作用は

\[
R_s:\ |m,n\rangle\mapsto(-1)^m|m,m-n\rangle.
\]

計算座標の共変電場、磁場には

\[
R_E^{uv}=A^{-T}\otimes R_s=A^T\otimes R_s,
\]

\[
R_H^{uv}=-A^{-T}\otimes R_s
\]

が作用する。磁場の負号は鏡映に対する axial-vector parity である。Cartesian 側は

\[
R_E^{xy}=\operatorname{diag}(1,-1)\otimes R_s,
\]

\[
R_H^{xy}=-\operatorname{diag}(1,-1)\otimes R_s.
\]

偏光 (chi=+1)（x）または (-1)（y）の部分空間は

\[
\Pi_{E,\chi}=\frac12(I+\chi R_E),\qquad
\Pi_{H,\chi}=\frac12(I+\chi R_H)
\]

の range である。実装は SVD でこれらの正規直交基底 (B_E,B_H) を作る。

### 9.3 短縮固有値問題

\[
P_\chi=B_E^\dagger PB_H,\qquad
Q_\chi=B_H^\dagger QB_E
\]

とし、

\[
P_\chi Q_\chi W_\chi=W_\chi\Lambda_\chi^2
\]

だけを解く。各層で

\[
r_P=\frac{\|PB_H-B_EP_\chi\|_F}{\|PB_H\|_F},
\qquad
r_Q=\frac{\|QB_E-B_HQ_\chi\|_F}{\|QB_E\|_F}
\]

を計算し、許容値を超えれば短縮を拒否する。(T) 後も同様に Cartesian sector residual を検査する。

次数 (M=3) では

\[
N_{\rm rect}=49,\quad 2N_{\rm rect}=98,
\quad N_\star=37.
\]

x または y の固有値問題は (98\times98) から (37\times37) へ縮む。次元比は 0.3776、dense eigensolve の漸近 cubic work 比は

\[
(37/98)^3\approx0.0538
\]

であり、固有値分解部分だけなら理論上約 18.6 倍の削減余地がある。実際の総時間には FFT、tensor factorization、(T) 構築、S cascade も含まれるため、全体が必ず 18.6 倍になるという意味ではない。

## 10. S 行列短縮

偏光短縮時は full (2N\times2N) 内部モードをダミーで埋めない。外部半空間の (E\to H) 行列も

\[
V_\chi=B_H^\dagger VB_E
\]

へ射影し、各層の (W_\chi,V_\chi,\Lambda_\chi) だけで Redheffer または Li algorithm 2a を実行する。最後に必要な外部 Cartesian 表現へ

\[
S_{\rm full,\chi}=B_E S_\chi B_E^\dagger
\]

として埋め戻す。

`smatrix_size='half'` は公開結果として (T_f,R_f)、`'quarter'` は (R_f) だけを保持する。
内部・外部場を要求した場合は、非公開の両方向full Sと各層の
`[c_l^+(0),c_l^-(d_l)]` couplingを短縮基底上で保存する。従って単一偏光短縮でも
full／half／quarterのいずれからも6成分場を再構成できる。

## 11. 使用例

```python
import torch
from rcwa_solver_auto import (
    ASROptions, AutoRCWA, Circle, GroupTheoryOptions,
    Lattice, LayerSpec, Material, OutputSpec,
)

sim = AutoRCWA(
    freq=1 / 1.55,
    order=[3, 3],
    lattice=Lattice.triangular(1.0),
    cascade="algo2a",                 # または "redheffer"
    outputs=OutputSpec(
        smatrix_size="quarter",       # "full" / "half" も可
        fields="all",                 # external/internal/none も可
    ),
    asr=ASROptions(
        circle_G=0.08,
        grid=(192, 192),               # 三角 D6 map は同数サンプル
        factorization_rules=True,
    ),
    group_theory=GroupTheoryOptions(
        enabled=True,
        polarization="x",             # "y" も可
    ),
    dtype=torch.complex128,
    device="cpu",
)

sim.add_input_layer(eps=1.0)
sim.add_output_layer(eps=1.0)
sim.set_incident_angle(0.0, 0.0)
sim.add_structured_layer(
    LayerSpec(
        thickness=0.18,
        geometry=Circle(radius=0.24),
        background=Material(eps=1.0),
        inclusion=Material(eps=4.0),
        method="matched-asr",
    )
)
sim.solve_global_smatrix()

print(sim.S[0])  # forward transmission
print(sim.S[1])  # forward reflection
print(sim.group_theory_diagnostics[-1])
```

## 12. 適用条件

三角格子 matched-ASR:

- `Lattice.triangular(L)`、すなわち 60°・等長 primitive vectors。
- primitive cell 中心の円が一つ。
- (0<R<L/2)。接触極限 (R=L/2) は outer Hermite 区間が消えるため対象外。
- `grid=(n,n)`。

x/y 短縮:

- 正規入射。
- `GroupTheoryOptions(enabled=True, polarization='x'|'y')`。
- `smatrix_size='full'`、`'half'`、`'quarter'`。
- `fields='none'`、`'external'`、`'internal'`、`'all'`。
- partial公開Sで場を要求すると、場専用full Sと前後方向couplingを追加保存する。
- x/y は `a1` が +x を向く solver local frame。`basis_rotation_deg` を使う場合、実験室座標との回転を利用者側で考慮する。

自動選択 `method='auto'` は円に対して従来どおり NVM を優先する。新しい三角 matched-ASR を指定するには `LayerSpec(method='matched-asr')` を使う。これは、NVM が円境界の既存・検証済み経路であり、新写像を暗黙に選ばないためである。

## 13. 数値検証結果

三角NVM偏光短縮を追加する前に実行した
`validate_triangular_matched_asr.py --integration --order 3 --grid 192` のbaseline結果は
38/38 PASSである。factorizationのon/offと、matched-ASR短縮quarter/halfの一致もこの
38項目に含む。現在の同スクリプトには三角NVM x/y sectorと独立full-star比較も追加した。

| 検証 | 結果 |
|---|---:|
| 六角 interface → 物理円 | 最大誤差 (1.11\times10^{-16}) |
| D6 60°回転 equivariance | (2.22\times10^{-16}) |
| D6 x鏡映 equivariance | (2.22\times10^{-16}) |
| 独立有限差分 Jacobian | 最小 (det J=7.00\times10^{-2}>0) |
| Li2a / Redheffer（full 4 blocks） | 最大 (7.56\times10^{-15}) |
| full / half / quarter | 最大 (7.60\times10^{-15}) |
| 損失なし (R+T), x/y | (1)（誤差 (\le 6.7\times10^{-16})） |
| reduced (P,Q) invariance | (\le 6.2\times10^{-16}) |
| reduced (T) sector invariance | (\le 7.3\times10^{-16}) |
| 直交 matched-ASR reduced / full source response | (\le1.7\times10^{-14}) |
| x+y sector / 未分解 full-star | (T:3.54\times10^{-5}, R:6.62\times10^{-5}) |

最後の full-star 比較だけ誤差が大きいのは、D6 の厳密縮退モードを未分解の非 Hermitian eigensolver が任意に混合し、固有ベクトル行列が悪条件になるためである。order 2 では同じ比較が (3.4\times10^{-15}) 以下、order 3 では sector 分解側の (P,Q,T) invariance と Li2a/Redheffer parity は引き続き (10^{-15}) 級である。このため、(2\times10^{-4}) を「未分解縮退基底との比較」専用の許容値とし、物理的・代数的検査は別の厳しい許容値で判定している。

matched-ASR と独立な Cartesian NVM の x 入射 source-response 差は

| order | 最大差 |
|---:|---:|
| 1 | 0.02824 |
| 2 | 0.02643 |
| 3 | 0.01584 |

となり、低次で完全一致はしないが次数増加で差が低下した。これは同一問題に対する異なる Fourier factorization/truncation の収束比較であり、order 3 の差を最終精度とみなしてはならない。

同じ変更前baselineでは、直交円matched-ASRが19/19、全機能回帰が51/51 PASSした。

## 14. 何を「達成」とし、何を主張しないか

達成したもの:

- 三角格子の周期的・orientation-preserving な円 matched-coordinate map。
- transformed tensor、(K_1,K_2)、一般 (T) を通した通常 S 行列との接続。
- D6-closed star 上での厳密な x/y mirror-sector eigensolve。
- sector 内の Redheffer/Li2a および half/quarter 出力。
- sector／完全D6の内部・外部6成分場と前後方向modal coupling。
- NVM、標準 S、直交 matched-ASR との数値照合。

主張しないもの:

- 本写像が Weiss 論文に掲載された「三角格子専用の同一式」であること。本写像は同論文の共変 matched-coordinate 原理に基づく新しい D6 構成である。
- x と y が D6 全体の別々の 1D 既約表現であること。両者は (E_1) をなす。単一偏光短縮は従来の (C_s) sectorに加え、完全D6の (E_1) matrix-unit x/y rowでも行える。
- exact touching (R=L/2) の支持。
- 総実行時間が常に固有値 cubic 比どおり短縮されること。

## 15. 半径・層厚の逆伝播

三角格子写像を \(\mathbf F(u,v;R)\) とすると、metric を通る半径勾配には
\(\partial_R\partial_u\mathbf F\) と
\(\partial_R\partial_v\mathbf F\) が必要である。修正版は写像 Jacobian を生成する
`torch.autograd.grad` に trainable radius の場合 `create_graph=True` を指定し、
写像値・Jacobian・行列式の `.detach()` を除去した。これにより full 問題だけでなく、
D6-closed starから作るCs sectorと、完全D6 (E_1) x/y rowにも同じ勾配が通る。

層厚は \(\exp(i\omega k_zd)\) の Tensor 経路を維持する。order 1、64×64 sampling の
中心差分試験では、三角 matched-ASR full の scaled error は radius 5.08e-8、
thickness 9.16e-10、y sector は radius 9.42e-8、thickness 8.30e-11 であった。
詳細は `design_gradient_report_ja.md` と `validate_design_gradients.py` を参照すること。

## 16. 参考文献

- T. Weiss et al., “Matched coordinates and adaptive spatial resolution in the Fourier modal method,” *Optics Express* 17, 8051–8061 (2009). [DOI](https://doi.org/10.1364/OE.17.008051)
- S. Essig and K. Busch, “Generation of adaptive coordinates and their use in the Fourier Modal Method,” *Optics Express* 18, 23258–23274 (2010). [DOI](https://doi.org/10.1364/OE.18.023258)
- J. Küchenmeister, “Generalization and modularization of two-dimensional adaptive coordinate transformations for the Fourier modal method,” *Optics Express* 22, 9404–9412 (2014). [DOI](https://doi.org/10.1364/OE.22.009404)
- L. Wang et al., “2D rigorous coupled wave analysis with adaptive spatial resolution for a multilayer periodic structure,” *Optics Express* 30, 21295–21308 (2022). [DOI](https://doi.org/10.1364/OE.459110)
- L. Li, “Use of Fourier series in the analysis of discontinuous periodic structures,” *JOSA A* 13, 1870–1876 (1996). [DOI](https://doi.org/10.1364/JOSAA.13.001870)
- L. Li, “Formulation and comparison of two recursive matrix algorithms for modeling layered diffraction gratings,” *JOSA A* 13, 1024–1035 (1996). [DOI](https://doi.org/10.1364/JOSAA.13.001024)
- B. Bai and L. Li, “Group-theoretic approach to the enhancement of the Fourier modal method for crossed gratings: C2 symmetry case,” *JOSA A* 22, 654–661 (2005). [DOI](https://doi.org/10.1364/JOSAA.22.000654)
- T. Schuster et al., “Normal vector method for convergence improvement using the RCWA for crossed gratings,” *JOSA A* 24, 2880–2890 (2007). [DOI](https://doi.org/10.1364/JOSAA.24.002880)
- P. Götz et al., “Normal vector method for the RCWA with automated vector field generation,” *Optics Express* 16, 17295–17301 (2008). [DOI](https://doi.org/10.1364/OE.16.017295)
