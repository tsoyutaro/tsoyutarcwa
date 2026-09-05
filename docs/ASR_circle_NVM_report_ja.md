# 円形NVMに対するASR導入：実装結果と数式根拠

## 結論

円形に対するASRは実装できた。ただし、実装した厳密な経路は「Cartesian NVM行列を作った後でASR行列を掛ける」方式ではなく、円周に一致する座標系でNVMと同じ法線・接線の界面条件をFourier因子分解する **matched-coordinate ASR-FR** である。APIでは誤解を避けるため `method="matched-asr"` とした。

既存NVMの射影行列とmatched-coordinate因子分解を同じ打切り空間で二重に適用する一意な式は、確認した文献にはない。単純な積は打切りToeplitz行列の非可換性により、Liの因子分解則を保つ保証がない。そのため、根拠のない「NVM+ASR後掛け」は実装していない。一方、ユーザーが求めた実用上の目的である、円形界面へのASR、写像 (T)、通常のS行列、Redheffer／Li algorithm 2a、full／half／quarterは実現した。

主な根拠は、円柱matched coordinatesとASRを実際に導出・検証した [Weiss et al., Optics Express 17, 8051–8061 (2009)](https://doi.org/10.1364/OE.17.008051)、変換座標のモードをCartesianモードへ戻す (T) を導出した [Wang et al., Optics Express 30, 21295–21308 (2022)](https://doi.org/10.1364/OE.459110) である。NVM側の界面条件との関係は [Schuster et al., JOSA A 24, 2880–2890 (2007)](https://doi.org/10.1364/JOSAA.24.002880) と [Götz et al., Optics Express 16, 17295–17301 (2008)](https://doi.org/10.1364/OE.16.017295) に基づく。

## 1. 円周に一致する座標写像

直交セルを (0\le x<L_x, 0\le y<L_y)、円の中心を ((L_x/2,L_y/2))、半径を (R) とする。matched座標の界面位置を

\[
b_x^\pm=\frac{L_x}{2}\pm\frac{R}{\sqrt2},\qquad
b_y^\pm=\frac{L_y}{2}\pm\frac{R}{\sqrt2}
\]

とする。例えば (x) 方向の2本の界面曲線は

\[
x^\pm(\tilde v)=
\begin{cases}
L_x/2\pm\sqrt{R^2-(\tilde v-L_y/2)^2},
&b_y^-\le\tilde v\le b_y^+,\\
b_x^\pm,&\text{otherwise}.
\end{cases}
\]

である。物理座標 (X(\tilde u,\tilde v)) は、

\[
X=
\begin{cases}
\dfrac{\tilde u}{b_x^-}x^-(\tilde v),&0\le\tilde u<b_x^-,\\[4pt]
\dfrac{b_x^+-\tilde u}{b_x^+-b_x^-}x^-(\tilde v)
+\dfrac{\tilde u-b_x^-}{b_x^+-b_x^-}x^+(\tilde v),
&b_x^-\le\tilde u\le b_x^+,\\[4pt]
\dfrac{L_x-\tilde u}{L_x-b_x^+}x^+(\tilde v)
+\dfrac{\tilde u-b_x^+}{L_x-b_x^+}L_x,&b_x^+<\tilde u<L_x.
\end{cases}
\]

(Y(\tilde u,\tilde v)) は (x\leftrightarrow y, \tilde u\leftrightarrow\tilde v) で得る。これはWeiss論文の式(37)–(38)であり、円周の4分割が (\tilde u=b_x^\pm) または (\tilde v=b_y^\pm) の座標面に一致する。

その前段に、各軸独立のASR写像 (\tilde u=f(u),\tilde v=g(v)) を置く。各区間 (u_l\le u<u_{l+1}) では

\[
f(u)=a_1+a_2u+\frac{a_3}{2\pi}
\sin\!\left(2\pi\frac{u-u_l}{\Delta u_l}\right),
\]

\[
f'(u)=a_2+\frac{a_3}{\Delta u_l}
\cos\!\left(2\pi\frac{u-u_l}{\Delta u_l}\right).
\]

区間幅はWang式(2)の正規化形 (\Delta u_l\propto\sqrt[3]{\Delta\tilde u_l}) とした。円形ではWeiss論文の (eta=0.97) に対応する最小傾き `circle_G=1-eta=0.03` を既定値とする。長方形用の既定 `G=0.001` を円形に流用するとJacobianが過度に小さくなるため、別パラメータにした。

## 2. Maxwell方程式と変換媒質テンソル

最終写像を

\[
\mathbf r(u,v,z)=(X(u,v),Y(u,v),z),\qquad
J=\frac{\partial(X,Y)}{\partial(u,v)}
=\begin{pmatrix}X_u&X_v\\Y_u&Y_v\end{pmatrix},
\quad h=\det J>0
\]

とする。Weiss論文の共変形式、すなわち

\[
\xi^{mnp}\partial_n E_p=i k_0\tilde\mu^{mn}H_n,qquad
\xi^{mnp}\partial_n H_p=-i k_0\tilde\varepsilon^{mn}E_n
\]

を用いると、等方媒質 (arepsilon,mu) は

\[
\tilde\varepsilon=hJ^{-1}(\varepsilon I)J^{-T},\qquad
\tilde\mu=hJ^{-1}(\mu I)J^{-T}
\]

へ変換される。横成分と縦成分は

\[
\begin{aligned}
\tilde\varepsilon^{11}&=\varepsilon\frac{X_v^2+Y_v^2}{h},&
\tilde\varepsilon^{12}=\tilde\varepsilon^{21}
&=-\varepsilon\frac{X_uX_v+Y_uY_v}{h},\\
\tilde\varepsilon^{22}&=\varepsilon\frac{X_u^2+Y_u^2}{h},&
\tilde\varepsilon^{33}&=\varepsilon h,
\end{aligned}
\]

であり、(mu) も同形である。実装では (h>0) を全サンプルで検査する。

## 3. NVMと同じ界面条件を満たす対称Fourier因子分解

NVMは、接線電場と法線変位が連続であることを使い、接線成分にLaurent則、法線成分にLiのinverse ruleを適用する。matched coordinatesでは円周が座標面なので、同じ条件を座標成分へ直接適用できる。

([f]_v) を (v) 方向の有限Toeplitz行列、(Delta_arepsilon=\varepsilon^{11}\varepsilon^{22}-\varepsilon^{12}\varepsilon^{21}) とする。まず (v) 方向に

\[
\begin{aligned}
A_{22}&=[1/\varepsilon^{22}]_v^{-1},\\
A_{21}&=[1/\varepsilon^{22}]_v^{-1}
[\varepsilon^{21}/\varepsilon^{22}]_v,\\
A_{12}&=[\varepsilon^{12}/\varepsilon^{22}]_v A_{22},\\
A_{11}&=[\Delta_\varepsilon/\varepsilon^{22}]_v
+[\varepsilon^{12}/\varepsilon^{22}]_v A_{21}.
\end{aligned}
\]

次に (u) 方向へ

\[
\varepsilon^F_{11}=[A_{11}^{-1}]_u^{-1},\qquad
\varepsilon^F_{12}=\varepsilon^F_{11}[A_{11}^{-1}A_{12}]_u.
\]

軸を交換した構成から

\[
\varepsilon^F_{22}=[B_{22}^{-1}]_v^{-1},\qquad
\varepsilon^F_{21}=\varepsilon^F_{22}[B_{22}^{-1}B_{21}]_v
\]

を作る。これはWeiss論文の式(29)–(36)を有限BTTB行列として実装したもので、(u\to v) だけを優先して生じる人工的な偏光非対称性を避ける。基礎となる積の因子分解則は [Li, JOSA A 13, 1870–1876 (1996)](https://doi.org/10.1364/JOSAA.13.001870) である。

## 4. 一般異方性P/Q行列

(K_u,K_v) を規格化横波数の対角行列とする。上の因子分解後テンソルを使い、実装した固有値問題は

\[
\frac{d\mathbf s}{d\tilde z}=P\mathbf u,\qquad
\frac{d\mathbf u}{d\tilde z}=Q\mathbf s
\]

で、

\[
P=\begin{pmatrix}
\mu_{21}^F+K_u\varepsilon_{33}^{-1}K_v &
\mu_{22}^F-K_u\varepsilon_{33}^{-1}K_u\\
K_v\varepsilon_{33}^{-1}K_v-\mu_{11}^F &
-\mu_{12}^F-K_v\varepsilon_{33}^{-1}K_u
\end{pmatrix},
\]

\[
Q=\begin{pmatrix}
-\varepsilon_{21}^F-K_u\mu_{33}^{-1}K_v &
K_u\mu_{33}^{-1}K_u-\varepsilon_{22}^F\\
\varepsilon_{11}^F-K_v\mu_{33}^{-1}K_v &
\varepsilon_{12}^F+K_v\mu_{33}^{-1}K_u
\end{pmatrix}.
\]

(PQW=W\Lambda) を解き、既存のtorcwa互換モード処理へ渡す。

## 5. 非分離写像の (T) と通常S行列への接続

一般写像では (T) はWang論文の対角ブロックではなく全4ブロックになる。共変場 (mathbf E_{uv}) とCartesian場の関係は

\[
\mathbf E_{xy}=J^{-T}\mathbf E_{uv},\qquad
dxdy=h,dudv.
\]

したがって (A=hJ^{-T}) とおけば、Cartesian次数 ((m,n))、変換座標次数 ((p,q)) 間の写像は

\[
T^{\alpha i}_{mn,pq}=
\frac{1}{L_xL_y}\int_0^{L_x}\!\int_0^{L_y}
A_{\alpha i}(u,v)
e^{i[\mathbf k_{pq}\cdot(u,v)-\mathbf k_{mn}\cdot(X,Y)]},du,dv.
\]

ここで

\[
A=\begin{pmatrix}Y_v&-Y_u\\-X_v&X_u\end{pmatrix}.
\]

縦成分は同様に

\[
(T_z)_{mn,pq}=\frac{1}{L_xL_y}\int h
e^{i[\mathbf k_{pq}\cdot(u,v)-\mathbf k_{mn}\cdot(X,Y)]},du,dv.
\]

を使う。固定したCartesian行ごとに2次元IFFTを用い、全変換次数を同時に求めるため、直接積分の (O(N^2N_g)) ではなく概ね (O(NN_g\log N_g)) で構築する。モードは

\[
W_{xy}=TW_{uv},\qquad V_{xy}=TV_{uv}
\]

として通常のCartesian境界条件へ戻る。その後は既存のRedheffer star productまたは [Liのstable S-matrix algorithm](https://doi.org/10.1364/JOSAA.13.001024) をそのまま使う。

この方法が削減するのは「同じ精度に必要なFourier次数」である。同じ次数なら固有値問題は従来と同じ (2N\times2N) であり、次元そのものが減るわけではない。`half`／`quarter` はS行列再帰の不要ブロックを省くが、層固有値問題は省略しない。

## 6. 実装API

```python
import torch
from rcwa_solver_auto import (
    ASROptions, AutoRCWA, Circle, Lattice,
    LayerSpec, Material, OutputSpec,
)

sim = AutoRCWA(
    freq=1 / 1.55,
    order=[5, 5],
    lattice=Lattice.rectangular(1.0, 1.2),
    cascade="algo2a",  # または "redheffer"
    outputs=OutputSpec(smatrix_size="half", fields="none"),
    asr=ASROptions(
        circle_G=0.03,
        grid=(192, 192),
        factorization_rules=True,
    ),
    dtype=torch.complex128,
    device="cpu",
)
sim.add_input_layer(eps=1.0)
sim.add_output_layer(eps=1.0)
sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
sim.add_structured_layer(
    LayerSpec(
        thickness=0.18,
        geometry=Circle(0.25),
        background=Material(1.0),
        inclusion=Material(4.0),
        method="matched-asr",
    )
)
sim.solve_global_smatrix()
Tf, Rf = sim.S[0], sim.S[1]
```

同心コアシェルで内外両円へ整合する場合は、構造層を次のように追加する。

```python
sim.add_layer_circle_shell_asr(
    thickness=0.18,
    core_radius=0.18,
    outer_radius=0.32,
    eps_bg=1.0,
    eps_shell=2.5,
    eps_core=4.0,
    radial_mapping="double",
    nx=192,
    ny=192,
)
```

`double` は単調性保証付き半径方向C2のquintic-Hermite写像と固定計算空間maskを用いるため、
`core_radius` と `outer_radius` の両方をTensor設計変数にできる。二重放射supportには
逐次u/v因子分解を流用せず、level-set法線 (n_i=\partial_i\rho) を使う一般化Li
因数分解を適用する。計算格子計量で法線projector (P_n) を作り、

\[
C^F=[CB^{-1}][B^{-1}]^{-1},\qquad
B=P_t+P_nC,\qquad C=gA
\]

として、連続な接線 (E) と法線 (D) を有限Fourier空間へ入れる。これはmatched空間内の
因数分解であり、Cartesian NVM射影を後掛けする混成法ではない。

零曲率quinticの物質界面には
`min(circle_G, 0.95*minimum_radial_secant)`を与える。中心と周期境界は、それぞれの
区間の最小割線勾配の0.95倍（上限1）まで勾配を戻す。各区間で両端勾配を割線勾配以下に
制限するため半径方向Jacobianは正に保たれ、同時に中心と周期境界での不必要な
`det(J) ~ circle_G^2`圧縮を避ける。

## 7. 検証結果

`validation/validate_circle_matched_asr.py --integration --order 5 --grid 192` は、単調写像、
一般化Li、S行列一致、エネルギー保存を含む全項目を検証する。

| 検証 | 最大誤差または結果 |
|---|---:|
| 円周とmatched座標線 | (1.11\times10^{-16}) |
| FFT版 (T) と直接二重積分 | (4.13\times10^{-16}) |
| FFT版 (T_z) と直接二重積分 | (4.31\times10^{-16}) |
| (T,T_z) の恒等写像極限 | (3.74\times10^{-16}) |
| 式(29)–(36)の定数テンソル極限 | (4.44\times10^{-16}) |
| 一般化Liの軸法線inverse/direct極限 | (1.11\times10^{-16}) |
| 一般化Liの60°斜交・定数異方性極限 | (8.88\times10^{-16}) |
| 一般化Li法線projectorのD6共変性 | 0 |
| Redheffer対Li-2a、4ブロック最大 | (1.69\times10^{-14}) |
| full対half/quarter最大 | (9.12\times10^{-15}) |
| 無損失 (R+T) | (0.9999999999999996) |
| (T) の条件数（次数5） | 18.64 |

同一のx偏光ゼロ次入射に対するmatched-ASRと既存円形NVMの最大振幅差は、次数3、4、5で

\[
1.746\times10^{-2},\quad 6.110\times10^{-3},\quad 2.103\times10^{-3}
\]

へ減少した。追加試験では次数7で (1.725\times10^{-3}) だった。単調ではないが両定式化が同じ解へ近づく挙動を確認した。三角NVM偏光短縮を追加する前のbaseline回帰試験は51/51項目合格した。

## 8. 現在の適用範囲（更新）

- 対応：正方・長方の直交格子、セル中心の単一円、非接触条件 (2R<\min(L_x,L_y))。
- 対応：60°・等辺の三角 Bravais 格子（2D六方最密配置）、セル中心の単一円、非接触条件 (2R<L)。
- 対応：直交・60°三角格子の同心コアシェル。`radial_mapping='outer'` は外円のみ、`'double'` は内外両円へmatchedする。
- 対応：直交格子はWeiss型円matched map、三角格子はD6-equivariantなhex-to-circle周期Hermite map。
- 対応：誘電・磁性材料（非ゼロの \(\epsilon,\mu\)）、Redheffer／Li-2a、full／half／quarter。
- 対応：正入射におけるmatched-ASR固有値問題のx/y単一偏光短縮。直交格子ではC2v、三角格子では `symmetry='auto'` でCs鏡映セクター、`symmetry='d6'` で完全D6のE1 matrix-unit source-rowを使う。
- 対応：正入射における円形NVM固有値問題のx/y単一偏光短縮。直交格子ではC2v、三角格子では誘電率・逆誘電率・法線射影を先にD6-closed starへ制限し、star内で逆則とNVMの \(P_\star,Q_\star\) を直接組み立てる。`symmetry='auto'` はCs、`symmetry='d6'` は完全D6 E1 source-rowを解く。
- 対応：`symmetry='d6', polarization='x'|'y'` による完全D6 E1 matrix-unit source-row短縮。三角NVM／matched-ASRの固有値問題とcascadeを (M(M+1)+1) 次元で実行する。
- 対応：radiusおよびthicknessのautograd。二重matchedコアシェルの内外半径、三角matched-ASR full／y sectorを含め、中心差分と照合済み。
- 対応：一般斜交NVMではCartesian x/yが同じC2 characterに属するため、両sourceが共有するC2 sectorだけを解く。xとyを別sectorにはしないが、sector内の交差偏光結合を含む厳密なsource-accessible短縮である。
- 対応：偏光短縮の内部・外部6成分電磁場。`fields='external'|'internal'|'all'` と full／half／quarterを任意に組み合わせられる。
- 制限：円が接触する真のclose-packed極限 (2R=L) ではJacobianが特異になり得るため、matched-ASRでは扱わず、小さい正のgapを設ける。
- 対応：D6-closed native star全体を `A1,A2,B1,B2,E1,E2` の6 isotypic blockへ分ける完全D6分解。三角NVM／matched-ASR、Redheffer／Li-2a、full／half／quarter S行列に対応する。元の矩形Fourier集合そのものはD6で閉じないため、そのcorner harmonicを含む集合の完全D6分解は数学的に行わない。
- 対応：完全D6経路の内部・外部6成分電磁場、前方・後方入射、Redheffer／Li-2a。partial公開Sの場合も場用full Sとmodal couplingを内部保存する。
- 対応：完全D6 E1 source-row経路の内部・外部場、Redheffer／Li-2a、full／half／quarter公開S。
- 非採用：既存Cartesian NVM射影行列とmatched-coordinate tensorを二重適用する
  `matched-nvm`。単一円はWeiss対称因数分解、二重matchedコアシェルは一般化Li
  normal-D/tangential-E因数分解によりmatched空間内で完結するため、二重補正は行わない。

したがって、直交格子と三角格子の円形matched-ASR、一般2次元 \((T,T_z)\) による通常S行列への接続、ならびに直交・三角・一般斜交円形NVMのsource-accessible x/y短縮は達成済みである。三角格子では矩形用分離写像を流用せず、斜交周期境界とD6対称性に適合する別の写像を導入している。詳細な三角写像・群論・検証結果は `triangular_matched_asr_math_report_ja.md`、`triangular_nvm_polarization_report_ja.md`、`rcwa_modular_math_guide_ja.md` を参照すること。
