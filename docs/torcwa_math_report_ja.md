# torcwa 0.1.4.2 数式・実装対応レポート

## 1. 対象と結論

本レポートは PyPI 版 `torcwa==0.1.4.2` の `torcwa/rcwa.py`、`geometry.py`、`torch_eig.py` を実際に読み、各処理を数式へ対応付けたものである。torcwa は PyTorch 上の 2D 周期 RCWA/Fourier modal method (FMM) であり、GPU と自動微分を主目的にしている。公開リポジトリも Lorentz–Heaviside 単位、(c=1)、時間因子 (e^{-i\omega t}) を明記している。[torcwa GitHub](https://github.com/kch3782/torcwa)

標準実装の計算経路は次である。

1. 周期面内の物性を FFT して畳み込み行列を作る。
2. Maxwell 方程式から横電磁場の (P,Q) 行列を作る。
3. (PQ) の固有値問題を解き、各層の固有モードと伝搬位相を得る。
4. 境界条件から単層 S 行列を作る。
5. Redheffer star product で多層を安定に接続する。

この構造は正統的な RCWA である。一方、0.1.4.2 本体は直交格子の rectangular Fourier truncation、通常の Laurent 積、全 (2N\times2N) 固有値問題、4 ブロック S 行列を前提とし、ASR、matched coordinates、NVM、斜交格子、対称セクター短縮は持たない。

## 2. 単位、Fourier 次数、Bloch 波数

`rcwa.__init__` は

\[
k_0=\omega=2\pi f,\qquad
G_x^{(n)}=\frac{1}{L_x f}=\frac{2\pi}{k_0L_x},\quad
G_y^{(n)}=\frac{1}{L_y f}
\]

を使う。上付き ((n)) は (k_0) で正規化された量を表す。次数集合は

\[
m=-M_x,\ldots,M_x,\qquad n=-M_y,\ldots,M_y,
\]

\[
N=(2M_x+1)(2M_y+1)
\]

である。`_kvectors` が入射側または出射側屈折率を用いて

\[
\bar k_{x,0}=\Re\sqrt{\epsilon_a\mu_a}\sin\theta\cos\phi,
\qquad
\bar k_{y,0}=\Re\sqrt{\epsilon_a\mu_a}\sin\theta\sin\phi
\]

を作り(sqrt{\epsilon_a\mu_a}は、相対屈折率)、各回折次数を

\[
\bar k_x(m)=\bar k_{x,0}+mG_x^{(n)},\qquad
\bar k_y(n)=\bar k_{y,0}+nG_y^{(n)}
\]

とする。これを対角行列 (K_x,K_y\in\mathbb C^{N\times N}) に格納する。

均質媒質の縦波数は

\[
\bar k_z=\sqrt{\epsilon\mu-\bar k_x^2-\bar k_y^2}
\]

であり、torcwa は虚部が負にならない枝を選ぶ。これは前向き伝搬波または (+z) 方向に減衰する evanescent 波を選ぶ操作である。

## 3. 物性の Fourier 畳み込み行列

`_material_conv` はサンプル (epsilon_{rs}) に `fft2` を適用し、離散 Fourier 係数

\[
\hat\epsilon_{pq}=\frac{1}{N_xN_y}
\sum_{r,s}\epsilon_{rs}
e^{-i2\pi(pr/N_x+qs/N_y)}
\]

を得る。畳み込み行列は

\[
[\![\epsilon]\!]_{(m,n),(m',n')}
=\hat\epsilon_{m-m',n-n'}
\]

である。Python の負添字が FFT 配列の周期添字として働くため、コードでは差次数をそのまま配列添字に使っている。

この行列化により、実空間の積 (epsilon E) は有限 Fourier 空間で

\[
\mathcal F[\epsilon E]\approx [\![\epsilon]\!]\,\mathbf E
\]

となる。ただし、不連続関数同士の積では単純 Laurent rule が常に正しいとは限らない。Li は同時不連続の向きに応じて direct rule と inverse rule を使い分ける必要を体系化した。[Li, JOSA A 13, 1870 (1996), DOI:10.1364/JOSAA.13.001870](https://doi.org/10.1364/JOSAA.13.001870)

## 4. Maxwell 方程式から (P,Q) へ

横電場・磁場係数を

\[
\mathbf e=\begin{bmatrix}\mathbf E_x\\\mathbf E_y\end{bmatrix},
\qquad
\mathbf h=\begin{bmatrix}\mathbf H_x\\\mathbf H_y\end{bmatrix}
\]

とする。各層内では

\[
\frac{1}{ik_0}\frac{d\mathbf e}{dz}=P\mathbf h,
\qquad
\frac{1}{ik_0}\frac{d\mathbf h}{dz}=Q\mathbf e
\]

の形に整理できる。`_eigen_decomposition` が作る標準等方媒質の行列は、(E=[\![\epsilon]\!])、(M=[\![\mu]\!]) として

\[
P=
\begin{bmatrix}
K_xE^{-1}K_y & M-K_xE^{-1}K_x\\
K_yE^{-1}K_y-M & -K_yE^{-1}K_x
\end{bmatrix},
\]

\[
Q=
\begin{bmatrix}
-K_xM^{-1}K_y & K_xM^{-1}K_x-E\\
E-K_yM^{-1}K_y & K_yM^{-1}K_x
\end{bmatrix}.
\]

これは (z) 成分

\[
E_z=E^{-1}(K_yH_x-K_xH_y),\qquad
H_z=M^{-1}(K_xE_y-K_yE_x)
\]

を Maxwell 方程式から消去した結果である。

均質層では `\_eigen_decomposition_homogenous` が (E=\epsilon I)、(M=\mu I) を使い、固有ベクトルを (W=I) として解析的な (ar k_z) を格納する。

## 5. 固有値問題とモード

二つの一次式を合成すると

\[
\frac{1}{(ik_0)^2}\frac{d^2\mathbf e}{dz^2}=PQ\mathbf e
\]

であり、層内モードは

\[
PQW=W\Lambda^2,
\qquad
\Lambda=\operatorname{diag}(\bar k_{z,\alpha})
\]

を満たす。torcwa は `torch.linalg.eig(P @ Q)` またはカスタム `Eig.apply` を用い、固有値の平方根から (ar k_z) を作る。磁場固有ベクトルは

\[
V=P^{-1}W\Lambda
\]

である。`avoid_Pinv_instability=True` の場合は (P^{-1}) の残差を調べ、悪条件なら等価な

\[
V=QW\Lambda^{-1}
\]

へ切り替える。

注意点として、一般の (PQ) は非 Hermitian であり、固有ベクトルは Euclidean 直交とは限らない。また、縮退固有値では固有ベクトルの選び方が一意でない。

## 6. 自動微分可能な固有値分解

`torch_eig.Eig` の forward は通常の複素固有値分解である。backward では固有値差

\[
s_{ij}=\lambda_j-\lambda_i
\]

に対し

\[
F_{ij}=\frac{s_{ij}^{*}}{|s_{ij}|^2+\delta},\qquad
F_{ii}=0,qquad \delta=10^{-10}
\]

という Lorentzian broadening を用いる。固有値が接近したときの (1/(\lambda_i-\lambda_j)) 発散を有限化する設計である。これは forward 解を変えず、backward の勾配だけを正則化する。

## 7. 均質半空間の (E\to H) 行列

`_kvectors` は各回折次数について自由空間基準 (V_f)、入力 (V_i)、出力 (V_o) を作る。等方媒質で

\[
V(\mu,k_z)=\frac{1}{\mu}
\begin{bmatrix}
-K_yK_xK_z^{-1} & -K_z-K_y^2K_z^{-1}\\
K_z+K_x^2K_z^{-1} & K_xK_yK_z^{-1}
\end{bmatrix}.
\]

したがって外部波の横磁場係数は (mathbf h=V\mathbf e) で表せる。入力界面と出力界面の S 行列は (V_f+V_{i/o}) の逆行列と差 (V_f-V_{i/o}) から構成される。

## 8. 単層 S 行列

厚さ (d) の層の前向き伝搬因子は

\[
X=\exp(ik_0\Lambda d)
\]

（コードでは対角行列）である。基準媒質 (V_f) に対して

\[
A=W+V_f^{-1}V,\qquad B=W-V_f^{-1}V
\]

と書くと、`_solve_layer_smatrix` が解く境界連立行列は

\[
C=
\begin{bmatrix}
A & BX\\
BX & A
\end{bmatrix}.
\]

前方入射・後方入射に対して

\[
C_f=C^{-1}\begin{bmatrix}2I\\0\end{bmatrix},
\qquad
C_b=C^{-1}\begin{bmatrix}0\\2I\end{bmatrix}
\]

を解き、(W,X,C_f,C_b) から (S_{11},S_{21},S_{12},S_{22}) を作る。torcwa のブロック順は

\[
[S_{11},S_{21},S_{12},S_{22}]
=[T_f,R_f,R_b,T_b]
\]

である。

## 9. Redheffer 接続

隣接する二つの散乱系 (S^{(m)},S^{(n)}) に対し、`_RS_prod` は

\[
D_1=(I-S_{12}^{(m)}S_{21}^{(n)})^{-1},\qquad
D_2=(I-S_{21}^{(n)}S_{12}^{(m)})^{-1}
\]

を用い、例えば

\[
S_{11}=S_{11}^{(n)}D_1S_{11}^{(m)},
\]

\[
S_{21}=S_{21}^{(m)}+S_{22}^{(m)}D_2S_{21}^{(n)}S_{11}^{(m)}
\]

などを計算する。指数的に増大する後退波を transfer matrix で直接長距離伝搬させないため、evanescent モードが多い厚い多層でも安定である。S/R 行列再帰の安定性と「増大指数関数を再帰に持ち込まない」という条件は Li の整理と一致する。[Li, JOSA A 13, 1024 (1996)](https://doi.org/10.1364/JOSAA.13.001024)

## 10. S パラメータと励振

`source_fourier` は指定回折次数に ((E_x,E_y)) を配置する。`notation='ps'` では入射面に基づく p/s 基底から x/y 基底へ回転する。

`S_parameters` は指定した入力次数・出力次数・偏光成分を S 行列から抽出する。`power_norm=True` では、おおまかに

\[
S_{\rm power}=S_{\rm field}
\sqrt{\frac{k_{z,\rm out}}{k_{z,\rm in}}}
\sqrt{\frac{1+(k_{\parallel,\rm out}/k_{z,\rm out})^2}
{1+(k_{\parallel,\rm in}/k_{z,\rm in})^2}}
\]

という規格化を行い、伝搬次数について (|S|^2) を電力比に対応させる。evanescent 判定された次数はゼロへ落とす。

## 11. 場再構成

`field_xz`、`field_yz`、`field_xy` は次を行う。

1. global S 行列と保存済み coupling (C_f,C_b) から各層の前後向きモード係数を得る。
2. (W,V,\exp(\pm ik_0\Lambda z)) で横電磁場を再構成する。
3. 消去式から (E_z,H_z) を戻す。
4. Fourier 位相 (e^{i(k_xx+k_yy)}) を掛け、指定面で和を取る。

このため標準 torcwa は内部場を得るために、全 S ブロックだけでなく各層の coupling 行列も保持する。
今回の拡張も同じ物理量を用いるが、`half`／`quarter` は公開Sの指定として扱う。場を要求した
場合だけ両方向full field Sとcouplingを非公開で構成するため、partial公開S、Li-2a、
x/y sector、native-star完全D6でも同じ `field_xy/xz/yz` APIを使用できる。D6-starの
\(E_z,H_z\) は矩形boxではなくscalar star内でlongitudinal inverseを解いてから埋め戻す。

## 12. geometry.py

`geometry` はセル中心サンプル

\[
x_i=\frac{L_x}{N_x}(i+1/2),\qquad
y_j=\frac{L_y}{N_y}(j+1/2)
\]

を作る。円ではレベル集合

\[
\ell=1-\sqrt{((x-C_x)/R)^2+((y-C_y)/R)^2}
\]

を sigmoid

\[
g(x,y)=\sigma(\beta\ell)
\]

へ通す。矩形、楕円、菱形、superellipse も同様の滑らかなレベル集合であり、union/intersection/difference は `maximum`、`minimum`、`minimum(A,1-B)` である。この平滑化により形状パラメータへの勾配を通せるが、有限 (eta) では境界が厳密な不連続ではない。

## 13. 標準 torcwa の適用範囲と今回拡張が必要な理由

標準 0.1.4.2 の実コードから確認できる制約は次である。

- 逆格子は (G_x,G_y) の直交 rectangular box。
- 物性積はスカラー畳み込み行列の通常則で、Li の方向別 factorization や NVM はない。
- 各非均質層で常に (2N\times2N) の (PQ) 固有値問題を解く。
- S 行列は常に 4 ブロックを作り、full/half/quarter の選択はない。
- 多層接続は Redheffer のみで、Li algorithm 2a を選ぶ API はない。
- 点群対称性や入射偏光に基づく部分空間短縮はない。
- ASR/matched coordinates の層と通常 Cartesian 層を結ぶ一般 (T) はない。

今回の拡張は、この基礎を置き換えるのではなく、同じ (P,Q,W,V,S) の流れを保ったまま、物性行列の作り方、座標変換、逆格子、固有空間、cascade 出力を一般化している。

## 14. 設計変数と自動微分

torcwa の層厚依存性は主として伝搬位相

\[
X_n(d)=\exp(i\omega k_{z,n}d)
\]

に入り、`thickness` が Tensor のままなら S 行列から設計変数まで逆伝播できる。
今回のラッパーでは standard、ASR、NVM の全層で、検証用 scalar と計算用 Tensor を
分離した。半径については torcwa の硬い geometry raster をそのまま使うと occupancy が
区分的に一定になるため、解析 Fourier–Bessel 係数を使う NVM、または境界を座標面へ
固定する matched-ASR を使用する。導出、実装上の graph-cut、有限差分照合は
`design_gradient_report_ja.md` にまとめた。

## 15. 参考文献

- C. Kim and B. Lee, “TORCWA: GPU-accelerated Fourier modal method and gradient-based optimization for metasurface design,” *Computer Physics Communications* 282, 108552 (2023). [DOI](https://doi.org/10.1016/j.cpc.2022.108552)
- L. Li, “Use of Fourier series in the analysis of discontinuous periodic structures,” *JOSA A* 13, 1870–1876 (1996). [DOI](https://doi.org/10.1364/JOSAA.13.001870)
- L. Li, “Formulation and comparison of two recursive matrix algorithms for modeling layered diffraction gratings,” *JOSA A* 13, 1024–1035 (1996). [Optica](https://opg.optica.org/abstract.cfm?uri=josaa-13-5-1024)
