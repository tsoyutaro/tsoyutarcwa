# 三角格子円形NVMのx/y偏光セクター短縮

## 1. 適用範囲

対象は、等長の基本並進ベクトルが60度をなす三角Bravais格子、primitive-cell中心の
単一円、正入射である。公開Sは `full`、`half`、`quarter`、場は `none`、`external`、
`internal`、`all` を独立に選べる。

```python
sim = AutoRCWA(
    freq=1 / 1.55,
    order=[5, 5],
    lattice=Lattice.triangular(1.0),
    outputs=OutputSpec(smatrix_size="quarter", fields="all"),
    group_theory=GroupTheoryOptions(
        enabled=True,
        strict=True,
        polarization="x",
    ),
)
```

`method="nvm"` または、円に対する既定の `method="auto"` で使用できる。
`fields="external"` と `"internal"` は指定した領域だけ、`"all"` は両方の場APIを
有効にする。公開Sがpartialでも場用の両方向full Sとmodal couplingは内部保存される。

## 2. 矩形打切りをそのまま使えない理由

torcwa型のindex box

\[
\mathcal B_M=\{(m,n):-M\le m,n\le M\}
\]

は60度回転で閉じない。例えば回転で得られるindexがboxの角から外れるため、
\(\mathcal B_M\) 上にはD6の表現行列を厳密には定義できない。

本実装はbox内部の

\[
\mathcal H_M=
\{(m,n):\max(|m|,|n|,|m-n|)\le M\}
\]

を採用する。これは六角形のreciprocal starであり、D6の全回転・鏡映で閉じる。
scalar embeddingを \(E_s:\mathcal H_M\to\mathcal B_M\)、2成分vector embeddingを
\(E_v=\operatorname{diag}(E_s,E_s)\) とする。

## 3. NVM演算子をstar内で組み立てる

円板の誘電率Fourier畳み込み行列を \([\epsilon]\)、逆誘電率の畳み込み行列を
\([\epsilon^{-1}]\)、Cartesian normal-vector projectionを \(N\) とする。まず

\[
\epsilon_\star=E_s^\dagger[\epsilon]E_s,
\qquad
\eta_\star=E_s^\dagger[\epsilon^{-1}]E_s,
\qquad
N_\star=E_v^\dagger N E_v
\]

を作る。NVMのtransverse effective tensorは

\[
\mathcal E_{t,\star}
=I_2\otimes\epsilon_\star
+\left\{I_2\otimes
(\eta_\star^{-1}-\epsilon_\star)\right\}N_\star
\]

である。longitudinal inverseも \(\epsilon_\star^{-1}\) を使う。斜交座標の波数行列
\(K_{1,\star}=E_s^\dagger K_1E_s\)、
\(K_{2,\star}=E_s^\dagger K_2E_s\) と合わせ、通常の斜交NVM式へ代入して
\(P_\star,Q_\star\) を直接作る。

この順序は必須である。一般に

\[
E_s^\dagger A^{-1}E_s
\ne
(E_s^\dagger A E_s)^{-1}
\]

なので、矩形box上で逆行列を計算してから切り出すと、除去した角harmonicを経由する
仮想結合が残り、D6/Cs不変性を壊す。

## 4. x/yセクター

物理x軸に関する鏡映を \(M\) とする。三角reciprocal indexでは

\[
(m,n)\mapsto(m,m-n)
\]

となり、cell中心を鏡映中心にする位相 \((-1)^m\) を含める。電気場と磁気場では
polar-vector／axial-vectorの違いを反映した表現 \(M_E,M_H\) を使う。

\[
M_EP_\star=P_\star M_H,
\qquad
M_HQ_\star=Q_\star M_E
\]

が成立するため、

\[
\Pi_\pm=\frac12(I\pm M)
\]

で偶・奇sectorへ分けられる。正入射のx sourceとy sourceは異なるsectorに属するので、
指定された一方の固有値問題だけを解く。得られたreduced S行列は元の
\(2N\times2N\) 配列へembeddingして返すが、未選択偏光の列は計算していない。

## 5. 計算量

矩形boxのscalar次元は \((2M+1)^2\)、D6 starのscalar次元は
\(3M(M+1)+1\) である。さらにCsの一方のsectorだけを解くため、固有値問題の次元は
概ねfull boxの半分より小さくなる。密行列固有値分解の主要計算量は次元の3乗なので、
高次数ほど短縮効果が大きい。

## 6. 一般斜交格子のx/y共通C2 sector

一般斜交格子のpoint groupは通常C2だけである。180度回転では

\[
(E_x,E_y)\mapsto(-E_x,-E_y)
\]

となり、xとyは同じcharacterを持つ。したがってC2射影演算子はx sourceとy sourceを
別々の不変部分空間へ分けない。しかし、両方のゼロ次sourceは同じscalar-even vector
sectorに属するため、その共通sectorだけを解くことはできる。このsector内ではxからy、
yからxへの交差偏光結合も保持される。

追加の鏡映を持つ特殊なrhombic格子なら鏡映固有偏光へ分けられる可能性があるが、一般に
その固有偏光はCartesian x/yではない。現在のAPIで一般斜交格子に
`polarization="x"`または`"y"`を指定すると、どちらも同じC2 source sectorを使う。
したがってx/yを別々にした場合ほどは縮まらないが、full C2両blockを解く必要はない。

## 7. 一からsolverを書けばD6完全分解できるか

可能だが、条件は「一から書くこと」ではなく「native Fourier基底をD6で閉じること」で
ある。矩形集合 \(\mathcal B_M\) を維持する限り、どの実装でも完全D6分解はできない。

本実装では最初から \(\mathcal H_M\) を採用し、D6の12群要素すべてのelectric polar-vector
表現とmagnetic axial-vector表現を作る。`symmetry="d6", polarization=None` では、A1、A2、
B1、B2、E1、E2の指標射影でnative star全modeをisotypic blockへ完全分解する。convolution
と固有値問題はnative-star index内で組み立て、interfaceとS行列cascadeも同じstar基底で
実行した後、公開結果と場のFourier係数をtorcwa互換の矩形配列へ埋め戻す。内部場再構成も
native-star modal amplitudeで実行し、縦場は `epsilon_zz`／`mu_zz` をstarへ制限した行列で
解いてからscalar embeddingで矩形配列へ戻す。したがってcorner harmonicを逆行列経由で
再混入させない。

E1/E2は2次元既約表現なので、character projector全体を対角化すると群強制の二重縮退を
含む。本実装はmatrix-unit rowを使い、multiplicity次元だけを対角化して相方を群作用から
再構成する。これにより固有値problemをさらに半分にし、重複固有値に対するautogradの
不安定性を避ける。

`symmetry="auto", polarization="x"|"y"` では同じD6-closed starとCs部分群を使う。
`symmetry="d6", polarization="x"|"y"` では12作用のReynolds平均後、sourceが属する
(E_1) matrix-unit rowだけを解く。その次元は (M(M+1)+1) で、Csの
(3M(M+1)+1) より漸近3分の1である。`polarization=None` の全6 isotypic block経路と
合わせ、用途の異なる三つの厳密なD6-star短縮を選べる。

## 8. 検証

`validate_triangular_matched_asr.py --integration` は次を検査する。

- D6 star closure
- NVM \(P_\star,Q_\star\) のCs不変性
- x/y各sectorの次元短縮
- x sectorとy sectorの和が独立full-star S行列と一致すること
- 6種類のD6指標projectorの冪等性・相互直交性・完全性と次元和
- E1/E2 matrix unitの積則、row rank、相方再構成
- signed-permutationによる \(O(D^2)\) Reynolds作用とdense作用の一致
- ランダムなD6-equivariant \(P:H\to E,Q:E\to H\) の全122 mode再構成残差
- NVM／matched-ASRの完全D6結果と独立full-star結果の一致
- NVM／matched-ASRの完全D6 E1 x/y rowと全既約表現解／Cs解のsource response一致
- E1-row次元 (M(M+1)+1)、source射影残差、12作用不変性
- complete D6／E1-row／Csの内部6成分場、前後入射、partial/full出力の一致
- 入出力界面における接線場の連続性
- 直交C2v NVM／matched-ASRの内部・外部場とLi-2a quarter/full一致
- fields-only出力と通常field出力の一致
- RedhefferとLi algorithm 2aの一致
- `quarter`反射と`half`反射の一致

一般回帰検証 `validate_rcwa_solver_auto.py` も三角NVMのx/y sectorを実行する。
