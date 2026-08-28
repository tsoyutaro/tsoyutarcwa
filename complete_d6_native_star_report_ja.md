# native-star 完全D6 solverと理論計算量

## 1. 実装範囲

三角Bravais格子、正入射、primitive-cell中心の単一円について、NVMとmatched-ASRの双方に
native-star完全D6固有値分解を追加した。指定は次のとおりである。

```python
group_theory=GroupTheoryOptions(
    enabled=True,
    symmetry="d6",
    strict=True,
    polarization=None,
)
```

matched-ASRでは、同じ中心を持つPMMAコア／金シェルのような同心円層にも適用できる。
写像が厳密にmatchedする境界は外円であり、内円は変換座標上で求積する。

特定の正入射x/y偏光だけが必要な場合は、全既約表現を再結合せず、sourceが属する
\(E_1\) matrix-unit rowだけを解く。

```python
group_theory=GroupTheoryOptions(
    enabled=True,
    symmetry="d6",
    strict=True,
    polarization="x",  # または "y"
)
```

散乱cascadeはRedhefferとLi algorithm 2a、S行列は `full`、`half`、`quarter`、場出力は
`none`、`external`、`internal`、`all` に対応する。公開S行列はtorcwa互換の矩形サイズへ
埋め戻すが、native starにないcorner harmonicの行・列はゼロになる。場を要求した場合は
公開Sの大きさによらず、両方向の完全なfield S行列と各層modal couplingを非公開で保持する。
`OutputSpec(smatrix=False, fields=...)` のfields-only出力にも対応し、公開Sを無効化しても
この非公開データから場を再構成する。
matched-ASRのCartesian変換についても \(T_\star=E_v^\dagger TE_v\)、
\(T_{z,\star}=E_s^\dagger T_zE_s\) を使用し、場再構成の最終段でfull-box変換を掛けて
native star外のcorner harmonicを再混入させない。

## 2. native Fourier star

torcwa型の矩形打切り

\[
\mathcal B_M=\{(m,n):-M\le m,n\le M\}
\]

は60度回転で閉じない。そのため、solverを一から書くだけではこの集合を完全D6分解できない。
本実装はD6で閉じる六角形star

\[
\mathcal H_M=\{(m,n):\max(|m|,|n|,|m-n|)\le M\}
\]

を採用する。そのscalar次元と面内vector次元は

\[
N_\star=3M(M+1)+1,\qquad D_\star=2N_\star
\]

である。比較対象の矩形vector次元は

\[
D_\Box=2(2M+1)^2,\qquad D_\star/D_\Box\longrightarrow 3/4
\]

となる。

## 3. D6指標射影

60度回転を \(r\)、x軸鏡映を \(s\) とし、12要素
\(r^0,\ldots,r^5,s,rs,\ldots,r^5s\) の作用をnative star上へ構成する。electric fieldには
polar-vector表現、magnetic fieldには鏡映で追加の符号を持つaxial-vector表現を用いる。
既約表現 \(\alpha\in\{A_1,A_2,B_1,B_2,E_1,E_2\}\) へのisotypic projectorは

\[
\Pi_\alpha=\frac{d_\alpha}{12}\sum_{g\in D_6}
\chi_\alpha(g)^*D(g)
\]

である。ここで \(d_\alpha=1\) はA/B表現、\(d_\alpha=2\) はE表現である。実装は

\[
\Pi_\alpha^2=\Pi_\alpha,\quad
\Pi_\alpha\Pi_\beta=0\ (\alpha\ne\beta),\quad
\sum_\alpha\Pi_\alpha=I
\]

およびelectric/magnetic rank一致、\(P,Q\) のblock不変性を検査する。有限gridのFourier化で
生じる微小な対称性誤差は、12作用のReynolds平均

\[
P_{D_6}=\frac1{12}\sum_gD_E(g)P D_H(g)^\dagger,
\qquad
Q_{D_6}=\frac1{12}\sum_gD_H(g)Q D_E(g)^\dagger
\]

で対称化し、その補正ノルムをdiagnosticへ記録する。群作用はsigned permutationと2成分
回転へ分け、denseな \(D(g)PD(g)^\dagger\) を作らずindex permutationで \(O(D^2)\) 適用する。

2次元既約表現E1/E2では、character projectorで得るisotypic空間をそのまま対角化しない。
標準既約表現行列 \(\Gamma^\alpha(g)\) からmatrix unit

\[
\Pi^\alpha_{ij}=\frac{d_\alpha}{12}\sum_{g\in D_6}
\Gamma^\alpha_{ij}(g)^*D(g)
\]

を作り、\(\Pi^\alpha_{00}\) のrange、すなわちmultiplicity空間だけを対角化する。二重縮退する
相方は \(\Pi^\alpha_{10}\) で再構成する。これにより、群対称性が強制する重複固有値を
`torch.linalg.eig` とそのbackwardへ直接渡さずに済む。

## 4. 理論的な計算量削減

以下は密行列固有値分解の主要項 \(O(D^3)\) に対する理想値であり、wall-clock全体の保証値では
ない。大きなstarで各既約表現が正則表現に近い比率で現れると、4個の1次元isotypic blockは
各 \(D_\star/12\)、2個の2次元isotypic blockは各 \(D_\star/3\) になる。E1/E2はmatrix-unit
rowにより各 \(D_\star/6\) のmultiplicity問題を一度だけ解けばよい。したがって実装済み
固有値分解の総主要計算量は

\[
4\left(\frac{D_\star}{12}\right)^3+
2\left(\frac{D_\star}{6}\right)^3
=\frac{5}{432}D_\star^3.
\]

従って、同じnative starを一括対角化する場合との理論比は

\[
\frac{D_\star^3}{(5/432)D_\star^3}=\frac{432}{5}=86.4
\]

倍である。矩形box一括対角化との漸近比はstar自体の \((3/4)^3\) も含め

\[
\frac{5}{432}\left(\frac34\right)^3=\frac{5}{1024},
\qquad \text{理論高速化}=\frac{1024}{5}=204.8
\]

倍となる。

block行列を全部同時に保持すると仮定した二乗メモリの和は

\[
4\left(\frac{D_\star}{12}\right)^2+
2\left(\frac{D_\star}{6}\right)^2=\frac1{12}D_\star^2,
\]

すなわち固有値workspaceの二乗和はstar一括の約1/12、矩形box一括の漸近約3/64である。
最大problemは約 \(D_\star/6\) なので、blockを逐次処理する場合の固有値workspace peakは
star一括の約1/36、矩形box一括の漸近約1/64になる。ただし現在の実装はnative-starの
\(P,Q\)、projector和、再構成後の全modeも保持するため、solver全体の実測peak memoryが
そのまま1/36になるわけではない。

次数 \(M=4\) の独立検証では \(D_\star=122\)、\(D_\Box=162\)、isotypic次元は
`A1:10, A2:10, B1:10, B2:10, E1:42, E2:40`、実際に対角化する次元は
`10,10,10,10,21,20` であった。実寸の三乗和から、固有値主要項はstar一括比約85.41倍、
矩形box一括比約199.97倍の削減となる。

## 5. S行列cascadeと単一偏光との違い

全モード完全D6経路は各blockで固有modeを求めた後、全modeをnative-star基底へ再結合して
interfaceとS行列cascadeを実行する。そのため約86.4倍という値は主に層固有値問題へ適用され、
cascade全体が同じ比率で短縮されるわけではない。cascadeの現実的な確実な次元削減は
\(D_\star/D_\Box\to3/4\) で、密行列積の主要項なら最大約 \((4/3)^3=2.37\) 倍である。

xまたはyのゼロ次sourceはD6の \(E_1\) 表現に属するため、他の
\(A_1,A_2,B_1,B_2,E_2\) blockは励起されない。新しいsource-row経路はxなら
\(\Pi^{E_1}_{00}\)、yなら \(\Pi^{E_1}_{11}\) のrangeだけで固有値問題、interface、伝搬、
S行列cascadeを実行する。その厳密な次元は

\[
D_{E_1,{\rm row}}=M(M+1)+1
\]

である。旧Cs経路の次元は \(D_{Cs}=N_\star=3M(M+1)+1\) なので、漸近的に

\[
\frac{D_{E_1,{\rm row}}}{D_{Cs}}\to\frac13.
\]

従って、密行列主要項は旧Cs比約1/27、二乗メモリは約1/9になる。native-star一括比では
主要項が約1/216、矩形box一括比では漸近約1/512、二乗メモリpeakはそれぞれ約1/36、
約1/64である。次数 \(M=4\) では `Cs=61`, `E1-row=21` で三乗比は約24.5倍、
\(M=22\) では `Cs=1519`, `E1-row=507` で約26.9倍となる。

有限gridで生じる微小非対称性にはCs平均ではなく12個すべてのD6作用によるReynolds平均を
適用してからE1 rowへ射影する。diagnosticには `symmetry="D6-E1-source-row"`、
`irrep="E1"`、row番号、source射影残差、演算子不変性残差、理論三乗比を保存する。

`fields="internal"|"external"|"all"` では、後方入射場とpartial公開Sからも場を一貫して
再構成するため、公開指定がhalf／quarterでも非公開cascadeは両方向4 blockを作る。
従って場を要求した実行では「公開S blockを減らす」ことによるcascade短縮は得られないが、
D6/Cs/E1-rowによる固有値問題とcascade次元の短縮はそのまま有効である。
場が不要な最速経路は `fields="none"` である。

実測wall-clockは行列構築、Bessel/Fourier係数、Reynolds平均、projector構築、GPU kernel起動、
S行列cascadeを含むため、上の理論値より小さくなる。最終的な全体速度向上は、固有値分解が
元の実行時間に占める比率に対するAmdahl則で評価する必要がある。

## 6. 検証状態

`validate_triangular_matched_asr.py` のNumPy独立検証は、D6-star閉性、12作用、6 projectors、
冪等性、相互直交性、完全性、E1/E2 matrix-unit、signed-permutation Reynolds作用、
ランダムMaxwell型intertwinerの全mode再構成と、独立な多層modal stateからの
前後方向内部field coupling再構成を含む15項目すべてに合格した。
projector完全性の最大誤差は \(1.84\times10^{-15}\)、matrix-unit誤差は
\(3.49\times10^{-15}\)、Maxwell型固有mode残差は \(6.54\times10^{-15}\)、
内部field couplingの境界残差は \(1.35\times10^{-15}\) 以下である。
PyTorch/torcwa統合検証コードにはNVM／matched-ASR、
Redheffer／Li-2a、`full`／`half`／`quarter`、独立full-star比較、内部場のpartial/full一致、
前後方向coupling、接線場の層界面連続性、空間場合成、fields-only公開S無効化に加え、
完全D6 E1 x/y rowと全既約表現解／旧Cs解のsource response一致、次元式、内部場一致を
含めている。
