# 半径・層厚を設計変数にするための自動微分（autograd）対応

## 結論

従来コードには、`Circle.__post_init__`、NVM 層追加、matched-ASR 写像生成で
`radius` を `float` に置き換える箇所がありました。この後で Bessel 関数や座標写像を
評価しても、元の設計 Tensor まで計算グラフが戻らないため、半径勾配は消失します。

修正版は次の規則に統一しています。

- 有限性・正値性・幾何適格性の判定だけに detached な Python scalar を使う。
- Fourier 係数、Bessel 関数、写像、Jacobian、固有値問題、伝搬位相には実 Tensor を使う。
- `Circle` は呼び出し側の Tensor をそのまま保持する。
- 硬いラスタ円は半径について微分不能なので、trainable な半径を渡して
  `method="standard"` を選んだ場合は例外にし、NVM または matched-ASR の使用を促す。

## 1. 二つの値を分離する理由

設計変数を (p) とすると、コード内では

\[
p_{\mathrm{tensor}}=\operatorname{to}(p),\qquad
p_{\mathrm{check}}=\operatorname{item}(\operatorname{detach}(p))
\]

を作ります。`p_check` は `if` 文やエラーチェック専用です。物理量
(F) は必ず (p_{\mathrm{tensor}}) から計算するため、

\[
\frac{\partial F}{\partial p}
\]

の計算グラフが維持されます。`torch.as_tensor(...).to(...)` による dtype/device 変換は
元 Tensor への autograd 経路を保ちます。

## 2. NVM の円板 Fourier–Bessel 係数

面積 (A)、半径 (R)、物性コントラスト
(\Delta\epsilon=\epsilon_{\mathrm{cyl}}-\epsilon_{\mathrm{bg}}) の円板について、
逆格子差 (\mathbf G) に対する係数は

\[
\epsilon_{\mathbf G}
=\epsilon_{\mathrm{bg}}\delta_{\mathbf G,0}
+\Delta\epsilon\,\frac{\pi R^2}{A}
\frac{2J_1(|\mathbf G|R)}{|\mathbf G|R}
\exp(-i\mathbf G\cdot\mathbf r_c)
\]

です。したがって半径は充填率 (\pi R^2/A) と

\[
\operatorname{jinc}(x)=\frac{2J_1(x)}{x},\qquad x=|\mathbf G|R
\]

の両方に現れます。使用した PyTorch 環境では
`torch.special.bessel_j1` 自体に backward がありません。そのため forward 値を変えず、

\[
\frac{d}{dx}\frac{2J_1(x)}{x}
=\frac{2J_0(x)}{x}-\frac{4J_1(x)}{x^2}
\]

を使う `_Jinc` の custom backward を追加しました。原点近傍では

\[
\operatorname{jinc}(x)=1-\frac{x^2}{8}+\frac{x^4}{192}+O(x^6),
\quad
\operatorname{jinc}'(x)=-\frac{x}{4}+\frac{x^3}{48}+O(x^5)
\]

を使い、(0/0) と桁落ちを避けます。NVM の normal-vector 投影に使う中心 taper 半径も
Tensor のまま計算します。このため直方格子と三角格子の円柱 NVM の双方で
(R\rightarrow(P,Q)\rightarrow(k_z,W,V)\rightarrow S) の勾配が通ります。

## 3. matched-ASR の半径勾配

matched-ASR では物理座標を

\[
\mathbf r=\mathbf F(\mathbf u;R),\qquad
J=\frac{\partial(x,y)}{\partial(u,v)}
\]

と書き、変換後の横テンソルと縦成分は概略

\[
\epsilon_t' = \epsilon\,\det J\,J^{-1}J^{-T},
\qquad
\epsilon_{zz}'=\epsilon\det J
\]

になります。半径微分は材料の hard mask を直接微分するのではなく、円境界に matched
させた写像と metric の

\[
\partial_R J,\quad \partial_R\det J,\quad
\partial_R\epsilon'
\]

として固有値問題へ入ります。

直方格子版では、半径依存 breakpoint を `torch.tensor([...])` で再生成すると要素が
detach されるため、0 次元 Tensor の `torch.stack` に変更しました。また周期端点を
`cumsum` の出力へ in-place 代入せず、定数端点との `cat` で構成します。

三角格子版では (x_u,x_v,y_u,y_v) を `torch.autograd.grad` で生成しています。
ここで `create_graph=False` のままだと

\[
\partial_R(\partial_u x),\quad \partial_R(\partial_v x)
\]

が失われます。trainable な半径の場合に `create_graph=True` とし、写像値・Jacobian・
行列式に付いていた `.detach()` も除去しました。D6-star 上の x/y 単一偏光セクターでも
同じ半径依存 (P,Q,T) を縮約するので、縮約後の固有値問題まで逆伝播できます。

三角NVMのx/y sectorでは、半径依存の \([\epsilon]\)、\([\epsilon^{-1}]\)、normal-vector
projectionをD6-starへ制限した後、star内でinverse ruleと \((P_\star,Q_\star)\) を再計算
します。この経路でもradiusをTensorのまま保持するため、sector固有値問題まで勾配が
接続されます。

## 4. 層厚の勾配

層厚 (d) は材料 Fourier 係数には入らず、主として伝搬行列

\[
X(d)=\operatorname{diag}\left[
\exp(i\omega k_{z,n}d)
\right]
\]

へ入ります。従って

\[
\frac{\partial X}{\partial d}
=\operatorname{diag}(i\omega k_{z,n})X
\]

です。standard、NVM、ASR の層追加関数すべてで thickness を実 Tensor として保存し、
Redheffer と Li algorithm 2a の位相評価まで同じ Tensor を渡します。検証用の
`float` 化は非負性判定だけに限定しました。

## 5. 最適化例

制約を満たすよう unconstrained parameter を sigmoid で写像する例です。

```python
import torch
from rcwa_solver_auto import AutoRCWA, Circle, LayerSpec, Material

raw_radius = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
raw_thickness = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
optimizer = torch.optim.Adam([raw_radius, raw_thickness], lr=2e-2)

for _ in range(100):
    optimizer.zero_grad()
    radius = 0.10 + 0.35 * torch.sigmoid(raw_radius)     # 0.10 < R < 0.45
    thickness = 0.02 + 0.48 * torch.sigmoid(raw_thickness)

    sim = AutoRCWA(...)  # 各反復で新しい simulation を構築
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=1.0)
    sim.set_incident_angle(0.0, 0.0)
    sim.add_structured_layer(LayerSpec(
        thickness=thickness,
        geometry=Circle(radius=radius),
        background=Material(1.0),
        inclusion=Material(4.0),
        method="nvm",  # または "matched-asr"
    ))
    sim.solve_global_smatrix()
    loss = -torch.sum(torch.abs(sim.S[0]) ** 2)
    loss.backward()
    optimizer.step()
```

同じ simulation オブジェクトを parameter 更新後に再利用せず、各反復で modal basis と
S 行列を再構築してください。幾何制約には clamp より sigmoid/softplus の再パラメータ化を
使う方が、制約境界で勾配を失いにくくなります。

## 6. 数値検証

`validation/validate_design_gradients.py` は目的関数

\[
\Phi=\lVert S_{\mathrm{Rf}}\rVert_F^2
\]

について autograd と中心差分（刻み (10^{-5})）を比較します。order 1、64×64 sampling、
PyTorch 2.13.0+cpu、torcwa 0.1.4.2 で全項目が合格しました。

| 経路 | radius の scaled error | thickness の scaled error |
|---|---:|---:|
| standard homogeneous | — | 1.18e-7 |
| 正方格子 NVM | 1.04e-9 | 5.00e-10 |
| 三角格子 NVM | 5.59e-9 | 9.40e-10 |
| 正方格子 matched-ASR | 7.75e-10 | 6.66e-10 |
| 三角格子 matched-ASR | 5.08e-8 | 9.16e-10 |
| 正方格子 matched-ASR x sector | 1.82e-9 | 6.07e-10 |
| 三角格子 matched-ASR y sector | 9.42e-8 | 8.30e-11 |

Jinc 単体の微分誤差は 8.85e-13 でした。また `Circle` が入力 Tensor の同一性を保つこと、
trainable radius を硬い standard raster へ渡すと明示的に拒否されることも検証しています。

上表は三角NVM偏光短縮追加前のbaseline結果です。現在の
`validation/validate_design_gradients.py` には `triangular NVM y sector`、x/y共通C2 sectorを使う
`oblique NVM x source sector`、NVM／matched-ASRの `complete D6` と
`complete-D6 E1 x row` も追加しており、同じ
autograd／中心差分比較を実行します。complete D6のE1/E2はmatrix-unit rowだけを対角化し、
群強制の重複固有値をeig backwardへ直接渡さない構成です。
さらに、完全D6全既約表現とE1 x-rowのNVM／matched-ASRについて `fields="all"` と
`smatrix_size="quarter"` を
組み合わせ、層中央の6成分Fourier場ノルムを目的関数とするradius／thickness勾配も
autogradと中心差分で照合します。内部場のmodal coupling、star内縦場inverse、空間合成まで
計算グラフを保持します。

## 7. 微分可能性の限界

- hard raster の occupancy `distance <= radius` は区分的に一定で、境界通過時は不連続です。
  この経路を半径最適化に使うことはできません。
- matched-ASR は固定トポロジー内の形状微分です。半径が接触条件に達する、最近接像が
  切り替わる、sample が区分境界を跨ぐ、といった離散的イベント上では微分不能です。
- 固有値の完全縮退点では固有ベクトル微分が不安定になり得ます。対称点では x/y sector
  短縮を使うか、目的関数の収束性を有限差分でも確認してください。
- 検証結果は order 1 の勾配整合性試験です。実設計では Fourier order と sampling grid を
  上げ、目的値と勾配の双方について収束試験を行ってください。
