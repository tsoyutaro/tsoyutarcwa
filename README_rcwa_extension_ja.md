# torcwa RCWA 拡張

成果物全体のフォルダ構成と推奨読解順は `README_ja.md` を参照してください。

## 追加した機能

- `Circle` を NVM（normal-vector method）で扱えます。
- `Lattice.hexagonal_close_packed(period)` と `Circle.close_packed(...)` で、2次元の三角格子上に円を配置できます。
- セル中心の単一円を、直交格子または60°三角格子の matched-coordinate ASR-FR で扱えます。明示的に `LayerSpec(method="matched-asr")` を指定します。
- 三角格子 matched-ASR では、D6-equivariant な周期写像、斜交基底の計量、一般2次元変換行列 `(T, T_z)` を使用します。
- S行列の連結法を `cascade="redheffer"`（通常のS行列）または `cascade="algo2a"` から選択できます。
- `OutputSpec.smatrix_size` を `"full"`、`"half"`、`"quarter"` から選択できます。
- NVM円形層で `GroupTheoryOptions(enabled=True)` を指定すると、直交格子では C2v、三角・斜交格子では C2 の全モードブロック固有値分解を試みます。
- 直交格子・正入射・単一円では、`polarization="x"` または `"y"` により、入射偏光が到達するC2vセクターだけを解けます。
- 三角格子の円形NVMおよびmatched-ASRでは、`symmetry="auto"` ならD6-closed star/Cs、`symmetry="d6"` なら完全D6のE1 matrix-unit rowを使い、正入射x/y単一偏光だけを解けます。
- 一般斜交格子の円形NVMでは、正入射x/yが共有するC2 source sectorだけを解き、交差偏光を保持したまま固有値問題を短縮できます。

ここでいう六方最密配置は、1個の円を三角 Bravais 格子の各格子点へ置く「2次元の円充填」です。3次元結晶の ABAB 型 HCP 積層ではありません。

## 使用例

```python
import torch

from rcwa_solver_auto import (
    AutoRCWA,
    Circle,
    GroupTheoryOptions,
    Lattice,
    LayerSpec,
    Material,
    NVMOptions,
    OutputSpec,
)

period = 1.0
sim = AutoRCWA(
    freq=1 / 1.55,
    order=[5, 5],
    lattice=Lattice.hexagonal_close_packed(period),
    cascade="algo2a",
    outputs=OutputSpec(smatrix_size="half", fields="none"),
    nvm=NVMOptions(grid=(256, 256)),
    group_theory=GroupTheoryOptions(
        enabled=True,
        strict=True,
        polarization="x",
    ),
    dtype=torch.complex128,
    device="cpu",
)
sim.add_input_layer(eps=1.0)
sim.add_output_layer(eps=2.25)
sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
sim.add_structured_layer(
    LayerSpec(
        thickness=0.22,
        geometry=Circle.close_packed(period, gap=0.08),
        background=Material(1.0),
        inclusion=Material(4.0),
    )
)
sim.solve_global_smatrix()

Tf, Rf = sim.S[0], sim.S[1]
print(sim.computed_smatrix_blocks)       # ('Tf', 'Rf')
print(sim.group_theory_diagnostics[-1])
```

三角格子の円形 matched-ASR を明示的に使う例です。`method="auto"` は円に対して
既定では解析円板係数を持つNVMを優先するため、matched-ASRは明示指定します。

```python
from rcwa_ext import ASROptions

sim = AutoRCWA(
    freq=1 / 1.55,
    order=[3, 3],
    lattice=Lattice.triangular(1.0),
    cascade="algo2a",
    outputs=OutputSpec(smatrix_size="half", fields="none"),
    asr=ASROptions(circle_G=0.08, grid=(192, 192)),
    group_theory=GroupTheoryOptions(
        enabled=True,
        strict=True,
        polarization="x",
    ),
    dtype=torch.complex128,
    device="cpu",
)
sim.add_input_layer(eps=1.0)
sim.add_output_layer(eps=1.0)
sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
sim.add_structured_layer(
    LayerSpec(
        thickness=0.22,
        geometry=Circle(radius=0.46),  # 2R < period
        background=Material(1.0),
        inclusion=Material(4.0),
        method="matched-asr",
    )
)
sim.solve_global_smatrix()
```

単一偏光だけを計算する場合は、散乱出力を `half` または `quarter` にします。

```python
sim = AutoRCWA(
    freq=1 / 1.55,
    order=[5, 5],
    lattice=Lattice.square(1.0),
    cascade="algo2a",
    outputs=OutputSpec(smatrix_size="half", fields="none"),
    group_theory=GroupTheoryOptions(
        enabled=True,
        strict=True,
        polarization="x",
    ),
    nvm=NVMOptions(grid=(256, 256)),
    dtype=torch.complex128,
    device="cpu",
)
```

直交格子では、全 `2N` 次元の固有値問題を作らず、x偏光ならC2vの第3セクター、y偏光なら第4セクターだけを解きます。三角格子では `symmetry="auto"` がD6-star/Cs、`symmetry="d6"` が完全D6 E1 source-rowを選びます。後者では未選択既約表現も未選択E1 rowも計算しません。一般斜交格子ではxとyが同じC2 source sectorに入るため、どちらを指定しても同じsectorを解き、そのsector内のx/y交差偏光応答は両方とも有効です。結果の `sim.S` はいずれも元の `2N × 2N` 配列へ埋め戻すため、既存の `S_parameters` 呼び出しと互換です。

## S行列サイズ

`sim.S` は torcwa 互換の `[Tf, Rf, Rb, Tb]` を維持します。未計算ブロックには同じ形状のゼロ行列が入ります。

| 設定 | 実際に計算するブロック | 用途 |
|---|---|---|
| `full` | `Tf, Rf, Rb, Tb` | 両方向散乱を公開 |
| `half` | `Tf, Rf` | 入力側からの透過・反射 |
| `quarter` | `Rf` | 入力側からの反射のみ |

`half` と `quarter` でも `fields="external"`、`"internal"`、`"all"` を指定できます。公開
S行列は要求されたblockだけを保持し、場再構成用には両方向の完全S行列と各層のmodal
couplingを非公開で保存します。そのため `quarter` でも出力側外部場、内部場、後方入射場を
再構成できますが、場を不要とする場合よりメモリ使用量は増えます。
`fields="external"` は入出力半空間、`"internal"` は構造層内、`"all"` は両方のAPIを
有効にします。指定外の領域を要求した場合は、誤って不完全な場を返さず例外にします。
場だけが必要なら `OutputSpec(smatrix=False, smatrix_size="quarter", fields="all")` と
指定できます。この場合も場用Sは内部計算しますが、公開 `sim.S` はゼロplaceholder、
`S_parameters()` は例外となります。旧APIの `enable_fields=True`／
`store_mode_couplings=True` もpartial Sと併用でき、`fields="all"` 相当として扱います。
三角matched-ASRでは横場の \(T\) と縦場の \(T_z\) もnative star内へ制限してから矩形配列へ
埋め戻すため、縦場inverse後に除去済みcorner harmonicを再混入させません。

## 群論オプション

群論による分解は、正入射かつセル中心の単一円を持つNVM層またはmatched-ASR層で使用します。適用される分解はbackendと格子で異なります。

- 直交NVM: C2v。x/y source-specific短縮にも対応
- 三角NVM: 全モード計算はC2または完全D6、x/y source-specific計算はCsまたは完全D6 E1-row短縮
- 一般斜交NVM: C2ブロック分解。x/yは同じC2 source sectorに属するため、どちらの指定でも同じ共通sectorだけを解く短縮に対応し、sector内の交差偏光結合も保持
- 直交matched-ASR: C2vのx/y source-specific短縮
- 三角matched-ASR: D6-closed star上のCs、または完全D6 E1-rowによるx/y source-specific短縮
- 対称条件を満たさない場合: 通常の固有値分解へフォールバック
- `GroupTheoryOptions(strict=True)` の場合: フォールバックせず例外を送出

各ブロックへの射影後に `P`、`Q` 演算子の不変性残差を検査します。`symmetry="auto", polarization="x"|"y"` は従来のCs鏡映sector、`symmetry="d6", polarization="x"|"y"` は12作用でReynolds平均した後のE1 matrix-unit rowだけを使用します。NVMの誘電率・逆誘電率・法線射影行列をstarへ制限してから逆行列と `P_star,Q_star` を直接組み立てるため、矩形打切りの隅調和波は逆行列を介して混入しません。

### D6の完全分解

六方（三角）格子の円形構造は、正入射では60度回転6個と鏡映6個からなる12要素の面内対称群 D6（光学では C6v と書くこともあります）を持ちます。「完全分解」は次をすべて満たすことを意味します。

1. 採用するFourier調和波集合が、D6の全回転・鏡映で閉じている。
2. 各群要素の表現行列を作り、指標射影演算子で全既約表現へ分割する。
3. 分割された全ブロックの次元合計が元の `2N` と一致する。
4. `P` と `Q` が対応するブロック間だけを写すことを残差で検証する。
5. 全ブロックを合わせた結果が通常固有値分解と一致する。

torcwa 型の矩形打切り `m,n = -M,...,M` 全体は、60度回転で一般に集合外へ出るためD6で閉じません。この集合そのものは、torcwaを書き換えても、一からsolverを書いても、D6の表現空間にはなりません。解決するには実装元ではなく打切り集合を変更し、例えば

`max(|m|,|n|,|m-n|) <= M`

という六角形star、またはD6 orbitの完全な和をnative Fourier基底として採用します。その基底上なら全D6既約表現への完全分解が可能です。本実装はこのnative star上に12個のD6作用を構成し、`A1, A2, B1, B2, E1, E2` の全指標射影を列挙します。E1/E2はmatrix-unitの1行だけを固有値分解し、二重縮退する相方を群作用から再構成します。x/y source-specific計算は、Cs sectorに加えて、sourceが属するE1のx/y rowだけを解く完全D6経路にも対応します。

実装済みの三角NVM／matched-ASR偏光短縮では、矩形打切り内部からD6で閉じるharmonicだけを選びます。`symmetry="auto"` はCs鏡映の偶・奇sector、`symmetry="d6"` とx/yの組合せはE1の対応matrix-unit rowだけを解きます。後者の次元は `M*(M+1)+1` で、Csの `3*M*(M+1)+1` よりさらに約3分の1です。一方 `symmetry="d6", polarization=None` は6種類の全isotypic blockを解きます。

さらに、`GroupTheoryOptions(enabled=True, symmetry="d6", polarization=None)` を指定すると、同じnative star上で12個のD6作用を作り、指標射影により `A1, A2, B1, B2, E1, E2` の全isotypic blockを列挙します。E1/E2はmatrix-unit rowへさらに縮約します。NVMとmatched-ASRの両方、RedhefferとLi-2a、`full`／`half`／`quarter` S行列、内部・外部電磁場再構成に対応します。公開S行列と場のFourier係数はtorcwa互換の矩形サイズへ埋め戻され、native starに含まれないcorner harmonicはゼロです。三角starの縦場は `epsilon_zz`／`mu_zz` の逆則をstar内で解いてから埋め戻します。

- 三角NVM: `symmetry="auto"` のCs短縮と、`symmetry="d6"` の完全D6 E1 source-row短縮に対応。
- 三角matched-ASR: `symmetry="auto"` のCs短縮と、`symmetry="d6"` の完全D6 E1 source-row短縮に対応。
- 完全D6 E1 x/y source-row: 三角NVM／matched-ASRで対応。固有値・cascadeともrow次元で実行。
- 全矩形基底のD6完全分解: 集合がD6で閉じないため数学的に不可能。
- native D6-star基底全体の完全既約分解: 三角NVM／matched-ASRで実装済み。

```python
sim = AutoRCWA(
    freq=1 / 1.55,
    order=[5, 5],
    lattice=Lattice.triangular(1.0),
    outputs=OutputSpec(smatrix_size="quarter", fields="all"),
    group_theory=GroupTheoryOptions(
        enabled=True,
        symmetry="d6",
        strict=True,
        polarization=None,
    ),
)
```

特定のx偏光だけを完全D6で解く場合:

```python
group_theory=GroupTheoryOptions(
    enabled=True,
    symmetry="d6",
    strict=True,
    polarization="x",
)
```

通常どおりsourceを設定して `solve_global_smatrix()` を実行した後、`field_xy`、`field_xz`、
`field_yz` を呼び出せます。Cs短縮と完全D6 E1-row短縮でも同じAPIを使用します。
偏光／群論短縮時に選択sector外のFourier sourceを渡した場合は、その成分を黙って捨てず
明示的に例外にします。任意sourceには群論短縮を無効化するか、D6/Csに適合するorbit和を
使用してください。

三角NVMのstar内inverse rule、偏光射影、一般斜交格子のx/y共通C2 sectorは
`docs/triangular_nvm_polarization_report_ja.md` に導出しています。

## 円形matched-ASRとNVMの関係

円形ASRはすでに実装済みです。ただし、NVM行列へASR行列を後から掛ける `ASR+NVM` 混成ではなく、円境界へ座標線を一致させた独立の **matched-coordinate ASR-FR** バックエンドです。APIでは `method="matched-asr"` を使います。

- 直交格子: Weiss型の円境界matched mapとseparable ASR stretchを組み合わせます。
- 60°三角格子: Wigner–Seitz六角形を円へ写すD6-equivariant周期Hermite mapを使い、斜交基底の計量と周期境界を同時に扱います。
- 両格子: Jacobianから変換媒質 `epsilon'`, `mu'` を作り、Fourier factorization後の `(P,Q)` を解き、一般変換 `(T,T_z)` で通常のCartesian S行列へ接続します。
- コアシェル: `add_layer_circle_shell_asr(..., radial_mapping="outer"|"double")` を選べます。`double` は計算空間の固定支持曲線を内外円へ写す単調性保証付き半径方向C2 quintic-Hermite写像で、両半径のautogradを保持します。界面勾配は全区間・全方向の最小割線勾配以下へ制限し、中心・周期境界の勾配は単調範囲内で大きくして過圧縮を避けます。
- 二重matched写像: level-set法線と計算格子計量から法線projectorを作り、連続な接線E／法線Dに一般化Li因数分解を適用します。`factorization_rules=False`の場合だけ比較用の直接Fourier畳み込みを使います。
- NVM: 円板の誘電率Toeplitz行列をFourier–Bessel式で解析的に作り、normal-vector projectionを使う独立バックエンドです。

したがって、現在は `nvm` と `matched-asr` の2経路を独立に選択・比較できます。
Cartesian NVM射影行列をmatched-coordinate tensorへ後掛けする二重補正は採用しません。
単一円はWeiss対称因数分解、二重matchedコアシェルは一般化Li因数分解により、
matched空間内だけで境界条件を処理します。

matched-ASRの適用条件は、セル中心の単一円または同心コアシェル、非接触条件
`2*outer_radius < period`、固定トポロジーです。真の接触極限や内外半径の一致極限では
写像Jacobianが特異になり得るため、正のgapとshell厚を設けてください。

## 検証

PyTorch と torcwa 0.1.4.2 が入った環境で、`rcwa_solver_auto.py`、`rcwa_ext/`、検証スクリプトを同じ成果物ディレクトリ構成のまま実行します。

```bash
python validation/validate_rcwa_solver_auto.py
```

検証内容は次の通りです。

- 直交格子の円形NVM層
- 三角格子の六方配置円形NVM層
- Redheffer と algo2a の `full` 一致
- `half` / `quarter` と `full` の対応ブロック一致
- 群論分解と通常固有値分解のS行列一致
- x/y単一偏光セクターとフル計算の透過・反射ベクトル一致
- 単一偏光版 Redheffer と algo2a の一致
- 三角matched-ASRのD6回転・鏡映equivarianceと周期境界
- 三角matched-ASR／NVMのD6-closed star上のx/y偏光短縮
- x/y両セクターの和と独立full-star source responseの一致
- native-star完全D6の6既約表現、次元和、projector完全性
- 完全D6 NVM／matched-ASRと独立full-star S行列の一致
- 完全D6 E1 x/y rowと全既約表現解／Cs解のsource response、次元式の一致
- 完全D6版Redheffer／Li-2aおよびfull／half／quarterの一致
- 完全D6／Csの内部場、両方向modal coupling、partial/full field一致
- 層界面での接線 `Ex,Ey,Hx,Hy` 連続性とstar内 `Ez,Hz` 再構成
- matched-ASRの `T/Tz` 適用後も内部6成分場のcorner harmonicがゼロであること
- 直交C2vのNVM／matched-ASRについて、Li-2a quarterとfullの内部・外部6成分場一致
- コアシェル二重matched写像の正方／三角両界面一致、Jacobian正値、内外半径autograd、一般化Liのinverse/direct極限・斜交定数極限・D6共変性、受動性、D6 E1 source-row併用
- `smatrix=False` のfields-only出力と通常field出力の一致、公開Sの無効化
- 一般斜交NVMで、x/y共通C2 source sectorとfull source responseが一致すること
- 既存の長方形 ASR-FR の回帰スモークテスト

追加の検証は次のとおりです。

```bash
python validation/validate_circle_matched_asr.py --integration --order 1 --grid 64
python validation/validate_triangular_matched_asr.py --integration --order 1 --grid 64
python validation/validate_design_gradients.py --json validation/results/design_gradient_validation.json
```

計算量を増やした収束確認例:

```bash
python validation/validate_rcwa_solver_auto.py --order 5 --grid 256 --json validation/results/modularization_validation.json
```

## 半径・層厚を直接最適化する場合

`Circle(radius=...)` と `LayerSpec(thickness=...)` には `requires_grad=True` の実 scalar
Tensor を渡せます。半径最適化には `method="nvm"` または `method="matched-asr"` を
使用してください。硬い standard raster の円は半径について微分不能なので、trainable
radius を渡した場合は明示的に例外になります。

```bash
python validation/validate_design_gradients.py --json validation/results/design_gradient_validation.json
```

この検証は standard thickness、直方・三角 NVM、直方・三角 matched-ASR、Csおよび
完全D6 E1-rowのx/y偏光短縮について autograd と中心差分を比較します。数式と最適化例は
`docs/design_gradient_report_ja.md` にあります。

## 金モスアイ／金基板の収束計算

高さ方向に半径が変化する金モスアイを円柱matched-ASR層へ分割し、スライス数、Fourier次数、
ASR sampling gridを交互に収束させるコードを追加しました。

```bash
python studies/gold_motheye/converge.py --device cuda
```

金はRakić Lorentz–Drude分散を既定で使用し、測定 `n,k` CSVにも切り替えられます。半無限
金基板では遠方透過率を0とし、モスアイ内吸収と基板へ流入して最終的に吸収されるpowerを
分けて出力します。三角格子の既定は完全D6 E1 x-source rowで、比較用の
`--symmetry-reduction cs-source` と群論無効化 `--no-symmetry` も選択できます。
仮定、数式、収束判定、必要な追加形状条件は
`studies/gold_motheye/README_ja.md` を参照してください。

## モジュール構成

実装は `rcwa_ext/` パッケージへ分割しました。旧 `rcwa_solver_auto.py` は互換 import
だけを提供する薄い facade なので、既存コードは変更不要です。新規コードでは

```python
from rcwa_ext import AutoRCWA, Circle, LayerSpec, Material
```

を推奨します。ファイル別の責務、支配方程式、ASR/NVM/群論/S 行列の導出は
`docs/rcwa_modular_math_guide_ja.md`、native-star完全D6の導出と計算量は
`docs/complete_d6_native_star_report_ja.md` を参照してください。

## PMMAモスアイへの金薄膜被覆

PMMAコア、同心金シェル、空気からなる三材料断面を扱う
`add_layer_circle_shell_asr` を追加しました。外側の金–空気境界をmatched-coordinate写像へ
整合する `radial_mapping="outer"` と、内側の金–PMMA境界にも同時に整合する
`radial_mapping="double"` を選べます。回折次数と高さ方向の層数を別々に収束させる
実行ファイルは次です。

```bash
python studies/pmma_gold_motheye/converge_order.py
python studies/pmma_gold_motheye/converge_layers.py
```

二重matched写像を使う場合は、各コマンドへ `--radial-mapping double` を追加します。

両者とも三角格子では完全D6 E1 x-source rowが既定です。
モデル、数式、判定規則、出力、実行順序は
`studies/pmma_gold_motheye/README_ja.md`、検証は
`studies/pmma_gold_motheye/validation/validate.py` を参照してください。

## Wang et al. (2022) 図8の正方形金属パッチ

論文 *2D rigorous coupled wave analysis with adaptive spatial resolution for a multilayer
periodic structure* の図8(a)(b)を、既存の分離ASR写像・Fourier因数分解・層間変換
`T` を用いて再現する専用コードを追加しました。

```bash
python paper_reproductions/wang2022_fig8/reproduce.py --study smoke --device cpu --no-plot
python paper_reproductions/wang2022_fig8/validation/validate.py
python paper_reproductions/wang2022_fig8/reproduce.py --study fig8 --device cuda
```

論文条件はperiod 30 mm、15 mm角patch、厚さ0.01 mm、2–18 GHz、`G=0.001`、
ASR-FR `N=M=8`、ASR `N=M=20`です。図8(b)のtotal-minus-zero-order power、
10 GHz付近のRayleigh cutoff、passivity、再開可能CSVも扱います。論文のHFSS元データは
公開されていないため、外部CSVがある場合だけ重ねます。数式、符号規約、実行順序、
検証範囲は `paper_reproductions/wang2022_fig8/README_ja.md` を参照してください。

## Peng–Zhang (2025) のAg–air–Ag円形aperture–particle配列

Fig. 2 のMI構造（周期62 µm、外半径30 µm、内半径14 µm、Ag厚1 µm、
Ag背景／空気環状開口／Ag中心粒子、半無限PI基板 `epsilon=3.5+0.009i`、1–3 THz）を
扱う正方格子コードと、三角格子primitiveを厳密に等価な直交二サイト・スーパーセルと
比較するコードを `paper_reproductions/peng2025/` にまとめました。

```bash
python paper_reproductions/peng2025/reproduce_square.py --study smoke --device cpu
python paper_reproductions/peng2025/compare_hex_supercell.py --study smoke --device cpu
python paper_reproductions/peng2025/validation/validate.py --device cpu
```

論文本文にないAg Drude定数とMI基板厚は再現仮定としてmetadataへ明記します。
三角格子比較ではmatched-ASR対hard rasterの差を格子差と誤認しないよう、standard-raster
primitive対standard-raster supercellで同値性を判定し、matched-ASR結果を別系列で示します。
条件、スーパーセルの導出、本計算コマンド、検証結果は
`paper_reproductions/peng2025/README_ja.md` を参照してください。通常結果は
`paper_reproductions/peng2025/results/`、検証結果は
`paper_reproductions/peng2025/validation/results/` に分離して保存します。
