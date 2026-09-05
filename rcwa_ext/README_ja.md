# rcwa_ext パッケージ

旧 `rcwa_solver_auto.py` の実装を、数学的責務ごとに分割したパッケージです。

| ファイル | 責務 |
|---|---|
| `config.py` | 格子、材料、geometry、設計 Tensor、options |
| `fields.py` | Fourier amplitude、D6/Cs内部場、縦場と空間場の再構成 |
| `scattering.py` | modal algebra、S 行列、Redheffer、Li-2a |
| `asr_maps.py` | separable、円、D6 matched-coordinate 写像 |
| `asr.py` | 変換媒質、Fourier factorization、T/Tz、ASR layer |
| `symmetry.py` | C2/C2v/D6 と x/y sector eigensolve |
| `reduced.py` | 偏光 sector 内の interface/cascade |
| `nvm.py` | 円板 Fourier–Bessel 係数、NVM tensor、斜交格子 layer |
| `auto.py` | backend 適格性判定と `AutoRCWA` |
| `__init__.py` | 公開 API |

## 現在の円形backend対応

| backend | 直交格子 | 60°三角格子 | x/y source-specific短縮 |
|---|---|---|---|
| `nvm` | 対応 | 対応 | 直交C2v、三角D6-star/Csまたは完全D6 E1-row |
| `matched-asr` | 対応 | 対応 | 直交C2v、三角D6-star/Csまたは完全D6 E1-row |
| `standard` hard raster | 固定半径のみ | 固定半径のみ | 非対応 |

三角NVMとmatched-ASRのx/y短縮は実装済みです。一般斜交格子ではx/yを異なるsectorへ
分けられませんが、両者が共有するC2 source sectorだけを解く短縮に対応します。
NVM射影行列とmatched-coordinate tensorを二重適用する `matched-nvm` は、有限Toeplitz
空間で二重補正になるため採用しません。二重matchedコアシェルは一般化Li因数分解で扱います。
native D6-star全体の `A1/A2/B1/B2/E1/E2` 完全isotypic分解は、三角NVMと
matched-ASRで実装済みです。Redheffer／Li-2a、full／half／quarter公開S、
`external`／`internal`／`all` の6成分場再構成を組み合わせられます。partial公開Sでも
場を指定した場合は、場専用の両方向full Sと各層modal couplingを非公開で保持します。

`symmetry="d6", polarization="x"|"y"` は、全D6分解のうちsourceが属するE1
matrix-unit rowだけを固有値分解とcascadeへ渡します。次元は `M*(M+1)+1` で、
従来Cs sectorの `3*M*(M+1)+1` より約3分の1です。

推奨 import:

```python
from rcwa_ext import AutoRCWA, Circle, LayerSpec, Material
```

従来の import も、親ディレクトリの互換 facade により動作します。

```python
from rcwa_solver_auto import AutoRCWA, Circle, LayerSpec, Material
```

各ファイルの数式、導出、依存関係、変更時の検証項目は
`../docs/rcwa_modular_math_guide_ja.md` にまとめています。三角NVMのD6-star/Cs偏光短縮と完全D6 E1 source-row短縮は
`../docs/triangular_nvm_polarization_report_ja.md`、native-star完全D6の全指標射影と計算量は
`../docs/complete_d6_native_star_report_ja.md` を参照してください。

応用例として、金円錐モスアイのmatched-ASRスライス数・Fourier次数・sampling gridを
収束させる `../studies/gold_motheye/converge.py`、金分散の
`../studies/shared/gold_dispersion.py`、導出と実行手順の
`../studies/gold_motheye/README_ja.md` があります。

PMMAコアへ金薄膜を被覆する同心三材料層は
`AutoRCWA.add_layer_circle_shell_asr(...)` で追加できます。`radial_mapping="outer"` は
外側境界だけに整合する従来方式、`radial_mapping="double"` は内外両境界に整合する
単調性保証付き半径方向C2写像です。端点勾配を各半径区間の最小割線勾配以下へ制限して
Jacobianの折り返しを防ぎ、内外半径Tensorの逆伝播にも対応します。二重写像では
逐次u/v factorizationを流用せず、level-set法線を用いる一般化Li
normal-D/tangential-E因数分解を使います。Cartesian NVMの後掛けは行いません。
ASRを用いない独立経路として、`AutoRCWA.add_layer_circle_shell_nvm(...)`も使用できます。
同心円の内外界面は法線方向を共有するため、一つの半径方向NVM射影と、両半径を含む
解析的Fourier-Bessel係数で三材料を扱います。
独立した回折次数／層数収束スクリプトと導出は
`../studies/pmma_gold_motheye/README_ja.md` を参照してください。

正方形金属パッチに対するWang et al. (2022)のASR-RCWA図8再現は
`../paper_reproductions/wang2022_fig8/reproduce.py`、静的および小次数統合検証は
`../paper_reproductions/wang2022_fig8/validation/validate.py`、条件と数式は
`../paper_reproductions/wang2022_fig8/README_ja.md` にあります。
この例は正方格子C4vであり、三角格子用D6短縮は適用しません。

Peng–Zhang (2025) Fig. 2 のAg背景／空気環状開口／Ag中心粒子とPI基板は
`../paper_reproductions/peng2025/reproduce_square.py`、同じ三角格子を1サイトprimitiveと直交二サイト・
supercellで比較する例は `../paper_reproductions/peng2025/compare_hex_supercell.py`、検証と再現上の制限は
`../paper_reproductions/peng2025/README_ja.md` にあります。
