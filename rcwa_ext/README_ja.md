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
| `nvm` | 対応 | 対応 | 直交C2v、三角D6-closed star/Cs |
| `matched-asr` | 対応 | 対応 | 直交C2v、三角D6-closed star/Cs |
| `standard` hard raster | 固定半径のみ | 固定半径のみ | 非対応 |

三角NVMとmatched-ASRのx/y短縮は実装済みです。一般斜交格子ではx/yを異なるsectorへ
分けられませんが、両者が共有するC2 source sectorだけを解く短縮に対応します。未実装
なのは、NVM射影行列とmatched-coordinate tensorを二重適用する `matched-nvm` です。
native D6-star全体の `A1/A2/B1/B2/E1/E2` 完全isotypic分解は、三角NVMと
matched-ASRで実装済みです。Redheffer／Li-2a、full／half／quarter公開S、
`external`／`internal`／`all` の6成分場再構成を組み合わせられます。partial公開Sでも
場を指定した場合は、場専用の両方向full Sと各層modal couplingを非公開で保持します。

推奨 import:

```python
from rcwa_ext import AutoRCWA, Circle, LayerSpec, Material
```

従来の import も、親ディレクトリの互換 facade により動作します。

```python
from rcwa_solver_auto import AutoRCWA, Circle, LayerSpec, Material
```

各ファイルの数式、導出、依存関係、変更時の検証項目は
`../rcwa_modular_math_guide_ja.md` にまとめています。三角NVMのD6-star/Cs偏光短縮は
`../triangular_nvm_polarization_report_ja.md`、native-star完全D6の全指標射影と計算量は
`../complete_d6_native_star_report_ja.md` を参照してください。

応用例として、金円錐モスアイのmatched-ASRスライス数・Fourier次数・sampling gridを
収束させる `../converge_gold_motheye.py`、金分散の `../gold_dispersion.py`、導出と実行手順の
`../gold_motheye_convergence_ja.md` があります。
