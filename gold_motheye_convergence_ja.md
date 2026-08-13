# 金モスアイ／金基板の収束判定

## 1. 現在の仮定

`converge_gold_motheye.py` は、まだ指定されていない形状条件を次の既定値で補っている。

| 項目 | 既定値 |
|---|---:|
| 配列 | 60°三角 Bravais 格子 |
| 周期 | 200 nm |
| モスアイ高さ | 500 nm |
| 形状 | 空気側が細い金円錐台（円柱スライス近似） |
| tip半径 | 5 nm |
| base半径 | 95 nm（隣接base間gap 10 nm） |
| 半径profile | 線形、`profile_power=1` |
| 入射 | 空気、垂直入射、x偏光 |
| 基板 | 半無限金 |
| 金分散 | Rakić Lorentz–Drude bulk-Au model |
| matched-ASR写像 | `circle_G=0.03`（CLIで変更可） |

半径は空気側から基板側へ

\[
r(u)=r_{\rm tip}+(r_{\rm base}-r_{\rm tip})u^p,
\qquad 0\le u\le1
\]

と増加し、各高さスライスの中点で評価する。`p=1` が円錐、`p>1` は先端側が細い時間が
長いprofile、`p<1` は早く太くなるprofileである。真の尖端 \(r=0\) はmatched mapが退化
するため、最小の正半径を用いる。

## 2. ASRによる離散化

各スライスは「空気背景中のセル中心金円柱」とし、`method="matched-asr"` を用いる。
円境界に座標線を一致させ、各スライスをCartesian S行列へ変換してRedhefferまたはLi-2aで
接続する。三角格子・正入射・中心円なので、既定ではD6-closed reciprocal star上のx偏光
Cs sectorだけを解く。

高さ方向の階段近似誤差、Fourier打切り誤差、ASR material tensorの数値Fourier積分誤差は
独立でない。このため次の3軸を交互に更新する。

1. 高さスライス数 `Nz = 12,16,24,32,48`
2. Fourier次数 `M = 3,4,5,6,7`
3. ASR sampling grid `96,128,192,256`

収束用波長は既定で400、550、700 nmである。ある候補と直前候補の最大絶対変化を、全波長と
全収束量について計算する。連続する2回のrefinementがとも `0.005` 以下、すなわち0.5
percentage point以下になった最小の中間候補を採用する。3軸の推奨値が変化しなくなるまで
最大3 cycle反復する。候補上限でも条件を満たさない場合は
`candidate_range_insufficient` とし、最大値を最終解と誤認しない。

## 3. 半無限金基板におけるR/T/A

入射、反射、基板へ進む場の単位セル平均Poynting fluxを

\[
P_z=\frac12\operatorname{Re}\sum_{mn}
\left(E_{x,mn}H_{y,mn}^*-E_{y,mn}H_{x,mn}^*\right)
\]

から計算する。半無限の損失金には裏面の透明portがないため、通常の遠方透過率は

\[
T_{\rm far}=0
\]

である。一方、モスアイ層と金基板の吸収を区別するため、構造／基板界面を通過するfluxを
\(P_{\rm sub}\) として保存する。

\[
R=-P_{\rm refl}/P_{\rm inc},
\]

\[
A_{\rm moth}=1-R-P_{\rm sub}/P_{\rm inc},
\qquad
A_{\rm sub}=P_{\rm sub}/P_{\rm inc},
\]

\[
A_{\rm total}=A_{\rm moth}+A_{\rm sub}=1-R.
\]

したがって半無限金で出力される `power_into_substrate` は遠方透過率ではなく、最終的に基板内で
吸収される割合である。有限金膜を選んだ場合は、裏面媒質へ出る通常の
\(T\) と \(A=1-R-T\) を返す。

## 4. 金の分散

既定モデルはRakićらのplasma energyとLorentz–Drude係数をそのまま実装する。solverの
\(\exp(-i\omega t)\) 規約に合わせ、passive goldが
\(\operatorname{Im}\epsilon>0\) になる符号を使用する。[Rakić et al., Applied Optics 37,
5271 (1998)](https://doi.org/10.1364/AO.37.005271)

Rakić論文自身が、金の低エネルギーinterband transition付近ではLorentz–Drudeより
Brendel–Bormann fitの方がよいと述べている。また薄膜の光学定数は成膜法、粗さ、粒径に依存
する。最終的な実験比較では、対象試料に近い測定 `n,k` CSVを渡すことを推奨する。古典的な
測定値の出典は [Johnson and Christy, Physical Review B 6, 4370
(1972)](https://doi.org/10.1103/PhysRevB.6.4370) である。

CSVは次のいずれかのheaderを持つ。

```text
wavelength_nm,n,k
400,...,...
...
```

または

```text
wavelength_nm,epsilon_real,epsilon_imag
400,...,...
...
```

範囲外外挿は行わず例外にする。

## 5. 実行方法

成果物ディレクトリで実行する。

```bash
python converge_gold_motheye.py --device cuda
```

測定CSVを使う例:

```bash
python converge_gold_motheye.py \
  --gold-model csv --gold-csv au_measured_nk.csv \
  --device cuda --output-prefix au_motheye_measured
```

推奨構成決定後に400–700 nmを5 nm間隔で同時に計算する例:

```bash
python converge_gold_motheye.py \
  --run-final-spectrum --spectrum-wavelengths 400:700:5 \
  --device cuda
```

主な出力:

- `gold_motheye_convergence.json`: 3軸収束履歴と推奨値
- `gold_motheye_anchor_spectrum.csv`: 推奨値でのanchor波長R/T/A
- `gold_motheye_all_cases.csv`: 実行済み全case
- `gold_motheye_checkpoint.json`: 中断再開用cache
- `gold_motheye_spectrum.csv/json`: `--run-final-spectrum` 指定時

同じprefixのcheckpointと物理条件が一致しない場合は、古い結果を混ぜず停止する。

分散・CSV補間だけの独立検証と、最小matched-ASR積層を含む統合検証は次で実行する。

```bash
python validate_gold_motheye_setup.py
python validate_gold_motheye_setup.py --integration --device cuda
```

## 6. 決定が必要な条件

最終スペクトルの前に、少なくとも次を確定する必要がある。

1. 金の突起か、金基板に形成した穴か。
2. 三角配列か正方配列か。
3. tip半径、base半径、側面profile。SEM断面があればそのprofileを使う。
4. 金基板が光学的に半無限か、有限膜か。有限なら膜厚と裏面媒質。
5. 金の成膜条件に対応する `n,k` データ。なければRakić bulk fitを使用する。
6. 必要な収束許容差。既定0.005は探索用で、最終値には0.001程度を推奨する。

周期200 nmは400–700 nmより短いため、空気側の遠方回折次数はゼロ次だけである。ただし金属
境界近傍のevanescent Fourier成分は吸収と場分布へ寄与するため、Fourier次数の収束確認は
省略できない。
