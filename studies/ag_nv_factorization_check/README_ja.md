# Ag M=14のNV因数分解を切り分ける比較

```bash
python -u run_ag_nv_factorization_check.py \
  --device cuda:0 \
  --order 14 --wavelength-nm 550 \
  --slices 210 --grids 512 \
  --factorizations symmetric-nv,galerkin \
  --reference-root studies/ag_nv_m14_diagnostics/results/M14_Nz210_Q384-1024_550nm \
  --output-root studies/ag_nv_factorization_check/results/M14_Nz210_Q512_550nm
```

この比較用ファイルは、既存の `run_ag_nv_m14_diagnostics.py` と同じフォルダに配置する。`rcwa_ext`や既存の診断ファイルは編集しない。必要なのは新しい `run_ag_nv_factorization_check.py` の追加だけ。

## 比較の目的

Ag単体・550 nm・M=14・210層・Q=512で、NVの横方向誘電率補正を外すと非物理的な増幅が消えるかを調べる。形状、複素誘電率、波長、偏光、フーリエ基底、D6縮約、層数、Redheffer計算、内部場の復元、層別パワーの検査は共通。

| 名前 | 横方向の誘電率行列 |
|---|---|
| `symmetric-nv` | 従来の対称形NV：ε_ab=δ_ab A+(Δ P_ab+P_ab Δ)/2、A=[ε]、Δ=[1/ε]⁻¹−[ε] |
| `galerkin` | 通常のフーリエ/Galerkin法：ε_xy=diag(A,A)。NV補正を使わない |

z方向の[ε]とその逆行列の扱いは両方で同じ。材料の虚部を増やす、負の固有値をクリップする、モードを除去するなどの補正は行わない。

通常のフーリエ法は、厳密なToeplitz行列の損失が半正定値となる比較対象である。これは有限次数での受動性の切り分けに役立つ。一方、金属の曲面境界でNV法と同じ次数の精度を保証するものではない。通常フーリエ法が物理範囲の値を返しても、その数値を最終解として採用したり、NV法の収束確認の代わりにしたりしない。

通常フーリエ法とNV法の収束速度の違い、およびNV因数分解の参考文献：

- *Normal vector method for the RCWA with automated vector field generation*, Optics Express 16, 17295 (2008): https://doi.org/10.1364/OE.16.017295
- *Accurate near-field calculation in the rigorous coupled-wave analysis method*, Eq. (6): https://arxiv.org/pdf/1507.06364

## 既存結果と継続処理

指定したreference-rootに条件・材料ハッシュ・ソースハッシュが一致するNVの診断結果があればコピーして再利用し、通常フーリエ法だけを新しく計算する。コピー先のreport.jsonには元の設定と保存元パスを残す。古い結果は書き換えない。一致しなければNVも新たに計算する。

`--no-reuse-reference` を付けると両方を再計算する。各方式・Qは別プロセスで計算し、非物理値も保存する。OOMや例外が起きても残りのケースを続ける。同じコマンドで完了済みの比較結果を再利用でき、失敗を再試行するには `--retry-failures` を付ける。

## 出力と判定

- `comparison.csv`：各方式のR、全A、モスアイA、基板A、負の吸収層数、物理範囲と内部場整合性の判定。
- `symmetric-nv/Q512/`：NVの従来結果と層別・モード診断。
- `galerkin/Q512/`：通常フーリエ法による同じ診断。
- `figures/factorization_Q512.png` / `.svg`：反射率・吸収率・負の吸収層数の比較。
- `figures/layer_comparison_Q512.png` / `.svg`：深さ方向のパワー・層吸収・損失固有値の比較。桁の大きく異なる正負の値があるため、縦軸は符号付き対数表示。

通常フーリエ法で負の吸収層がなくなり、境界・体積積分の整合性も保たれるなら、NVの横方向補正が増幅を生む要因であることを支持する。両方で異常が残るなら、共通の演算子組立や層接続も点検する。いずれの場合も、次に採用する方式の次数・層数収束は別途確認する。

## 検証

小規模CPU計算で、現行NVに負の層吸収が残る例を再現し、通常フーリエ法では全層の吸収が非負となること、既存の半S行列と内部場の結果が一致すること、層のパワー差と損失体積積分が一致することを確認した。実材料の複素誘電率の保持、実行後のオーバーライド解除、元のソースファイルの不変性、結果再利用時のハッシュ照合、OOM後の継続も自動テストで確認する。本番のM=14・210層の新しい比較計算は未実行。
