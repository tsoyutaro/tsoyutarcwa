# Au・700 nmの高さ依存診断

```bash
python -u run_au_nv_height_diagnostics.py \
  --device cuda:0 \
  --orders Au:36 \
  --structures both \
  --wavelengths 700 \
  --slices 210 --grids 512 \
  --heights-nm 500,400,300,200 \
  --reference-root studies/metals_nv_diagnostics/results/Au_M36_Nz210_Q512_3wl2 \
  --output-root studies/au_nv_height_diagnostics/results/M36_Nz210_Q512_700nm_H500-200
```

高さ4条件×単体・PMMA被覆の両構造で8条件を診断する。元のプロジェクト直下にこの新しい実行ファイルを置き、既存の`run_metals_nv_diagnostics.py`と依存ファイルをそのまま使う。既存ソルバー・診断ファイルの変更は不要。

## 固定する条件

- 波長700 nm、M=36、Nz=210、Q=512。NV/D6 E1 x、垂直入射、complex128、Redheffer、Lanczosなし。
- 周期200 nm。金属単体の先端半径5 nm・底面半径95 nm。
- PMMAの先端半径5 nm・底面半径65 nm。Auの側面・上部・谷部の膜厚はすべて30 nmのまま維持する。
- Auは既存の測定n/kデータ、PMMAは既存のSzczurowskiデータを使う。

`--heights-nm`はモスアイのプロファイル高さである。PMMA被覆では上部の金属キャップ30 nmを別に加えるため、全パターン高さは530/430/330/230 nm、全パターン層数は211層となる。単体の全パターン層数は210層。

高さを下げると実際の形状と光学特性が変わる。この比較は、500 nm構造の収束試験ではない。金属単体では同じ層番号の断面半径は全高さで共通。PMMAでは谷部の実膜厚30 nmを固定するため、その割合とプロファイルの層配分は高さに応じて変わる。

## 既存結果の再利用

`--reference-root`の元の診断結果と、材料・形状・M/Nz/Q・波長・数値精度・診断設定・ソースハッシュ・ファイル一式が一致する場合だけ再利用する。指定例では高さ500 nmの両構造が候補となり、互換なら新しい計算は残りの6条件となる。負の層吸収がある完成結果も比較の基準として保持する。保存元は変更せず、保存先のreport.jsonに出典と元レポートのハッシュを記録する。

条件やソースが一致しない場合は自動で再計算する。すべて再計算したい場合は`--no-reuse-reference`を追加する。

各条件は独立プロセスで実行し、OOM・例外・非物理値でも残りの高さ・構造を続ける。同じコマンドで完了済み条件を再利用できる。失敗済みを再試行する場合は`--retry-failures`を追加する。高さを下げてもMとNzが同じなら必要な行列サイズはほぼ変わらないため、以前M=36で成功したGPUメモリ容量を使う。

## 出力

- `height_summary.csv` / `height_summary.json`: 高さ別のR/T/A、負の層数、最小層吸収、負の損失固有値成分への寄与、整合性、再利用元。
- `height_RTA_*.png/.svg`: 高さに対するR/T/Aの比較。
- `height_passivity_*.png/.svg`: 負の層数・最小層吸収・負の損失固有値成分への寄与の比較。
- `H500/metal/au/...`等の各条件フォルダ: 従来と同じ`report.json`、`far_field.json`、`layer_power.csv`、`operator_diagnostics.csv`、`excited_modes.csv`、`mode_excitation_map.npz`、層別の図。

`physical_result`と`diagnostic_checks_passed`を別々に見る。`sample_checks_passed`は厳しい負の固有値成分への寄与の条件も含むため、FalseというだけではR/T/Aの誤差を示さない。実際の負の層吸収が消えた場合も、新しい高さでの次数・層数収束は別途確認する。

コマンドに`--dry-run`を追加すると、材料・形状・条件数を確認し、計算・ファイル作成を行わない。`--plot-only`はこの高さ掃引の保存結果から図だけを作成する。
