# Ag の NV 次数走査の補完と Galerkin 比較

`run_ag_nv_missing_orders.py` をリポジトリ直下に配置して実行する。
既存の `run_cu_ag_al_motheye_order_nv.py` のワーカーをそのまま利用し、NV の式、材料データ、complex128、D6、メモリ節約処理は変更しない。
新しいスクリプト以外の既存ファイルを上書きする必要はない。

## NV と Galerkin を両方実行

```sh
python -u run_ag_nv_missing_orders.py \
  --device cuda:0 \
  --factorizations symmetric-nv,galerkin \
  --structures both \
  --wavelengths 400,550,700 \
  --slices 210 --grid 512 \
  --start-order 4 --step 2 --max-order 36 \
  --source-root studies/au_ag_nv_order_slices/results/orders_only_Nz210_M4-36_Q512_3wl \
  --output-root studies/ag_nv_order_completion/results/Nz210_Q512_M4-36_3wl
```

手法の指定を省略すると従来どおりNVのみ。`--factorizations galerkin` でGalerkinだけを選べる。
NV側の保存形式・条件署名は従来版から維持しているため、前のNV補完コマンドで計算済みでも、同じ出力先にGalerkinを追加できる。
Galerkinは出力先内の `galerkin/` 以下に保存し、NVの値をGalerkinの結果として流用しない。
既存の単一点診断ディレクトリにあるGalerkin結果は、この走査には自動取り込みしない。
手元の元データだけから始める場合、新規予定はNV40条件、Galerkin102条件で合計142条件（実行前に `--dry-run` で確認可能）。

Galerkin側は既存の `run_ag_nv_factorization_check.py` の `factorization_scope("galerkin")` を使用する。
横方向の誘電率作用素を `diag(T(epsilon), T(epsilon))` に置き換える通常のフーリエ畳み込みの比較計算。
NVの横方向補正を適用せず、同じ形状、材料、epsilon_zz、D6、complex128、Redheffer、メモリ節約処理を使う。
界面法線方向にNV/Liの逆則補正を適用する別方式を追加したわけではない。
両手法の結果が有限次数で一致するとは限らず、値の物理範囲だけで収束や正確さは判定しない。

手法ごとに独立したプロセスで順次計算する。一方でOOMになっても、もう一方の手法は続行する。

## 実行

```sh
python -u run_ag_nv_missing_orders.py \
  --device cuda:0 \
  --structures both \
  --wavelengths 400,550,700 \
  --slices 210 --grid 512 \
  --start-order 4 --step 2 --max-order 36 \
  --source-root studies/au_ag_nv_order_slices/results/orders_only_Nz210_M4-36_Q512_3wl \
  --output-root studies/ag_nv_order_completion/results/Nz210_Q512_M4-36_3wl
```

末尾に `--dry-run` を付けると、再利用条件と実行予定条件を表示する。計算・ファイル作成は行わない。
条件は周期200 nm、高さ500 nm、垂直入射、x偏光。金属単体は半無限Ag基板上のAg構造。
PMMA+AgはPMMA構造に側面・上部・谷部各30 nmのAg膜、PMMA基板。
PMMA構造では210分割に上部キャップ層が加わる。

2026-10-22に確認した既存結果では、全102条件のうち61条件が正常終了、1条件が非物理値として記録済み。未計算は次の40条件。

| 構造 | 波長 (nm) | 補完する次数 M | 条件数 |
|---|---:|---|---:|
| Ag単体 | 550 | 16,18,…,34 | 10 |
| Ag単体 | 700 | 12,14,…,36 | 13 |
| PMMA+Ag | 700 | 4,6,…,36 | 17 |

400 nmの両構造、および550 nmのPMMA+Agは既存結果を再利用する。
Ag単体・550 nmのM36も保存済み。

## 続行と再開

- Ag単体・550 nm・M14の既存結果（A約−12.50%）は `nonphysical` として残す。既定では再実行しない。
- 新規の非物理値もそのまま記録し、次の次数へ進む。負値をゼロに補正しない。
- CUDA OOM時は、その手法・構造・波長での高次数側だけを省略する。他の手法・構造・波長は続行する。
- その他の計算エラーも記録して次の条件へ進む。
- 同じコマンドで再開すると保存済み条件を再利用する。`--retry-failures` を付けた場合は、非物理値・失敗・OOM省略条件を再試行する。
- 各条件を独立したプロセスで実行し、各試行に別の結果ファイルを使う。途中終了時に古い試行結果を読み間違えない。
- 元の結果ディレクトリは読み取り専用で参照する。材料、ソルバーソース、固定条件が一致するデータだけを再利用する。Linux/Windows間の材料ファイル絶対パス差と走査範囲の差は内容ハッシュで照合する。
- 新しい出力先の保存条件やソルバーが変わった場合は別の `--output-root` を指定する。

## 出力

- `all_cases.csv`：R/T/A (%)、手法、構造、波長、次数、状態。非物理値の行も含む。
- `summary.json`：正常終了、非物理値、OOM、省略などの件数。
- 構造・波長・次数別の `report.json`：結果、材料・ソルバーのハッシュ、既存結果の参照元など。
- `Ag_reflectance_vs_order.png` / `.svg`：反射率の次数依存性。
- `Ag_transmittance_vs_order.png` / `.svg`：透過率の次数依存性。
- `Ag_absorptance_vs_order.png` / `.svg`：吸収率の次数依存性。

図はNVを実線、Galerkinを破線で重ねて表示する。欠測部分は線を切り、非物理値も表示する。Xマークは使わない。

全条件の計算を試みても、非物理値などが残る場合の終了コードは2。既知のM14非物理値を再利用した場合も2となる。
計算の完了は収束や内部電磁場の精度を保証しない。
ZIPは作成しない。

## 検証

```sh
python -m unittest validation.test_ag_nv_missing_orders
```
