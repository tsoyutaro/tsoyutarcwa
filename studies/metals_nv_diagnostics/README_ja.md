# 金属単体・PMMA被覆のNV受動性と層別パワー診断

## 最初の確認

既存スペクトルに採用した設定で、8材料・両構造・3波長の計48条件を診断する。

```bash
python -u run_metals_nv_diagnostics.py \
  --device cuda:0 \
  --orders Cr:24,Fe:24,Ni:24,Ti:24,V:24,Al:36,Cu:36,W:36 \
  --structures both \
  --wavelengths 400,550,700 \
  --slices 60 --grids 512 \
  --output-root studies/metals_nv_diagnostics/results/adopted_Nz60_Q512_3wl
```

実行前に同じ指定へ `--dry-run` を追加すると、材料データと48条件の設定を確認し、計算もファイル作成も行わない。

このファイルと既存の `run_ag_nv_m14_diagnostics.py`、`run_ag_nv_factorization_check.py`、`run_metals_nv_spectra.py`、各材料の実行ファイル、`studies`、`rcwa_ext` を同じプロジェクト内で使う。既存のソルバー・診断ファイルは変更していない。

## 共通条件と検査内容

- 既存の材料別実行ファイルを直接呼び出すため、形状・材料補間・層の組立は既存スペクトルと同じ。
- 周期200 nm、高さ500 nm、三角格子、垂直入射、x偏光、complex128、NV/D6 E1、Redheffer、Lanczosなし。
- 単体は先端半径5 nm、底面半径95 nm、同じ金属の半無限基板。
- PMMAは先端半径5 nm、底面半径65 nm、側面・上部・谷部の金属膜厚30 nm。既存の部分谷部スライスを維持し、上部キャップ1層を加える。`Nz=60`なら全パターン61層。
- Wは添付の `Werner(W).csv`、Auは既存の測定n/k CSV。PMMAの400 nmは既存の `linear-to-400` 方針を維持。
- 内部場を減衰波のみで復元し、層上下面のパワー差と、実際に離散化された損失演算子による独立の体積積分を比較する。
- 負の層吸収、負の損失固有値成分への励起、モード振幅、損失固有値、境界残差、元の遠方場との一致を保存する。数値のクリップ、材料損失の変更、モードの除去は行わない。
- 全空間の層行列は従来の省メモリ処理で解放し、追加の損失行列はCPUへ保存する。ただし内部診断は通常スペクトルより追加の時間・メモリを必要とし、高次数のGPU実行成功を保証しない。

各条件を独立プロセスで順番に実行する。OOM・例外・非物理値でも、他の波長・構造・材料へ進む。完了済みは同じコマンドで再利用する。失敗済みを再試行する場合だけ `--retry-failures` を付ける。材料・ソース・物理条件が変わった保存結果は再利用しない。材料や検査する次数・層数・Qの追加は、互換な既存条件を再利用できる。

## 出力と判定

`diagnostics_summary.csv` と `batch_summary.json` に集計する。

- `physical_result`: 遠方場とパターン全体の値が物理範囲内で、各層吸収が -1e-8 以上。
- `diagnostic_checks_passed`: 境界残差 <=1e-7、元の遠方場とのパワー差 <=1e-8、層和の収支誤差 <=1e-7、独立した体積積分との最大差 <=1e-7。これは計算の整合性であり、受動性を意味しない。
- `excited_negative_loss_passed`: 実際の入射場が励起した負の損失固有値成分の寄与の絶対値が、入射パワー比1e-6以下。`--negative-loss-tolerance`で集計基準を変更できる。1e-6は0.0001パーセントポイントである。
- `sample_checks_passed`: 上の3条件を全て満たす。すべての可能な入射場に対する損失演算子の半正定値性や、次数・層数・Qの収束を保証するものではない。
- `negative_loss_sector_pp`: 負の損失固有値成分からの、符号を残した体積積分の総和。層全体の吸収が正でも、内部で負の寄与が大きい場合は別に検出する。

PMMA構造では基板へ出たパワーを透過率Tとして扱い、基板吸収は0。全吸収Aは金属被覆の吸収である。金属単体では遠方場T=0、全吸収Aはパターン吸収と半無限金属基板への流入の和。

各条件の保存先例:

```text
metal/cr/symmetric-nv/M24_Nz60_Q512/wl550/
pmma/cr/symmetric-nv/M24_Nz60_Q512/wl550/
```

`report.json`、`far_field.json`、`layer_power.csv`、`operator_diagnostics.csv`、`excited_modes.csv`、`mode_excitation_map.npz`、`layer_diagnostics.png/.svg`を出力する。図は簡潔な英語ラベルで、パワー流・層吸収・負の損失成分への励起を横3列で表示する。モード振幅の指標は干渉を含むため、足して全パワーになる量ではない。

## 層数・Q・次数の追加検証

採用設定の受動性検査だけでは収束の検証にならない。例えば同じ出力先で層数とQを追加すると、既存条件を再利用できる。

```bash
python -u run_metals_nv_diagnostics.py \
  --device cuda:0 \
  --orders Cr:24,Fe:24,Ni:24,Ti:24,V:24,Al:36,Cu:36,W:36 \
  --structures both --wavelengths 400,550,700 \
  --slices 60,90 --grids 512,768 \
  --tolerance 0.005 \
  --output-root studies/metals_nv_diagnostics/results/adopted_Nz60_Q512_3wl
```

`refinement_changes.csv`で、他の設定を固定したNz・Qの隣接比較を出力する。R/T/Aに加えてパターンと基板の吸収配分も比較する。`--tolerance 0.005`は0.5パーセントポイントで、比較した両条件の`sample_checks_passed`も合格して初めて`refinement_check_passed=True`になる。1段階の変化が小さいだけで完全収束とは呼ばない。

次数を追加する場合は `--order-offsets=-2,0,2` のように指定する。M=36の上限近くではOOMとなる可能性があるが、他の条件は続ける。最大次数を超えず既存末尾を調べるなら `--order-offsets=-4,-2,0` でM=32,34,36になる。

Au/Agを優先して調べる例:

```bash
python -u run_metals_nv_diagnostics.py \
  --device cuda:0 --orders Au:36,Ag:36 \
  --structures both --wavelengths 400,550,700 \
  --slices 210 --grids 512 \
  --output-root studies/metals_nv_diagnostics/results/AuAg_M36_Nz210_Q512_3wl
```

特定条件で通常フーリエ/Galerkinとの比較も行う場合は `--factorizations symmetric-nv,galerkin` を追加する。両方式で同じMの一致を収束の判定に使わず、各方式の収束を別々に確認する。
