# 8材料・単体とPMMA複合：NV法の61波長スペクトルと平均値

追加ファイルは `run_metals_nv_spectra.py`。サーバーの既存プロジェクト直下へコピーします。ZIPは作成しません。
既存の `run_transition_metals_nv.py`、`run_cu_ag_al_motheye_order_nv.py`、`run_pmma_cr_motheye_order_nv.py` とその依存ファイルを使用します。
Cr単体にも両構造に対応した既存のCrランナーを使います。ソルバーの計算式は変更しません。

```sh
cd /home/7/uq06557/common2/20260813/tsoyutarcwa
python -u run_metals_nv_spectra.py \
  --device cuda:0 \
  --orders Cr:24,Fe:24,Ni:24,Ti:24,V:24,Al:36,Cu:36,W:36 \
  --structures both \
  --wavelengths 400:700:5 \
  --slices 60 --grid 512 \
  --output-root studies/metals_nv_spectra/results/Nz60_Q512_400-700_61pt
```

400〜700 nmを両端を含む5 nm刻み、61波長で計算します。8材料×2構造×61波長＝976条件です。
金属単体のみなら `--structures metal`、PMMA複合のみなら `--structures pmma`。
一部の材料だけ実行する場合は `--orders` に必要な材料・次数だけを列挙します。

## 計算条件

- NV/NVM、complex128、D6 E1 x入射成分、正入射x偏光、Lanczosなし、Redheffer接続。
- 周期200 nm、高さ500 nm、三角配列、Nz60、法線場格子Q512×512。
- 単体：先端半径5 nm、底面半径95 nm、半無限同一金属基板。
- PMMA複合：コア先端半径5 nm、底面半径65 nm、側面・上面・谷部金属膜30 nm、半無限無吸収PMMA基板。
- PMMAのNz60は形状部分の分割数。上面キャップを加えて61構造層、最上面高さ530 nm。
- 各CSVのn,kからε=(n+ik)^2を作り、複素εを波長で線形補間。
- WはWerner(W).csv。CrはJohnson.csv。他材料は既存の添付CSV。
- PMMAはSzczurowski.csv、k=0。400 nmは既存のlinear-to-400方針でnを線形外挿。

単体のTは半無限金属を通り抜けた遠方透過で0、A=1-Rはモスアイ＋金属基板の吸収合計です。
PMMA複合のTは無吸収PMMA基板への流入、A=1-R-Tは金属膜での吸収です。

## 平均値

主表示は波長に一様な帯域平均：

`mean_R_band = integral_400^700 R(lambda) d(lambda) / 300`

61点の間を台形積分します。A,Tも同様です。太陽光スペクトル等の重みは付けません。
別に61点の単純平均 `mean_R_sample = sum(R_i)/61` も保存します。
両端の重みが異なるため、帯域平均と単純平均は一般にわずかに異なります。
CSVには比率と百分率の両方を記録し、図には両平均を表示します。
全点が成功した場合、帯域平均・単純平均の両方で平均R+A+T=1を確認します。
単体ではモスアイ吸収・基板吸収・基板流入パワーの平均も保存します。

## 失敗・再開

各波長は既存の省メモリNVワーカーを別プロセスで実行します。計算間でGPUメモリを持ち越しません。
OOM・数値異常は記録し、他の波長・材料・構造を続行します。
波長は資源使用量に対して単調な増加軸ではないため、失敗後の長波長をまとめて省略しません。
欠測点は0で埋めず、補間せず、61点が揃うまでそのスペクトルの平均値は未確定とします。
図では欠測部分の線を切り、INCOMPLETEと表示します。
同じコマンドで、成功済み点を省き、未完了・失敗点だけを再試行できます。
設定・材料・ソースコードが変わった場合は別の出力ルートを指定します。
Ctrl+Cでは成功済み結果を保存して中断します。
全条件が揃えば終了コード0、失敗点が残れば全条件の処理後に終了コード1です。

## 保存先

出力ルート直下：

- `all_spectra.csv`：全成功条件のR/A/T、単体の吸収内訳、計算時間、ピークGPUメモリ等。
- `averages.csv`：16スペクトルの完了状態と帯域平均・単純平均。欠測ありは平均欄が空欄。
- `failed_cases.csv`：失敗条件と理由。OOMと通常エラーを区別。
- `batch_summary.json`：完了点数、完了スペクトル数、平均値。
- `<材料>_spectrum_NV.png/.svg`：各材料の図。上段単体、下段PMMA複合、左R・中央A・右T。各パネルに平均値を表示。
- `metal/cr/`、`pmma/cr/`等：`spectrum.csv`、`checkpoint.json`、`report.json`、`plan.json`、波長ごとのワーカー結果。

`--dry-run` は材料データ範囲と全設定を確認するだけでRCWA計算を行いません。
保存済み結果から平均値と図を再作成する場合：

```sh
python -u run_metals_nv_spectra.py --plot-only \
  --output-root studies/metals_nv_spectra/results/Nz60_Q512_400-700_61pt
```

描画に失敗しても計算結果・平均値は保存し、plot_error.txtへ理由を記録します。描画を省く場合は `--no-plots`。

検証：`python -m unittest validation.test_metals_nv_spectra -v`
