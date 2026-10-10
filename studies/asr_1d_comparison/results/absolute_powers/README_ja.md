前に紹介した隣接次数差の図の縦軸は、反射率・透過率・吸収率そのものではなく、保持次数を増やしたときの絶対変化量です。

各物理量Q、各偏光について、`100 × max_波長 |Q(M) − Q(M_previous)|` を表示しています。
波長は400,450,500,550,600,650,700 nmの7点。単位はパーセントポイント（pp）で、縦軸は対数です。
例えばM=48の点はM=40との比較です。Rが0.40から0.41になれば変化は1 ppです。
ここで「隣接」は計算した保持次数の一覧で隣り合うことを意味し、常にMとM−1の比較という意味ではありません。

有限Li参照との差の図（reference_difference / convergence_overview）の縦軸は、有限Li M96との差です。convergence_overviewはさらに複数の物理量について最大化しています。
一方、前回作成したtm_retained_order.pngは、既に実際の物理量を%で表示しています。

今回の図は、縦軸を入射電力に対する比としてのR/T/Aそのものにしました。0.5は50%、1は100%です。
横軸は最大保持次数M（保持するフーリエ成分数N=2M+1）。TE/TMを上下、R/T/Aを左右に分け、波長ごとに別ファイルにしています。
差分・参照差・波長平均・波長ごとの最大化は行っていません。

各波長に次の2種類があります。どちらも縦軸は同じR/T/Aの実値です。

- powers_波長nm.png / .svg：縦軸0〜1の全体図。
- powers_波長nm_zoom.png / .svg：縦軸の範囲を拡大した線形表示。小さなTでは科学表記の乗数に注意してください。

**全保持次数の比較：既存の内部4N計算**

M=2,4,8,12,16,24,32,40,48。G=0.001、正入射、Galerkin、CUDA complex128の保存結果を使用。
ASRの積分点数下限は192、実際の点数はMに応じて増加します。最新6N/Q4096の条件と区別してあります。
金格子は層分割420、金被覆PMMA格子はPMMA分割300・有限層301です。

| 波長 nm | PMMA＋金膜（全体） | PMMA＋金膜（拡大） | 金格子（全体） | 金格子（拡大） |
|---:|---|---|---|---|
| 400 | [図](pmma_gold_grating_1d/powers_400nm.png) | [図](pmma_gold_grating_1d/powers_400nm_zoom.png) | [図](gold_grating_1d/powers_400nm.png) | [図](gold_grating_1d/powers_400nm_zoom.png) |
| 450 | [図](pmma_gold_grating_1d/powers_450nm.png) | [図](pmma_gold_grating_1d/powers_450nm_zoom.png) | [図](gold_grating_1d/powers_450nm.png) | [図](gold_grating_1d/powers_450nm_zoom.png) |
| 500 | [図](pmma_gold_grating_1d/powers_500nm.png) | [図](pmma_gold_grating_1d/powers_500nm_zoom.png) | [図](gold_grating_1d/powers_500nm.png) | [図](gold_grating_1d/powers_500nm_zoom.png) |
| 550 | [図](pmma_gold_grating_1d/powers_550nm.png) | [図](pmma_gold_grating_1d/powers_550nm_zoom.png) | [図](gold_grating_1d/powers_550nm.png) | [図](gold_grating_1d/powers_550nm_zoom.png) |
| 600 | [図](pmma_gold_grating_1d/powers_600nm.png) | [図](pmma_gold_grating_1d/powers_600nm_zoom.png) | [図](gold_grating_1d/powers_600nm.png) | [図](gold_grating_1d/powers_600nm_zoom.png) |
| 650 | [図](pmma_gold_grating_1d/powers_650nm.png) | [図](pmma_gold_grating_1d/powers_650nm_zoom.png) | [図](gold_grating_1d/powers_650nm.png) | [図](gold_grating_1d/powers_650nm_zoom.png) |
| 700 | [図](pmma_gold_grating_1d/powers_700nm.png) | [図](pmma_gold_grating_1d/powers_700nm_zoom.png) | [図](gold_grating_1d/powers_700nm.png) | [図](gold_grating_1d/powers_700nm_zoom.png) |

**最新のPMMA：内部6N/7N、Q4096**

Liは既存M=2〜48と、固定次数監査で計算したM=64,88,96,120,128を統合しました。
重複するLi結果の一致、形状・材料・数値ソースの一致を確認しています。Liは解析的フーリエ係数なのでASRの積分点数には依存しません。
ASR 6N/7NはM=48,64の2点のみ。2点を結ぶ線は両点間の比較を見やすくするもので、未計算の次数の値を表すものではありません。
今回の内部6N/7N・Q4096の系列と、前の4N・積分点数下限192の系列を接続していません。

| 波長 nm | 全体 | 拡大 |
|---:|---|---|
| 650 | [図](pmma_latest_6N_7N/powers_650nm.png) | [図](pmma_latest_6N_7N/powers_650nm_zoom.png) |
| 700 | [図](pmma_latest_6N_7N/powers_700nm.png) | [図](pmma_latest_6N_7N/powers_700nm_zoom.png) |

**R/T/Aの物理的な定義**

PMMA＋金膜ではRは全反射電力、Tは損失のないPMMA基板へ入る全透過電力、A=1−R−Tです。
これは全回折次数の電力の和です。単独のゼロ次R₀/T₀を描いた図ではありません。

金格子は半無限の吸収性金基板上にあるため、遠方透過T_far=0、全吸収A_total=1−Rです。
金の基板界面へ入る電力P_subは、その後基板で吸収される電力であり、遠方透過とは別の物理量です。
前の金格子の隣接差図が描いたP_subとA_relief（格子部のみの吸収）は、今回のT_farとA_totalとは定義が異なります。
今回の金格子の図は、構造全体のR/T/Aを示します。

Aはポート電力の残差から求めた値です。体積吸収積分による独立な検証ではありません。
曲線が平らに見えることは、真の解への収束を単独では保証しません。

**再生成用Pythonファイル**

描画スクリプトは20260806のプロジェクト内の `studies/asr_1d_comparison/plot_absolute_powers.py` に反映しました。
プロジェクトのルートから次のコマンドで保存済み結果を図示できます。

```text
python -m studies.asr_1d_comparison.plot_absolute_powers
```

計算済みcheckpoint/planを読み込んで描画します。Matplotlibが必要です。RCWAの再計算やGPUの確保は不要です。
元の結果フォルダは変更していません。plotted_values.csvに576行の表示値、figure_manifest.jsonにデータの出所を保存しました。
