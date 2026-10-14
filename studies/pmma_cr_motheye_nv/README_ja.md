# PMMA＋CrのNV法による回折次数掃引

既存のPMMA＋CrのASR計算と同じ物理形状・光学定数で、NV法の次数依存を確認するランナーです。
今回追加したrun_pmma_cr_motheye_order_nv.pyとplot_pmma_cr_motheye_order_nv.pyをサーバーのプロジェクト直下へコピーしてください。
既に導入した5金属用NVのstudies/transition_metals_nv/runtime.py、既存のrcwa_ext、PMMAモデルを利用します。
ZIPは作成していません。

```sh
cd /home/7/uq06557/common2/20260813/tsoyutarcwa

python -u run_pmma_cr_motheye_order_nv.py \
  --device cuda:0 --structures pmma \
  --slices 60 --grid 512 \
  --start-order 4 --step 2 --max-order 36 \
  --wavelengths 400,550,700 \
  --tolerance 0.0005 \
  --output-root studies/pmma_cr_motheye_nv/results/Nz60_Q512_M4-36
```

M=4,6,…,36の17次数×3波長、計51ケースです。NzとNV法線場格子Qは固定します。
計算前に入力だけを確認するなら上のPythonコマンドに--dry-runを追加してください。
上限なしで次数を増やすなら--max-order 0です。OOM、数値異常、または固定格子の制限で止まります。
成功した波長は直ちに保存し、同じコマンドで保存済みケースを飛ばして再開します。
最大次数だけを延長する場合も同じ出力先を使えます。材料・ソルバー・固定条件を変更したら別の出力先を使ってください。

固定条件は周期200 nm、高さ500 nm、三角格子、垂直入射、x偏光、空気入射、半無限の無吸収PMMA基板です。
PMMAコアの先端半径5 nm、底面半径65 nm、Cr側面膜厚30 nm、頂部と谷部の膜厚も30 nmです。
Nz=60はPMMAの高さ500 nmを60分割する数です。頂部キャップ1層を加えて計61構造層になります。
Crはstudies/cr_motheye/data/Johnson.csv、PMMAはstudies/pmma_cr_motheye/data/Szczurowski.csvです。
--cr-csv、--pmma-csvで別データを指定できます。PMMAの400 nmのnのみ従来と同じlinear-to-400で外挿します。

円形・同心コアシェル用NV因子分解、D6 E1のx入射成分、complex128、Lanczosなし、Redheffer結合を使います。
縮小モードの取得後に不要な層行列を解放し、層S行列を1枚ずつ結合します。各ケースを別プロセスで実行します。

出力ルートのpmma/cr/all_cases.csvにR・T・A、次数、実際の層数、実行時間、GPUピークメモリ等を保存します。
PMMAは無吸収なので、A=1−R−TはCr膜の吸収率、TはPMMA基板へ入るパワーです。
checkpoint.json・report.json・plan.json・profile_layers.csvも同じフォルダに保存します。
CSVのパワーは割合です。図では100倍して%に変換します。

許容差0.0005は0.05パーセントポイントです。全3波長について最後の2回の次数差がR/T/Aのすべてで許容差以内かを調べます。
有限の指定範囲を完了した場合のみpassed_at_fixed_Nz_Qを判定します。途中停止した計算には判定通過を付けません。
これは固定Nz・Qでの次数依存の確認であり、層数・法線場格子や全波長スペクトルの収束は別に確認してください。

作図コマンド：

```sh
python plot_pmma_cr_motheye_order_nv.py \
  --output-root studies/pmma_cr_motheye_nv/results/Nz60_Q512_M4-36
```

figures/PMMA_Cr_R_A_T_vs_order.png/.svgにR・A・Tの横3列の図を保存します。
figures/pmma_order_changes.png/.svgには、隣接次数の全波長での最大変化と許容差を表示します。

ローカルCPUでNz=60、M=2、Q=32の3波長を計算し、省メモリ化前後のR/T/Aの一致を検証しました。
既存ASR計算の形状・CrとPMMAの材料ハッシュの一致も確認しました。
これは実装の確認であり、GPUの高次数での収束結果ではありません。
