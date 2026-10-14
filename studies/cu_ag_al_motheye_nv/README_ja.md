# Cu・Ag・Al：省メモリNV法による回折次数掃引

今回の添付CSVで、金属単体とPMMAモスアイに金属を蒸着した構造を計算します。ZIPは作成していません。

## サーバーへの配置

run_cu_ag_al_motheye_order_nv.pyとplot_cu_ag_al_motheye_order_nv.pyを、既存tsoyutarcwaのプロジェクト直下にコピーしてください。
以前の5金属用NVのstudies/transition_metals_nv/runtime.pyと、既存のrcwa_ext・PMMAモデルを利用します。
CSVは以下の場所に置きます。Alのファイル名は空白と括弧を含めてそのままです。

```text
studies/cu_ag_al_motheye_nv/data/Cu.csv
studies/cu_ag_al_motheye_nv/data/Ag.csv
studies/cu_ag_al_motheye_nv/data/Rakic (Al).csv
```

計算用コマンド：

```sh
cd /home/7/uq06557/common2/20260813/tsoyutarcwa
mkdir -p studies/cu_ag_al_motheye_nv/data
```

このフォルダに添付の3つのCSVをコピーした後、実行してください。

```sh
python -u run_cu_ag_al_motheye_order_nv.py \
  --device cuda:0 \
  --materials Cu,Ag,Al --structures both \
  --slices 60 --grid 512 \
  --start-order 4 --step 2 --max-order 36 \
  --wavelengths 400,550,700 \
  --tolerance 0.0005 \
  --output-root studies/cu_ag_al_motheye_nv/results/Nz60_Q512_M4-36
```

M=4,6,…,36の17次数×3波長×3金属×2構造なので、すべて成功すれば306ケースです。
実行順はCu単体、PMMA＋Cu、Ag単体、PMMA＋Ag、Al単体、PMMA＋Alです。
入力条件だけを確認するなら上のPythonコマンドに--dry-runを追加します。
--materials Ag --structures pmmaのように、金属と構造を絞ることもできます。
別のCSV保存フォルダを使う場合は--data-dir path/to/dataを指定してください。

上限なしで増やすなら--max-order 0です。OOM、数値異常、または固定格子の限界でその掃引を止め、次の構造へ進みます。
成功した波長は直ちに保存します。同じコマンドで保存済みケースを飛ばして再開できます。
最大次数だけを大きくして延長する場合も同じ出力先を使えます。
材料・ソルバー・層数・格子等が変わった場合は、別の出力先を指定してください。

## データと固定条件

添付のwlはµmとしてnmへ変換します。nとkの波長列の一致を検査し、ε=(n+ik)²を作ってεを波長について線形補間します。
CuとAgは187.9–1937 nm、Alは0.12399–200000 nmを含むため、400・550・700 nmは金属の外挿なしで計算できます。
データのコピーは添付の内容とバイト単位で一致し、data/manifest.jsonに元ファイル・ハッシュ・波長範囲・3波長のεを記録しました。
Alは添付のRakic (Al).csvの表を使います。このランナー内で別の解析的Rakićモデルに置き換える処理はありません。

周期200 nm、高さ500 nm、三角格子、先端半径5 nm、半径は深さに対して直線、垂直入射・x偏光です。
単体では底面半径95 nm、半無限の同じ金属基板を使います。
PMMA側はコア底面半径65 nm、側面・頂部・谷部の金属膜厚が各30 nm、半無限の無吸収PMMA基板です。
Nz=60は高さ500 nmの60分割数で、PMMA側は頂部キャップを加えて61構造層になります。
PMMAのCSVは既存studies/pmma_cr_motheye/data/Szczurowski.csvです。--pmma-csvで変更できます。
PMMAのnだけは400 nmで従来のlinear-to-400の外挿を使います。PMMAのkは0です。

NV/NVMの円形と同心コアシェル用因子分解、D6 E1のx入射成分、complex128、Lanczosなし、Redheffer結合を使います。
Q=512は法線場の512×512標本化格子です。
縮小モードを求めた後に不要な層行列を解放し、層のS行列を1枚ずつ結合します。各ケースは別プロセスで実行します。

## 結果と作図

出力ルートのmetal/cu、pmma/cu、metal/ag、pmma/ag、metal/al、pmma/alに、all_cases.csv・checkpoint.json・report.json・plan.jsonを保存します。
PMMA側にはprofile_layers.csvも保存します。
単体のAは構造と半無限金属基板を合わせた全吸収率でA=1−R、T=0です。
PMMA側のA=1−R−Tは金属膜の吸収率、Tは無吸収PMMA基板へ入るパワーです。
単体のモスアイ吸収・基板吸収、計算時間、GPUピークメモリもCSVに記録します。
CSVのパワーは割合です。図では100倍して%表示します。

```sh
python plot_cu_ag_al_motheye_order_nv.py \
  --output-root studies/cu_ag_al_motheye_nv/results/Nz60_Q512_M4-36
```

figuresにCu・Ag・Alそれぞれの*_metal_and_PMMA_R_A_T_vs_order.png/.svgを出力します。
上段が単体、下段がPMMA蒸着、左からR・A・Tです。
*_metal_order_changes、*_pmma_order_changesには、隣接次数の全波長での最大差と許容差を表示します。
出力ルートのorder_convergence_summary.csvで各金属・構造の状態と判定を一覧できます。

許容差0.0005は0.05パーセントポイントです。全3波長が完了した隣接次数の最後の2比較が許容差以内かを調べます。
単体はR・基板流入・モスアイ吸収、PMMAはR/T/Aを比較します。
有限の指定範囲を完了した場合にだけpassed_at_fixed_Nz_Qを判定し、途中停止は正式な判定通過として扱いません。
固定NzとQでの次数依存の確認であり、層数・法線場格子や全波長スペクトルの収束は別に確認してください。
GPUでの高次数計算は、この作成時点では未実行です。

## ローカル検証

CPU・complex128で3金属×2構造×3波長の18条件を、実行予定と同じNz=60（PMMAは61構造層）、M=4、Q=64で確認しました。
全条件で省メモリ化前後のR/T/Aと吸収分配が絶対差1e-10以内で一致し、非物理的パワーの検査を通過しています。
OOM後の他構造への継続、保存済みケースを飛ばす再開、全6件の入力の事前確認、Agの実際の子プロセス実行と作図も確認しました。
これらは実装の確認であり、M=36やQ=512での収束確認を示すものではありません。
