# Au単体・PMMA＋Au：省メモリNV法による回折次数掃引

他の5金属と同じ2Dモスアイの形状で、金単体とPMMAに金を蒸着した構造を計算します。
Nzと法線場の格子Qは固定し、Mだけを増やします。

## サーバーでの実行

既存のtsoyutarcwaプロジェクトのルートへZIPをコピーし、展開してください。
今回の金専用ファイルを追加するパッケージです。

```sh
cd /home/7/uq06557/common2/20260813/tsoyutarcwa
unzip -o gold_motheye_NV_bundle.zip

python -u run_gold_motheye_order_nv.py \
  --device cuda:0 \
  --structures both \
  --gold-model csv \
  --slices 60 --grid 512 \
  --start-order 4 --step 2 --max-order 36 \
  --wavelengths 400,550,700 \
  --tolerance 0.0005 \
  --output-root studies/gold_motheye_nv/results/Nz60_Q512_M4-36_measured
```

金単体を先に、PMMA＋Auを次に計算します。全ケースが成功すれば17次数×3波長×2構造=102ケースです。
計算前に入力条件だけを確認する場合は、上のPythonコマンドに`--dry-run`を追加してください。
金単体だけなら`--structures metal`、PMMA＋Auだけなら`--structures pmma`です。

1つの構造でOOMや非物理的パワーが発生すると、その掃引を止めて次の構造へ進みます。
それまでに成功した波長は保存します。同じコマンドで保存済みケースを飛ばして再開できます。
上限なしで増やすなら`--max-order 0`です。OOM・数値異常・固定格子の制限まで進みます。
Q≥max(32,4M+4)を条件とするため、Q=512の偶数次数ではM=126が格子側の上限です。
その次数までGPUで実行できることを示す設定ではありません。

## 金の光学定数

既定のCSVは今回同梱した`studies/gold_motheye_nv/data/au_measured_nk.csv`です。
これは既存のgold_motheye3/data/au_measured_nk.csvをそのままコピーしたもので、187.9–1937 nmを含みます。
PMMA＋Auで以前使ったPMMA側のAu CSVと、400/550/700 nmの誘電率が一致することも確認しました。
各標本点のε=(n+ik)²を波長について線形補間します。3波長ともAuの外挿はありません。
別のAuデータを指定する場合は`--gold-csv path/to/Au.csv`を追加してください。
CSVには波長/n/k、または従来のAuローダー形式の波長/epsilon_real/epsilon_imagを使えます。

Rakić Lorentz–Drudeモデルで比較したい場合は、材料モデルと出力先を変えます。

```sh
python -u run_gold_motheye_order_nv.py \
  --device cuda:0 --structures both --gold-model rakic-ld \
  --slices 60 --grid 512 \
  --start-order 4 --step 2 --max-order 36 \
  --wavelengths 400,550,700 --tolerance 0.0005 \
  --output-root studies/gold_motheye_nv/results/Nz60_Q512_M4-36_rakic_ld
```

CSVとRakićでは金の誘電率が異なります。過去のASR結果との比較では材料モデルもそろえてください。
材料モデル・層数・格子・コードを変えた場合、既存チェックポイントは混ぜず、別の出力先を指定してください。
最大次数だけの延長は同じ出力先で再開できます。

## 固定条件

| 項目 | Au単体 | PMMA＋Au |
|---|---|---|
| 周期・格子 | 200 nm、三角格子 | 同左 |
| 高さ | 金属モスアイ500 nm | PMMAコア500 nm |
| 先端半径 | 金属5 nm | PMMAコア5 nm |
| 底面半径 | 金属95 nm | PMMAコア65 nm、外周95 nm |
| 半径の深さ依存 | 直線、各層の中点で評価 | 同左 |
| 金属膜厚 | 構造全体がAu | 側面・頂部・谷部が各30 nm |
| 基板 | 半無限Au | 半無限・無吸収PMMA |
| 深さ分割 | 60層 | 60分割＋頂部キャップ1層、計61構造層 |

PMMAの60分割は上側56層のコア/金属シェル/空気と、下側4層の谷部膜を含みます。
厚さ30 nm・半径35 nmの頂部キャップを加えるため、基板から最上面の高さは530 nmです。
これは今回のFe/Ni/Ti/W/Vとの比較をそろえるための30 nm膜条件です。
過去にPMMA＋Auで20 nm膜を使った結果との比較では、膜厚・コア底面半径の違いにも注意してください。

入射は空気側から垂直・x偏光、比透磁率1です。
NV/NVMの円と同心コアシェル用因子分解、D6 E1 x入射成分、complex128、Redheffer結合、半分のS行列を使います。
Lanczosフィルターは使いません。Q=512は法線場の512×512標本化格子です。
層の縮小モードを求めた後に不要な全空間行列を解放し、S行列は1層ずつ作って結合します。
各ケースを別プロセスで実行するため、ケース間でGPUメモリを持ち越しません。

PMMAのnはSzczurowski.csvを使い、k=0です。
データは404.7 nmから始まるため、400 nmだけは従来どおり最初の区間を線形外挿するlinear-to-400方針です。

## 保存内容と判定

出力ルートのmetal/auとpmma/auに、all_cases.csv・checkpoint.json・report.json・plan.jsonを保存します。
PMMA側には60分割の形状を記録したprofile_layers.csvも保存します。
R/A/T、単体のモスアイ吸収と基板吸収、計算時間、GPUピークメモリを記録します。

単体では半無限Au基板内で吸収されるためT=0、全吸収A=1−Rです。
基板へ入るパワーを基板吸収として、モスアイ部分の吸収を別に記録します。
PMMA＋AuではTを無吸収PMMA基板へ入るパワー、A=1−R−TをAu膜での吸収とします。
CSVの値は割合なので、0.01は1%です。

0.0005は入射パワーを1とした絶対差、0.05パーセントポイントです。
全3波長が完了した隣接次数について、最後の2回の比較が両方許容差以内かを調べます。
単体はR・基板流入・モスアイ吸収、PMMAはR/T/Aを比較します。
有限の指定次数範囲を完了してこの判定を満たした場合だけ、passed_at_fixed_Nz_Q=trueになります。
OOMなどで途中停止したケースは、正式な判定通過として扱いません。
これは固定NzとQでの次数依存の確認であり、層数・格子や61点スペクトルの収束とは別です。

## 作図

```sh
python plot_gold_motheye_order_nv.py \
  --output-root studies/gold_motheye_nv/results/Nz60_Q512_M4-36_measured
```

figures/Au_and_PMMA_Au_R_A_T_vs_order.png/.svgは、上段が単体、下段がPMMA＋Au、左からR・A・Tです。
metal_order_changes.png/.svgとpmma_order_changes.png/.svgには、隣接次数の最大変化と許容差を表示します。
Rakićで実行した場合は作図コマンドの出力先もそのフォルダへ変えてください。

## 実装の検証

CPUでCSV/Rakićの2モデル×2構造×3波長の12条件と、両モデル・両構造のNz=60の4条件を確認しました。
全16条件で省メモリ化前後のR/T/Aと吸収分配が絶対差1e-10以内で一致しています。
材料モデルの混入拒否、次数上限の延長、部分次数からの再開、OOM後の次構造への継続もテスト済みです。
GPUでの高次数計算は未実行です。高次数の収束性そのものは、サーバーで得られる結果から判断してください。
