# Fe・Ni・Ti・W・VのNV法による次数収束調査

金属単体と、PMMAモスアイに金属を蒸着した構造を調べる専用ランナーです。
直前のCrのNV計算と同様に、回折次数Mだけを変え、層数NzとNVの標本化格子Qを固定します。
Wには添付されたWerner(W).csvを使います。

## サーバーへの配置

既存のtsoyutarcwaプロジェクトのルートで、配布ZIPを展開してください。

```sh
cd /home/7/uq06557/common2/20260813/tsoyutarcwa
unzip -o transition_metals_nv_bundle.zip
```

ZIPには今回追加したランナー、作図スクリプト、NV用補助モジュール、光学定数、説明書が入っています。
既存のrcwa_ext、rcwa_solver_auto.py、studies/gold_motheye、studies/pmma_metal_motheye等を利用します。
既存のCr用ランナー、ASR用ランナー、以前のOrdal(W).csvは変更しません。

## 実行

最初に入力と10件の計算条件だけを確認できます。このコマンドはGPU計算を開始しません。

```sh
python -u run_transition_metals_nv.py --dry-run
```

全5金属 × 単体・PMMA蒸着の計10件を順番に実行します。

```sh
python -u run_transition_metals_nv.py \
  --device cuda:0 \
  --materials Fe,Ni,Ti,W,V --structures both \
  --slices 60 --grid 512 \
  --start-order 4 --step 2 --max-order 36 \
  --wavelengths 400,550,700 \
  --tolerance 0.0005 \
  --output-root studies/transition_metals_nv/results/Nz60_Q512_M4-36_Werner
```

M=4,6,…,36の17次数、3波長、10件なので、すべて完了した場合は510ケースです。
上のコマンドをもう一度実行すると、同じ条件で保存済みのケースを飛ばして再開します。
最大次数だけを大きくして同じ出力先を使えば、その先の次数を追加できます。
材料・ソルバー・層数・格子等が変わった場合は、別の出力先を指定してください。

WのPMMA蒸着構造だけを実行する例です。

```sh
python -u run_transition_metals_nv.py \
  --device cuda:0 --materials W --structures pmma \
  --slices 60 --grid 512 \
  --start-order 4 --step 2 --max-order 36 \
  --wavelengths 400,550,700 --tolerance 0.0005 \
  --output-root studies/transition_metals_nv/results/W_pmma_Nz60_Q512
```

`--structures metal`なら金属単体だけを計算します。
回折次数を上限なしで増やしたい場合は、最初の全金属コマンドの`--max-order 36`を`--max-order 0`にしてください。
OOM、非物理的な出力、または固定格子の下限条件Q≥max(32,4M+4)に達すると、その金属・構造の計算を停止します。
Q=512、偶数次数ではM=126までが格子の許容範囲です。この値までGPUで実行できることを保証するものではありません。
停止後も他の金属・構造の計算へ進み、成功した波長のデータは保存されます。

## 計算条件

| 項目 | 金属単体 | PMMAへの蒸着 |
|---|---|---|
| 平面周期 | 200 nm、三角格子 | 200 nm、三角格子 |
| モスアイの高さ | 500 nm | PMMAコア500 nm |
| コア/金属先端半径 | 金属5 nm | PMMAコア5 nm |
| 底面半径 | 金属95 nm | PMMAコア65 nm、外周95 nm |
| 半径の深さ依存 | 直線 | 直線 |
| 金属膜 | 構造全体が金属 | 側面・頂部・谷部それぞれ30 nm |
| 基板 | 同じ金属、半無限 | 無吸収PMMA、半無限 |
| 深さ分割 | 60層 | 高さ500 nmを60分割＋頂部キャップ1層 |
| 計算上の構造層数 | 60 | 61 |

PMMA側のNz=60は従来モデルと同じ「高さ500 nmの深さ分割数」です。
そのうち上側56層がコア・金属シェル・空気、下側4層が谷部の膜を表すPMMA円柱・金属背景の層です。
これらに厚さ30 nm、半径35 nmの頂部金属キャップを追加します。
PMMAの基板面からキャップ最上面までは530 nmです。
層ごとの形状は各PMMAジョブのprofile_layers.csvに出力します。このCSVは60分割を記録し、追加キャップはplan.jsonのgeometryで確認できます。

共通条件は垂直入射、x偏光、入射側空気、比透磁率1です。
NV/NVMの円形・同心コアシェル用フーリエ係数を使い、NV場の標本化格子を512×512に固定します。
D6対称性のE1のx入射成分で行列を縮小し、complex128、Redheffer結合、半分のS行列を使います。
Lanczosフィルターは使いません。

各層の縮小モードが求まった後、不要な全空間行列・場復元用行列を解放します。
Redheffer結合では層のS行列を1枚ずつ作って消費します。
各ケースを別プロセスで実行するため、波長・次数をまたいでGPUメモリが残りません。
電場分布や勾配の計算はこのランナーには含まれていません。

## 光学定数

NV用データはstudies/transition_metals_nv/dataに同梱しています。

| 金属 | データ |
|---|---|
| Fe | Fe.csv |
| Ni | Ni.csv |
| Ti | Ti.csv |
| W | Werner(W).csv（今回の添付） |
| V | Vanadium.csv |

Wの表の波長範囲は17.586–2479.684 nmです。400・550・700 nmはすべて表の範囲内で補間します。
以前のOrdal表とは光学定数が異なるため、Wの旧計算との違いには材料データの変更も含まれます。
添付CSVのwlはµmとして読み、nmへ変換します。各標本点でε=(n+ik)²を作り、従来と同じく複素誘電率εを波長について線形補間します。
元ファイルの同一性・範囲はdata/manifest.jsonに記録しています。

PMMAは従来と同じSzczurowski.csvのnとk=0を使います。
PMMAの表は404.7 nmから始まるため、400 nmだけは従来と同じlinear-to-400方針で最初の区間を線形外挿します。
金属の400 nmには外挿を使いません。PMMAの外挿の有無はall_cases.csvにも記録します。

## 収束判定と保存内容

許容差0.0005は入射パワーを1とした絶対差、つまり0.05パーセントポイントです。
全3波長が完了した隣接次数を比較し、高次数側の最後の2回の比較がともに許容差以内かを調べます。
したがって判定には少なくとも3つの完了次数が必要です。

- 金属単体：R、基板への流入パワー、モスアイ部分の吸収率の差を比較。
- PMMA蒸着：R、T、Aの差を比較。

金属単体では半無限金属内で最終的に吸収されるのでT=0、全吸収率A=1−Rです。
基板への流入パワーを基板吸収とし、それを全吸収から引いたものをモスアイ吸収とします。
PMMA蒸着ではTは無吸収PMMA基板へ入るパワー、A=1−R−Tは金属膜での吸収です。

report.jsonのorder_convergenceには次の項目があります。

- tail_two_steps_passed：保存済みの最後の2比較が許容差以内。
- requested_range_complete：指定した有限の次数範囲を完了したか。
- passed_at_fixed_Nz_Q：上の2項目が両方true。OOMや数値異常で途中停止した結果には付けません。
- working_order：上記を満たす場合に、高次数側3点の中央の次数を記録。

これは「指定した次数範囲の末尾で、固定Nz・Qでの次数依存が許容差以内」という判定です。
さらに高い次数で変化が再開する可能性は残り、層数・格子の収束や全波長スペクトルの収束は別途確認が必要です。
working_orderもその限定された範囲での候補です。
今回のランナーは400・550・700 nmの次数掃引だけを実行し、61点スペクトルは計算しません。

出力構成は次のとおりです。

```text
studies/transition_metals_nv/results/Nz60_Q512_M4-36_Werner/
  batch_summary.json
  metal/fe/   （ni、ti、w、vも同様）
    all_cases.csv
    checkpoint.json
    report.json
    plan.json
    worker_results/
  pmma/fe/    （ni、ti、w、vも同様）
    all_cases.csv
    checkpoint.json
    report.json
    plan.json
    profile_layers.csv
    worker_results/
```

all_cases.csvのR/T/Aは割合です。例えば0.01は1%です。
GPUのピークallocated/reservedバイト数、NV使用の記録、実際の構造層数も保存します。
異常が出たケースは成功データに混ぜず、worker_resultsとreport.jsonに失敗内容を残します。
終了コードは0が判定通過、2が途中停止または次数判定未達、1が計算・設定失敗です。dry-run成功も0です。

## 作図

計算中または完了後に、以下で保存済み結果を図にできます。

```sh
python plot_transition_metals_nv.py \
  --output-root studies/transition_metals_nv/results/Nz60_Q512_M4-36_Werner
```

各ジョブのfiguresにreflectance_vs_order.png/.svgとorder_changes.png/.svgを保存します。
反射率は波長ごとの横3列、次数差は各指標の全波長での最大差と許容差を表示します。
出力ルートのorder_convergence_summary.csvで、各金属・構造の完了次数と次数判定を一覧できます。
計算途中の場合、その時点までにbatch_summary.jsonへ登録されたジョブを表示します。

## ローカル検証

CPU・complex128で5金属×2構造×3波長の30条件と、WのPMMA蒸着のNz=60の1条件を計算しました。
全31条件でメモリ削減前後のR/T/Aと吸収分配が絶対差1e-10以内で一致しました。
途中停止からの再開、最大次数の延長、異なる条件の混入拒否、失敗後の他ジョブ継続のテストも通過しています。
加えてWの2構造・3波長で実際の子プロセス実行と作図を確認しています。
これらは実装の検証であり、高次数の物理量が収束したことを示す結果ではありません。
GPUでの高次数計算と実メモリ量はサーバーで確認してください。
