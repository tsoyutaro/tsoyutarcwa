# 共通RCWA高速化・金属最適化 v3

添付のCUDAログでは、n2.5とPMMAの正方格子・六方格子について、小規模試験に加えてM8/grid512/15層/31波長/Adam 3更新の接続検証も全ケース合格しています。ソース照合の差は改行/BOMのみです。これは接続検証の合格で、500更新後の収束やM16の次数収束を意味するものではありません。

## 反映した内容

共通 `rcwa_ext` と、既存のn2.5・Cr最適化、材料別に分かれた43条件、新設16条件へ適用しました。

- 正方格子：C2vのスカラー対称性ブロック内で逆行列を計算し、縮約P/Qを直接構築。
- 六方格子：native star、D6のE1 partner 2行と重複しないスカラーirrep逆行列を使用。矩形基底での重複構築と12回の平均化を除去。
- 一定屈折率：波長に依存しないP/Q係数を共有。
- PMMAと金属分散：disk形状行列と法線場を共有し、各波長の複素誘電率・逆誘電率係数・P/Q・portは更新。
- 波長ごとの共役勾配を合算し、形状のVJPを1回実行。その後、元の単調softmaxへ戻してlogits勾配を計算。
- portと幾何基底だけを上限付きでキャッシュ。形状依存行列と計算グラフは更新ごとに作り直す。
- 層checkpointのOFF/ONに対応。

共通AutoRCWAの既定は `acceleration='auto'` です。対象は、法線入射・x偏光・空気背景・セル中心の非接触円・complex128・fieldsなし・half S・正方格子/C2vまたは六方格子/D6のNVMです。対象外の形状、コーティング、斜入射、y偏光、場の再構成などは従来経路を使用します。ASRなど別の定式化も元の実装を使用します。

`acceleration='reference'` で共通RCWAの行列高速化を停止できます。最適化側の `rcwa_acceleration` は `auto`（行列＋波長共有）、`matrix`（波長ごとの行列高速化）、`reference`（従来経路）です。旧ワーカーの設定にこのキーがない場合は、最適化の従来経路を使います。

## 新設した金属フォルダ

各条件は、他の研究フォルダを必要とせずコピーできる独立パッケージです。

| 材料 | フォルダ例 | 条件 |
|---|---|---|
| Al | `Al/al_motheye_P200_H500_Nz15` | P200 nm、H500 nm、5/10/15/20層 |
| Cu | `Cu/cu_motheye_P200_H500_Nz15` | 同上 |
| Au | `Au/au_motheye_P200_H500_Nz15` | 同上 |
| Ag | `Ag/ag_motheye_P200_H500_Nz15` | 同上 |

金属柱と同じ金属の半無限基板、空気入射、コーティングなしです。既存研究で使われている `Rakic (Al).csv`、`Cu.csv`、`au_measured_nk.csv`、`Ag.csv` を各条件内のdataへ同梱しています。金属の epsilon=(n+ik)^2 を波長に対して線形補間し、外挿は行いません。データの出典パスとハッシュは配置記録に保存しています。

目的関数は波長平均反射率の最小化です。この半無限基板モデルでは、全吸収率A_total=1-Rの最大化と等価です。`power_into_substrate` は金属基板の表面を通過する電力であり、遠方透過率ではありません。`motheye_absorptance=1-R-power_into_substrate`、`substrate_absorptance=power_into_substrate` を保存します。これらは表面電力収支から求める量で、体積損失積分による独立検証ではありません。

## 最適化で一度終了する

例として `Optimization/Al/al_motheye_P200_H500_Nz15` 内へ移動して実行します。どの材料・条件でも同じ引数を使用できます。

```bash
python run.py --device cuda:0 --mode optimize --output-root results/NV_shared_v3
```

既定でもmodeはoptimizeです。従来の `--mode all` もoptimizeへの互換aliasとし、最適化後の検証は自動実行しません。指定した格子のOptuna/Adamが完了するとbest形状を保存し、プロセスを終了します。通常の終了直前に、best_lossと形状を一致させるための最終順計算は行います。M16等の後処理は開始しません。

正方格子・六方格子を別PCで実行する場合は、条件フォルダをそれぞれコピーし、`--lattices square` または `--lattices hexagonal` を追加してください。既存Crの元ワークフローは六方格子専用です。

以下のコマンドは最適化終了後に、同じ条件フォルダ・同じoutput-rootで明示的に実行します。Optuna/Adamを追加で進めることはありません。

## convergence：次数・gridの収束を確認する

```bash
python run.py --device cuda:0 --mode convergence --orders 8,12,16 --grids 512,1024 --wavelengths 400,550,700 --output-root results/NV_shared_v3
```

同じbest形状を固定し、指定したMとgridの直積を評価します。最大M・最大gridの結果との差を `convergence.csv` と `report.json` へ保存します。最高2次数の比較と最高2gridの比較が、設定のverification_tolerance以下であるかを `numerical_convergence_passed` に記録します。次数またはgridを1種類しか指定していない場合は、両方の収束を検証したとはみなしません。最大条件との差だけで無限次数への収束を証明するものではありません。

## timing：従来版と共有版の計算時間を比較する

```bash
python run.py --device cuda:0 --mode timing --order 8 --grid 512 --wavelengths 400:700:10 --repeats 3 --output-root results/NV_shared_v3
```

既定では損失＋全logits勾配までを測定します。順計算だけなら `--forward-only` を追加します。各経路の初回時間をcold_seconds、初回を除く繰り返しをwarm_seconds、その中央値と1波長換算値を保存します。CUDA同期とGPUピーク割当メモリを記録します。時間比較の前提として、従来版と共有版の損失・R/Tまたは基板電力・勾配の一致もチェックします。

単一波長の時間を見たい場合は `--wavelengths 550` とします。波長共有の効果を見る場合は、31波長のコマンドを使ってください。測定値には本番評価関数の処理を含み、worker起動時間は別項目です。

## spectrum：最適形状のスペクトルを取得する

```bash
python run.py --device cuda:0 --mode spectrum --order 16 --grid 1024 --wavelengths 400:700:3 --output-root results/NV_shared_v3
```

`spectrum.csv`、`spectrum.png`、詳細JSONを保存します。M/grid/波長範囲は指定できます。計算結果は `results/NV_shared_v3/<格子>/postprocess/<モード>/<識別子>/` に分離し、`latest_spectrum.json` などが最新の保存先を示します。元のbest.jsonとAdam checkpointは変更しません。

従来経路で確認する場合は、後処理コマンドに `--analysis-backend reference` を追加します。既存の `--mode verify`、`--mode audit`、`--mode plot` も引き続き明示実行できます。

最適化時に `--checkpoint-layers`、`--config` 等を指定した場合は、後処理にも同じ設定を指定してください。実験の設定・ソフトウェア・ソースが異なるoutput-rootの流用は拒否します。今回の更新は計算経路の変更を伴うため、従来の計算用output-rootやcheckpointをそのまま新方式の続きとして使用せず、新しい `results/NV_shared_v3` を使用してください。

## 更新した実装をCUDAで追加確認する

この更新を展開した研究ルート（rcwa_extとOptimizationが並ぶ場所）で実行します。

```bash
python validate_rcwa_acceleration.py --device cuda:0 --check-fd --output Optimization/results/acceleration_cuda_v3.json
```

既定ではM2/grid48/5層/3波長で、n1.5/n2.0/n2.5/PMMA/Al/Cu/Au/Agの両格子と、既存Crの六方格子の17ケースを照合します。`status=passed`、`cuda_execution_performed=true`、17ケース完了を確認します。

金属の本番サイズも確認する場合は次を実行します。

```bash
python validate_rcwa_acceleration.py --device cuda:0 --production --materials Cr Al Cu Au Ag --output Optimization/results/metals_cuda_M8_v3.json
```

金属のこの共通化実装はCPUで検証済みですが、こちらのPCではCUDA実機試験を行っていません。既存のn2.5/PMMAのCUDA合格を、金属のCUDA合格とは扱っていません。

## 配布と記録

配布用 `RCWA高速化と金属最適化_v3.zip` は、研究ルートへ展開する構成です。`Optimization/` と `rcwa_ext/` がZIP直下にあります。既存の計算結果やjournal、checkpointを含めません。各条件フォルダだけを別PCへコピーすることもできます。

過去の検証パッケージは比較基準として凍結し、ソースを変更していません。古い配布ZIPも履歴として残しています。新規計算にはv3のソースを使ってください。

初期・最終の勾配監査も自動実行しません。必要な場合は、同じ設定で `python run.py --device cuda:0 --mode audit --output-root results/NV_shared_v3` を別途実行してください。最適化コマンドは勾配の次数収束確認を自動で開始しません。
