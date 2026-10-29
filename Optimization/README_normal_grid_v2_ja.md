# 法線場の正則化OFF比較（v2）

ZIP内のOptimizationの内容を既存のOptimizationに追加してください。既存のcontinue_convergence.pyを利用します。v1のvalidate_normal_grid.pyや本番RCWAソースは変更しません。

## Al・15層・次数30・3波長

```bash
( set -e; cd /home/7/uq06557/common2/20260813/tsoyutarcwa/Optimization; python validate_normal_grid_v2.py --case-dir Al/al_motheye_P200_H500_Nz15 --run-root results/hex_trial1_v3 --device cuda:0 --order 30 --normal-mode off --grids 1024,2048 --wavelengths 400,550,700; )
```

同じ保存済み最適形状を固定し、grid1024と2048の2条件、各400/550/700 nmを計算します。逆伝播・最適化は実行しません。--normal-reference-gridsはOFF時には指定しません。

OFFは中心側・セル間の両方のsmoothstep補正を計算せず、最も近い円柱中心に対するradial normalのn n^Tを使います。中心点は角度平均I/2、等距離点は全ての等距離中心に対する射影行列の平均です。等距離・中心の判定には機械精度に基づく値を使用し、gridに比例する有限幅の接続は作りません。これはサンプリング点での規約であり、材料界面の形状を平滑化しません。

- 通常点：P = n n^T。
- 円中心：P = I/2。
- セル間の等距離点：全ての最近接中心のPを平均。六方格子の3重等距離点も含めます。
- 等距離判定：距離の2乗差が64 * float64 epsilon * max(Lx,Ly)^2以下。
- 円の誘電率・逆誘電率の解析的Fourier係数、材料分散、次数、形状は維持。

本番ファイルを編集せず、独立した子プロセスのメモリ内でnormal_fields生成を差し替えます。元のmanifest、ソース、実行環境の照合を行い、保護した保存結果が変更されていないことも確認します。

## 結果

元の実行フォルダ内：

`hexagonal/postprocess/normal_grid_v2/<識別子>/report.json`

`hexagonal/latest_normal_grid_v2.json`にも結果へのポインタを保存します。v1の結果は保持します。

- normal_modeとnormal_definition：今回採用した法線場の定義。
- sampling_comparisons：OFFの定義を固定したgrid比較。
- details.reflectance：反射率の最大絶対差。
- details.power_into_substrate：基板流入パワーの最大差。
- details.motheye_absorptance：モスアイ内吸収率の最大差。
- passed：RとP_subの最大差が元のverification_tolerance以下か。吸収率の差は個別に確認。
- production_files_unchanged：保護した元の実験ファイルが実行前後で同一か。

OFF時のnormal_reference_grid=nullは、平滑化幅を指定していないため正常です。smoothing_comparisonsは空配列です。条件は別プロセスで順番に実行します。非物理値はクリップせず記録し、OOMまたはworker不明終了時は残りを保守的に省略します。

同じコマンドの再実行では完了条件を再利用します。失敗条件の再試行は--retry-failuresを追加します。

## 固定幅モード

v2でも従来の4条件比較ができます。v1の結果はそのまま残りますが、スクリプトと出力先が異なるためv2の初回は新しく計算します。

```bash
( set -e; cd /home/7/uq06557/common2/20260813/tsoyutarcwa/Optimization; python validate_normal_grid_v2.py --case-dir Al/al_motheye_P200_H500_Nz15 --run-root results/hex_trial1_v3 --device cuda:0 --order 30 --normal-mode fixed --grids 1024,2048 --normal-reference-grids 1024,2048 --wavelengths 400,550,700; )
```

## どの定義・値を採用するか

0.5859375 nmと0.29296875 nmは物理定数ではなく、旧実装のgrid1024/2048から作った診断用の幅です。どちらかを正解として選ぶ根拠は現時点ではありません。次の順に判断します。

1. 材料界面で正しい法線方向を保つこと。補助場の変更で材料形状を変えないこと。
2. 一つの法線場の定義を固定し、サンプリングgridを変更してR、P_sub、必要な吸収率の差を確認すること。
3. gridが十分な状態で、同じ定義を維持してFourier次数を上げたときの安定性を確認すること。
4. 複数の妥当な法線場定義または独立した電磁場解析法との一致を確認すること。同じ法線場を使うreferenceとの一致だけでは、定義自体の正しさを判断しない。
5. 要求精度を満たす候補の中で、計算負荷と勾配の安定性を比較すること。本番最適化へ反映する前に、採用する定義で自動微分と有限差分の一致も確認する。

有限次数で反射率が小さく出る定義を選ぶ方法ではありません。OFFであっても有限次数・サンプリングの誤差は残り、セル間の不連続性によって収束が遅くなる可能性もあります。

これらは固定した候補形状を評価するための基準です。最適化の全stepで完全収束を要求するものではありません。Al単一金属・六方格子の診断であり、PMMA複合や他の格子へ本番反映する際には、それぞれのモデルで確認が必要です。

## 検証

```bash
python test_normal_grid_v2.py
```

CPUで16項目合格。従来固定幅の等価性に加え、OFFでsmoothstepが呼ばれない、材料Fourier係数が不変、補助行列が半径に依存しない、通常点の射影、中心・2重/3重等距離点の規約、Hermitian性、3波長forwardの有限性などを確認。

独立したCPU用検証フォルダで、Alの保存済み15層形状、M2、grid48/64、3波長のOFF比較が正常終了し、元の実験ファイル不変と再実行時のキャッシュ再利用を確認しました。CUDA・次数30は未実施です。
