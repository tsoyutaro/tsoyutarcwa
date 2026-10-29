# Alの法線場：gridと平滑化幅の切り分け

既存のOptimization/continue_convergence.pyと同じフォルダにvalidate_normal_grid.pyを置いてください。既存の最適化ソース、材料CSV、manifest、Journal、Adamチェックポイントは変更しません。対象は六方格子の単一金属円柱／同じ金属の半無限基板です。

## Al・15層・次数30・3波長

以下は1行のLinuxコマンドです。

```bash
( set -e; cd /home/7/uq06557/common2/20260813/tsoyutarcwa/Optimization; python validate_normal_grid.py --case-dir Al/al_motheye_P200_H500_Nz15 --run-root results/hex_trial1_v3 --device cuda:0 --order 30 --grids 1024,2048 --normal-reference-grids 1024,2048 --wavelengths 400,550,700; )
```

元の実行フォルダのhexagonal/report.jsonに保存された最良logitsを固定します。最適化や逆伝播は実行しません。サンプリング数Qと法線場の幅を決める参照値Nを独立に変えます。Nは実際のサンプリング数ではなく、平滑化幅を固定するための数値です。

| 実際のgrid Q | 法線場の参照値 N | 距離差に対する平滑化パラメータ 3P/N | 役割 |
|---|---|---|---|
| 1024 | 1024 | 0.5859375 nm | 従来のgrid1024と同じ定義 |
| 1024 | 2048 | 0.29296875 nm | grid1024のまま幅を変更 |
| 2048 | 1024 | 0.5859375 nm | 幅を固定して細かくサンプリング |
| 2048 | 2048 | 0.29296875 nm | 従来のgrid2048と同じ定義 |

4条件×400/550/700 nmの計12波長計算です。各条件を独立した子プロセスで順番に実行します。

## 実装範囲

保存済み実験のソース・設定・実行環境を既存のruntime_policyで照合します。native_inputsのresolution = max(lx/grid,ly/grid)という代入が1か所だけ存在することをASTで検査し、子プロセスのメモリ内でのみgridを固定参照値Nに置き換えます。円の解析的Fourier係数、サンプリング点、次数、波長、材料、形状は変更しません。resolutionを使う中心側とセル間の補正幅をともに固定し、定義がgridで変わらない比較にします。

本番のPythonファイルは編集しません。元の最適化へこの変更を適用するものではなく、原因を切り分ける診断です。実行中は元のcontroller.lockを使用し、終了・Ctrl+Cで解放します。既存lockは自動削除しません。

## 結果

保存先：元のrun-root内のhexagonal/postprocess/normal_grid/<識別子>/report.json。

hexagonal/latest_normal_grid.jsonに今回のreportへのポインタも保存します。

- sampling_comparisons：同じN（平滑化幅）で、Qだけ変えた差。
- smoothing_comparisons：同じQで、N（平滑化幅）だけ変えた差。
- details.reflectance：反射率の最大絶対差と、その波長。
- details.power_into_substrate：基板流入パワーの差。
- details.motheye_absorptance：モスアイ部分の吸収率の差。
- passed：RとP_subの最大絶対差が元のverification_tolerance以下か。A_mothの差は別記。
- production_files_unchanged：保護したmanifest、保存済みreport、Journalなどの直下ファイルとAdamチェックポイントが実行前後で同一か。

samplingの差が小さく、smoothingの差が大きければ、従来のgrid比較に平滑化幅の変更が強く影響していた根拠になります。判定は固定次数における有限の比較であり、正しい法線場の構成や次数収束の証明とは分けて扱います。

生のR、P_sub、1−R、1−R−P_subを記録し、非物理値はクリップしません。非物理値の条件も記録して他条件を続けます。CUDA/CPU OOM、結果を返さないworker終了があれば、残りを保守的に省略します。

同じコマンドの再実行では、同じソース・設定・GPU構成・形状・条件の記録を再利用します。失敗条件の再試行は--retry-failuresを追加します。既存のcontinue_convergence結果とは別の診断なので、この4条件は初回に新しく計算します。

## ローカル検証

```bash
python test_normal_grid.py
```

CPUで8項目に合格：15層の全半径でN=Qの入力行列が従来と完全一致、幅の変更で円の誘電率・逆誘電率係数が不変、実際のサンプリング点数がQを維持、M2/Q48とQ64で従来と3波長のforwardが一致、交差条件が有限、予期しないソース式の拒否、比較軸と非物理値の扱い。

さらに、独立したCPU用の検証フォルダで、Alの保存済み15層形状・M2・Q48/64・N48/64・3波長の4条件をCLIから実行し、全条件完了、保護ファイル不変、再実行時の4条件キャッシュ再利用を確認しました。CUDA・M30の4条件は未実施です。低次数CPU検証の数値から本番条件の物理的結論を出さないでください。
