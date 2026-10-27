# 電力収支エラーの診断（v3の最適化結果を変更しない追加ツール）

`Invalid shared power flux.` は、R、P_sub、1-R、1-R-P_subのいずれかが有限値でないか、許容範囲[-tol,1+tol]を外れた場合に発生します。金属の標準tolは1e-6です。エラーだけでは、波長・数値・原因は特定できません。

`Optimization stopped before verification` は旧v3ドライバーが失敗時にも表示する終了メッセージです。成功判定ではありません。`trial 0 failed` が出たCuは失敗としてjournalへ記録されます。set -eの連続コマンドでは、このエラーでAu/Agへ進まず停止します。失敗試行もtrials=1に数えられるため、同じ最適化コマンドの再実行だけではCuの新しい試行は始まりません。

この追加ツールは、現在のcheckpoint形状について、共有版・従来版・行列高速化版の順計算の生の電力収支を比較します。非物理値を観測する診断のため、そのまま記録しますが、最適化側の判定を無効にするものではありません。OptunaやAdamは再開せず、checkpoint、journal、best.json、既存ソースを変更しません。ファイルはrun-root/diagnosticsへ保存します。計算中の実験と同時に比較せず、停止した状態で実行してください。

## 転送と実行

diagnose_power_flux.pyを研究フォルダのOptimization直下へ置いてください。ZIPを使う場合は、rcwa_extとOptimizationが並ぶ研究ルートへ展開します。v3の本体ソースを上書きする必要はありません。実験に記録されたソースと一致するかを起動時に確認します。

Linuxで1行のまま実行します。

```bash
( set -e; cd /home/7/uq06557/common2/20260813/tsoyutarcwa/Optimization; python diagnose_power_flux.py --case-dir Cu/cu_motheye_P200_H500_Nz15 --device cuda:0; )
```

既定ではresults/hex_trial1_v3/hexagonal/trials/trial_0000/adam_checkpoint.ptとresolved_config.jsonを読み、保存状態のM/gridと最適化の全波長を評価します。GPUのCUDAが使用できない場合はCPUへ自動で切り替えません。

各波長でR、P_sub、A_mothとpassiveの真偽を表示します。最後に次の項目を表示します。

- next_step / successful_gradient_evaluations：保存済みAdam更新数・成功済み勾配評価数。
- nonpassive_wavelengths_by_method：各経路で許容範囲外の波長。
- reference_comparisons_passed：従来版と他の経路のR/P_subの差が1e-8以内か。
- checkpoint_unchanged：診断の前後でcheckpointが同一だったか。

診断ファイルには全logitsと半径、各波長の生の値・許容範囲外の項目・経路間差を保存します。status=diagnostic_completeは診断が終わった意味で、最適化成功や物理妥当性の合格ではありません。

## 結果の解釈

従来版・行列版・共有版が一致しているのに、全経路で同じ波長の電力収支が外れる場合は、高速化だけの不一致とは区別できます。その保存形状に対して、有限次数/gridによる誤差やNVMの数値的な問題を次に調べます。共有版だけ、または行列版も従来版と異なる場合は、その経路を詳しく点検します。診断で再現しない場合は、実際の失敗時の状態・勾配有効時・CUDA差も追加で確認します。

例えば問題の波長を特定した後、同じ形状で --order 12 --grid 1024 --wavelengths 550 のように評価条件を変更できます。元の最適化設定は変えず、別の診断ファイルへ記録します。波長は実際に問題が出た値へ置き換えてください。

## ローカル確認

Cuの5層/M2/grid48/400,550,700 nmで3経路が一致し、checkpointを変更しないことを確認しました。Cuの15層/M8/grid512/31波長の初期形状も、CPU共有版では電力収支チェックを通過しました。これらはHPCで失敗した実際のcheckpointやCUDAでの再現結果ではありません。実際の保存形状の診断結果を使って原因を判断します。
