# OFF Optuna30最適化後の一括評価

通常版の`run_metal_off_optuna30.py --grid 512 --materials Al,Ti,Cr`で
最適化が完了した保存済み最良形状を使います。3材料とも既定の最適化次数M8です。
最適化は再実行せず、Alの収束確認→Alのスペクトル→Ti→Crの順に処理します。

```sh
cd /home/7/uq06557/common2/20260813/tsoyutarcwa/Optimization &&
python postprocess_metal_off_optuna30.py \
  --device cuda:0 \
  --materials Al,Ti,Cr \
  --orders 4:40:2 \
  --grids 512,1024 \
  --convergence-wavelengths 400,550,700 \
  --spectrum-order 40 \
  --spectrum-grid 512 \
  --spectrum-wavelengths 400:700:3
```

- 収束確認は偶数次数4,6,…,40の19条件×grid 512/1024×3波長です。
- スペクトルは400～700 nmを3 nm間隔で計算し、両端を含め101点です。
- 元のconfigと`results/hex_OFF_optuna30_v1_M8_Q512`を読み込みます。
  Al/Tiはそれぞれ`Al/al_motheye_P200_H500_Nz15`と`Ti/ti_motheye_P200_H500_Nz15`、
  Crは`cr_motheye`です。旧1-trialや`_normal_off_v1`の結果は選択しません。
- 各材料の最適化完了状態・最良形状・controller lockを確認します。
  未完了・ロック中・読込失敗の材料は記録し、次の材料へ進みます。
- 非物理値は同一次数の残りのgridを省略し、次の次数へ進みます。
- OOMではその材料の残りの収束計算とM40スペクトルを省略し、次の材料へ進みます。
- M40/Q512で3波長の計算が完了しない場合も、その材料の101点計算を省略します。
  M40/Q512が物理的に計算できれば、有限次数の差が許容値未満になっていなくても
  スペクトルを計算します。処理完了と数値収束の合格は別の情報です。
- スペクトルの他の波長で非物理値やOOMが起きた場合も、その材料を失敗として記録し次へ進みます。
- 手動中断では全体を停止します。ロックの自動削除は行いません。
- 原ソルバ・Adam・ケースdriver・ソース同一性の検査は変更しません。
  Crは元のdriverと同じ厳密なconfig/source/device/runtime比較を維持します。

## 保存先

集計は`Optimization/postprocess_OFF_optuna30_v1/summary.json`と`summary.csv`です。
材料ごとの処理状況、101点の平均反射率、所要時間、詳細report/CSVのパスを保存します。
詳細な収束結果とスペクトルは元のrun-root内の`hexagonal/postprocess/`に保存します。
再実行では同一条件の成功済み収束計算を再利用します。スペクトルは再計算します。

材料を1つだけ評価するには`--materials Ti`などに変更します。
`--prepare-only`は保存先・完了状態の確認と実行計画の保存だけを行い、RCWAは実行しません。
`--output-root`で集計の保存先、`--cases-file`で明示的なcase/config/run-root一覧も指定できます。
後者の形式は`continue_convergence.py`と同じです。

自動次数変更版の既知の次数を評価する場合は、例えばAuに対して
`--materials Au --run-tag hex_OFF_optuna30_auto_v1 --optimization-order 16`を指定します。
自動次数変更の最終次数を推測する機能はありません。最適化が完了した次数を指定してください。

## 計算時間の目安

過去の15層・六方格子・H100のforward計算は、M40で概ね140～170秒/波長でした。
31点のmatrix評価と3点のauto評価など異なるバッチ・経路の記録を含むため、予測は概算です。
同等のH100でM40まで完走すると、1材料あたり収束確認約1～1.3時間、
101点スペクトル約4～5時間、3材料を順に処理して合計約15～20時間が目安です。
CrのM40実測はなく、同じ行列サイズから推定しています。
最適化形状・GPU割当・他ジョブの負荷により変動します。最適化自体の時間は含みません。

## ローカル検証

`test_metal_postprocess.py`と`test_continue_convergence.py`で、順序・非物理値・OOM・
次材料への継続・Crのidentity/runtime検査を確認します。
小規模CPU実行での接続確認と、本番CUDA/M40での性能・収束確認は区別してください。
