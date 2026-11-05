# Cu: trial 0～14を引き継ぎ、trial 15から復旧付きで再実行

周期200 nm・高さ500 nm・15層・六方格子、正則化OFF、回折次数14、grid 512、400～700 nmの31点の算術平均反射率を使用します。既存Cuフォルダと結果を保存したまま、この独立フォルダに新しいstudyを作成します。

## 引き継ぐ結果

- 元の `hex_OFF_optuna30_auto_v1_M14_Q512` のtrial 0～14を、パラメータ・目的関数値・最良形状・中間評価値とともに取り込みます。
- 元の13件のCOMPLETEと、trial 12・13のPRUNEDをそのまま保持します。PRUNEDをCOMPLETEには変更しません。
- 元の失敗したtrial 15と同じ `lr`・`init_top`・`delta_ratio` を予約します。元の初期形状からAdamのstep 0で再計算し、失敗時点のAdam状態は引き継ぎません。
- TPE samplerの保存状態を引き継ぎます。trial 15以降で復旧方式を使用し、trial 29まで、取り込み分を含めて合計30 trialとします。
- 元の実行環境と同じOptunaバージョンが必要です。確認済みの元データはOptuna 4.6.0です。バージョン、材料データ、物理計算ソース、学習条件が一致しない場合、取り込みを停止します。

## Linux実行コマンド

このフォルダをLinux側の同じ `Optimization/Cu` 配下に反映してから、元のCuジョブが終了した状態で実行してください。

```bash
(
set -e
cd /home/7/uq06557/common2/20260813/tsoyutarcwa/Optimization

d="$PWD/Cu/cu_motheye_P200_H500_Nz15_backtracking_v1"
source_run="$PWD/Cu/cu_motheye_P200_H500_Nz15/results/hex_OFF_optuna30_auto_v1_M14_Q512"
r="$d/results/hex_OFF_optuna30_backtracking_from15_v1_M14_Q512"

python "$d/import_completed_trials.py" \
  --source-run "$source_run" \
  --output-root "$r" \
  --device cuda:0

python "$d/run.py" \
  --output-root "$r" \
  --device cuda:0 \
  --mode optimize \
  --lattices hexagonal \
  --timeout-hours 0
)
```

同じ取り込みを再実行しても、取り込み済みのstudyを初期化しません。以降の中断からの再開でも同じ出力先を使用します。停止済みプロセスのロックだけが残っている場合は、`run.py` の引数に `--recover-lock` を追加します。

## 非物理値が発生した場合

直前に正常評価した形状とAdamのモーメント・step・scheduler状態に戻します。その更新の学習率を半分にして最大4回再試行します。各再試行の学習率は、失敗前に対して1/2、1/4、1/8、1/16です。正常な更新が通れば、その学習率で続行し、連続再試行回数をリセットします。

4回とも失敗した場合は、そのtrialの最良正常形状で早期終了して次のtrialに進みます。終了理由は `nonphysical_retry_exhausted`、Optunaの状態はCOMPLETEです。初期形状から正常評価が一度も得られない場合はFAILです。OOMや通常の実行エラーを、この早期終了による成功には置き換えません。

回折次数は14で固定します。最適化完了後は検証前に停止します。

## 保存情報・確認

- `import_completed_trials.json`: 元結果のハッシュと取り込み元、予約したtrial 15のパラメータ。
- `hexagonal/trials.csv`: 元の状態・値、および新trialの終了理由。
- `hexagonal/trials/trial_0015/adam_checkpoint.pt`: 再実行trial 15の正常点・Adam状態・復旧状態。
- `recovery_attempts` は元の失敗評価も数えます。元の失敗1回と再試行4回が全て失敗した場合は5です。
- 新エンジンの `history` は初期評価step 0を含みます。`next_step` が受理済み更新数で、正常終了時の履歴数は通常 `next_step + 1` です。

`test_backtracking.py` の14件、`test_import_completed_trials.py` の8件、小規模CPU実計算を確認済みです。取り込みテストでは元trial 0～14の保持、PRUNEDの保持、trial 15から16への継続、取り込みの再実行、元結果の不変を確認しました。CUDA実行は利用先で確認してください。
