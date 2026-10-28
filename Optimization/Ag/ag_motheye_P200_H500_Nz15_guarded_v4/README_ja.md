# Ag：物理チェックを維持したAdamの再試行版 v4

対象は六方格子、周期200 nm、高さ500 nm、15層、M20、grid512、波長400–700 nm（10 nm間隔）、1 Optuna trialです。

## 今回の診断

旧M20・grid512の実行は `next_step=5`、成功した勾配評価5回でした。初期評価は通り、5回目の更新後の形状を評価する際に停止した状況です。最後に通過した評価はstep4で、平均反射率0.7258330357304851、学習率0.03、勾配ノルム8.405026394368498でした。最後の評価時間137.853520097211秒÷31波長＝約4.45秒/波長で、これは逆伝播を含む実測です。Adam更新・保存・棄却候補の時間は別です。

## 配置と変更範囲

ZIPをLinuxの `/home/7/uq06557/common2/20260813/tsoyutarcwa` に展開すると、次の独立したケースが追加されます。

`Optimization/Ag/ag_motheye_P200_H500_Nz15_guarded_v4`

旧ケースやその結果は置き換えません。新しいソースのため、旧v3のチェックポイント・Journalをこのフォルダへコピーして再開しないでください。本版は新しいtrialを初期形状から実行します。

RCWAの行列構築、材料データ、幾何パラメータ化、目的関数、物理チェックの許容差はv3と同じです。変更はAdamの更新受理・再試行、状態保存、停止理由の表示、およびエラー時の波長・電力値の表示です。

## 更新の受理と棄却

1. 初期形状を、本番の勾配計算を有効にして評価します。初期形状が非物理なら停止します。
2. 物理チェックを通過した形状とその勾配からAdamの候補更新を作ります。
3. 候補形状を全最適化波長で評価し、物理チェックと勾配の有限性を確認します。
4. 通過した候補だけを受理します。その勾配を次の更新に再利用します。
5. 非物理な候補では、形状・Adamのstepカウンタとモーメント・schedulerを更新前へ戻し、学習率を半分にして再試行します。非物理な電力値を目的関数に採用したり、反射率をクリップしたりしません。

通常の受理更新ごとに、別の確認用forwardは追加しません。初期形状と各候補を勾配付きで評価するため、500更新では原則501回の勾配付き評価です。棄却があればその評価分の時間が追加されます。

設定は `physical_backtracking_factor=0.5`、`physical_backtracking_max_retries=12`、`physical_backtracking_min_lr=1e-6` です。最初の提案と最大12回の縮小再試行を行います。上限・下限に達しても候補が通らない場合は `backtracking_exhausted` として明示的に停止し、trialは成功扱いにしません。通過済みの最良形状はtrial内の `best.json` と `best_profile.csv` に保存します。

OOM、割り込み、実装上の例外は物理的な棄却として隠しません。中断時には最後に通過した形状とAdam状態から再開できます。初期評価前のチェックポイントだけは未検証で、`current_state_validated=false` と記録します。

## 実行

以下はそれぞれ1行のLinuxコマンドです。

### 検証コード（CPU、小規模）

```bash
( set -e; cd /home/7/uq06557/common2/20260813/tsoyutarcwa/Optimization/Ag/ag_motheye_P200_H500_Nz15_guarded_v4; python validate_guarded_adam.py; )
```

### 本番の1 trial（CUDA、M20・grid512・15層）

```bash
( set -e; cd /home/7/uq06557/common2/20260813/tsoyutarcwa/Optimization/Ag/ag_motheye_P200_H500_Nz15_guarded_v4; python run.py --device cuda:0 --mode optimize --lattices hexagonal --skip-audit --timeout-hours 0; )
```

出力はケース内の `results/hex_trial1_M20_Q512_guarded_v4` です。500回の受理更新、または既存のpatienceによる早期終了後に停止します。検証・スペクトル取得は自動では始まりません。中断後の再開も同じコマンドです。FAILとなったtrialは、同じ `trials=1` の出力先を指定し直しても新たなtrialにはなりません。

### 実行中のログ（別の端末）

```bash
tail -f /home/7/uq06557/common2/20260813/tsoyutarcwa/Optimization/Ag/ag_motheye_P200_H500_Nz15_guarded_v4/results/hex_trial1_M20_Q512_guarded_v4/hexagonal/trials/trial_0000/worker/worker.log
```

`REJECT update ...` に棄却した候補の波長・電力値・学習率、`accepted step ...` に受理した更新が出ます。

## 終了後の解析

### 次数・gridの確認

```bash
( set -e; cd /home/7/uq06557/common2/20260813/tsoyutarcwa/Optimization/Ag/ag_motheye_P200_H500_Nz15_guarded_v4; python run.py --device cuda:0 --mode convergence --lattices hexagonal --orders 20,24,28 --grids 512,1024 --wavelengths 400:700:10 --timeout-hours 0; )
```

これは次数・gridの収束判定を行う後処理です。Savedやstatus=completedだけでは収束を意味しません。`numerical_convergence_passed` と各条件の差を確認します。

### 計算時間（逆伝播込み）

```bash
( set -e; cd /home/7/uq06557/common2/20260813/tsoyutarcwa/Optimization/Ag/ag_motheye_P200_H500_Nz15_guarded_v4; python run.py --device cuda:0 --mode timing --lattices hexagonal --order 20 --grid 512 --wavelengths 400:700:10 --repeats 3 --timeout-hours 0; )
```

このtimingモードは比較のためreferenceとautoの両方を実行します。autoの `warm_median_seconds_per_wavelength` が波長間共有ありの値です。`--forward-only` を付けると逆伝播を除きます。

実際の最適化における受理候補の評価時間は `history.csv` の `evaluation_seconds`、棄却候補を含む評価時間はチェックポイントの `stats.total_evaluation_seconds` に記録されます。物理的な候補更新の再試行は `backtracking.csv` に記録されます。

### スペクトル（次数・gridは収束確認に応じて指定）

```bash
( set -e; cd /home/7/uq06557/common2/20260813/tsoyutarcwa/Optimization/Ag/ag_motheye_P200_H500_Nz15_guarded_v4; python run.py --device cuda:0 --mode spectrum --lattices hexagonal --order 20 --grid 512 --wavelengths 400:700:3 --timeout-hours 0; )
```

この例のM20・grid512が収束済みという意味ではありません。採用した次数・gridに置き換えてください。波長間隔を細かくした際に新たな非物理値が見つかる可能性もあるため、物理チェックは後処理でも維持します。

再試行上限で停止したtrial内の通過済み最良形状を解析する場合は、上記の後処理コマンドに次を追加できます。

`--profile-json results/hex_trial1_M20_Q512_guarded_v4/hexagonal/trials/trial_0000/best.json`

これは未完了の最適化から保存した候補形状の解析です。trialが完走したことにはなりません。

## 検証済みの範囲

CPUの9件の回帰テストが通過しました。棄却なしで旧Adamと同じ更新軌跡となること、初回およびモーメント蓄積後の棄却でAdam内部状態・stepカウンタが復元されること、中断後の学習率・更新軌跡の一致、再試行上限で成功扱いにしないこと、初期形状の不合格、非有限勾配、予期しない例外の伝播、実際のAg RCWA（M2・grid48・5層・3波長）で2回の受理更新を確認しています。

本番のCUDA・M20・grid512・15層・31波長で完走できるかは、まだ実測していません。小規模CPUテストはその保証ではありません。

追加の接続確認として、実際のOptunaの1 trial（2受理更新）、最適化後の停止、逆伝播込みtimingのreference/auto一致、スペクトルCSV・図の生成も、小規模CPU条件で通過しています。M20・grid512・15層の設定読み込みとprepare-onlyも確認しています。
