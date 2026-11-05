# Au M20のAdam backtracking

周期200 nm、高さ500 nm、15層、六方格子、正則化OFF、grid 512、
400～700 nmの31波長の算術平均反射率を最小化します。
Optunaは従来の3変数（初期学習率・初期上端半径・広がり率）を選び、
Adamは単調な全層の半径を更新します。30 trial、各trial最大500回の採用更新です。
初期学習率の探索範囲は従来の0.01～1.0で、trial 0は従来の基準形状とlr=0.03です。

```bash
cd /home/7/uq06557/common2/20260813/tsoyutarcwa/Optimization &&
python Au/au_motheye_P200_H500_Nz15_backtracking_v1/run.py \
  --device cuda:0 --mode optimize --lattices hexagonal --timeout-hours 0
```

設定はこのフォルダのconfig.json、結果は
results/hex_OFF_optuna30_backtracking_v1_M20_Q512 に保存します。
この専用フォルダに新しいstudyを作成し、trial 0から開始します。
旧M20のcheckpointは未評価の更新後形状を含み、更新前のAdam状態を保存していないため、
旧checkpointをこの方式の途中再開データとして読み込むことはできません。

## 非物理値が出た場合

1. 更新前の正常な形状、Adamの両モーメント・step数、schedulerを復元します。
2. 学習率を半分にして、同じ更新元から候補を作り直します。
3. 全31波長で物理判定が通り、目的関数と勾配が有限なら、その更新を採用します。
4. 通常候補の失敗後、lr/2、lr/4、lr/8、lr/16の最大4回を再試行します。
5. 4回とも失敗した場合、そのtrialで最も低い平均反射率を得た正常形状を返します。
   termination_reason=nonphysical_retry_exhaustedとして正常終了し、Optunaは次のtrialへ進みます。

成功した再試行の学習率を、その後の更新でも使います。再試行回数の上限は
各更新ごとにリセットします。反射率の単調な低下は採用条件に含めず、
元のpatienceとMedianPrunerも使います。回折次数はM20に固定します。

最初の形状から非物理値で、正常な復帰先が一度も得られないtrialはFAILとなり、
次のtrialへ進みます。30 trialの数にはCOMPLETE・PRUNED・FAILを含みます。
OOMは数値的なbacktrackingの対象に含めず、そのtrialを再開可能な状態で停止します。

## 保存・再開

Adam checkpointは、全波長で評価できた現在の形状と、その形状の勾配を保持します。
中断した候補やOOM中の候補が、最後に検証した状態を上書きすることを防ぎます。
再試行中の中断では、半減済みの学習率と再試行番号を保存して再開します。

同じコマンドで再開します。ロックが残っている場合は、前の同じジョブが停止済みであることを
確認してから--recover-lockを追加します。

- history.csvは正常評価できた点の履歴です。step 0は初期点で、step kは採用更新k回後の形状です。
- next_stepは採用更新回数です。初期点を含むので、正常勾配評価数は通常next_step+1です。
- recovery.jsonは棄却した候補、学習率、復帰元の評価値を保存します。
- recovery_attemptsは失敗した評価の総数で、通常候補の失敗も含みます。
  例えば通常候補と半減後4回がすべて失敗すると、この値は5です。
- best.jsonのbest_lossとbest_logitsは同じ正常評価形状に対応します。
- trials.csvにはtermination_reasonとrecovery_attemptsも保存します。

## 検証

```bash
python Au/au_motheye_P200_H500_Nz15_backtracking_v1/test_backtracking.py -v
```

rollback後のAdam状態、4回の半減、最良正常形状の選択、中断・時間制限・OOM後の再開、
pruning、旧checkpointの拒否、完了直前の中断を検証します。
本番のM20 CUDA計算は利用者のGPU上で実施します。
