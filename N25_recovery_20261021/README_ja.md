# Gitで取得して復元する（ZIP不要）

このフォルダ全体をGitに追加し、GPU側では `git pull origin master` で取得する。
復元ツールだけでなく、`snapshot/` の保存データ124ファイルと `snapshot_index.json` も必要。
ZIPのダウンロード・展開や、依存ライブラリの再インストールは不要である。

## 実行

元のジョブと誤実行したジョブの両方が停止してから、従来と同じPython環境で実行する。

```bash
cd /home/7/uq06557/common2/20260813/tsoyutarcwa
git pull origin master
python N25_recovery_20261021/restore_n25_snapshot.py --project-root .
python run_n25_nv_optuna_adam.py --device cuda \
  --output-root Optimization/dielectric_motheye_n25/results/NV_Optuna_Adam_M8_verifyM16_recovered_20261021_0909 \
  --timeout-hours 0
```

復元時には新しい結果フォルダを作成する。同名フォルダが既にあれば上書きせず停止する。
元の結果フォルダ、ソースコード、事故後の失敗記録は変更しない。
初回の起動に `--recover-lock` は不要。同じ復元先への同時実行は行わない。

起動時の `Resuming square trial 13` が再開の目印である。最初の進捗表示はstep 160となる見込み。
`--timeout-hours 0` はプログラム側の時間制限をなくす。GPU計算機のジョブ時間制限は別途適用される。

## 復元する状態

- 正方格子のTrial 0～12：完了済みの結果を保持。
- Trial 13：step 155までの履歴と、更新後のAdam状態を保持し、step 156から再開。
- Optuna journal、TPE sampler、Adamのモーメント、学習率調整、最良形状を保持。
- 完了済み最良値：Trial 12、31波長平均反射率0.02790122594028907%（M8）。
- 六方格子：未着手。

2026年10月21日の誤実行前に共有された保存データである。
共有後に元のGPUジョブで進んだ分は含まれず、事故直前の最新状態への完全な復元を保証するものではない。
元のOptunaは4.6.0。通常のcontrollerがソフトウェア環境と計算条件を照合し、
既知のOOM停止処理の更新履歴を記録しつつ元のcheckpoint signatureを保持する。

## 保存データの検査

復元せず、Gitで取得した保存データの整合性だけを確認する場合：

```bash
python N25_recovery_20261021/restore_n25_snapshot.py --inspect-only
```

SHA256を確認してから復元する。`.gitattributes` で `snapshot/` 内の改行変換を無効にしている。
Gitへの追加時にこの属性ファイルも含めること。ソースファイルの照合ではLF/CRLFの差だけを正規化する。
復元の来歴は新しい結果フォルダの `recovery_provenance.json` に記録する。

## 検証

ZIPを使わない復元、既存結果の保持、既存復元先への上書き拒否、破損データと異なる計算コードの拒否、
Gitの改行変換を有効にしたcheckoutでの124ファイルの同一性を、6件のテストで確認した。
実GPUでの再開そのものは未実施。テストは以下で実行できる。

```bash
python N25_recovery_20261021/test_restore_n25_snapshot.py
```

今回の追加は復元ファイルの配置・読み込み方法だけを変更しており、RCWA・NV・Optuna/Adamの計算コードは変更しない。
