# Al・Cu・Auの周期・高さ別最適化と再開

`run_metal_period_backtracking.py` は、既存の復旧版 `run.py` を使い、条件ごとに独立したOptuna studyを実行します。
既定値は高さ500 nm、15層、六方格子、grid 512、30 trial、Adam最大500更新、正則化OFFです。
次数の既定値はAlが8、Cuが14、Auが20です。非物理値に対する復旧は従来と同じで、直前の正常状態に戻し、学習率を半減して最大4回再試行します。
初期形状自体が非物理値となる場合の扱いを含め、既存の最適化エンジンは変更していません。

## 実行例

Optimizationフォルダで、材料ごとに別のGPU・サーバーから実行できます。

```bash
python run_metal_period_backtracking.py \
  --materials Al --periods 100,150,250,300,350 \
  --height-nm 500 --order 8 --grid 512
```

```bash
python run_metal_period_backtracking.py \
  --materials Cu --periods 100,150,250,300,350 \
  --height-nm 500 --order 14 --grid 512
```

1周期だけ実行・再開する場合は `--period-nm` を使います。

```bash
python run_metal_period_backtracking.py \
  --materials Al --period-nm 150 --height-nm 500 --order 8 --grid 512
```

`--periods` と `--period-nm` は同時には指定できません。
必要に応じて `--device cuda:0`、`--layers`、`--trials`、`--adam-steps` も指定できます。

Auの高さ依存性を調べる場合は、次のように指定します。高さごとに独立したstudyを順に実行します。

```bash
python run_metal_period_backtracking.py \
  --materials Au --period-nm 200 --heights 100,200,300,400,500 \
  --order 20 --grid 512
```

1つの高さだけ実行する場合は、`--heights ...` を `--height-nm 300` などに置き換えます。
`--heights` と `--height-nm` は同時には指定できません。

## 再開

同じ設定・コマンドを再実行すると、同じ出力先のOptuna journalとAdam checkpointから再開します。
完了済みの条件はスキップし、次の周期へ進みます。最適化後の収束確認・スペクトル計算は自動実行しません。

古いロックで止まる場合は、前のcontrollerとworkerが停止済みであることを確認して `--recover-lock` を追加します。

```bash
python run_metal_period_backtracking.py \
  --materials Al --periods 100,150,250,300,350 \
  --height-nm 500 --order 8 --grid 512 --recover-lock
```

同じホストのロック所有プロセスが生存している場合は、`--recover-lock` があっても起動を拒否します。
別ホストのプロセスは判定できないため、`--recover-lock` はその旧ジョブも停止済みであることを指定するものです。
同じ条件・出力先を複数ホストから同時実行しないでください。

既存の実験を再開するときに設定内容を上書きしません。指定条件が保存設定と異なる場合は停止します。
次数・gridなどを変える場合は別の実験として扱います。

## 既存結果との対応

- Alの周期別ケースは既存の `Al/al_motheye_P..._H..._Nz..._backtracking_v1` を再利用します。
  次数14には `_M14` が付くため、次数8の新規結果と混在しません。
- AlのP100/H500/M8では、存在する `backtracking_from10_v1` のstudyを優先します。
  未作成で旧M8 studyがある場合は、既存のimport処理でtrial 0～9を引き継ぎ、trial 10から復旧版を実行します。
- Cuの周期・高さ別ケースは `Cu/cu_motheye_P200_H500_Nz15_backtracking_v1/geometry_cases/P..._H..._Nz...` を使います。
- CuのP200/H500/M14では、既存の `backtracking_from15_v1` があれば再利用します。
- Auの周期・高さ別ケースは `Au/au_motheye_P200_H500_Nz15_backtracking_v1/geometry_cases/P..._H..._Nz...` を使います。
  P200/H500/M20では、従来の解析に使用した `backtracking_from5_v1` を優先します。
  古い中断済みの通常studyも保存したままにします。完了済みのfrom5 studyは再計算しません。
- 周期に合わせて半径上限・初期形状の探索範囲を設定します。既存のCu高さ別設定も再利用できます。
- Al・Cuで既存の基準studyとimport済みstudyの両方がある場合は、曖昧な選択を避けて停止します。
  `--run-tag hex_OFF_optuna30_backtracking_from10_v1` のように出力先の接頭辞を明示できます。

設定と起動コマンドだけ確認し、RCWAを開始しない場合は `--prepare-only` を追加してください。
この指定では、必要なケースと設定ファイルを作成しますが、trialのimport・Optuna・RCWAは起動しません。

## 起動処理の検証

```bash
python -m unittest test_run_metal_period_backtracking -v
```

一時フォルダを使い、設定分離・checkpoint保持・再開先選択・完了済みスキップ・中断・ロック保護を確認します。
このテストは実際のRCWA計算を行いません。
