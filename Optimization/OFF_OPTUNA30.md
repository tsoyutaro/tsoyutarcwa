# 正則化OFFでの金属モスアイ最適化

通常のRCWAと独立実行用ケースの `NVMOptions.normal_mode` の既定値は `"off"` です。
NVMと補助法線場は保持し、中心・最近接セル境界での有限幅smoothstep平滑化を除きます。
中心点はI/2、等距離点は最近接する全法線射影の平均です。Lanczosとは別の設定です。
以前の場は `NVMOptions(normal_mode="on")`、最適化設定では `"normal_mode": "on"` で明示できます。
参照計算、直接行列構築、波長間共有、streaming検証で同じ設定を使用します。

## 実行

Linuxの更新済みOptimizationフォルダで実行します。

```sh
cd /home/7/uq06557/common2/20260813/tsoyutarcwa/Optimization
python validate_off_optuna30.py --device cuda:0 --output results/cuda/off_optuna30_connection.json
python run_metal_off_optuna30.py --device cuda:0 --grid 512 --materials Al,Ti,Cr,Cu,Au
```

検証コマンドは15層・M=2・Q=48・3波長での接続・勾配確認です。本番次数での収束検証ではありません。
設定と実際の起動コマンドだけを確認するには `--prepare-only` を追加します。

| 材料 | 既存の実行フォルダ | 最適化次数M | grid Q |
|---|---|---:|---:|
| Al | Al/al_motheye_P200_H500_Nz15 | 8 | 512 |
| Ti | Ti/ti_motheye_P200_H500_Nz15 | 8 | 512 |
| Cr | cr_motheye | 8 | 512 |
| Cu | Cu/cu_motheye_P200_H500_Nz15 | 12 | 512 |
| Au | Au/au_motheye_P200_H500_Nz15 | 14 | 512 |

全ケースは周期200 nm、高さ500 nm、15層、六方格子、分散あり、法線入射・x偏光です。
400–700 nmを10 nm間隔の31点で計算し、平均反射率を最小化します。
各材料30 trial、Adam最大500 stepです。早期終了とMedianPrunerは従来どおりです。

最適化gridの既定値は512です。`--grid`で明示して変更できます。
実行用configは各ケース内の `config_hex_OFF_optuna30_M{M}_Q512.json`、
結果は `results/hex_OFF_optuna30_v1_M{M}_Q512` に保存します。
旧結果への追記を避けるため、新しい保存先を使います。既存run.pyを子プロセスとして実行します。
`_normal_off_v1`版の学習率のみの探索は使用しません。

## OptunaとAdamの役割

- `lr`: Adamの初期学習率。従来の0.01–1.0を対数探索します。
- `init_top`: 初期形状の最上層半径。1–99 nmを探索します。
- `delta_ratio`: 下端半径を決める0–1の係数です。

下端半径は `init_top + delta_ratio * (radius_max_nm - init_top)` で、
上限radius_max_nm直前に丸めます。層間を線形補間して初期形状を作成します。
Adamはその後、単調性を保つsoftmaxパラメータを通して全層の半径を更新します。

trial 0は従来の既定円錐台（層中央の半径8,14,…,92 nm、lr=0.03）です。
それ以降はOptunaが3変数を選びます。以前のON最適形状は初期値に読み込みません。
30 trialにはCOMPLETE・PRUNED・FAILを含み、30個の成功形状を保証する設定ではありません。

## 停止・再開と検証

`--mode optimize`で各材料の最適化完了時に終了し、検証やスペクトルを自動実行しません。
通常実行では、非物理値はそのtrialをFAILとして次のtrialへ進みます。
`--auto-order`を追加すると、確認済みの非物理的なpower fluxが出たtrialをFAILとして保存し、
その次数のstudyを停止します。次のtrialには進まず、M+2の別studyをtrial 0から開始します。
OOMではその材料を停止し、バッチは次の材料へ進みます。OOMによる次数の増加は行いません。

途中で停止した場合は同じコマンド・config・保存先で再開できます。
ロックが残っている場合のみ、前のプロセスが終了したことを確認して `--recover-lock` を追加してください。
ソースや設定が異なる実験への混在はmanifestで拒否します。

特定材料だけ実行する例:

```sh
python run_metal_off_optuna30.py --device cuda:0 --grid 512 --materials Cu
```

完了後のconvergence・timing・spectrumは、各ケースの従来のrun.pyに同じconfigとoutput-rootを指定して別に実行します。
従来の固定スナップショット検証、既存のOFF専用版、実験的guarded版は再現用として変更していません。

## 非物理値で次数を自動的に上げる

AuのM=14で既に非物理値を確認した場合、現在のAuプロセスをCtrl+Cで終了し、
終了を確認してから、次のコマンドでM=16を新しく開始できます。他材料を停止する必要はありません。

```sh
python run_metal_off_optuna30.py --device cuda:0 --grid 512 --materials Au --auto-order --start-order 16 --max-order 40
```

- M=16で30 trialを開始し、非物理値を検知したら18、20…と2ずつ増加します。
- 各次数は新しいjournal・Adam初期状態・trial 0から開始し、下位次数の最適形状を引き継ぎません。
- OOM・手動中断・上限次数40での非物理値では、それ以上の次数を実行しません。
- `--start-order`省略時は材料別の既定次数から開始し、`--max-order`省略時の上限は40です。
- `--auto-order`の保存先は `results/hex_OFF_optuna30_auto_v1_M{M}_Q512` です。
  通常実行の `hex_OFF_optuna30_v1_M14_Q512` は保持します。
- 再実行時は中断した同一次数のstudyを再開します。既に非物理値で棄却した次数は、
  元のmanifest確認を行ってから、保存済みの `order_retry.json` に従って次の次数へ進みます。
- 発生trial・エラー・要求次数は `order_retry.json` と `order_retry_history/` に記録します。
- 最適化次数が従来のverification_orders上限を超えた場合、新しいconfigの検証次数にその次数を追加します。

次数切り替えは別プロセスの起動制御で行います。数値ソルバ・Adam・従来driver・runtime_policyを変更しないため、
他材料の実行中ジョブや元のsource manifestへの影響を避けられます。更新は起動済みのAuプロセスには適用されません。

## 周期・高さごとに独立した最適化

`--period-nm` と `--height-nm` で1条件を指定できます。15層・六方格子・正則化OFF、
31波長、30 trial、Adam最大500 stepと3変数のOptuna探索を引き継ぎます。
周期200 nm・高さ500 nm以外では、元の材料ケース内の
`geometry_cases/P{P}_H{H}_Nz15/` にconfigとresultsを分けて保存します。
各条件は独立したstudy・checkpoint・lockを持ち、別GPU・別PCで個別に起動できます。
ソルバと材料CSVは元のケースから使用するため、別PCへはリポジトリ全体を配置してください。

```sh
python run_metal_off_optuna30.py --device cuda:0 --grid 512 --materials Al --period-nm 200 --height-nm 100 --auto-order --max-order 40
```

高さ4条件（周期200 nm）を順に起動する例。`materials`を`Al`など1材料に変更できます。
通常の実行エラーやOOMでは、その条件の材料を止め、残りの材料・条件へ進みます。
手動中断などexit code 128以上では全体を終了します。

```sh
( cd /home/7/uq06557/common2/20260813/tsoyutarcwa/Optimization || exit 1
materials=Al,Ti,Cr,Cu,Au
failed=0
for h in 100 200 300 400; do
  if python run_metal_off_optuna30.py --device cuda:0 --grid 512 --materials "$materials" --period-nm 200 --height-nm "$h" --auto-order --max-order 40; then rc=0; else rc=$?; fi
  if [ "$rc" -ge 128 ]; then exit "$rc"; fi
  if [ "$rc" -ne 0 ]; then failed=1; fi
done
exit "$failed"
)
```

周期5条件（高さ500 nm）は、上記の`for`ループを次のように置き換えます。

```sh
for p in 100 150 250 300 350; do
  if python run_metal_off_optuna30.py --device cuda:0 --grid 512 --materials "$materials" --period-nm "$p" --height-nm 500 --auto-order --max-order 40; then rc=0; else rc=$?; fi
  if [ "$rc" -ge 128 ]; then exit "$rc"; fi
  if [ "$rc" -ne 0 ]; then failed=1; fi
done
```

半径上限は`P/2 - 0.001 nm`です。trial 0の連続円錐台の頂部・底部半径は
`0.025P`・`0.475P`とし、従来の誘電体の周期別ケースと同じ比率を用います。
初期最上層半径の探索範囲は`1 nm`から`min(99 nm, P/2 - 0.011 nm)`です。
これは既存Cr driverの探索範囲にも合わせています。Adamはこの初期値範囲に限定されず、
全層の半径を`radius_min_nm`から`radius_max_nm`の範囲で更新します。
開始次数はAl/Ti/Cr=8、Cu=12、Au=14です。Auだけ16から始める場合は
`--materials Au --start-order 16`を指定します。
非物理値を検知した場合は、その条件だけM+2の新しいstudyをtrial 0から開始します。
最適化完了後の検証・スペクトルは従来どおり別コマンドで実行します。
