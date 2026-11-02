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
非物理値はそのtrialをFAILとして次のtrialへ進みます。OOMではその材料を停止し、
バッチは次の材料へ進みます。回折次数の自動変更は行いません。

途中で停止した場合は同じコマンド・config・保存先で再開できます。
ロックが残っている場合のみ、前のプロセスが終了したことを確認して `--recover-lock` を追加してください。
ソースや設定が異なる実験への混在はmanifestで拒否します。

特定材料だけ実行する例:

```sh
python run_metal_off_optuna30.py --device cuda:0 --grid 512 --materials Cu
```

完了後のconvergence・timing・spectrumは、各ケースの従来のrun.pyに同じconfigとoutput-rootを指定して別に実行します。
従来の固定スナップショット検証、既存のOFF専用版、実験的guarded版は再現用として変更していません。
