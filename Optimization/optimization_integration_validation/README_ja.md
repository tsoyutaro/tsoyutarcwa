# 本番最適化と波長共有版の接続検証

本番の geometry.py、physics.py、engine.py を変更せずコピーし、従来の評価関数と FastEvaluator を、同じ Adam エンジンの evaluator 引数に渡して比較します。対象は n=2.5 と、CSV の分散を使う PMMA、それぞれの正方格子・六方格子です。

配置先は `20260806側のoutputs/Optimization/optimization_integration_validation` です。このフォルダ全体、または同階層の `本番最適化_波長共有_接続検証.zip` を別PCへコピーできます。既存の CUDA 検証が動いた PyTorch 環境を使ってください。追加の Optuna インストールは不要です。

## CUDAで実行するコマンド

現在 `square_spectral_cuda_validation` または `hex_spectral_cuda_validation` 内にいる場合は、最初に移動します。

```bash
cd ../optimization_integration_validation
```

まず小規模の接続検証を実行します。1コマンドで n2.5・PMMA × 正方格子・六方格子の4ケースを順に実行します。M=2、NVM grid=48、5層、400/550/700 nm、Adam 2更新です。

```bash
python validate_connection.py --device cuda:0 --suite quick --output results/cuda/quick
```

合格したら、本番と同じ M=8、grid=512、15層、周期200 nm、高さ500 nm、400–700 nmを10 nm刻みの31波長にした検証を実行します。Adamは接続確認のため3更新に限定します。

```bash
python validate_connection.py --device cuda:0 --suite production --output results/cuda/production_M8_L15
```

別のPCへ分ける場合は、次の4コマンドを各PCで実行できます。全ケースを上の1コマンドで実行済みなら、重ねて実行する必要はありません。

```bash
python validate_connection.py --device cuda:0 --suite production --materials n2.5 --lattices square --output results/cuda/n25_square_M8_L15
python validate_connection.py --device cuda:0 --suite production --materials n2.5 --lattices hexagonal --output results/cuda/n25_hex_M8_L15
python validate_connection.py --device cuda:0 --suite production --materials PMMA --lattices square --output results/cuda/pmma_square_M8_L15
python validate_connection.py --device cuda:0 --suite production --materials PMMA --lattices hexagonal --output results/cuda/pmma_hex_M8_L15
```

各コマンドは `--output` のフォルダに `report.json` を作ります。次を確認してください。

- 全体の `status` が `passed`、`cuda_execution_performed` が `true`。
- 全4ケース、または指定したケースの `status` が `passed`。
- `adam_trajectory`、`checkpoint_resume`、`same_order_streaming_verification` の `passed` が `true`。
- `wrong_backend_checkpoint_rejected` が `true`。

全ケース完了前は全体の status が running になります。終了コードは全ケース合格で0、失敗で1です。失敗したケースは同じ report.json に traceback を記録します。CUDAが使えない場合はCPUへ自動で切り替えず、起動時に停止します。

## 確認する接点

1. 本番の単調 softmax 変換を使い、logits → 半径[nm] → 半径/周期の連鎖を検証します。半径に対する勾配だけでなく、全 layers+1 個の logits 勾配を比較します。
2. 層 checkpoint をOFF/ONそれぞれで損失・R/T・勾配を比較し、OFF/ON間も照合します。基準形状に加え、半径の上下限に近い形状も比較します。
3. 本番の順計算による中心差分と、共有版のlogits方向微分を比較します。差分幅は 1e-4 と 5e-5 の2段階です。
4. 本番で使う arithmetic 平均と、trapezoidal 重みの両方を比較します。
5. コピーした本番 engine.train をそのまま呼び、従来版・共有版の Adam 更新、最終/best logits、半径、履歴の損失と学習率、optimizer の moments と step、scheduler 状態、best_step、stale を照合します。固定初期値は init_top=8 nm、delta_ratio=0.8、lr=0.03、訓練時の層 checkpoint はONです。
6. 共有版を1更新後に制御された KeyboardInterrupt で中断し、保存済み adam_checkpoint.pt を読み、FastEvaluator を作り直して再開します。連続実行とモデル・optimizer・schedulerの状態を比較します。
7. backend の署名が違う checkpoint の読み込みが拒否され、ファイルが書き換わらないことを確認します。
8. 得られた best 形状を本番の solve_verify で再評価します。同じM・gridでの層ストリーミング計算と、共有版の全波長R/T・保存best_lossが一致することを確認します。

production の損失・全logits勾配・Adam軌跡・best形状のストリーミング照合は31波長です。独立有限差分は計算量を抑えるため既定で400/550/700 nmの3波長です。全31波長でも差分を確認したい場合は、別の出力先に `--fd-wavelengths 400:700:10` を追加します。report の finite_difference.full_objective_coverage に範囲を記録します。

本番のOptuna探索・pruning制御、500更新後の収束、M=16による次数収束、Crの複素誘電率・吸収は、この検証には含めません。今回使う本番Adamとの接点にOptunaの trial は渡しません。

## PMMAとキャッシュ

PMMAの屈折率は、コピーした本番 materials.material_index で同じ `Szczurowski.csv` を補間します。各波長で n(lambda)^2、逆誘電率の係数、P/Q、portを更新し、共有するのは形状のdisk行列と法線場です。元の本番と同じ k=0 のモデルを使い、404.7 nmより短い400 nmは既存の線形外挿方針に従います。

一定屈折率では波長に依存しないP/Qの係数を共有します。各波長の共役勾配を集め、形状のVJPを1回行った後、本番softmaxのVJPを通してlogits勾配を返します。この評価関数はAdamで使う一次勾配を対象としています。

キャッシュするのは格子・次数・grid・層数・周期・高さ・波長列・材料屈折率列・deviceが同じ場合の静的演算子とportです。半径に依存する行列・計算グラフは評価ごとに再構築します。層checkpointは共有形状部分と各波長の層モード計算に適用します。

n2.5の六方格子の従来版は既存の共通AutoRCWAです。追加PMMAケースの六方格子は、既存設定ですでに D6 native を使用しています。したがってPMMAの比較基準にはD6 nativeの改善がすでに含まれます。

## 保存済みの実際の形状も照合する場合

本番のbest.jsonを指定すると、そこに保存されたbest_logitsの全勾配・損失も追加で比較します。層数はその形状に合わせてください。

```bash
python validate_connection.py --device cuda:0 --suite production --materials n2.5 --lattices square --profile-json /path/to/your/trial/best.json --output results/cuda/n25_saved_profile_M8
```

この指定は保存形状の追加照合です。接続確認用のAdamは上記の固定初期値から実行し、既存の本番checkpointを引き継いで更新することはありません。

## 出力と再開

出力フォルダは新しい空の場所を指定してください。コード・材料設定・実行条件・PyTorch/torcwaとGPUの識別情報を記録し、条件が異なる出力先の流用は拒否します。同じコマンドで再実行した場合は、完了したケースを再利用し、未完了ケースは数値照合を再実行したうえでAdamをcheckpointから再開します。同じ出力先への同時起動は避けてください。

`reference/`、`shared/`、`shared_resumed/` ごとにcheckpoint・history.csv・best.jsonを分けます。既存のn2.5/Crジョブのソース・設定・結果は変更しません。今回の検証で、本番の全RCWAを自動的に切り替える処理はありません。

所要時間の report 値は、短い診断訓練の参考値です。初期化、最終順計算、ファイル保存、キャッシュの有無が含まれるため、従来の benchmark_device.py の速度比と直接比較しないでください。速度測定は既存の正方格子・六方格子のCUDA検証パッケージを使います。

## 同梱ソース

- `validate_connection.py`: 検証コマンドの入口。
- `fast_objective.py`: 本番 evaluator インターフェースに接続する共有版。
- `Optimization/dielectric_motheye_n25/`: 本番n2.5の geometry / physics / engine / common と設定の凍結コピー。
- `pmma_case/optimizer/` と `pmma_case/data/`: 本番PMMAのコードとCSVの凍結コピー。
- `rcwa_ext/`、`d6_nvm.py`、`legacy_block_assembly.py`、`benchmark_square.py`、`hex_core.py` など: 既存の検証済み演算を再利用するためのモジュール。ここでは validate_connection.py から使用します。
- `frozen_production_sources.json`: 48ファイルについてコピー元とSHA256を記録。起動時に凍結コピーの一致を確認します。別PCでは元のWindowsパスにアクセスする必要はありません。
- `verified_cpu_smoke/`: 下記のCPU小規模試験結果。CUDA実機の結果とは区別しています。

## ローカルCPUで実施した結果（2026-10-27）

Python 3.13.5、PyTorch 2.8.0+cpu、torcwa 0.1.4.2 を使用しました。下記はM=2、grid=48、400/550/700 nm、Adam 2更新の接続確認です。CUDAの接続試験とM=8の接続試験は、このPCでは未実行です。

| 材料 | 格子 | 層数 | 損失の最大絶対差 | 全logits勾配の最大絶対差 | Adam最終logitsの最大絶対差 | 中断再開 | streaming照合 |
|---|---|---:|---:|---:|---:|---|---|
| n2.5 | square | 5 | 3.37e-16 | 1.59e-15 | 1.10e-15 | passed | passed |
| n2.5 | hexagonal | 5 | 2.22e-16 | 5.13e-16 | 1.11e-16 | passed | passed |
| PMMA | square | 5 | 2.37e-16 | 2.64e-16 | 3.61e-16 | passed | passed |
| PMMA | hexagonal | 5 | 3.53e-17 | 8.28e-17 | 1.11e-16 | passed | passed |
| n2.5 | square | 15 | 3.62e-16 | 7.24e-16 | 2.64e-14 | passed | passed |
| n2.5 | hexagonal | 15 | 6.59e-17 | 1.47e-15 | 8.19e-16 | passed | passed |

15層の2ケースでは、既存n2.5計算で保存された正方格子のbest_logitsも追加照合しました。同じ半径列を六方格子にも与えて照合しています。次数はM=2へ下げており、この表の値をM=8の実機合格とみなすことはできません。既存のbest.jsonとcheckpointは読み取りのみで、更新していません。

一次勾配の有限差分照合、層checkpoint OFF/ON、arithmetic/trapezoidal、誤ったcheckpoint署名の拒否も全ケースで合格しました。元コードの48ファイルのコピーはすべてSHA256が一致しています。

詳細は [CPU小規模4ケースのreport.json](verified_cpu_smoke/report.json) と [保存済み15層形状のreport.json](verified_cpu_saved_profile/report.json) を参照してください。
