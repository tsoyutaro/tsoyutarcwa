# 非物理値で次数を+2して、初期状態から1 trial再最適化

2026-11-02追加。既存の単純なforループ/run.py単独実行には、次数を上げる機能はありません。この機能は `optimize_adaptive_order.py` で実行します。

| 金属 | 開始次数 | 上限 |
|---|---:|---:|
| Al | 8 | 32 |
| Ti | 8 | 32 |
| Cr | 8 | 32 |
| Cu | 12 | 32 |
| Au | 16 | 32 |

正則化OFF（normal_mode=off、LanczosフィルタOFF）、六方格子、周期200 nm、高さ500 nm、15層、grid512、400–700 nmの10 nm間隔31波長。PMMA芯と金属外側の両半径を最適化し、平均A=1−R−Tを最大化します。各次数は独立した1 trialで、Adam最大500受理更新、patience100です。初期学習率0.03、初期半径はPMMA芯5→65 nm、金属外側35→95 nmを使用します。

## 実行

配布ZIPの中は `Optimization/PMMA_metal/` です。研究ルート `/home/7/uq06557/common2/20260813/tsoyutarcwa` に展開して更新した後、次を実行します。

```bash
(
  set -e
  cd /home/7/uq06557/common2/20260813/tsoyutarcwa/Optimization/PMMA_metal
  python optimize_adaptive_order.py \
    --device cuda:0 \
    --materials Al,Ti,Cr,Cu,Au \
    --start-orders Al:8,Ti:8,Cr:8,Cu:12,Au:16 \
    --max-order 32 \
    --grid 512 \
    --output-root results/adaptive_trial1_normal_off_v1
)
```

個別に実行する場合は、例えばCuなら `--materials Cu --start-orders Cu:12` として、別の出力先を使います。GPUメモリを抑える既存の層チェックポイント機能は `--checkpoint-layers` で使用できます。

## 非物理値を検知した場合の処理

1. 初期形状または最適化中の候補更新で `NonphysicalResult` が発生すると、その次数のtrialを失敗として保存します。有限性や受動性の判定条件は従来どおりです。非有限の目的関数・勾配も同じ型で検知します。
2. この自動モードでは、学習率を下げて同じ次数で候補を再試行しません。最初の非物理値で、その次数の最適化を終了します。
3. 次数を+2します。例えばCuは12→14→16→…→32です。
4. 新しい次数専用の別フォルダで、初期半径・初期学習率・Adamのモーメント・scheduler・Optuna studyを新規に作成します。前の次数の最良形状やAdam状態を初期値として流用しません。
5. 最適化が正常に完了したら、その金属の次数上昇を終了し、次の金属へ進みます。
6. M32でも非物理値なら、その金属を `max_order_reached` として終了し、次の金属へ進みます。M34は実行しません。
7. CUDA/CPU OOMの場合は次数を上げず、その金属を停止して次へ進みます。通常のコード・設定エラーはキャンペーン全体を停止します。

正常終了はearly stoppingを含む1 trial完了を意味します。物理チェック通過だけで回折次数の収束が証明されるわけではありません。収束確認は保存した最良形状で別途行います。

## 結果と再開

```text
results/adaptive_trial1_normal_off_v1/
  campaign_manifest.json
  campaign_state.json
  Al/M8/config.json
  Al/M8/controller.log
  Al/M8/run/hexagonal/trials/trial_0000/
  Al/M10/...   # M8で非物理値が出た場合
  Ti/...
  Cr/...
  Cu/M12/...
  Au/M16/...
```

`campaign_state.json` の各金属のstatus、selected_order、attemptsを確認します。completed、max_order_reached、cuda_oom/cpu_oomは区別して記録します。1つでもOOM/上限到達があれば、全体はpartial、終了コードは1です。終了済みの金属の結果は保持します。

同じコマンドで再実行すると、非物理値で終了した次数は飛ばし、完了済みの金属は再実行しません。通常の中断やOOMが起きた次数は、その次数の保存済みAdam状態から再開します。これは「非物理値で次の次数に進む際は初期状態に戻す」という動作とは異なります。

lockが残った場合は元のジョブが終了していることを確認したうえで `--recover-lock` を追加します。設定・ソースが異なる出力先への再開は拒否します。更新前の結果とは混ぜず、新しい出力先で開始してください。

`--prepare-only` は初期次数の設定展開のみを行います。準備用には別の出力先を使います。実行済みキャンペーンの履歴を準備処理で置き換えることは拒否します。

## 追加・変更した部分

- `optimize_adaptive_order.py`：金属ごとの開始次数、+2の制御、上限32、結果保存、再開。
- 各ケースの`optimizer/engine.py`：`nonphysical_action=restart-higher-order`なら最初の非物理候補で停止。省略時は既存のbacktrack処理を維持。
- `optimizer/driver.py`：構造化した非物理終了をtrial失敗として記録。
- `config_adaptive_trial1_M*_Q512.json`：開始条件を明示した設定。
- Crの独立ケース：既存 `Optimization/cr_motheye/data/Johnson.csv` をそのままコピー。金属CSVのraw SHA256は `c88e310d75af57a460a267ec3f00ab24613bab0b8903455775f9db6e3f8d610f`。PMMAの表と形状定義は他のPMMA＋金属ケースと同じです。
- Agの既存ケースも保持していますが、今回の既定実行対象はAl/Ti/Cr/Cu/Auです。

## 検証

CPUで43件のテストを、必要なプロセス分離・短いWindows作業パスで確認しました。初期・候補の非物理値、初期化、+2、上限32、OOM、再開、既存backtrackingを含みます。従来の収束制御テストはtorch未importを前提とするため、torchを使うテストとは別プロセスで実行します。

実際のOptuna/Adam/RCWAを使った検証コピーでは、M1の2回目の評価で非物理値を注入し、M1停止→M3で初期状態から再開始→1更新完了を確認しました。注入コードは検証コピーだけにあります。本番physics.pyは変更していません。

Crは3層/M2/grid48/3波長で、reference/autoのR/T/A・全勾配一致、半径有限差分、lossless、Fresnel対照も通過しました。全5金属の本番開始次数の設定展開も確認しました。CUDAでの500更新完走および最適形状の数値収束は未検証です。

詳細は `validation_reports/adaptive_order_20261102/` に保存しています。
