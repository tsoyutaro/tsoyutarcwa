# GIL両界面の次数収束からM640スペクトルまで一括実行

RCWAルートで次の1コマンドを実行します。

```sh
python studies/pmma_sheet_relief_1d/automation/run_gil640_spectra101.py --device cuda:0
```

1. GIL=0/0.5/1/2 nm、両界面について、700 nmでM16から16刻み、最大640まで自動収束を実行します。
2. 4条件すべてが収束またはM640上限に達したら、各GILでM640固定のスペクトルを順番に計算します。
3. 波長400～700 nm、3 nm刻みの101点、p偏光、空気溝/水溝です。4 GIL×101波長×2溝=808点です。

共通の形状はP=長辺3720 nm、短辺=高さ1500 nm、Au公称50 nm、Nz960指定です。位相平均は許容差0.00005、最大262144点。次数比較の許容差は0.001です。FFF射影やGILの定義、既存の高速化は変更していません。

収束すれば640未満でも第1段は停止します。スペクトルは常に640固定です。M640上限で未収束でも指定どおりスペクトルへ進みます。700 nmの次数収束だけでは、全波長・層数の収束を保証しません。

## 保存先と再開

- 収束：`results/fff1d/P3720_w700_Nz960_GIL/`。前回と同じ場所で、0 nmとbothの完了済み次数を再利用します。親のgil_summaryは今回の4 GILの一覧になります。
- スペクトル：`results/fff1d/spectrum_P3720_M640_Nz960_GIL0_both/`、`GIL0.5_both/`、`GIL1_both/`、`GIL2_both/`。
- 全体状態とログ：`results/fff1d/workflow_M640_spectra101_both/workflow.json` と `logs/`。

中断後は同じコマンドで再開します。既存の収束チェックポイントと波長点キャッシュを使います。完成済みスペクトルも出力を再生成しますが、同じ条件で保存済みの波長点は再計算しません。GPU側では前回と同じコード・Python環境・計算ルートを使ってください。

各スペクトルの `spectra.csv`、`diffraction_orders.csv`、`completion.json` とR/T/A図は従来と同じです。

## 停止と確認

次数上限到達による終了コード2は通常の続行対象です。位相未収束もコード2になる場合があるため、4条件のgil_summaryを検査し、`converged` / `limit_reached` 以外ではスペクトルへ進みません。OOM・その他の失敗でも進みません。収束ドライバ内部は既存どおり、あるGILが失敗しても残りのGILを試す場合があります。

スペクトルでOOM・位相未収束・計算エラーが出たら、残りのGILを始めず停止します。有限の非受動値（負の吸収等）は保存して4 GILを計算し終え、`completed_with_nonpassive_points` と終了コード2で報告します。数値を切り詰めません。

計算を始めず、実行予定の5コマンドを表示するには末尾に `--dry-run` を付けます。保存先を変える場合は `--convergence-dir`、`--spectrum-root`、`--output-dir`、別のRCWAルートには `--solver-root` を指定できます。

## 導入と検証

追加ファイルは `automation/` と `validation/workflows/` に配置し、元のルートPythonおよび `fff1d/*.py` は変更していません。既存の通常RCWA・FFFの物理ソース署名とキャッシュキーを維持しています。

別のGPU計算機では追加ZIPをRCWAルートへ展開してください。先頭パスは `studies/pmma_sheet_relief_1d/` です。新しい実行ファイルを `fff1d/` 直下へ置かないでください。

実行制御の検査：

```sh
python studies/pmma_sheet_relief_1d/validation/workflows/test_gil640_spectra101.py
```

上限到達・早期収束・位相失敗・OOM・中断・同時実行・非受動値の扱いを、模擬子計算で検査します。これらの検査で大規模なGPU計算は実行しません。
