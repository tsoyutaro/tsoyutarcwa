# 屈折率2.5版：同じGPUモデルでのMIG構成変更を許容する再開修正

既存の `tsoyutarcwa` フォルダ（`run_n25_nv_optuna_adam.py` がある場所）へZIPを展開する。
前回のジョブが停止した状態で、次を実行する。

```bash
unzip -o N25_GPU_MIG_resume_fix.zip
python run_n25_nv_optuna_adam.py --device cuda --recover-lock
```

現在のPython環境をそのまま使う。依存ライブラリの再インストールは不要。
時間制限を23時間にする場合は `--timeout-hours 23`、プログラム内の制限を無効にする場合は `--timeout-hours 0` を追加する。
クラスタのジョブ時間制限は別に適用され、計算の区切りでの保存時間にも余裕を確保する。

更新される実行ファイルは次の2個。

- `Optimization/dielectric_motheye_n25/run.py`
- `nv_runtime_policy.py`（新規、プロジェクト直下）

計算条件・半径モデル・NV法・群論・目的関数・Adam・Optunaの数値計算本体は変更しない。
Cr版のドライバはこの修正では変更しない。

## 許容する変更

- `NVIDIA H100 MIG 4g.47gb` → `NVIDIA H100 MIG 3g.47gb` など、同じGPUモデル内のMIG構成変更。
- 同じGPUモデル内でのGPUメモリ容量の変化。
- CUDAデバイス番号の変化。`--device cuda:0` → `--device cuda:1` もCUDA同士の場合に限り許可。

GPUモデルは、報告されたGPU名から ` MIG ...` 部分を除いて比較する。
`NVIDIA H100` と `NVIDIA A100` の切替、CPUとCUDAの切替は拒否する。
小さいメモリへの切替では、既存のOOM処理が引き続き適用される。
速度や浮動小数点の丸めは変わる可能性があり、ビット単位の同一結果は保証しない。

## 引き続き拒否する変更

- Python、PyTorch、PyTorch側CUDA、torcwa、Optunaのバージョン変更。
- 計算条件、回折次数、層数、波長、形状範囲、seedなどの設定変更。
- 電磁場ソルバー・半径モデル・目的関数・Adamなどのソース変更。
- 未知の旧ドライバからの移行。

前回の記録では Python 3.9.25 / PyTorch 2.6.0+cu124 / CUDA 12.4 /
torcwa 0.1.4.2 / Optuna 4.6.0 を使用している。
再開のためにOptunaを4.5.0へ変更しない。元の実験のバージョンを維持する。
`nvidia-smi` のCUDA表示はドライバの対応CUDAであり、ここで照合する値は `torch.version.cuda`。

## 既存結果・再現性の記録

今回の旧版ドライバのSHA-256
`2bc7ef243312b0afa0943ceea2db1b3169a613e579399f431d435548e8e8329a`
から、今回の再開処理だけが変わった場合に、互換移行を許可する。
その他のソース・設定の一致を確認し、元の実験signatureを保持する。
既存のAdamチェックポイント、Optuna試行履歴・サンプラー状態、保存済み検証を引き継ぐ。
保存済み結果ファイルのsignatureを手で書き換える必要はない。

`manifest.json` に次を保存する。

- 元の `signature`、`source_sha256`：従来のチェックポイント識別として保持。
- `active_source_sha256`、`active_execution_signature`：更新後の実行ソース。
- `source_upgrade_history`：今回の既知の再開処理の更新履歴。
- `initial_runtime_environment`：最初の環境。
- `runtime_environment`：今回実行した環境。
- `runtime_history`：変更前後のGPU構成、変更項目、UTC時刻。

manifestの更新前には `manifest.backup.*.json` を自動保存する。
別のコントローラがロックを持つ場合は停止し、ロック取得後のmanifestの競合も検査する。
`--recover-lock` は前回のプロセスが終了している場合に使用する。

GPU変更だけを許容する今回の履歴があっても、異なるGPUモデルやソフトウェア変更には
新しい実験を用意する必要がある。旧ドライバの識別が一致しない場合は、
エラーに表示されるソース差分を確認し、チェックを削除して回避しない。

## 確認範囲

同じH100のMIG構成変更の許可、ソフトウェア・別GPUモデル・CPU/CUDA切替の拒否、
旧manifestの互換移行、元ファイルのバックアップ、競合検出を単体テストで確認。
共有された試行4の実チェックポイントをコピー上で読み込み、次のstep 228、
最良値・パラメータ・履歴を保持することを確認した。
旧版のCPU実行結果を更新版で再開し、既存Optuna studyを引き継ぐことも確認。
実GPUでの4g→3g切替は手元では実行していない。
