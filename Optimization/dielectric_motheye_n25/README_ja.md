# NV法のみ：Optuna＋Adamによる15層モスアイ最適化

20260806の現行`rcwa_ext`を使用する。最適化は**M=8**、固定された最終形状の
検証は**M=16**を既定とする。ASRでの計算は行わない。
正方格子と六方対称（三角Bravais）格子を独立したOptuna studyで計算する。

## GPU環境での実行

20260806のプロジェクト直下（`rcwa_solver_auto.py`がある場所）へファイルを反映する。
既存のCUDA版PyTorch・torcwa環境を使用し、新たにOptunaを追加する。

```bash
python -m pip install -r Optimization/dielectric_motheye_n25/requirements.txt
python run_n25_nv_optuna_adam.py --device cuda --prepare-only
python run_n25_nv_optuna_adam.py --device cuda
```

同じ実行ファイルを専用フォルダから起動してもよい：

```bash
python Optimization/dielectric_motheye_n25/run.py --device cuda
```

既定の保存先は
`Optimization/dielectric_motheye_n25/results/NV_Optuna_Adam_M8_verifyM16/`。
各格子について、初期勾配監査 → Optuna＋Adam → 初期／最適形状の順計算検証 → 最終勾配監査 → 図、
の順で実行する。初期勾配監査に不合格でも指定されたM=8を自動変更せず、判定を記録して探索を続ける。
監査の合否と、Optunaの完了状態は別の指標である。

計算予算は既定24時間で、Adamのステップ間／検証波長の間で途中停止する。
同じコマンドでチェックポイントから再開する。時間制限を外すには`--timeout-hours 0`。
30試行×500ステップ×31波長は**1格子につき最大465,000回の波長別勾配評価**となる。
EarlyStoppingとOptunaのpruningにより実際の回数は減るが、24時間で完了する保証はない。
まず1格子の短時間実行で、`history.csv`の実測ステップ時間とメモリを確認するとよい。

```bash
python run_n25_nv_optuna_adam.py --device cuda --lattices square --timeout-hours 1
```

最適化だけを進める／保存済みの最適形状を検証する：

```bash
python run_n25_nv_optuna_adam.py --device cuda --mode optimize
python run_n25_nv_optuna_adam.py --device cuda --mode verify
```

高次数で追加検証する（最適化のM=8は変わらない）：

```bash
python run_n25_nv_optuna_adam.py --device cuda --mode verify \
  --verification-orders 8,16,20,24,28,32
```

同じ形状・M・Q・波長の成功した検証は再利用する。OOMとなった条件は再実行時に省略する。
GPUメモリを空けて再試行する場合は`--retry-oom`を付ける。
同じ出力先への同時起動は排他ロックで拒否する。
ジョブが強制終了してロックが残った場合は、前のジョブが終了したことを確認して`--recover-lock`を使用する。
両格子を別GPUで並行実行する場合は、`--lattices square`／`hexagonal`に加え、別の`--output-root`を指定する。

## 物理モデルと設計変数

| 項目 | 設定 |
|---|---|
| 材料と半無限基板 | 無損失n=2.5、ε=6.25、μ=1、波長非依存 |
| 周期／高さ | 最近接周期200 nm／高さ500 nm、固定 |
| 高さ分割 | 15層、等厚500/15 nm |
| 設計変数 | 15層の半径。上から下へ単調増加 |
| 半径範囲 | 0.001～99.999 nm。0と100 nmの特異な端点を避ける微小余裕 |
| 入射 | 空気から垂直入射、x偏光 |
| ソルバー | complex128、解析円Fourier係数、NV法、Lanczosなし、Redheffer接続 |
| 対称性 | 正方C2vのx入射セクター、六方D6の入射行 |
| 初期比較形状 | 先端5 nm・底部95 nmの円錐台を15層の中点で評価 |

初期・最適形状の半径は、各層の**等厚円柱の半径**である。
表面の連続補間や、最適化後の勝手な層数増加は実行しない。
初期比較形状の最上層／最下層半径は8／92 nm。
探索範囲は添付に近い0～100 nmであり、底部隙間10 nmは制約として課していない。
製造制約を導入するときは`radius_max_nm`等を変更し、別の実験出力先を使う。

単調Softmaxモデルは独立に実装し、式を明示している。
16個のlogitをθ、f=1e−6、w_i=f+(1−16f)softmax(θ)_iとし、
15個の半径をr_j=r_min+(r_max−r_min)Σ_{i=0}^j w_i（j=0,…,14）とする。
16番目の重みは上限までの余白。Softmaxの共通加算不変性により、実質15自由度。
Adam更新後のlogitは±20へ制限する。このパラメータ範囲も実験条件として保存する。

## 目的関数とOptuna／Adam

添付に合わせ、既定の目的関数は31波長の**算術平均反射率**：

```text
J(θ) = (1/31) Σ R(400 + 10k nm; θ),  k=0,...,30
```

Rは全伝搬反射次数のフラックス合計。基板側Tも全伝搬次数を集計し、R+T=1を確認する。
今回のP=200 nm・λ=400～700 nmでは空気側の反射はゼロ次だけが伝搬する。
目的関数を連続波長積分と呼ばない。`objective_weighting=trapezoidal`で台形積分平均も選べる。
高密度検証の101波長（400～700 nm、3 nm刻み）平均は、31波長の訓練目的関数と区別して保存する。

OptunaのTPEは学習率（0.01～1、対数）、初期最上層半径、初期半径差の割合を探索する。
各試行の15層形状はAdamで最大500ステップ更新する。
最初の試行は初期比較円錐台と学習率0.03で固定し、比較の基準を残す。
MedianPruner、ReduceLROnPlateau、EarlyStoppingを使用する。
試行数30はCOMPLETE／PRUNED／FAILを合計した終了済み試行の予算。
数値エラーはFAILとして記録し、低い目的関数を割り当てない。
最良形状は、損失を評価した時点のパラメータと対応させて保存する。
最後のAdam更新後の形状も評価し、最良試行の保存形状をそのまま検証する。
検証時にM=16でAdamを再実行することはない。

## 速度とメモリ

**最適化**ではM=8、Q=512を使用し、同じ波長の全層モードを保持して微分する。
波長ごとに勾配を計算して加算するため、31波長分の逆伝播グラフを同時保持しない。
不要な全空間P/Qや場の保存配列の参照は各層で解放し、応答は反射・透過ベクトルだけを取り出す。
CUDAのキャッシュを毎波長空にする処理は既定で加えず、再割当の負担を避ける。

最適化の逆伝播でOOMが出る場合は、層モードを逆伝播中に再計算する方式を選べる。
速度との交換条件になるため既定では無効。変更後は別出力先で開始する。

```bash
python run_n25_nv_optuna_adam.py --device cuda --checkpoint-layers \
  --output-root Optimization/dielectric_motheye_n25/results/NV_checkpointed_M8
```

**検証**では、勾配の保存を止め、層を下から1層ずつ固有値計算してRedheffer接続する。
保持するのはポート、現在の1層、累積反射／透過であり、全層のモードや層S行列を同時保存しない。
全Fourier空間へ大きなS行列を展開せず、入射源に対する反射／透過ベクトルだけでR/Tを求める。
波長・次数・Qの各計算を別プロセスとし、終了時にCUDAコンテキストも解放する。
起動の負担があるため、検証は最適化よりメモリを優先する構成。

既定の検証条件はM8/Q512、M8/Q1024、M16/Q512、M16/Q1024。
同じQでMだけを比較し、同じMでQだけを比較することで、次数と積分格子の影響を分離する。
Q≥4M+4を満たさない条件は実行しない。
層方向の保存量は減るが、各層の固有値問題・NV因数分解やFourierモード数に依存する
メモリは残るため、高次数でもOOMが起こる場合がある。
OOMの次数・波長・ピーク割当／予約量を保存し、未完了検証を成功扱いしない。

GPU上で通常の順計算経路と低メモリ検証経路の一致・ピークメモリを確認する専用コードもある。
各経路を別プロセスで測定するため、前の経路のCUDAキャッシュが測定へ混入しない。

```bash
python Optimization/dielectric_motheye_n25/validate_memory.py --device cuda --orders 8,16 --grid 512
```

`results/memory_parity.json`へR/T差とピーク割当／予約量を保存する。
高次数で通常経路がOOMになっても、低メモリ経路の結果は別に保存する。

## 勾配監査と論文で主張できる範囲

初期と最適形状で全logitの自動微分を計算し、2方向の中央差分h／h/2を照合する。
M=8と16の勾配ベクトルの差とcosineも保存する。
既定では400／550／700 nmの3波長で確認するため、31波長目的関数全体の勾配収束を保証しない。
論文用の確認ではconfigの`audit_wavelengths_nm`を`400:700:10`にした別実験を用意し、
必要なら方向数を増やし、重要な設計変数の成分別差分や途中形状も検査する。
近似勾配の合格条件は絶対許容値1e−7＋相対5%、次数間のcosine≥0.99。
目的関数の改善が小さい場合は、要求精度に応じて許容値を厳しくする。

```bash
python run_n25_nv_optuna_adam.py --device cuda --mode audit
```

保存済み最良形状がない場合は初期形状のみを監査する。
試験的に監査を省略するには`--skip-audit`。省略は結果へ記録される。
監査不合格・OOM・省略を、勾配検証の合格とみなさない。

論文化時には、最終形状でM16→20→24等の追加次数、Qの影響、波長点の密度を確認する。
別seedで独立した探索を繰り返し、平均だけでなく最良・分散・計算予算を報告する。
15段の構造を物理モデルとする場合と、連続形状を15層で近似する場合を区別する。
連続形状を主張するなら、形状を保った層数収束も別途必要。
Optuna＋Adamで得た結果は、指定された探索空間内の得られた最良解であり、数学的な大域最適の証明ではない。

## 出力・再現性

- `manifest.json`：設定、ソースSHA256、Python／PyTorch／torcwa／Optuna／CUDA／GPU。
- `resolved_config.json`、`baseline_profile.csv`：実際の条件と初期形状。
- `workflow.json`、各格子の`report.json`：処理状況、最良形状、監査と検証。
- `<格子>/optuna_journal.log`、`sampler.pkl`：試行履歴とTPEの乱数状態。
- `<格子>/trials.csv`：COMPLETE／PRUNED／FAILを区別した試行表。
- `<格子>/trials/trial_*/adam_checkpoint.pt`：次のステップ、現在／最良パラメータ、Adam、scheduler、EarlyStopping。
- 同フォルダの`history.csv`、`best.json`、`best_profile.csv`：損失・時間と対応する形状。
- `<格子>/optimized_profile.csv`：最良形状の半径・直径・隙間・深さ。
- `<格子>/verification/<形状識別子>/M*_Q*.json/csv`：成功した波長、完全性、31／101波長平均。
- `<格子>/audits/`：勾配、方向差分、次数間比較、失敗時のエラー。
- `figures/optimization_<格子>.png/svg`：Optuna履歴、Adam履歴、検証スペクトル、15層形状。
- 各`workers`／`worker`フォルダ：要求・結果・ログ。PyTorchのCUDAピーク割当／予約量。

ソース・物理条件・訓練設定・要求デバイスを変更した場合は別の出力先を使う。
runtimeが異なる実行からの再開も拒否し、異なる精度の結果を混在させない。
検証の追加次数と実行時間予算はコマンドから変更可能。
途中の波長をNaNで埋めたり、失敗波長を除いた平均を正式な検証平均として出したりしない。
JournalStorage＋通常ファイルの排他ロックを使用し、SQLiteや特殊なsymlink権限に依存しない。
チェックポイントとpickleは、このコードが作成した自分の実験フォルダのものだけを読み込む。

図だけ再生成：

```bash
python run_n25_nv_optuna_adam.py --mode plot
```

## 必要なファイルと添付との関係

新しいコードには形状モデル・EarlyStopping相当の状態・ロック・再開機能を組み込んだ。
**追加の材料CSVや、旧`geometry_models.py`／`ml_utils.py`／`hpc_utils.py`の提出は実行に不要。**
材料n=2.5はここでの仮定であり、実材料の分散を扱う論文に変更する場合は測定n/kデータが必要になる。
旧コードの数値結果やSoftmaxモデルを厳密に再現したい場合には、上記の旧補助ファイルが必要。
今回のコードは、添付の探索方針を現行20260806ソルバーへ移したもの。
添付の旧ソルバーや未提供の形状モデルと同一の実装だとは主張しない。

参考API：
[OptunaのAsk/Tell](https://optuna.readthedocs.io/en/v4.5.0/reference/generated/optuna.study.Study.html)、
[JournalFileBackend](https://optuna.readthedocs.io/en/v4.5.0/reference/generated/optuna.storages.journal.JournalFileBackend.html)。

## 手元の動作確認

```bash
python -m unittest Optimization.dielectric_motheye_n25.test_optimization -v
python run_n25_nv_optuna_adam.py --device cpu --smoke --output-root work/nv_optuna_smoke
```

`--smoke`は15層を保ち、M1/M2・小さいQ・2試行×2ステップへ減らす。
実際のOptuna／Adam／NV計算／勾配監査／低メモリ検証を通すための試験であり、
M8/M16のGPU本計算や論文用の最適解ではない。

2026-10-14のローカル確認では、単体テスト7件が合格。
実際のOptunaを使い両格子で合計4試行・8回のAdam更新・32件の波長別検証を完了した。
保存した最良形状の目的関数と再評価値は一致した。
さらに15層・550 nm・Q=96のCPU計算で、正方／六方×M8／M16の4条件について
通常の順計算経路と1層ずつ処理する経路のR/Tが一致した。
小規模M1/M2試験では方向有限差分は合格したが、次数間の勾配収束は不合格として記録した。
GPU本計算、実際のCUDA OOM、GPUピークメモリの削減量は手元では実測していない。


## CUDA OOM表記と同時実行からの復元（2026-10-21）

`CUDA error: out of memory` もCUDA OOMとして停止する。端末のOOMを通常エラーとして
FAILにして次の試行へ進める問題を修正した。既知のGPU/MIG対応版からのcontroller更新を
記録し、保存済みのsignatureと物理計算・最適化条件を維持する。
`--recover-lock` は前のジョブが停止したことを確認してから使う。他ホストの生存確認を行う機能ではない。
誤実行でjournalにFAILが追記された場合は、ロック解除だけでは元に戻らない。
事故前のjournal・sampler・Adam checkpointを一式で別フォルダに復元し、元の失敗記録は残す。

## GPU/MIG構成変更時の既存学習の再開

同じGPUモデル内のMIG構成・メモリ容量・CUDA番号の変更を許容する。
Python/PyTorch/CUDA/torcwa/Optuna、計算条件、計算本体のソースは引き続き照合する。
既知の旧版から再開処理だけを更新した場合は、元のチェックポイントsignatureを保持し、
GPU変更履歴とソース更新履歴、manifestのバックアップを保存する。
既存実験を再開するときは依存を再インストールせず、開始時の環境を維持する。
詳細は `GPU_MIG_resume_ja.md` を参照。
