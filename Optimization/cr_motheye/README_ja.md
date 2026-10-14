# 六方格子Cr単体モスアイ：NV法によるOptuna + Adam最適化

20260806フォルダの `rcwa_ext` を使い、Cr突起／半無限Cr基板の15層半径を最適化する。
材料を除く既定条件・半径モデル・Optuna/Adamの設定は、先の屈折率2.5版を引き継ぐ。
この実行コードは六方格子のみ、NV法のみを受け付ける。

## 配置と実行

配布ZIPの内容を、`rcwa_solver_auto.py` がある20260806フォルダへ展開する。
ZIPには必要なソルバー・共有処理・Cr光学定数も含まれるため、別の空フォルダでも実行できる。
既存フォルダへ展開するときは、含まれる `rcwa_ext` が実行予定の版と一致することを確認する。

既存のCUDA版PyTorch・torcwa環境を有効にして、そのフォルダで実行する。
CUDA版PyTorchをCPU版へ置き換えないこと。

```bash
python -m pip install -r Optimization/cr_motheye/requirements.txt
python run_cr_nv_optuna_adam.py --device cuda --timeout-hours 0
```

最初のコマンドはOptuna 4.5.0等の補助依存を導入する。PyTorchとtorcwaは既存環境を使う。
`--timeout-hours 0` はプログラム内の時間制限なし。既定値は24時間で、更新境界で保存して一時停止する。
クラスタのジョブ制限を超えない時間を選ぶ。TSUBAMEのジョブ投入設定は既存のGPUジョブ設定を使用する。

結果保存先の既定値：
`Optimization/cr_motheye/results/NV_Optuna_Adam_M8_verifyM16/`

環境・コード・材料・設定が同じであれば、同じコマンドで中断したAdam試行とOptuna探索を再開する。
強制終了で `controller.lock` が残った場合は、元のジョブが終了したことを確認して `--recover-lock` を付ける。
設定・材料・seedを変更した独立実験は、別の `--output-root` を指定する。

## 既定条件

| 項目 | 条件 |
|---|---|
| 格子 | 六方格子（三角Bravais格子）、基本並進ベクトルの長さ200 nm、角度60° |
| 配列 | 単位胞ごとに中心に円形断面1個。最近接の柱中心間距離200 nm |
| 入射 | 空気側から垂直入射、x方向直線偏光、比透磁率1 |
| 構造 | 高さ500 nm、15層、各層厚さ500/15 nm |
| 材料 | 円形突起と半無限基板がともにCr、周囲は空気。酸化膜・PMMA・被覆なし |
| 設計変数 | 15層の半径のみ。高さ・周期・材料・層厚は固定 |
| 半径 | 0～100 nmに近い範囲として0.001～99.999 nm、上から下へ厳密に増加 |
| 基準形状 | 線形テーパーの層中点、半径8,14,...,92 nm。最上下層半径も最適化する |
| 最適化 | NV、次数8、法線場格子512×512、Optuna 30試行、Adam各試行最大500更新 |
| 目的関数 | 400～700 nm、10 nm刻み31点の反射率の算術平均を最小化 |
| 検証 | 400～700 nm、3 nm刻み101点。最適化用の31点も別途同じ条件で再評価 |
| 比較条件 | (M,Q)=(8,512),(8,1024),(16,512),(16,1024)。M変更とQ変更を別々に比較 |
| 学習率 | Optunaが0.01～1を対数探索。初回は基準形状・学習率0.03 |
| 初期形状探索 | 最上層半径1～99 nm、底面への増加比率0～1。単調モデルへ変換 |
| 早期終了 | 改善が小さい状態が100更新。改善判定の絶対閾値1e-8 |
| 学習率調整 | ReduceLROnPlateau、patience=40、factor=0.5 |
| 再現性 | Optuna seed=42、PyTorch seed=42+試行番号、監査方向seed=755。サンプラー状態保存 |
| 精度 | 形状float64、電磁場complex128 |
| 群論 | D6-E1 source-row縮約、x偏光、strict=True。不適合なケースへ自動切替しない |
| フィルタ | use_lanczos=False。NV法線場内部のsmoothstep処理は存在する |
| 層接続 | Redheffer、内部応答ベクトルのみ。近接場は保存しない |
| メモリ方針 | 最適化は層モードを保持して勾配計算、検証はno_gradで層を下から逐次処理 |

## Crの分散と吸収

`data/Johnson.csv` は、20260806の既存Cr計算のユーザー提供データをそのままコピーしたもの。
49点、波長188～1937 nm。`wl,n` と `wl,k` の2ブロックで、`wl` の単位はµm。

出典：P. B. Johnson and R. W. Christy, *Optical constants of transition metals*,
Phys. Rev. B **9**, 5056–5070 (1974), https://doi.org/10.1103/PhysRevB.9.5056 。
掲載ページ：https://refractiveindex.info/?shelf=main&book=Cr&page=Johnson 。

各標本点で複素比誘電率 `epsilon=(n+i*k)^2` を求め、波長に対してepsilonを線形補間する。
固定屈折率2.5は使用しない。400～700 nmの全最適化・検証波長で、突起と基板に同じepsilonを適用する。
範囲外への外挿を拒否し、受動性・波長対応・単位を検証する。
材料ファイルのSHA-256、出典、補間規則、計算に使う分散表を保存する。

| 波長nm | 複素比誘電率epsilon（線形補間） |
|---:|---:|
| 400 | -4.055700 + 11.482900i |
| 550 | -0.962715 + 21.181018i |
| 700 | -2.137858 + 20.675640i |

別の実測Crデータを使用する場合：

```bash
python run_cr_nv_optuna_adam.py --device cuda --cr-csv path/to/cr_nk.csv --output-root Optimization/cr_motheye/results/custom_Cr --timeout-hours 0
```

`wavelength_nm,n,k` の3列CSVも受け付ける。カスタムCSVの相対パスは実行ディレクトリ基準。
論文では実際に使用したデータと成膜条件との対応を記載する。

半無限の吸収性Cr基板には、その裏側の遠方透過ポートがないため `T_far=0` とする。
一方、突起の直下からCr基板へ流入する電力 `P_sub` は計算し、0に置き換えない。

```
A_total = 1-R
A_substrate = P_sub
A_motheye = 1-R-P_sub
```

吸収率は表面の電力収支から求める。独立した体積損失積分による検証ではない。
R、P_sub、A_total、A_motheyeの有限性と0～1の受動性を許容差1e-6で検査し、異常値を丸めて隠さない。
反射率を最小化することは、この半無限Cr系では全吸収率を最大化することと等価。

## 単調半径と勾配

16個のsoftmax変数thetaから `w_i = f + (1-16*f)*softmax(theta)_i`、`f=1e-6` を作り、
15層の半径を `r_j = r_min + (r_max-r_min)*sum(w_1,...,w_j)` とする。
最後のw_16は上限までの余裕。15層には16個の区間が必要で、重みの総和は1。
並べ替えや半径の画素丸めを使わず、円のフーリエ係数・NV法線場・行列計算へ勾配を伝える。
softmaxの共通シフトは冗長であり、形状の自由度は15。
Adam更新後にthetaを±20へ射影するため、端点近傍では制約付きの探索となる。

各波長の目的関数を微分してから勾配を加算するので、全31波長の平均の微分に対応する。
最適化では層チェックポイントを既定で無効にして再計算を避ける。
勾配メモリが不足する場合は `--checkpoint-layers` を付けると、計算時間と引き換えに保持を減らせる。

初期・最適形状の監査は既定で有効。400/550/700 nmで、M8とM16、Q512を比較する。
各次数で全16変数の勾配を計算し、2方向の中央有限差分をh=0.001と0.0005で比較する。
判定は絶対1e-7＋相対5%、次数間のコサイン類似度0.99以上。
**監査に不合格でもM8の最適化を続け、その結果を記録する。監査合格は実行完了とは別。**
監査は3波長の方向微分確認であり、31波長全体・全形状の勾配収束を証明しない。
論文化に向けては `audit_wavelengths_nm` を `400:700:10` に変更した独立実験と、
途中の代表形状、さらに高次数・層数について検証する。
誘電体でのM8/M16収束結果は、Crの勾配・反射率の収束保証にはならない。

## 個別実行・高次数検証・動作確認

最適化と監査のみ：

```bash
python run_cr_nv_optuna_adam.py --device cuda --mode optimize --timeout-hours 0
```

最適形状をさらに高い次数で検証する（同じ結果保存先で実行）：

```bash
python run_cr_nv_optuna_adam.py --device cuda --mode verify --verification-orders 8,16,20,24 --timeout-hours 0
```

これは高次数での前方計算であり、高次数でのAdam再最適化ではない。
各検証波長を別プロセスで実行し、ある(M,Q)条件でOOMが出たらその条件を止め、次の条件へ進む。
OOMを記録した条件は再開時にスキップし、明示的な再試行は `--retry-oom` を付ける。
最適化のOOMはそのAdam試行のチェックポイントを残して実行を停止する。

CPUでの小規模実行確認（本番計算とは別の保存先）：

```bash
python run_cr_nv_optuna_adam.py --device cpu --smoke --timeout-hours 0 --output-root Optimization/cr_motheye/results/cpu_smoke
```

これは15層、M1/M2、Q32/Q48、Optuna 2試行、Adam各2更新の実計算で、物理収束の確認ではない。

GPU上で通常経路と逐次経路の電力一致・メモリ測定：

```bash
python Optimization/cr_motheye/validate_memory.py --device cuda --orders 8,16 --grid 512
```

計算を行わず、設定・材料・出力先の確認だけを行う：

```bash
python run_cr_nv_optuna_adam.py --device cuda --prepare-only
```

## 保存される結果と論文での確認

- `resolved_config.json`、`manifest.json`：設定、ソースSHA、実行環境、実験識別。
- `material_provenance.json`、`material_dispersion.csv`：材料出典・SHA・各波長のepsilon。
- `hexagonal/trials.csv`、`optuna_journal.log`、`sampler.pkl`：探索履歴と乱数状態。
- `hexagonal/trials/trial_*/history.csv`、`adam_checkpoint.pt`、`best.json`：Adam履歴・再開状態。
- `hexagonal/optimized_profile.csv`、`report.json`：最良形状とその形状で測定した目的関数。
- `hexagonal/verification/*.csv`：R、P_sub、T_far、全吸収、突起吸収、基板吸収。
- `hexagonal/audits/*/result.json`：有限差分比較・次数による勾配変化。
- `figures/optimization_hexagonal.png`、`.svg`：Optuna/Adam履歴、反射スペクトル、形状。

小規模の実装検証と本番GPUでの科学的検証を区別する。独立seedでの探索、
同じ最適化波長によるM比較、法線場格子・層分割の収束も確認する。
光学定数の補間は帯域内の数値モデルであり、因果性を拘束した材料フィットではない。
最良形状を大域最適解と解釈しない。
