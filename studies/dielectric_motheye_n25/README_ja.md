# n=2.5モスアイ：六方／正方格子、NV／ASRの比較

無損失の屈折率2.5のモスアイを、同じ材料の半無限基板上に配置する。
NV法と円形境界適合ASRを同じソルバー基盤で明示的に選び、次の4ケースを比較する。

| 格子 | NV | ASR |
|---|---|---|
| 六方対称（三角Bravais格子） | `hexagonal/nv` | `hexagonal/asr` |
| 正方格子 | `square/nv` | `square/asr` |

目的は400–700 nmの平均反射率を下げるため、どちらの計算法を探索・最適化に使うか判断すること。
最適化そのものは実行せず、収束・計算時間・勾配を検査する。
基板側への透過は実在する。材料の吸収はなく、A=1−R−Tは離散化／丸め誤差の指標。

## 初期条件

| 項目 | 既定値 |
|---|---|
| 材料・基板 | n=2.5、k=0、ε=6.25、μ=1、波長非依存 |
| 周期・高さ | P=200 nm、H=500 nm |
| 先端・底部半径 | 5 nm、95 nm。最近接柱間の底部隙間10 nm |
| 形状 | 深さに比例して半径が増える円錐台。各z層の中点半径を使用 |
| 入射 | 空気から垂直入射、x偏光 |
| 四つの計算に共通 | complex128、Redheffer接続、half S行列、場保存なし |
| 対称性 | 六方はD6の入射行、正方はC2vのx入射セクター。同一格子内の両方法で共通 |
| NV | 円の誘電率Fourier係数は解析式、法線ベクトル投影はQ×Q格子、Lanczosフィルタなし |
| ASR | `matched-asr`、一般化因数分解あり、G=0.03、Q×Q積分格子。両格子とも`local-radial`、公称形状の計算座標上の円周を固定 |
| 保持次数M | 4,6,8,10,12,16,20 |
| 高さ分割Nz | 20,40,60,80 |
| NV投影／ASR積分Q | 256,384,512 |
| 初期の判定波長 | 400,550,700 nm。最終チェックで不合格なら必要な波長を自動追加 |
| 時間・勾配監査波長 | 400,430,436,475,495,505,550,600,645,650,700 nm（11点） |
| 方法間の許容変化 | R/T/Aの絶対変化0.001＝0.1パーセントポイント |
| 自己収束の許容変化 | 既定0.00025。方法間許容値の1/4を各方法の誤差予算に使用 |
| 無損失の条件 | −1e−7≦R,T≦1+1e−7、abs(R+T−1)≦1e−6 |
| 最終スペクトル | 指定時、収束したケースだけ400–700 nm、5 nm刻みの61点 |

正方格子では基板側の最初の回折が開く境界がnP=500 nm、六方ではn√3P/2≈433.013 nm。
閾値付近は最終スペクトルと11点の監査で確認する。
R/Tはゼロ次だけでなく全伝搬回折次数のフラックスを集計する。
最終スペクトルには500 nmも含め、閾値でのエネルギー保存を確認する。
`--run-final-spectrum`では全61波長について、M、Nz、Qをそれぞれ候補上限に
増やした3条件と、三軸を同時に上限へ増やした条件も比較する。
すべての最大変化が許容値以下で、エネルギー保存を満たす場合だけ監査へ進む。
不合格なら、各細分化条件で差が最大の波長と非物理的な波長を判定点へ追加して再走査する。
追加は既定で最大2回（`--max-anchor-refinements`）。合格済みの個別ケースは再利用する。
追加後も候補上限が足りない場合は不合格のままとし、監査へ進まない。
指定波長の間の精度や、候補上限より先の収束まで保証するものではない。

両格子は最近接周期を200 nmにそろえる。セル面積が異なるため底面の充填率は
六方約0.8185、正方約0.7088。格子間で光学応答が違うのは計算法の差だけではない。
比較・推奨は**格子ごと**に行う。

対称性を使う六方のscalar harmonic数は3M(M+1)+1、正方は(2M+1)²。
実際の固有値問題の縮約次元も各ケースへ保存する。
これは2次元の円形境界適合ASRで、以前の1D ASRにおける内部3N/4N設定とは別の実装。

## 実行

遠隔環境に更新した`outputs`一式を反映し、プロジェクトのルートで実行する。
新規ファイルに加え、既存`rcwa_ext`と`studies/pmma_gold_motheye/common.py`の補助配列解放処理を利用する。
PyTorch、torcwaが必要。PNG/SVGグラフにはMatplotlib、設定と形状SVGだけの確認は標準ライブラリのみ。

```text
cd /home/7/uq06557/common2/20260813/tsoyutarcwa
python run_n25_motheye_comparison.py --device cuda --dry-run
python run_n25_motheye_comparison.py --device cuda --run-final-spectrum
```

2026-10-12の実装修正では光学ソルバーのハッシュも変わるため、
`repaired_v2`の光学キャッシュは再利用できない。
必要な転送ファイルと診断根拠は[IMPLEMENTATION_FIX_20261012.md](IMPLEMENTATION_FIX_20261012.md)を参照。
まず、以前に不安定だった正方格子M=16の550/700 nm、Q=384/512と
小規模の両格子半径勾配を確認する：

```text
python run_n25_motheye_repair_validation.py --device cuda
```

`status=passed`を確認したら、4ケースの収束、61点スペクトル、時間・勾配監査へ進む。
前段の検証は全帯域の収束証明ではない。

```text
python run_n25_motheye_comparison.py --device cuda --orders 8,12,16,20 --slices 40,60,80,100 --grids 256,384,512 --run-final-spectrum --output-root studies/dielectric_motheye_n25/results/repaired_v3
```

初期3点だけで全帯域の収束を保証するわけではない。
旧写像の時間見積もりは[RUNTIME_ESTIMATE_20261012.md](RUNTIME_ESTIMATE_20261012.md)に保存している。
写像と界面計算法が変わったため、新版の実行時間は再測定が必要。
保存済みGPUケースから見積もりを更新する場合（光学計算なし）：

```text
python estimate_n25_motheye_runtime.py --output-root studies/dielectric_motheye_n25/results/repaired_v3
```

2格子×2方法の収束確認、最終スペクトル、計算時間、勾配の確認、図と比較JSONの出力まで行う。
CUDAを明示した場合はCPUへ自動で戻さない。行列・固有値・界面接続・フラックス・自動微分はGPU、
設定・幾何条件・材料準備・ファイル出力はCPU。`--device auto`ならGPUがなければCPU。

GPU本計算での負荷を分けたいときは、格子ごとに実行できる：

```text
python run_n25_motheye_comparison.py --device cuda --lattices hexagonal --run-final-spectrum
python run_n25_motheye_comparison.py --device cuda --lattices square --run-final-spectrum
```

同じ出力先で順番に実行すれば、整合した過去のもう一方の結果も比較JSONに取り込む。
同じ出力先への同時実行はしない。`--methods nv`等の片方だけの計算では、
相手の結果がなければ推奨は`insufficient_evidence`になる。

候補不足の場合の拡張例：

```text
python run_n25_motheye_comparison.py --device cuda --orders 6,8,10,12,14,16 --slices 40,60,80,100 --grids 256,384,512 --max-cycles 4 --run-final-spectrum
```

同じ形状・材料・ソルバーなら、完了した光学ケースを再利用する。
候補・判定波長・許容値を変更しても個別ケースの再利用が可能。
形状・材料・G・対称性・ソルバーソース・インストール済みtorcwaソースが変わった場合は別出力先が必要。
CPUからGPUへ移った場合は光学キャッシュを使えるが、時間・勾配の監査はデバイス情報で区別して再実行する。

形状例：

```text
python run_n25_motheye_comparison.py --device cuda --height-nm 600 --base-radius-nm 90 --run-final-spectrum --output-root studies/dielectric_motheye_n25/results/H600_r90
```

Gの影響を独立に調べる場合も別の`--output-root`で`--asr-circle-g`を変える。
格子・形状・入射条件を変えた後は、初期形状で得た次数を無条件に採用しない。

図だけ再生成：

```text
python run_n25_motheye_comparison.py --plot-only
```

## 収束と最適化への適性

各軸の高精度側の最後の2区間で、全判定波長におけるR/T/Aの変化が許容値以下、
かつ無損失のフラックス条件を満たすことを要求する。途中の平坦部だけでは採用しない。
そのうえで、高精度側まで安定した区間の先頭のfine候補を選び、
採用候補とそれより上の全候補の差も許容値以下であることを確認する。
例えば8→10、10→12だけが通ればM=10。4→6から12まで安定し、
M=6と全上位候補も一致すればM=6を採用する。隣接差の累積で誤差が増えるケースも除外する。
両方法が別の次数で収束する場合は、各方法の採用条件で計算時間を比べる。
方法間比較の既定0.001は絶対誤差0.1パーセントポイントで、Rに対する相対誤差0.1%ではない。
自己収束と全スペクトル細分化では既定0.00025を要求し、方法間比較に余裕を残す。
`--mesh-tolerance`で指定可能。ただし比較許容値より緩くできない。
低反射率の小さな改善を比較する場合は、例えば`--tolerance 0.0001`にして再検証する。
Nz→M→Qを更新し、三軸の設定が安定し、同じ固定設定で全軸が通った場合に`converged`。
これは有限の収束チェックであり、厳密な誤差上限ではない。

自己収束が通っても、NVとASRの結果が許容値内で一致しなければ`methods_disagree`。
その場合は高速な方を選ばず、両方法の上限・格子・Gを再確認する。
方法間の比較は最終スペクトルを要求したときは全最終波長、それ以外は判定波長で行う。
保存済みの片方の判定条件が緩い場合や、波長セット・デバイスが異なる時間測定は採用しない。

収束した各方法で、次を確認する：

1. ウォームアップを除いた目的関数評価時間の中央値（既定3回）。CUDAは測定の前後で同期。
2. CUDA使用時のPyTorchピーク割当メモリ。ドライバ／他ライブラリの全GPU使用量ではない。
3. 高さと底部半径に関する平均Rの自動微分。
4. 中央差分h、h/2との一致（高さh=0.1 nm、半径h=0.02 nm）。
5. M、Nz、Qをそれぞれ上位候補へ増やしたときの勾配の安定性。
6. 両方法の勾配も一致するか。

勾配の判定はabs(diff)≦1e−7 /nm＋0.05×max(abs(g1),abs(g2))。
`--gradient-atol`、`--gradient-rtol`で変更できる。
監査の目的関数は**監査用11波長を使った台形積分の波長平均R**であり、61点の平均の勾配とは限らない。
初期の収束走査を3点に減らしても監査波長と目的関数は変えない。
`--audit-wavelengths`で明示的に変更できるが、波長を減らすと監査する平均Rの目的関数も変わる。
波長ごとに勾配を加算するため、全波長の計算グラフを同時には保持しない。
対称性縮約を使うASRの半径勾配では、各層の写像・積分・固有値問題を
`torch.utils.checkpoint`で逆伝播時に再計算する。縮約した層モードは保持するが、
全層の大きな補助計算グラフを同時には保持しない。計算時間は増える可能性がある。
`--no-layer-checkpoint`で従来の保持方式へ戻せる。`--no-symmetry`ではこの対策は使わない。
CPUで順計算と両形状勾配の一致を検証した。GPUピークメモリは再計算の監査で測定する。

両格子の既定`--asr-profile local-radial`は、内接円の内部だけを放射状に変形し、
角形領域を円へ変換する際の接線方向の大きな変形を避ける。
公称形状の円周を計算座標上の固定界面とし、物理半径の変化を写像とJacobianへ伝える。
有限差分の±hでも同じ公称界面を保持する。境界の二値マスクが変化する勾配は使用しない。
最適化でこの離散勾配を使う場合も、比較する局所変形の間は同じ参照形状を保持する。
参照形状を作り直したときは順計算と勾配の収束を再確認する。
`radial`、`equalized`、`weiss2009`、`identity`も比較用に選べるが、新規の別出力先を使う。
従来の`--asr-square-profile`は別名として残す。

有限差分と物理検査は通ったが勾配のM/Nz/Q細分化が不合格の場合、
不合格の軸の採用下限を次の候補へ上げ、順計算・全スペクトル・監査を再実行する。
既定で最大2回（`--max-numerical-refinements`）。FD不一致やエネルギー保存違反は
次数増加で隠さず、未検証のままとする。候補上限を採用するだけでは収束と認定しない。
`--min-order 16`等で採用下限を明示できるが、その上に検証用の次数を残す必要がある。

`recommendations`は格子ごとに二種類を出す：

- `derivative_free_method`：順計算時間から、勾配を使わない探索に向く方法を比較。
- `gradient_method`：勾配の一致・精度・安定性が通った場合、順計算＋逆伝播時間から比較。

時間差が15%未満なら`comparable`。NVの保持次数が少ないことだけでは勝者を決めない。
片方だけ勾配監査に通った場合は`one_method_gradient_candidate`という暫定候補に留める。
`--skip-gradient-audit`なら順計算とその時間だけを検査し、勾配法の判断は行わない。
高さ・底部半径の監査は、各層半径を個別に変える任意形状の全勾配を検証したことにはならない。
最適化後の形状でも三軸・方法間の一致を再確認する。

## 出力

既定では`studies/dielectric_motheye_n25/results/`に保存する。

- `plan.json`、`geometry.svg`、`profile_layers.csv`：設定、連続形状、初期Nz候補の層半径。
- `<格子>/<方法>/checkpoint.json`：光学ケースの再開用。
- 同フォルダの`profile_layers.csv`：その方法の推奨／最後の保持設定Nzに対応する中点半径。
- 同フォルダの`convergence.json`、`all_cases.csv`、`anchor_spectrum.csv`、`spectrum.csv`：収束結果。
- `convergence.json`内の`final_spectrum_mesh_validation`：全最終波長の三軸別／同時細分化結果。
- `anchor_refinements`：最終チェックで追加された判定波長と不合格の理由。
- `audits/M*_Nz*_Q*/audit_checkpoint.json`、`audit.json`：採用条件別の時間・メモリ・勾配の途中／最終結果。
- 同フォルダ直下の`audit.json`：最新の採用条件の監査結果。
- `audit_history`、`numerical_refinements`：勾配細分化に伴う条件引き上げの根拠。
- `comparison.json`：4ケースと、格子ごとの方法選択。まずこの`recommendations`を確認する。
- `figures/absolute_<格子>_<軸>.png/svg`：M/Nz/Qに対するR・Tそのもの。
- `figures/changes_and_time_*.png/svg`：変化量（パーセントポイント）と個別波長の計算時間。
- `figures/spectrum_*.png/svg`：Rの比較とR+T−1。
- `figures/optimization_cost_*.png/svg`：収束した条件でのウォーム評価時間。
- `figures/gradients_*.png/svg`：自動微分・有限差分・細分化後の勾配。

`comparison.json`の生データと条件を優先する。絶対値図は見やすくするため縦軸を自動調整し、
R/T/Aそのものをクリップしない。曲線の固定Nz/Qは方法ごとに異なることがあり、図のタイトルで示す。
単発時間の図には初回準備の影響が含まれるため、方法選択にはウォーム中央値を使う。

ステータスは`converged`、`candidate_range_insufficient`、`cycle_limit_reached`、
`nonphysical_results`、`nonphysical_final_spectrum`、`final_spectrum_not_converged`、`calculation_failed`、`interrupted`。
実行中は`running`、`spectrum_running`、`spectrum_validation_running`、`audit_running`で処理段階を示す。
未収束なら最終スペクトル／最適化監査を省略し、完了ケースは保存する。
本番の終了コードは比較成立0、検査不足・不一致2、計算失敗1。
既定では勾配監査も要求するため、順計算だけが通った場合や片方だけ勾配が通った場合は2。
`--skip-gradient-audit`を指定したときは、順計算の比較成立だけで0となる。

## ローカル検証

```text
python -m unittest studies.dielectric_motheye_n25.test_comparison
```

CPUの小規模計算で4ケースのエネルギー保存と形状勾配を検査する。
ローカルの`results/local_cpu_550nm`等は550 nmでの小規模評価であり、
GPU本計算や400–700 nm全体の手法選択の根拠として扱わない。

**修正前の旧写像**による550 nm、Nz=12、Q=96、CPU・2スレッド、許容値0.001の予備評価では、
六方はM=6を採用し、RはNV=0.00636095、ASR=0.00589262。
正方はM=4を採用し、RはNV=0.00771255、ASR=0.00857250。
両方法の差は許容値以下で、NVのウォーム順計算は六方約2.0倍、正方約3.5倍速かった。
時間測定は1回のため、この比は予備値。
四つのケースとも高さ勾配の細分化安定性は未達で、六方ASRの底部半径勾配は有限差分のh/h2とも不一致。
この条件で勾配最適化に適した方法は確定していない。
監査が不合格の場合、`audit.json`の`fd_checks`と`refinement_checks`を区別して確認する。
