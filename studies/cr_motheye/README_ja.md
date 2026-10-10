# Crモスアイ／Cr基板の収束確認

`gold_motheye` と同じ構造・matched-ASR・D6縮約を使い、突起と基板の両方をCrへ変更する。
1DのLi/ASR比較とは別の、面内2方向に周期的な円形断面構造である。

## 材料と見込み

添付された `Johnson (2).csv` を `data/Johnson.csv` に原文のまま保存した。
`wl,n` と `wl,k` の2ブロック、49波長、188–1937 nm。
`wl` はµmなので1000倍してnmへ変換し、nとkの波長対応・重複・有限性・受動性を検証する。
各データ点で ε=(n+ik)² とし、εを波長に対して線形補間する。範囲外へ外挿しない。
既存の金CSV実装と同じε補間である。

|波長 nm|Cr ε（添付）|Au ε（既存Rakić LD）|
|---:|---:|---:|
|400|−4.05570 + 11.48290i|−1.06116 + 4.92069i|
|550|−0.96272 + 21.18102i|−5.37137 + 2.35816i|
|700|−2.13786 + 20.67564i|−13.75426 + 1.91049i|

特に550–700 nmでCrの損失は大きく、金の鋭い共鳴が弱まる可能性がある。
これは材料データに基づく見込みであり、RCWA収束の証明ではない。
Crの誘電率の絶対値も大きいため、界面の不連続性・Fourier打切り・階段近似・写像積分の
難しさは残る。Crへ変更すれば必ず少ない次数で収束するとは判断しない。

データの出典: [Johnson and Christy, Phys. Rev. B 9, 5056–5070 (1974)](https://doi.org/10.1103/PhysRevB.9.5056)、
[Cr / Johnson データ掲載ページ](https://refractiveindex.info/?shelf=main&book=Cr&page=Johnson)。
実際のCr膜の酸化・成膜条件はこの計算に含めていない。

## 既定の物理・数値条件

|項目|条件|
|---|---|
|配列|60°三角Bravais格子（六方配列）、最近接周期200 nm|
|突起|Cr円錐台、高さ500 nm、先端半径5 nm、底面半径95 nm、線形profile|
|入射|空気、垂直入射、x偏光|
|基板|半無限Cr|
|手法|円境界に合わせたmatched-ASR、Fourier因子分解、Redheffer接続|
|縮約|D6-E1 source-row（円形断面・六方配列・垂直入射）|
|精度|complex128|
|代表波長|400・550・700 nm|
|次数M|4,6,8,10,12,14,16,18,20|
|高さ分割Nz|50,60,70,80,90,100|
|積分格子|96,128,192,256 の各正方格子|
|許容差|最大絶対変化0.005 = 0.5パーセントポイント|

Mは面内Fourier打切りパラメータで、D6の縮約後の固有値問題次元はM(M+1)+1。
空気中で伝搬する回折次数の数とは異なり、近接場に必要なエバネッセント成分も増やす。
3軸を交互に更新し、上限側で連続2回のrefinementが許容差内かつ物理的なパワーであり、
次のcycleでも採用構成が変化しなければ `converged` とする。
候補上限で変化が大きい場合は `candidate_range_insufficient`。
3軸の判定が通っても採用構成の固定点をcycle上限までに確認できなければ `cycle_limit_reached`。
どちらも収束済みではなく、終了コード2を返す。

この試験は代表3波長・G=0.03に対する離散化安定性の確認である。
全可視域の誤差上限、Gに対する安定性、独立した体積吸収との一致までは保証しない。
最終的に0.1パーセントポイントが必要なら `--tolerance 0.001` で再評価する。

## 実行

更新した20260806のプロジェクトをGPU計算機へ反映し、プロジェクトルート
`/home/7/uq06557/common2/20260813/tsoyutarcwa` から実行する。

```bash
python run_cr_motheye_convergence.py --device cuda --dry-run
python run_cr_motheye_convergence.py --device cuda
```

最初のコマンドは条件と材料値を表示するだけで、固有値計算を行わない。
本計算はCUDAを明示選択し、利用できない場合は開始前に停止する。
`--device auto` ならCUDAが利用可能なときGPU、それ以外はCPUを選ぶ。
幾何・材料・積分格子の準備の一部はCPUで行い、solverの線形代数を指定デバイスで行う。

各caseをcheckpointへ保存するため、中断後は同じコマンドで再開できる。
材料とsolverソースのハッシュを記録し、異なる設定・実装のcheckpointとの混用を拒否する。
次数・層数・格子の候補リストは署名から独立しており、同じ物理設定のまま候補を追加できる。
許容差の変更も既存caseを使って再判定する。Gや形状を変える場合は別の `--output-prefix` を使う。

収束後のスペクトルも計算する場合:

```bash
python run_cr_motheye_convergence.py --device cuda --run-final-spectrum
```

400–700 nm、5 nm刻み。未収束なら最終スペクトルを計算しない。

## 出力と図

既定保存先は `studies/cr_motheye/results/`。

- `cr_motheye_plan.json`: 条件、材料値、solverハッシュ
- `cr_motheye_checkpoint.json`: 計算済みcase
- `cr_motheye_convergence.json`: 判定・3軸の履歴
- `cr_motheye_all_cases.csv`: 実行時間を含む全case
- `cr_motheye_anchor_spectrum.csv`: 採用候補の3波長結果。未収束なら暫定値
- `cr_motheye_figures/convergence.svg`: 隣接候補間の最大変化、縦軸はパーセントポイント
- `cr_motheye_figures/spectrum.svg`: R・T・吸収分配
- `cr_motheye_figures/absolute_order.png/svg`: 回折次数に対する実際のパワー比
- `cr_motheye_figures/absolute_slices.png/svg`: 層数に対する実際のパワー比
- `cr_motheye_figures/absolute_grid.png/svg`: 積分格子に対する実際のパワー比

絶対値の図は各軸の最新cycleの固定条件を表示し、縦軸1が100%。
Matplotlibがない場合、最初の2つのSVGのみ生成し、その旨を表示する。
図だけ再生成するコマンド:

```bash
python -m studies.cr_motheye.plot_results
```

半無限Crには裏面の遠方透過portがないためT_far=0、A_total=1−R。
P_subは構造／基板界面を通るパワーで、最終的に基板で吸収される。
A_moth=1−R−P_subでモスアイ領域の吸収を区別する。
この等式自体は独立した体積吸収の検証ではない。

## 検証と金の比較データ

```bash
python -m studies.cr_motheye.validate --integration --device cuda
```

CSVの単位・受動符号・補間・外挿拒否に加え、平坦な空気／Cr界面のFresnel解と
パターン付き最小構造のD6/Cs一致を確認する。
ローカルではCPUでこれらと小規模な3軸実行を確認する。本来のM=4–20・Nz=50–100の
GPU本計算の結果は、実行後に判定する。

金の旧 `gold_motheye_convergence.json` には、前方透過ブロックを計算しなかった時期の
無効な吸収分配が含まれる。材料比較にはそれを使わず、修正済みの `gold_motheye_corrected*`
を用いる。現在保存されている修正済み金checkpointは部分計算で、全3軸の収束値は
まだ確定していない。Cr版は前方透過・反射の両ブロックを計算する修正済み経路を共有する。
