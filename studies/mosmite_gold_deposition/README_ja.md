# MOSMITEの金被覆：真空蒸着とスパッタの2モデル

20260806の共有`rcwa_ext`を利用する。既存ソルバーや未被覆モデルは変更しない。
`studies/mosmite_gold_deposition/`をプロジェクト直下へ追加する。
単独の2実行ファイルは、このフォルダ内の共通モジュールとdataを使用する。

## 実行ファイル

- `run_evaporation.py`：垂直方向の理想真空蒸着。頂部・谷部30 nm、側面は半径方向3.6 nm。
- `run_sputter.py`：スパッタの現象論的な近似。頂部・谷部30 nm、側面は半径方向8 nm。

どちらも空気→金被覆樹脂モスアイ→PET相当基材→粘着層→石英ガラス→空気。
実試料へ合わせた形状の同定や自動最適化は行わない。
金の成長・粒子輸送・結晶粒・島状成長を計算するプログラムではない。

## 実行（PowerShellでもbashでも使える一行コマンド）

`rcwa_ext/`があるプロジェクト直下で、既存のPyTorch/Torcwa環境から実行する。
CUDAが使えない環境では、以下の`--device cuda`を`--device cpu`へ変更する。

最初に形状を確認する。プレビューはtorchを読み込まず、光学計算をしない。

```text
python studies/mosmite_gold_deposition/run_evaporation.py --preview-only
python studies/mosmite_gold_deposition/run_sputter.py --preview-only
```

プレビューで生成する`geometry.png`と`geometry_layers.csv`は、実際に計算へ渡す分割形状。
`settings.json`に厚さと材料参照を保存する。

400/600/800 nmで、次数・高さ分割・格子の収束を確認する。

```text
python studies/mosmite_gold_deposition/run_evaporation.py --device cuda --wavelengths "400,600,800" --slices 150 --order 7 --grid 512 --check-convergence
python studies/mosmite_gold_deposition/run_sputter.py --device cuda --wavelengths "400,600,800" --slices 150 --order 7 --grid 512 --check-convergence
```

各モデルの`results/evaporation/convergence/`と`results/sputter/convergence/`へ保存する。
既定では(M,Nz,grid)=(7,150,512),(9,150,512),(9,300,512),(9,300,1024)を比較する。
Nzはモスアイ部の分割数。頂部Auの分割数も高さ分割の増加に合わせて増やす。
R/T/Aおよびx/y偏光のR/Tの段階間差、初期値と最終値の差、最終値のx/y差を検査する。
許容差は1e-4（0.01パーセントポイント）。不合格ならCSVを残してエラー終了する。
`settings.json`の`status`と`convergence_passed`を確認する。
M=7は計算開始用の候補であり、金被覆モデルでの収束を保証する値ではない。
金属の連結領域は特に高次数を要する可能性がある。条件を上げた場合は本計算にも反映する。

合格した条件で全81点の本計算を行う。

```text
python studies/mosmite_gold_deposition/run_evaporation.py --device cuda --wavelengths "400:800:5" --slices 150 --order 7 --grid 512
python studies/mosmite_gold_deposition/run_sputter.py --device cuda --wavelengths "400:800:5" --slices 150 --order 7 --grid 512
```

既定の保存先はそれぞれ`results/evaporation/reference/`と`results/sputter/reference/`。
`--output-dir PATH`で変更可能。過去の結果を比較するときは異なる保存先を使う。
3波長の収束確認は全81点の収束確認を意味しない。狭いスペクトル構造を見つけた場合は
`--check-wavelengths "400,430,600,800"`などでその波長も指定する。

## 高次数でGPUメモリが不足する場合（2026-10-18追加）

`--layer-storage reduced`で、反射・透過に不要な層ごとのCartesian行列・材料格子を
追加直後に解放する。D6では独立した縮約固有モードを保持し、4種類の層S行列は
1層ずつ生成してRedheffer接続する。全矩形基底では、追加時に層Sを接続して消費する。
真空蒸着・スパッタの両方に対応し、共有`rcwa_ext`の変更はない。
既定値は`--layer-storage full`で従来の保持方法。材料・形状・次数・分割数・NVM・
complex128・厚層の位相平均は変更しない。有限試料に必要なTf/Rf/Rb/Tbをすべて保持する。
内部電磁場の復元や勾配計算には、この省メモリ設定を使わない。

最初に、すでに完了したM=18、Nz=150、grid=512を再計算し、元のCSVと照合する。
過去の設定ファイル・生のCSVを使うので、画面に丸めて表示された数値との比較ではない。

```bash
python studies/mosmite_gold_deposition/run_evaporation.py --device cuda --wavelengths "400,600,800" --slices 150 --order 18 --grid 512 --layer-storage reduced --reference-dir studies/mosmite_gold_deposition/results/evaporation/M18_Nz150_G512 --output-dir studies/mosmite_gold_deposition/results/evaporation/M18_Nz150_G512_reduced
```

材料データ・共有ソルバー・形状・厚層計算のソース、および数値条件の一致を先に確認する。
R/T/A・偏光ごとのR/T・交差偏光の絶対差が1e-8以下なら、`PASS storage parity`を表示する。
`storage_parity.csv`と`settings.json`にも保存し、不一致ならエラー終了する。
`--reference-dir`と`--output-dir`には別のフォルダを指定する。

照合に合格した後、層数と格子を固定したまま次の次数へ進める（bash/sh）。

```bash
for M in 20 22 24 26; do
  python studies/mosmite_gold_deposition/run_evaporation.py \
    --device cuda --wavelengths "400,600,800" \
    --slices 150 --order "$M" --grid 512 --layer-storage reduced \
    --output-dir "studies/mosmite_gold_deposition/results/evaporation/M${M}_Nz150_G512_reduced" \
    || break
done
```

各波長のCUDAピーク使用量（allocated/reserved）を`spectra.csv`に記録し、allocatedを画面表示する。
`retained_before_solve_bytes`は接続直前に実際に参照されているテンソルの容量であり、
一時作業領域やCUDAアロケータのキャッシュを含むピーク使用量とは異なる。
波長間で不要なPython参照と未使用CUDAキャッシュも回収するが、実際に参照中の行列を
解放する処理が主な対策である。環境変数による断片化対策だけには依存しない。

CPUで小さい次数および150層の保持方法の同等性を検証するコマンド：

```text
python studies/mosmite_gold_deposition/validate_memory.py --device cpu
```

省メモリ化は収束の改善とは別であり、実行可能な最大次数を保証しない。
高次数のCUDAピークやM=18とのGPU照合は実際の計算機で確認する。
M=14/16/18の既存結果は特に600/800 nmで次数依存が残り、まだ収束済みではない。
層数150とgrid512の十分性も、次数が安定した後に別々に確認する。

## 具体的な形状

| 条件 | 真空蒸着 | スパッタ |
|---|---|---|
| 周期・配列 | 100 nm、三角格子 | 同左 |
| 樹脂突起 | 高さ200 nm、底面直径98 nm、先端直径50 nmの円錐台 | 同左 |
| 頂部Auのz方向厚さ | 30 nm | 30 nm |
| 谷部Auのz方向厚さ | 30 nm | 30 nm |
| 側面Auの半径方向厚さ | 30*(49-25)/200=3.6 nm | 8 nm（参考値） |
| 上端の形状 | z=200 nmで外半径28.6 nmから、z=230 nmで半径25 nmまで線形に細くなる | 半径33 nmの平坦な金キャップ近似 |
| モスアイ部の分割 | 150層（谷部を含む） | 150層（谷部を含む） |
| 頂部Auの分割 | 追加23層 | 追加23層 |
| 金被覆構造部の総層数 | 173層 | 173層 |
| 金を含む最大高さ | 230 nm | 230 nm |

真空蒸着の既定形状は、線形円錐台の表面高さg(x,y)に対して、
金の領域をg(x,y)<z<g(x,y)+30 nmとする理想的な垂直成長形状に一致する。
側面に垂直な方向での金厚は約3.57 nm。金の側面厚を30 nmとはしていない。
`--side-gold-nm 6`などで半径方向厚さを変更すると、理想垂直成長から外れた感度比較になる。

スパッタの8 nmは、Kobayashiら(2019)のMOSMITE試料に対するモデルを参考にした値。
実試料のスパッタ条件や断面測定から決めた値ではない。
頂部半径33 nmの平坦キャップは本実装の形状近似である。
https://doi.org/10.1364/OME.9.003744

谷部は0<z<30 nmにおいて樹脂突起の外側をAuで満たす。樹脂の下にAu層を挟まない。
谷部30 nmはモスアイ高さ200 nmの内数なので、全高さを260 nmにはしない。
z=30 nm、z=200 nm、z=230 nmの境界を明示し、側面の金同士が接触する高さも分割境界にする。

## 金がつながる領域の数値表現

スパッタの側面8 nmでは、谷部より上の約30--58.33 nmにも隣接するAu殻の合体がある。
その領域を孤立円殻で計算したり、セル端で金を切り捨てたりしない。
周期格子点への最短距離を使ってAuの周期的な和集合を作り、実空間格子で表現する。
合体領域は標準RCWA、非合体円形断面は解析円板係数を使うNVMとする。
共有ソルバーの現状ではこの混在とD6 source縮約は併用できないため、
スパッタ既定条件では全矩形Fourier基底を使う。

真空蒸着の既定形状には谷部以外の殻の合体がないので、D6-star source縮約を使う。
`--basis full`を両方へ指定すると、両モデルを同じ全矩形基底で比較できる。
M=7のD6-starは169調和波、全矩形基底は225調和波。単にMが同じだけでは打切り集合は同じでない。
全基底の計算では零次の2x2 Jones反射・透過行列を保持し、交差偏光を含めてx/y強度を平均する。
D6縮約の場合は、正入射・等方材料のx=yの対称性を用いる。

## 材料と厚い層

- PET：同梱Zhangデータのn,kを各波長で線形補間。加工樹脂も同じPETデータを代用。
- 加工済み基材：金を含まず55 um。突起200 nmを差し引き、均一な光学層を54.8 umとする。
- `--residual-um 0`：未知の残膜をPET相当層へ統合する扱い。実物の残膜がないという主張ではない。
- 粘着層：20 um、n=1.50,k=0。分散は未測定の仮定。
- 溶融石英ガラス：1 mm、Malitson分散、k=0。
- Au：両モデル・全領域とも、まずJohnson--Christyの同梱参照n,kを使用。
- 厚いPET・粘着層・ガラスは位相平均。界面の多重反射、金とPETの吸収、ガラス裏面反射を含む。
- 非零回折次数は厚層内でエバネッセントになる条件を検査し、厚層を零次Jones振幅で接続する。
- 位相平均は16点/層と32点/層を各波長で比較する。`--coherence coherent`で固定位相へ変更可能。

金は連続した等方媒質として扱う。3--4 nm程度の膜が島状になる影響は、この基準モデルに含まない。
実物への一致を判断するときは、成膜条件・連続性・粗さに対応する光学データで検討する。

別の金データを全体へ適用する例（25 nm膜の比較用であり、30 nm実測値ではない）：

```text
python studies/mosmite_gold_deposition/run_evaporation.py --device cuda --gold-csv studies/mosmite_gold_deposition/data/Au_Yakubovsky_25nm.csv --output-dir studies/mosmite_gold_deposition/results/evaporation/yakubovsky25_comparison
```

領域ごとのデータを持つ場合は、`--top-gold-csv`、`--side-gold-csv`、`--valley-gold-csv`で指定できる。
未指定の領域は`--gold-csv`へ従う。CSVは`wavelength_nm,n,k`または`wavelength_um,n,k`。
位置に応じたn,kの変更は自動では行わない。波長範囲外へ外挿しない。
データの出典・測定試料の説明は`data/README_ja.md`に記載する。

## 出力と検証

- `spectra.csv`：無偏光のR,T,A（0--1）、x/yそれぞれのR,T、交差偏光成分、位相平均差、基底、層数、計算時間。
- `spectra.png`：R,T,Aの3段グラフ（%）。
- `geometry.png`、`geometry_layers.csv`：実際の分割形状。
- `materials.csv`：各波長の全材料・全Au領域のn,k。
- `settings.json`：仮定、データ・ソルバー・実行ファイルのSHA256、状態、完了点数。
- `convergence.csv`：収束確認の途中結果も逐次保存。

独立検証を実行する：

```text
python studies/mosmite_gold_deposition/validate.py --integration --device cpu
```

形状の高さ・層数・谷部境界、垂直蒸着の幾何学的一致、Airy式、位相平均の解析式、
交差偏光の強度計算、受動な金のR/T/A、無吸収時のエネルギー保存、
Auを除いた場合と独立に組み立てた未被覆RCWAとの一致を確認する。
これらの物理整合性検証は、次数・高さ・面内格子の収束確認とは別である。
