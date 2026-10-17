# NV法：長方形・楕円の収束確認と、等軸楕円／円の一致確認

周期200 nm、高さ500 nm、構造n=2.5、基板n=1.5、空気から垂直入射を既定値とする。
**NV法の確認には `nv_convergence.py` を使用する。長方形・楕円・比較用の円をすべてNV法で計算する。**
実行済みの結果と未収束の項目は[NV_RESULTS_ja.md](NV_RESULTS_ja.md)に記録した。
更新されたCUDA結果の波長別評価と、M=11→13だけを追加する手順は[NV_NEXT_STEPS_ja.md](NV_NEXT_STEPS_ja.md)を参照。

この環境のコード一式のルートは次の場所：

```text
C:\Users\tsoyu\Documents\Codex\2026-08-06\files-mentioned-by-the-user-extensions\outputs
```

以下のコマンドは、20260806のルート（`rcwa_ext` と `Optimization` がある場所）で実行する。
ここで直接実行する場合は最初に移動する：

```powershell
Set-Location 'C:\Users\tsoyu\Documents\Codex\2026-08-06\files-mentioned-by-the-user-extensions\outputs'
```

## 1. 等軸楕円と既存の円NV法の一致確認

```powershell
python Optimization/achromatic_quarter_wave/nv_convergence.py circle-check --index 2.5 --orders 3 5 --grids 128 256 --layers 15 --wavelengths 400 550 700 --device cpu --output Optimization/achromatic_quarter_wave/results/nv_circle_check_M3_5
```

既定の比較対象は半径30 nm・70 nmの一定断面柱、および半径10→50→90 nmの15層テーパー。
一定断面柱は高さ500 nmの1層として計算し、テーパーは指定した15層で計算する。
楕円の各層で半径a=b=rと設定し、既存の `Circle(r)` のNV法と比較する。
**楕円側はa=bでも円のルーチンに切り替えない。**
楕円の解析Fourier係数・法線場・層の組立を通して計算するため、円への置換による自明な一致ではない。

比較する項目：

- x/yの0次複素透過振幅（実部・虚部）、および各偏光の透過位相。
- x/yの透過率・反射率と、前方透過・反射のS行列全体。
- 誘電率のFourier畳み込み行列と、MaxwellのP/Q行列。
- 等軸断面の位相差がほぼ0°となること、およびR+T=1。

保存先には `circle_equivalence.csv`、`circle_equivalence_summary.json`、各条件の`cache/`が作られる。
既定の一致許容値は振幅・S行列・パワー差10^-8、透過位相差10^-6°、P/Q行列差10^-9。
不一致があれば結果を保存したうえで終了コード2を返す。

## 2. 長方形・楕円のNV収束確認

```powershell
python Optimization/achromatic_quarter_wave/nv_convergence.py converge --index 2.5 --shapes rectangle ellipse --orders 3 5 7 --grids 128 256 --layers 15 30 --wavelengths 400 550 700 --device cpu --output Optimization/achromatic_quarter_wave/results/nv_convergence_M3_5_7
```

既定の形状は前回の比較候補と同じ寸法：

| z | x半径／半幅 | y半径／半幅 |
|---|---:|---:|
| 0 nm | 90.5 nm | 48.2 nm |
| 250 nm | 92.3 nm | 50.0 nm |
| 500 nm | 94.1 nm | 51.8 nm |

楕円では半径a,b、長方形では半幅a,bであり、全軸長・全幅は2a,2b。
層の中央で寸法を補間するため、分割数を変えても同じ連続テーパーを近似する。
同じ外接寸法を比較しており、面積は楕円πab、長方形4abと異なる。

以下を別々に変える：

1. M=3→5→7：Q=128、15層を固定。
2. Q=128→256：M=7、15層を固定。
3. 15→30層：M=7、Q=256を固定。
4. 最終Q=256・30層でもM=5→7を再確認。

材料のεと1/εには解析Fourier係数を使う。**Qは形状の二値画像の解像度ではなく、法線射影場のサンプリング数。**
法線場の中心・セル境界の正則化は既存の円NV法に合わせている。
この正則化のセル境界幅もQに依存するため、Qの依存性を省略してはいけない。

保存される主なファイル：

| ファイル | 内容 |
|---|---|
| `nv_convergence.csv` | 各条件の位相差、x/y透過率・反射率、複素透過係数、エネルギー誤差 |
| `convergence_summary.json` | M・Q・z分割ごとの最終2段階の差と判定 |
| `convergence_summary.csv` | 判定を表形式で確認するファイル |
| `run_metadata.json` | 条件、形状、計算方法、参照したコードのSHA256 |
| `cache/` | 成功した各条件の計算結果。同じコマンドで再利用する |

判定の既定値は、位相差と各偏光の透過位相の変化がともに0.1°以下、
透過率・反射率の絶対変化が0.001以下、エネルギー誤差が10^-7以下。
等軸断面では位相差が0°でも、共通の透過位相や透過率が未収束のことがあるので、両方を調べる。
一つしか値を指定していない軸は `insufficient_levels` と記録し、収束合格にはしない。

**計算が正常に終了したことと、指定許容値まで収束したことは別。**
`needs_refinement` があれば、そのM・Q・分割数を増やす。
`--require-convergence` を付けると、未収束・段階不足の場合に終了コード2を返す。
未指定の場合は未収束でも結果を保存して通常終了する。

より高次数・細かいグリッドへ進める例（新しい保存先を指定）：

```powershell
python Optimization/achromatic_quarter_wave/nv_convergence.py converge --index 2.5 --shapes rectangle ellipse --orders 5 7 9 --grids 256 512 --layers 15 30 --wavelengths 400 550 700 --device cpu --output Optimization/achromatic_quarter_wave/results/nv_convergence_M5_7_9
```

400–700 nmを細かく確認する段階では `--wavelengths 400 425 450 475 500 525 550 575 600 625 650 675 700` などに増やす。
まず3波長で収束条件を調べ、その後、候補となる波長板の形状について帯域全体を検証する。
高次数・多数の分割層ではCPUの計算時間・使用メモリが増える。CUDA版PyTorch環境があれば `--device cuda` を指定できる。
この環境での実行確認はCPU・complex128・1スレッドで行った。

より強いテーパーを調べる例：上のconvergeコマンドに
`--x-knots 10 55 95 --y-knots 10 30 50` と別の `--output` を指定する。
基板をn=2.5にする場合は `--substrate-index 2.5` を指定するが、周期200 nmでは400 nm側で高次回折光が発生する。
全透過率と0次透過率は両方保存する。ちょうど外部回折次数のカットオフになる波長は避ける。

## 3. 図の作成と基本チェック

```powershell
python Optimization/achromatic_quarter_wave/plot_nv_convergence.py --input Optimization/achromatic_quarter_wave/results/nv_convergence_M3_5_7
```

同じ保存先に `nv_convergence.png` と `nv_convergence.svg` を作る。
グラフは位相差の確認用で、収束判定にはCSV/JSONの透過率・各偏光の透過位相も使用する。

```powershell
python Optimization/achromatic_quarter_wave/validate_nv.py --output Optimization/achromatic_quarter_wave/results/nv_validation.json
```

境界法線の向き、Fresnel係数、等軸断面、軸交換、一定断面の層分割に対する不変性を確認する。

## 実装上の範囲

`nv_shapes.py` の局所的なNV拡張を使う。既存の円NVルーチンは比較用の参照として使用する。
NVの対称なFourier因子分解とRedheffer接続を共有し、Lanczosフィルター・偏光や群対称性による縮約を使わない。
長方形の法線場は各辺で辺に垂直となる解析的な拡張を使い、法線が定義できない頂点は等方値で正則化する。
対応範囲は中心を揃えた軸平行の断面、直交格子、μ=1、散乱係数のみ。内部電磁界の再構成は対象外。
既存の `run.py` による初期探索の方式は、長方形ASR・楕円standardのまま。
NV収束確認の結果と、過去の方式による結果は計算方法の欄を区別して読む。

同じ条件・同じコードで再実行するとキャッシュを再利用する。
条件や参照コードを変えた場合は誤った再利用を避けるため、別の `--output` が必要。
同じ保存先で複数プロセスを同時実行しない。
