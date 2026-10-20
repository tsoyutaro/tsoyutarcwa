# 金モスアイ：傾斜法線を含む2次元周期FFFの収束・物理検査

`paper_reproductions/mohamad2020` の1次元TM計算を出発点として、
2方向の周期と3成分の法線を扱う専用studyを作成しました。
円錐の側面傾斜による `n_z` と `epsilon_xz / epsilon_yz` を含む
4成分の接線電磁場を解きます。元の1次元論文に載っている2次元結果の再現ではなく、
3次元法線場を使うベクトルFFFへの拡張・比較実装です。

**現時点の結果:** 独立な物理・実装検査12件は成功しました。
一方、実際のAu円錐ではM=3・700 nmで反射率が1を超える低次数の不安定点を検出しました。
高次数収束と全条件の物理妥当性は未確認です。保存済み結果の詳細は
[ASSESSMENT_ja.md](results/ASSESSMENT_ja.md)を参照してください。

## 形状・材料

|項目|既定条件|
|---|---|
|配列|三角格子、周期200 nm|
|円錐台|高さ500 nm、上端半径5 nm、底面半径95 nm、線形テーパー|
|入射|空気から正入射、x偏光（`--polarizations x,y` で比較可）|
|基板|半無限Au|
|材料|`studies/shared/gold_dispersion.py` と `DEFAULT_GOLD_CSV` を共用|
|Au CSV|`studies/gold_motheye2/data/au_measured_nk.csv`|
|補間|既存ローダーの誘電率の線形補間、範囲外への外挿なし|
|波長|400,550,700 nm|
|精度|complex128|
|GIL|既定0 nm。任意に `--gil-nm 1` などで比較|

形状・材料条件は既存の金単体モスアイstudyに合わせています。PMMA/Auコアシェルは今回の対象に含めません。
GILは円錐の外側の**水平方向・半径方向**に追加します。法線方向の一定厚さではありません。
複素屈折率を線形変化させ、同心環で積分します。`--gil-rings` で環の分割数を変えられます。
GILは物理形状を変更するため、GILあり・なしは別の結果として評価してください。

## 手法

|`--methods`|内容|
|---|---|
|`rcwa-fff`|円錐の3成分法線、対称FFF、各区間の中点で行列を固定して局所行列指数を計算|
|`dm-fff`|同じベクトルMaxwell方程式をz依存のまま標準RK4で積分|
|`slice-nvm`|同じソルバーで `n_z=0` とした円柱スライスNV比較系|
|`direct`|通常のLaurent積だけを使うベクトル比較系|

既定は `rcwa-fff,slice-nvm` です。`dm-fff` は行列を多数のz点で作り直すため計算量が増えます。
`slice-nvm` は既存 `rcwa_ext` を呼ぶラッパーではなく、同じ新規基底・法線継続で比較する系です。
既存NVランナーとのビット単位の一致は主張しません。

三角格子ではD6で閉じた六角形のFourier集合を使い、スカラー調和数は
`B=1+3*M*(M+1)`、電磁場の行列次元は `4*B` です。
例: M=1,2,3,8,16でB=7,19,37,217,817です。
既存studyの正方形打ち切り `(2*M+1)^2` とMだけで精度・計算量を比較しないでください。
このstudyではD6の**モード縮約**は行いません。正入射でも4B次元全体を解きます。

## 最初の実行

既存プロジェクト直下（`studies/` と `rcwa_ext/` がある `outputs/`）で実行します。
Python 3.10以上、NumPy、SciPy、Matplotlib、PyTorchが必要です。既存RCWA環境を使用できます。

```powershell
Set-Location 'C:\Users\tsoyu\Documents\Codex\2026-08-06\files-mentioned-by-the-user-extensions\outputs'
python -m studies.gold_motheye_fff2d.validate --device cpu
python -m studies.gold_motheye_fff2d.run --device cpu --orders 1,2,3 --slices 24 --grid 64 --output-dir studies/gold_motheye_fff2d/results/my_cpu_check
```

低次数では非物理値を検出して終了コード1となることがあります。値を切り詰めず、
CSVとチェックポイントに `UNPHYSICAL` として残します。これは収束成功を意味しません。

## GPUで回折次数を増やす

CUDA対応PyTorchの既存環境で、最初に入力と行列規模を確認します。
`--dry-run` はソルバーを実行せず計画を表示します。

```bash
python -m studies.gold_motheye_fff2d.run --device cuda:0 \
  --orders 2:16:2 --slices 100 --grid 512 --dry-run
```

次数掃引と、追加の構成行列の受動性診断を行う例:

```bash
python -u -m studies.gold_motheye_fff2d.run --device cuda:0 \
  --orders 2:16:2 --slices 100 --grid 512 \
  --methods rcwa-fff,slice-nvm --wavelengths 400,550,700 \
  --loss-diagnostics \
  --output-dir studies/gold_motheye_fff2d/results/order_Nz100_Q512
```

PowerShellでは行継続の `\` を使わず、1行で実行してください。
メモリ計画の「20行列分」は目安で、実測ピークの保証ではありません。
高次数のCUDA計算はこのPCでは未実施です。GPU上でも最初に物理検査が実行されます。
OOM時は保存済み点を維持して停止します。

`--gil-nm 1` の比較は出力先を変えて実行します。GILがあれば必ず安定するとはしていません。

```bash
python -u -m studies.gold_motheye_fff2d.run --device cuda:0 \
  --orders 2:16:2 --slices 100 --grid 512 --gil-nm 1 \
  --methods rcwa-fff --loss-diagnostics \
  --output-dir studies/gold_motheye_fff2d/results/order_GIL1_Nz100_Q512
```

## 層数・法線格子の収束を別に調べる

```bash
python -m studies.gold_motheye_fff2d.run --device cuda:0 \
  --axis slices --values 30,60,100,140 --order 8 --grid 512 \
  --output-dir studies/gold_motheye_fff2d/results/slices_M8_Q512

python -m studies.gold_motheye_fff2d.run --device cuda:0 \
  --axis grid --values 128,256,384,512 --order 8 --slices 100 \
  --output-dir studies/gold_motheye_fff2d/results/grid_M8_Nz100
```

既定の次数収束判定は、全指定波長で最後の2回の隣接差について
`R`、基板流入、モスアイ吸収の各絶対差が0.0005以下であることです。
3点未満では収束と判定しません。固定Nz・Qでの次数収束と、Nz・Qの収束を分けています。
共通の低次数物理検査を省く `--skip-physics` を指定した場合、総合 `passed` はtrueになりません。

## DM-FFFと体積吸収を確認する

まず低次数・少数波長から始めます。

```bash
python -m studies.gold_motheye_fff2d.run --device cpu \
  --methods dm-fff --orders 1,2 --wavelengths 550 \
  --slices 24 --grid 48 --dm-step-nm 1 \
  --output-dir studies/gold_motheye_fff2d/results/dm_cpu_small
```

DMの刻みは `--axis dm-step --values 1,0.5,0.25 --methods dm-fff --order 2` で検査できます。
この軸は大きい刻みから小さい刻みの順に評価します。

全点で独立な体積吸収積分を求めたい場合は `--volume-absorption` を追加します。
層間の前進・後退場を求めるため層S行列を保持し、メモリと計算量が増えます。
体積積分の精度は `--absorption-quadrature`（既定7）と層数でも確認できます。

## 物理検査で調べる内容

- 平坦な吸収膜のFresnel/Fabry–Perot解析解との比較。
- 損失なしの傾斜構造に対する `R+T=1` と、任意の開放チャネル重ね合わせのパワー保存。
- 正入射でのx/y偏光の一致とポート規格化した相反性。
- 元の1次元TM式へ縮約したときの行列の一致。
- 体積散逸 `k0/(2*Pinc) * integral Im(E^H D) dz` と界面パワー差の一致。
- DMの積分刻みを半分にしたときのRK4誤差の減少。
- 各Au計算点の反射・基板流入・吸収が有限で受動系の範囲内にあること。
- `--loss-diagnostics` では深さ10%,50%,90%の構成行列の反Hermitian部分の固有値。

半無限Auに対し遠方透過率 `T_far=0`、基板流入は `P_sub`、
モスアイ吸収は `A_moth=1-R-P_sub`、全吸収は `A_total=1-R` です。
`R+A_total=1` は定義から成り立つため、それを独立なエネルギー検査には使いません。
構成行列の負の損失固有値は、指定入射でRが1以下でも人工的な利得を許すことを示します。
診断した深さ以外の全領域の受動性を保証する検査ではありません。

## 保存と再開

`checkpoint.json` を各点の直後に保存し、同じ設定・出力先では完了点を飛ばして再開します。
次数上限や波長を追加できます。材料、形状、固定Nz/Q、因子分解、コードが変わった場合は
混用を拒否するため、別の `--output-dir` を使ってください。
保存済み `UNPHYSICAL` 点もそのまま保持し、FAILED点は再試行します。

`cases.csv`、`report.json`、`plan.json`、`physics_validation.json`、
`convergence.png`、`changes.png` を出力します。各次数の回折効率はcheckpoint内に保存します。
同じコマンドに `--report-only` を追加すると保存済みデータから表・図を再生成します。

終了コードは、実行・物理異常で1、通常完了で0です。
未収束も終了コードで扱う場合は `--require-convergence` を指定し、その場合は未収束で2です。
収束判定の詳細は常に `report.json` を確認してください。

数式・文献・有限打ち切りの注意は [FORMULATION.md](FORMULATION.md) にまとめています。
