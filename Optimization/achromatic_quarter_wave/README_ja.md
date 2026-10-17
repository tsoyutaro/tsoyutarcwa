# 可視広帯域・異方性モスアイ四分の一波長板

既存の `Optimization/cr_motheye`、`Optimization/dielectric_motheye_n25` と並ぶ研究用フォルダ。
プロジェクト直下の `rcwa_ext` を使用する。既存ソルバー本体は変更しない。
今回の実行結果は[INITIAL_RESULTS_ja.md](INITIAL_RESULTS_ja.md)にまとめた。
長方形・楕円を両方NV法で検証する手順と、等軸楕円／円の一致確認は[NV_README_ja.md](NV_README_ja.md)を参照。

## 最初に調べる内容

1. 周期200 nm、高さ500 nmで、偏光間位相差がどの程度得られるか。
2. 同じ形状で400–700 nmの位相差を90°に近づけられるか。
3. x/y両偏光の透過率、反射率、45°直線偏光入射時の円偏光度を両立できるか。

最初は**正方格子で楕円・長方形の両方**を比較する。
長方形格子にしなくても、断面のx/y寸法を変えることで異方性を作れる。
格子の効果と断面の効果を分けるため、格子周期を追加の最適化変数にするのは次段階とする。
長方形格子も `--period-x` / `--period-y` で指定できる。

長方形は細長い断面や大きな充填率を試しやすい。楕円は角のない形状として比較する価値がある。
同じ外接幅でも占有面積は異なり、楕円はπab、長方形は4abになる。
したがって「同じ縦横寸法での比較」と「各形状で最適化した性能の比較」は区別する。

## 物理モデルと初期値

| 項目 | 初期設定 |
|---|---|
| x/y周期 | 200 / 200 nm、正方格子 |
| 高さ | 500 nm |
| 構造屈折率 | 1.5、2.0、2.5。実数、波長非依存 |
| 周囲 | 空気、n=1 |
| 入射・基板 | 空気から垂直入射、半無限基板n=1.5 |
| z分割 | 等厚15層。実材料の15種類の積層ではなく、テーパーの階段近似 |
| 断面 | 中心を揃えた楕円、または軸に平行な長方形 |
| 寸法の定義 | 楕円では半径a,b、長方形では半幅a,b。全幅は2a,2b |
| 寸法範囲 | 半幅5–95 nm、隣接構造の最小隙間10 nm。変更可能 |
| 高さ方向の制約 | a(z),b(z)の両方が上から下へ単調非減少 |
| 設計変数 | 初期は各軸3個の節点。softmaxで正の間隔を作り、単調性を保証 |
| 初期探索 | 6個の初期形状 + Sobol候補。全材料・形状に同じ候補を使用 |
| 数値計算 | complex128、Redheffer接続、前方透過・反射のS行列 |
| 楕円 | Raster + standard RCWA。既存の円専用NV法は転用しない |
| 長方形 | ASR-FR。`--rectangle-method raster` で独立な比較も可能 |
| 透過率条件 | 初期は各評価波長でTx00,Ty00ともに80%以上 |

基板n=1.5は今回の初期仮定。構造と同じ屈折率の基板を使う場合は、材料ごとに
`--substrate-index` を変更する。周期200 nm・基板n=2.5では500 nm未満で基板側に
高次回折光が伝搬し得るため、全透過率と0次透過率を区別する必要がある。
このコードは両方を出力し、波長板の評価には0次透過率を用いる。
基板n=1.5・周期200 nmでは400–700 nmに外部伝搬高次回折はない。
基板裏面、有限厚基板の干渉、実材料の分散・吸収、斜入射は今回のモデルに含めない。

## 「位相」の意味

計算するのは共通の透過位相ではなく、

\[
\delta=\arg(t_{xx}t_{yy}^{*})
\]

というx/y偏光間の位相差。主値は−180°〜180°で保存する。
90°と270°（主値−90°）は逆の符号の四分の一波長板に対応する。
主値の絶対値の最大値と、360°の整数倍を含む光路位相差の最大値は同じではない。
低透過率で不安定な位相を性能と誤認しないよう、無制約の最大値と透過率制約付きの最大値を併記する。

45°入射で円偏光を得るには、位相差±90°に加えて、透過後のx/y振幅が等しいことが必要。
`stokes3_45` は完全な透過Jones行列から計算する規格化Stokes S3で、絶対値1が円偏光、0が直線偏光。
左右の呼び方は時間・観測方向の規約に依存するため、このコードでは符号で記録する。
広帯域の指標は同じ符号のS3を帯域全体で維持することも評価する。

伝搬位相だけで見積もると、500 nmで四分の一波長板に必要な平均実効複屈折は
Δn_eff≈λ/(4h)=0.20（400 nm）、0.275（550 nm）、0.35（700 nm）。
これは共振・界面位相を無視した目安であり、実現可否の厳密な上限ではない。
広帯域化では位相差の大きさだけでなく、その波長依存性の設計が必要になる。

## 実行方法

プロジェクト直下（`rcwa_solver_auto.py` がある場所）で既存のPython環境を使用する。
Optunaは不要。依存ライブラリが不足する場合だけ `requirements.txt` を使用する。

まず物理チェック：

```powershell
python Optimization/achromatic_quarter_wave/validate.py --output Optimization/achromatic_quarter_wave/results/validation.json
```

初期探索（M=3は高速な候補抽出用で、収束済みの答えではない）：

```powershell
python Optimization/achromatic_quarter_wave/run.py screen --samples 32 --output Optimization/achromatic_quarter_wave/results/screen_M3
```

`summary.csv` はケース別の結果、`per_wavelength_maxima.csv` は各波長で探索中に観測した最大位相差。
各波長の最大値は異なる形状から得られることがあり、一本の広帯域波長板の性能を示す表ではない。
最大位相差の候補は `max_phase_*.json` にも個別保存する。
`best_maximin_phase_*.json` は全評価点で透過率条件を満たす候補のうち、最小位相差が最大の形状。
`best_qwp_*.json` は同一形状で90°に近づける目的関数の最良候補。
`history.jsonl` に全候補の形状とスペクトルを保存する。同じ設定・出力先でscreenを再実行すると既存候補を再利用する。

各波長の最大候補を固定してM=5で再計算する場合：

```powershell
python Optimization/achromatic_quarter_wave/recheck_maxima.py --screen Optimization/achromatic_quarter_wave/results/screen_M3 --order 5 --grid 256 --output Optimization/achromatic_quarter_wave/results/recheck_M5
```

これは固定候補の再計算であり、高次数で再最適化した最大値ではない。

同一形状の四分の一波長板最適化（例：n=2.5の長方形、7波長点）：

```powershell
python Optimization/achromatic_quarter_wave/run.py optimize --shapes rectangle --indices 2.5 --goal qwp --design Optimization/achromatic_quarter_wave/results/screen_M3/best_qwp_n2p5_rectangle.json --order 5 --grid 256 --wavelengths 400 450 500 550 600 650 700 --maxiter 20 --output Optimization/achromatic_quarter_wave/results/qwp_n25_rectangle_M5
```

`--goal max-phase` は評価点のうち最大の位相差、`--goal maximin-phase` は評価点のうち最小の位相差を大きくする。
`--goal qwp` は90°からの最悪誤差、同一符号の円偏光度、透過率不足のペナルティを小さくする。
制約は目的関数内のペナルティなので、結果の `passes_transmission` も必ず確認する。
`--nodes 15 --layers 15` で形状の自由度を増やせる。高さ500 nmは固定。
各節点はz=0〜h上の寸法で、各層の中央で補間した半幅をRCWAに渡す。
勾配が不連続になるラスタ楕円にも対応するため、現段階は勾配不要の差分進化法を用いる。

固定形状の再計算：

```powershell
python Optimization/achromatic_quarter_wave/run.py verify --design Optimization/achromatic_quarter_wave/results/qwp_n25_rectangle_M5/best_design.json --orders 3 5 7 --verify-grid 256 --wavelengths 400 450 500 550 600 650 700 --output Optimization/achromatic_quarter_wave/results/verify_n25_rectangle
```

verifyはMを順番に増やし、最後にQを2倍、その後z分割数を2倍にする。
`convergence.csv` の隣接段階を比較し、M/Q/zそれぞれの依存性を確認する。
最終2段階の位相差の変化も保存するが、これだけで全項目の収束合格とは判定しない。
波長を10 nm間隔以下に増やすなどして、評価点間の共振や性能劣化も確認する。
M=3では(2M+1)^2=49個、M=7では225個の空間調和波を保持し、偏光を含む行列はさらに大きくなる。
特にCPUでMを増やすと計算時間は急増する。初期確認では `--threads 1` を既定とした。
CUDA環境では `--device cuda` を指定できる。

## 検証と解釈の注意

`validate.py` は空気→ガラスのFresnel透過率、等軸断面のゼロ位相差、軸交換による符号反転、
交差偏光、エネルギー保存、高次回折を含む透過パワーの計数を確認する。
既存ASR法は有限Mで微小な軸交換誤差を持つため、M=3のASR対称性チェックは位相0.05°、パワー2×10^-4を許容する。
ラスタ形状の対称性は機械精度に近い許容値で確認する。
エネルギー保存が成立しても位相差が収束したとは限らない。

探索結果は指定した寸法範囲・節点数・候補数・評価波長・計算次数の中での到達値。
真の大域最大値や実製品の性能とは区別する。
特に楕円のstandard RCWAは境界のFourier展開が遅く収束することがあるため、MとQの再計算を省略しない。
最終的に実材料のn(λ)+ik(λ)、残膜、基板、加工寸法の制約に置き換えて評価する。
