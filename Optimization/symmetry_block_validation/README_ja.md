# 対称性ブロックによるNVM行列構築の検証

既存のRCWA/NVMソルバーを基準に、対称性ブロックの逆行列を使って必要なP・Qだけを直接組み立てる試作実装を比較する。正方格子と六方格子の両方を対象とし、本番のOptimizationコードや計算結果は変更しない。このフォルダだけを別の計算機へコピーして実行できる。

## ファイル

- `validate.py`：比較、勾配・有限差分・負例試験、結果保存、終了コードによる合否判定。
- `block_assembly.py`：実験用の対称性ブロック逆行列とP・Qの直接構築。
- `rcwa_ext/`：既存ソルバーの固定コピー。基準となる光学応答はこのコードをそのまま呼び出す。
- `data/Szczurowski.csv`：既存のPMMA条件と同一の屈折率データ。波長の単位はµm。
- `verified_results/`：作成時の実行結果を収録する。

## 実行環境

Python 3.10以降、PyTorch、NumPy、torcwaを用いる。Optunaは不要。
既存のGPU環境では、そのGPUに対応するCUDA版PyTorchを使う。

```powershell
python -m pip install -r requirements.txt
```

このコマンドの前に使用するPython環境を選択する。検証に使うGPU用PyTorchを新たに入れる場合は、GPU/CUDAに適合する公式の配布を選ぶ。

## 実行コマンド

このREADMEと同じフォルダを作業ディレクトリとする。

### 小規模な確認：CPU

```powershell
python validate.py --suite quick --device cpu --output validation_results/quick_cpu
```

M1/M2で正方格子・六方格子を確認する。行列8件、負例2件、光学応答と勾配6件の計16件。

### M4/M8と材料条件を増やした確認：GPU

```powershell
python validate.py --suite standard --device cuda --benchmark-repeats 3 --output validation_results/standard_gpu
```

行列8件、負例2件、M8の光学応答と勾配12件の計22件。一定屈折率1.5/2.0/2.5、PMMAを含む。PMMAの400/550/700 nmも確認する。
GPUを使用しない場合は `--device cpu` に置き換える。

### M8/M16の行列比較だけを実行

```powershell
python validate.py --suite standard --device cuda --orders 8 16 --matrix-only --benchmark-repeats 3 --output validation_results/matrix_M8_M16_gpu
```

M16の基底生成・基準行列には多くのメモリと時間を要する。格子ごとに別の計算機で実行する場合：

```powershell
python validate.py --suite standard --device cuda --orders 8 16 --matrix-only --lattices square --output validation_results/square_M8_M16
python validate.py --suite standard --device cuda --orders 8 16 --matrix-only --lattices hexagonal --output validation_results/hexagonal_M8_M16
```

### 実際の空間サンプリングに近づける

```powershell
python validate.py --suite standard --device cuda --orders 8 --grid 512 --benchmark-repeats 3 --output validation_results/M8_Q512_gpu
```

既定のサンプリング数はquickで48、standardで128以上。`--grid 512` により最適化と同じQ512でも比較できる。いずれも検証用の3層構造を使い、43条件全体の最適化は開始しない。

## 合格条件

全件合格すると `PASSED` を表示し、終了コード0を返す。不一致・非有限値・例外は `FAILED` と終了コード1になる。PowerShellでは `$LASTEXITCODE` で確認できる。

`--output` に指定したフォルダへ `report.json` と `summary.csv` を保存する。途中結果も各試験の終了後にJSONへ保存する。別の実行結果を残す場合は保存先を分ける。

比較内容：

1. 誘電率と逆数誘電率のFourier行列が、それぞれの対称性ブロックを保存するか。
2. 基準のP・Qを射影した行列と、試作版で直接作った小行列が一致するか。P、Q、PQの相対誤差はいずれも2×10⁻⁹以下。
3. 基準計算と試作版で、3層の反射・透過電場ベクトル、反射率R、透過率Tが一致するか。R/Tの絶対誤差は10⁻⁸以下、電場の相対誤差は2×10⁻⁷以下。
4. 損失なしの条件で、RとTが物理的範囲に入り、|R+T−1|が10⁻⁶以下か。
5. Rの各層半径に対する自動微分勾配が、基準計算と一致するか。絶対許容差10⁻⁷、相対許容差2×10⁻⁵。
6. 試作版の自動微分による方向微分が、中心有限差分と一致するか。刻み2×10⁻⁵、絶対許容差10⁻⁶、相対許容差3×10⁻⁴。
7. 対称性を壊した行列、中心からずらした構造、斜入射を、適用範囲外として拒否するか。

勾配の変数は規格化半径ρ=r/P。nm単位の半径rに対する勾配は、出力された勾配を周期P[nm]で割る。
PMMAは既存コードと同じ線形補間・k=0を使用し、400 nmでは最初の区間による限定的な外挿を用いる。

## 正方格子での変更内容

M8では289次元のスカラーFourier空間をee/eo/oe/ooの81/72/72/64次元に分け、それぞれで逆行列を計算する。大きな逆行列へ再展開せず、小ブロックをベクトルへ作用させる。

NV法の対称化された因子分解

`epsilon_ab = delta_ab E + (Delta N_ab + N_ab Delta)/2`

を維持する。ここでEは誘電率のFourier行列、Deltaは逆数誘電率のFourier行列の逆行列からEを引いたもの。電場と磁場の基底、および微分で偶奇が入れ替わる関係を保ったまま、145次元のP・Qを直接構築する。

## 六方格子に適用する際の違い

六方格子の現行ソルバーは、矩形の次数集合からD6で閉じたnative starへ移し、そのstar上で逆行列を計算している。現行コードには、その分岐前に矩形全体の逆行列・P・Qを構築する処理も残る。この重複処理の除去は優先して検討できる。

今回の試作版は次の手順を使う。

1. 基準と同じ `max(|m|,|n|,|m−n|) ≤ M` のstarを使用する。
2. スカラー誘電率行列をD6のA1/A2/B1/B2/E1/E2のisotypic成分へ分解し、空でない成分ごとに逆行列を計算する。二次元既約表現を含むため、正方格子の偶奇4分割を流用しない。
3. 電場・磁場のE1表現のx入射に対応する行を使用し、必要なP・Qを直接構築する。M8では73次元。
4. 基準に含まれる12操作のReynolds平均も保ち、同じ離散化を比較する。

**starへの制限と逆行列の順序は入れ替えない。** 矩形全体の逆行列を後からstarへ切り取ると、除外した次数を経由する結合が残り、基準と異なる計算になる。

六方格子の試作版では、確認しやすさを優先して12操作の平均化を作用計算で繰り返す。これは最終的な高速化実装ではない。正しく一致しても、速度が改善するとは限らない。採用判断には、重複処理の除去、基底の再利用、平均化の整理とGPUでの測定が必要。

## 測定の範囲と残っている処理

`--benchmark-repeats 3` は、共通の入力行列と再利用可能な対称性基底を作った後の、逆行列・P/Q構築のみを計測する。ウォームアップ後の各回の時間と中央値を保存する。CUDAでは同期を取り、その区間の追加ピーク割当量も保存する。最適化全体の実行時間、総GPUメモリ、基底生成時間を表す数値ではない。

試作版は全サイズのスカラー逆行列、NV誘電率テンソル、P・Qを構築しない。一方、材料のFourier畳み込み行列と法線射影行列、および境界ポートの行列は既存コードで作る。これらもすべて小ブロックから生成する最終実装までは含めていない。

今回の検証は、同じRCWA/NVの離散化に対する実装の一致を確認するもの。得られたR/Tは最適化の成果データには含めない。

## 数式の根拠

- 既存コードが使用するNV法：[Weismann et al., Accurate near-field calculation in the rigorous coupled-wave analysis method](https://arxiv.org/abs/1507.06364)
- 対称性による不変部分空間とSchurの補題：[MIT, Representations of Finite Groups](https://symm4ml.mit.edu/symm4ml_s26/notes/representations)
