# 六方格子のD6専用NVM実装

前回の12操作を反復する検証用実装を置き換え、D6のE1表現に対応する対の2行で同じ平均化を計算する。検証用構造で基準との一致とCPU時間の改善を確認した。GPU速度は未測定。

## 変更した処理

`d6_nvm.py` の `D6FastRCWA` は、垂直入射、中央の円形断面、等方的な損失なし材料、x偏光、非磁性、fieldsなしの六方格子用の追加クラスである。既存のrcwa_extを直接変更しない。

1. 誘電率、逆数誘電率、法線射影のFourier行列を、D6で閉じたnative star上で直接生成する。矩形全体の材料行列・逆行列・P/Qを経由しない。入力生成も凍結した基準コードと別々に比較する。
2. スカラーA1/A2/B1/B2/E1/E2のmultiplicity空間で逆行列を計算する。二次元表現では、2行で共通する小行列の逆行列を1回だけ求める。
3. Cartesian成分で必要なP/Qの作用を計算し、E1の2行をバッチで処理する。12個の群操作を層ごとに反復しない。
4. Fourier次数、格子、dtype、deviceが同じ場合、半径と波長に依存しない基底をプロセス内で再利用する。キャッシュは最大4条件。

M8ではscalar starは217次元、電場・磁場のE1 source rowは73次元。M16では817次元、273次元となる。NV法の対称化された因子分解とRedheffer cascadeは維持する。

## 平均化を省略したのではない

E1のrow 0の基底をU、matrix unitで対応づけたrow 1の基底をVとすると、12操作のReynolds平均の必要な行列要素を

`P_reduced = (U_E^H P U_H + V_E^H P V_H) / 2`

で厳密に計算できる。Qも電場と磁場の基底を入れ替えて同様に計算する。任意の非対称な複素乱数行列についても、12操作による平均と一致することを `check_e1_identity.py` で確認する。

理論の根拠：[MIT, Schur's lemma and the orthogonality theorem](https://symm4ml.mit.edu/symm4ml_s26/notes/representations)。既存のNV因子分解：[Weismann et al.](https://arxiv.org/abs/1507.06364)。上式のこの実装への適用は、これらの表現論の性質から導出し、数値比較でも確認した。

## ファイル

- `d6_nvm.py`：最適化コードにも同梱するD6専用クラス。
- `validate.py`：行列、光学応答、勾配、有限差分、負例、時間を比較する。
- `check_e1_identity.py`：任意の複素行列で、12操作とE1の2行の計算が一致するかを調べる。
- `block_assembly.py`：検証用adapter。D6専用クラスと基準を同じ入出力で呼ぶ。
- `legacy_block_assembly.py`：前回の12操作を反復する試作。比較用に維持する。
- `rcwa_ext/`：既存ソルバーの凍結コピー。基準の光学応答はこの実装を直接呼ぶ。
- `verified_results/` と `D6検証結果_ja.md`：作成時の実測結果。

## 実行環境

Python 3.10以降、PyTorch、NumPy、torcwa。Optunaはこの比較検証には不要。
GPU環境ではGPU/CUDAに対応したPyTorchを用意する。依存が未導入の場合：

```powershell
python -m pip install -r requirements.txt
```

## 検証コマンド

このREADMEと同じフォルダで実行する。

```powershell
# 小規模、両格子。前回試作の12操作との比較も含む
python validate.py --suite quick --device cpu --compare-legacy --output validation_results/quick_cpu

# 六方格子のM4/M8、材料・波長を増やした比較と5回の時間測定
python validate.py --suite standard --device cuda --lattices hexagonal --compare-legacy --benchmark-repeats 5 --output validation_results/standard_gpu

# M8、Q512、15層、屈折率2.5、550 nm。応答と全15半径の勾配を比較
python validate.py --suite quick --device cuda --orders 8 --grid 512 --layers 15 --materials n2.5 --lattices hexagonal --benchmark-repeats 3 --output validation_results/Q512_L15_gpu

# M16の行列比較。光学応答・勾配は計算しない
python validate.py --suite quick --device cuda --orders 16 --grid 128 --matrix-only --lattices hexagonal --output validation_results/M16_gpu

# 12操作と2行による平均化の独立チェック
python check_e1_identity.py --device cpu --output validation_results/e1_identity.json
```

GPUを使わない場合は `--device cpu` とする。`--materials` は光学応答・勾配の材料を選ぶ。端の半径に対する行列試験は材料選択にかかわらず実行する。

全件合格時は `PASSED` と終了コード0、不一致時は `FAILED` と終了コード1。`--output` 配下の `report.json` と `summary.csv` に判定・誤差・実行環境・ソースSHA256・時間を保存する。

## 時間の範囲

`reference_timing`、`candidate_timing`、`legacy12_timing` は、共通入力と再利用する基底を用意した後の逆行列・P/Q構築の測定。ウォームアップ後の中央値を記録する。`reference_input_timing` と `candidate_input_timing` は材料・法線行列の生成を別に測る。

光学応答の `reference_forward_backward_seconds` と `candidate_forward_backward_seconds` は、層の行列生成、固有値計算、cascade、反射率、逆伝播を含む。基底・境界ポートの初期化、Python起動、Optuna controllerの時間は含まない。最適化全体の所要時間を直接表す値ではない。

境界ポートの行列は既存のfull-space APIを使う。初回のD6基底生成にも全群の行列を一時的に使うため、総メモリが常に縮約行列だけになるわけではない。

## 最適化フォルダでの使用

追加した未実行の条件では、`config.json` の `hexagonal_backend` が `d6-native-e1-pair-v1` の場合に、六方格子のみD6専用版を使用する。正方格子は既存のC2v実装を維持する。各条件は引き続き単独で別PCへコピー・実行できる。

```powershell
python run.py --prepare-only
python run.py --device cuda --lattices hexagonal
```

元の六方格子実装を選ぶ場合は `hexagonal_backend` を `reference` として、空の別出力先で実行する。構成またはソースが変わった結果への途中再開は、manifestの検査で拒否する。D6専用モジュールのSHA256も検査対象に含めた。

進行中の元の `dielectric_motheye_n25` や他の既存計算は更新しない。

## 検証の区別

同じFourier次数で基準と新版が一致することと、次数を増やして物理量・勾配が収束することは別々に判定する。PMMAの小規模最適化ではM1→M2の勾配収束は両実装とも未達、同じ次数での実装差は許容差内、M2→M4への精密化は合格だった。判定の許容差は変更していない。

今回の検証値は高速化の比較用であり、最適化の成果データや論文の性能値として扱わない。
