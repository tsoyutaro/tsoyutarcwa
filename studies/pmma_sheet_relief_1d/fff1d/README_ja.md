# 1次元RCWA-FFFとGIL別の次数収束

2026-11-15。提供論文の式(4)～(11)の積順序でRCWA-FFFを追加しました。定式化・角度・数値伝搬は [FORMULATION_ja.md](FORMULATION_ja.md)、実測検証は [検証報告](../validation/fff1d/README_ja.md) を参照してください。従来の通常RCWAは元の実行ファイルで引き続き使えます。

以下は `rcwa_ext/` と `studies/` があるRCWAルートで実行します。別の場所では `--solver-root PATH` を追加してください。既存と同じPython環境を使います。依存関係は既存の `requirements.txt` の範囲です。CPUではSciPyのsolve/expm、CUDAではtorchのsolve/matrix_expを使い、CUDA未対応環境でCPUへ自動変更しません。

## 指定された条件でGIL別に実行

周期=長辺3720 nm、短辺=高さ1500 nm、Au公称50 nm、波長700 nm、p偏光、Nz960、空気溝/水溝、最終透過先は水です。相似形状は既存の37.2:15:15のままです。Au膜厚は周期に比例させません。

```sh
python studies/pmma_sheet_relief_1d/fff1d/converge_gil.py --gil-values "0,0.5,1,2" --gil-interface-values all --start 16 --step 16 --max-order 128 --periods 3720 --wavelengths 700 --polarization p --slices 960 --coating gold --gold-nm 50 --voids both --device cuda:0 --phase-tol 0.00005 --phase-max 262144 --tolerance 0.001 --output-dir studies/pmma_sheet_relief_1d/results/fff1d/P3720_w700_Nz960_GIL
```

**128は実行例の上限であり、収束確認済みの推奨次数ではありません。** Mは保持する最大の正の次数で、物理的なフーリエ次数は−M～+Mの2M+1本です。

| オプション | 指定方法・意味 |
|---|---|
| `--gil-values "0,0.5,1,2"` | 横方向の全遷移幅δ、nm。任意の非負値を指定可能 |
| `--gil-interface-values inner` | PMMA/Auのみ |
| `--gil-interface-values outer` | Au/溝媒質のみ。溝媒質は空気または水 |
| `--gil-interface-values both` | 上記の両界面 |
| `--gil-interface-values all` | inner、outer、bothをそれぞれ独立に計算 |
| `--gil-interface-values inner,both` | 必要な組合せのみ選択 |
| `--max-order 128` | 全GILで共通の上限 |
| `--max-orders "256,192,128,128"` | GIL値の並び順に各幅の上限を指定。上の共通上限と同時には使わない |
| `--start 16 --step 16` | 16,32,48,…と増加。上限が刻みと合わない場合も上限ちょうどで計算 |
| `--tolerance 0.001` | 比較するパワー比の絶対差。0.1パーセントポイント |
| `--gil-quadrature 16` | GILのフーリエ係数積分の精度。32等と比較可能 |

例えば上のコマンドの `--max-order 128` を `--max-orders "256,192,128,128"` に置き換えると、δ=0/0.5/1/2 nmの上限がそれぞれ256/192/128/128になります。同じ幅でのinner/outer/bothには同じ上限を使います。界面別にも上限を変える場合は界面を一つずつ選び、出力先を分けて実行してください。

δ=0では界面選択による違いがないため一度だけ実行します。既定候補とallでは合計10キャンペーンです。空気溝と水溝は各キャンペーン内で両方を比較し、一方だけ合格しても収束とはしません。

## 収束判定・停止・再開

全出射次数R_m/T_m、合計R/T/A、反射20°以内/外の最大差を調べ、連続する2比較が許容値以下で、位相平均も条件を満たした場合に合格します。有限次数で負の吸収などが出た点はCSVに保存し、合格から除外します。値の切り詰めは行いません。

上限、OOM、位相平均未収束、その他のエラーでそのキャンペーンを停止し、次のGIL条件へ進みます。Ctrl+Cでは全体を中断します。再開は同じコマンド・同じ出力先です。完了済み点を再計算せず確認し、必要な次数から続行します。最大次数を上げて再開することもできます。GILや層数などの計算条件、開始次数、刻み、ソースを変更するときは新しい出力先を使ってください。

親の `gil_summary.csv/json` に各GILの状態、最後に完了した次数、上限を保存します。各子フォルダの `convergence.json` に停止理由、`adjacent_changes.csv` に比較結果、`order_M/spectra.csv` と `diffraction_orders.csv` に生データ、`logs/` に実行ログを保存します。`discretization_converged=False` は各単独点の扱いであり、キャンペーンの判定は `convergence.json` を参照します。

終了コードは全条件合格=0、上限等で未収束=2、OOMを含む=3、設定/計算エラーを含む=4、中断=130です。位相未収束は各子の状態で識別できます。停止はAu構造の収束を証明するものではありません。次数と層数は独立に確認する必要があります。

## 単独計算と従来法への還元検査

```sh
# 通常の斜面法線を使い、PMMA/Auだけ1 nmのGILを追加
python studies/pmma_sheet_relief_1d/fff1d/run_fff.py --periods 3720 --wavelengths 700 --polarization p --slices 960 --order 64 --gil-nm 1 --gil-interfaces inner --device cuda:0 --output-dir studies/pmma_sheet_relief_1d/results/fff1d/single_inner1

# GPU上で従来法とのθ=90°一致・対称性・GIL積分などを検証
python studies/pmma_sheet_relief_1d/validation/fff1d/validate.py --device cuda:0 --suite standard --output-dir studies/pmma_sheet_relief_1d/validation/fff1d/cuda_0

# θ=90°の通常RCWA因子分解。構造と入射角は変更しない
python studies/pmma_sheet_relief_1d/fff1d/run_fff.py --theta-deg 90 --gil-nm 0 --section-sampling midpoint --periods 3720 --wavelengths 700 --polarization p --slices 960 --order 64 --device cuda:0 --output-dir studies/pmma_sheet_relief_1d/results/fff1d/theta90
```

`--theta-deg` は入射角ではなくFFF射影角です。省略時は実際の斜面角53.4985588795°を使用します。GIL=0かつ同じ中点層分割でθ=90°にすると、従来の1次元ソルバーへ還元します。論文の層入口サンプリングを使うときは `--section-sampling entrance` を指定します。`--symmetry full` で左右対称性の縮約を解除して比較できます。

GILは複素屈折率nを横方向に線形に変化させます。εを線形補間する指定ではありません。水平な端面をz方向にぼかす処理は追加しません。δは法線方向のAu膜厚ではありません。

## 維持した高速化と計算量

波長間の形状/フーリエ係数共有、左右鏡映対称性、s/p分離、位相平均のPreparedSheet、点キャッシュ、CSV追記、逐次S行列接続を維持しています。形状計画のキャッシュは既存とFFFそれぞれに上限128 MiBを設けています。GILの積分節点は共有し、材料分散を含む補正値は波長ごとに計算します。

FFFでは一般に状態行列が2(M+1)次元になるため、通常RCWAのM+1次元の固有値問題より、同じMの1点計算が速いとは限りません。小さいMで必要な精度へ到達できるかを比較してください。複数層をまとめるGPU処理は追加していません。今回の指数行列伝搬でも層ごとの呼び出しです。

通常RCWAの反射/透過点をFFFの点として使い回すことはできません。FFF専用の `results/fff1d/cache/` に、ソース・材料・GIL・次数・層数・精度設定・device等を含めたキーで保存します。元の通常RCWAのソース署名と既存結果は維持しています。
