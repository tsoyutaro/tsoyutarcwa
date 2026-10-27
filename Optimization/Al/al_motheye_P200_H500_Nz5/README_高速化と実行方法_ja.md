# al_motheye_P200_H500_Nz5

周期200 nm、高さ500 nm、5層の独立した最適化パッケージです。このフォルダ全体を別PCへコピーできます。最適化は既定でM8/grid512、400–700 nmを10 nm刻み、Optuna 30試行・Adam最大500更新です。

最適化後は検証を自動実行せず、best形状を保存してプロセスを終了します。従来のmode allもoptimizeのaliasです。以下の各コマンドを、このフォルダ内で別々に実行してください。

```bash
python run.py --device cuda:0 --mode optimize --output-root results/NV_shared_v3
```

終了後、同じbest形状を固定して次数・gridの収束を確認します。

```bash
python run.py --device cuda:0 --mode convergence --orders 8,12,16 --grids 512,1024 --wavelengths 400,550,700 --output-root results/NV_shared_v3
```

従来版と共有版の損失＋勾配の時間を比較します。初回時間と、初回を除いた繰り返しの中央値を記録し、数値一致も検査します。単一波長ならwavelengthsを550に変更できます。順計算だけならforward-onlyを追加します。

```bash
python run.py --device cuda:0 --mode timing --order 8 --grid 512 --wavelengths 400:700:10 --repeats 3 --output-root results/NV_shared_v3
```

スペクトルCSV・PNG・JSONを取得します。

```bash
python run.py --device cuda:0 --mode spectrum --order 16 --grid 1024 --wavelengths 400:700:3 --output-root results/NV_shared_v3
```

正方格子と六方格子を分ける場合は、lattices square または lattices hexagonal を各コマンドへ追加します（引数には先頭に -- を付けます）。出力先は results/NV_shared_v3/<格子>/postprocess/<モード>/<識別子>/ で、latest_convergence.json等に保存先を記録します。convergenceのnumerical_convergence_passedで最高2次数・最高2grid間の許容差を確認します。有限の比較であり、無限次数への収束の証明ではありません。

最適化時にcheckpoint-layers等を指定した場合は、後処理も同じ設定で実行します。source/config/runtimeが異なる古い出力先の流用は拒否します。旧版の最適化結果はそのまま保持し、新方式には新しい出力先を使ってください。

configのrcwa_accelerationはauto（行列＋波長共有）、matrix（行列のみ）、reference（従来）の3種類です。後処理のみ従来経路で計算する場合は --analysis-backend reference を指定します。fields・偏光・形状が高速経路の対象外なら従来RCWAを使います。

金属柱＋同じ金属の半無限基板、空気入射、コーティングなしです。同梱CSVの複素epsilon=(n+ik)^2を波長に対して線形補間し、外挿しません。目的は平均Rの最小化で、全吸収率1-Rの最大化と等価です。power_into_substrateは基板表面へ流れる電力で、遠方透過率ではありません。A_total=1-R、A_moth=1-R-power_into_substrateを表面電力収支から計算します。体積損失積分による独立検証は含みません。

共通化実装のCPU照合は合格済みです。金属のCUDA試験は手元のPCでは未実施です。研究ルートのvalidate_rcwa_acceleration.pyで追加確認できます。

初期・最終の勾配監査も自動実行しません。必要な場合は、同じ設定で `python run.py --device cuda:0 --mode audit --output-root results/NV_shared_v3` を別途実行してください。最適化コマンドは勾配の次数収束確認を自動で開始しません。
