# n1.5・n2.0・n2.5・PMMA：最適化の終了と後処理

20260806側の材料別43条件と、既存の dielectric_motheye_n25 の入口を確認しました。すべてv3の「最適化だけを実行して終了する」動作に対応しています。配布済み RCWA高速化と金属最適化_v3.zip のドライバーと後処理コードも、配置済みファイルと一致しています。

| 対象 | 条件数 | 実行入口 |
|---|---:|---|
| n1.5 | 13 | n1.5/<条件>/run.py |
| n2.0 | 13 | n2.0/<条件>/run.py |
| n2.5 | 13 | n2.5/<条件>/run.py |
| PMMA | 4 | PMMA/<条件>/run.py |
| 既存n2.5 | 1 | dielectric_motheye_n25/run.py または研究ルートの run_n25_nv_optuna_adam.py |

最適化・勾配監査・収束確認・計算時間測定・スペクトル取得は、別々のコマンドで実行します。最適化終了時に検証やスペクトル取得を自動で始める処理はありません。初期勾配監査も --mode audit で明示した場合だけ行います。mode allはoptimizeの互換aliasです。

## 各条件フォルダ内でのコマンド

例として、PMMA/pmma_motheye_P200_H500_Nz15 へ移動してください。他の材料・高さ・周期・層数の条件フォルダでも同じコマンドを使えます。元の設定にある試行数・Adam更新数で実行するため、以下は1 trialに制限する指定ではありません。

最適化を実行し、終了します。

```bash
python run.py --device cuda:0 --mode optimize --lattices hexagonal --output-root results/NV_shared_v3
```

終了後、必要なコマンドを別途実行します。以下は保存されたbest形状を固定し、最適化を追加で進めません。

次数8/12/16、grid512/1024、最適化と同じ31波長の収束確認です。

```bash
python run.py --device cuda:0 --mode convergence --lattices hexagonal --orders 8,12,16 --grids 512,1024 --wavelengths 400:700:10 --output-root results/NV_shared_v3
```

従来版と共有版の損失＋全logits勾配の計算時間比較です。

```bash
python run.py --device cuda:0 --mode timing --lattices hexagonal --order 8 --grid 512 --wavelengths 400:700:10 --repeats 3 --output-root results/NV_shared_v3
```

最適形状のスペクトルをCSV・PNG・JSONへ保存します。収束確認で採用した次数とgridを指定してください。

```bash
python run.py --device cuda:0 --mode spectrum --lattices hexagonal --order 16 --grid 1024 --wavelengths 400:700:3 --output-root results/NV_shared_v3
```

必要な場合の勾配監査です。

```bash
python run.py --device cuda:0 --mode audit --lattices hexagonal --output-root results/NV_shared_v3
```

正方格子は --lattices square、両格子は --lattices square,hexagonal を指定します。

## 出力と設定の指定

後処理は results/NV_shared_v3/<格子>/postprocess/<モード>/<ID>/ に保存し、latest_convergence.json / latest_timing.json / latest_spectrum.json に保存先を記録します。

- convergence.csv / report.json：同じ形状での次数・grid比較。numerical_convergence_passedは最高2次数と最高2gridの差を判定します。M8の誤差は各行の最高条件との差で別途確認できます。
- timingのreport.json：初回時間、warm測定の中央値、1波長換算、速度比、数値一致、CUDAピーク割当メモリ。
- spectrum.csv / spectrum.png：R/Tのスペクトル。

最適化を --config config_hex_trial1.json --output-root results/hex_trial1_v3 のような別設定・別出力先で実行した場合は、後処理にも必ず同じ --config と --output-root を指定してください。--checkpoint-layers を使用した場合も、同じ設定で実行します。現在のコードと設定に対応する出力先を使用し、旧版の実験へ新しいコードを混在させることはありません。

既に起動済みの旧版プロセスは、ソースを転送してもその制御が置き換わるわけではありません。停止と後処理の分離は、更新版で新しく起動したコントローラーに適用されます。
