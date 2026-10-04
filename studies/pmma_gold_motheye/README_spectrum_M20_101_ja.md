# M=20・100層・400–700 nm（101点）のスペクトル計算

プロジェクトの `outputs` ディレクトリから、CUDAを使えるLinux Python環境で実行する。

```bash
python3 studies/pmma_gold_motheye/run_spectrum_M20_400_700_101.py --prepare-only
python3 studies/pmma_gold_motheye/run_spectrum_M20_400_700_101.py
```

実行するのは `.py` ファイル。1行目は材料と形状の準備だけを行い、2行目でRCWAを開始する。
結果は `studies/pmma_gold_motheye/results/measured_30nm_Nz100_M20_400_700_101pt/` に保存する。
同じコマンドと出力先で再実行すると、完了した波長点を `checkpoint.json` から読み、未完了の点だけを計算する。
全点完了後、`spectrum.csv` と `report.json` を書く。単一次数のため `report.json` の状態は
`single_order_unverified` となる。既存のM=20次数収束結果でも700 nm側には差が残る。

計算条件は波長400, 403, …, 700 nm、次数M=20、PMMAモスアイ100層、頂部金円板1層、
ASR格子256×256、`outer` 写像、谷部金30 nm、CUDA。周期200 nm、モスアイ高さ500 nm、
金の側面半径方向厚さ30 nm、頂部金円板の直径70 nm・厚さ30 nm、半無限PMMA基板を使う。

元のPMMA測定CSVは404.7 nmから始まる。このランチャーは元CSVを変更せず、最初の2測定点から
400 nmの屈折率を線形外挿した別CSVを出力先に作る。400 nmと403 nmだけがこの外挿に依存する。
`pmma_400nm_extension.json` に方法、元CSVのハッシュ、計算した値を記録する。
金CSVは既存の測定値をそのまま使う。

既存のM=20・100層・3波長計算に記録された時間は192.8、219.4、202.3秒/波長で、
平均204.8秒/波長。101点では約20,688秒＝5時間45分を中心値として見込む。
同じGPU環境でも波長や混雑によって変わるため、実行枠は6～8時間程度を見込む。
チェックポイントは各波長点の計算後に保存する。
