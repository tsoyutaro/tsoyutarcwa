# pmma_motheye_P200_H500_Nz10

このフォルダだけを別のWindows/Linux計算機へコピーして単独実行できる。
親プロジェクトや他条件のフォルダを参照しない。全条件を一括で走らせる機能はない。

## 条件

材料 pmma、周期 200 nm、高さ 500 nm、層数 10。
正方格子と六方対称格子は独立study。既定はM8/Q512で最適化しM8/M16とQ512/Q1024で検証。
データ・checkpoint・journal・manifestはこのフォルダのresultsへ保存する。
本番計算は未実行。

## 実行

フォルダ内で次を実行する。既存のCUDA版PyTorchを使用し、他ジョブの環境を変更しない。

```bash
python -m pip install -r requirements.txt
python run.py --prepare-only
python run.py --device cuda
```

CPU動作確認は本番と別保存先で実施する。

```bash
python run.py --device cpu --smoke --timeout-hours 0 --output-root results/cpu_smoke
```

格子を分けて異なる計算機で実行する場合:

```bash
python run.py --device cuda --lattices square --output-root results/square_run
python run.py --device cuda --lattices hexagonal --output-root results/hexagonal_run
```

同じコマンドで中断したcheckpointから再開できる。同じ保存先への二重起動は排他ロックで拒否する。
異なる条件または格子を別フォルダへコピーして実行する限り、共通ファイルの書き込みは発生しない。
既存runの途中結果を別機へ移して再開する場合、ソフトウェアの版とGPUモデルなどの整合確認を行う。
異なる機種へ新規条件を配布する場合は空のresultsから開始すればよい。

同梱のrcwa_extは共有ソルバーの固定コピー。optimizer以下がこの条件専用の最適化コード。
io_utilsとruntime_policyも同梱している。OS依存の絶対パスは設定に含めていない。

PMMA突起と基板に同じn(λ)を適用。k=0を仮定し、404.7 nm未満は400 nmまで線形外挿する。材料CSVはdata/Szczurowski.csv。

## 六方格子のD6専用実装

六方格子ではD6 native-star上のE1対成分を使用し、矩形基底の重複計算と12操作の反復を省く。正方格子の実装は従来どおり。`config.json` の `hexagonal_backend` を `reference` にすると元の六方格子実装を選択できる。実装変更時は空の出力先を指定する。ソースのSHA256は実行manifestに含まれる。
