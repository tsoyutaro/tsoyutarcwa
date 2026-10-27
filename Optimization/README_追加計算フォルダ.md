# 材料別・条件別の独立実行フォルダ

43条件を材料別の4フォルダに整理した。各条件フォルダにソルバー、最適化コード、設定、材料データ、独立したresultsを同梱している。各条件フォルダだけを別の計算機へコピーして実行できる。

## 保存先と構成

```
C:\Users\tsoyu\Documents\Codex\2026-08-06\files-mentioned-by-the-user-extensions\outputs\Optimization
├─ PMMA/   4条件：P200 nm、H500 nm、Nz5/10/15/20
├─ n1.5/  13条件
├─ n2.0/  13条件
└─ n2.5/  13条件
```

各材料フォルダの直下に条件フォルダを配置した。例：

```
Optimization/PMMA/pmma_motheye_P200_H500_Nz5/
Optimization/n1.5/dielectric_motheye_n15_P200_H500_Nz15/
```

一定屈折率の各13条件は、基準のP200/H500/Nz15、周期P100/150/250/300/350（H500/Nz15）、高さH100/200/300/400（P200/Nz15）、層数Nz5/10/20（P200/H500）で構成する。PとHの単位はnm。

既存の進行中の `Optimization/dielectric_motheye_n25` は元の場所で維持した。今回の `n2.5/dielectric_motheye_n25_P200_H500_Nz15` は同一物理条件を新規実行するための独立コピーであり、進行中の結果・checkpointを複製していない。他の既存プロジェクトも移動していない。

## 別の計算機での実行

材料フォルダ内の必要な条件フォルダを丸ごとコピーする。材料フォルダや親プロジェクトのコードを一緒にコピーする必要はない。各計算機では対応するGPU用PyTorchと実行環境を用意し、その条件フォルダ内で次を実行する。

```bash
python -m pip install -r requirements.txt
python run.py --prepare-only
python run.py --device cuda
```

各条件の正方格子と六方格子を既定では順に実行する。格子ごとに別の計算機へ配布する場合は、同じ条件フォルダを各計算機へコピーして次を実行する。

```bash
# 計算機A
python run.py --device cuda --lattices square --output-root results/square_run
# 計算機B
python run.py --device cuda --lattices hexagonal --output-root results/hexagonal_run
```

出力、checkpoint、journalは各条件内のresultsへ保存する。同じコマンドで中断再開でき、同じ保存先への二重起動はロックで拒否する。新規に配布した43条件のGPU本計算は開始していない。

## 確認状況

移動前後で各条件内の全ファイルのSHA256が一致した。移動後の全43条件について、プロジェクト外を作業ディレクトリとし、Pythonの `-I` と `--prepare-only` で起動を確認した。PMMAの材料CSVも各条件フォルダ内に存在する。

前回の確認では、PMMA5層を日本語と空白を含む別パスへコピーし、低次数のCPU試験でOptuna、Adam、勾配監査、再評価まで完了した。この試験は本計算結果には含めない。

Word原稿は今回変更していない。9本の数式は編集可能なWord標準のOMMLで保存済みで、Microsoft Word for Microsoft 365のPDF書き出しによる全10ページの描画結果を前回確認済み。未計算の性能値は［計算後記入］としている。

## 条件一覧

配布用の全条件ZIPにも、上記と同じ `Optimization/材料/条件/` の階層を収録した。個別ZIPは材料別に整理し、展開すると条件フォルダを1つ得られる。

## 六方格子のD6専用版への更新

未実行の追加43条件にD6専用版を同梱した。六方格子は `hexagonal_backend: d6-native-e1-pair-v1` でnative starとE1の対の2行による処理を使う。正方格子は既存のC2v実装を維持する。`reference` で元の六方格子実装を選べるが、構成変更時は空の別出力先を使用する。

CPU検証ではM8/Q512・15層の順計算＋逆伝播が約52秒から約17秒へ短縮し、R/Tと全半径の勾配も一致した。その後、ユーザー提供CUDAログでn2.5/PMMAのM8・15層の波長共有の一致と速度改善を確認した。進行中の既存計算は変更していない。検証コードはOptimization内のd6_optimizedに配置した。


## 波長共有・CUDA検証コード

検証コード一式をOptimization直下へ追加しました。[検証コードの場所とCUDA実行方法](検証コードの場所とCUDA実行方法.md)を参照してください。今回の反映は検証用ファイルの配置で、本番の最適化経路は未切替です。

## 本番Adamへの接続検証

[optimization_integration_validation/README_ja.md](optimization_integration_validation/README_ja.md)へ、softmaxと全logits勾配、実際のAdam更新、中断再開、best形状のストリーミング検証を行うコードとCUDAコマンドを追加しました。M2のCPU試験6ケースは合格です。本番共通RCWAへの波長共有の自動切替は行っていません。
