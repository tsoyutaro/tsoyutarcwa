# 波長共有・CUDA検証コードの場所と実行方法

検証一式を、この20260806側のOptimizationフォルダへ配置しました。

| フォルダ | 内容 |
|---|---|
| [optimization_integration_validation](optimization_integration_validation/README_ja.md) | 本番softmax・全logits勾配・Adam更新・checkpoint再開・best形状のストリーミング再評価を従来版と比較。 |
| [square_spectral_cuda_validation](square_spectral_cuda_validation/README_CUDA_ja.md) | 正方格子のCUDA対応ドライバー。一定屈折率1.5/2.0/2.5とPMMA分散の波長共有を検証。 |
| [hex_spectral_cuda_validation](hex_spectral_cuda_validation/README_CUDA_ja.md) | 六方格子のCUDA対応ドライバー。従来版・D6専用版・波長共有版を比較。 |
| [square_spectral_validation](square_spectral_validation/README_ja.md) | 正方格子のCPU検証コードと、次数8・15層の実測結果。 |
| [spectral_reuse_validation](spectral_reuse_validation/README_ja.md) | 六方格子の波長共有・PMMA分散のCPU検証コードと結果。 |
| [d6_optimized](d6_optimized/README_ja.md) | 先に配置済みのD6専用の行列構築・検証コード。 |

各パッケージのZIPもこのフォルダ直下に配置しました。別PCへはZIPをコピーして展開できます。

## CUDAの実行

CUDA対応PyTorchを利用するPython環境を有効にして、PowerShellで実行してください。別PCでは作業フォルダのパスを変更します。

```powershell
Set-Location -LiteralPath 'C:/Users/tsoyu/Documents/Codex/2026-08-06/files-mentioned-by-the-user-extensions/outputs/Optimization/square_spectral_cuda_validation'
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available()"
```

先にCPUとの照合と有限差分を含む小規模検証を実行します。

```powershell
python benchmark_device.py --device cuda:0 --order 2 --grid 48 --layers 5 --material n2.5 --wavelengths 400,550,700 --check-fd --compare-cpu --output results/cuda/small_n25.json
python benchmark_device.py --device cuda:0 --order 2 --grid 48 --layers 5 --material PMMA --wavelengths 400,550,700 --check-fd --compare-cpu --output results/cuda/small_pmma.json
```

次数8・15層・周期200 nm・高さ500 nm・31波長の検証です。

```powershell
python benchmark_device.py --device cuda:0 --order 8 --grid 512 --layers 15 --material n1.5 --wavelengths 400:700:10 --output results/cuda/n15_M8_L15_31.json
python benchmark_device.py --device cuda:0 --order 8 --grid 512 --layers 15 --material n2.0 --wavelengths 400:700:10 --output results/cuda/n20_M8_L15_31.json
python benchmark_device.py --device cuda:0 --order 8 --grid 512 --layers 15 --material n2.5 --wavelengths 400:700:10 --output results/cuda/n25_M8_L15_31.json
python benchmark_device.py --device cuda:0 --order 8 --grid 512 --layers 15 --material PMMA --wavelengths 400:700:10 --output results/cuda/pmma_M8_L15_31.json
```

速度比較は同じGPU上で1本ずつ実行します。合格時はJSONのstatusがpassedで、cuda_execution_performedがtrueであることを確認します。
550 nm単独測定、チェックポイント再開、詳細な適用範囲は[CUDA実行手順](square_spectral_cuda_validation/README_CUDA_ja.md)を参照してください。

## 反映した範囲

反映したのは、検証コード・既存の検証結果・データCSV・実行手順・配布用ZIPの配置です。
検証済みコードとCSVのSHA256、ZIPのCRCを確認しています。実行場所に関係するMarkdown内のパスだけを20260806側へ変更しました。

正方格子・六方格子のCUDAは、ユーザー提供ログで小規模のCPU照合・有限差分と、次数8・15層・31波長のn2.5/PMMAの合格を確認しました。Cr専用の複素誘電率・吸収・勾配の検証は未実施です。
このため本番RCWAの計算経路や、進行中のdielectric_motheye_n25・cr_motheyeの最適化コードと設定は切り替えていません。
今回、本番Adamとの接続検証用パッケージを追加しました。CPUの接続試験は合格し、CUDAの接続試験は下記の手順で実行できます。Crは対象外です。
共通RCWAへの組込みは適用条件を判定する高速経路として行い、対象外は従来経路を維持する方針です。

CUDA対応版に付属するverified_cpu_smokeはCPU上での動作確認結果です。CUDA実機の合格結果ではありません。


## 六方格子のCUDA検証

[hex_spectral_cuda_validation/README_CUDA_ja.md](hex_spectral_cuda_validation/README_CUDA_ja.md)に、小規模CPU照合・有限差分、次数8・15層・31波長、550 nm単独測定のコマンドを追加しました。

## 本番最適化への接続確認（2026-10-27）

[接続検証の手順とCPU結果](optimization_integration_validation/README_ja.md)を追加しました。既存のbenchmarkのフォルダから、次を実行します。

```bash
cd ../optimization_integration_validation
python validate_connection.py --device cuda:0 --suite quick --output results/cuda/quick_v2
python validate_connection.py --device cuda:0 --suite production --output results/cuda/production_M8_L15_v2
```

quick合格後にproductionを実行します。productionはM8/grid512/15層/31波長・Adam 3更新で、既定でn2.5/PMMAと正方格子/六方格子の4ケースを実行します。材料・格子ごとに別PCへ分けるコマンドも手順に記載しました。配布用は[本番最適化_波長共有_接続検証.zip](本番最適化_波長共有_接続検証.zip)です。

ローカルCPUではM2/grid48の5層4ケースと、保存済み形状を含むn2.5の15層2ケースが合格しました。CUDAの接続確認はこのPCでは未実行です。既存の本番ソース・設定40ファイルはSHA256が一致し、今回の配置による変更はありません。

## ソース整合性チェックの改行対応（v2）

`Frozen production source changed: rcwa_ext/asr.py` で停止する場合に備え、改行・UTF-8 BOMの差だけを許容する照合へ修正しました。配布元asr.pyのCRLF/LF混在を確認済みですが、転送先の実際の相違原因はログだけでは断定できません。内容差は引き続き拒否します。

[修正版フルZIP](本番最適化_波長共有_接続検証_v2.zip) または [差分パッチ](本番最適化_接続検証_改行対応パッチ.zip) を使用してください。[詳しい手順](optimization_integration_validation/README_改行対応.md)のとおり、先に `python source_integrity.py` を実行します。パッチZIP内のファイルは検証フォルダ直下へ展開します。既存本番RCWAは対象外です。

LF+BOMへ変換したコピーで、CPU小規模の4ケースが再び合格しました。実際の内容変更やファイル欠落が拒否されることも確認しています。

## v3統合の完了

本番サイズのCUDA接続ログ（n2.5/PMMAの両格子、M8/15層/31波長/Adam 3更新）の合格後、共通RCWAと最適化へ高速経路を反映しました。新規金属16条件、最適化後の停止、独立したconvergence/timing/spectrumの実行方法は[こちら](README_RCWA高速化と金属最適化_v3.md)です。旧検証パッケージは比較基準として凍結しています。この文書内の未切替という説明は旧版の記録です。
