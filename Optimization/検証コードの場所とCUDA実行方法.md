# 波長共有・CUDA検証コードの場所と実行方法

検証一式を、この20260806側のOptimizationフォルダへ配置しました。

| フォルダ | 内容 |
|---|---|
| [square_spectral_cuda_validation](square_spectral_cuda_validation/README_CUDA_ja.md) | 正方格子のCUDA対応ドライバー。一定屈折率1.5/2.0/2.5とPMMA分散の波長共有を検証。 |
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

CUDA実機での検証とCr専用の複素誘電率・吸収・勾配の検証は未完了です。
このため本番RCWAの計算経路や、進行中のdielectric_motheye_n25・cr_motheyeの最適化コードと設定は切り替えていません。
正方格子の合格を六方格子・Crの合格とみなすことはできません。
共通RCWAへの組込みは適用条件を判定する高速経路として行い、対象外は従来経路を維持する方針です。

CUDA対応版に付属するverified_cpu_smokeはCPU上での動作確認結果です。CUDA実機の合格結果ではありません。

