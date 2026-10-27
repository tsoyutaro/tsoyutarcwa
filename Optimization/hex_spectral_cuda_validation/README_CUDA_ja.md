# 六方格子：D6専用行列構築と波長共有のCUDA検証

正方格子のCUDA検証と同じ形式で、六方格子の3経路を比較する。入口はこのフォルダの `benchmark_device.py`。
**こちらの作成環境でCUDA実機テストは未実施**。CPUでの動作・数値照合・有限差分の結果を付属する。

| 経路 | 内容 |
|---|---|
| reference | 凍結した元のAutoRCWAを直接呼び出す。六方格子の元のD6処理、矩形側で生成する入力、全体行列の構築を含む。 |
| block | D6専用版。native starの入力、multiplicity空間の逆行列、E1の対の2行によるP/Q構築を使用。 |
| shared | 上記D6専用版に波長間共有を追加。一定屈折率はP/Q係数、PMMA分散は形状Fourier係数と法線射影のみ共有する。 |

referenceとblockは波長ごとに再構築する。sharedも全波長の固有値・伝播・散乱・随伴計算を実行する。
共有部分から半径へのVJPは最後にまとめて1回計算する。PMMAの円柱と基板には各波長のn(lambda)を使用し、材料の逆行列とP/Qは各波長で再計算する。
半径は規格化0.10～0.39。法線入射、x偏光、中心円柱、NVM、非磁性、fields=none、非吸収性材料が対象。
次数8ではscalar starが217、source rowの固有値問題が73次元。Crは今回の対象外。

## 実行するフォルダ

20260806側に配置済み：

```text
Optimization/hex_spectral_cuda_validation/
```

Linux等で正方格子の検証フォルダから移動する場合（両フォルダを同じ階層に置いた場合）：

```bash
cd ../hex_spectral_cuda_validation
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available()"
```

WindowsのPowerShellでは：

```powershell
Set-Location -LiteralPath 'C:/Users/tsoyu/Documents/Codex/2026-08-06/files-mentioned-by-the-user-extensions/outputs/Optimization/hex_spectral_cuda_validation'
```

CUDA対応PyTorchの環境を使用する。旧spectral_reuse_validationのbenchmark.py/benchmark_pmma.pyはCPU検証用のまま維持し、このフォルダにある新しい入口を使う。

## 最初の小規模検証

次数2・5層・400/550/700 nm。従来CUDA版、D6専用CUDA版、共有CUDA版の一致に加え、独立したCPU参照と有限差分を確認する。

```bash
python benchmark_device.py --device cuda:0 --order 2 --grid 48 --layers 5 --material n2.5 --wavelengths 400,550,700 --check-fd --compare-cpu --output results/cuda/hex_small_n25.json
python benchmark_device.py --device cuda:0 --order 2 --grid 48 --layers 5 --material PMMA --wavelengths 400,550,700 --check-fd --compare-cpu --output results/cuda/hex_small_pmma.json
```

## 次数8・15層・31波長

周期200 nm、高さ500 nm、NVM grid 512×512。

```bash
python benchmark_device.py --device cuda:0 --order 8 --grid 512 --layers 15 --material n2.5 --wavelengths 400:700:10 --output results/cuda/hex_n25_M8_L15_31.json
python benchmark_device.py --device cuda:0 --order 8 --grid 512 --layers 15 --material PMMA --wavelengths 400:700:10 --output results/cuda/hex_pmma_M8_L15_31.json
```

一定屈折率1.5・2.0も同じ入口で選択できる。

```bash
python benchmark_device.py --device cuda:0 --order 8 --grid 512 --layers 15 --material n1.5 --wavelengths 400:700:10 --output results/cuda/hex_n15_M8_L15_31.json
python benchmark_device.py --device cuda:0 --order 8 --grid 512 --layers 15 --material n2.0 --wavelengths 400:700:10 --output results/cuda/hex_n20_M8_L15_31.json
```

## 550 nm単独の時間と有限差分

各経路3回測定。PMMA単独波長はその波長の屈折率であり、分散の確認自体は複数波長のテストで行う。

```bash
python benchmark_device.py --device cuda:0 --order 8 --grid 512 --layers 15 --material n2.5 --wavelengths 550 --repeats 3 --check-fd --output results/cuda/hex_n25_M8_L15_single.json
python benchmark_device.py --device cuda:0 --order 8 --grid 512 --layers 15 --material PMMA --wavelengths 550 --repeats 3 --check-fd --output results/cuda/hex_pmma_M8_L15_single.json
```

## 判定と時間

- 合格にはJSONの `status: "passed"` と `cuda_execution_performed: true` を確認する。latticeはhexagonal、symmetryはd6となる。
- native入力と元の入力、P/Q/PQ、R/T、全半径勾配、エネルギー保存と値域を確認する。従来版とD6専用版は波長別の全層勾配も照合する。
- --check-fd指定時は有限差分、--compare-cpu指定時は独立したCPU参照との照合も合格条件になる。
- PMMAの複数波長では、屈折率をn(550 nm)へ誤って固定した負例が一致しないことも判定する。
- 元のCPU検証と同じ許容差を使い、GPU用に緩和していない。R/T絶対差1e-8、エネルギー1e-6、勾配atol=1e-7/rtol=2e-5、有限差分atol=1e-6/rtol=3e-4。native入力の相対差2e-12、P/Q/PQの相対差2e-9。
- GPU時間は同期を挟んだ経過時間。初期化用VJP、共通基底、1波長のforward/backwardウォームアップを除く。CUDAのCPU待ち時間を中断とは扱わない。
- GPU名・メモリ・CUDA/PyTorch/Python・コードとCSVのハッシュを記録する。GPUメモリのピークも保存する。
- `block_speedup` は元実装に対するD6専用版の比。`shared_speedup_with_port_preparation` は元実装に対するD6＋共有の比。共有による追加倍率はblock時間を「shared時間＋port準備時間」で割って求める。
- 同じGPUでは1本ずつ実行する。共有の1波長平均は全波長時間を波長数で割った値であり、単独1波長の時間と区別する。

従来版・D6専用版は波長ごとにチェックポイントを保存する。共有版は全波長と形状VJPをまとめた単位で保存する。
同じPC/GPU・同じ実行環境・同じソースで、同じコマンドを再実行すると再開する。測り直す場合は新しい--outputを指定する。

## D6行列構築だけを独立に検証する場合

既存のd6_optimized/validate.pyはCUDA対応。このテストはD6専用化・1波長の光学応答と勾配の確認であり、波長共有は対象に含まれない。

```bash
cd ../d6_optimized
python validate.py --suite quick --device cuda:0 --orders 8 --grid 512 --layers 15 --materials n2.5 PMMA --lattices hexagonal --benchmark-repeats 3 --output validation_results/hex_M8_L15_gpu
```

## 今回の確認範囲

verified_cpu_smokeに、CPU上で実施した次数2の4材料、および次数4のn2.5/PMMAの6ケースを収録する。すべて5層・3波長で有限差分まで合格した。n2.5の2回測定とチェックポイント再開も確認した。これらはCUDA実機の合格結果ではない。

d6_nvm.py、D6用のadapterとlegacy参照、元の六方格子CPUコードは凍結コピーを使用する。frozen_source_origins.jsonに出典とSHA256を記録した。
benchmark_square.pyは入力検査・PMMA補間・比較処理などの共通utilityの供給元であり、このドライバーの格子はhex_core.pyで六方格子に固定される。

このテスト合格は同じ次数で元実装と一致することを示す。次数16での収束性、実際の最適化パラメータへの勾配、Adam/checkpoint接続、Crの複素誘電率と吸収の確認は別である。本番の最適化・共通RCWAの計算経路は今回変更していない。

