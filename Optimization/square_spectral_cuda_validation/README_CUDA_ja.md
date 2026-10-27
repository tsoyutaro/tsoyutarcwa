# 正方格子のCUDA対応・波長共有検証

前回CPUで検証した光学計算を凍結コピーし、`benchmark_device.py` からCPU/CUDAへ配置して呼び出す。元のCPU検証フォルダ、本番RCWA、進行中の最適化は変更していない。**CUDA実機での検証は未実施**。付属する実測結果はCPU上でのドライバー動作確認であり、GPU合格を意味しない。

## CUDA対応の内容

- 半径、有限差分の方向、波数、基底、境界行列と勾配計算を指定デバイスへ配置する。complex128を使用する。
- 全体時間・行列構築時間は、計測の前後で `torch.cuda.synchronize(device)` を実行した経過時間。CPU起動・制御処理も含む。凍結コード内のCPU向け区間時計はCUDAの結果から除外する。
- 計測前に1波長のforward/backwardをウォームアップし、初回VJP初期化・共通基底とともに時間から除く。
- CUDAではCPUがGPUを待つ時間も正常な実行時間である。CPU専用版の「経過時間と実CPU時間の差による中断判定」は適用しない。
- GPU名・容量・CUDAランタイム・PyTorch/Python・デバイス番号・GPU UUID（取得できる場合）、GPUピークメモリ、ソースとCSVのSHA256を保存する。
- 従来版とブロック版は波長ごとにチェックポイントを保存する。共有版は全波長の逆伝播と形状VJPを1回の単位で保存する。
- `--compare-cpu` は独立した従来CPU計算と、各デバイス経路のR/T・半径勾配を照合する。CPU照合はGPUの速度計測から除く。
- 半径勾配の平均だけでなく、従来版とブロック版の全波長の勾配ベクトルも照合する。

対象は正方格子・中心円柱・法線入射・x偏光・NVM・fields=none・非吸収性材料。一定屈折率ではP/Q係数を共有し、PMMAでは形状Fourier係数と法線射影だけを共有する。PMMAの円柱・基板には波長ごとの同じn(lambda)を適用し、材料の逆行列とP/Qは各波長で再計算する。

## 環境確認

CUDA対応PyTorchの入ったPython環境で実行する。以前のCPU版Pythonを固定指定したコマンドは、そのままGPU環境を選ぶことにはならない。必要な依存はrequirements.txtを参照する。インストールはこの検証では変更していない。

```powershell
Set-Location -LiteralPath 'C:/Users/tsoyu/Documents/Codex/2026-08-06/files-mentioned-by-the-user-extensions/outputs/Optimization/square_spectral_cuda_validation'
python -c "import torch; print(torch.__version__, torch.version.cuda); print('CUDA:', torch.cuda.is_available()); assert torch.cuda.is_available()"
```

この作業で使用したPythonはPyTorch 2.8.0+cpuであり、上のCUDA確認は通らない。利用者のCUDA環境で以下を実行する。

## 先に実行する小規模検証

CPUとの照合と有限差分を含む。--compare-cpuは小規模での使用を推奨する。

```powershell
python benchmark_device.py --device cuda:0 --order 2 --grid 48 --layers 5 --material n2.5 --wavelengths 400,550,700 --check-fd --compare-cpu --output results/cuda/small_n25.json
python benchmark_device.py --device cuda:0 --order 2 --grid 48 --layers 5 --material PMMA --wavelengths 400,550,700 --check-fd --compare-cpu --output results/cuda/small_pmma.json
```

## 次数8・15層・31波長

周期200 nm、高さ500 nm。従来版・ブロック構築版・波長共有版を同じGPUで比較する。

```powershell
python benchmark_device.py --device cuda:0 --order 8 --grid 512 --layers 15 --material n1.5 --wavelengths 400:700:10 --output results/cuda/n15_M8_L15_31.json
python benchmark_device.py --device cuda:0 --order 8 --grid 512 --layers 15 --material n2.0 --wavelengths 400:700:10 --output results/cuda/n20_M8_L15_31.json
python benchmark_device.py --device cuda:0 --order 8 --grid 512 --layers 15 --material n2.5 --wavelengths 400:700:10 --output results/cuda/n25_M8_L15_31.json
python benchmark_device.py --device cuda:0 --order 8 --grid 512 --layers 15 --material PMMA --wavelengths 400:700:10 --output results/cuda/pmma_M8_L15_31.json
```

## 550 nmの単独実行

各経路3回の中央値と有限差分を確認する。

```powershell
python benchmark_device.py --device cuda:0 --order 8 --grid 512 --layers 15 --material n2.5 --wavelengths 550 --repeats 3 --check-fd --output results/cuda/n25_M8_L15_single.json
python benchmark_device.py --device cuda:0 --order 8 --grid 512 --layers 15 --material PMMA --wavelengths 550 --repeats 3 --check-fd --output results/cuda/pmma_M8_L15_single.json
```

速度比較は同じGPUで1本ずつ行う。同じ出力先の再実行はチェックポイントを再利用する。測り直す場合は新しい--outputを使う。同じPC・GPU・実行環境で再開し、環境が変わった場合も出力先を変更する。共有版の準備込み時間と、31波長の時間を割った平均、単独1波長の時間を区別する。以前の6.92倍・4.69倍はCPU実測であり、CUDAの倍率ではない。

合格時はJSONの `status: "passed"` に加え、`cuda_execution_performed: true`、`runtime.device: "cuda:0"` を確認する。--compare-cpu指定時はCPU照合も合格条件に入る。比較閾値はCPU版と同じ（R/T絶対差1e-8、エネルギー1e-6、勾配atol=1e-7/rtol=2e-5、有限差分atol=1e-6/rtol=3e-4）で、GPU用に緩和していない。

## 本番RCWAへの適用範囲

今回のテスト合格は、対象の正方格子の数値計算の一致を示す。全RCWA・最適化フロー全体への一括適用の判定にはならない。

| 対象 | 合格後の判断 |
|---|---|
| 正方格子・一定屈折率2.5のNVM | 対象条件の高速経路として組込み候補。実際の最良形状と半径範囲、softmax logitsへの勾配、短いAdam運転、checkpoint ON/OFF、次数16の監査を元の経路と照合してから新規計算で有効化する。 |
| 六方格子・一定屈折率2.5 | 正方格子の合格では確認できない。D6基底と六方格子の波長共有についてCUDAで別途検証する。現行n2.5設定には正方・六方の両方が含まれる。 |
| Crの六方格子NVM | 今回の対象外。Johnson.csvの複素epsilon(lambda)と半無限Cr基板で、R・基板流入パワー・吸収・半径およびlogits勾配をCr用の元実装と比較する。 |
| 斜入射、非対称形状、ASR、場分布の出力など | この高速経路の適用条件に含まれない。従来経路を保持する。 |

現行Crコードは、epsilon=(n+ik)^2を波長で補間し、円柱と半無限基板へ適用する。基板上端への流入パワーP_subは遠方透過率ではない。コードではA_total=1-R、A_moth=1-R-P_subとして表面のパワー収支から吸収を推定している。これらは独立した体積損失積分ではない。Cr検証に非吸収系のR+T=1判定を流用できない。

Crでも形状部分を共有する考え方は検討できるが、今回のコードはCrを受け付けず、負の実部を持つ複素誘電率・吸収・金属の条件数を検証していない。PMMA合格をCr合格へ読み替えない。

共通RCWAライブラリには、適用条件を判定するオプションの高速経路として追加し、対象外条件では従来実装へ戻す構成が適切。進行中の最適化のソースを途中で差し替えるのではなく、別出力先の比較運転を経て新規実験に適用する。この作業では本番コードの反映は行っていない。

## 今回の確認結果

`verified_cpu_smoke/` は、この新ドライバーをCPUで動作確認した4材料・次数2・5層・3波長の結果。全てR/T・半径勾配・有限差分が合格し、n2.5の2回測定とPMMAのチェックポイント再開も確認した。GPUの同期・メモリ・CPU対GPU照合を含むCUDA実機経路は未実施。

`frozen_cpu_source_sha256.json` は、元の検証済みCPUコード・CSVの凍結コピーのハッシュ。元のCPU検証結果は元フォルダに保存したままである。

