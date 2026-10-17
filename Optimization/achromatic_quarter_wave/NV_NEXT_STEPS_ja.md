# pull後のNV結果の再評価と次の計算

確認したHEADは`4ac9c8a`。直近の追加はCUDA実行結果で、NV計算コードに変更はなかった。
実行時のSHA256と現コードを照合し、`rcwa_ext`の10ファイルの差はCRLF/LFの改行差だけと確認した。
元の実験メタデータ・キャッシュは変更していない。

## 保存済みデータを波長別に読むと

最新CUDA実験のQ=1024・60層におけるM=9→11の比較：

| 断面 | 波長 | 位相差の変化 | 各偏光の透過位相の最大変化 | 透過率・反射率の最大変化 |
|---|---:|---:|---:|---:|
| 長方形 | 400 nm | 0.2048° | 0.2651° | 0.0616ポイント |
| 長方形 | 550 nm | 0.4297° | 0.5111° | 0.0966ポイント |
| 長方形 | 700 nm | 0.4354° | 0.5008° | 0.0424ポイント |
| 楕円 | 400 nm | 1.1302° | 0.9838° | 0.6427ポイント |
| 楕円 | 550 nm | 0.3118° | 0.2515° | 0.0117ポイント |
| 楕円 | 700 nm | 0.2097° | 0.1743° | 0.0157ポイント |

「ポイント」は透過率等を%表示したときの絶対差。たとえばパワー差0.001は0.1ポイント。
全波長でMの0.1°条件は未達。重点的に診断するのは長方形550・700 nmと、楕円400 nm。
楕円の400 nmの主値位相差は約−170°で、±180°の折り返しを補正して差を計算している。
1.13°の差は単なる360°の表示上の折り返しではない。

Q=512→1024の比較、および30→60層の比較は現在の判定基準内。
ただしQ・zの誤差がゼロになったことを意味せず、いまの形状と3波長の隣接段階の確認に限られる。

## 別方式との照合

同じ形状、30層、Q=256で長方形ASR-FRをCPU・complex128で再計算した。

| 波長 | ASR M=5 | ASR M=7 | NV M=11・Q=1024 |
|---|---:|---:|---:|
| 550 nm | 119.5354° | 119.5530° | 119.9753° |
| 700 nm | 94.6379° | 94.6880° | 95.0327° |

ASR自身にも有限次数誤差はあるが、この比較ではNVの次数依存性の追加確認が優先。
ASR値を確定した真値と扱うことはしない。

楕円400 nmのstandard-raster法も同条件で照合したが、M=5→7で位相差が
143.6643°→128.3877°と大きく変化した。こちらも未収束で、NVを判定する基準値にはできない。
これらの結果だけでNVに実装不良があるとも、正しい収束値が得られたとも判断しない。

## 次はMだけを追加する

次の3点を診断する：

1. 長方形550 nm、M=11→13、Q=1024・30層。
2. 長方形700 nm、同じ条件。
3. 楕円400 nm、同じ条件。

今回の30→60層の位相差変化は最大約0.051°なので、次数の診断は30層で進める。
最終検証では60層に戻し、残りの波長も確認する。この3点だけで全帯域の収束を認定しない。

`nv_order_refine.py`を追加した。既存のNV数式・演算コードは変えていない。
参照実験のモデル・形状を読み込み、ソースが同じことを確認してから、同じM/Q/z/波長のキャッシュを再利用する。
CRLF/LFだけの違いは許容するが、数式等のソース変更は拒否する。
Qとzを固定した実験としてMだけを判定するため、固定した軸に`insufficient_levels`を付けて総合失敗にすることはない。

リポジトリのルートで、追加スクリプトを計算環境へ反映後に実行：

```bash
python Optimization/achromatic_quarter_wave/nv_order_refine.py --reference Optimization/achromatic_quarter_wave/results/nv_convergence_M7_9_11_Q512_1024_cuda_run01 --shape rectangle --orders 11 13 --grid 1024 --layers 30 --wavelengths 550 700 --device cuda --output Optimization/achromatic_quarter_wave/results/nv_rectangle_M11_13_focus
```

```bash
python Optimization/achromatic_quarter_wave/nv_order_refine.py --reference Optimization/achromatic_quarter_wave/results/nv_convergence_M7_9_11_Q512_1024_cuda_run01 --shape ellipse --orders 11 13 --grid 1024 --layers 30 --wavelengths 400 --device cuda --output Optimization/achromatic_quarter_wave/results/nv_ellipse_M11_13_focus
```

両方を合わせて既存M=11の3点を再利用し、新規RCWA評価はM=13の3点。
`reference_cache`と表示される点は再計算しない。
新しい保存先にも成功した結果を保存するので、同じコマンドで途中から再開できる。

出力は`order_spectra.csv`、`order_comparisons.csv`、`order_summary.json`、`refinement_metadata.json`。
`order_summary.json`の`order_within_tolerances`は、指定した波長・固定Q/zでの最後のM比較だけの判定。

まだ0.1°を超える場合は、NVのM依存の推移と別方式の値を合わせて判断する。
楕円400 nmで不規則な変化が続く場合は、近傍波長の応答と異なる法線場の拡張方法を比較し、
単に全条件を高次数で再計算し続ける進め方は避ける。

## 補助スクリプトの確認

- GPUの既存6結果をGPU初期化なしで読み、前回のM=9→11の波長別差を再現した。
- CPUで新規M=2を計算し、保存済みM=3を再利用する経路を確認した。
- 参照ソースのハッシュを意図的に変更したテストで再利用拒否を確認した。
- CUDAでのM=13新規計算はこの環境では未実行。実行環境はCPUのみ。

ZIPは作成していない。

## 2026-10-18：M11→13の追加結果と次の比較

以下はユーザーが提示したCUDA実行結果の記録。M13のキャッシュ本体はこのローカル環境にはまだない。

| 断面・波長 | M11の位相差 | M13の位相差 | 位相差の変化 | 各偏光の透過位相の最大変化 | パワーの最大変化 |
|---|---:|---:|---:|---:|---:|
| 長方形550 nm | 119.975304° | 119.512206° | 0.463098° | 0.396880° | 0.078934ポイント |
| 長方形700 nm | 95.032748° | 94.663837° | 0.368911° | 0.331161° | 0.026646ポイント |
| 楕円400 nm | −170.342027° | −170.333016° | 0.009011° | 0.154072° | 0.235339ポイント |

楕円400 nmは偏光間の位相差だけなら0.1°の基準内だが、各偏光の透過位相とパワー変化は基準外。
ほぼ共通する透過位相の変化が位相差では相殺されていると解釈できる。
この一対での位相差の一致は絶対誤差が0.009°という保証にはならず、低反射・偏光変換特性を含む収束完了とはしない。
また、約−170°は代表形状の応答であり、90°を狙った最適化結果ではない。

次は同じQ=1024・30層で、長方形550/700 nm、楕円400 nmのM13→15を比較する。
`nv_order_refine.py`に`--reuse`を追加した。過去の次数追加実験について材料、形状、Q、層数、デバイス、
NVソースのハッシュと精度を照合してからキャッシュを再利用する。
NVの演算や既定の収束基準は変更していない。更新版を計算環境へ反映後、リポジトリのルートで実行：

```bash
python Optimization/achromatic_quarter_wave/nv_order_refine.py --reference Optimization/achromatic_quarter_wave/results/nv_convergence_M7_9_11_Q512_1024_cuda_run01 --reuse Optimization/achromatic_quarter_wave/results/nv_rectangle_M11_13_focus --shape rectangle --orders 13 15 --grid 1024 --layers 30 --wavelengths 550 700 --device cuda --output Optimization/achromatic_quarter_wave/results/nv_rectangle_M13_15_focus
```

```bash
python Optimization/achromatic_quarter_wave/nv_order_refine.py --reference Optimization/achromatic_quarter_wave/results/nv_convergence_M7_9_11_Q512_1024_cuda_run01 --reuse Optimization/achromatic_quarter_wave/results/nv_ellipse_M11_13_focus --shape ellipse --orders 13 15 --grid 1024 --layers 30 --wavelengths 400 --device cuda --output Optimization/achromatic_quarter_wave/results/nv_ellipse_M13_15_focus
```

M13は`refinement_cache`となり、新規計算はM15の3点だけ。
引数・ドライバーが変わるので、M11→13の出力フォルダはそのまま保存し、新しい出力先を使う。
この3点を通過しても、残りの波長、最終次数でのQ/z、最適化後の形状、400–700 nmの波長走査は別途確認する。
不規則な次数依存性や停滞が続く場合は、法線場の拡張や別方式との照合に戻る。
