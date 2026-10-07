# Vallius & Honkanen (2002) 図6・7・9・11の再現

対象論文は T. Vallius and M. Honkanen, *Reformulation of the Fourier modal
method with adaptive spatial resolution: application to multilevel profiles*,
Optics Express **10**, 24–34 (2002),
[DOI: 10.1364/OE.10.000024](https://doi.org/10.1364/OE.10.000024) です。

通常FMMを(a)、論文のparametric representationを(b)として、ゼロ次透過回折効率
`eta0 = T0` を計算します。`T` は全伝搬回折次数の透過率で、図の縦軸とは異なります。
独立した1次元TE/TMソルバーなので、既存の2次元 `rcwa_ext` の選択設定は必要ありません。

保存済み計算では図9はよく一致しました。図6・7には論文曲線との差が残り、
図5の横寸法は推定値です。図11の定量再現は未達です。
詳細は [精度評価](results/ASSESSMENT_ja.md) を参照してください。

追加添付された図5・8・10の材料・周期・高さ・配置と、実装および保存済み条件を
照合し、指定値にずれがないことを確認しました。
[設定照合](results/ATTACHED_SETTINGS_CHECK_ja.md) に表と実装形状を保存しています。

## ファイル

| ファイル | 内容 |
|---|---|
| `solver.py` | 1次元TE/TM FMM・ASR、物理座標への射影、多層S行列 |
| `devices.py` / `torch_backend.py` | CPU/CUDA選択、倍精度PyTorch計算経路 |
| `geometries.py` | 論文の図5・8・10の形状、図別条件 |
| `reproduce.py` | 図6・7・9・11の波長掃引、CSV・PNG・JSON出力 |
| `extract_reference.py` | 添付PDF内の実際のベクトル曲線・マーカーの抽出 |
| `reference/` | 元PDF座標・source hash付きの独立参照データ |
| `validation/test_solver.py` | 解析解と物理的性質による10件の検証 |
| `validation/test_backends.py` / `validation/check_backends.py` | デバイス選択・物理検証、CPUとCUDA計算経路の比較 |
| `results/paper/` | 実際に計算した各図、比較図、条件と精度の記録 |
| `results/ASSESSMENT_ja.md` | 保存済み計算の論文との比較と制約 |

## 実行

既存リポジトリのルート（`rcwa_ext/` と `paper_reproductions/` がある `outputs/`）で
実行します。Python 3.10以上とNumPy、SciPy、Matplotlibが必要です。
既定の `--device auto` は、PyTorchでCUDAが利用できればGPUを選び、
利用できなければ従来のSciPy/CPUを選びます。両方とも複素倍精度 `complex128` です。
判定はCUDAの利用可否によるもので、計算速度のベンチマークによる判定ではありません。

```bash
python -m pip install -r paper_reproductions/vallius2002/requirements.txt
python -m unittest paper_reproductions.vallius2002.validation.test_solver paper_reproductions.vallius2002.validation.test_backends -v
python -m paper_reproductions.vallius2002.reproduce --study smoke --output-dir paper_reproductions/vallius2002/results/smoke
```

論文のモード設定で4図を計算する例:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m paper_reproductions.vallius2002.reproduce \
  --study all --overlay-reference --include-reference-grid \
  --output-dir paper_reproductions/vallius2002/results/paper
```

## GPU・CPUの指定

CUDA GPUにはCUDA対応のPyTorchが追加で必要です。CPU版PyTorchではGPU計算できません。
GPU環境に合うインストールコマンドは [PyTorch公式](https://pytorch.org/get-started/locally/)
でOSとCUDAを選択してください。CPU実行だけならPyTorchは不要です。

まず使用中のPython環境からCUDAが見えるか確認できます。

```bash
python -c "import torch; print('torch=', torch.__version__, 'CUDA build=', torch.version.cuda, 'CUDA available=', torch.cuda.is_available(), 'GPUs=', torch.cuda.device_count())"
```

以下はリポジトリのルートで実行します。

```bash
# 利用可能ならGPU、なければCPU（既定動作）
python -m paper_reproductions.vallius2002.reproduce --study all --device auto --overlay-reference --include-reference-grid --output-dir paper_reproductions/vallius2002/results/auto

# GPUを明示指定。利用できなければエラーで終了
python -m paper_reproductions.vallius2002.reproduce --study all --device cuda --overlay-reference --include-reference-grid --output-dir paper_reproductions/vallius2002/results/gpu

# CPUを明示指定
python -m paper_reproductions.vallius2002.reproduce --study all --device cpu --overlay-reference --include-reference-grid --output-dir paper_reproductions/vallius2002/results/cpu

# GPU実機でCPUとの一致を検証（4図・FMM/ASR・両モード設定、48ケース）
python -m paper_reproductions.vallius2002.validation.check_backends --device cuda --output paper_reproductions/vallius2002/results/diagnostics/cuda_parity.json
```

`--device cuda:0`、`cuda:1` で可視GPUの番号を指定できます。
スケジューラなどが `CUDA_VISIBLE_DEVICES` を設定している場合、その環境内の番号です。
開始時に選択したデバイス・GPU名・計算経路を表示し、各CSVに `device`、`backend`、
metadata JSONに `execution`（PyTorch/CUDAの版、倍精度、選択理由）を保存します。
明示指定したGPUが使えない場合やGPU計算が失敗した場合、CPUへ自動で切り替えません。

GPU経路では形状・Gauss積分の係数を最初にCPUで組み立て、行列を一度転送します。
波長掃引の逆行列・固有値計算・射影・S行列接続は指定デバイスのPyTorch APIを使います。
無損失層の一般化Hermitian問題はCholesky変換を使って解きます。
最終的なスペクトルと描画データはCPUへ戻します。低次数ではGPUの方が速いとは限りません。
境界条件数はCPUのLAPACK推定値とGPUのLU逆行列からの厳密1ノルムで定義が異なるため、
その差をスペクトルの不一致と解釈しないでください。定義も `execution` に記録します。

Python APIでも `PreparedStack(..., device="auto")` / `device="cuda"` / `device="cpu"`
で指定できます。検証用の `device="cpu", backend="torch"` とCLIの
`--device cpu --backend torch` はGPUと同じテンソル計算経路をCPU上で実行します。

今回の開発環境は `PyTorch 2.8.0+cpu` でCUDAが利用できません。
同じテンソル経路をCPUで検証し、4図48ケースの電力値の最大絶対差は約 `1.08e-7` でした。
18件の物理・選択テストが通り、CUDA実機用4件はスキップしました。
[比較記録](results/diagnostics/torch_cpu_parity.json) を保存しています。
GPU実機での実行確認は未実施です。以前の `results/paper/` はSciPy/CPUで計算した結果です。
図6・7の高次数参照 `N=481` もTE/TMで別途比較し、電力値の最大絶対差は
約 `1.20e-11` でした（[高次数比較](results/diagnostics/torch_cpu_high_reference.json)）。
論文との未解決の差の評価は [精度評価](results/ASSESSMENT_ja.md) のままです。

PowerShellでは、先に環境変数を指定します。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python -m paper_reproductions.vallius2002.reproduce --study all --overlay-reference --include-reference-grid
```

`--study fig6`、`fig7`、`fig9`、`fig11` で個別実行できます。
図6・7の240設定のFMM破線も既定で再計算します。これは低次数の系列より計算量が大きく、
既定では51点の別掃引です。省く場合は `--reference-modes 0` を指定します。
smokeではこの高次数系列を省きます。

図9は既定401点、他の図は241点です。`--include-reference-grid` はPDFの元波長点も
計算格子へ追加します。鋭い共鳴を含むため、粗い等間隔格子の結果を補間して参照と比較すると、
実際の計算誤差より大きく見えることがあります。PDFの丸めにより端点が表示範囲から
約1e-5はみ出す場合も、その波長を記録して直接評価します。

## 論文条件と実装上の仮定

全例で周期 `d=1`、正常入射、上下の屈折率 `n1=n3=1`、`G=0.001` です。
材料の屈折率は論文の固定値を使用し、追加の波長分散は導入していません。

| 図 | 偏光 | 波長範囲 | captionの設定 | 構造 |
|---|---|---|---|---|
| 6 | TE | 0.95–1.05 | 5 / 10、FMM参照240 | 4層の金属/空気階段形状 |
| 7 | TM | 0.95–1.05 | 12 / 24、FMM参照240 | 図6と同じ |
| 9 | TM | 1.16–1.18 | 19 / 38 | 屈折率5 / 1.5、各厚さ10の交替二層 |
| 11 | TE | 0.90–0.96 | 7 / 14 | 屈折率5の厚さ1の層中の半径0.25の空気円柱 |

図5では金属の屈折率は `0.1217+3.2966i`、各層厚は0.125です。
**横方向の遷移位置は本文・キャプションに数値がありません。** ベクトル図の位置から、
金属の中心を0.25、下から幅 `[0.3, 13/30, 17/30, 0.7]` と推定しました。
これは論文に明記された条件ではありません。`--metal-widths w1,w2,w3,w4` で変更でき、
実際の遷移位置と複素誘電率を各metadata JSONに保存します。

図11では円中心を `(x,z)=(0.25,0.25),(0.75,0.75)` とし、120等厚スライスの
中央で円幅を評価します。論文は120スライスと記載していますが、スライス内の評価位置は
明記していません。`--slices` で層数を変更できます。

論文の式(20)は `m=-M,...,M` と書き、本文ではモード数と行列サイズの表記が一致しません。
図9の通常FMMを元PDF波長点で検算すると、captionのMを半次数とした
**保持数 `N=2M+1`** が曲線と一致するため、この解釈を既定にしました。
`--mode-convention count` で文字通り `N=M` とする診断も可能です。
ASR内部固有値問題は `3N` 次元、境界はN次元とし、論文の記述通り絶対値が小さい
固有値をN個残します。実際の行列サイズはCSVの `harmonics`、`eigen_dimension` に保存します。

各物理材料区間の計算座標幅を等しく配分します。このu遷移配分も論文では指定されていません。
同一材料をセル端で分断せず、周期的につながる区間を一つの区間として写像します。

## 数式との対応

`D=diag(2*pi*m/d)`、`F=[f]`、`A=[epsilon*f]`、`B=[f/epsilon]` とすると、
ASR固有値問題は論文式(14),(15)です。

```text
TE: (k² A - D F⁻¹ D) E = gamma² F E
TM: (k² F - D A⁻¹ D) H = gamma² B H
```

通常FMMでは `f=1` とし、TEは `k²[epsilon]-D²`、TMはLiのFourier factorizationを
用いる `inv([1/epsilon])*(k²I-D*inv([epsilon])*D)` に還元します。
実誘電率の層では一般化Hermitian固有値問題として解きます。

区間 `[u0,u1]` から `[x0,x1]` への写像は式(16)–(19)です。

```text
du=u1-u0; s=(x1-x0)/du
x(u)=x0+s*(u-u0)+(G-s)*du/(2*pi)*sin(2*pi*(u-u0)/du)
f(u)=s+(G-s)*cos(2*pi*(u-u0)/du)
```

式(25)の `K[p,m]=integral f exp(-i*alpha_p*x+i*alpha_m*u) du/d` により、
各層を同一の物理x空間Fourier基底へ射影します。TEの境界対は `E, gamma*E`、
TMは `H, gamma*(H/epsilon)` です。TMの `Q=H/epsilon` はu空間の連続関数を
区間積分して射影し、物理x空間で改めて不連続係数を掛けません。
`--q-projection laurent` では有限のu空間Laurent積を先に作る比較もできます。
論文はQの有限打ち切り方法を別に指定していません。

写像積分は材料界面ごとのGauss–Legendre積分で、入力quadrature値と
`2*(内部次元+保持数)` の大きい方以上の点数を自動使用します。
`--quadrature` は最低値です。層接続はRedheffer S行列で行い、増大する
エバネッセント波の指数を作りません。`lambda=1` の厳密なRayleighカットオフは
相対1e-12上側の極限で評価し、`evaluation_wavelength` に記録します。

## 検証と参照の意味

10件のテストは、独立Fabry–Perot解析解、上下媒質の電力換算、受動吸収、
通常FMMの全次数電力保存、周期移動、単位スケーリング、ASR恒等写像、
写像の連続性・積分・界面勾配、周期セル端、ASRの次数収束を確認します。
低次数ASRでは境界の射影打ち切り誤差があり、無損失でも `A=1-R-T` がゼロから
ずれる場合があります。図9・11のAは物理吸収ではなく残差です。

`reference/` は著者の元数値データではなく、添付PDFの実際のベクトル曲線と
菱形中心の抽出値です。計算値を参照データへ加工していません。出所のhash、
元座標、軸較正、図5の測定値を `reference/metadata.json` に保存しています。
PDF抽出をやり直す場合だけpypdf/pdfplumberが追加で必要です。

```bash
python -m pip install pypdf pdfplumber
python -m paper_reproductions.vallius2002.extract_reference --pdf "/path/to/Reformulation of the Fourier modal method.pdf"
```

CSV、曲線重ね合わせPNG、JSONの誤差指標を組にして確認してください。
保存済みの精度と未解決の差は `results/ASSESSMENT_ja.md` に記載しています。

`figN_paper_sampling.png` は元PDFと同じ波長で計算した比較図です。
`figN.png` と `figN_comparison.png` は細かい計算格子のスペクトルで、
PDFの粗い点数では捉えていない極細の共鳴も表示します。

図11の独立ソルバーとの確認は、既存torch/torcwa環境がある場合に再実行できます。

```bash
python -m paper_reproductions.vallius2002.validation.compare_torcwa
```

同検査はtorcwaの固有値・界面・多層接続・電力換算を使用し、材料のFourier係数のみ
独立した区間積分で与えます。`--grid 2048` は通常のtorcwa画素化でも確認する設定です。
