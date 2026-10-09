**既存の一次元金格子：Cartesian Li と Vallius 1D ASR の次数収束比較**

`gold_grating_1d` と `pmma_gold_grating_1d` の既存のLi実装を呼び、同じ形状・分散材料・波長・偏光・基板で `paper_reproductions.vallius2002` のASRを計算します。共通の `rcwa_ext` と既存studyの物理モデルは変更しません。元のリポジトリルートから実行してください。

**両studyを計算する**

```bash
python -m studies.asr_1d_comparison.compare --study all --device auto --output-root studies/asr_1d_comparison/results/galerkin
```

CUDAが利用可能ならGPU、利用できなければCPUで計算します。`--device cuda` で明示したGPUが利用できない場合はエラーになり、CPUへ切り替えません。ASRのCPUは既定でSciPyです。`--device cpu --backend torch` を指定するとGPUと同じテンソル計算経路をCPUで使えます。既存Li側はCPU/GPUともPyTorchです。

初期値は `M=2,4,8,12,16,24`、`N=2M+1`、ASR内部次元 `4N`、G=0.001。波長は既存configの400,450,500,550,600,650,700 nm、垂直入射、TE/TMです。金は420層、PMMAは300分割＋金キャップ1層を固定します。保存済みの探索で使われた層数を採用していますが、この次数比較で層数の収束を再判定するわけではありません。

横軸MはFourier展開の打切り次数です。エバネッセント成分を含めて−M～+MのN成分を保持します。TEは電場が溝方向、TMは磁場が溝方向です。

金のLi側は既存の解析的Fourier係数の経路にそろえます。横方向のサンプリングgridは使いません。PMMA側は従来から解析的な区間積分です。各studyの材料読み込み・補間関数をそのまま使用し、LiとASRに同じ複素誘電率を渡します。金studyの線形epsilon補間と、PMMA studyの線形n,k補間後の二乗は、それぞれ元studyに合わせます。元の材料CSVは変更しません。

ASRのTM境界場は、既定で `--q-projection galerkin` を使います。内部固有モードの選択は従来どおり |γ²| の小さいN個です。物理磁場W=KHはそのまま、電場側の境界場Vを
`W^H V = H^H [f/epsilon] H Gamma` から構成し、縮約された層内系と境界の電力内積を一致させます。
SciPyとPyTorch/CUDAの両経路に実装しました。TE・既存Li・通常FMM・界面の連続条件・Redheffer接続は従来と同じです。

従来の `--q-projection direct` は磁場と相方の電場を独立にFourier投影しており、有限打切りでは電力内積が一致しませんでした。PMMA300分割、4N、M=16、600/650/700 nmの負吸収はこの段階で生じた人工的な増幅が主因でした。内部倍率を5Nへ増やしても解消せず、SciPyでも同じ問題がありました。galerkinはこの投影を改良した方法であり、元論文の有限投影と同一とは扱いません。論文再現CLIではdirectを既定値として維持しています。

受動性の回復は精度・収束の保証ではありません。内部倍率4、G、保持次数、層数、積分点による変化と有限Li参照との差を引き続き確認してください。旧 `gpu_4N` などの保存値はdirectの結果です。修正後はソースと投影設定のハッシュが変わるため、新しい出力先を使います。

同じ条件をGPUで再計算：

```bash
python -m studies.asr_1d_comparison.compare --study all --device cuda --q-projection galerkin --output-root studies/asr_1d_comparison/results/gpu_galerkin
```

3N/4Nを並べる場合：

```bash
python -m studies.asr_1d_comparison.compare --study all --device cuda \
  --asr-ratios 3,4 --orders 2,4,8,12,16,24,32,40 \
  --reference-order 80 --reference-check-order 72 \
  --output-root studies/asr_1d_comparison/results/gpu_3N_4N
```

M=80まで拡張する例（参照は掃引の最大次数より上に設定）：

```bash
python -m studies.asr_1d_comparison.compare --study all --device cuda \
  --orders 4,8,12,16,24,32,40,48,56,64,72,80 --asr-ratios 4 \
  --reference-order 96 --reference-check-order 88 \
  --output-root studies/asr_1d_comparison/results/gpu_M80
```

同じ保持Nで比較しています。ASRはrN個の内部固有モードからN個を選ぶため、行列サイズ・計算時間まで同じにはなりません。大きいM・内部倍率・層数では時間とメモリが増えます。時間のCSVは各波長のTE/TM両方を含む時間で、偏光別行に重複して記録しているため、単純に全行を合計しないでください。

今回のCPU結果は両studyを別プロセスで同時に実行したものです。記録した時間には他の処理・待機の影響が含まれます。この図からLiとASRの速度比較は行いません。必要な依存関係は同フォルダの `requirements.txt` に示します。CUDA用PyTorchは使用するGPU環境に合ったビルドを使ってください。

**studyごとの入口**

```bash
python studies/gold_grating_1d/compare_asr.py --device cuda --slices 420 --output-root studies/asr_1d_comparison/results/gpu_galerkin
python studies/pmma_gold_grating_1d/compare_asr.py --device cuda --slices 300 --output-root studies/asr_1d_comparison/results/gpu_galerkin
```

特定波長、形状config、G、積分点数、内部倍率も変更できます。例：

```bash
python studies/pmma_gold_grating_1d/compare_asr.py --device cuda \
  --wavelengths 400,550,700 --asr-ratios 3,4,5 --G 0.001 \
  --quadrature 512 --diagnostics \
  --output-root studies/asr_1d_comparison/results/internal_check
```

`--quadrature` は各材料領域の積分点数の下限です。実際の点数はASR本体の `max(48, 2*(内部次元+保持N), 指定下限)` によって増え、CSVにも保存されます。`--diagnostics` はASRの界面・S行列接続の条件数を追加します。条件数だけでは収束を保証しません。

**出力と確認**

既定の出力先は各studyの `results/li_vs_asr_1d/` です。`--output-root` を指定した場合は、その下のstudy名のフォルダに保存します。

両studyを実行すると `convergence_overview.png/.svg` と `overview.csv` も作成します。各偏光で全指標・全波長の有限Li参照からの最大差を4パネルで比較します。既定では `studies/asr_1d_comparison/results/li_vs_asr_1d/`、`--output-root` 指定時はそのフォルダ直下です。

- `values.png/.svg`：各波長のR・基板電力・吸収対M。TE/TMの6パネル。
- `reference_difference.png/.svg`：有限次数Li参照との差の絶対値を波長間で最大化。縦軸はパーセントポイントの対数目盛。
- `adjacent_difference.png/.svg`：隣接するM間の絶対差の最大値。絶対誤差とは区別します。
- `cases.csv`：物理量、M/N/内部次元、層数、積分点、実行環境、有限参照との差、受動性判定。
- `plan.json`：形状、材料値、材料・数値ソースのハッシュ、全計算条件。
- `checkpoint.json`：完了した各波長ケースと実行環境。各ケース終了時に保存。
- `report.json` / `RESULTS_ja.md`：隣接差、有限参照の確認、対象範囲。
- `invalid_cases.json`：受動性の範囲を破った値と計算失敗。

金基板の `P_sub` は金に入る界面電力で、遠方透過 `T_far` は0です。全ての界面フラックス成分を加算するため、金側の高次成分を空気中のエバネッセント次数として捨てません。PMMA基板のTは、半無限・無損失PMMAへ入る透過電力です。両studyの吸収はポート電力の残差であり、内部場から独立に求めた吸収積分ではありません。

物理的な範囲を破った計算は赤い×にして有効な曲線から除外し、元の値はCSVに残します。有限参照との差の最大値は、その偏光で全波長が受動性範囲を満たすMだけを結びます。受動性に整合することは収束の十分条件ではありません。

参照は既定でM=80のLiで、M=72との変化を確認します。この有限参照が厳密解だとは扱いません。最後の二つの隣接M区間が、全波長・TE/TM・全指標で0.005以内かを記録します。これは固定した層数と試した波長での判定です。有限参照との差も図から確認してください。

`report.json` には最終Mの有限参照からの最大差も保存します。隣接差が小さくても参照値から離れている場合は、両者が許容差内で整合したとは判定しません。0.005は割合で、図では0.5パーセントポイントです。

計算せずに条件だけ確認：

```bash
python -m studies.asr_1d_comparison.compare --study all --prepare-only --output-root studies/asr_1d_comparison/results/galerkin
```

同じ引数で再実行すると完了済みケースを再利用します。M・波長・内部倍率の範囲を広げることもできます。材料・形状・数値ソース・G・積分点を変更した場合は混在を拒否するので、新しい `--output-root` を指定してください。CPU/GPUを再開時に変更した場合の実行環境はケースごとに残ります。

保存した範囲と同じ引数で再作図（計算はしない）：

```bash
python -m studies.asr_1d_comparison.compare --study all --report-only --output-root studies/asr_1d_comparison/results/galerkin
```

既定と異なる `--orders`、`--wavelengths`、`--asr-ratios`、参照次数などで計算した場合、再作図にもその引数を指定してください。

**検証**

```bash
python -m unittest studies.asr_1d_comparison.test_comparison
python -m unittest paper_reproductions.vallius2002.validation.test_galerkin paper_reproductions.vallius2002.validation.test_backends
```

実studyのLiと変換した層の通常FMMの一致、金・PMMAのポート電力、層の順序・総厚・谷底境界、3材料の区間、ASRのCPUテンソル経路との一致、非受動な値を収束として扱わないことを検査します。検証成功から500 nm構造の全次数・層数収束を保証することはできません。

追加のGalerkin検証は、無損失層の電力保存、任意の伝搬ポート励振に対する受動性、内部材料損失の体積積分との一致、3材料層・斜入射のバックエンド一致を含みます。CUDA環境では実GPUのテストも実行し、CUDAがない環境では明示的にスキップします。

固有モード・投影・界面の段階別検査（指定デバイス、代表層と界面）：

```bash
python -m studies.asr_1d_comparison.diagnose_passivity --study pmma_gold_grating_1d --order 16 --wavelength 600 --slices 300 --device cuda
```

`tm_stage_audit.json` に、内部・縮約された損失行列の最小固有値、投影後の人工増幅、電力内積の不一致、界面の場と電力の連続誤差を保存します。損失行列の符号は行列ノルムで規格化して数値丸めと区別します。

**保持次数を固定した条件比較をPythonファイルで実行**

```bash
python -m studies.asr_1d_comparison.run_fixed_order_audit --device cuda
```

`run_fixed_order_audit.py` をプロジェクトのルートに置いた場合は、
`python run_fixed_order_audit.py --device cuda` でも実行できます。

保持次数M=48（N=97）、波長650/700 nm、金420分割、PMMA300分割＋キャップ、
Galerkin投影、有限Li参照M=96/88を固定します。
G=0.001で3N/4N/5Nを積分2048/4096点それぞれで比較し、
次に4N・4096点でG=0.0003/0.003を計算します。
G=0.001の基準は4N・4096点の結果を使います。
積分点数が内部倍率ごとに自動増加することを避けるため、
指定する全積分点数が最大内部次元での自動下限以上かを検査します。

比較計算は指定デバイスで実行し、界面・S行列接続の条件数も保存します。
投影行列W=KHの2ノルム条件数と固有値・電力内積の残差は、
比較と同じ指定デバイス・バックエンドによるTM代表層診断で調べます。
CUDA診断では、本体の `TorchPreparedLayer.modes()` が生成するW/Vと内部固有対を使い、
固有値・投影・SVD条件数・損失行列・界面接続の線形代数をGPUで実行します。
大きい診断行列をCPUへ転送して計算しません。形状・材料・積分係数の準備は従来どおりCPUです。
代表層の診断であり、全層・全スタックの電力残差を測るものではありません。

既定の出力は `studies/asr_1d_comparison/results/fixed_order_audit_M48/` です。
`audit_plan.json` に計算条件と呼び出すコマンドを記録し、
条件別の元出力に加えて `comparison_summary.csv` と `projection_summary.csv` に集約します。
前者には物理量と `max_boundary_condition`、後者には
`projected_field_condition_2norm` と各残差が入ります。

計画だけ確認する場合は `--dry-run` を追加します（計算・出力ファイル作成なし）。
`--stage compare` で比較計算のみ、`--stage projection` で投影診断のみ、
`--stage internal` / `quadrature` / `g` で比較の各段階を選べます。
`quadrature` は2048点の基準に対する4096点の計算、`g` はG=0.001の基準に対する
G=0.0003/0.003の計算です。基準も必要な場合は先に `compare` を実行してください。

同じ条件で再実行すると比較計算の完了済みケースを再利用します。
投影診断は再計算します。CUDAが未利用の場合、明示したCUDA指定はCPUへ切り替わりません。
条件を変更する場合は新しい `--output-root` を指定します。
比較モジュールがケース単位のエラーを保存して正常終了した場合も、
このランナーは不足・失敗ケースと条件数の欠落を検出して停止します。
保持次数は1点だけなので、既存レポートの隣接次数による収束判定とは区別して、
内部倍率・積分・Gによる変化を評価してください。

投影診断の `threadpoolctl` は任意依存です。未インストールでも停止せず、
NumPy/SciPyの読み込み前にBLAS/OpenMPの環境変数でCPUスレッド数1を要求します。
利用可能な場合は従来どおり `threadpool_limits(limits=1)` を使います。
診断JSONの `cpu_thread_control` に制御方法を保存します。
GPU比較を終えた後に投影診断で停止した場合は、同じ条件・出力先で
`python run_fixed_order_audit.py --device cuda --stage projection` により診断から再開できます。
診断・ランナーの変更は比較の数値ソースを変更しないため、
保存済みのGPU比較チェックポイントはそのまま利用します。
GPU診断は `projection/cuda_auto/`（既定のCUDA・backend=auto）に保存し、旧CPU診断と分けます。
`projection_summary.csv` のbackend/device/linear_algebra_deviceとmodal_trace_sourceで実行経路を確認できます。
実行前の検査は `python -m studies.asr_1d_comparison.diagnose_passivity --device cuda --check-only`、
デバイス診断の検証は `python -m unittest studies.asr_1d_comparison.test_device_diagnostics` です。
後者はCUDA環境でGPU上の行列を検査し、CUDAがなければGPU試験を明示的にスキップします。
