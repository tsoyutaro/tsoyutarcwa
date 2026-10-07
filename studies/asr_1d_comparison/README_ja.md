**既存の一次元金格子：Cartesian Li と Vallius 1D ASR の次数収束比較**

`gold_grating_1d` と `pmma_gold_grating_1d` の既存のLi実装を呼び、同じ形状・分散材料・波長・偏光・基板で `paper_reproductions.vallius2002` のASRを計算します。共通の `rcwa_ext` と既存studyの物理モデルは変更しません。元のリポジトリルートから実行してください。

**両studyを計算する**

```bash
python -m studies.asr_1d_comparison.compare --study all --device auto
```

CUDAが利用可能ならGPU、利用できなければCPUで計算します。`--device cuda` で明示したGPUが利用できない場合はエラーになり、CPUへ切り替えません。ASRのCPUは既定でSciPyです。`--device cpu --backend torch` を指定するとGPUと同じテンソル計算経路をCPUで使えます。既存Li側はCPU/GPUともPyTorchです。

初期値は `M=2,4,8,12,16,24`、`N=2M+1`、ASR内部次元 `4N`、G=0.001。波長は既存configの400,450,500,550,600,650,700 nm、垂直入射、TE/TMです。金は420層、PMMAは300分割＋金キャップ1層を固定します。保存済みの探索で使われた層数を採用していますが、この次数比較で層数の収束を再判定するわけではありません。

横軸MはFourier展開の打切り次数です。エバネッセント成分を含めて−M～+MのN成分を保持します。TEは電場が溝方向、TMは磁場が溝方向です。

金のLi側は既存の解析的Fourier係数の経路にそろえます。横方向のサンプリングgridは使いません。PMMA側は従来から解析的な区間積分です。各studyの材料読み込み・補間関数をそのまま使用し、LiとASRに同じ複素誘電率を渡します。金studyの線形epsilon補間と、PMMA studyの線形n,k補間後の二乗は、それぞれ元studyに合わせます。元の材料CSVは変更しません。

既定のASR倍率を4にしているのは、今回のPMMA被覆格子で3NのTMが受動性を大きく破る検証ケースが見つかったためです。4Nがすべての波長・次数で十分という保証はありません。3N/4Nを並べる場合：

本番のPMMA300分割では4NのM=16にも受動性違反が出ました。M=16・600 nmを5Nにした確認でも改善せず、界面接続の最大条件数は約4×10²⁵でした。現行ASRのTMに安定性の問題があり、赤い×の結果から収束を判断できません。内部倍率だけでなく、モード選択・投影・界面接続の検証が必要です。

同じ4N・600 nmをSciPy側でも確認し、PyTorch側と同様に負の吸収になりました。今回の不安定性は計算経路を切り替えても残ります。

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
python studies/gold_grating_1d/compare_asr.py --device cuda --slices 420
python studies/pmma_gold_grating_1d/compare_asr.py --device cuda --slices 300
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
python -m studies.asr_1d_comparison.compare --study all --prepare-only
```

同じ引数で再実行すると完了済みケースを再利用します。M・波長・内部倍率の範囲を広げることもできます。材料・形状・数値ソース・G・積分点を変更した場合は混在を拒否するので、新しい `--output-root` を指定してください。CPU/GPUを再開時に変更した場合の実行環境はケースごとに残ります。

保存した範囲と同じ引数で再作図（計算はしない）：

```bash
python -m studies.asr_1d_comparison.compare --study all --report-only
```

既定と異なる `--orders`、`--wavelengths`、`--asr-ratios`、参照次数などで計算した場合、再作図にもその引数を指定してください。

**検証**

```bash
python -m unittest studies.asr_1d_comparison.test_comparison
```

実studyのLiと変換した層の通常FMMの一致、金・PMMAのポート電力、層の順序・総厚・谷底境界、3材料の区間、ASRのCPUテンソル経路との一致、非受動な値を収束として扱わないことを検査します。検証成功から500 nm構造の全次数・層数収束を保証することはできません。
