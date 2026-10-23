# 金モスアイ：通常NVと3次元法線FFFの比較（2026-10-22修正版）

`studies/gold_motheye_fff2d` は、x/yの2方向周期とz方向の側面傾斜を
同時に扱うベクトル計算です。三角格子上の**円錐台**を対象にします。
三角格子と三角錐は異なり、三角錐の面形状は今回実装していません。
Mohamad2020の1D論文そのものに2Dモスアイの参照解があるわけではありません。

## CUDA初回起動の修正（2026-10-22）

ユーザ提供ログでは `--device cuda` の19検査と円柱18点は成功しています。
`cuda:0` の初回起動はCUDA初期化前にメモリ統計をリセットしていたため、
`RuntimeError: Invalid device argument` になりました。CUDA Backend作成時に
`torch.cuda.init()` を呼ぶよう修正しました。相反性検査のKeyErrorは先行検査失敗の影響です。

失敗時は例外名を標準出力、完全なスタックをJSONへ保存します。
失敗済みの事前検査は次回実行で再試行します。
単独validateの既定ファイル名もデバイスに合わせました（cuda:0ならphysics_cuda_0.json）。
計算式、金CSV、物理検査の許容誤差は今回変更していません。

```bash
python -m unittest studies.gold_motheye_fff2d.validation.test_cuda_startup -v
python -m studies.gold_motheye_fff2d.validate --device cuda:0 --json studies/gold_motheye_fff2d/results/physics_cuda_0_initfix_20261022.json
```

新しいPythonプロセスからのcuda/cuda:0起動を確認する実GPUテストを含みます。
CUDAがない環境では実GPUテストのみSKIPし、初期化順序の回帰テストはCPUで実行します。
修正版のGPU実機再検証は実行先で行ってください。

## 流束・損失演算子・収束の表示を分離

`--loss-diagnostics`では、R/P_sub/Aが0〜1にあっても、損失行列に負固有値があれば
総合判定をUNPHYSICALにします。この場合は計算した入射場で負の吸収を観測したとは限りません。
従来のpassive_flux_bounds表示はこの演算子検査も含めていたため、実際の流束検査だけの値へ修正しました。

- `passive_flux_bounds_passed`：R/P_sub/A_moth/A_totalの有限性と範囲。
- `physical_checks_passed`：演算子の受動性などを含む従来と同じ総合判定。
- `constitutive_loss_flagged_cases`：損失演算子が受動性検査に不合格だった点数。
- `axis_numerically_converged`：固定した他条件の下で最後の2回の数値差が許容値以下か。
- `axis_converged` / `passed`：従来どおり物理検査合格も要求する厳しい判定。

各UNPHYSICAL点にfailed_checksと最小損失固有値を出力します。判定閾値と終了条件は維持しています。
数値差が小さいことだけでは物理精度を保証できません。今回変更したのは表示・集計です。
数式・Au材料・伝搬処理は変更していません。新しいコードでの計算には新しいoutput-dirを指定してください。

## 1Dの状況と今回の修正

1Dの全条件が解決したという意味ではありません。保存済み
`paper_reproductions/mohamad2020/results/paper_fixed_20261020` は383/950点、
12点がUNPHYSICALです。図5のDM-FFF N=100の参照値に対する相対誤差は約0.0641%ですが、
図9a GILなし N=100ではA=-0.090006が残り、次数収束・物理妥当性は未達です。
外部計算機の現在のジョブ稼働状況はこの保存コピーだけでは分かりません。

2Dには次を反映しました。

- DMの既定を固定RK4から適応CF4へ変更。全Sブロックで局所誤差を検査し、
  微小区間ごとのS行列を接続します。最小刻みでも誤差を満たさなければ停止します。
- GILの既定を半径rの両側にdelta/2ずつ置く配置へ変更。全幅はdeltaです。
  `--gil-side dielectric` は以前の外側だけの配置です。
- 3D法線の延長部分のPzzが傾斜ゼロでも残っていた問題を修正。
  円柱では通常NVと同じ法線射影・Maxwell行列へ厳密に戻ります。
- 体積吸収用の内部場も左右のS行列から再構成し、成長する伝達行列による桁落ちを避けます。
- 独立検査が失敗したときは長い掃引を開始しません。手法間差分をreport.jsonに保存します。

1DのFFT材料界面交差処理はそのまま移植しません。2Dでは円・環の材料Fourier係数を
Bessel積分で解析的に求めるので、材料境界のFFT画素交差はありません。
法線射影のFFT格子Q、次数M、層数Nzはそれぞれ収束確認が必要です。

## 材料・手法

Auは既存の `studies/shared/gold_dispersion.py` と
`studies/gold_motheye2/data/au_measured_nk.csv` を共用します。
各波長で誘電率を補間し、GILの複素nも再計算します。GIL厚さを固定して広帯域を計算できます。
GILは半径方向の全幅であり、斜面法線方向に一定の厚さではありません。
GILは物理モデルを変えるので、幅・環数の収束も別に比較してください。

|方法|内容|
|---|---|
|slice-nvm|通常のスライスNV。nz=0、横方向の同じ対称FFFを使用|
|rcwa-fff|3成分法線(nr,nz)、xz/yz連成を含み、各層中点の行列指数で伝搬|
|dm-fff|同じ3D法線Maxwell方程式を連続z依存で適応CF4積分|
|direct|FFFを用いないLaurent積の比較系|

`slice-nvm`自体は同じ新ソルバー内の比較系ですが、独立した既存
`rcwa_ext.AutoRCWA`の固有モードNVとも照合しています。矩形Fourier集合、
Lanczosなし、対称性縮約なしに条件を揃え、正方・三角格子の金円柱と誘電体テーパーで
全次数の複素振幅・パワーを比較します。SciPyとTorchのBessel関数評価差も記録します。
円柱での一致は実装の検証であり、斜面での精度向上を証明するものではありません。

既定は周期200 nm、高さ500 nm、先端半径5 nm、底面半径95 nm、空気正入射、半無限Auです。
三角格子の既定Fourier集合はD6で閉じた六角形で、B=1+3M(M+1)、場の次元4Bです。
`--harmonic-shape rectangular` では既存NVと同じ(2M+1)^2に揃えられます。
両手法の比較は必ず同じFourier集合・層数・Q・材料・GILで行ってください。

## 実行

`studies`、`paper_reproductions`、`rcwa_ext` があるプロジェクト直下で実行します。
このPCでは次の場所です。Python環境は既存RCWA環境を使用します。

```powershell
Set-Location 'C:\Users\tsoyu\Documents\Codex\2026-08-06\files-mentioned-by-the-user-extensions\outputs'
python -m studies.gold_motheye_fff2d.validate --device cpu --json studies/gold_motheye_fff2d/results/verify_user_20261022/physics.json
```

CPUでの19検査を実行済みです。独立NV比較、垂直円柱の全S行列の一致、Fresnel解析解、
損失なし保存則、相反性、任意開放チャネル、1D TMへの縮約、体積散逸と境界流束、
変係数DMと独立DOP853積分との比較、最小刻み停止を含みます。
GPUでは同じ検査を再度実行してください。
実際の金モスアイM=3・700 nm・Nz=24にはR=3.3284の非物理点が残ります。
Nzを48/96へ増やしても損失演算子の負固有値と未収束があり、精度向上は未認定です。
[今回の検証結果](results/verification_20261022_final/ASSESSMENT_ja.md)を参照してください。

円柱比較の小規模実行:

```powershell
python -u -m studies.gold_motheye_fff2d.run --device cpu --profile cylinder --height-nm 80 --cylinder-radius-nm 60 --orders 1,2,3 --slices 8 --grid 96 --wavelengths 400,550,700 --methods slice-nvm,rcwa-fff --output-dir studies/gold_motheye_fff2d/results/cylinder_user_20261022
```

モスアイの小規模比較（低次数での物理異常を含めて検出）:

```powershell
python -u -m studies.gold_motheye_fff2d.run --device cpu --orders 1,2,3 --slices 24 --grid 96 --wavelengths 400,550,700 --methods slice-nvm,rcwa-fff --output-dir studies/gold_motheye_fff2d/results/cone_small_user_20261022
```

CUDA環境での次数収束。本計算前に同じコマンドへ `--dry-run` を追加し行列規模を確認します。
このPCのPyTorchはCPU版なので、高次数CUDA実行は未検証です。

```powershell
python -u -m studies.gold_motheye_fff2d.run --device cuda:0 --orders 2:12:2 --slices 100 --grid 256 --wavelengths 400,550,700 --polarizations x,y --methods slice-nvm,rcwa-fff --loss-diagnostics --require-convergence --output-dir studies/gold_motheye_fff2d/results/cone_order_M12_Nz100_Q256_20261022
```

層数・法線格子も別に確認します。M=8は実行例で、次数収束を確認したMへ変更してください。

```powershell
python -u -m studies.gold_motheye_fff2d.run --device cuda:0 --axis slices --values 40,80,120,160 --order 8 --grid 256 --methods slice-nvm,rcwa-fff --loss-diagnostics --require-convergence --output-dir studies/gold_motheye_fff2d/results/cone_slices_M8_Q256_20261022
python -u -m studies.gold_motheye_fff2d.run --device cuda:0 --axis grid --values 128,256,384,512 --order 8 --slices 120 --methods slice-nvm,rcwa-fff --loss-diagnostics --require-convergence --output-dir studies/gold_motheye_fff2d/results/cone_grid_M8_Nz120_20261022
```

GILは別フォルダで同条件比較します。各方法に同じGILを与えます。

```powershell
python -u -m studies.gold_motheye_fff2d.run --device cuda:0 --orders 2:12:2 --slices 100 --grid 256 --gil-nm 1 --gil-side centered --gil-rings 24 --methods slice-nvm,rcwa-fff --loss-diagnostics --output-dir studies/gold_motheye_fff2d/results/cone_GIL1_order_20261022
```

DMによる斜面計算の追加照合。最初は低次数・単波長で計算量を確認します。

```powershell
python -u -m studies.gold_motheye_fff2d.run --device cpu --orders 1,2 --slices 24 --grid 64 --wavelengths 550 --methods slice-nvm,rcwa-fff,dm-fff --dm-integrator cf4-adaptive --dm-step-nm 2 --dm-rtol 1e-6 --dm-atol 1e-9 --output-dir studies/gold_motheye_fff2d/results/cone_dm_small_20261022
```

`--axis dm-step --values 2,1,0.5 --methods dm-fff --order 2` は最大試行刻みの確認です。
適応積分ではさらにrtol/atolを1桁ずつ厳しくする比較が必要です。
固定CF4と旧RK4は `--dm-integrator cf4` / `rk4` で選べます。

## 出力・判定

- `cases.csv`、`convergence.png`：R、Au基板流入P_sub、モスアイ域吸収A_moth。
- `checkpoint.json`：各次数の回折効率、物理判定、設定、積分診断。
- `report.json`：各手法の収束と `method_comparisons` 内のNVとの差。
- `physics_validation.json`：独立検査。失敗時は掃引を開始せず終了コード1。
- `plan.json`：共有材料のパスとハッシュ、形状、Fourier集合、メモリ目安。

半無限AuではT_far=0、A_total=1-Rです。A_moth=1-R-P_sub。
R+A_total=1は定義であり、独立な検証には使いません。
`--volume-absorption --absorption-quadrature 7` で体積散逸との照合を追加できます。
体積積分も切断した構成行列に基づくため、連続体解への精度保証ではありません。

`--loss-diagnostics` は深さ10%,50%,90%で構成行列の損失固有値を検査します。
複素金属では対称FFFでも有限次数で負になる可能性があります。
Rが0〜1にあっても、人工的利得の方向が検出された点はUNPHYSICALにします。
この場合、斜面を入れた方が高精度と結論しないでください。
負の吸収・R>1を切り詰めたり、規格化で隠したりしません。

収束判定は全指定点が物理検査に合格し、最後の隣接2回の
R/P_sub/A_mothの絶対差が全て既定5e-4以下の場合です。少なくとも3点必要です。
ある軸だけの収束は他の軸の収束を保証しません。
低次数に異常があって高次数で回復した場合も、その全点を含む実行の総合判定は不合格です。
結果を見て高次数の範囲を別フォルダで追加検証できます。
通常完了0、実行・物理異常1、`--require-convergence`指定時の未収束2。
`--skip-physics`時は総合passed=trueにはしません。

同じコード・設定・出力先で再開できます。コード変更やGIL/形状/固定層数変更は混用を拒否します。
今回の修正前の出力先には追加せず、新しいフォルダを指定してください。
数式と文献は [FORMULATION.md](FORMULATION.md) を参照してください。
