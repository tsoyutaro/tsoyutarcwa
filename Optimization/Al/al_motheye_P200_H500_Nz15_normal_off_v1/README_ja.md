# Al：NVMを維持した正則化OFFの追加最適化

この独立ケースはAlの単一金属円柱／同じAlの半無限基板を対象にします。PMMA複合ケースではありません。既存のAlケースと実行結果は変更しません。

## OFFの意味

NVMの補助法線場、法線・接線分解、Fourier factorization、対称性による縮約、波長間共有を維持します。円中心側とセル間のsmoothstep補正を外し、最近接円柱中心への径方向からn n^Tを作ります。円中心のサンプルは角度平均I/2、等距離点は全ての最近接中心の射影行列の平均です。等距離判定は機械精度で行い、gridに連動する有限幅の補正を作りません。Lanczosは従来と同じくOFFです。

normal_field_definitionはnearest-radial-center-and-tie-mean-v1です。referenceと共有計算の両経路に同じ定義を実装しています。正則化OFFの法線行列は半径・波長に依存しないため、幾何に依存するキャッシュとして再利用し、定数行列を半径へのVJP対象から外しています。

OFFが物理的に正しい、または有限次数ですべて収束するという意味ではありません。固定形状での次数依存性は引き続き確認してください。

## 条件と初期形状

- 六方格子、15層、周期200 nm、高さ500 nm。
- 最適化はM8/Q1024、400–700 nmの10 nm刻み31波長、1 Optuna trial、Adam最大500 step。早期停止条件は従来どおりです。
- 目的関数は従来と同じ平均反射率の最小化。全吸収率1−Rの最大化に対応します。モスアイ部分のみの吸収率1−R−P_subは目的関数に含めていません。
- 初期値は旧Alケース results/hex_trial1_v3 の保存済み最良形状です。initial_profile.jsonとconfig内initial_logitsに固定しています。
- 旧Adam/Optunaの状態はコピーせず、新しい状態から始めます。1 trialの初期learning rateは0.03です。
- 新しい結果フォルダはresults/hex_trial1_M8_Q1024_off_v1です。最適化後に自動検証は実行しません。

## 1. 実際のOFF目的関数の勾配検証

以下はLinuxで貼り付ける1行です。検証波長は400/550/700 nmの3点です。

```bash
( set -e; cd /home/7/uq06557/common2/20260813/tsoyutarcwa/Optimization; d="$PWD/Al/al_motheye_P200_H500_Nz15_normal_off_v1"; python "$d/validate_off.py" --device cuda:0 --order 8 --grid 1024 --wavelengths 400,550,700; )
```

全16 logitsについて、reference forwardによる中心差分（h=1e-4とh/2）を共有計算の自動微分と比較します。共有計算とreferenceの損失・パワー・全勾配、層checkpoint OFF/ONの一致も確認します。FDの許容差はabsolute1e-6、relative0.002です。これは微分の整合性の判定であり、Fourier次数やgridの収束判定ではありません。

status=passedを確認してから次へ進んでください。結果はvalidation/<識別子>/report.json、ポインタはlatest_off_validation.jsonです。31波長全てに対する網羅的な勾配検証ではなく、3波長に対する検証です。

## 2. 別フォルダで1 trial

```bash
( set -e; cd /home/7/uq06557/common2/20260813/tsoyutarcwa/Optimization; d="$PWD/Al/al_motheye_P200_H500_Nz15_normal_off_v1"; python "$d/run.py" --config "$d/config_hex_trial1_M8_Q1024_off.json" --device cuda:0 --mode optimize --lattices hexagonal --skip-audit --timeout-hours 0 --output-root "$d/results/hex_trial1_M8_Q1024_off_v1"; )
```

最適化では従来どおり物理的なパワーの範囲をチェックし、非物理値をクリップしません。非物理値でtrialがFAILした場合、同じ出力フォルダでtrial数1のコマンドを再実行しても、新しいtrialは自動追加されません。OOM・中断などでRUNNINGの状態は同一コマンドで再開します。stale controller.lockの回復は、旧プロセス停止を確認してから--recover-lockを指定します。

## 3. 旧・新形状を同じOFF・高次数で評価

```bash
( set -e; cd /home/7/uq06557/common2/20260813/tsoyutarcwa/Optimization; d="$PWD/Al/al_motheye_P200_H500_Nz15_normal_off_v1"; python "$d/compare_off_profiles.py" --device cuda:0 --run-root results/hex_trial1_M8_Q1024_off_v1 --order 30 --grid 1024 --wavelengths 400,550,700; )
```

旧形状はinitial_profile.json、新形状は新しい実行フォルダの最良logitsです。両方を新しいOFFソルバーで評価します。旧ON計算の反射率を、そのまま新OFF計算の反射率と比較する処理ではありません。

結果は新しい実行フォルダのhexagonal/postprocess/off_old_new_comparison/<識別子>/report.json、comparison.csv、comparison.pngです。ポインタはhexagonal/latest_off_comparison.jsonです。元のmanifest、Journal、Adamチェックポイント、最良形状の記録が比較中に変更されていないことをチェックします。

M30は今回の共通評価条件であり、収束値として確定した次数ではありません。3波長の平均値の改善は、全31波長の改善や全帯域の収束を保証しません。探索の各stepに完全収束を要求するものではありません。

## 文献と今回の実装

NVMの提案本体はSchuster et al. (2007)、以前引用したGötz et al. (2008)は補助法線場の自動生成を扱う続編です。2007年論文は円に対する径方向場、他の構成方法、不連続線と点特異点の扱いを比較しています。2008年論文は境界の法線から内部を補間する方法を示し、普遍的に優れた構成の選択則は示していません。今回のsmoothstep幅やOFFの平均規約は本実装の選択であり、論文が指定した値ではありません。

- Schuster et al., JOSA A 24, 2880–2890 (2007): https://doi.org/10.1364/JOSAA.24.002880
- Götz et al., Optics Express 16, 17295–17301 (2008): https://doi.org/10.1364/OE.16.017295

## ローカル検証の範囲

validation_reports/とnormal_off_case_validation_summary.jsonを参照してください。CPUの15層・M2/Q48で全16 logitsのreference有限差分・自動微分・checkpointの一致を確認し、M8/Q1024ではCPU共有計算の1方向の有限差分とcheckpoint一致を確認しています。CPUで実際のOptuna/Adamを1 trial・2 step実行し、旧・新形状比較の動作も確認しています。

CUDAでの全16変数チェック、M8/Q1024の500 step、M30での旧・新形状比較は未実施です。上のコマンドで実行してください。
