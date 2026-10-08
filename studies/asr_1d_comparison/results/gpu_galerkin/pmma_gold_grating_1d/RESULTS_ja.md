**pmma_gold_grating_1d: Cartesian Li と Vallius 1D ASR の次数比較**

元studyのconfig・材料関数を使用。周期=200.0 nm。形状はplan.jsonのgeometryに保存。
Nz=300固定、波長=[400, 450, 500, 550, 600, 650, 700] nm、TE/TM、N=2M+1。
M=[2, 4, 8, 12, 16, 24]、ASR内部倍率=[4]、G=0.001。
ASR TM境界場=galerkin。galerkinは電力内積を保つ改良版で、TEの境界場は従来と同じです。
Li参照M=80、参照の確認M=72。
参照間の全波長・全偏光・全指標の最大絶対差=0.0022229086832319667。
参照は有限次数の計算で、厳密解ではありません。層数の収束もこの次数掃引では再判定していません。

金格子のP_subは吸収性金基板に入る界面電力で、遠方透過T_far=0。PMMA格子のTはPMMA基板への透過電力です。
図の物理量は%、誤差と隣接差はパーセントポイントです。

受動性の範囲を破った値・計算失敗は赤い×で表示し、元の値はCSV/invalid_cases.jsonに保存します。
最大誤差の図は、その偏光で全波長が受動性範囲を満たす点だけを結びます。

| 系列 | 無効な偏光ケース数 | 最後の2区間が許容差内 | 最終次数と有限参照の差も許容差内 |
|---|---:|---|---|
| li | 0 | False | True |
| asr_r4 | 0 | False | False |

[物理量](values.png)・[有限Li参照との差](reference_difference.png)・[隣接次数の差](adjacent_difference.png)

条件とソース・材料のハッシュはplan.json/checkpoint.json、各ケースの実行環境はcheckpoint.jsonに保存しています。
ASRは内部固有モードから|γ²|が小さいN個を選び、物理x空間のN成分へ射影します。GPU化はこの打切り誤差を減らしません。
