# 保存済み Peng 2025 結果の判定

この文書は、リポジトリに保存されている既存結果を、受動性と次数収束の観点から判定したものです。

## `square_nvm_convergence`

- N=23, 24, 26 の透過率は 0.890279, 0.888912, 0.894530。
- 末尾3点の変動幅は、Rが約0.00139、Tが約0.00562、Aが約0.00664。
- N=26では A=-0.001381 で、小さいものの厳密な受動性を満たさない。

したがって、解析Fourier NVMは `provisional_small_passivity_error`（ほぼ安定だが未確定）と判定する。
論文Fig. 2(c)の曲線との差をコード誤差だけに帰属することはできない。論文にはAg Drude定数と
Fig. 2のPI厚 `h2` の数値がなく、保存結果は半無限PIと既定Drude定数を仮定しているためである。

## `square_matched-asr_outer_convergence`

- 13点中9点が受動性診断に失敗。
- N=23, 24, 26でもRの変動幅は約0.124で、透過率がほぼゼロへ崩れている。

この結果は `not_converged` であり、論文再現結果として使用しない。原因はouter-only ASR経路が
曲面界面のNV因数分解を持たず、Ag/空気の極端なコントラストに対して不安定だったことである。

## `matched-nvm outer` の添付実行結果

- N=8でT=1.1866、N=24でR=10.2193となり、13点中6点が非受動。
- 高次数側でもTがほぼ0へ崩れ、末尾は安定しない。
- `R/p=30/62`、grid=256、`G=0.001`のouter-only非分離写像では、
  `min(det J)`が約`1.2e-10`となり、写像が数値的にほぼ特異である。

これは物理解ではなく `not_converged` である。修正版では同条件のouter写像を固有値計算前に
拒否し、`auto`で単調性保証付きdouble写像を選ぶ。また一般化Li NV補正は誘電率tensorだけへ
適用し、透磁率tensorはWeiss対称ASR因数分解とする。

## `square/square_mi.*`

これは旧 `matched-asr outer` の未収束スペクトルで、R>1またはA<0の点を多数含む。
比較図・再現図として使用しない。修正版は方式と試験ごとの別フォルダへ出力するため、今後この
旧結果へ上書きしない。

## 修正版で最初に行う計算

```bash
python -m paper_reproductions.peng2025.reproduce_square \
  --study convergence \
  --solver matched-nvm \
  --radial-mapping auto \
  --orders 4,6,8,10,12,14,16,18,20,22,23,24,26 \
  --grid 256 \
  --device cuda
```

metadataの `convergence_assessment.status` が `converged` になってから、同じ方式と次数で
`--study spectrum`を実行する。`provisional_small_passivity_error`または`not_converged`のままなら、
提示された次次数まで拡張し、PI厚とDrude定数の感度は収束後に分離して調べる。
