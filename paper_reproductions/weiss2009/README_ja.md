# Weiss et al. (2009) matched-coordinate ASR 再現

対象は T. Weiss et al., *Matched coordinates and adaptive spatial resolution
in the Fourier modal method*, Optics Express 17, 8051-8061 (2009),
DOI: 10.1364/OE.17.008051 である。

## 再現範囲

既存の `rcwa_ext` に、論文の数値例で用いられた固定界面 ASR 写像を追加した。

- 式 (37)-(38): 正方格子内の円へ整合する 2 次元 matched-coordinate 写像
- 式 (41)-(42): `eta=0.97`、新旧の界面位置を同一とした 1 次元 ASR 写像
- 式 (29)-(36): x1/x2 の扱いを対称にした Fourier factorization
- Fig. 2(a): ASR を含む座標線
- Fig. 2(b): 誘電体円柱の基本 HE11 モード伝搬定数の収束
- Fig. 3(a): 高さ 50 nm の無損失誘電体円柱に対するエネルギー保存誤差
- Fig. 4(a): Drude 金円柱配列の T/R/A スペクトル
- Fig. 4(b): 370 THz における透過率の次数収束
- Fig. 5(b): 対称 factorization における x/y 偏光の擬似非対称性

Fig. 2(b)、3(a)、4(b)、5(b) の黒い小マーカーは Li (2003) の別の逐次
factorization である。現コードは Weiss 論文の対称式を実装しているが、この比較用
Li 版を同一 API には追加していない。このため Fig. 3(b) と Fig. 5(a)、および各図の
Li マーカーは再現対象外である。論文の zigzag Cartesian 近似も比較対象外とした。
したがって本実装は論文の中心手法と主要数値例の再現であり、全マーカーの完全複製ではない。

## 論文条件

| 計算 | 条件 |
|---|---|
| 誘電体円柱 | 周期 1.5 um、半径 0.5 um、内側 epsilon=4、外側真空 |
| 固有モード | 619.5 THz、解析的 HE11 円柱分散式を基準 |
| エネルギー保存 | 上記円柱、高さ 0.05 um、上下真空 |
| 金円柱 | 周期 0.7 um、半径 0.15 um、高さ 0.05 um、上下真空 |
| Drude 金 | plasma frequency 1.37e16 rad/s、damping 0.85e14 rad/s |
| ASR | eta=0.97、したがって界面で d(tilde x)/dx=1-eta=0.03 |
| Fig. 4(a) | 625 harmonics、すなわち各軸 Fourier order 12 |

金の高周波誘電率は、論文に別値の記載がないため `epsilon_inf=1` とした。時間依存は
論文と同じ `exp(-i omega t)` で、受動金属の虚部は正になる。

## 実装上の指定

円用 ASR profile は次の三つから選択できる。

- `matched_asr_profile="equalized"`: 既存の区間再配分型 profile（既定、後方互換）
- `matched_asr_profile="weiss2009"`: 今回追加した論文式 (41)-(42)
- `matched_asr_profile="identity"`: matched coordinates のみで ASR なし

`AutoRCWA` では `ASROptions(circle_profile="weiss2009", circle_G=0.03)` と指定する。
`circle_G` はこの profile では `1-eta` を表す。

## 実行

`outputs` をカレントディレクトリにするか、その親を `PYTHONPATH` に含める。
PyTorch、torcwa 0.1.4.2、matplotlib が必要である。

高速な一貫性確認:

```powershell
python -m paper_reproductions.weiss2009.reproduce --study smoke --device cpu `
  --output-dir paper_reproductions/weiss2009/results/smoke
```

論文次数の全計算:

```powershell
python -m paper_reproductions.weiss2009.reproduce --study paper --device cuda `
  --orders 6:15 --grid 256 --identity-grid 1024 --frequencies 250:470:5 `
  --spectrum-order 12 --output-dir paper_reproductions/weiss2009/results/paper
```

個別には `--study mapping`、`fig2`、`fig3`、`fig4-spectrum`、
`fig4-convergence` を選べる。次数 `N` に対する調和波数は `(2N+1)^2` で、論文の
169、361、625、961 harmonics はそれぞれ `N=6,9,12,15` に対応する。
ASR なしの matched-coordinate Jacobian は区分的に不連続なので、既定ではその系列だけ
`--identity-grid 1024` を使う。`N=12` では grid 256 が基本モードを別枝へ誤追跡したのに対し、
grid 1024 で解析解に対する相対誤差が約 `8.8e-6` へ戻ることを確認した。ASR 系列は
`--grid 256` を使う。

## 出力

各 study は CSV、PNG、条件を保存した JSON を生成する。`fig4-convergence` は同じ
RCWA 解から Fig. 4(b) の透過率と Fig. 5(b) の x/y 差を出す。論文の元数値データは
公開されていないため、画像から値を捏造せず、論文条件で再計算した値だけを保存する。

検証:

```powershell
python -m paper_reproductions.weiss2009.validation.validate --device cpu
```

検証は式 (41)-(42) の界面勾配、界面位置、写像 Jacobian の正値、解析モード基準、
Drude 符号、低次数散乱の有限性、x/y 対称性を確認する。
