# Weiss et al. (2009) matched-coordinate ASR 再現

対象は T. Weiss et al., *Matched coordinates and adaptive spatial resolution
in the Fourier modal method*, Optics Express 17, 8051-8061 (2009),
DOI: 10.1364/OE.17.008051 である。

## 最初に確認するファイル

| 場所 | 内容 |
|---|---|
| `reproduce.py` | Fig. 2～5 の計算とプロット。`--use-symmetry` はここで選ぶ。 |
| `verify_fig4a.py` | Fig. 4(a) 全スペクトルの論文曲線との一致確認。 |
| `fig4a_spectrum_only.py` | 論文曲線を重ねない計算スペクトル。 |
| `diagnose_fig3.py` | 無損失 Fig. 3 の固有値・条件数・電力流診断。 |
| `compare_boundaries.py` | Fig. 3 の境界投影方式を比較。 |
| `reference/` | 論文図から抽出した参照CSVと出所情報。 |
| `validation/` | 検証・テスト。 |
| `docs/` | [診断方法](docs/DIAGNOSE_ja.md)、[境界比較](docs/COMPARE_BOUNDARIES_ja.md)、[保存済み結果の評価](docs/RESULTS_ja.md)。 |
| `results/` | [結果フォルダの索引](results/README_ja.md)。計算条件・出所と一緒に保存。 |

現時点では、Fig. 4(a) の625調和波・全スペクトルは PDF 曲線との規定許容差内で一致した。
Fig. 3 の ASR エネルギー保存と Fig. 4(b) の高次数収束は未達成である。
特に Fig. 4(b) は次数 15～18 で T/R/A が大きく変動する。
図ごとの評価範囲は [保存済み結果の評価](docs/RESULTS_ja.md) にまとめた。

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
この再現用CLIの `--use-symmetry` は、完全な C2v の4ブロックで固有値を解き、
全モードと両偏光を保持する。境界行列は全サイズのままで、高次数の数値安定性を
保証するものではない。

## 実行

`outputs` をカレントディレクトリにするか、その親を `PYTHONPATH` に含める。
PyTorch、torcwa 0.1.4.2、matplotlib が必要である。

高速な一貫性確認（以下は `outputs` で実行する bash 例）:

```bash
python -m paper_reproductions.weiss2009.reproduce --study smoke --device cpu \
  --output-dir paper_reproductions/weiss2009/results/smoke
```

論文次数の全計算:

```bash
python -m paper_reproductions.weiss2009.reproduce --study paper --device cuda \
  --orders 6:15 --grid 256 --identity-grid 1024 --frequencies 250:470:5 \
  --spectrum-order 12 --output-dir paper_reproductions/weiss2009/results/paper
```

個別には `--study mapping`、`fig2`、`fig3`、`fig4-spectrum`、
`fig4-convergence` を選べる。次数 `N` に対する調和波数は `(2N+1)^2` で、論文の
169、361、625、961 harmonics はそれぞれ `N=6,9,12,15` に対応する。
ASR なしの matched-coordinate Jacobian は区分的に不連続なので、既定ではその系列だけ
`--identity-grid 1024` を使う。`N=12` では grid 256 が基本モードを別枝へ誤追跡したのに対し、
grid 1024 で解析解に対する相対誤差が約 `8.8e-6` へ戻ることを確認した。ASR 系列は
`--grid 256` を使う。
高次数の Fig. 4(b) に `--use-symmetry` を指定する場合も、
[保存済み結果の評価](docs/RESULTS_ja.md) のとおり収束を確認する必要がある。

## 出力

各 study は CSV、PNG、条件を保存した JSON を生成する。`fig4-convergence` は同じ
RCWA 解から Fig. 4(b) の透過率と Fig. 5(b) の x/y 差を出す。著者の元数値データは
同梱していない。下記の全スペクトル検証では、PDF内の実際のベクトル曲線から抽出した
参照値を、再計算結果と明確に区別して使用する。
`reproduce.py --study fig4-spectrum` の既存PNGは全回折次数の T/R を描き、
下記 `verify_fig4a.py` は論文図との比較にゼロ次 T/R を使用する。
高周波側では回折次数が開くため、二つのプロットの縦軸定義を混同しない。

検証:

```bash
python -m paper_reproductions.weiss2009.validation.validate --device cpu
python -m unittest paper_reproductions.weiss2009.validation.test_asr_symmetry \
  paper_reproductions.weiss2009.validation.test_compare_boundaries \
  paper_reproductions.weiss2009.validation.test_fig4a_verification
```

検証は式 (41)-(42) の界面勾配、界面位置、写像 Jacobian の正値、解析モード基準、
Drude 符号、低次数散乱の有限性、x/y 対称性を確認する。

## Fig. 4(a) 全スペクトルの一致確認

`verify_fig4a.py` は既存の円形ASRと一般2次元変換行列Tの計算経路を使う。
物理計算の式は変更していない。`reference/fig4a_reference.csv` は論文PDFの
11ページにあるT/R/Aベクトル曲線から抽出した値であり、著者の元数値データではない。
抽出元PDF・CSVのSHA256、軸の較正、元ベクトル座標をJSONへ記録している。
T/Rは各101頂点、Aは94頂点を持ち、AのみT/Rの周波数位置へ線形補間した。
曲線の平滑化、物理モデルへのフィット、R+T+Aの正規化はしていない。

図の描画範囲はラベルの250～450 THzより広い。既定では抽出した全101周波数点
（約241.8～483.6 THz）を使い、従来の250～470 THz指定で端を欠くことを避ける。

既存の `outputs` ディレクトリから実行:

```bash
python -m paper_reproductions.weiss2009.verify_fig4a --device cuda --resume
```

既定条件は `--order 12 --grids 256,512 --frequencies reference`。
625調和波・complex128・正入射・x/y両偏光で、2格子×101周波数＝202ケースを計算する。
周期700 nm、半径150 nm、高さ50 nm、上下と背景は真空、eta=0.97。
金のDrude定数は元論文設定で、高周波誘電率は既存実装と同じ1とする。
CPUでは `--device cpu --threads 4` を指定できる。

出力先は `paper_reproductions/weiss2009/results/fig4a_verification`。

- `fig4a_comparison.png`: 論文曲線との重ね描き、符号付き誤差、格子差
- `fig4a_spectrum.csv`: 全ケースのT/R/A、両偏光、計算格子、実行時間
- `fig4a_pointwise_errors.csv`: 最高密度格子と論文曲線の各周波数での比較
- `fig4a_grid_errors.csv`: 2つの最高密度格子間の差（両偏光）
- `fig4a_report.json` / `.md`: 判定、最大絶対誤差、RMSE、最大誤差周波数、標本点の極値
- `fig4a_checkpoint.json`: 1ケースごとの再開用保存

### ゼロ次と全回折次数の区別

図4(a)の参照曲線は約428 THz以上でT+R+Aが1より小さくなる。最初の回折閾値
`c/P = 428.27494 THz` と対応するため、既定の比較はゼロ次の `T0` / `R0` とした。
これは図からの推定であり、論文キャプションに回折次数の明示があるわけではない。
`--paper-power total` で全次数の合計との比較へ切り替えられる。
どちらを選んでも、もう一方の定義での誤差もJSONに保存する。誤差の小さい方式を
自動選択して「一致」にすることはない。

CSVの `T_x/R_x` は従来と同じ全回折次数の合計、`T0_x/R0_x` はゼロ次で、y偏光も同様。
吸収は常に `A=1-T_total-R_total`。`1-T0-R0` は高次回折光も含むため吸収にしない。
この追加は `reproduce.py` の返り値・CSV列への追加で、既存 `T_x/R_x/A_x` の意味を変えない。

合格条件は、全範囲を3 THz以下の間隔でカバーし、625調和波を使用し、全指定ケースが
完了していること。その上で受動性（許容誤差1e-6）、正のJacobian、x/y偏光差1e-6以下、
2つの最高密度格子間の全帯域T/R/A差0.005以下を要求する。
論文との比較は各T/R/Aについて最大絶対誤差0.02以下、RMSE 0.01以下を既定値とする。
0.02はパワー比で2パーセントポイントであり相対誤差2%ではない。
これはPDFの線幅・座標較正の不確かさを考慮した実用的な利用者基準で、論文が保証する
誤差限界ではない。`--paper-atol` / `--paper-rmse` / `--grid-atol` で変更できる。
吸収はA=1-R-Tなので、R+T+A=1を独立の正しさの証拠として使わない。

判定は `matched_within_tolerance`（基準内一致）、`mismatch`（格子収束済みだが論文と不一致）、
`inconclusive`（未完了・部分帯域・格子未収束・非物理値など）を区別する。
紙面と同じ625調和波での比較であり、Fourier次数を無限に増やした収束の証明ではない。
`--strict-exit-code` は一致時0、不一致2、判定保留3を返す。通常は結果を保存して0を返す。

`--resume` は同じ物理条件・次数・精度・cascade・参照CSV・ソースハッシュのときだけ再利用する。
格子や周波数を追加して再開でき、現在指定した範囲だけを比較対象にする。
ソース変更や次数変更の場合は、新しい `--output-dir` を使う。
同じディレクトリで同時に複数プロセスを実行しないこと。

格子差が大きい場合の追加計算:

```bash
python -m paper_reproductions.weiss2009.verify_fig4a --device cuda --grids 256,512,768 --resume
```

既存結果を使って許容誤差だけ変更し再評価（torch不要）:

```bash
python -m paper_reproductions.weiss2009.verify_fig4a --analyze-only --paper-atol 0.01 --paper-rmse 0.005
```

格子・周波数・次数を既定値から変更した結果を再評価するときは、同じ指定を併記する。
中断後でも解析でき、未計算点があれば必ず判定保留になる。

短い実計算テスト（全スペクトル一致の証明にはならない）:

```bash
python -m paper_reproductions.weiss2009.verify_fig4a --device cpu --order 2 --grids 32,48 --frequencies 350,370,430 --output-dir paper_reproductions/weiss2009/results/verification_smoke --resume
python -m unittest paper_reproductions.weiss2009.validation.test_fig4a_verification -v
```

参照値を元PDFから再抽出する場合のみ、`pdfplumber` が必要:

```bash
python -m paper_reproductions.weiss2009.extract_fig4a_reference --pdf "/path/to/Matched coordinates and adaptive.pdf"
```

抽出器は今回の出版社PDFのレイアウト専用で、軸や曲線が認識できないPDFでは推測せず停止する。
通常の比較実行ではPDF・pdfplumberは不要。

参照CSVの整合性検査はUTF-8テキストをLF・末尾改行へ正規化したSHA256を使う。
GitやWindows/Linux間の転送によるCRLF/LF変換は許容するが、数値・列・文字の変更は拒否する。
