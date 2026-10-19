# Mohamad et al. (2020) の1次元TM回折格子計算

対象論文: H. Mohamad et al., *Fast Fourier factorization for differential method and RCWA: a powerful tool for the modeling of non-lamellar metallic diffraction gratings*, Optical and Quantum Electronics **52**, 127 (2020), [DOI: 10.1007/s11082-020-2240-y](https://doi.org/10.1007/s11082-020-2240-y)。

通常RCWA、RCWA-FFF、DM-FFFを独立した1次元TMソルバーで比較します。
図5のアルミニウム正弦波格子、図6の電場分布、図9・10の金三角波薄膜格子と
graded index layer（GIL）を対象にしています。既存の `rcwa_ext` の2次元ソルバーの
設定変更は不要です。実装を用意したことと、論文の全曲線との定量的一致は区別してください。
MC-Grating/C-Methodは実装していません。

## 実行コマンド

PowerShellで、既存の `paper_reproductions/` がある `outputs/` を作業ディレクトリにします。
Python 3.10以上が必要です。

```powershell
Set-Location 'C:\Users\tsoyu\Documents\Codex\2026-08-06\files-mentioned-by-the-user-extensions\outputs'
python -m pip install -r paper_reproductions/mohamad2020/requirements.txt
python -m unittest paper_reproductions.mohamad2020.validation.test_solver paper_reproductions.mohamad2020.validation.test_materials -v
python -m paper_reproductions.mohamad2020.reproduce --study smoke --output-dir paper_reproductions/mohamad2020/results/smoke
```

まず少数の次数で動作と曲線の傾向を確認します。`quick` は計算量を減らした設定であり、
論文の数値条件ではありません。

```powershell
python -m paper_reproductions.mohamad2020.reproduce --study fig5 --profile quick --output-dir paper_reproductions/mohamad2020/results/fig5_quick
python -m paper_reproductions.mohamad2020.reproduce --study all --profile quick --output-dir paper_reproductions/mohamad2020/results/all_quick
```

図5の論文向け設定は次です。`N` は打ち切り最大次数で、実際のフーリエ成分数は `2*N+1` です。
`2:100:2` は端点を含む `2,4,...,100` を意味します。

```powershell
python -m paper_reproductions.mohamad2020.reproduce --study fig5 --profile paper --orders 2:100:2 --section-nm 1 --dm-substeps 4 --nfft 4096 --include-nfft1024 --output-dir paper_reproductions/mohamad2020/results/fig5_paper
```

図6、9a、9b、10aを個別に実行する場合:

```powershell
python -m paper_reproductions.mohamad2020.reproduce --study fig6 --profile paper --output-dir paper_reproductions/mohamad2020/results/fig6_paper
python -m paper_reproductions.mohamad2020.reproduce --study fig9a --profile paper --output-dir paper_reproductions/mohamad2020/results/fig9a_paper
python -m paper_reproductions.mohamad2020.reproduce --study fig9b --profile paper --output-dir paper_reproductions/mohamad2020/results/fig9b_paper
python -m paper_reproductions.mohamad2020.reproduce --study fig10a --profile paper --output-dir paper_reproductions/mohamad2020/results/fig10a_paper
```

図10bを含めて全studyを一括計算する場合:

```powershell
python -m paper_reproductions.mohamad2020.reproduce --study all --profile paper --output-dir paper_reproductions/mohamad2020/results/paper
```

`paper` の高次数掃引は、密行列演算と多数のz断面を使うため長時間かかります。
まず `--orders 10,20,30` のような限定した次数で実行できます。
各点が終わるたびにCSVと設定JSONを更新しますが、同じ出力先の再実行は上書きです。
同時実行には別々の `--output-dir` を指定してください。

モジュール起動以外に、作業ディレクトリに依存しない直接起動もできます。

```powershell
python 'C:\Users\tsoyu\Documents\Codex\2026-08-06\files-mentioned-by-the-user-extensions\outputs\paper_reproductions\mohamad2020\reproduce.py' --study fig5 --profile quick
```

## 図と計算条件

| `--study` | 計算内容 |
|---|---|
| `smoke` | 正弦波格子、3手法、N=2、粗いz分割での実行確認 |
| `fig5` | λ=632.8 nm、θ=40°、3手法のR−1と参照値に対する相対誤差 |
| `fig6` | 同格子の各手法の電場絶対値。paperではN=50、quickではN=8 |
| `fig9a` | λ=810 nm、θ=11°、GILなし、DM-FFFの積分刻みDz=1,0.5,0.25 nm |
| `fig9b` | λ=810 nm、θ=11°、DM-FFF、GIL厚さδ=0,1,5,7,10 nm |
| `fig10a` | λ=810 nm、θ=15°、DM-FFFのδ=0,1、RCWA-FFFのδ=1、RCWAのδ=0 |
| `fig10b` | θ=15°、DM-FFFのδ=0,0.3 nm、既存の共通Au CSVによるT0波長掃引 |
| `all` | 図5・6・9a・9b・10a・10bを実行。CSVの追加指定は不要 |

図5の参照値は論文記載の `Rref(-1)=0.8785309971543` です。
相対誤差は `abs(R(-1)-Rref)/Rref` として計算し、C-Methodの数値を推定していません。
正弦波格子の周期500 nm、高さ200 nm、Alの屈折率1.3+7.6i、入射側は空気です。
金格子は周期400 nm、基板屈折率1.55、空気側へ出射し、810 nmで金の屈折率0.16+5.26iです。
`paper` の既定次数は2,4,...,100に、論文が離散的発散を報告したN=95を加えます。
`--orders` を明示した場合は、その指定だけを使用します。

`--section-nm` は一つのS行列を構成する厚さです。DM-FFFはその中を
`--dm-substeps`（既定4）に分けて積分します。したがって、section=1 nmなら
DMの積分刻みは0.25 nm、RCWAの断面間隔は1 nmです。図9aは論文のDzを積分刻みと解釈し、
Dzに合わせてsectionを4,2,1 nmに変更します。CSVには両方の厚さを記録します。
`paper` の図9b・10a・10bではDM-FFFのsection=dm_substeps nm、積分刻みDz=1 nmです。
既定のdm_substeps=4ならsection=4 nmです。
図10aのRCWA・RCWA-FFFはsection=1 nmです。図5・6は全手法でsection=1 nmです。
明示した `--section-nm` はこれらの図別既定値を上書きします。図9aを上書きした場合は、
指定sectionに対応するDzの1曲線だけを計算します。
z方向の材料評価点は既定 `--sampling entrance` で、`midpoint` への変更は別設定として記録します。

## 図10b: 既存の共通Au CSVを使用

既定で `studies/gold_motheye2/data/au_measured_nk.csv` を直接読みます。
新しいCSVの用意やコピーは不要です。共通既定パスは
`studies/shared/gold_dispersion.py` の `DEFAULT_GOLD_CSV` で定義しました。
このCSVは `gold_motheye3`、`gold_motheye_nv` の既存CSVと同じSHA-256です。
データ範囲は187.9–1937 nmで、既定の400–850 nmの掃引を含みます。

読み込みと補間には、他のAu計算で使用している
`studies.shared.gold_dispersion.load_gold_csv` をそのまま使用します。
各データ点で `epsilon=(n+i*k)**2` としてから波長に対して誘電率を線形補間し、
その受動系の平方根をGILの屈折率へ渡します。範囲外へは外挿しません。
前版にあったこの再現実装独自のn,k補間処理は削除しました。

```powershell
python -m paper_reproductions.mohamad2020.reproduce --study fig10b --profile paper --orders 50 --output-dir paper_reproductions/mohamad2020/results/fig10b_paper
```

別の材料表を使う場合だけ `--gold-csv PATH`（従来名 `--gold-data PATH` も有効）を指定します。
共通ローダーが扱う `wavelength_nm,n,k` と
`wavelength_nm,epsilon_real,epsilon_imag` の形式に対応します。

図9・10aの単一波長計算は、論文指定条件を維持して810 nmで `n=0.16+5.26i` を使います。
図10bは既存CSVの分散モデルを使います。CSVの810 nmの値は約 `0.15677+4.99490i` なので、
両者の材料条件は同一ではありません。論文の分散表が明記されていない点は残りますが、
既存プロジェクトと同じ材料モデルでスペクトルを計算できます。

## 保存内容と判定

各図のCSVには、R−1、R0、T0、全次数のR・T、吸収A、計算時間、設定、状態を保存します。
PNGは計算した値のプロットです。`settings.json` に実行日時、コマンド、Python・依存パッケージの
バージョン、ソースファイルのSHA-256、形状、各計算の詳細を保存します。
元PDFがユーザー指定の場所に存在すれば、そのSHA-256も保存します。
図6には再解析用の複素フーリエ係数と絶対値マップをNPZで保存します。

保存済みの実行・精度確認は [検証結果](results/ASSESSMENT_ja.md) を参照してください。
`results/all_shared_gold_quick/` は図10bを含む全studyの出力確認、
`results/all_quick/` は前版の5studyの記録です。`results/fig5_verified/` は論文のz分割・FFT条件で
N=20,30,50を計算した結果です。高次数の全掃引は未実行です。

数値線形代数の例外・警告・非有限値は `FAILED` としてNaNと原因を保存します。
有限値でも受動系のエネルギー収支を破る点は `UNPHYSICAL` として値を保存し、そのまま描きます。
発散自体が図9・10で検討されているため、R0>1の値を1に切り詰めたり隠したりしません。
`FAILED` があるとCLI終了コードは1です。`UNPHYSICAL` のみの場合は終了コード0ですが、
値の収束・妥当性を確認してください。

BLASの並列数は、環境変数に指定がない場合に1へ設定します。既存の環境設定を優先します。
使用した値はJSONに保存します。GPUバックエンドはありません。

## 再現の範囲と制約

- 図8には283.2 nmの寸法が明記されており、ここでは三角波の谷から山までの高さと解釈しています。50 nmの金薄膜厚さをz方向の厚さと解釈しています。法線方向一定厚さのコーティングとは異なり、この解釈による系統差があり得ます。
- GILは各z断面の界面に沿うx方向の厚さδとして扱い、屈折率を線形補間した後で `epsilon=n**2` を計算します。εそのものの線形補間や法線方向の一定厚さとは異なります。
- 外部MC-GratingによるC-Methodの比較曲線は含めません。図10bには既存プロジェクトのAu CSVを採用しており、論文で使われた分散表と同一であることは確認できていません。
- 図6は計算領域内のS行列境界におけるExを直接フーリエ再合成します。界面付近のGibbs振動とz分割への依存が残るため、論文画像との画素単位の一致を保証しません。表示上端は高さ200 nm、下端は0 nmで、論文図の共通高さオフセットは省略します。
- `quick` は動作確認専用です。`paper` でもN、z刻み、FFT点数、形状解釈を個別に確認し、収束した値を評価してください。
- DMでは標準的な4次Runge–Kuttaを積分刻み `section_nm/dm_substeps` で実行します。論文のns=4という記述には内部の評価点・実装の全詳細がないため、原著コードと完全に同じ段階処理であるとは保証しません。
