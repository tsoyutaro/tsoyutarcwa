# 未被覆MOSMITEの反射・透過スペクトル

20260806の共有`rcwa_ext`を利用する。表面の微細構造をRCWAで解き、有限厚の
基材・粘着剤・石英ガラスを電場振幅の散乱行列として接続する。
空気から入射し、ガラス裏面から空気へ出る試料全体のR、T、Aを計算する。
金属膜はない。実測へのフィッティングは行わない。

## 配置

この`mosmite_uncoated`フォルダを、20260806の`studies/`直下へ配置する。
`rcwa_ext/`と`studies/`は同じ階層に置く。既存Python環境のnumpy、torch、torcwa、
matplotlibを利用する。公開ソルバーのコードは変更しない。

## コマンド（bash / TSUBAME）

`rcwa_ext/`が存在するプロジェクト直下で実行する。

```bash
# 数値設定の確認。400,600,800 nmでM、Nz、gridを順に上げて比較する。
python studies/mosmite_uncoated/run_spectrum.py \
  --device cuda --wavelengths 400,600,800 \
  --order 7 --slices 100 --grid 256 \
  --check-convergence --convergence-tol 0.0001 \
  --output-dir studies/mosmite_uncoated/results/convergence

# スペクトル本計算。上の確認が不合格なら次数・層数・gridを上げてから実行する。
python studies/mosmite_uncoated/run_spectrum.py \
  --device cuda --wavelengths 400:800:5 \
  --period-nm 100 --height-nm 200 \
  --base-diameter-nm 98 --tip-diameter-nm 50 --profile-power 1 \
  --film-total-um 55 --residual-um 0 --adhesive-um 20 --glass-um 1000 \
  --adhesive-n 1.50 --adhesive-k 0 \
  --method nvm --order 7 --slices 100 --grid 256 \
  --coherence phase-average --phase-points 16 \
  --output-dir studies/mosmite_uncoated/results/reference
```

CPUでは`--device cpu`を使う。PowerShellでは行末の`\`をバッククォートへ変更するか、
各コマンドを一行にまとめる。起動場所が別でも`--solver-root`でソルバー直下を指定できる。

`--check-convergence`は400,600,800 nmのうち指定範囲に入る波長について、
`(M,Nz,grid)=(7,100,256),(9,100,256),(9,200,256),(9,200,512)`を比較する。
隣接する計算のR/T/Aの最大差、または各比較値と開始条件の差が
1e-4（0.01パーセントポイント）を超えると、
CSVを保存したうえでエラー終了する。これは3波長での確認であり、全波長での収束保証ではない。
`--check-wavelengths 400,450,600,800`や`--check-wavelengths 400:800:5`で検査波長を変更できる。
さらに厳密な定量比較が必要なら全スペクトルを高精度設定でも計算して比較する。
位相平均は各波長で16点/層と32点/層を比較し、同じ許容差を超えると停止する。

## ガラスを外した構成（2026-10-23追加）

更新版は`--glass-um 0`に対応する。厚さ0のガラスは、界面の接続にも位相平均にも入れない。
最後に残る層と空気の界面を接続する。従来版は0を拒否するため、更新したrun_spectrum.pyを使用する。
金なし、空気→樹脂モスアイ200 nm→PET相当54.8 um→粘着20 um→空気が対象。
周期100 nm、底面直径98 nm、先端直径50 nm、フィルム全厚55 um、残膜はPET相当層へ統合。
PETと表面樹脂は同梱PET分散、粘着層はn=1.50,k=0の仮定を維持する。

形状を変えず、必要な回折次数・高さ分割・面内グリッドを調べる収束検査を行う。
以下はM3、Nz50、grid128を開始候補とし、
(3,50,128)→(5,50,128)→(5,100,128)→(5,100,256)を比較する。
開始条件と各比較条件のR/T/A、および隣接段階のR/T/Aを検査する。
全ての組合せを探索して世界最小の計算量を保証する最適化ではなく、軽い候補の十分性を確認する。

許容差0.005は**絶対差0.5パーセントポイント**であり、計算値に対する相対0.5%ではない。
0.01パーセントポイントで評価したい場合は、両コマンドで`--convergence-tol 0.0001`を使う。
これらは測定精度や識別したい差に応じて選ぶ基準であり、RCWAの普遍的な必須値ではない。

```bash
python studies/mosmite_uncoated/run_spectrum.py \
  --device cuda --wavelengths "400,600,800" \
  --glass-um 0 --adhesive-um 20 \
  --method nvm --order 3 --slices 50 --grid 128 \
  --coherence phase-average --phase-points 16 \
  --check-convergence --convergence-tol 0.005 \
  --output-dir studies/mosmite_uncoated/results/no_glass/convergence &&
python studies/mosmite_uncoated/run_spectrum.py \
  --device cuda --wavelengths "400:800:5" \
  --glass-um 0 --adhesive-um 20 \
  --method nvm --order 3 --slices 50 --grid 128 \
  --coherence phase-average --phase-points 16 \
  --convergence-tol 0.005 \
  --output-dir studies/mosmite_uncoated/results/no_glass/reference
```

`&&`により、収束確認が成功した場合だけ400～800 nm・5 nm刻み81点を計算する。
結果は`results/no_glass/reference/spectra.csv`と`spectra.png`。
検査結果は`results/no_glass/convergence/convergence.csv`と`settings.json`のconvergence_passedへ保存する。
3波長だけの検査は、全81波長の収束を認定するものではない。
全域を確認したい場合は、最初のコマンドの両方の波長指定を
`--wavelengths "400:800:5" --check-wavelengths "400:800:5"`へ置き換える。
不合格の場合は基準を確認し、例えば開始条件をM5、Nz100、grid256へ上げて同じ検査を再実行する。
本計算も、合格した開始条件のM/Nz/gridに合わせる。

粘着層も除き、PET相当層の裏面を空気に接する構成を扱う場合は、
両コマンドの`--adhesive-um 20`を`--adhesive-um 0`へ変更し、保存先も別名へ変更する。
その場合も収束検査を行う。厚さ0の粘着層も位相平均から除外する。

```text
python studies/mosmite_uncoated/validate_no_glass.py
```

この補助検証は層の除外、無吸収膜の解析式、均一層RCWAとの一致、位相平均、
および粘着層の有無を確認する。`--legacy-script PATH`で旧版との光学結果の一致も確認できる。

## 材料分散と未確定値

PETの参照データはZhangら、Applied Optics 59,2337--2344(2020)、
DOI https://doi.org/10.1364/AO.383831 に対応する。
同梱CSVは原論文の添付ファイルから直接ダウンロードしたものではなく、
refractiveindex.infoデータベースの公開表（CC0）の400--800 nm部分を抽出した。
表は10 nm刻み、計算ではn,kを別々に線形補間し、範囲外には外挿しない。

公開表のURL：
https://github.com/polyanskiy/refractiveindex.info-database/blob/main/database/data/organic/%28C10H8O4%29n%20-%20polyethylene%20terephthalate/nk/Zhang.yml

| 波長 | PET n | PET k |
|---|---|---|
| 400 nm | 1.61027 | 2.31e-6 |
| 600 nm | 1.56811 | 1.29e-6 |
| 800 nm | 1.55533 | 7.56e-7 |

MOSMITE固有のPETや表面加工樹脂を測定したデータではない。
初期モデルでは、未同定の表面加工樹脂も同じPETの光学定数で代表する。
実物の加工樹脂がPETであるという主張ではない。PETの配向・複屈折は扱わない。
`--resin-nk-csv`で別の表面樹脂の分散表に変更できる。

石英ガラスは溶融石英としてMalitson(1965)のSellmeier式を使い、k=0とする。
DOI https://doi.org/10.1364/JOSA.55.001205 。結晶石英の異方性モデルではない。

粘着層はアクリル系微粘着20 umを参考にするが、n,kの分散は公開資料で未確認。
既定のn=1.50,k=0は波長一定の仮定であり、分散を測定した値ではない。
全材料について測定分散を入力したモデルではないことに注意する。
測定値を入手した場合は`--adhesive-nk-csv`で置換できる。
両CSVオプションは`wavelength_nm,n,k`または`wavelength_um,n,k`のヘッダーを受け付ける。
受動媒質n>0,k>=0、時間依存exp(-i omega t)、epsilon=(n+i k)^2を使用する。

## 厚さ55 umの扱い

SF-1000Nの資料では「ポリエステル系樹脂／モスマイト処理」の基材55 um、
アクリル系微粘着20 um、総厚75 umと記載される。
https://www.cuttingsheet.com/hpgen/pdf/catalog_sf1000n.pdf

55 umを加工済み基材全体の厚さと解釈する初期モデルを採用した。
高さ0.2 umを差し引き、既定の均一な光学層を54.8 umとする。
PET単体の厚さが54.8 umであると実測・確認したわけではない。
表面加工樹脂とPETの光学定数を同じに置くので、残膜をその均一層へ統合する。
`--residual-um 0`は実物に残膜がないという意味ではない。
別の樹脂分散と残膜厚を仮定する場合は、例えば
`--resin-nk-csv resin.csv --residual-um 5`とし、
PET相当厚は55-0.2-5=49.8 umとなる。55 umに5 umを上乗せしない。
この5 umは使用例であり、公開資料によって特定した残膜厚ではない。

## 光学計算の範囲

六方（三角）格子の基本ベクトルはa1=(100,0)nm、a2=(50,86.60254)nm。
単位胞内に円断面の突起1個。底面直径98 nm、先端直径50 nmの円錐台を
高さ方向100層の中点断面で近似する。寸法は初期仮定であり、SF-1000Nの形状測定値ではない。
正入射、各材料を等方媒質とするためD6対称性からx/y偏光のR,Tは等しい。
既定はx偏光のD6 source sectorを解き、無偏光の値としても使用できる。
M=7はnative D6-starの169調和波に対応する。矩形225調和波の計算とは異なる。
複素倍精度、NVM、Redheffer連結、内部電場保存なし、両方向のSブロックを使う。

表面構造内部はコヒーレントに計算する。55 um基材・20 um粘着層・1 mmガラスの
厚層干渉は、各層の往復位相を独立に平均する。
測定で細かい干渉縞が分解されない場合の初期モデルであり、装置分解能との照合が必要。
干渉縞を計算する場合は`--coherence coherent`に変更し、波長刻みも細かくする。
400--800 nmの範囲では非零回折次数は厚層中でエバネッセントなので、
十分離れた基材下面から後方の多層部分を零次の振幅で接続できる。
複素屈折率の内部ポートを介して強度だけを接続せず、電場振幅を接続後、
外部空気中のR=|r|^2,T=|t|^2を平均する。吸収と多重反射を含む。

ガラス裏面は未加工の空気界面を想定し、その反射も含む。
したがって全体Rはモスアイ面単独の反射率より大きくなり、数%になりうる。
散乱・ヘーズ・気泡・測定開口・角度分布は含まない。
R+T<1の差をすべて実物の吸収に帰属させるモデルではない。

## 出力

- `spectra.csv`：wavelength_nm、R、T、A（0--1）、位相平均の差、計算時間。
- `spectra.png`：反射率・透過率の2段グラフ（縦軸%）。
- `materials.csv`：各計算波長で実際に使用した各材料のn,k。
- `settings.json`：仮定、厚さ内訳、材料表とソルバーのSHA256、実行引数。
- `convergence.csv`：`--check-convergence`時の比較結果。

補助検証`python studies/mosmite_uncoated/validate.py`は、独立なAiry式、
無吸収膜の多重反射式、有限厚多層の均一RCWA、複素Fresnel振幅、
位相平均の刻み、x/y対称性、材料範囲外拒否を確認する。
これらの物理整合性チェックと、実形状のFourier次数・高さ分割の収束確認は別である。
