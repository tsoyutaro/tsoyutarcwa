# MOSMITE：解析的な混合コヒーレント／インコヒーレント計算

2026-11-09更新。対象は未蒸着MOSMITE、および金の真空蒸着・スパッタ成膜モデルのPET・粘着層・ガラスです。PMMA芯＋金属被覆の最適化は、従来どおり半無限PMMAへ接続したRCWAであり、本変更の対象ではありません。

## 方法とコード

解析的な強度計算の共有実装は `studies/mosmite_optics.py` です。未蒸着・真空蒸着・スパッタの位相平均は `studies/mosmite_phase_average.py` を共用します。

- モスアイ・金被覆の局所構造：従来と同じコヒーレントRCWA。
- PET・粘着層・ガラス：Fresnelの電力係数と単一通過吸収を使う解析的な強度伝達。
- 無限回の往復反射：強度の連立方程式を解いて合算。
- 既定の解析計算は各波長で強度収支を直接解く。位相平均を選んだ場合は各波長で等間隔の位相点を使う。
- 実測比較：`compare_pointwise.py`で各計算波長の実測値を直接使用。移動窓・補間・正規化倍率は加えない。

## スカラーの場合の理論式

層jの単一通過強度透過率は、正入射で

\[
P_j=\exp\left(-\frac{4\pi k_jd_j}{\lambda}\right)
\]

です。左右の部分系を1、2とし、順方向・逆方向をf、bとします。その間の層を往復する反射の公比は

\[
q=R_1^b R_2^f P_j^2
\]

であり、全多重反射を等比級数として足すと

\[
T=\frac{T_1^f P_j T_2^f}{1-R_1^bR_2^fP_j^2},\qquad
R=R_1^f+\frac{T_1^fT_1^bR_2^fP_j^2}{1-R_1^bR_2^fP_j^2}.
\]

複数層でも同じ関係を逐次接続します。実装では転送行列の指数増大を避けるため、散乱係数の連立方程式を解く形にしています。

複素屈折率の界面では、正入射の電場振幅tから

\[
T_{ij}=\frac{\operatorname{Re}n_j}{\operatorname{Re}n_i}|t_{ij}|^2,\qquad R_{ij}=|r_{ij}|^2
\]

を求めます。吸収のある入射側の内部ポートでは、単一界面のR+T=1を強制しません。Byrnesの付録Bで説明されている電力規約です。試料の入射側・出射側はいずれも空気で、試料全体の吸収はA=1-R-Tです。層ごとの吸収積分をこの更新で追加したわけではありません。

金モデルの2×2 Jones係数では、偏光間のコヒーレンシー行列Cを保ちます。各ブロックの作用はC→J C J†、列ベクトル化するとK(J)=J*⊗Jです。往復反射を(I−K_b K_r)^−1で解きます。異なる往復経路間の干渉を捨てても、同一ビーム内の偏光コヒーレンスを勝手に消さないためです。D6・正入射ではスカラー式へ帰着します。

## 対応文献

1. **Steven J. Byrnes, “Multilayer optical calculations,” arXiv:1603.02720.**
   [公開全文6.2節](https://arxiv.org/html/1603.02720v5#S6.SS2)。式(25)が強度の接続、式(26)が単一通過吸収、式(27)–(30)が多層の接続とR,Tの計算です。現在のスカラー実装に直接対応し、独立した2×2強度伝達行列との数値照合も行いました。吸収性内部ポートの規約は付録B参照。
2. **E. Centurioni, “Generalized matrix method for calculation of internal light energy flux in mixed coherent and incoherent multilayers,” Applied Optics 44,7532–7539(2005).**
   [DOI:10.1364/AO.44.007532](https://doi.org/10.1364/AO.44.007532)。コヒーレント／インコヒーレント層が混在する系を、エネルギーフラックスに基づいて扱う査読論文です。
3. **C. C. Katsidis and D. I. Siapkas, “General transfer-matrix method for optical multilayer systems with coherent, partially coherent, and incoherent interference,” Applied Optics 41,3978–3987(2002).**
   [DOI:10.1364/AO.41.003978](https://doi.org/10.1364/AO.41.003978)。コヒーレント薄膜と非干渉厚膜を含む一般化多層膜法を記述しています。ここで部分コヒーレンスの全機能を実装したという意味ではありません。

## 非干渉とする層の指定

- `--coherence incoherent`：各一様層を非干渉とする。既定。
- `--coherence incoherent --coherent-layers PET,adhesive`：PETと粘着層は固定位相のコヒーレント計算、ガラスを非干渉とする。
- `--coherence coherent`：すべての層を固定位相で計算。
- `--coherence phase-average --phase-points 16`：各一様層の往復位相を16点ずつ等間隔に取り、全組合せの反射率・透過率を平均。`--phase-points` の既定値は16。1点では公称厚さの固定位相を使う。
- `--coherence phase-average --coherent-layers PET,adhesive`：PETと粘着層は固定位相とし、存在するガラス層の位相を平均。

位相平均で2点以上を使う場合、位相点数を各層で倍にして再計算し、差の最大値を `spectra.csv` の `phase_average_error`（0〜1の分率）に記録します。差が `--convergence-tol` を超えると計算を停止します。

この方法も非干渉の仮定の下での理論計算です。厚いという理由だけで非干渉性が確定するわけではありません。PET5 µmや粘着層20 µmの干渉縞を再現する場合は、その層をコヒーレントに指定してください。装置の1 nmバンド幅との畳み込みは今回の処理に含めていません。

## 未蒸着・ガラスなしの実行例

ZIPを既存の研究ルートに展開します。研究ルート直下から：

```bash
python studies/mosmite_uncoated/run_spectrum.py \
  --device cuda --wavelengths "400:1000:5" \
  --pet-nk-csv studies/mosmite_uncoated/data/PET_Zhang_400_1000nm.csv \
  --adhesive-nk-csv studies/mosmite_uncoated/data/adhesive_reference_Cauchy_20261027.csv \
  --period-nm 100 --height-nm 200 \
  --base-diameter-nm 98 --tip-diameter-nm 50 --profile-power 1 \
  --film-total-um 5.2 --residual-um 0 \
  --adhesive-um 20 --glass-um 0 \
  --method nvm --order 3 --slices 50 --grid 128 \
  --coherence incoherent \
  --output-dir studies/mosmite_uncoated/results/no_glass/analytic_incoherent_v1
```

ここでのPETは一様層5 µm、突起200 nmを含むCLI総厚が5.2 µmです。粘着層のCauchy表は以前のガラスなし測定に対する無吸収フィットを固定した参照で、実材料の測定値ではありません。この更新で材料定数を再フィットしていません。従来のn=1.50,k=0を使う場合はadhesive-nk-csvを外し、`--adhesive-n 1.50 --adhesive-k 0`を指定します。

ガラスありは `--glass-um 1000`へ変更し、別の出力先を指定します。各層の干渉も保持する混合モデルなら `--coherent-layers PET,adhesive`を追加します。数値収束の確認には、従来どおり`--check-convergence --check-wavelengths "400,600,800,1000" --convergence-tol 0.005`などを指定します。上のM3等は従来の比較条件で、今回その空間離散化を再最適化したわけではありません。

## 金蒸着・スパッタ

以下は解析的強度計算の実行例です。従来の位相平均で計算する場合は `--coherence phase-average --phase-points 16` に切り替えます。手法を比較する場合はそれぞれ別の出力先を指定します。

```bash
python studies/mosmite_gold_deposition/run_evaporation.py \
  --device cuda --wavelengths "400,600,800" \
  --slices 150 --order 32 --grid 512 \
  --layer-storage reduced --coherence incoherent \
  --output-dir studies/mosmite_gold_deposition/results/evaporation/analytic_incoherent_v1
```

スパッタは`run_sputter.py`を使用します。金モデルの既定PET一様厚は54.8 µm、ガラスは1 mmで、今回これらの形状条件は変更していません。

## 移動平均なしの実測比較

```bash
python studies/mosmite_uncoated/compare_pointwise.py \
  --spectrum studies/mosmite_uncoated/results/no_glass/analytic_incoherent_v1/spectra.csv \
  --measurement /path/to/measurement.csv \
  --measurement-column T_grand_mean_percent \
  --output-dir studies/mosmite_uncoated/results/no_glass/analytic_incoherent_v1/pointwise
```

比較する実測は空気基準の透過率を想定します。ガラス基準の場合は、シミュレーションも試料T/裸ガラスTに規格化してから比較する必要があります。

JASCOのXYDATA原CSV、または`wavelength_nm`と透過率[%]列を持つCSVを読めます。JASCOでは列指定は不要です。処理済みCSVでは列名を指定してください。計算波長が実測の点に存在しない場合はエラーにし、補間・平均を自動で追加しません。比較のRMSEは各波長の残差から求めます。この誤差指標や実測の試料平均は、スペクトル平滑化とは別です。

以前の`work/no_glass/fit_wideband_20261027.py`などと過去の25/50 nm区間平均結果は、過去解析として保持しています。今回の新しい計算・比較経路では使用しません。

## 検証と出力の範囲

- 単層の無吸収・吸収を含む非干渉閉形式、Airy式との一致。
- 複数の複素屈折率層について、独立実装したByrnes式(25)–(30)の2×2強度伝達行列と一致。
- コヒーレントAR膜＋非干渉基板、薄いコヒーレント空隙、Jones交差偏光、強吸収極限。
- 位相配列生成やサンプル平均を呼ばないこと、波長比較で移動平均や補間をしないこと。
- 未蒸着MOSMITEのRCWA・Fresnel・残膜・x/y検証。
- 金の蒸着・スパッタ、合体した金属断面、lossless制御を含む26項目のCPU検証。

同じ2026-10-27の前面RCWA係数を固定した4波長・ガラス有無の比較では、旧位相16点法との差は最大約0.0000554ポイントでした。これは厚層計算法のみの比較です。現在のrcwa_extは当時からauto.py/config.py/nvm.pyが更新されているため、旧RCWAキャッシュを現在のソルバーと同一だとみなす比較には使っていません。

実行済みのCLI確認はCPUの小規模条件です。CUDAでの全波長・高次数計算は実行していません。更新後は新しい出力先を使い、settings.jsonのcoherence・理論出典・ソースSHA256を確認してください。
