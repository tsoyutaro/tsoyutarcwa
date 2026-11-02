# Schuster (2007) 図10と正則化ON/OFFの比較

対象は Schuster et al., JOSA A 24, 2880-2890 (2007), Figure 10。
DOI: https://doi.org/10.1364/JOSAA.24.002880
Götz (2008)の別論文とは区別する。

## 再現する条件

- 正方格子。周期x/y = 1000 nm。
- 金属薄膜に直径500 nmの円形空気孔。膜厚50 nmの単一構造層。
- 上部は空気、下部は屈折率1.5の半無限ガラス基板。
- 金属の屈折率は1.75+1.5i、したがって誘電率は0.8125+5.25i。
- 真空波長500 nm、垂直入射。
- 比較する量は反射ゼロ次回折効率R00。全反射率total_Rとは異なる。
- x偏光[1,0]と、正規化した対角偏光[1,1]/sqrt(2)。
- 両方向に-M...Mの矩形Fourier集合、M=4...13。
- Lanczosなし、ASRなし、形状・材質の追加フィルターなし。

正方格子の円形孔の対称性により、R00は上記2偏光で同じになる。計算本体はx偏光のC2v source sectorを使用し、M4/M13で全偏光S行列からx/y/対角偏光を直接求めて検証する。対角偏光の線はこの対称性を使って表示する。

## 本番実装との関係

本番Al用フォルダのON/OFFの`rcwa_ext`を独立した名前空間で読み込む。材質は本図の指定値へ置き換える。半径の最適化は行わない。

ASTでNVMクラスの差が`_projection_matrix`だけであることと、config/symmetry/reduced/scatteringモジュールが同じであることを検査する。本番のソースや最適化結果は変更しない。ソースSHA256は`manifest.json`へ記録する。

ONは中心付近と最近接中心が切り替わる境界付近のsmoothstep正則化を使用する。OFFは最近接中心からのradial projectorを使用し、中心点はI/2、最近接中心の等距離点はprojectorを平均する。OFFでも補助法線場およびNVMの因子分解を使用する。

主比較は両方とも現在の対称化式`eps_ab = delta_ab E + (D N_ab + N_ab D)/2`を用いる。論文の式(7)は`eps_ab = delta_ab E + D N_ab`であり、有限次数で同じ式ではない。`--paper-equation-check`はOFFに限り式(7)を追加で評価し、主比較と分けて保存する。

円の物質係数は解析Bessel係数、補助法線projectorの係数はFFTで構築する。論文図10の元の数値表・計算gridは入手できておらず、原実装との完全一致は保証しない。

## 図から読み取った参照値

`digitize.py`は添付PDF内の元サイズ850x1523のFig.10 JPEGから、実線「NV radial」だけを読み取る。円マーカーの「NV electrostatic」と混同しない。各次数間の線分の中央を直線fitし、マーカー近傍を避ける。

画素座標、軸の校正、線分fit残差、元画像SHA256を保存する。`digitization_overlay.png`で選択した曲線を確認できる。読取り誤差の目安は±2.5画素、R00で±4.44e-5。これは統計的な信頼区間や厳密解の誤差ではない。

図からのMAEが小さいことは、厳密解に近いことや高次数収束の証明ではない。参照値をこの読取幅内で動かしたときにON/OFFのMAE順位が変わるかも評価する。

## 実行方法

リポジトリのルート（Optimizationの一つ上）で実行する。

```bash
python paper_reproductions/schuster2007_nv_fig10/reproduce.py --device cuda:0 --orders 4:13 --grids 512,1024 --validate --paper-equation-check --output paper_reproductions/schuster2007_nv_fig10/results/cuda
```

CPUの場合は`--device cpu`。PyTorchとtorcwaが必要。CUDA環境では実際にCUDAのcomplex128を使用する。計算済みの同じソース・device・PyTorch・threads・条件の結果は再利用する。別deviceや環境の時間を測る場合は別の`--output`を使用する。

図の再作成は、配布フォルダのrootを指定して以下を実行する。matplotlib/numpyが必要。

```bash
python plot_comparison.py .
```

配布ZIPは`source_snapshot/ON/rcwa_ext`と`source_snapshot/OFF/rcwa_ext`も含む。元リポジトリから独立して計算する場合は次のように指定する。

```bash
python reproduce.py --source-root source_snapshot --device cuda:0 --orders 4:13 --grids 512,1024 --validate --output results_cuda
```

## 数値検証

grid512/1024を全次数で比較し、M4/8/13ではgrid2048も比較する。非物理値は補正・クリップせず、そのまま診断する。

波長500 nmは空気側の(±2,0)/(0,±2)次数がちょうどRayleigh cutoffになる。ここではポート行列の数値条件が悪いため、full/source-sectorのR00差を1e-7以下で検証し、厳しい1e-9での成否も別に記録する。実測差は最大9.43e-9であり、図の読取幅より3桁以上小さい。初回の厳しいチェック結果は`validation_initial_strict.json`に残す。

さらに500*(1±1e-8) nmでR00の感度を確認し、cutoffから離れた501 nmではfull/source-sectorを1e-11で検証する。grazing次数の法線パワーは0なので、全R/Tの診断ではその次数を除外し、ゼロ除算を避ける。

このベンチマークでの一致を、Al/Ti/Cu/Au/Agの別形状・別波長での正則化方式の優劣へそのまま一般化しない。
