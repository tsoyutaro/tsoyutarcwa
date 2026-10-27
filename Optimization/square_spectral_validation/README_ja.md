# 正方格子：行列のブロック構築と波長間共有の検証

正方格子、法線入射、x偏光、単位胞中心の円柱、非吸収性材料を対象とする独立した検証パッケージ。既存の43条件や実行中の最適化は変更しない。

`benchmark_square.py` は次の3経路を、同じ形状・材料・波長・Fourier基底・NVM設定で比較する。

1. `reference`：凍結した従来の本番ソルバー。各波長で形状と材料行列を構築し、平均反射率の半径勾配を加算する。
2. `block`：形状Fourier係数と法線射影行列の生成は従来と同じ。誘電率行列の逆行列をC2vの4ブロックで計算し、全サイズの逆行列やP・Qを再構成せず、光源に対応する部分空間のP・Qを直接構築する。
3. `shared`：上記に波長間共有を追加。各波長の固有値・伝播・散乱計算は継続する。波長ごとの随伴勾配を共有テンソル上に加算し、形状から半径へのVJPを最後に1回計算する。

`block_assembly.py` は先の検証パッケージから凍結コピーしたもの。正方格子の経路のみを今回のコードから呼び出す。C4vへ変更した実験ではない。次数8ではscalar Fourier数289、光源に対応する固有値問題145次、scalar逆行列ブロックは81・72・72・64次となる。

`block` と `shared` は縮約空間内で電磁場モードを作るため、従来経路に残っている全空間へのモード再構成や、`fields=none` では不要な場の回復用格子データ・全サイズ行列の保存も省く。したがって全体時間の差には、逆行列・P/Q構築の変更に加えて、この付随処理の削減も含まれる。逆行列・P/Q構築だけの効果は別に測定する。

## 材料ごとの共有内容

一定屈折率では、法線入射の規格化波数を用いて `P(lambda)=P0+(lambda/period)^2 P2`、`Q(lambda)=Q0+(lambda/period)^2 Q2` と分ける。各層のP2・Q0とその半径への微分を共有する。P0・Q2は半径に依存しない。

PMMAでは `data/Szczurowski.csv` の屈折率を波長で線形補間する。400 nmだけは従来の方針と同じく最初の区間を外挿する。消衰係数k=0を仮定する。円柱部分と基板の両方に同じn(lambda)を設定する。

PMMAで共有するのは、円形のFourier形状行列Fと、その形状の法線射影行列だけ。各波長で `epsilon=I+(n(lambda)^2-1)F` と inverse-rule の畳み込み行列 `I+(1/n(lambda)^2-1)F` を生成し、それぞれのブロック逆行列とP・Qを再計算する。材料の逆行列やP・Qを波長間で使い回さない。

## 実行方法（PowerShell）

このフォルダへ移動し、torch・numpy・torcwaが利用できるPythonで実行する。Optunaは不要。CPU実装であり、CUDAでの速度は未測定。

小規模で一定屈折率とPMMAの値・勾配・有限差分を確認：

```powershell
python benchmark_square.py --order 2 --grid 48 --layers 5 --material n1.5 --wavelengths 400,550,700 --check-fd --output results/n15_small.json
python benchmark_square.py --order 2 --grid 48 --layers 5 --material n2.0 --wavelengths 400,550,700 --check-fd --output results/n20_small.json
python benchmark_square.py --order 2 --grid 48 --layers 5 --material n2.5 --wavelengths 400,550,700 --check-fd --output results/n25_small.json
python benchmark_square.py --order 2 --grid 48 --layers 5 --material PMMA --wavelengths 400,550,700 --check-fd --output results/pmma_small.json
```

次数8・15層・NVM grid 512・31波長の比較：

```powershell
python benchmark_square.py --order 8 --grid 512 --layers 15 --material n2.5 --wavelengths 400:700:10 --output results/n25_M8_L15_31.json
python benchmark_square.py --order 8 --grid 512 --layers 15 --material PMMA --wavelengths 400:700:10 --output results/pmma_M8_L15_31.json
```

長時間の測定では、従来版・ブロック版の進捗・実CPU時間を波長ごとに保存する次のドライバーも利用できる。同じコマンドを再実行すると、ソースと設定が一致するチェックポイントから再開する。同じPC・Python/PyTorch環境で再開し、実行環境を変える場合は新しい出力先を使う。光学計算は上記と同じ `benchmark_square.py` の関数を呼ぶ。

```powershell
python benchmark_resumable.py --order 8 --grid 512 --layers 15 --material PMMA --wavelengths 400:700:10 --output results/pmma_progress.json
```

今回のPMMA・次数8・31波長の実測には、このドライバーを使用する。従来版とブロック版は各波長のソルバー経過時間を合計し、進捗表示・チェックポイント書き込みをその合計から除く。各波長の半径勾配を最後に平均する。共有版は全波長と形状VJPをまとめて計測する。`経過時間−実CPU時間 > max(10秒, 実CPU時間の10%)` となる大きな中断・CPU待ちを検出した試行は記録に残して再測定する。3回続く場合はチェックポイントを保存して停止する。中断原因をOSの休止と断定する処理ではない。

550 nmの1波長を単独で評価する比較（3回測定、有限差分を含む）：

```powershell
python benchmark_square.py --order 8 --grid 512 --layers 15 --material n2.5 --wavelengths 550 --repeats 3 --check-fd --output results/n25_M8_L15_single.json
python benchmark_square.py --order 8 --grid 512 --layers 15 --material PMMA --wavelengths 550 --repeats 3 --check-fd --output results/pmma_M8_L15_single.json
```

`--repeats` を増やすと3経路の実行順を交互にし、各経路の中央値を報告する。既定は1回測定。`--check-fd` は全層の半径を同時に変える方向微分の中央差分を確認する。

## 検証と時間の定義

- P・Q・PQの相対誤差、反射率R・透過率T、全層の半径勾配、R+T=1と値域を検証する。
- PMMAの複数波長では、誤ってn(550 nm)を固定した場合の差が検出されることも確認する。
- 時間はforwardと平均反射率の全層半径勾配を含む。Optuna・Adam全体の時間ではない。
- 「1波長当たり平均」は、**実測した全波長の時間を波長数で割った値**。1波長を単独実行した所要時間と区別する。
- 共有経路は境界行列を事前生成する。報告にはその準備時間、準備込みの速度比、再利用時の速度比を併記する。従来版とブロック版は各波長の初期化を時間に含む。
- 共通の対称性基底、Python起動、PyTorchの初回VJP初期化は除く。
- 行列構築だけの時間は1層分・forwardのみで、形状入力生成・固有値計算・backwardを含まない。構築回数を15倍した推定を全体時間として扱わない。
- 形状は規格化半径0.10～0.39を層数に応じて等間隔配置。周期200 nm、高さ500 nm。各層の厚さは等しい。
- 本番のsoftmaxパラメータ、checkpoint、再開・監査、CUDAへの接続はこの検証の対象外。

実測結果は `検証結果_ja.md` と `verified_results/` に収録する。JSONには実行環境、設定、全スペクトル、全勾配、誤差、時間、ソースとPMMAデータのSHA256を記録する。
