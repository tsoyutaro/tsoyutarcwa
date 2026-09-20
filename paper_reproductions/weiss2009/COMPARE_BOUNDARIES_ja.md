# 変換・境界接続の比較実験

ZIP 内のファイルを reproduce.py と同じディレクトリに配置してください。
コアソルバーは変更しません。diagnose_fig3.py の共通関数を利用します。

```bash
python compare_boundaries.py --orders 11,14,15 --grid 512 --identity-grid 1024 --device cuda --output-dir results/boundary_comparison
```

complex128、G=0.03（reproduce.py の実値を記録）、無損失誘電体の Fig.3 条件を使います。
各次数・profile で内部モードは一度だけ計算し、全方式で共有します。
接続比較を単純にするため、単層・真空入出射・垂直入射・Redheffer 固定です。
`--dtype`、`--cascade`、`--use-symmetry` はありません。
`identity` も matched 座標を使いますが、ASR の伸縮を行いません。

計算する方式：

| 名前 | 境界の内部場 | 境界の外部場 | 目的 |
|---|---|---|---|
| production | TW, TV | I, Vf | 現行ソルバーの基準値 |
| cartesian_control | TW, TV | I, Vf | 入射2偏光だけを解く比較コードの照合 |
| inverse_T_control | W, V | T^-1, T^-1 Vf | 有限行列として既存方式と同値な対照 |
| direct_pullback | W, V | B, B Vf | 外部平面波を共変成分へ直接展開する別の有限次数近似 |

B は Eu=x_u Ex+y_u Ey、Ev=x_v Ex+y_v Ey を計算座標の Fourier 基底へ投影して作ります。
物理座標の平面波 exp(i kx x(u,v)+i ky y(u,v)) を用い、FFT で積分します。
積分変数は du dv なので detJ を追加しません。T の逆行列を B として使っていません。
無限基底で対応する変換でも、有限次数では B と T^-1 は一般に一致しません。
この実験は Weiss 論文の外部媒質固有モード計算そのものを実装するものではありません。
direct_pullback がエネルギー保存を改善する保証もありません。

出力は一意の run_* ディレクトリ内の metadata.json とケース別 report.json です。
各方式の R/T/A、境界・伝搬残差、上下界面の外部電力流と内部共変電力流、条件数、
production との差を保存します。電力流は x,y の順に入射電力で規格化しています。
BT/TB と単位行列との差は投影の診断値であり、0でなければ実装誤りという意味ではありません。

結果の読み方：

- 最初に cartesian_control と production の R/T/A が一致することを確認します。
- inverse_T_control の差は数学的な手法差ではなく、悪条件下の演算経路の感度を示します。
- direct_pullback と対照の差から有限次数の境界投影の影響を調べます。
- direct_pullback の A が小さくても、それだけで正解としません。次数・格子収束、R/T、両偏光、境界残差も確認します。
- 大きな境界残差や対照間の差がある場合、投影法の優劣より先に数値安定性を検討します。

条件数計算を省略するには `--skip-conditions`、ASR だけなら `--profiles weiss2009` を指定します。
元の計算に追加のFFT・行列演算が必要なため、時間とGPUメモリ使用量は増えます。
各方式の完了時に保存します。途中終了の status=partial や error.json も共有してください。
再開機能はありません。status=complete は実験完了を表し、物理的な合格判定ではありません。

短い動作確認：

```bash
python test_compare_boundaries.py
python compare_boundaries.py --orders 1 --grid 32 --identity-grid 32 --device cpu --output-dir results/boundary_smoke
```
