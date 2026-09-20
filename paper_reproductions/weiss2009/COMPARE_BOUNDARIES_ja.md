# 変換・境界接続の比較実験

今回の対称性更新ZIPはプロジェクトルートで展開してください（末尾参照）。
比較実験は既存の境界式を置き換えず、別経路で評価します。
コアにはオプションの対称性固有値計算を追加しています。diagnose_fig3.py の共通関数を利用します。

```bash
python compare_boundaries.py --orders 11,14,15 --grid 512 --identity-grid 1024 --device cuda --output-dir results/boundary_comparison
```

complex128、G=0.03（reproduce.py の実値を記録）、無損失誘電体の Fig.3 条件を使います。
各次数・profile で内部モードは一度だけ計算し、全方式で共有します。
接続比較を単純にするため、単層・真空入出射・垂直入射・Redheffer 固定です。
`--dtype`、`--cascade` はありません。`--use-symmetry` は以下の更新で追加しました。
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

## 対称性と高次数計算

`compare_boundaries.py` と `reproduce.py` は `--use-symmetry` に対応します。
P/Q の C2v 不変性を検査し、全4ブロックの固有値問題を解いて完全なモード基底に戻します。
片方の偏光を複製する方式ではありません。両偏光を独立に計算します。
不変性検査に失敗した場合は停止します。適用したブロック数・残差を JSON/CSV に記録します。
固有値計算を軽くしますが、変換行列の構築、全サイズの境界接続、SVD は縮小されません。
したがって実行時間全体が必ず短くなるとは限りません。
悪条件の高次数では演算経路による差があり得るため、まず既存次数の対称性なし結果と比較してください。

```bash
python compare_boundaries.py --orders 11,14,15 --grid 512 --identity-grid 1024 --device cuda --use-symmetry --output-dir results/boundary_symmetry_check
python compare_boundaries.py --orders 15,16,17,18 --grid 512 --identity-grid 1024 --device cuda --use-symmetry --output-dir results/boundary_high_order
python reproduce.py --study fig4-convergence --orders 15,16,17,18 --grid 512 --identity-grid 1024 --device cuda --dtype complex128 --use-symmetry --output-dir results/fig4b_high_order_grid512
python reproduce.py --study fig4-convergence --orders 15,16,17,18 --grid 1024 --identity-grid 1024 --device cuda --dtype complex128 --use-symmetry --output-dir results/fig4b_high_order_grid1024
```

境界比較は Fig.3 の誘電体です。Fig.4(b) の金属収束性は reproduce.py で計算します。
高次数の --use-symmetry あり/なし一致やCUDAでの高速化は、利用環境で確認してください。
CPU complex128 の低次数では誘電体・金属、ASRあり/なしで R/T/A の差が1e-9未満であることをテストします。

この更新ZIPはプロジェクトルート（rcwa_ext と paper_reproductions がある場所）で展開してください。
rcwa_ext/asr.py の更新も必要です。古いファイルは上書き前にバックアップしてください。
reproduce.py/asr.py のハッシュが変わるため、verify_fig4a の古いチェックポイントに --resume すると拒否される可能性があります。
その場合は新しい出力先で再計算し、署名チェックを外さないでください。
