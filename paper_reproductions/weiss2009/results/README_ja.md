# 結果フォルダの索引

この階層はプログラムが書き込む既定の場所です。ケースの `metadata.json`、
CSV、`run_*/report.json` を一緒に残し、出所が分からなくならないようにします。
フォルダ名は実行コマンドでも指定されているため、既存結果は移動していません。
結果の解釈は [保存済み結果の評価](../docs/RESULTS_ja.md) を参照してください。

| フォルダ | 内容・位置づけ |
|---|---|
| `paper/` | 初期の論文図一式。Fig.4(a) のプロットは全回折次数の T/R を描く。 |
| `smoke/` | 低次数の動作確認。論文との一致判定ではない。 |
| `fig4a_verification/` | PDF曲線と全スペクトルの比較。101周波数×3格子、ゼロ次T/Rで許容範囲内一致。 |
| `fig4a_spectrum_only/` | 論文曲線の縦軸を使用しない計算スペクトル。 |
| `diagnostic_fig3_grid512/`, `diagnostic_fig3_grid1024/` | Fig.3の格子数比較。 |
| `diagnostic_fig3_grid512_algo2a/` | Fig.3の接続アルゴリズム比較。 |
| `diagnostic_fig3_internal/` | 固有値・条件数・電力流の内部診断。 |
| `boundary_comparison/` | 対称性なし、境界投影の初回比較。 |
| `boundary_symmetry_check/` | 次数11・14・15、C2vを使った境界比較。 |
| `boundary_high_order/` | 次数15～18、C2vを使った境界比較。収束未確認。 |
| `fig4b_high_order_grid512/` | 金属370THz、次数15～18、grid512、C2v。収束未確認。 |

`fig4b_high_order_grid1024/` は現在この同期済み結果にはありません。
計算を追加するときは、条件ごとに専用フォルダを使ってください。
`run_*` は実行ごとに作られ、同名ケースの上書きを防ぎます。
