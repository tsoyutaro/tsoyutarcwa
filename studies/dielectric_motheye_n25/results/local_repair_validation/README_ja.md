# 修正のローカル検証

すべて2026-10-11のCPU検証または旧GPU結果の読み取り集計。
400–700 nmの修正後GPU収束結果ではない。

- `saved_gpu_results_review.json`：旧GPUの2研究の結果を変更せず集計。
- `diagnose_n25_trace.json`：旧正方写像のM=10の変換条件数。
- `check_n25_high_order.json`：新正方放射状写像の550 nmでの次数／積分格子依存。
- `check_n25_quadrature.json`：M=12、Nz=20のQ=384/512での追加計算。
- `check_n25_repair.json`：M=8、Nz=12、Q=128でのG依存。特定のGの採用根拠ではない。
- `n25_checkpoint_memory.json`：両格子のcheckpoint有無によるR・勾配の一致と、
  逆伝播用保存テンソルサイズ合計。重複を含み、GPUピークメモリとは異なる。
- `triangular_map.json`：既存の六方写像検証31項目。

`../../RESULTS_REVIEW_20261011.md`に原因・修正・限界・GPU再実行コマンドを記載。
