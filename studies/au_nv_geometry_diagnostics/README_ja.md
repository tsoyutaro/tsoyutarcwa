# Auの周期・高さ・半径を明示したNV診断

周期100 nm、高さ200 nm、Au膜厚30 nmの両構造を診断する。

```bash
python -u run_au_nv_geometry_diagnostics.py \
  --device cuda:0 \
  --orders Au:36 --structures both --wavelengths 700 \
  --slices 210 --grids 512 \
  --period-nm 100 --height-nm 200 \
  --tip-radius-nm 5 \
  --metal-base-radius-nm 47.5 --pmma-base-radius-nm 17.5 \
  --metal-side-thickness-nm 30 \
  --metal-top-thickness-nm 30 \
  --metal-valley-thickness-nm 30 \
  --output-root studies/au_nv_geometry_diagnostics/results/P100_H200_M36_Nz210_Q512_700nm_Au30
```

- Au単体：先端半径5 nm、底面半径47.5 nm、高さ200 nm、Au半無限基板。底面間の隙間5 nm。
- PMMA+Au：PMMA先端半径5 nm、底面半径17.5 nm、高さ200 nm。Au側面・上部・谷部30 nm。外周底面半径47.5 nm、上部キャップ半径35 nm。全高230 nm、PMMA半無限基板。
- M36、Nz210、Q512、700 nm、垂直入射x偏光、三角格子。NV/D6、complex128、Redheffer、Lanczosなし。Au・PMMAの光学定数は従来と同一。
- Au単体210層、PMMA+Auはキャップを加えて211層。既存のメモリ節約ソルバーと層別パワー診断をそのまま使用する。

半径を省略した場合、Au単体の底面半径は0.475×周期、PMMA底面半径はその半径から側面膜厚を引いた値になる。先端半径は自動縮小しない。元の200 nm周期形状を単純に一様縮小した形状ではない。半径・膜厚・層数の組み合わせを事前検証し、底面の接触・重なりや無効なPMMA形状は計算前に拒否する。

各条件は別プロセスで実行し、OOMや非物理値が出ても次の構造・条件を続ける。完了済みは同じコマンドで再利用でき、失敗済みの再試行には`--retry-failures`を使う。周期変更によってMやNzの行列サイズは減らないので、従来M36/Nz210で成功した容量のGPUを使う。

`--dry-run`で形状と材料を確認でき、計算・保存は行わない。`--plot-only`は同じ設定の保存結果から図を再作成する。元の周期200 nmの結果は再利用・変更しない。形状・材料・ソースが異なる保存先には誤って追記できない。

出力：`geometry_summary.csv/json`（R/T/A、半径・膜厚、負の層吸収、診断チェック）、`geometry_RTA_*.png/svg`、条件ごとの`report.json`・`layer_power.csv`・`operator_diagnostics.csv`・励起モードと層別の図。

周期変更は別構造の比較であり、元の構造の収束改善を証明するものではない。`physical_result`、`diagnostic_checks_passed`、厳しい負の固有値成分のチェックは別々に読む。新形状でもM/Nz/Qに対する収束確認は必要。
