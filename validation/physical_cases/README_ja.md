# 独立した物理検証ケース

このフォルダは `paper_reproductions` と別に、RCWAの出力が基本的な物理条件を満たすか調べる。
`cases.json` が条件、`run.py` が実行ファイル、`results/quick/` が結果置き場。
既定は低次数 M=2、PMMA形状4層、ASR格子96×96、複素倍精度で、素早い整合性確認用。
高次数・層数・格子の収束や実験との一致を証明する設定ではない。

Linuxのプロジェクトルートで、最初にPyTorchなしでも可能な準備確認をする。

```bash
python3 validation/physical_cases/run.py --prepare-only
```

`results/quick/plan.json` に構造、光学定数、各ソースのハッシュを保存する。
以前の100層・M=4～20の27結果について、PyTorchなしで受動性を点検するには
`python3 validation/physical_cases/audit_saved_results.py` を実行する。
吸収率Aが1−R−Tから作られるため、この監査のエネルギー和は独立した保存則の証明ではない。
PyTorchとRCWAがある環境では全7ケースを実行する。CUDAが利用可能なら自動選択する。

```bash
python3 validation/physical_cases/run.py --device auto
```

CPUを使う場合は `--device cpu`、GPUを必須とする場合は `--device cuda`。
各ケース終了時に `checkpoint.json` を保存し、同じ条件で再実行すると合格済みを再利用する。
ソースまたは設定が変わったときは旧 `checkpoint.json`・`report.json`・`plan.json` を
`results/quick/history/` に移してから全ケースを再実行する。
個別実行は `--case air_to_pmma_550` のように指定する。
最終的な判定と数値は `report.json` に保存する。

| ケース | 主な確認 |
|---|---|
| `air_to_pmma_550` | 平坦な空気／PMMA界面のR・TをFresnel式と比較 |
| `lossless_relief_550` | 吸収のないモスアイでR＋T≈1 |
| `au_coated_relief_550`, `au_coated_relief_700` | 金を含む構造のR・T・Aの受動性 |
| `auxiliary_parity_rt_550` | 補助テンソル解放の前後でR・T・Aが一致 |
| `symmetry_parity_rt_550` | 同じD6閉集合でE1行縮約と全既約表現のR・T・Aが一致 |
| `auxiliary_parity_gradient_550` | 金柱の半径勾配とRが補助テンソル解放前後で一致 |

補助テンソルの比較ケースは両方式の実行秒数を記録し、CUDAでは各方式の
`peak_cuda_allocated_bytes` も記録する。低次数での1回ずつの比較なので、
速度差を断定する場合は実際に使う次数・層数でも測定する。

`symmetry_parity_rt_550` はD6閉集合の完全計算を合否基準にする。
M=2ではD6閉集合は19個、矩形の全次元計算は25個のFourierモードを含む。
異なる打ち切りの結果も参考値として保存するが、その差に1e-6の一致条件は課さない。

金入りPMMAモスアイは周期200 nm、PMMA高さ500 nm、先端／底面直径10／130 nm、
金側面厚さ30 nm、トップの金円盤の厚さ30 nm・直径70 nm、谷部金30 nm、
半無限PMMA基板、空気側からの正入射x偏光。PMMAと金の屈折率はstudyの測定CSVを使う。

失敗したケースは `report.json` に例外と短いトレースを残す。
これらは物理的な必要条件の検査であり、通過してもモスアイ近似形状や金膜厚が
実試料と一致するとは限らない。金膜の蒸着角度・遮蔽などは別途評価が必要。
