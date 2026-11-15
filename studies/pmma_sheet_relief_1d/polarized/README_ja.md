# p偏光だけの計算（2026-11-15）

`--polarization p` を自動収束コマンドへ追加しました。**新規計算ではs偏光の固有値計算・層接続・位相平均を行いません。** complex128、鏡映対称縮約、p偏光で必要な奇空間の逆行列を保持したLi因子分解、逐次Redheffer接続を使用します。

既定値は `both` で、従来と同じs/p同時計算です。`s` だけの指定も可能です。このオプションは更新後の `automation/converge_until.py` と `polarized/run_selected.py` で使用します。従来のstudy直下の `run.py` や `converge.py` へ直接渡すオプションではありません。

## M240を基準にpだけで続ける

現在のs/p実行をCtrl+Cで停止してから、更新ZIPを既存RCWAルートへ展開し、以下を実行します。M240→256→272→…と増やし、連続2回の比較が合格するか、OOMなどの停止条件に達するまで進めます。p偏光だけを収束判定します。

```bash
python studies/pmma_sheet_relief_1d/automation/converge_until.py \
  --axis order --start 240 --step 16 --polarization p \
  --coating gold --device cuda --acceleration fast --slices 960 \
  --wavelengths "700" --periods "3720" --voids both \
  --geometry-mode scaled --gold-nm 50 --tolerance 0.001 \
  --phase-tol 0.00005 --phase-max 262144 \
  --output-dir studies/pmma_sheet_relief_1d/results/gold_gpu_auto_p_P3720_w700_Nz960
```

偏光の設定が変わるため、s/p実行とは別の出力先を使います。前回と同じ点キャッシュを参照できれば、完了済みM240のpを再利用します。M256の一部が完了していれば、その波長点も再利用します。計算途中で中断された点は再計算します。

前回 `--cache-dir` を明示していた場合は、今回も同じディレクトリを指定してください。既定は従来どおり `studies/pmma_sheet_relief_1d/results/v2/cache/` です。

## キャッシュの互換性

study直下の既存Python・材料ファイルは変更していないため、従来のs/p点キャッシュの署名を維持しています。新機能は `polarized/` に追加しました。

キャッシュの利用順序は、同条件の新しい単一偏光キャッシュ、次に同条件の旧s/pキャッシュです。旧s/pキャッシュのキーを、元のソース署名・材料・幾何・M/Nz・波長・位相設定・device・torch/numpy版から再構成し、そのp行だけを選びます。異なるソースや設定の結果は流用しません。

旧s/pキャッシュから選んだ結果は数値を変更せず、新しい偏光別キャッシュに保存します。`completion.json` の `legacy_both_cache_hits` が旧キャッシュの再利用数です。キャッシュ由来の `runtime_seconds` は元の計算時間なので、p専用の速度測定には使えません。

既存のs/p自動収束キャンペーンは、偏光指定を省略すれば再開できます。新しいp専用キャンペーンでも、同じコマンド・同じ出力先で再開できます。

## 単一解像度のスペクトル

```bash
python studies/pmma_sheet_relief_1d/polarized/run_selected.py \
  --polarization p --coating gold --device cuda --acceleration fast \
  --order 256 --slices 960 --periods 3720 --wavelengths 700 --voids both \
  --phase-tol 0.00005 --phase-max 262144 \
  --output-dir studies/pmma_sheet_relief_1d/results/gold_gpu_p_M256_Nz960
```

`--acceleration reference` は照合用です。この場合は元の全s/pソルバーから対象偏光を選ぶため、固有値計算のp専用化による短縮はありません。p専用計算は既定の `fast`、または明示的な `--acceleration fast` で使います。

## 一致検証

CPUで27ケース・s/p計54条件を比較しました。Auなし/50 nm、空気/水、P3720/P3920、400/550/700 nm、M24/Nz120、追加のM96/Nz120とM240/Nz30を含みます。単一偏光の固有値呼出しが各層で1偏光だけであることも確認しました。

高次数のCPU照合では、元のs/pソルバーのTorch/MKL線形solveがエラーとなったため、M240の診断プロセス内だけ、両経路に同じSciPy線形solveを使用しました。実運用のソースにはこの代替を導入していません。GPUでの一致・速度測定はローカルでは未実施です。

キャッシュのないp専用実行、旧s/pキャッシュの利用、pだけの自動収束、再開、既存s/pキャンペーンの互換性を確認しました。詳細は `validation/selected_polarization_20261115/` を参照してください。

計算する偏光を絞る機能であり、p偏光の次数収束自体を改善する手法ではありません。s偏光とp偏光、次数と層数の収束はそれぞれ区別します。
