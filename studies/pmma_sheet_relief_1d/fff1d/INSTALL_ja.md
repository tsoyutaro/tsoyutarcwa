# 配布物と反映方法

ローカルの2026-08-06側 `studies/pmma_sheet_relief_1d/` には反映済みです。元の通常RCWAのルートPython・材料データ・既存resultsは維持しています。

別のGPU計算機へ反映するときは、v2またはv2のp偏光対応版があるRCWAルート（`rcwa_ext/` と `studies/` がある場所）で `pmma_sheet_relief_1d_fff_patch_20261115.zip` を展開してください。ZIP内の先頭は `studies/pmma_sheet_relief_1d/` です。新規 `fff1d/`、共通自動実行の互換拡張、p偏光用補助、説明・検証記録を含みます。結果フォルダは含みません。ZIP内の `validation/fff1d/` は今回の予備計算記録です。

全v2ソース一式は `pmma_sheet_relief_1d_v2_fff_20261115.zip` です。先頭が `pmma_sheet_relief_1d/` なので `studies/` の下へ配置してください。既存の `rcwa_ext` と `studies/gold_grating_1d/solver.py` に依存します。RCWAルート全体を再配布するZIPではありません。

通常RCWAとFFFの実行ファイルを分けています。FFFで計算する場合は [README_ja.md](README_ja.md) の `fff1d/converge_gil.py` または `fff1d/run_fff.py` を使ってください。従来の `automation/converge_until.py` を単独で呼ぶと通常RCWAです。

検証記録はCPUで作成しました。CUDA実行の前に `validation/fff1d/validate.py --device cuda:0 --suite standard` で同じ照合ができます。最終ソースのSHA-256はスタディ直下の `DELIVERY_MANIFEST.json` に記録しています。
