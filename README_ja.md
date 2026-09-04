# RCWA拡張成果物ガイド

このディレクトリは、公開solver、応用計算、論文再現、検証、数式資料を用途別に分離しています。
すべてのコマンドは、この `outputs/` を作業ディレクトリとして実行してください。

## ディレクトリ構成

```text
outputs/
├─ rcwa_ext/                  # solver本体
├─ rcwa_solver_auto.py        # 旧import互換facade
├─ studies/                   # モスアイなどの応用・収束計算
│  ├─ shared/                 # 共通材料モデル（金分散）
│  ├─ gold_motheye/           # 金モスアイ／金基板
│  └─ pmma_gold_motheye/      # 金被覆PMMAモスアイ
├─ paper_reproductions/       # 論文ごとの再現実装
│  ├─ wang2022_fig8/          # ASR-RCWA論文 Fig. 8
│  └─ peng2025/               # Ag–air–Ag円形構造
├─ validation/                # solver横断の独立検証
├─ docs/                      # 数式導出・設計資料
└─ README_rcwa_extension_ja.md # APIと全機能の詳細
```

各研究フォルダでは、実行コード、`README_ja.md`、`results/`、必要なら
`validation/` を同じ場所にまとめています。検証出力は通常計算結果と混ざりません。

## 推奨して読む順番

1. `README_rcwa_extension_ja.md` — 対応機能と公開API
2. `rcwa_ext/README_ja.md` — solver内部のファイル分担
3. `docs/rcwa_modular_math_guide_ja.md` — Maxwell方程式から実装までの全体導出
4. `docs/ASR_circle_NVM_report_ja.md` — 円形NVMとmatched-ASR
5. `docs/triangular_matched_asr_math_report_ja.md` — 三角格子matched-ASR
6. `docs/complete_d6_native_star_report_ja.md` — native-star完全D6分解
7. 目的に応じて `studies/*/README_ja.md` または `paper_reproductions/*/README_ja.md`

## よく使うコマンド

```bash
python validation/validate_rcwa_solver_auto.py
python validation/validate_circle_matched_asr.py --integration
python validation/validate_triangular_matched_asr.py --integration
python studies/pmma_gold_motheye/validation/validate.py --integration
python paper_reproductions/wang2022_fig8/validation/validate.py --integration
python paper_reproductions/peng2025/validation/validate.py --device cpu
```

移動後の各実行ファイルは、自身から `outputs/` を解決して `sys.path` に追加するため、
絶対パスで直接実行しても同じsolverを読み込みます。
