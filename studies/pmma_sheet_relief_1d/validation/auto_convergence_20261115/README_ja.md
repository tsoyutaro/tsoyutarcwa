# 自動収束制御の検証

2026-11-15。追加は `automation/converge_until.py` と説明文書です。計算本体・材料データは変更していません。

- `unit_tests.json`: 制御テスト20件の合否。OOM、強制終了、中断、再開、通常エラー、位相未収束、同時実行ロック、上限、次数別比較などを検証。OOMは模擬エラーであり、GPUの実メモリを使い切る試験はしていません。
- `integration_checks.json`: 実RCWAと従来の収束コマンドとの比較。CPU、Auなし、周期200 nm、550 nm、水溝、s/p。次数M2/4/6（Nz4固定）と層数Nz4/8/16（M4固定）。各段階の3種類の結果CSV計18ファイルと、隣接比較CSV2ファイルがバイト単位で同一。実データでの再開も確認。
- `order_*_adjacent_changes.csv` / `slices_*_adjacent_changes.csv`: 自動版と従来版の比較値。
- `installation_checks.json`: 実studyへの配置、配置先からの実行、元の計算本体・キャッシュ署名・過去の結果の保存確認。

物理ソース署名は更新前後で `6597efe7fab2f6c1e6330ba1f2cb41861b42b80f62578d9dbd312d8deecce3b5` と同一です。従来のAu付き計算結果やGPU結果のキャッシュキーを変更しません。

この検証は自動制御と既存計算の一致を確認するものです。Au付きM96/Nz960などの収束、より大きな解像度でのGPU OOM発生点は、対象GPUでの実行によって確認してください。
