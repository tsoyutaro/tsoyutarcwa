# Ti：補助法線場の正則化OFFによる独立最適化

NVMと補助法線場は維持し、有限幅smoothstep補正のみを外します。定義はnearest-radial-center-and-tie-mean-v1で、先行Al OFFケースと同じです。

15層、六方格子、周期200 nm、高さ500 nm、次数8、grid 1024。同じ金属の半無限基板上に置く単一金属構造で、PMMA複合構造ではありません。材料CSVは当該金属の従来ケースと同一です。

1 Optuna trial、Adam最大500 step（早期停止あり）、400:700:10の31波長で平均反射率を最小化します。最適化終了後は停止し、自動で検証を実行しません。非物理値をクリップしたり無視したりする処理はありません。

初期形状は当該材料の旧完了trialの最良形状で、Adam/Optunaは新しく開始します。

実行例：`python run.py --config config_hex_trial1_M8_Q1024_off.json --device cuda:0 --mode optimize --lattices hexagonal --skip-audit --timeout-hours 0 --output-root results/hex_trial1_M8_Q1024_off_v1`

`--skip-audit`は勾配検証を省略します。別途`python validate_off.py --device cuda:0`で初期形状の全16変数の有限差分・reference・checkpoint整合性を検証できます。`python check_status.py`は結果を変更せず進捗を表示します。

完了後の`python compare_off_profiles.py --device cuda:0 --order 30 --grid 1024 --wavelengths 400,550,700`では、保存済み初期形状と新しい最良形状を同一OFF条件で比較します。M30を収束値と断定しません。Agの比較対象は旧最適形状ではなく未最適化初期形状です。

新ケースはソースとデータを独立して保持します。既存ケースのソース・結果は変更しません。CUDA本番最適化はローカルでは実行していません。
