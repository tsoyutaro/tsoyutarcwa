# 改行差によるソース照合停止への修正版（v2）

旧版はファイルの生バイトのSHA256で照合していました。配布元の asr.py にはCRLF/LFが混在しており、Gitや転送ツールでLFに統一しただけでも一致しなくなります。今回のエラーログだけでは、転送先で改行だけが変わったのか、実際の内容も変わったのかまでは判定できません。

v2は元の検証済みファイルから作った信頼済みのハッシュ表を使い、UTF-8 BOMとCRLF/LF/CRの違いだけを正規化して照合します。空白、コメント、数値、処理内容の変更は引き続き検出します。チェックを無効にしたり、転送先のファイルから期待ハッシュを作り直したりする必要はありません。

パッチZIPを使用する場合は、中のファイルを現在の optimization_integration_validation 直下へ上書きして転送してください。フルZIPなら、既存と同じ名前の検証フォルダ全体が入っています。既存の results や本番の n2.5/Cr フォルダはパッチの対象ではありません。

最初に、計算を行わない整合性チェックを実行します。標準Pythonのみで動き、PyTorchやCUDAは使用しません。

```bash
python source_integrity.py
```

status が passed なら、そのまま次を実行できます。newline_or_bom_only_files に表示されるファイルは、内容が同一で改行またはBOMだけが違うものです。

```bash
python validate_connection.py --device cuda:0 --suite quick --output results/cuda/quick_v2
```

quick合格後に実行します。

```bash
python validate_connection.py --device cuda:0 --suite production --output results/cuda/production_M8_L15_v2
```

ソース識別方式が変わるため、v1と異なる出力先を使います。元の計算で何も出力されていなくても、この名前で構いません。

status が failed で reason が content_changed の場合は、改行以外の内容差があります。mismatches に該当ファイルと期待値・実値を表示します。修正版フルZIPの検証用コピーへ戻して確認してください。期待ハッシュの作り直しやチェック行の削除で通過させないでください。

ローカルでは全テキストをLF+BOMへ変換したコピーを作って48ファイルの照合を確認し、ソースへの実際の追記・ファイル欠落・旧manifestの拒否も確認しました。数値検証もLF+BOM変換後のCPUのn2.5/PMMA × 正方格子/六方格子の4ケースで再実行し、全ケース合格しました。CUDA実機の検証はこのPCでは実施できません。
