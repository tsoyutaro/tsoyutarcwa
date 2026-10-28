# 非物理値・OOMが出ても収束確認を続ける

既存のrun.pyのconvergenceは、1条件が失敗するとその材料の処理を終了します。本スクリプトは別ファイルとして追加し、既存の最適化ソースを変更せず、固定した最良形状を各条件で評価します。学習や最適化の再実行は行いません。

## 使い方

ZIPをLinuxの `/home/7/uq06557/common2/20260813/tsoyutarcwa` に展開し、`Optimization/continue_convergence.py` を追加してください。既存の最適化結果はそのまま使います。

```bash
( set -e; cd /home/7/uq06557/common2/20260813/tsoyutarcwa/Optimization; python continue_convergence.py --device cuda:0 --materials Al,Ti,Cu,Au --orders 4:40:2 --grids 512,1024 --wavelengths 400:700:10 --oom-policy stop-material; )
```

Al → Ti → Cu → Auの順、15層・六方格子です。各材料で次数4,6,...,40、各次数でgrid512→1024の順に評価します。本スクリプトの--ordersは「開始:終了:刻み」またはカンマ区切りのリストに対応します。

計算はauto（波長間共有あり）のforwardのみです。referenceとの比較や逆伝播は行いません。物理チェックは有効なままで、非物理値をクリップしたり正常値として採用したりしません。

## 継続・省略のルール

- 非物理値：失敗を記録し、その次数の残りgridを省略して次の次数に進みます。例：M4/grid512で非物理値ならM4/grid1024を省略し、M6/grid512へ進みます。
- OOM：既定のstop-materialでは、その材料の残りの未計算条件をすべて省略し、次の材料へ進みます。OOMになった条件以上の高次数を同じ材料で再試行しません。CUDA OOMとCPU OOMの両方を対象にします。
- 各条件は別プロセスです。OOMのプロセスが終了してから次の条件・材料を開始し、GPUメモリを持った親プロセスを残しません。
- 結果を返さずworkerが終了した場合はworker_lostとして記録し、OOMと断定せず、その材料の残りを省略します。
- 他の計算エラーも記録し、その次数の残りgridを省略して次の次数へ進みます。設定・ソース整合性の不一致、結果未保存、既存controller.lockによる競合では、その材料を省略して次の材料へ進みます。
- Ctrl+CやSIGTERMは利用者・実行環境からの停止要求として扱い、全体を中断します。

OOMによる省略は材料ごとです。次の材料は低次数から開始します。高gridでOOMになっても、次の次数の低gridを続けたい場合は、次のオプションに変更できます。

`--oom-policy dominated`

この場合、OOMが出た条件と「同じ以上の次数、かつ同じ以上のgrid」の組合せだけを省略します。例：M20/grid1024でOOMなら、以降のgrid1024を省略し、M22/grid512は実行します。ご指定に合わせた既定のコマンドは、より保守的なstop-materialです。

## 使用する結果と設定

既定の対応は次のとおりです。すべて `Optimization/<材料>/<小文字材料>_motheye_P200_H500_Nz15/` 内です。

| 材料 | 設定ファイル | 結果フォルダ |
|---|---|---|
| Al | config_hex_trial1.json | results/hex_trial1_v3 |
| Ti | config_hex_trial1_M8_Q512.json | results/hex_trial1_M8_Q512_v3 |
| Cu | config_hex_trial1_M12.json | results/hex_trial1_M12_v3 |
| Au | config_hex_trial1_M12.json | results/hex_trial1_M12_v3 |

最良形状が保存済みである必要があります。元のmanifest、設定、ソース、材料CSV、実行環境を既存のポリシーで照合します。解析開始時に最良logitsを固定するので、条件間で形状を変えません。計算対象のソースや材料CSVが途中で変わった場合もエラーとして記録します。

元の最適化結果への変更は、解析用のpostprocess結果・latestポインタ・実行中のcontroller.lockのみです。元のmanifest、Journal、Adamチェックポイント、report内の最良形状は書き換えません。controller.lockは材料の解析終了時に解放します。既存のlockを自動削除しません。強制終了でlockが残った場合は、前のプロセスが終了したことを確認してから対処してください。

## 保存先

各材料の元の結果フォルダ内：

`hexagonal/postprocess/convergence_continue/<識別子>/`

- report.json：全条件の成否、省略理由、計算時間、成功した条件のスペクトル、比較結果。
- convergence.csv：次数・gridごとのstatus、平均反射率、参照との差、計算時間、失敗理由。
- context.json：固定形状、設定、元の学習signature、ソースhash、実行環境。
- Mxx_Qxxx/worker.log、result.json：条件ごとのログと結果。

`hexagonal/latest_convergence_continue.json` にreportの場所を保存します。元のlatest_convergence.jsonは変更しません。

4材料全体の結果は `Optimization/convergence_continue_summary.json` です。スキップや非物理値があっても計画した走査が終わればcompleted_with_skipsとなります。これは数値収束の合格を意味しません。

## 収束比較の意味

非物理値・OOM・省略条件は収束判定の参照に使いません。成功した条件のうち「最高次数、その次数で最大grid」をcomparison_referenceにします。M40/grid1024が計算できなければ、それより低い成功条件が参照になります。

その参照に対し、同じgridの1つ前の成功次数、および同じ次数の1つ小さい成功gridの差を確認します。両方の比較が可能な場合だけnumerical_convergence_passedをtrue/falseにします。片方でも欠けた場合はnullです。参照自身との差0は収束判定に使いません。有限の成功条件間の比較であり、無限次数の正解を保証しません。

## 再実行

同じコマンドを再度実行すると、同じ形状・ソース・設定・実行環境・走査条件で保存済みの結果を再利用します。成功済み条件は再計算しません。非物理値やOOMも記録を再利用し、同じ省略規則を適用します。

失敗条件を再試行したい場合は `--retry-failures` を追加します。成功条件は再計算せず、失敗・省略条件を再検討します。GPUの空き容量が変わった場合などに使えます。GPUモデルやMIGメモリサイズ、ソフトウェア、形状、走査条件が変わると別の識別子になります。

記録を消して再実行する必要はありません。CPU/CUDAの切り替えやソースの変更を、元の実験のmanifestを書き換えて回避しないでください。

## 検証

```bash
python test_continue_convergence.py
```

11件のテストで、非物理値後の次次数への継続、CUDA/CPU OOM後の省略、次材料への継続、再実行時の再利用、成功・失敗の分離、不完全な収束比較を合格扱いにしないこと等を確認しています。OOM分岐は模擬的な構造化エラーで試験しています。

実際のTiケース（15層）でも、CPUのM2/grid48,64・3波長で2条件を計算し、再実行時の成功結果の再利用と、元のmanifest・Journal・チェックポイントが不変なことを確認しています。CUDAでの実際のOOM発生や、M4〜40の全条件は手元では未実行です。

## 別の結果フォルダを使う場合

--cases-fileにJSONを指定できます。例：

```json
[
  {
    "material": "Ti",
    "case_dir": "/absolute/Optimization/Ti/ti_motheye_P200_H500_Nz15",
    "config": "/absolute/Optimization/Ti/ti_motheye_P200_H500_Nz15/config_hex_trial1_M8_Q512.json",
    "run_root": "/absolute/Optimization/Ti/ti_motheye_P200_H500_Nz15/results/hex_trial1_M8_Q512_v3"
  }
]
```

この例では--materials Tiも指定します。パスは絶対パスを推奨します。--rootは全体のsummaryの出力先と、既定の各材料の探索元です。
