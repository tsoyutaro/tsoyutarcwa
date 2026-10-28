# Tiモスアイ構造の最適化（Alと同じv3方式）

周期200 nm・高さ500 nmで、5・10・15・20層の独立フォルダを追加しています。

- ti_motheye_P200_H500_Nz5
- ti_motheye_P200_H500_Nz10
- ti_motheye_P200_H500_Nz15
- ti_motheye_P200_H500_Nz20

各フォルダにはrun.py、optimizer、rcwa_ext、材料データ、依存ライブラリ一覧を同梱しています。フォルダ単位で別PCへコピーして実行できます。計算コードは対応するAlケースと同一で、Tiの複素屈折率データを指定しています。Ag用の更新棄却版v4とは別の、Alと同じv3のAdamです。

## 材料データ

既存の `studies/shared/transition_metals_data/Ti.csv` をバイト単位で同一コピーしています。このファイルは `studies/transition_metals_nv/data/Ti.csv` とも一致しています。元データはユーザー提供CSVです。

SHA256: `1b13b2ee9306617d9dd5a1a0e2746d7c6bd880324c7a607f50853b8e6e819216`

wlはmicrometre、nとkはCSV内の別ブロックです。既存の金属用ローダーでepsilon=(n+ik)^2を計算し、複素epsilonを波長に対して線形補間します。外挿はしません。Ti柱＋同じTiの半無限基板、空気入射、コーティングなしです。

## 設定

通常のconfig.jsonはAlと同様、正方格子・六方格子、M8・grid512、30 Optuna trials/格子、Adam最大500更新です。

各フォルダの `config_hex_trial1_M8_Q512.json` は確認用で、六方格子、M8、grid512、1 trial、Adam最大500更新、波長400–700 nmを10 nm刻み（31波長）です。15層の確認にはNz15フォルダの設定を指定してください。既存と同じpatience=100の早期終了も有効です。1 trialは完走や収束を保証する設定ではなく、試行数の指定です。

## 配置

ZIPをLinuxの `/home/7/uq06557/common2/20260813/tsoyutarcwa` に展開してください。`Optimization/Ti/` が追加されます。

## 次数8・15層の1 trialをCUDAで実行

以下は1行のコマンドです。

```bash
( set -e; cd /home/7/uq06557/common2/20260813/tsoyutarcwa/Optimization; d="$PWD/Ti/ti_motheye_P200_H500_Nz15"; python "$d/run.py" --config "$d/config_hex_trial1_M8_Q512.json" --device cuda:0 --mode optimize --lattices hexagonal --skip-audit --timeout-hours 0 --output-root "$d/results/hex_trial1_M8_Q512_v3"; )
```

最適化後は停止し、監査・収束確認・スペクトル取得は自動実行しません。CUDAが利用できない場合にCPUへ自動フォールバックしません。

## 終了状態の確認（RCWA再計算なし）

```bash
( set -e; cd /home/7/uq06557/common2/20260813/tsoyutarcwa/Optimization; python Ti/check_trial_status.py; )
```

`optimization_completed: true`、`optuna_trial_state: COMPLETE`、`worker_status: completed` を確認します。元のv3では、失敗時にも `Optimization stopped before verification` が表示されるため、このメッセージだけでは完走と判定しないでください。

この確認コードは保存された結果を読むだけです。Adam更新回数、成功した勾配評価数、最後の評価時間/波長、最良の平均反射率も表示します。実行中・未確定は終了コード2、失敗は1、完了は0です。

物理チェックに抵触した場合、v3ではtrialがFAILとなります。その場合も学習履歴とチェックポイントは保持されます。FAILもtrials=1の試行数に数えられるため、同じ出力先を指定しても新しいtrialは追加されません。

## 初期形状の事前チェック（任意、最適化なし）

```bash
( set -e; cd /home/7/uq06557/common2/20260813/tsoyutarcwa/Optimization; python Ti/validate_ti.py --device cuda:0 --check-m8-initial --output Ti/verification_cuda_initial.json; )
```

4フォルダの設定・CSV、分散データ、小規模なreference/sharedの値と勾配、logitsに対する有限差分を調べます。check-m8-initialを付けると、本番と同じ最初のtrialの初期形状について、M8・grid512・15層・31波長のforwardと勾配付き評価を比較し、物理チェックを行います。確認用の計算だけで、Optunaのtrialや最適化結果は変更しません。

初期形状での通過は、その後の全ての更新形状や細かい波長サンプリングでの通過を保証しません。数値収束の確認は最適化後に行います。

## 最適化後の収束確認

```bash
( set -e; cd /home/7/uq06557/common2/20260813/tsoyutarcwa/Optimization; d="$PWD/Ti/ti_motheye_P200_H500_Nz15"; python "$d/run.py" --config "$d/config_hex_trial1_M8_Q512.json" --device cuda:0 --mode convergence --lattices hexagonal --orders 8,12,16,20 --grids 512,1024 --wavelengths 400:700:10 --timeout-hours 0 --output-root "$d/results/hex_trial1_M8_Q512_v3"; )
```

`numerical_convergence_passed` と各条件の差を確認します。計算完了やSavedは数値収束と同じ意味ではありません。

## 逆伝播込みの計算時間

```bash
( set -e; cd /home/7/uq06557/common2/20260813/tsoyutarcwa/Optimization; d="$PWD/Ti/ti_motheye_P200_H500_Nz15"; python "$d/run.py" --config "$d/config_hex_trial1_M8_Q512.json" --device cuda:0 --mode timing --lattices hexagonal --order 8 --grid 512 --wavelengths 400:700:10 --repeats 3 --timeout-hours 0 --output-root "$d/results/hex_trial1_M8_Q512_v3"; )
```

timingは比較のためreferenceとautoの両方を実行します。autoのwarm_median_seconds_per_wavelengthが波長間共有ありの時間です。forwardのみなら--forward-onlyを追加してください。最適化中の最後の成功評価の時間はcheck_trial_status.pyでも表示できます。

## 最良形状のスペクトル

```bash
( set -e; cd /home/7/uq06557/common2/20260813/tsoyutarcwa/Optimization; d="$PWD/Ti/ti_motheye_P200_H500_Nz15"; python "$d/run.py" --config "$d/config_hex_trial1_M8_Q512.json" --device cuda:0 --mode spectrum --lattices hexagonal --order 16 --grid 1024 --wavelengths 400:700:3 --timeout-hours 0 --output-root "$d/results/hex_trial1_M8_Q512_v3"; )
```

上のM16・grid1024は指定例です。収束確認に応じて採用する次数・gridに置き換えてください。金属のpower_into_substrateは基板表面への流入電力で、遠方への透過率ではありません。

## 検証結果

同梱のverification_*.jsonを参照してください。手元はCPU環境で、CUDAの500更新までの本番trialは未実行です。GPUでの完走は上記の1 trial用コマンドと終了状態の確認で判断してください。

CPUでの確認結果：4ケースのprepare-only、Tiの分散データ、小規模条件でのreference/sharedの電力・勾配一致、有限差分照合が通過しました。さらにM8・grid512・15層・31波長の初期形状でforwardと勾配付き評価が通過し、電力値の差は0、初期平均反射率は0.012946556889293338でした。これらはCPU結果です。

さらに、CPUでM8・grid512・15層・31波長を使い、実際のOptuna 1 trialをAdam 2更新に限定して実行し、完了状態の確認と、監査・検証へ進まず停止することを確認しました。最良平均反射率は 0.00902825642177 です。配布する1 trial用設定は最大500更新であり、CPU試験用の2更新設定とは分けています。
