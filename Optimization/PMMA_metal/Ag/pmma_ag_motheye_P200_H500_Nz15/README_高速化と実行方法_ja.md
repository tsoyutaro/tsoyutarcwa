# PMMA core / Ag shell：15層の両半径最適化

周期200 nm、高さ500 nm、六方格子。PMMA基板上のPMMA芯とAg被覆を同心円状に配置します。
各15個のPMMA半径と金属外半径の両方を最適化し、それぞれ深さ方向の単調増加と包含関係を保ちます。
固定の上部キャップや谷部金属膜は追加しません。被覆厚さ一定の拘束はありません。

目的はmean(A)=mean(1-R-T)の最大化です。PMMAはlosslessで、TはPMMA基板へ透過する電力です。

確認用の1 trial：

```bash
python run.py --config config_hex_trial1_M8_Q512.json --device cuda:0 --mode optimize --lattices hexagonal --skip-audit --timeout-hours 0 --output-root results/hex_trial1_M8_Q512_v1
```

終了状態：

```bash
python check_status.py --run-root results/hex_trial1_M8_Q512_v1
```

通常の中断・OOM後は、同じ設定・出力先で再実行します。FAILになったtrialは自動再試行しません。
config.jsonは30 trials、確認用config_hex_trial1_M8_Q512.jsonは1 trialです。最大500受理更新、patience=100です。
最適化後に停止し、検証・スペクトル取得は別コマンドです。データはこのフォルダ内に同梱しています。

初期・勾配の小規模検証：

```bash
python validate_case.py --device cuda:0 --suite small
```

M8・grid512・15層・全31波長の初期形状と勾配：

```bash
python validate_case.py --device cuda:0 --suite initial --with-gradient
```

収束確認：

```bash
python run.py --config config_hex_trial1_M8_Q512.json --device cuda:0 --mode convergence --orders 8,12,16,20 --grids 512,1024 --wavelengths 400:700:10 --timeout-hours 0 --output-root results/hex_trial1_M8_Q512_v1
```

計算時間（autoのみ、逆伝播込み）：

```bash
python run.py --config config_hex_trial1_M8_Q512.json --device cuda:0 --mode timing --order 8 --grid 512 --wavelengths 400:700:10 --timing-backends auto --repeats 3 --timeout-hours 0 --output-root results/hex_trial1_M8_Q512_v1
```

スペクトル（次数・gridは収束確認に応じて変更）：

```bash
python run.py --config config_hex_trial1_M8_Q512.json --device cuda:0 --mode spectrum --order 16 --grid 1024 --wavelengths 400:700:3 --timeout-hours 0 --output-root results/hex_trial1_M8_Q512_v1
```

被覆厚さの数値下限は0.001 nmです。実際の最小膜厚を指定する場合はmin_shell_thickness_nmを変更し、新しい出力先を使います。
PMMAの400〜404.7 nmは既存の線形延長方針を引き継ぎます。k=0はモデル上の仮定です。
D6 core-shellの行列構築・形状部分の波長共有を使い、両材料の分散を波長ごとに反映します。
CPUの初期・小規模検証は合格していますが、CUDAの500更新の完走や数値収束は未確認です。
