# 電力収支診断v2：NVMと法線場補正なしの対照計算

## 今回の結果

Cuの6更新後の形状・680 nmでは、従来NVMと共有NVMはM20まで一致しています。しかし、M16/grid1024→2048のR差は0.000567570167、grid1024でM16→20のR差は0.036357609836です。現在の収束許容差0.00025を両方とも超えています。受動性の範囲に入ったことと、数値収束したことは区別します。

従来版referenceも同じNVMの式を使っているため、厳密解や別の数値手法との比較ではありません。

## 追加した診断

`diagnose_power_flux_v2.py` に `--methods galerkin` を追加しました。これは円の解析的な誘電率フーリエ係数を使い、NVMの法線場補正を使わずに標準Cartesian Maxwell P/Qを構築するFourier–Galerkinの対照計算です。固有値計算・散乱行列・電力評価は同じ実装を使います。別の完全独立ソルバーではありません。

この対照計算自体も有限次数の近似で、受動性を満たすだけでは正解と判断できません。NVMとどちらが正しいかを一度の計算だけで決めず、それぞれの次数依存を比較します。最適化への適用は行っていません。

Galerkinは円の解析的フーリエ係数を使い、法線場用gridでサンプリングしません。grid引数はNVM側の比較条件・実行記録として扱います。

## 転送と実行

ZIPを、Optimizationとrcwa_extが並ぶ研究ルートへ展開してください。旧diagnose_power_flux.pyはそのまま残します。最適化本体、checkpoint、Optuna journal、実験manifestは変更しません。

保存されたCuの同じ形状について、680 nmをM8/12/16/20で評価します。既にNVMの結果があるため、追加計算はGalerkinだけに限定します。Linux端末に1行で貼り付けてください。

```bash
( set -e; cd /home/7/uq06557/common2/20260813/tsoyutarcwa/Optimization; for m in 8 12 16 20; do python diagnose_power_flux_v2.py --case-dir Cu/cu_motheye_P200_H500_Nz15 --device cuda:0 --order "$m" --grid 1024 --wavelengths 680 --methods galerkin; done; )
```

同じ実行でNVM側も再計算したい場合は、`--methods shared galerkin` を指定します。

`reference_comparisons_passed: null` は、今回は従来NVMとの実装一致比較を要求していないことを示します。対照法が失敗したという意味ではありません。`galerkin` のR・P_sub・A_moth・passiveと、保存形状のchecksumを確認してください。

新しい診断は `results/hex_trial1_v3/diagnostics/power_flux_v2_*.json` へ保存します。旧診断ファイルと区別します。診断完了は最適化完了の判定ではなく、FAIL試行の状態も変更しません。

## ローカル検証範囲

正方格子・六方格子のM2で、P/Qが標準Cartesian Maxwell行列を直接構築・射影した結果と一致しました（最大絶対差約3.6e-14）。均一金属層および均一誘電体層では、Fresnel反射と有限厚さの減衰に一致しました。Cuの低次数の円形層および初期15層/M8のCPU順計算も確認しました。

GPUの実際の6更新後形状に対する対照計算は未実行です。現時点でGalerkinへの置き換えや、M20での再最適化を推奨するものではありません。
