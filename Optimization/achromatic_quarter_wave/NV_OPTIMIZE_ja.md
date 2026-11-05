# 楕円・長方形モスアイの OFF-NVM 最適化

`nv_optimize.py` は、`n=1.5, 2.0, 2.5` と楕円・長方形の組み合わせを独立に最適化する。構造材と半無限基板は同じ実数屈折率で、溝と入射側は空気。既存の波長板検証に合わせて正方格子の周期は200 nm、高さは500 nm、垂直入射、厚さが等しい15層。各層の x/y 半径（楕円）または半幅（長方形）の計30寸法を、先端から基板に向かって単調に広がるように最適化する。半径・半幅の範囲は5〜95 nm。

前方計算は、補助法線場の有限幅平滑化と Lanczos フィルターを OFF にした NVM を使う。誘電率と逆誘電率は解析フーリエ係数で表し、法線射影行列のグリッドだけを `--grid` で変える。固有値分解の逆伝播には既存 torcwa の安定化を使う。

## 実行

プロジェクトルート（`Optimization` と `rcwa_ext` があるフォルダ）で実行する。依存パッケージは同じフォルダの `requirements.txt` に記載。CUDA 実行には CUDA 対応版 PyTorch が必要。

```powershell
Set-Location 'C:\Users\tsoyu\Documents\Codex\2026-08-06\files-mentioned-by-the-user-extensions\outputs'
python -m Optimization.achromatic_quarter_wave.nv_optimize --goal reflectance --device cuda:0
python -m Optimization.achromatic_quarter_wave.nv_optimize --goal phase --device cuda:0
```

- `--goal reflectance` は x/y 入射偏光の全反射率を400〜700 nmの10 nm刻みで算術平均し、最小化する。
- `--goal phase` は0次の `arg(txx * conj(tyy))` と +90° の差を `[-180°,180°]` に折り返し、波長平均二乗誤差を最小化する。出力には位相の RMS 誤差と最大絶対誤差も度で保存する。
- `--shapes ellipse` または `--shapes rectangle` と、例えば `--indices 2.5` を指定すると1ケースに絞れる。既定では各目的につき6ケースを実行する。
- Optuna の TPE は Adam の初期学習率と各軸の初期先端寸法・テーパー量を探索する。Adam は各試行内で15層すべての x/y 寸法を更新する。既定は各ケース30試行、試行あたり最大500更新、改善が100更新止まったら打ち切り。既定の `M=3, Q=128` は探索用。
- 屈折率2.0の400 nmと屈折率2.5の500 nmでは、基板側の回折次数がカットオフになる。該当する要求波長は長波長側へ相対量 `10^-8` ずらして評価し、要求値と評価値をスペクトルに残す。

短い動作確認の例。

```powershell
python -m Optimization.achromatic_quarter_wave.nv_optimize --goal phase `
  --shapes ellipse --indices 2.5 --device cpu --order 1 --grid 32 `
  --wavelengths 400 500 700 --trials 1 --adam-steps 2 `
  --output Optimization/achromatic_quarter_wave/results/nv_adam_smoke
```

結果は既定で `results/nv_adam_optuna/<形状>_n<屈折率>/<目的>` に保存する。`best_design.json` は形状・材料・設定と両指標、`best_profile.csv` は15層の半径または半幅、`best_spectrum.csv` は波長ごとの反射率・透過率・位相差、`trials.csv` は各試行の設定と結果を記録する。Optuna の `study.sqlite3` を使い、同じ設定と出力先で再実行した場合は完了した試行を再利用する。設定を変える場合は `--output` で出力先を分ける。

反射率と位相誤差の単独最適化の比較では、形状・屈折率・波長・次数・グリッドを揃える。最終候補は高い次数で再評価してから比較する。
