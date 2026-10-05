# NV/ASR修正を既存GitHubリポジトリからTSUBAMEへ反映する

新リポジトリは不要です。既存リポジトリtsoyutaro/tsoyutarcwaのmaster上のソースを
修正しています。研究コードのディレクトリ構成と既存studies/の実行コマンドを維持します。
新しいGUIを起動する手順ではありません。

## ローカルからGitHubへ送る

この修正を確認してcommit/pushした後、TSUBAMEでpullできます。GitHubへpushする前は
TSUBAMEのgit pullだけでは変更が届きません。この更新は未commit・未pushです。
既存20260806のoutputs/（.gitとrcwa_extがあるフォルダ）で次を実行してください。

```bash
git diff --check
git diff --stat
git add README_ja.md rcwa_ext docs validation/validate_nv_asr_physics.py validation/validate_design_gradients.py validation/validate_triangular_matched_asr.py paper_reproductions/wang2022_fig8/validation/validate.py
git commit -m "Fix NV factorization and ASR power and internal-field consistency"
git push origin master
```

## TSUBAMEで更新・確認

普段使用しているtsoyutarcwaのプロジェクト直下に移動し、既存Python環境を有効にして実行します。

```bash
git status --short
git pull --ff-only origin master
python3 validation/validate_nv_asr_physics.py --device cuda --json results/nv_asr_v1_physics_cuda.json
```

git statusにソース変更が出た場合は、変更を保存してからpullしてください。
ローカル修正を消すreset --hardは不要です。git pullで反映した場合、追加パッチの適用や
ZIPの解凍は不要です。既存のPyTorch/Torcwa環境を維持します。
新しい物理検査20項目を通過した後、普段のstudies/の実行へ進みます。
このCUDA検査はローカルのGPU実機では未実施です。

## 以前の最適化形状を再評価する

形状は再利用できます。旧スコアやAdam momentsは新版へそのままresumeしません。
新しいoutput-dirで、同じ材料・次数・層数・格子・波長点の条件を用いて再評価します。
例として、140層・13波長追加探索の保存済み形状を高精度条件で確認する場合：

```bash
python3 studies/gold_motheye2/validate_adam_profile.py --device cuda \
  --run-dir studies/gold_motheye2/results/adam_refine_from_best_M8_Nz140_13wl \
  --orders 18,20 --slices 140 --grids 1024 --wavelengths 400:700:10 \
  --output-dir studies/gold_motheye2/results/validation_nv_asr_v1_Nz140
```

これは保存済み形状の評価です。追加最適化を行う場合は--initial-run-dirで形状だけを引き継ぎ、
新しい--output-dirを使います。既存ソースhashの保護を解除しません。
GPUの大規模な高次数・多数層計算はこのローカル作業では実行していません。

共有ソルバーの修正詳細はNV_ASR_CORRECTIONS_ja.mdです。
元のソースのバックアップと適用記録は、このGitリポジトリの外に保存しています。
