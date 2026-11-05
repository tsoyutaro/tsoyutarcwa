# Kikuta et al. (1997) 図7の再現

図6(a)の単純ラメラ格子に空気側から垂直入射する。基準波長を `λ₀ = 1` とし、周期を `0.4λ₀`、溝深さを `1.146λ₀`、高屈折率部の幅を周期の70%とする。溝と入射媒質の屈折率は1.0、リッジと半無限基板の屈折率は2.0。波長 `λ/λ₀ = 0.8–1.2` を201点で計算する。[原論文](https://doi.org/10.1364/AO.36.001566)

## 実行

親リポジトリのルート（`paper_reproductions` がある `outputs`）で実行する。CPU には NumPy、SciPy、Matplotlib が必要。CUDA 実行には CUDA 対応版の PyTorch も必要。`--device cuda` は CUDA が使用できない場合にエラーで終了する。[PyTorch のインストール手順](https://docs.pytorch.org/get-started/locally/)

```powershell
python -m paper_reproductions.kikuta1997_fig7.reproduce --device cpu
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m paper_reproductions.kikuta1997_fig7.reproduce --device cuda --output-dir paper_reproductions/kikuta1997_fig7/results/cuda
```

`--device cuda:0` のように GPU を指定できる。既定の `--harmonics 61` は `m = −30, …, +30` を意味する。収束を確認する場合は `--harmonics 121` を指定して出力先を分ける。実行すると `fig7.png` と `fig7.csv` を保存し、図から読み取った21点との差とエネルギー収支の残差を表示する。既定の出力先はこの実装フォルダ内の `results`。

## 図7(a) 位相差

- Rigorous は既存の1次元 RCWA/FMM を倍精度で実行し、0次透過振幅の `arg(t_TE / t_TM)` を度で表示する。TE は電場が溝に平行、TM は磁場が溝に平行。基板の1次回折がちょうどカットオフとなる `λ/λ₀ = 0.8` は、既存ソルバーが長波長側の極限で評価する。CSV には要求波長と評価波長を両方記録する。
- EMT は無限ラメラ媒質の基本 Bloch モードから `360° × h × (n_TE − n_TM) / λ` を計算する。TM については磁場と `(1/ε) ∂H/∂x` の連続条件から、分散式の係数を `(ε₂/ε₁) k₁z² + (ε₁/ε₂) k₂z²` とした。印刷された式(4)の誘電率比とは逆の対応で、図2の TM 有効屈折率と図7(a)の EMT 曲線を再現する。
- 石英の比較線は、複屈折を一定として `λ₀` で90°になる `90° / (λ/λ₀)` で近似した。図の再現に用いたモデルの絶対波長と石英の材料分散は指定されていない。

## 図7(b) 透過率

図中の白丸（TE 表記）は RCWA の TM 0次透過パワーの二乗、黒丸（TM 表記）は TE 0次透過パワーの二乗と一致する。この対応は PDF のプロットとの照合から得たもので、論文本文には二乗や偏光ラベルの対応について説明がない。図の再現値は `fig7b_te_label` と `fig7b_tm_label` に記録する。物理的な0次透過パワーは `T0_te` と `T0_tm` に記録する。

`fig7_markers.csv` は添付PDFの図7からマーカー中心を読み取った概算値。CPU の61モード計算では、読み取り値に対する最大差は位相差で約0.53°、図7(b)の透過率で約0.0045（0.45パーセントポイント）。TE、TM ともに `|1 − R − T|` の最大値は `2 × 10⁻¹⁵` 未満だった。
