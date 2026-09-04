"""Convergence versus Fourier diffraction order for Au-coated PMMA moth-eye."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# TSUBAME / CUI環境用のヘッドレス描画バックエンド
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_OUTPUTS_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = Path(__file__).resolve().parent
if str(_OUTPUTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_OUTPUTS_ROOT))

from studies.pmma_gold_motheye.common import (
    NumericalConfig,
    add_shared_arguments,
    parse_int_list,
    run_axis_convergence,
)


def display_and_plot_results(checkpoint_path: Path, output_png_path: Path) -> None:
    """_checkpoint.json から計算結果を読み込み、コンソール表示と PNG 保存を行う。"""
    if not checkpoint_path.exists():
        print(f"[Warning] Checkpoint file not found: {checkpoint_path}")
        return

    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    cases = list(checkpoint.get("cases", {}).values())
    if not cases:
        print("[Warning] No simulation cases found in checkpoint.")
        return

    # 波長ごとにデータをグループ化
    data_by_wl: dict[float, list[dict]] = defaultdict(list)
    for case in cases:
        data_by_wl[case["wavelength_nm"]].append(case)

    wavelengths = sorted(data_by_wl.keys())

    # 1. コンソールへのテキストテーブル表示（波長別）
    for wl in wavelengths:
        records = sorted(data_by_wl[wl], key=lambda x: x["order"])
        print("\n" + "=" * 82)
        print(f" Wavelength: {wl:g} nm")
        print("=" * 82)
        print(
            f"{'Order (M)':^10} | {'Dim':^6} | {'Reflectance (R)':^14} | "
            f"{'Transmittance (T)':^14} | {'Absorptance (A)':^14} | {'Time (s)':^8}"
        )
        print("-" * 82)
        for r in records:
            order = r["order"]
            ref = r["reflectance"]
            trans = r["transmittance"]
            abso = r["absorptance"]
            dimension = r.get("reduced_dimension", "-")
            sec = r.get("runtime_seconds", 0.0)
            print(
                f"{order:^10} | {dimension!s:^6} | {ref:^14.6f} | {trans:^14.6f} | "
                f"{abso:^14.6f} | {sec:^8.2f}"
            )
        print("=" * 82)

    # 2. PNG グラフの描画と保存
    n_wl = len(wavelengths)
    fig, axes = plt.subplots(
        1,
        n_wl,
        figsize=(6 * n_wl if n_wl > 1 else 6.5, 4.8),
        dpi=300,
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for idx, wl in enumerate(wavelengths):
        ax = axes_flat[idx]
        records = sorted(data_by_wl[wl], key=lambda x: x["order"])
        orders = [r["order"] for r in records]
        r_vals = [r["reflectance"] for r in records]
        t_vals = [r["transmittance"] for r in records]
        a_vals = [r["absorptance"] for r in records]

        ax.plot(orders, r_vals, marker="o", linewidth=1.8, label="Reflectance (R)")
        ax.plot(orders, t_vals, marker="s", linewidth=1.8, label="Transmittance (T)")
        ax.plot(orders, a_vals, marker="^", linewidth=1.8, label="Absorptance (A)")

        ax.set_title(rf"$\lambda = {wl:g}$ nm", fontsize=12, fontweight="bold")
        ax.set_xlabel("Fourier Order ($M$)", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Optical Response ($R, T, A$)", fontsize=11)
        ax.set_xticks(orders)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(frameon=True, fontsize=9)

    fig.suptitle("Convergence vs. Fourier Diffraction Order ($M$)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_png_path, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[Success] Convergence plot saved to: {output_png_path.resolve()}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refine only the Fourier order M at fixed z-slice count."
    )
    add_shared_arguments(parser)
    parser.add_argument("--orders", default="10,12,14,16,18,20,22")
    parser.add_argument("--fixed-slices", type=int, default=32)
    parser.add_argument("--grid", type=int, default=256)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=_PACKAGE_ROOT / "results" / "order" / "pmma_gold_order",
    )
    args = parser.parse_args()

    candidates = parse_int_list(args.orders, "orders")
    fixed = NumericalConfig(
        order=candidates[0],
        slices=args.fixed_slices,
        grid=args.grid,
        radial_mapping=args.radial_mapping,
    )
    report, report_path = run_axis_convergence(
        axis="order", candidates=candidates, fixed=fixed, args=args
    )

    # チェックポイントファイルからデータを読み出して表示・描画
    checkpoint_path = args.output_prefix.with_name(
        args.output_prefix.name + "_checkpoint.json"
    )
    png_path = report_path.with_suffix(".png")
    display_and_plot_results(checkpoint_path, png_path)

    print(
        json.dumps(
            {
                "status": report.get("status"),
                "recommendation": report.get("recommendation"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"report json: {report_path.resolve()}")
    return 0 if report.get("status") == "converged" else 2


if __name__ == "__main__":
    raise SystemExit(main())
