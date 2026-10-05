#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # TSUBAMEなどGUIのない環境用
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Plot reflectance, transmittance, and absorptance spectra."
    )
    parser.add_argument(
        "csv_file",
        type=str,
        help="Input CSV file"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="spectrum.png",
        help="Output PNG file (default: spectrum.png)"
    )
    args = parser.parse_args()

    # =========================
    # CSV読み込み
    # =========================
    csv_path = Path(args.csv_file)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # 必要な列の確認
    required_columns = [
        "wavelength_nm",
        "reflectance",
        "transmittance",
        "absorptance",
    ]

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # =========================
    # データ取得
    # =========================
    wavelength = df["wavelength_nm"]
    reflectance = df["reflectance"]
    transmittance = df["transmittance"]
    absorptance = df["absorptance"]

    # =========================
    # プロット
    # =========================
    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.plot(
        wavelength,
        reflectance * 100,
        label="Reflectance (R)",
        linewidth=2.0,
    )

    ax.plot(
        wavelength,
        transmittance * 100,
        label="Transmittance (T)",
        linewidth=2.0,
    )

    ax.plot(
        wavelength,
        absorptance * 100,
        label="Absorptance (A)",
        linewidth=2.0,
    )

    # =========================
    # 軸・タイトル
    # =========================
    ax.set_xlabel("Wavelength (nm)", fontsize=13)
    ax.set_ylabel("Ratio (%)", fontsize=13)

    ax.set_xlim(wavelength.min(), wavelength.max())
    ax.set_ylim(0, 100)

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    ax.set_title("Optical Spectrum", fontsize=14)

    fig.tight_layout()

    # =========================
    # 保存
    # =========================
    output_path = Path(args.output)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()