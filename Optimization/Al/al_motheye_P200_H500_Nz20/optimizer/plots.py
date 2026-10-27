"""Export publication-editable SVGs and numerical provenance, without clipping."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def make_plots(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    c = json.loads((output/"resolved_config.json").read_text(encoding="utf-8"))
    destination = output/"figures"
    destination.mkdir(exist_ok=True)
    for lattice in c["lattices"]:
        folder = output/lattice
        if not (folder/"report.json").exists():
            continue
        report = json.loads((folder/"report.json").read_text(encoding="utf-8"))
        best = report.get("best")
        if not best:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        ax = axes[0, 0]
        if (folder/"trials.csv").exists():
            with (folder/"trials.csv").open(encoding="utf-8", newline="") as file:
                trials = list(csv.DictReader(file))
            finished = [t for t in trials if t["state"] == "COMPLETE" and t.get("objective")]
            if finished:
                x, y = [int(t["trial_number"]) for t in finished], [100*float(t["objective"]) for t in finished]
                ax.scatter(x, y, label="Complete trial", alpha=.7)
                ax.step(x, np.minimum.accumulate(y), where="post", color="#d56318", label="Best completed")
        ax.set(xlabel="Optuna trial number", ylabel="Training mean R (%)")
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        ax = axes[0, 1]
        history_path = Path(best["trial_folder"])/"history.csv"
        if history_path.exists():
            with history_path.open(encoding="utf-8", newline="") as file:
                history = list(csv.DictReader(file))
            ax.plot([int(r["step"]) for r in history], [100*float(r["objective"]) for r in history], label="Measured iterate")
            ax.plot([int(r["step"]) for r in history], [100*float(r["best_objective"]) for r in history], label="Best measured", color="#d56318")
            best_path = Path(best["trial_folder"])/"best.json"
            if best_path.exists():
                selected = json.loads(best_path.read_text(encoding="utf-8"))
                ax.scatter([selected["best_step"]], [100*selected["best_loss"]], marker="*", s=110,
                           color="#d56318", label="Selected best state", zorder=5)
            ax.legend()
        ax.set(xlabel="Adam step", ylabel=f"Mean R at M={c['optimization_order']} (%)")
        ax = axes[1, 0]
        verification = report.get("verification", {})
        for condition in verification.get("best", {}).get("conditions", []):
            rows = sorted(condition["cases"].values(), key=lambda r: r["wavelength_nm"])
            rows = [r for r in rows if r["wavelength_nm"] in c["verification_wavelengths_nm"]]
            label = f"Best M={condition['order']}, Q={condition['grid']}"
            if not condition.get("complete"):
                label += " (partial)"
            ax.plot([r["wavelength_nm"] for r in rows], [100*r["reflectance"] for r in rows], label=label)
        initial = verification.get("baseline", {}).get("conditions", [])
        if initial:
            condition = max(initial, key=lambda r: (r["order"], r["grid"]))
            rows = sorted(condition["cases"].values(), key=lambda r: r["wavelength_nm"])
            rows = [r for r in rows if r["wavelength_nm"] in c["verification_wavelengths_nm"]]
            ax.plot([r["wavelength_nm"] for r in rows], [100*r["reflectance"] for r in rows], "--", color="gray",
                    label=f"Initial M={condition['order']}, Q={condition['grid']}"+(" (partial)" if not condition.get("complete") else ""))
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
        ax.set(xlabel="Wavelength (nm)", ylabel="Total R (%)")
        ax = axes[1, 1]
        radii = best["radii_nm"]
        depth = c["height_nm"]/c["layers"]
        for i, r in enumerate(radii):
            ax.add_patch(plt.Rectangle((-r, i*depth), 2*r, depth, facecolor="#bddff5", edgecolor="#205781", linewidth=.6))
        ax.set(xlim=(-c["period_nm"]*.55, c["period_nm"]*.55), ylim=(c["height_nm"], 0),
               xlabel="Position (nm)", ylabel="Depth (nm)")
        for a in axes.flat:
            a.grid(alpha=.2)
        fig.suptitle(f"NV Optuna + Adam; {lattice}; material={c.get('material_model', 'constant')}, Nz={c['layers']}; best trial {best['trial_number']}")
        fig.tight_layout()
        for extension in ("png", "svg"):
            fig.savefig(destination/f"optimization_{lattice}.{extension}", dpi=180)
        plt.close(fig)
