"""Run Fe/Ni/Ti/W/V moth-eye convergence studies sequentially on one device."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from studies.shared.transition_metals import material_symbol, select_material, default_wavelengths
from studies.pmma_metal_motheye.converge import wavelengths
from studies.pmma_metal_motheye.model import PMMAIndex, write_json

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materials", default="Fe,Ni,Ti,W,V")
    parser.add_argument("--structures", choices=("both", "metal", "pmma"), default="both")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--orders", help="Override candidates for both structure families")
    parser.add_argument("--slices", help="Override candidates for both structure families")
    parser.add_argument("--grids", help="Override candidates for both structure families")
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--max-cycles", type=int)
    parser.add_argument("--run-final-spectrum", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--output-root", type=Path,
                        help="Optional batch output root, with metal/<symbol> and pmma/<symbol> subdirectories")
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    try:
        symbols = list(dict.fromkeys(material_symbol(s) for s in args.materials.split(",")))
        if not symbols:
            raise ValueError("At least one material is required.")
        if args.cpu_threads <= 0:
            raise ValueError("cpu-threads must be positive.")
        # Reject missing or out-of-range constants for all materials before any solve.
        pmma = PMMAIndex(ROOT / "studies/pmma_cr_motheye/data/Szczurowski.csv", "linear-to-400")
        for symbol in symbols:
            _, _, metal = select_material(symbol)
            anchors, final = default_wavelengths(symbol)
            for w in (*wavelengths(anchors), *(wavelengths(final) if args.run_final_spectrum else ())):
                metal(w)
                if args.structures != "metal":
                    pmma(w)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    structures = ("metal", "pmma") if args.structures == "both" else (args.structures,)
    destination = args.output_root.resolve() if args.output_root else None
    summary_path = args.summary or ((destination or ROOT / "studies/metal_motheye/results") /
                                    ("batch_plan.json" if args.dry_run else "batch_summary.json"))
    summary = {"status": "running", "dry_run": args.dry_run, "materials": symbols,
               "structures": list(structures), "requested_device": args.device,
               "scheduling": "sequential; no concurrent GPU processes",
               "jobs": []}
    write_json(summary_path, summary)
    failed = False
    unconverged = False
    for symbol in symbols:
        for structure in structures:
            package = "metal_motheye" if structure == "metal" else "pmma_metal_motheye"
            script = f"run_{package}_convergence.py"
            output = (destination / structure / symbol.lower() if destination else
                      ROOT / "studies" / package / "results" / symbol.lower())
            command = [sys.executable, "-u", str(ROOT / script), "--material", symbol,
                       "--device", args.device, "--cpu-threads", str(args.cpu_threads),
                       "--output-root", str(output)]
            for option in ("orders", "slices", "grids", "tolerance", "max_cycles"):
                value = getattr(args, option)
                if value is not None:
                    command.extend(("--" + option.replace("_", "-"), str(value)))
            for option in ("run_final_spectrum", "dry_run", "no_plots"):
                if getattr(args, option):
                    command.append("--" + option.replace("_", "-"))
            record = {"material": symbol, "structure": structure,
                      "output_root": str(output), "command": command, "status": "running"}
            report = output / ("metal_motheye_convergence.json" if structure == "metal"
                               else "pmma_metal_motheye_convergence.json")
            previous_stamp = report.stat().st_mtime_ns if report.exists() else None
            summary["jobs"].append(record)
            write_json(summary_path, summary)
            print(f"\n=== {symbol} / {structure} ===", flush=True)
            try:
                completed = subprocess.run(command, cwd=ROOT, check=False)
            except KeyboardInterrupt:
                record["status"] = "interrupted"
                summary["status"] = "interrupted"
                write_json(summary_path, summary)
                print("Interrupted. Completed cases remain in per-material checkpoints.")
                return 130
            record["exit_code"] = completed.returncode
            record["report"] = str(report)
            if args.dry_run:
                record["status"] = "planned" if completed.returncode == 0 else "plan_failed"
            elif (completed.returncode in (0, 2) and report.exists()
                  and report.stat().st_mtime_ns != previous_stamp):
                saved = json.loads(report.read_text(encoding="utf-8"))
                record["status"] = saved["status"]
                record["recommendation"] = saved["recommendation"]
            else:
                # A failed solve may leave an older report. Do not call that a success.
                record["status"] = "calculation_failed"
            failed |= (completed.returncode not in (0, 2)
                       or record["status"] in ("calculation_failed", "plan_failed"))
            unconverged |= completed.returncode == 2 and record["status"] not in ("calculation_failed", "plan_failed")
            write_json(summary_path, summary)
    summary["status"] = ("failed" if failed else "some_unconverged" if unconverged
                         else "planned" if args.dry_run else "converged")
    write_json(summary_path, summary)
    print(f"\nBatch status: {summary['status']}\nSummary: {summary_path.resolve()}")
    return 1 if failed else 2 if unconverged else 0


if __name__ == "__main__":
    raise SystemExit(main())
