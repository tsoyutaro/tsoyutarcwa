"""Layer-resolved passivity and power diagnostics for metal/PMMA NV studies.

Uses the existing material-specific geometry builders without changing solver
equations. Every condition runs in a fresh subprocess. Failed/nonphysical
conditions are retained and do not stop other materials or wavelengths.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from itertools import product
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

import run_ag_nv_m14_diagnostics as diagnostic
import run_ag_nv_factorization_check as control
import run_metals_nv_spectra as spectra
import run_gold_motheye_order_nv as gold

ROOT = Path(__file__).resolve().parent
nv = diagnostic.nv
BACKENDS = dict(spectra.BACKENDS, Au=gold, Ag=spectra.cu_al)
DEFAULT_ORDERS = spectra.DEFAULT_ORDERS


def numbers(text, kind=int):
    values = sorted(set(kind(x.strip()) for x in text.split(",")))
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError("Use a nonempty comma-separated list of finite numbers.")
    return values


def orders(text):
    aliases = {name.lower(): name for name in BACKENDS}
    result = {}
    for part in text.split(","):
        name, value = part.strip().split(":")
        material, order = aliases[name.strip().lower()], int(value)
        if order < 1 or material in result:
            raise ValueError("Use distinct materials and positive orders.")
        result[material] = order
    return result


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--orders", default=DEFAULT_ORDERS)
    p.add_argument("--order-offsets", default="0", help="Offsets from each material's order; use --order-offsets=-2,0,2.")
    p.add_argument("--structures", choices=("both", "metal", "pmma"), default="both")
    p.add_argument("--wavelengths", default="400,550,700")
    p.add_argument("--slices", default="60", help="Comma-separated profile slice counts.")
    p.add_argument("--grids", default="512", help="Comma-separated normal-vector sampling grids.")
    p.add_argument("--factorizations", default="symmetric-nv", help="symmetric-nv and/or galerkin; NV only by default.")
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--quadrature", type=int, default=16)
    p.add_argument("--tolerance", type=float, default=.005, help="Refinement comparison in power fraction; .005 = .5 pp.")
    p.add_argument("--negative-loss-tolerance", type=float, default=1e-6,
                   help="Maximum magnitude of the total excited negative-loss contribution, as incident-power fraction.")
    p.add_argument("--pmma-csv", type=Path)
    p.add_argument("--w-csv", type=Path)
    p.add_argument("--output-root", type=Path, default=ROOT/"studies/metals_nv_diagnostics/results/adopted_settings_3wl")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--retry-failures", action="store_true")
    p.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    return p


def build_jobs(args):
    selected = orders(args.orders)
    offsets, slices, grids = numbers(args.order_offsets), numbers(args.slices), numbers(args.grids)
    waves = numbers(args.wavelengths, float)
    methods = list(dict.fromkeys(x.strip() for x in args.factorizations.split(",")))
    if (min(slices) < 1 or min(waves) <= 0 or args.cpu_threads < 1 or args.quadrature < 2
            or any(m not in control.LABELS for m in methods)
            or not math.isfinite(args.tolerance) or args.tolerance <= 0
            or not math.isfinite(args.negative_loss_tolerance) or args.negative_loss_tolerance < 0):
        raise ValueError("Check positive slices/wavelengths/threads/tolerances, quadrature >=2, and factorization names.")
    if args.device not in ("cpu", "cuda") and not (args.device.startswith("cuda:") and args.device[5:].isdigit()):
        raise ValueError("Use cpu, cuda or cuda:<index>.")
    structures = ["metal", "pmma"] if args.structures == "both" else [args.structures]
    extra_sources = {p.name: nv.digest(p) for p in (Path(__file__), Path(diagnostic.__file__),
                                                  Path(control.__file__), Path(spectra.__file__))}
    jobs = []
    for material, base_order in selected.items():
        backend = BACKENDS[material]
        for structure, offset, nz, q, wave, method in product(structures, offsets, slices, grids, waves, methods):
            order = base_order + offset
            if order < 1 or q < max(32, 4*order+4):
                raise ValueError("Every order must be positive and every Q >= max(32,4*M+4).")
            options = backend.build_parser().parse_args([])
            options.device, options.cpu_threads = args.device, args.cpu_threads
            options.slices, options.grid, options.start_order = nz, q, order
            if args.pmma_csv is not None:
                options.pmma_csv = args.pmma_csv
            if material == "W" and args.w_csv is not None:
                options.w_csv = args.w_csv
            base = backend.build_identity(options, material, structure, [wave])
            sources = dict(base.pop("source_sha256"), **extra_sources)
            # Scheduling choices are not part of a single physical case.
            for key in ("wavelengths_nm", "start_order", "step", "tolerance"):
                base.pop(key, None)
            identity = dict(version=1, material=material, structure=structure, order=order,
                slices=nz, grid=q, wavelength_nm=wave, factorization=method,
                device=args.device, cpu_threads=args.cpu_threads, quadrature=args.quadrature,
                backend_identity=base, sources=sources)
            relative = f"{structure}/{material.lower()}/{method}/M{order}_Nz{nz}_Q{q}/wl{wave:g}"
            jobs.append(dict(identity=identity, directory=relative))
    return jobs


def layer_metadata(base):
    g, nz = base["geometry"], base["slices"]
    if base["structure"] == "metal":
        return [dict(kind="metal_profile", core_radius_nm=None,
            radius_nm=g["tip_radius_nm"]+(g["base_radius_nm"]-g["tip_radius_nm"])*((i+.5)/nz)**g["profile_power"])
            for i in range(nz)]
    from studies.pmma_metal_motheye.model import Geometry
    result = []
    if g["metal_top_thickness_nm"] > 0:
        result.append(dict(kind="top_cap", core_radius_nm=None,
                           radius_nm=g["tip_radius_nm"]+g["metal_side_thickness_nm"]))
    for layer in Geometry(**g).profile(nz):
        result.append(dict(kind=layer.kind, core_radius_nm=layer.core_radius_nm,
            radius_nm=layer.core_radius_nm if layer.kind == "valley" else layer.outer_radius_nm))
    return result


@contextmanager
def instrument_solver(torch, operators, assembly_rows, output):
    """Capture the actual reduced loss operators before memory-safe release."""
    import rcwa_solver_auto as solver
    original = solver.AutoRCWA

    class DiagnosticNV(original):
        def _d6_symmetrized_cartesian_pq(self, *args, **kwargs):
            result = super()._d6_symmetrized_cartesian_pq(*args, **kwargs)
            self._diagnostic_pq = result[0], result[3], result[4]
            return result

        def _d6_source_eigendecomposition(self, *args, **kwargs):
            result = super()._d6_source_eigendecomposition(*args, **kwargs)
            embed, p, q = self._diagnostic_pq
            be, bh = (embed.mH@basis for basis in self._polarization_bases)
            ps, qs = be.mH@p@bh, bh.mH@q@be
            n = be.shape[0]//2
            pairing = be.mH@torch.cat((bh[n:], -bh[:n]), 0)
            le = diagnostic.antihermitian_part(pairing@qs)
            lh = diagnostic.antihermitian_part(pairing.mH@ps)
            operators.append((le.detach().cpu(), lh.detach().cpu()))
            layer = self._polarized_layers[-1]
            we, wh = torch.linalg.eigvalsh(le), torch.linalg.eigvalsh(lh)
            assembly_rows.append(dict(layer=len(operators),
                electric_loss_min_eigenvalue=float(we.min()),
                electric_loss_negative_eigenvalues=int((we < -1e-8).sum()),
                magnetic_loss_min_eigenvalue=float(wh.min()),
                magnetic_loss_negative_eigenvalues=int((wh < -1e-8).sum()),
                electric_modal_condition_number=float(torch.linalg.cond(layer["electric"])),
                eigenproblem_relative_residual=diagnostic.relative_residual(torch, ps@qs@layer["electric"],
                    layer["electric"]*layer["kz"][None, :].square()),
                symmetry_invariance_residual=float(self.group_theory_diagnostics[-1]["max_invariance_residual"])))
            diagnostic.write_csv(output/"operator_diagnostics.csv", assembly_rows)
            del self._diagnostic_pq
            return result

    solver.AutoRCWA = DiagnosticNV
    try:
        yield
    finally:
        solver.AutoRCWA = original


def compute_case(identity, output):
    import torch
    from studies.gold_motheye.converge import _zero_order_x_source, _mean_poynting_z
    torch.set_num_threads(identity["cpu_threads"])
    device = nv.initialize_device(torch, identity["device"])
    base = identity["backend_identity"]
    backend = BACKENDS[identity["material"]]
    metadata = layer_metadata(base)
    operators, assembly_rows, reconstructed = [], [], {}
    coated = identity["structure"] == "pmma"
    started = time.perf_counter()

    def observe(sim):
        if len(operators) != sim.layer_N or len(metadata) != sim.layer_N:
            raise RuntimeError("Every cap/profile/valley layer must have captured loss operators.")
        sim._diagnostic_radii_nm = [row["radius_nm"] for row in metadata]
        sim._diagnostic_period_nm = base["geometry"]["period_nm"]
        sim._diagnostic_order = identity["order"]
        source = _zero_order_x_source(sim)
        pin = _mean_poynting_z(source, sim.Vi, direction=1)
        r = -_mean_poynting_z(sim.S[1]@source, sim.Vi, direction=-1)/pin
        p = _mean_poynting_z(sim.S[0]@source, sim.Vo, direction=1)/pin
        t = p if coated else 0.
        nv.write_json(output/"far_field.json", dict(reflectance=r, transmittance=t,
            absorptance=1-r-t, power_into_substrate=p, pattern_absorptance=1-r-p,
            substrate_absorptance=0. if coated else p))
        reconstructed.update(diagnostic.reconstruct(torch, sim, operators, source, output, identity["quadrature"]))

    with control.factorization_scope(identity["factorization"]), instrument_solver(torch, operators, assembly_rows, output):
        original = backend.compute_case(base, identity["order"], identity["wavelength_nm"], device, observer=observe)
    result = dict(reconstructed)
    p = result["substrate_absorptance"]
    result["power_into_substrate"] = p
    result["transmittance"] = result["transmittance_far"] = p if coated else 0.
    result["absorptance"] = 1-result["reflectance"]-result["transmittance"]
    result["pattern_absorptance"] = result["motheye_absorptance"]
    result["substrate_absorptance"] = 0. if coated else p
    result["motheye_absorptance"] = None if coated else result["pattern_absorptance"]
    result["original_power_parity_max_error"] = max(abs(result[name]-original[name])
        for name in ("reflectance", "transmittance", "absorptance", "power_into_substrate"))
    result["original_powers"] = {name: original[name] for name in
        ("reflectance", "transmittance", "absorptance", "power_into_substrate")}
    result["far_field_physical"] = all(math.isfinite(value) and -1e-8 <= value <= 1+1e-8 for value in
        (result["reflectance"], result["transmittance"], result["absorptance"], p, result["pattern_absorptance"]))
    result["layer_flux_passive"] = result["negative_absorption_layers"] == 0
    result["physical_result"] = result["far_field_physical"] and result["layer_flux_passive"]
    result["diagnostic_checks_passed"] = (result["max_boundary_residual"] <= 1e-7
        and result["original_power_parity_max_error"] <= 1e-8
        and result["layer_sum_power_balance_error"] <= 1e-7 and result["max_volume_flux_error"] <= 1e-7)
    result.update(epsilon_real=original["epsilon_material_real"], epsilon_imag=original["epsilon_material_imag"],
        pmma_index=original["pmma_index"], total_pattern_layers=original["total_pattern_layers"],
        runtime_seconds=time.perf_counter()-started,
        peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
        interpretation="Passivity, diagnostic consistency and refinement stability are separate. Negative loss is retained. Mode proxies are not additive powers.")
    with (output/"layer_power.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    diagnostic.write_csv(output/"layer_power.csv", [dict(row, **meta) for row, meta in zip(rows, metadata)])
    return result


def verify_sources(identity):
    base = identity["backend_identity"]
    if nv.digest(base["metal_csv"]) != base["metal_sha256_lf"]:
        raise ValueError("Metal CSV changed after preflight.")
    if base["pmma"] and nv.digest(base["pmma"]["path"]) != base["pmma"]["sha256_lf"]:
        raise ValueError("PMMA CSV changed after preflight.")
    if any(nv.digest(ROOT/path) != value for path, value in identity["sources"].items()):
        raise ValueError("Diagnostic or solver source changed after preflight.")


def worker(plan_path):
    identity = json.loads(plan_path.read_text(encoding="utf-8"))["identity"]
    output = plan_path.parent
    signature, torch_module = nv.identity_signature(identity), None
    try:
        verify_sources(identity)
        import torch as torch_module
        with torch_module.no_grad():
            result = compute_case(identity, output)
        report = dict(status="diagnostics_complete", signature=signature, identity=identity, result=result)
        print(f"  R={100*result['reflectance']:.6f}%, T={100*result['transmittance']:.6f}%, "
              f"A={100*result['absorptance']:.6f}%; negative layers={result['negative_absorption_layers']}, "
              f"physical={result['physical_result']}, checks={result['diagnostic_checks_passed']}", flush=True)
    except Exception as error:
        oom = torch_module is not None and isinstance(error, torch_module.cuda.OutOfMemoryError) and identity["device"].startswith("cuda")
        report = dict(status="oom" if oom else "calculation_failed", signature=signature, identity=identity,
                      error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        if (output/"far_field.json").exists():
            report["far_field"] = json.loads((output/"far_field.json").read_text(encoding="utf-8"))
        print(f"  {report['status']}: {report['error']}; other cases will continue.", flush=True)
    nv.write_json(output/"report.json", report)
    return 0 if report["status"] == "diagnostics_complete" else 2


def collect(jobs, output, negative_loss_tolerance):
    rows = []
    for job in jobs:
        identity = job["identity"]
        path = output/job["directory"]/"report.json"
        saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else dict(status="pending")
        result = saved.get("result", {})
        complete = saved["status"] == "diagnostics_complete"
        sector = result.get("total_negative_loss_sector")
        sector_passed = sector is not None and abs(sector) <= negative_loss_tolerance
        row = {k: identity[k] for k in ("material", "structure", "factorization", "wavelength_nm", "order", "slices", "grid")}
        row.update(status=saved["status"], sample_checks_passed=bool(complete and result.get("physical_result")
            and result.get("diagnostic_checks_passed") and sector_passed), physical_result=result.get("physical_result"),
            diagnostic_checks_passed=result.get("diagnostic_checks_passed"), excited_negative_loss_passed=sector_passed if complete else None)
        for name, field in (("R_percent", "reflectance"), ("T_percent", "transmittance"), ("A_percent", "absorptance"),
                            ("pattern_A_percent", "pattern_absorptance"), ("substrate_A_percent", "substrate_absorptance"),
                            ("negative_loss_sector_pp", "total_negative_loss_sector")):
            row[name] = None if field not in result else 100*result[field]
        for field in ("negative_absorption_layers", "minimum_layer_absorptance", "max_boundary_residual",
                      "max_volume_flux_error", "original_power_parity_max_error", "layer_sum_power_balance_error",
                      "total_pattern_layers", "peak_cuda_allocated_bytes"):
            row[field] = result.get(field)
        row.update(error=saved.get("error", ""), directory=job["directory"])
        rows.append(row)
    return rows


def export(jobs, output, args):
    rows = collect(jobs, output, args.negative_loss_tolerance)
    diagnostic.write_csv(output/"diagnostics_summary.csv", rows)
    comparisons = []
    key_fields = ("material", "structure", "factorization", "wavelength_nm", "order", "slices", "grid")
    for axis in ("order", "slices", "grid"):
        groups = {}
        for row in rows:
            key = tuple(row[k] for k in key_fields if k != axis)
            groups.setdefault(key, []).append(row)
        for group in groups.values():
            ordered = sorted(group, key=lambda r: r[axis])
            for left, right in zip(ordered, ordered[1:]):
                change = {k: left[k] for k in key_fields if k != axis}
                change.update(axis=axis, coarse=left[axis], fine=right[axis])
                ready = left["status"] == right["status"] == "diagnostics_complete"
                fields = ("R_percent", "T_percent", "A_percent", "pattern_A_percent", "substrate_A_percent")
                for field in fields:
                    change["delta_"+field.replace("_percent", "_pp")] = abs(left[field]-right[field]) if ready else None
                change["maximum_change_pp"] = max(abs(left[f]-right[f]) for f in fields) if ready else None
                change["refinement_check_passed"] = bool(ready and left["sample_checks_passed"] and right["sample_checks_passed"]
                                                        and change["maximum_change_pp"] <= 100*args.tolerance)
                comparisons.append(change)
    # Axis-specific fixed settings produce different dictionaries; use a common header.
    if comparisons:
        fields = list(dict.fromkeys(k for row in comparisons for k in row))
        diagnostic.write_csv(output/"refinement_changes.csv", [{k: row.get(k) for k in fields} for row in comparisons])
    else:
        nv.atomic_text(output/"refinement_changes.csv", "axis,coarse,fine,maximum_change_pp,refinement_check_passed\n")
    counts = {status: sum(row["status"] == status for row in rows) for status in sorted({r["status"] for r in rows})}
    summary = dict(counts=counts, requested_cases=len(rows),
        sample_checks_passed=sum(row["sample_checks_passed"] for row in rows),
        negative_loss_tolerance=args.negative_loss_tolerance, refinement_tolerance_pp=100*args.tolerance,
        tested_axes=sorted({row["axis"] for row in comparisons}),
        refinement_checks=len(comparisons), refinement_checks_passed=sum(r["refinement_check_passed"] for r in comparisons),
        interpretation="Single M/Nz/Q only checks the sampled case; no convergence claim. Refinement checks refer only to the requested intervals and waves.")
    nv.write_json(output/"batch_summary.json", summary)
    return rows, summary


def plot_case(directory, identity):
    path = directory/"layer_power.csv"
    if not path.exists():
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    depth = [float(r["depth_mid_nm"]) for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot([float(r["depth_top_nm"]) for r in rows]+[float(rows[-1]["depth_bottom_nm"])],
                 [100*float(r["flux_top"]) for r in rows]+[100*float(rows[-1]["flux_bottom"])])
    axes[0].set(title="Net downward power", ylabel="Incident power (%)")
    axes[1].plot(depth, [float(r["layer_absorptance_pp"]) for r in rows])
    axes[1].set(title="Layer absorption", ylabel="Percentage points")
    axes[2].plot(depth, [100*(float(r["negative_electric_sector"])+float(r["negative_magnetic_sector"])) for r in rows])
    axes[2].set(title="Excited negative-loss contribution", ylabel="Percentage points")
    for ax in axes:
        ax.axhline(0, color="black", lw=.7)
        ax.set_xlabel("Depth (nm)")
        ax.grid(alpha=.25)
    fig.suptitle(f"{identity['material']} | {identity['structure']} | {identity['factorization']} | "
        f"{identity['wavelength_nm']:g} nm | M={identity['order']}, Nz={identity['slices']}, Q={identity['grid']}")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(directory/f"layer_diagnostics.{ext}", dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args.worker)
    try:
        jobs = build_jobs(args)
        output = args.output_root.resolve()
        # Check every requested cache before writing or launching anything.
        for job in jobs:
            path = output/job["directory"]/"report.json"
            if path.exists() and json.loads(path.read_text(encoding="utf-8")).get("signature") != nv.identity_signature(job["identity"]):
                raise ValueError(f"Saved settings/source changed at {path}; use a new --output-root.")
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    print(f"Metal diagnostics: {len(jobs)} conditions; structures={args.structures}; "
          f"orders={args.orders}; Nz={args.slices}; Q={args.grids}; wavelengths={args.wavelengths}\nOutput: {output}", flush=True)
    if args.dry_run:
        print("Dry run: no calculation or file changes. OOM, failed and nonphysical cases do not stop other cases.")
        return 0
    output.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        nv.write_json(output/"plan.json", dict(jobs=jobs))
        for i, job in enumerate(jobs):
            directory = output/job["directory"]
            directory.mkdir(parents=True, exist_ok=True)
            path = directory/"report.json"
            saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
            if saved and (saved["status"] == "diagnostics_complete" or not args.retry_failures):
                print(f"Reuse {job['directory']}: {saved['status']}", flush=True)
                continue
            identity = job["identity"]
            nv.write_json(directory/"plan.json", dict(identity=identity))
            print(f"\n=== {i+1}/{len(jobs)}: {job['directory']} ===", flush=True)
            process = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
                                      "--worker", str(directory/"plan.json")], check=False)
            if not path.exists() or json.loads(path.read_text(encoding="utf-8")).get("signature") != nv.identity_signature(identity):
                nv.write_json(path, dict(status="calculation_failed", signature=nv.identity_signature(identity), identity=identity,
                                        error=f"Worker exited {process.returncode} without a matching report."))
            export(jobs, output, args)
    rows, summary = export(jobs, output, args)
    if not args.no_plots:
        for row, job in zip(rows, jobs):
            if row["status"] == "diagnostics_complete":
                plot_case(output/job["directory"], job["identity"])
    print(f"Counts: {summary['counts']}; sampled checks passed: {summary['sample_checks_passed']}/{len(rows)}", flush=True)
    print(f"Summary: {output/'diagnostics_summary.csv'}", flush=True)
    return 0 if all(row["status"] == "diagnostics_complete" for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
