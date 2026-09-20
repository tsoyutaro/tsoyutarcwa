"""Diagnose the lossless Fig. 3 calculation without changing its solver.

Run beside reproduce.py, or with python -m paper_reproductions.weiss2009.diagnose_fig3.
JSON files are saved after each case; existing files are never silently reused.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper_reproductions.weiss2009 import reproduce as r


def scalar(x):
    return float(x.detach().cpu())


def norm(x):
    return torch.linalg.vector_norm(x)


def relative(a, b):
    return scalar(norm(a - b) / (norm(a) + norm(b)).clamp_min(1e-300))


def condition(a):
    s = torch.linalg.svdvals(a)
    return {"sigma_max": scalar(s.max()), "sigma_min": scalar(s.min()),
            "condition_2": scalar(s.max() / s.min())}


def flux(e, h):
    """Signed z flux, omitting common 1/(2 Z0) and cell-area factors.

    For covariant u/v components the coordinate Jacobian cancels in the
    integrated cross product. Cartesian truncated modes use Parseval.
    """
    n = e.shape[0] // 2
    return (e[:n].conj() * h[n:] - e[n:].conj() * h[:n]).sum(dim=0).real


def clean(value):
    # Strict JSON: failed/nonfinite diagnostics are null, not nonstandard NaN.
    import math
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2), encoding="utf-8")
    temporary.replace(path)


@torch.inference_mode()
def diagnose(order, profile, grid, args, case_dir):
    started = time.perf_counter()
    sim = r._new_simulation(
        frequency_thz=r.DIELECTRIC_FREQUENCY_THZ, order=order,
        period_um=r.DIELECTRIC_PERIOD_UM, profile=profile, grid=grid,
        dtype=getattr(torch, args.dtype), device=torch.device(args.device),
        cascade=args.cascade, smatrix_size="half")
    solves = []
    original_solve = sim._solve

    def audited_solve(a, b):
        x = original_solve(a, b)
        caller = inspect.currentframe().f_back
        residual = norm(a @ x - b)
        solves.append({"caller": caller.f_code.co_name, "line": caller.f_lineno,
                       "size": list(a.shape),
                       "backward_error_frobenius": scalar(residual /
                           (norm(a) * norm(x) + norm(b)).clamp_min(1e-300)),
                       "relative_rhs_residual": scalar(residual / norm(b).clamp_min(1e-300))})
        return x

    sim._solve = audited_solve
    r._add_circle(sim, height_um=r.DIELECTRIC_HEIGHT_UM,
                  radius_um=r.DIELECTRIC_RADIUS_UM,
                  epsilon=r.DIELECTRIC_EPSILON, grid=grid)
    sim.solve_global_smatrix()
    # The audit covers production solves only. Diagnostics below use separate solves.
    sim._solve = original_solve
    power = {pol: r.power_for_polarization(sim, pol) for pol in ("x", "y")}
    p, q = sim.P[-1], sim.Q[-1]
    w, v = sim.E_eigvec_uv[-1], sim.H_eigvec_uv[-1]
    e, h = sim.E_eigvec[-1], sim.H_eigvec[-1]
    transform, kz = sim.asr_T_matrices[-1], sim.kz_norm[-1]
    pqw, wk2 = p @ (q @ w), w * kz.square()[None, :]
    per_mode = torch.linalg.vector_norm(pqw - wk2, dim=0) / (
        torch.linalg.vector_norm(pqw, dim=0) +
        torch.linalg.vector_norm(wk2, dim=0)).clamp_min(1e-300)
    uv_flux, cart_flux = flux(w, v), flux(e, h)
    mode_path = case_dir / "modes.csv"
    columns = torch.stack([kz.real, kz.imag, per_mode, uv_flux, cart_flux], dim=1).cpu().tolist()
    with mode_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["mode", "kz_real", "kz_imag", "pq_relative_residual",
                         "forward_uv_flux", "forward_cartesian_flux"])
        writer.writerows([i] + row for i, row in enumerate(columns))
    report = {"order": order, "profile": profile, "grid": grid,
              "cascade": args.cascade, "power": power, "production_solves": solves,
              "minimum_jacobian": scalar(sim.asr_mappings[-1].det_j.min()),
              "residuals": {"PQW_Wkz2": relative(pqw, wk2),
                            "QW_Vkz": relative(q @ w, v * kz[None, :]),
                            "PV_Wkz": relative(p @ v, w * kz[None, :]),
                            "TW_E": relative(transform @ w, e),
                            "TV_H": relative(transform @ v, h)},
              "kz": {"negative_real_count": int((kz.real < -1e-10).sum()),
                     "negative_imag_count": int((kz.imag < -1e-10).sum()),
                     "near_zero_count": int((kz.abs() < 1e-8).sum()),
                     "nonfinite_count": int((~torch.isfinite(kz)).sum())}}
    # Save useful results before the more expensive SVD diagnostics.
    save(case_dir / "report.json", dict(report, status="partial"))
    print("  modal residuals and power saved; checking boundaries", flush=True)
    identity = torch.eye(e.shape[0], dtype=e.dtype, device=e.device)
    ei = torch.linalg.solve(e, identity)
    hv = torch.linalg.solve(h, sim.Vf)
    a, b = ei + hv, ei - hv
    phase = torch.exp(1j * sim.omega * kz * sim.thickness[-1])
    xb = phase[:, None] * b
    core = a - xb @ torch.linalg.solve(a, xb)
    # Layer S matrices reference Vf on both sides. Inspect two incident zeroth
    # Cartesian polarizations, independently of external port normalization.
    orders = torch.cartesian_prod(sim.order_x, sim.order_y)
    zero = int(torch.nonzero((orders == 0).all(dim=1), as_tuple=False)[0, 0])
    incident = identity[:, [zero, zero + sim.order_N]]
    reflected = sim.layer_S21[-1] @ incident
    transmitted = sim.layer_S11[-1] @ incident
    e_top, h_top = incident + reflected, sim.Vf @ (incident - reflected)
    e_bottom, h_bottom = transmitted, sim.Vf @ transmitted
    et, ht = torch.linalg.solve(e, e_top), torch.linalg.solve(h, h_top)
    eb, hb = torch.linalg.solve(e, e_bottom), torch.linalg.solve(h, h_bottom)
    ft, bt, fb, bb = (et + ht) / 2, (et - ht) / 2, (eb + hb) / 2, (eb - hb) / 2
    report["boundary"] = {
        "forward_propagation_residual": relative(fb, phase[:, None] * ft),
        "backward_propagation_residual": relative(bt, phase[:, None] * bb),
        "top_E_reconstruction": relative(e @ et, e_top),
        "top_H_reconstruction": relative(h @ ht, h_top),
        "bottom_E_reconstruction": relative(e @ eb, e_bottom),
        "bottom_H_reconstruction": relative(h @ hb, h_bottom)}
    incoming_flux = flux(incident, sim.Vf @ incident)
    report["surface_flux_normalized"] = {
        "cartesian_top": (flux(e_top, h_top) / incoming_flux).cpu().tolist(),
        "cartesian_bottom": (flux(e_bottom, h_bottom) / incoming_flux).cpu().tolist(),
        "covariant_top": (flux(w @ et, v @ ht) / incoming_flux).cpu().tolist(),
        "covariant_bottom": (flux(w @ eb, v @ hb) / incoming_flux).cpu().tolist()}
    report["conditions"] = {}
    if not args.skip_conditions:
        for name, matrix in [("T", transform), ("W_uv", w), ("E_cartesian", e),
                             ("H_cartesian", h), ("boundary_A", a), ("boundary_core", core)]:
            print("  SVD condition: " + name, flush=True)
            report["conditions"][name] = condition(matrix)
            save(case_dir / "report.json", dict(report, status="partial"))
    report.update(status="complete", elapsed_s=time.perf_counter() - started)
    save(case_dir / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", default="11,14,15")
    parser.add_argument("--profiles", default="identity,weiss2009")
    parser.add_argument("--grid", type=int, default=512)
    parser.add_argument("--identity-grid", type=int, default=1024)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["complex64", "complex128"], default="complex128")
    parser.add_argument("--cascade", choices=["redheffer", "algo2a"], default="algo2a")
    parser.add_argument("--skip-conditions", action="store_true", help="Skip expensive SVDs")
    parser.add_argument("--output-dir", type=Path, default=Path("results/diagnostic_fig3_internal"))
    args = parser.parse_args()
    orders = [int(x) for x in args.orders.split(",")]
    profiles = args.profiles.split(",")
    if min(orders) < 0 or min(args.grid, args.identity_grid) < 1 or any(p not in r.PROFILES for p in profiles):
        parser.error("Invalid orders, grid, or profiles")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available; select --device cpu")
    # Unique run subdirectory avoids mixing checkpoints from different code/settings.
    import tempfile
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=time.strftime("run_%Y%m%d_%H%M%S_"), dir=args.output_dir))
    sources = [Path(__file__), Path(r.__file__)] + sorted((Path(r.__file__).parents[2] / "rcwa_ext").glob("*.py"))
    save(root / "metadata.json", {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
         "torch": torch.__version__, "python": sys.version, "cuda": torch.version.cuda,
         "G": r.INTERFACE_SLOPE, "frequency_THz": r.DIELECTRIC_FREQUENCY_THZ,
         "period_um": r.DIELECTRIC_PERIOD_UM, "radius_um": r.DIELECTRIC_RADIUS_UM,
         "height_um": r.DIELECTRIC_HEIGHT_UM, "epsilon": r.DIELECTRIC_EPSILON,
         "source_sha256": {str(p.relative_to(Path(r.__file__).parents[2])): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}})
    print("Results: " + str(root.resolve()), flush=True)
    failed = False
    for profile in profiles:
        for order in orders:
            grid = args.identity_grid if profile == "identity" else args.grid
            case = root / (profile + "_N" + str(order))
            case.mkdir()
            print("Starting " + case.name, flush=True)
            try:
                result = diagnose(order, profile, grid, args, case)
                print("  A_x={:.8g}, A_y={:.8g}".format(result["power"]["x"]["A"], result["power"]["y"]["A"]), flush=True)
            except Exception:
                failed = True
                save(case / "error.json", {"traceback": traceback.format_exc()})
                traceback.print_exc()
            if args.device == "cuda":
                torch.cuda.empty_cache()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
