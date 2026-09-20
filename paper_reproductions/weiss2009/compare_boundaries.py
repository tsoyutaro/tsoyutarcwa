"""Controlled boundary-projection experiment for the lossless Weiss Fig. 3.

All methods use exactly the same P,Q,kz,W,V. The direct pullback is a
different finite Fourier projection, not a reproduction of Weiss's solver.
Normal incidence, orthogonal cell, vacuum ports and a single layer only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper_reproductions.weiss2009 import reproduce as r
from paper_reproductions.weiss2009.diagnose_fig3 import condition, flux, relative, save, scalar


def direct_pullback(sim):
    """B: Cartesian plane waves -> covariant Fourier coefficients.

    [Eu,Ev]^t = J^t [Ex,Ey]^t; integrate with du dv, not dx dy.
    This is computed independently by quadrature, never by inverting T.
    """
    m = sim.asr_mappings[-1]
    n = sim.order_N
    nx, ny = m.x.shape
    weights = torch.stack((m.x_u, m.y_u, m.x_v, m.y_v)).to(sim._dtype)
    rows_x = torch.remainder(sim.order_x, nx).long()
    rows_y = torch.remainder(sim.order_y, ny).long()
    blocks = torch.empty((4, n, n), dtype=sim._dtype, device=sim._device)
    for col, (kx, ky) in enumerate(zip(sim.Kx_norm_dn.flatten(), sim.Ky_norm_dn.flatten())):
        plane = torch.exp(1j * sim.omega * (kx * m.x + ky * m.y))
        coefficients = torch.fft.fft2(weights * plane[None], dim=(-2, -1)) / (nx * ny)
        blocks[:, :, col] = coefficients[:, rows_x[:, None], rows_y[None, :]].reshape(4, n)
    return torch.cat((torch.cat((blocks[0], blocks[1]), dim=1),
                      torch.cat((blocks[2], blocks[3]), dim=1)), dim=0)


def connect(w, v, exterior_e, exterior_h, phase, incident):
    """Eq.29 algebra in a common boundary basis; external amplitudes stay Cartesian."""
    we = torch.linalg.solve(w, exterior_e)
    vh = torch.linalg.solve(v, exterior_h)
    a, b = we + vh, we - vh
    xb = phase[:, None] * b
    core = a - xb @ torch.linalg.solve(a, xb)
    # Only two incident columns are needed, avoiding full output S matrices.
    bi, ai = b @ incident, a @ incident
    reflection_rhs = xb @ torch.linalg.solve(a, phase[:, None] * ai) - bi
    transmission_rhs = phase[:, None] * (ai - b @ torch.linalg.solve(a, bi))
    reflected = torch.linalg.solve(core, reflection_rhs)
    transmitted = torch.linalg.solve(core, transmission_rhs)
    forward_top = (ai + b @ reflected) / 2
    backward_top = (bi + a @ reflected) / 2
    forward_bottom = a @ transmitted / 2
    backward_bottom = b @ transmitted / 2
    return reflected, transmitted, {
        "forward_propagation": relative(forward_bottom, phase[:, None] * forward_top),
        "backward_propagation": relative(backward_top, phase[:, None] * backward_bottom),
        "reflection_solve": relative(core @ reflected, reflection_rhs),
        "transmission_solve": relative(core @ transmitted, transmission_rhs)}, a, core


def summarize(sim, reflected, transmitted, incident):
    vf = sim.Vf
    incoming = flux(incident, vf @ incident)
    reflection = flux(reflected, vf @ reflected) / incoming
    transmission = flux(transmitted, vf @ transmitted) / incoming
    return {pol: {"R": scalar(reflection[i]), "T": scalar(transmission[i]),
                  "A": scalar(1 - reflection[i] - transmission[i])}
            for i, pol in enumerate(("x", "y"))}


@torch.inference_mode()
def run_case(args, order, profile, directory):
    grid = args.identity_grid if profile == "identity" else args.grid
    sim = r._new_simulation(frequency_thz=r.DIELECTRIC_FREQUENCY_THZ,
        order=order, period_um=r.DIELECTRIC_PERIOD_UM, profile=profile, grid=grid,
        dtype=torch.complex128, device=torch.device(args.device),
        cascade="redheffer", smatrix_size="half", use_symmetry=args.use_symmetry)
    r._add_circle(sim, height_um=r.DIELECTRIC_HEIGHT_UM, radius_um=r.DIELECTRIC_RADIUS_UM,
                  epsilon=r.DIELECTRIC_EPSILON, grid=grid)
    sim.solve_global_smatrix()
    w, v = sim.E_eigvec_uv[-1], sim.H_eigvec_uv[-1]
    e, h = sim.E_eigvec[-1], sim.H_eigvec[-1]
    transform = sim.asr_T_matrices[-1]
    eye = torch.eye(w.shape[0], dtype=w.dtype, device=w.device)
    orders = torch.cartesian_prod(sim.order_x, sim.order_y)
    zero = int(torch.nonzero((orders == 0).all(dim=1))[0, 0])
    incident = eye[:, [zero, zero + sim.order_N]]
    phase = torch.exp(1j * sim.omega * sim.kz_norm[-1] * sim.thickness[-1])
    baseline_r, baseline_t = sim.S[1] @ incident, sim.S[0] @ incident
    report = {"order": order, "profile": profile, "grid": grid, "G": r.INTERFACE_SLOPE,
        "use_symmetry": args.use_symmetry,
        "symmetry_diagnostics": sim.group_theory_diagnostics,
        "status": "partial", "methods": {"production": {
            "power": {p: r.power_for_polarization(sim, p) for p in ("x", "y")},
            "flux_power": summarize(sim, baseline_r, baseline_t, incident)}}}
    save(directory / "report.json", report)
    print("  building direct external-field pullback B", flush=True)
    pullback = direct_pullback(sim)
    report["projection"] = {"BT_identity_residual": relative(pullback @ transform, eye),
                            "TB_identity_residual": relative(transform @ pullback, eye)}
    if not args.skip_conditions:
        report["projection"].update(T=condition(transform), B=condition(pullback))
    # The inverse-T control is algebraically equivalent to the Cartesian method.
    # Direct quadrature B is the actual alternative discretization.
    for name in ("cartesian_control", "inverse_T_control", "direct_pullback"):
        print("  connecting: " + name, flush=True)
        if name == "cartesian_control":
            wi, vi, ex, hx = e, h, eye, sim.Vf
        else:
            ex = torch.linalg.solve(transform, eye) if name == "inverse_T_control" else pullback
            wi, vi, hx = w, v, ex @ sim.Vf
        rr, tt, residuals, a, core = connect(wi, vi, ex, hx, phase, incident)
        result = {"power": summarize(sim, rr, tt, incident), "residuals": residuals,
            "relative_amplitude_difference_from_production": {
                "reflection": relative(rr, baseline_r), "transmission": relative(tt, baseline_t)}}
        # Independently reconstruct internal fields from projected exterior fields.
        et = torch.linalg.solve(wi, ex @ (incident + rr))
        ht = torch.linalg.solve(vi, hx @ (incident - rr))
        eb = torch.linalg.solve(wi, ex @ tt)
        hb = torch.linalg.solve(vi, hx @ tt)
        incoming = flux(incident, sim.Vf @ incident)
        result["surface_flux"] = {
            "exterior_top": (flux(incident + rr, sim.Vf @ (incident - rr)) / incoming).cpu().tolist(),
            "exterior_bottom": (flux(tt, sim.Vf @ tt) / incoming).cpu().tolist(),
            "internal_covariant_top": (flux(w @ et, v @ ht) / incoming).cpu().tolist(),
            "internal_covariant_bottom": (flux(w @ eb, v @ hb) / incoming).cpu().tolist()}
        if not args.skip_conditions:
            result["conditions"] = {"boundary_A": condition(a), "boundary_core": condition(core)}
        report["methods"][name] = result
        save(directory / "report.json", report)
        print("    A_x={:.9g}, A_y={:.9g}".format(result["power"]["x"]["A"], result["power"]["y"]["A"]), flush=True)
    report["status"] = "complete"
    save(directory / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", default="11,14,15")
    parser.add_argument("--profiles", default="identity,weiss2009")
    parser.add_argument("--grid", type=int, default=512)
    parser.add_argument("--identity-grid", type=int, default=1024)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--skip-conditions", action="store_true")
    parser.add_argument("--use-symmetry", action="store_true",
                        help="Use complete C2v eigensystem blocks, retaining all modes")
    parser.add_argument("--output-dir", type=Path, default=Path("results/boundary_comparison"))
    args = parser.parse_args()
    orders, profiles = [int(n) for n in args.orders.split(",")], args.profiles.split(",")
    if min(orders) < 0 or any(p not in r.PROFILES for p in profiles):
        parser.error("Invalid order or profile")
    if min(args.grid, args.identity_grid) < max(32, 4 * max(orders) + 4):
        parser.error("Both grids must be at least max(32, 4*max_order+4)")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=time.strftime("run_%Y%m%d_%H%M%S_"), dir=args.output_dir))
    source_root = Path(r.__file__).parents[2]
    sources = [Path(__file__), Path(r.__file__), Path(__file__).with_name("diagnose_fig3.py")] + sorted((source_root / "rcwa_ext").glob("*.py"))
    save(root / "metadata.json", {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
         "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
         "dtype": "complex128", "cascade": "redheffer", "G": r.INTERFACE_SLOPE,
         "frequency_THz": r.DIELECTRIC_FREQUENCY_THZ, "period_um": r.DIELECTRIC_PERIOD_UM,
         "radius_um": r.DIELECTRIC_RADIUS_UM, "height_um": r.DIELECTRIC_HEIGHT_UM,
         "epsilon": r.DIELECTRIC_EPSILON,
         "source_sha256": {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}})
    print("Results: " + str(root.resolve()), flush=True)
    failures = 0
    for profile in profiles:
        for order in orders:
            directory = root / (profile + "_N" + str(order))
            directory.mkdir()
            print("Starting " + directory.name, flush=True)
            try:
                run_case(args, order, profile, directory)
            except Exception:
                failures += 1
                save(directory / "error.json", {"traceback": traceback.format_exc()})
                traceback.print_exc()
            if args.device == "cuda":
                torch.cuda.empty_cache()
    return int(failures > 0)


if __name__ == "__main__":
    sys.exit(main())
