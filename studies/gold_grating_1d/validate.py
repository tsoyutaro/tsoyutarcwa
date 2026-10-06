"""Small physical and operator checks, separate from discretization convergence."""
from __future__ import annotations

import argparse
import cmath
import math
import time
from common import HERE, load_config, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", default=str(HERE/"results"/"validation"))
    args = p.parse_args()
    import torch
    from solver import _simulation, port_results
    from rcwa_ext.asr import CustomRCWA_ASR_FR
    config, model = load_config(HERE/"config.json")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        p.error("CUDA is unavailable.")
    eps, wave, period = model(550), 550.0, 200.0
    n = cmath.sqrt(eps)
    fresnel_r = abs((1-n)/(1+n))**2
    checks = {}
    with torch.no_grad():
        start = time.perf_counter()
        sim = _simulation(3, wave, period, eps, device)
        flat = port_results(sim)
        error = max(abs(row["reflectance"]-fresnel_r) for row in flat.values())
        flux_error = max(abs(row["power_into_substrate"]-(1-fresnel_r)) for row in flat.values())
        checks["air_to_au_fresnel"] = {"passed": max(error, flux_error) <= 1e-9,
            "absolute_R_error": error, "absolute_substrate_flux_error": flux_error,
            "computed": flat, "runtime_seconds": time.perf_counter()-start}

        start = time.perf_counter()
        sim = _simulation(3, wave, period, eps, device)
        sim.add_layer(30.0/period, eps=eps, mu=1.0)
        homogeneous = port_results(sim)
        expected_p = (1-fresnel_r)*math.exp(-4*math.pi*n.imag*30/wave)
        error = max(max(abs(row["reflectance"]-fresnel_r),
                        abs(row["power_into_substrate"]-expected_p))
                    for row in homogeneous.values())
        checks["homogeneous_au_30nm_then_same_au"] = {"passed": error <= 1e-9,
            "maximum_absolute_error": error, "expected_power_into_substrate": expected_p,
            "computed": homogeneous, "runtime_seconds": time.perf_counter()-start}

        start = time.perf_counter()
        sim = _simulation(4, wave, period, 2.25, device)
        audits = [sim.add_ridge(0.25, fill, 2.25, 128, audit=True)
                  for fill in (0.15, 0.4, 0.7, 0.9)]
        lossless = port_results(sim)
        balance = max(abs(1-row["reflectance"]-row["power_into_substrate"])
                      for row in lossless.values())
        count = sim.order_N
        cross = max(float(torch.max(torch.abs(sim.S[1][:count, count:])).item()),
                    float(torch.max(torch.abs(sim.S[1][count:, :count])).item()))
        eigen_error = max(a["relative_eigen_residual"] for a in audits)
        checks["lossless_1d_grating_and_polarization_decoupling"] = {
            "passed": balance <= 1e-8 and cross <= 1e-8 and eigen_error <= 1e-9,
            "maximum_energy_balance_error": balance, "maximum_cross_polarization_amplitude": cross,
            "maximum_relative_eigen_residual": eigen_error, "computed": lossless,
            "runtime_seconds": time.perf_counter()-start}

        start = time.perf_counter()
        sim = _simulation(3, wave, period, eps, device)
        u = (torch.arange(64, dtype=torch.float64, device=device)+0.5)/64
        material = torch.where(torch.abs(u-0.5) < 0.3,
                               torch.as_tensor(eps, dtype=torch.complex128, device=device),
                               torch.ones(64, dtype=torch.complex128, device=device))[:, None]
        ones = torch.ones_like(material)
        p_generic, q_generic, _, _ = CustomRCWA_ASR_FR._build_asr_pq(
            sim, material, material, material, ones, ones, ones, factorization_rules=True)
        sim.add_ridge(0.1, 0.6, eps, 64)
        # The core FFT samples j/N; study quadrature samples (j+1/2)/N.
        phase = torch.exp(-1j*torch.pi*sim.order_x.to(torch.float64)/64)
        d = torch.diag(torch.cat((phase, phase)))
        errors = [float((torch.linalg.norm(actual-d @ generic @ d.mH)/
                         torch.linalg.norm(actual)).item())
                  for actual, generic in ((sim.P[-1], p_generic), (sim.Q[-1], q_generic))]
        checks["one_dimensional_Li_operators_vs_generic_core"] = {
            "passed": max(errors) <= 1e-10, "maximum_relative_operator_error": max(errors),
            "runtime_seconds": time.perf_counter()-start}

        start = time.perf_counter()
        analytic, analytic_reciprocal, _ = sim.ridge_convolutions(0.4, eps, 64, analytic=True)
        sampled, sampled_reciprocal, _ = sim.ridge_convolutions(0.4, eps, 65536)
        relative = max(float((torch.linalg.norm(a-b)/torch.linalg.norm(a)).item())
                       for a, b in ((analytic, sampled), (analytic_reciprocal, sampled_reciprocal)))
        checks["analytic_Fourier_coefficients_vs_fine_quadrature"] = {
            "passed": relative <= 5e-5, "maximum_relative_operator_error": relative,
            "runtime_seconds": time.perf_counter()-start}
    passed = all(c["passed"] for c in checks.values())
    for name, check in checks.items():
        print(f"{'PASS' if check['passed'] else 'FAIL'} {name}")
    from pathlib import Path
    path = Path(args.output_dir)/"report.json"
    write_json(path, {"passed": passed, "cases": checks, "torch": torch.__version__,
                     "scope": "Small physical/operator checks; not M/Nz/grid convergence."})
    print(f"report: {path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
