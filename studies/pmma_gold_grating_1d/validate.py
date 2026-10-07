"""Small physical checks for PMMA transmission and Au absorption, not convergence."""
from __future__ import annotations

import argparse
import cmath
import copy
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from studies.pmma_gold_grating_1d.common import HERE, load_config, write_json
from studies.pmma_gold_grating_1d.geometry import Layer, build_layers


def run_checks(config, model, device_name):
    import torch
    from studies.pmma_gold_grating_1d.solver import simulation, simulate
    from studies.gold_grating_1d.solver import port_results
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    wave, period = 550., config["geometry"]["period_nm"]
    pmma, gold = model(wave)
    n_pmma, n_gold = math.sqrt(pmma.real), cmath.sqrt(gold)
    checks = {}
    with torch.no_grad():
        sim = simulation(3, wave, period, pmma, device)
        ports = port_results(sim)
        expected_r = ((1-n_pmma)/(1+n_pmma))**2
        error = max(max(abs(row["reflectance"]-expected_r),
                        abs(row["power_into_substrate"]-(1-expected_r))) for row in ports.values())
        checks["air_to_pmma_fresnel"] = {"passed": error < 1e-10, "maximum_absolute_error": error}

        film_config = copy.deepcopy(config)
        g = film_config["geometry"]
        g.update(top_width_nm=period, bottom_width_nm=period, gold_side_thickness_x_nm=0.,
                 gold_cap_width_nm=period, gold_cap_thickness_nm=30., gold_valley_thickness_nm=0.)
        film = simulate(film_config, {"order": 3, "slices": 2}, wave, (pmma, gold), device_name)
        rab, rbc = (1-n_gold)/(1+n_gold), (n_gold-n_pmma)/(n_gold+n_pmma)
        phase = cmath.exp(2j*math.pi*n_gold*30/wave)
        denominator = 1+rab*rbc*phase**2
        r = (rab+rbc*phase**2)/denominator
        t = (2/(1+n_gold))*(2*n_gold/(n_gold+n_pmma))*phase/denominator
        reference = {"reflectance": abs(r)**2, "transmittance": n_pmma*abs(t)**2}
        reference["absorptance"] = 1-reference["reflectance"]-reference["transmittance"]
        errors = {pol: {name: abs(row[name]-value) for name, value in reference.items()}
                  for pol, row in film["polarizations"].items()}
        maximum = max(v for values in errors.values() for v in values.values())
        checks["uniform_30nm_au_film_on_pmma"] = {"passed": maximum < 1e-9,
            "maximum_absolute_error": maximum, "reference": reference, "computed": film["polarizations"]}

        sim = simulation(4, wave, period, 2.25, device)
        # Lossless surrogate: gold-shaped regions have real epsilon=2.89.
        for layer in build_layers(config["geometry"], 4):
            sim.add_coated_layer(layer, (2.25, 2.89))
        ports = port_results(sim)
        balance = max(abs(1-row["reflectance"]-row["power_into_substrate"]) for row in ports.values())
        n = sim.order_N
        cross = max(float(torch.max(torch.abs(sim.S[1][:n, n:])).item()),
                    float(torch.max(torch.abs(sim.S[1][n:, :n])).item()))
        checks["lossless_three_material_relief"] = {"passed": max(balance, cross) < 1e-9,
            "maximum_energy_balance_error": balance, "maximum_cross_polarization_amplitude": cross}

        sim = simulation(3, wave, period, pmma, device)
        layer = Layer(0., 10., .23*period, .53*period, "air", "shell")
        direct, reciprocal, _ = sim.ridge_convolutions(layer, (pmma, gold), 1, analytic=True)
        grid = 65536
        x = (torch.arange(grid, dtype=torch.float64, device=device)+.5)/grid
        samples = torch.ones(grid, dtype=torch.complex128, device=device)
        samples[torch.abs(x-.5) < .53/2] = gold
        samples[torch.abs(x-.5) < .23/2] = pmma
        delta = sim.order_x[:, None]-sim.order_x[None, :]
        phase = torch.exp(-1j*torch.pi*delta/grid)
        reference = [(torch.fft.fft(samples)/grid)[delta]*phase,
                     (torch.fft.fft(1/samples)/grid)[delta]*phase]
        relative = max(float((torch.linalg.norm(a-b)/torch.linalg.norm(a)).item())
                       for a, b in zip((direct, reciprocal), reference))
        checks["three_material_coefficients_vs_fine_quadrature"] = {"passed": relative < 1e-4,
            "maximum_relative_error": relative, "quadrature_points": grid}

        coated = simulate(config, {"order": 4, "slices": 6}, wave, (pmma, gold), device_name)
        physical = all(-1e-8 <= value <= 1+1e-8 for row in coated["polarizations"].values() for value in row.values())
        residual = max(d["relative_eigen_residual"] for d in coated["selected_layer_diagnostics"])
        checks["au_coated_pmma_passivity_and_eigen_residual"] = {"passed": physical and residual < 1e-9,
            "maximum_relative_eigen_residual": residual, "computed": coated["polarizations"]}
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--output-dir", type=Path, default=HERE/"results/validation")
    args = p.parse_args()
    config, model = load_config(HERE/"config.json")
    started = time.perf_counter()
    checks = run_checks(config, model, args.device)
    passed = all(check["passed"] for check in checks.values())
    for name, result in checks.items():
        print(f"{'PASS' if result['passed'] else 'FAIL'} {name}")
    write_json(args.output_dir/"report.json", {"passed": passed, "cases": checks, "device": args.device,
               "runtime_seconds": time.perf_counter()-started,
               "scope": "Small physical checks, not order/slices convergence or deposition morphology validation."})
    print(f"report: {args.output_dir/'report.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
