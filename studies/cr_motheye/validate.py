"""Validate Cr input units, passive signs, Fresnel power, and D6/Cs agreement."""
from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

from studies.shared.tabulated_nk import load_tabulated_nk
from studies.gold_motheye.converge import GeometryConfig, NumericalConfig, _choose_candidate

PACKAGE = Path(__file__).resolve().parent


def core_checks():
    model = load_tabulated_nk(PACKAGE / "data/Johnson.csv")
    assert len(model.wavelength_nm) == 49
    assert (model.wavelength_nm[0], model.wavelength_nm[-1]) == (188.0, 1937.0)
    assert abs(model(549) - complex(3.18, 3.33) ** 2) < 1e-12
    for w in (400, 550, 700):
        assert model(w).imag > 0
    for w in (187, 1938, float("nan")):
        try:
            model(w)
        except ValueError:
            pass
        else:
            raise AssertionError("Extrapolation/nonfinite input was accepted.")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "test.csv"
        path.write_text("wavelength_nm,n,k\n400,2,3\n700,3,4\n", encoding="utf-8")
        assert abs(load_tabulated_nk(path)(550) - 0.5 * (complex(2, 3)**2 + complex(3, 4)**2)) < 1e-12
        for bad in ("wl,n\n0.4,2\n0.7,3\nwl,k\n0.4,3\n0.8,4\n",
                    "wavelength_nm,n,k\n400,2,-3\n700,3,4\n",
                    "wavelength_nm,n,k\n400,nan,3\n700,3,4\n"):
            path.write_text(bad, encoding="utf-8")
            try:
                load_tabulated_nk(path)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid optical constants were accepted.")
    rows = {m: [{"convergence_metrics": {"reflectance": 1.1}, "passivity_warning": True}]
            for m in (1, 2, 3)}
    assert not _choose_candidate((1, 2, 3), rows, 0.005, reject_unphysical=True)[1]
    assert _choose_candidate((1, 2, 3), rows, 0.005)[1]  # Au legacy default is preserved.
    return {"name": "CSV units, samples, interpolation, invalid-data rejection and physical convergence gate", "passed": True}


def integration_checks(device):
    import torch
    from rcwa_solver_auto import AutoRCWA, Lattice, OutputSpec
    from studies.gold_motheye.converge import simulate_case, _zero_order_x_source, _mean_poynting_z
    model = load_tabulated_nk(PACKAGE / "data/Johnson.csv")
    checks = []
    for w in (400, 550, 700):
        eps = model(w)
        sim = AutoRCWA(freq=200/w, order=[1, 1], lattice=Lattice.triangular(1.0),
                       outputs=OutputSpec(smatrix_size="half", fields="none"),
                       dtype=torch.complex128, device=torch.device(device))
        sim.add_input_layer(eps=1.0)
        sim.add_output_layer(eps=eps)
        sim.set_incident_angle(0.0, 0.0)
        sim.solve_global_smatrix()
        src = _zero_order_x_source(sim)
        pin = _mean_poynting_z(src, sim.Vi, direction=1)
        r = -_mean_poynting_z(sim.S[1] @ src, sim.Vi, direction=-1) / pin
        p = _mean_poynting_z(sim.S[0] @ src, sim.Vo, direction=1) / pin
        fresnel = abs((1 - eps**0.5) / (1 + eps**0.5))**2
        error = max(abs(r - fresnel), abs(r + p - 1))
        assert error < 1e-10
        checks.append({"name": f"Flat air/Cr Fresnel interface {w} nm", "passed": True,
                       "max_error": error, "R": r, "P_sub": p})
    results = []
    for reduction in ("d6-source", "cs-source"):
        result = simulate_case(550, NumericalConfig(1, 2, 64), GeometryConfig(), model,
                               cascade="redheffer", use_symmetry=True,
                               symmetry_reduction=reduction, device=torch.device(device), material_name="Cr")
        assert not result["passivity_warning"]
        assert "epsilon_material_real" in result and "epsilon_gold_real" not in result
        assert all(math.isfinite(result[key]) and -1e-5 <= result[key] <= 1+1e-5
                   for key in ("reflectance", "absorptance_total", "motheye_absorptance", "power_into_substrate"))
        results.append(result)
    error = max(abs(results[0][key] - results[1][key]) for key in
                ("reflectance", "absorptance_total", "motheye_absorptance", "power_into_substrate"))
    assert error < 2e-4
    assert results[0]["reduced_dimension"] == 3 and results[1]["reduced_dimension"] == 7
    checks.append({"name": "Cr patterned stack D6/Cs cross-check", "passed": True,
                   "max_error": error, "results": results})
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integration", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", type=Path, default=PACKAGE / "validation/results.json")
    args = parser.parse_args(argv)
    checks = [core_checks()]
    if args.integration:
        checks.extend(integration_checks(args.device))
    payload = {"passed": True, "checks": checks}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
