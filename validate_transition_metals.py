"""Check supplied material constants, Fresnel flux, solver parity and checkpoint safety.

Small integration checks do not establish convergence at production orders.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from studies.shared.transition_metals import FILES, DATA, select_material, default_wavelengths
from studies.pmma_metal_motheye.converge import wavelengths
from studies.pmma_metal_motheye.model import (
    Geometry, Numerical, Study, PMMAIndex, content_hash, physical,
    choose_candidate, write_json,
)

ROOT = Path(__file__).resolve().parent


def core_checks():
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    result = []
    for symbol, filename in FILES.items():
        _, path, model = select_material(symbol)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["materials"][symbol]["original_sha256"]
        for w, n, k in zip(model.wavelength_nm, model.n, model.k):
            assert model(w) == complex(n, k) ** 2
        anchors, final = default_wavelengths(symbol)
        assert len(wavelengths(final)) == (8 if symbol == "W" else 61)
        for w in wavelengths(anchors) + wavelengths(final):
            assert model(w).imag > 0
        for w in (model.wavelength_nm[0] - 1, model.wavelength_nm[-1] + 1, float("nan")):
            try:
                model(w)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid wavelength or extrapolation was accepted.")
        if symbol == "W":
            assert model.wavelength_nm[0] == 667
            try:
                model(400)
            except ValueError:
                pass
            else:
                raise AssertionError("W table was extrapolated to 400 nm.")
        result.append({"name": f"{symbol} CSV hash, units, interpolation and range",
                       "passed": True, "samples": len(model.wavelength_nm),
                       "range_nm": [model.wavelength_nm[0], model.wavelength_nm[-1]]})
    geometry = Geometry()
    geometry.validate()
    for nz in (4, 40, 80, 100):
        layers = geometry.profile(nz)
        assert len(layers) == nz
        assert abs(sum(l.bottom_depth_nm-l.top_depth_nm for l in layers)-500) < 1e-10
        valleys = [l for l in layers if l.kind == "valley"]
        assert valleys[0].top_depth_nm == 470
    result.append({"name": "PMMA geometry and valley boundary", "passed": True})
    # A plateau at low order must not hide a change at higher order.
    spectra = {i: [{"wavelength_nm": 550, "reflectance": r, "transmittance": .1,
                    "absorptance": .9-r, "passivity_warning": False}]
               for i, r in enumerate((.2, .2001, .2002, .4), 1)}
    assert not choose_candidate([1, 2, 3, 4], spectra, .005)[1]
    result.append({"name": "Early plateau does not establish tail convergence", "passed": True})
    return result


def integration_checks(device):
    import torch
    from rcwa_solver_auto import AutoRCWA, Lattice, OutputSpec
    from studies.gold_motheye.converge import (
        simulate_case, GeometryConfig, NumericalConfig,
        _zero_order_x_source, _mean_poynting_z,
    )
    from studies.pmma_cr_motheye.model import Study as CrStudy, Geometry as CrGeometry
    from studies.pmma_cr_motheye.model import Numerical as CrNumerical
    torch.set_num_threads(1)
    target = torch.device(device)
    pmma_csv = ROOT / "studies/pmma_cr_motheye/data/Szczurowski.csv"
    validation_work = (ROOT / "studies/metal_motheye/validation/work").resolve()
    validation_work.mkdir(parents=True, exist_ok=True)
    checks = []
    for symbol in FILES:
        _, path, model = select_material(symbol)
        w = 680.0 if symbol == "W" else 550.0
        eps = model(w)
        flat = AutoRCWA(freq=200/w, order=[1, 1], lattice=Lattice.triangular(1),
                        outputs=OutputSpec(smatrix_size="half", fields="none"),
                        dtype=torch.complex128, device=target)
        flat.add_input_layer(eps=1)
        flat.add_output_layer(eps=eps)
        flat.set_incident_angle(0, 0)
        flat.solve_global_smatrix()
        src = _zero_order_x_source(flat)
        incident = _mean_poynting_z(src, flat.Vi, direction=1)
        r = -_mean_poynting_z(flat.S[1] @ src, flat.Vi, direction=-1)/incident
        p = _mean_poynting_z(flat.S[0] @ src, flat.Vo, direction=1)/incident
        fresnel = abs((1-eps**.5)/(1+eps**.5))**2
        error = max(abs(r-fresnel), abs(r+p-1))
        assert error < 1e-10
        checks.append({"name": f"{symbol} flat-interface Fresnel power", "passed": True,
                       "max_error": error, "R": r, "P_sub": p})
        # Same input geometry/material through Cr's shared engine, with two reductions.
        pure = [simulate_case(w, NumericalConfig(1, 2, 64), GeometryConfig(), model,
                             cascade="redheffer", use_symmetry=True,
                             symmetry_reduction=reduction, device=target, material_name=symbol)
                for reduction in ("d6-source", "cs-source")]
        metrics = ("reflectance", "absorptance_total", "motheye_absorptance", "power_into_substrate")
        assert all(not row["passivity_warning"] for row in pure)
        error = max(abs(pure[0][k]-pure[1][k]) for k in metrics)
        assert error < 2e-4
        checks.append({"name": f"{symbol} patterned pure-metal D6/Cs parity", "passed": True,
                       "max_error": error, "numerical": {"M": 1, "Nz": 2, "Q": 64},
                       "powers": {k: pure[0][k] for k in metrics}})
        with tempfile.TemporaryDirectory(dir=validation_work) as folder:
            assert Path(folder).resolve().is_relative_to(validation_work)
            shared = dict(pmma_csv=pmma_csv, pmma_policy="linear-to-400", device=target,
                          radial_mapping="outer", cascade="redheffer", use_symmetry=True,
                          signature="integration-"+symbol)
            generic = Study(geometry=Geometry(), metal_csv=path, material_name=symbol,
                            checkpoint=Path(folder)/"generic.json", **shared)
            new = generic.evaluate(Numerical(2, 4, 64), w)
            old = CrStudy(geometry=CrGeometry(), cr_csv=path,
                          checkpoint=Path(folder)/"cr_engine.json", **shared).evaluate(CrNumerical(2, 4, 64), w)
            assert physical(new) and new["material"] == symbol
            assert "epsilon_metal_real" in new and "epsilon_cr_real" not in new
            error = max(abs(new[k]-old[k]) for k in ("reflectance", "transmittance", "absorptance"))
            assert error < 1e-12
            restored = Study(geometry=Geometry(), metal_csv=path, material_name=symbol,
                             checkpoint=Path(folder)/"generic.json", **shared)
            with patch("studies.pmma_gold_motheye.common.simulate_case", side_effect=AssertionError("Unexpected repeat solve")):
                assert restored.evaluate(Numerical(2, 4, 64), w) == new
            changed = dict(shared, signature="different-material")
            try:
                Study(geometry=Geometry(), metal_csv=path, material_name=symbol,
                      checkpoint=Path(folder)/"generic.json", **changed)
            except ValueError:
                pass
            else:
                raise AssertionError("Mismatched checkpoint signature was accepted.")
            checks.append({"name": f"PMMA/{symbol} Cr-engine parity and resume safety", "passed": True,
                           "max_error": error, "numerical": {"M": 2, "Nz": 4, "Q": 64},
                           "powers": {k: new[k] for k in ("reflectance", "transmittance", "absorptance")}})
    return checks


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--integration", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--json", type=Path,
                   default=ROOT / "studies/metal_motheye/validation/verification_20261011.json")
    args = p.parse_args(argv)
    checks = core_checks()
    if args.integration:
        checks.extend(integration_checks(args.device))
    payload = {"passed": True, "scope": "software and small-case physics validation; not production convergence",
               "checks": checks}
    write_json(args.json, payload)
    print(json.dumps({"passed": True, "checks": len(checks), "json": str(args.json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
