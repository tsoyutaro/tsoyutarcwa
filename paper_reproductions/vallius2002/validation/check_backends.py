"""Compare the CUDA tensor backend with the independent SciPy CPU path.

Example: python -m paper_reproductions.vallius2002.validation.check_backends --device cuda
``--device cpu`` exercises exactly the same tensor kernels without CUDA.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from ..devices import resolve_execution
from ..geometries import FIGURES, geometry_for_figure
from ..solver import PreparedStack


WAVELENGTHS = {6: (0.97, 1.0, 1.01), 7: (0.97, 1.0, 1.003),
               9: (1.161, 1.1723, 1.178), 11: (0.908, 0.928, 0.953)}


def compare(device="cuda", *, atol=2e-7, rtol=2e-6):
    execution = resolve_execution(device, "torch")
    started, checks = time.perf_counter(), []
    for figure, wavelengths in WAVELENGTHS.items():
        setting, layers = FIGURES[figure], geometry_for_figure(figure, slices=120)
        for method in ("fmm", "asr"):
            for label in setting["modes"]:
                options = dict(polarization=setting["polarization"], method=method,
                               harmonics=2 * label + 1, quadrature=192)
                cpu = PreparedStack(layers, **options, device="cpu", backend="scipy")
                tensor = PreparedStack(layers, **options, device=execution["device"], backend="torch")
                # All prepared operators must really reside on the selected device.
                for layer in tensor.prepared_layers:
                    assert str(layer.identity.device) == execution["device"]
                    if not layer.homogeneous:
                        assert str(layer.a.device) == execution["device"]
                for wavelength in wavelengths:
                    first, second = cpu.solve(wavelength), tensor.solve(wavelength)
                    error = {}
                    for key in ("T0", "T", "R", "A", "T_orders", "R_orders", "t", "r"):
                        np.testing.assert_allclose(second[key], first[key], atol=atol, rtol=rtol,
                                                   err_msg=f"Fig {figure}, {method}, M={label}, lambda={wavelength}, {key}")
                        error[key] = float(np.max(np.abs(np.asarray(second[key]) - first[key])))
                    assert first["evaluation_wavelength"] == second["evaluation_wavelength"]
                    checks.append({"figure": figure, "method": method, "modes": label,
                                   "harmonics": options["harmonics"], "wavelength": wavelength,
                                   "maximum_absolute_errors": error})
                print(f"PASS Fig {figure}, {method}, M={label}, device={execution['device']}", flush=True)
    return {"execution": execution, "passed": True, "case_count": len(checks),
            "atol": atol, "rtol": rtol, "seconds": time.perf_counter() - started,
            "cuda_executed": execution["device"].startswith("cuda"),
            "maximum_power_error": max(c["maximum_absolute_errors"][k] for c in checks for k in ("T0", "T", "R", "A")),
            "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", help="cuda, cuda:<index>, auto, or cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = compare(args.device)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"All {report['case_count']} cases passed; maximum power error={report['maximum_power_error']:.3g}; "
          f"CUDA executed={report['cuda_executed']}")


if __name__ == "__main__":
    main()
