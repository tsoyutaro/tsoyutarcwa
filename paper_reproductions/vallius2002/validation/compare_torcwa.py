"""Optional independent native torcwa check of the Fig. 11 TE calculation.

This uses native vector eigensystems, interfaces, cascade and power. Only
the raster Fourier integral is replaced by an independent antiderivative
when --grid=0, so numerical rasterization does not obscure the comparison.
Requires the repository's optional torch/torcwa installation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torcwa

from ..geometries import cylinder_layers
from ..solver import PreparedStack


class ExactIntervalRCWA(torcwa.rcwa):
    def _material_conv(self, raster):
        differences = (self.order_x[:, None] - self.order_x[None, :]).cpu().numpy()
        frequency = 2 * np.pi * differences / self.current_spec.period
        convolution = np.zeros(frequency.shape, complex)
        nonzero = differences != 0
        for left, right, epsilon in zip(self.current_spec.breaks[:-1],
                                       self.current_spec.breaks[1:], self.current_spec.epsilon):
            integral = np.full(frequency.shape, (right - left) / self.current_spec.period, complex)
            integral[nonzero] = (np.exp(-1j * frequency[nonzero] * left)
                                 - np.exp(-1j * frequency[nonzero] * right)) / (
                                     1j * frequency[nonzero] * self.current_spec.period)
            convolution += epsilon * integral
        return torch.tensor(convolution, dtype=self._dtype, device=self._device)


def native_power(sim, orders, port):
    amplitudes = [sim.S_parameters(orders, direction="forward", port=port,
                                  polarization=pol, power_norm=True) for pol in ("yy", "xy")]
    return float(sum(torch.sum(torch.abs(value) ** 2).item() for value in amplitudes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=0, help="0: exact independent integrals; positive: native raster.")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "diagnostics" / "torcwa_check.json")
    args = parser.parse_args()
    if args.grid < 0:
        parser.error("grid must be nonnegative")
    torch.set_num_threads(1)
    layers = cylinder_layers(120)
    dedicated = PreparedStack(layers, "TE", "fmm", 15, diagnostics=False)
    rows = []
    with torch.inference_mode():
        for wavelength in (0.92, 0.93, 0.96):
            cls = torcwa.rcwa if args.grid else ExactIntervalRCWA
            sim = cls(1 / wavelength, [7, 0], [1.0, 1.0], dtype=torch.complex128,
                      device=torch.device("cpu"), stable_eig_grad=False)
            sim.add_input_layer(eps=1.0)
            sim.add_output_layer(eps=1.0)
            sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
            for layer in layers:
                if args.grid:
                    coordinate = (np.arange(args.grid) + 0.5) / args.grid
                    index = np.searchsorted(layer.breaks, coordinate, side="right") - 1
                    raster = torch.tensor(np.array(layer.epsilon)[index, None], dtype=torch.complex128)
                else:
                    sim.current_spec = layer
                    raster = torch.ones((2, 1), dtype=torch.complex128)
                sim.add_layer(layer.thickness, eps=raster)
            sim.solve_global_smatrix()
            orders = [[m, 0] for m in range(-7, 8)]
            native = {"T0": native_power(sim, [[0, 0]], "transmission"),
                      "T": native_power(sim, orders, "transmission"),
                      "R": native_power(sim, orders, "reflection")}
            scalar = dedicated.solve(wavelength)
            rows.append({"wavelength": wavelength, "grid": args.grid, "slices": 120,
                         "native": native, "scalar": {k: scalar[k] for k in native},
                         "difference_scalar_minus_native": {k: scalar[k] - native[k] for k in native}})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"torcwa_version": getattr(torcwa, "__version__", "unknown"),
                                      "torch_version": torch.__version__, "rows": rows}, indent=2), encoding="utf-8")
    maximum = max(abs(value) for row in rows for value in row["difference_scalar_minus_native"].values())
    print(f"Maximum T0/T/R difference: {maximum:.6g}; {args.output}")
    if args.grid == 0 and maximum > 1e-7:
        raise SystemExit("Independent native comparison exceeded tolerance 1e-7")


if __name__ == "__main__":
    main()
