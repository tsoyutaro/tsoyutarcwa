"""Replay a saved ellipse/rectangle convergence sweep with NVM smoothing off."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch

from rcwa_ext import Lattice, NVMOptions, OutputSpec
from rcwa_ext.config import _as_float

from .geometry import slices
from .model import Config, scattering_metrics
from .nv_convergence import cache_job, differences, fingerprint
from .nv_shapes import ShapeNVRCWA
from .run import save_csv, save_json


class UnsmoothedShapeNVRCWA(ShapeNVRCWA):
    """Use the centered shape normal and average tied periodic projectors."""

    def shape_projection(self, shape, half_x, half_y, nx, ny):
        minimum = max(32, 4 * max(int(v) for v in self.order) + 4)
        if min(nx, ny) < minimum:
            raise ValueError(f"NV normal grid requires Q >= {minimum}.")
        lx, ly = _as_float(self.L[0]), _as_float(self.L[1])
        x = torch.arange(nx, dtype=torch.float64, device=self._device) * (lx / nx) - lx / 2
        y = torch.arange(ny, dtype=torch.float64, device=self._device) * (ly / ny) - ly / 2
        dx, dy = torch.meshgrid(x, y, indexing="ij")
        vx, vy, length = self.normal_direction(shape, dx, dy, half_x, half_y)
        defined = length > 1e-14
        # At an edge of this orthogonal single-center cell, the tied image
        # flips one normal component. At a cell corner both signs are sampled.
        tie = (dx.abs() == lx / 2) | (dy.abs() == ly / 2)
        pxx = torch.where(defined, vx.square(), 0.5)
        pyy = torch.where(defined, vy.square(), 0.5)
        pxy = torch.where(defined & ~tie, vx * vy, 0.0)
        count_y = len(self.order_y)
        px = self.order_x.repeat_interleave(count_y)
        py = self.order_y.repeat(len(self.order_x))
        ix, iy = px[:, None] - px[None, :], py[:, None] - py[None, :]

        def conv(field):
            result = torch.fft.fft2(field.to(self._dtype))[ix, iy] / (nx * ny)
            return 0.5 * (result + result.mH)

        xx, yy, xy = conv(pxx), conv(pyy), conv(pxy)
        return torch.cat((torch.cat((xx, xy), 1), torch.cat((xy.mH, yy), 1)), 0)


@torch.no_grad()
def solve(knots, shape, index, wavelength, cfg):
    cfg.validate()
    scale = cfg.period_x_nm
    sim = UnsmoothedShapeNVRCWA(
        freq=scale / wavelength, order=[cfg.order, cfg.order],
        lattice=Lattice.rectangular(1., cfg.period_y_nm / scale),
        outputs=OutputSpec(smatrix_size="half", fields="none"),
        nvm=NVMOptions(grid=(cfg.grid, cfg.grid), use_lanczos=False, normal_mode="off"),
        dtype=torch.complex128, device=cfg.device,
        cascade="redheffer", verify_cascade=False,
    )
    sim.add_input_layer(eps=cfg.n_input**2)
    sim.add_output_layer(eps=cfg.n_substrate**2)
    sim.set_incident_angle(0., 0.)
    for a, b in slices(knots, cfg):
        sim.add_layer_shape_nv(cfg.height_nm / cfg.layers / scale, shape,
                               a / scale, b / scale, 1., index**2,
                               nx=cfg.grid, ny=cfg.grid)
    sim.solve_global_smatrix()
    return scattering_metrics(sim, cfg, wavelength)


def compare(rows, settings):
    summaries = []
    for shape, series in dict.fromkeys((r["shape"], r["series"]) for r in rows):
        field = "order" if series.startswith("orders") else "grid" if series == "grids" else "layers"
        group = [r for r in rows if r["shape"] == shape and r["series"] == series]
        levels = sorted({r[field] for r in group})
        for lower, upper in zip(levels, levels[1:]):
            for wavelength in sorted({r["wavelength_nm"] for r in group}):
                a, b = (next(r for r in group if r[field] == level and r["wavelength_nm"] == wavelength)
                        for level in (lower, upper))
                delta = differences(a, b)
                phase = delta["retardance_change_deg"]
                tphase = delta["transmission_phase_change_deg"]
                energy = max(a["energy_error"], b["energy_error"])
                passed = (phase is not None and tphase is not None
                          and max(phase, tphase) <= settings["phase_tol"]
                          and delta["power_change"] <= settings["power_tol"]
                          and energy <= settings["energy_tol"])
                summaries.append(dict(shape=shape, series=series, lower=lower, upper=upper,
                                      wavelength_nm=wavelength, **delta, energy_error=energy,
                                      passed=bool(passed)))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True,
                        help="Existing nv_convergence.py output directory defining the sweep")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shapes", choices=("rectangle", "ellipse"), nargs="+",
                        default=["rectangle", "ellipse"])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    ref = args.reference / "run_metadata.json"
    schedule = args.reference / "nv_convergence.csv"
    settings = json.loads(ref.read_text(encoding="utf-8"))["arguments"]
    if settings["mode"] != "converge":
        parser.error("--reference must be an nv_convergence.py converge run")
    if args.output.resolve() == args.reference.resolve():
        parser.error("Choose a separate --output for the OFF calculation")
    with schedule.open(newline="", encoding="utf-8-sig") as stream:
        cases = [row for row in csv.DictReader(stream) if row["shape"] in args.shapes]
    if {row["shape"] for row in cases} != set(args.shapes):
        parser.error("The reference does not contain all requested shapes")
    sources = fingerprint()
    sources[Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata = dict(normal_mode="off", use_lanczos=False, shapes=args.shapes,
                    device=args.device, threads=args.threads, reference=settings,
                    reference_csv_sha256=hashlib.sha256(schedule.read_bytes()).hexdigest(),
                    source_sha256=sources)
    path = args.output / "run_metadata.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != metadata:
        parser.error("Output settings or source code changed; choose a new --output")
    save_json(path, metadata)
    base = Config(period_x_nm=settings["period"], period_y_nm=settings["period"],
                  height_nm=settings["height"], n_substrate=settings["substrate_index"],
                  device=args.device)
    knots = [settings["x_knots"], settings["y_knots"]]
    rows = []
    on_off = []
    for case in cases:
        shape, series = case["shape"], case["series"]
        m, q, z = (int(case[k]) for k in ("order", "grid", "layers"))
        w = float(case["wavelength_nm"])
        cfg = replace(base, order=m, grid=q, layers=z)
        result = cache_job(args.output, f"{shape}_M{m}_Q{q}_Z{z}_w{w:g}",
                           lambda: solve(knots, shape, settings["index"], w, cfg))
        key = dict(shape=shape, series=series, order=m, grid=q, layers=z)
        rows.append(dict(**key, **result))
        on_off.append(dict(**key, wavelength_nm=w,
                           retardance_on_deg=float(case["retardance_deg"]),
                           retardance_off_deg=result["retardance_deg"],
                           Tx00_on=float(case["Tx00"]), Tx00_off=result["Tx00"],
                           Ty00_on=float(case["Ty00"]), Ty00_off=result["Ty00"]))
        save_csv(args.output / "nv_convergence.csv", rows)
    save_csv(args.output / "convergence_comparisons.csv", compare(rows, settings))
    save_csv(args.output / "on_off_comparisons.csv", on_off)


if __name__ == "__main__":
    main()
