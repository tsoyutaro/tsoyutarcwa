"""Reproduce the cylinder examples of Weiss et al. (2009).

The implementation uses the paper's matched-coordinate map (Eqs. 37-38),
the fixed-interface adaptive map (Eqs. 41-42), and the symmetric Fourier
factorization (Eqs. 29-36) implemented in :mod:`rcwa_ext`.

Examples
--------
Fast end-to-end check::

    python -m paper_reproductions.weiss2009.reproduce --study smoke --device cpu

Paper sweeps (dense eigensolves; CUDA is recommended)::

    python -m paper_reproductions.weiss2009.reproduce --study paper --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import torch

_PACKAGE_ROOT = Path(__file__).resolve().parent
_OUTPUTS_ROOT = Path(__file__).resolve().parents[2]
if str(_OUTPUTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_OUTPUTS_ROOT))

from rcwa_ext import CustomRCWA_ASR_FR


PAPER_DOI = "10.1364/OE.17.008051"
C_UM_PER_PS = 299.792458
ETA = 0.97
INTERFACE_SLOPE = 1.0 - ETA

DIELECTRIC_FREQUENCY_THZ = 619.5
DIELECTRIC_PERIOD_UM = 1.5
DIELECTRIC_RADIUS_UM = 0.5
DIELECTRIC_EPSILON = 4.0
DIELECTRIC_HEIGHT_UM = 0.05

METAL_PERIOD_UM = 0.7
METAL_RADIUS_UM = 0.15
METAL_HEIGHT_UM = 0.05
GOLD_PLASMA_RAD_S = 1.37e16
GOLD_DAMPING_RAD_S = 0.85e14

PROFILES = ("identity", "weiss2009")
PROFILE_LABELS = {
    "identity": "matched coordinates (no ASR)",
    "weiss2009": "matched coordinates + ASR",
}
REVISION = "weiss2009-v1-fixed-interface-eq41"


def torcwa_frequency(frequency_thz: float) -> float:
    """Convert THz (= ps^-1) to torcwa frequency in inverse micrometres."""
    return frequency_thz / C_UM_PER_PS


def gold_drude_epsilon(frequency_thz: float) -> complex:
    """Paper Drude model for the exp(-i*omega*t) convention."""
    omega = 2.0 * math.pi * frequency_thz * 1.0e12
    return 1.0 - GOLD_PLASMA_RAD_S**2 / (
        omega * (omega + 1j * GOLD_DAMPING_RAD_S)
    )


def parse_ints(text: str) -> list[int]:
    values: set[int] = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            parts = [int(value) for value in item.split(":")]
            if len(parts) not in (2, 3):
                raise ValueError(f"Invalid integer range: {item!r}")
            start, stop = parts[:2]
            step = parts[2] if len(parts) == 3 else 1
            if step <= 0 or stop < start:
                raise ValueError(f"Invalid integer range: {item!r}")
            values.update(range(start, stop + 1, step))
        else:
            values.add(int(item))
    result = sorted(values)
    if not result or result[0] < 1:
        raise ValueError("Fourier orders must be positive integers.")
    return result


def parse_floats(text: str) -> list[float]:
    values: list[float] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            parts = [float(value) for value in item.split(":")]
            if len(parts) != 3:
                raise ValueError(
                    "Float ranges must use start:stop:step notation."
                )
            start, stop, step = parts
            if step <= 0 or stop < start:
                raise ValueError(f"Invalid float range: {item!r}")
            count = int(math.floor((stop - start) / step + 1.0e-12))
            values.extend(start + index * step for index in range(count + 1))
        else:
            values.append(float(item))
    result = sorted(set(round(value, 12) for value in values))
    if not result or result[0] <= 0.0:
        raise ValueError("Frequencies must be positive.")
    return result


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def harmonics(order: int) -> int:
    return (2 * order + 1) ** 2


def analytical_dielectric_rod_beta() -> float:
    """Return the exact HE11 beta of the isolated dielectric cylinder.

    This solves the full-vector step-index cylinder characteristic equation,
    rather than using the rounded ``25 1/um`` value printed in the paper.
    """
    real_dtype = torch.float64
    n_core = math.sqrt(DIELECTRIC_EPSILON)
    n_clad = 1.0
    radius = DIELECTRIC_RADIUS_UM
    k0 = 2.0 * math.pi * torcwa_frequency(DIELECTRIC_FREQUENCY_THZ)
    v_number = radius * k0 * math.sqrt(n_core**2 - n_clad**2)

    def residual(u: torch.Tensor) -> torch.Tensor:
        w = torch.sqrt(torch.clamp(v_number**2 - u**2, min=1.0e-30))
        j0 = torch.special.bessel_j0(u)
        j1 = torch.special.bessel_j1(u)
        k0_value = torch.special.modified_bessel_k0(w)
        k1 = torch.special.modified_bessel_k1(w)
        j_ratio = (j0 - j1 / u) / (u * j1)
        k_ratio = (-k0_value - k1 / w) / (w * k1)
        neff_squared = n_core**2 - (u / (radius * k0)) ** 2
        return (
            (j_ratio + k_ratio)
            * (n_core**2 * j_ratio + n_clad**2 * k_ratio)
            - neff_squared * (1.0 / u**2 + 1.0 / w**2) ** 2
        )

    scan = torch.linspace(
        v_number * 1.0e-6,
        v_number * (1.0 - 1.0e-6),
        50000,
        dtype=real_dtype,
    )
    values = residual(scan)
    valid = torch.isfinite(values) & (torch.abs(values) < 1.0e10)
    crossings = torch.nonzero(
        valid[:-1] & valid[1:] & (values[:-1] * values[1:] < 0.0)
    ).flatten()
    roots: list[float] = []
    for index in crossings.tolist():
        lo = float(scan[index])
        hi = float(scan[index + 1])
        flo = float(values[index])
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            fmid = float(residual(torch.tensor(mid, dtype=real_dtype)))
            if flo * fmid <= 0.0:
                hi = mid
            else:
                lo, flo = mid, fmid
        root = 0.5 * (lo + hi)
        root_residual = abs(float(residual(torch.tensor(root, dtype=real_dtype))))
        if root_residual < 1.0e-5 and all(abs(root - old) > 1.0e-8 for old in roots):
            roots.append(root)
    if not roots:
        raise RuntimeError("No guided HE11 root was found.")
    betas = [
        math.sqrt((n_core * k0) ** 2 - (root / radius) ** 2)
        for root in roots
    ]
    guided = [beta for beta in betas if n_clad * k0 < beta < n_core * k0]
    if not guided:
        raise RuntimeError("The characteristic roots contained no guided mode.")
    return max(guided)


def _new_simulation(
    *,
    frequency_thz: float,
    order: int,
    period_um: float,
    profile: str,
    grid: int,
    dtype: torch.dtype,
    device: torch.device,
    cascade: str,
    smatrix_size: str,
) -> CustomRCWA_ASR_FR:
    simulation = CustomRCWA_ASR_FR(
        torcwa_frequency(frequency_thz),
        [order, order],
        [period_um, period_um],
        dtype=dtype,
        device=device,
        cascade=cascade,
        smatrix_size=smatrix_size,
        store_mode_couplings=False,
        verify_cascade=False,
        stable_eig_grad=False,
        compute_condition_numbers=False,
        matched_asr_G=INTERFACE_SLOPE,
        matched_asr_profile=profile,
        matched_asr_min_jacobian=1.0e-12,
        asr_quadrature_grid=max(grid, 4 * order + 4),
    )
    simulation.add_input_layer(eps=1.0, mu=1.0)
    simulation.add_output_layer(eps=1.0, mu=1.0)
    zero = torch.tensor(0.0, dtype=torch.float64, device=device)
    simulation.set_incident_angle(inc_ang=zero, azi_ang=zero)
    return simulation


def _add_circle(
    simulation: CustomRCWA_ASR_FR,
    *,
    height_um: float,
    radius_um: float,
    epsilon: complex,
    grid: int,
) -> None:
    simulation.add_layer_circle_asr(
        thickness=height_um,
        radius=radius_um,
        eps_bg=1.0,
        eps_cyl=epsilon,
        mu_bg=1.0,
        mu_cyl=1.0,
        nx=grid,
        ny=grid,
        factorization_rules=True,
        normal_vector_factorization=False,
    )


def _tensor_float(value: torch.Tensor) -> float:
    return float(torch.real(value).detach().cpu().item())


def power_for_polarization(
    simulation: CustomRCWA_ASR_FR, polarization: str
) -> dict[str, float]:
    """Return total power for Cartesian x or y input at normal incidence."""
    all_orders = torch.cartesian_prod(simulation.order_x, simulation.order_y)
    if polarization == "x":
        co, cross = "pp", "sp"
    elif polarization == "y":
        co, cross = "ss", "ps"
    else:
        raise ValueError("polarization must be 'x' or 'y'.")

    def port_power(port: str) -> torch.Tensor:
        a = simulation.S_parameters(
            all_orders,
            direction="forward",
            port=port,
            polarization=co,
            power_norm=True,
        )
        b = simulation.S_parameters(
            all_orders,
            direction="forward",
            port=port,
            polarization=cross,
            power_norm=True,
        )
        return torch.sum(torch.abs(a) ** 2 + torch.abs(b) ** 2)

    reflection = port_power("reflection")
    transmission = port_power("transmission")
    absorption = 1.0 - reflection - transmission
    return {
        "R": _tensor_float(reflection),
        "T": _tensor_float(transmission),
        "A": _tensor_float(absorption),
        "balance": _tensor_float(reflection + transmission),
    }


@torch.inference_mode()
def simulate_mode(
    *,
    order: int,
    profile: str,
    grid: int,
    beta_reference: float,
    dtype: torch.dtype,
    device: torch.device,
    cascade: str,
) -> dict[str, object]:
    started = time.perf_counter()
    simulation = _new_simulation(
        frequency_thz=DIELECTRIC_FREQUENCY_THZ,
        order=order,
        period_um=DIELECTRIC_PERIOD_UM,
        profile=profile,
        grid=grid,
        dtype=dtype,
        device=device,
        cascade=cascade,
        smatrix_size="half",
    )
    _add_circle(
        simulation,
        height_um=DIELECTRIC_HEIGHT_UM,
        radius_um=DIELECTRIC_RADIUS_UM,
        epsilon=DIELECTRIC_EPSILON,
        grid=grid,
    )
    kz = simulation.kz_norm[-1].detach()
    target_neff = beta_reference / (
        2.0 * math.pi * torcwa_frequency(DIELECTRIC_FREQUENCY_THZ)
    )
    finite = torch.isfinite(kz.real) & torch.isfinite(kz.imag)
    guided = finite & (kz.real > 1.0) & (kz.real < 2.05) & (torch.abs(kz.imag) < 1.0e-4)
    candidates = kz[guided]
    if candidates.numel() == 0:
        candidates = kz[finite]
    selected = candidates[torch.argmin(torch.abs(candidates - target_neff))]
    beta = float(selected.real.cpu()) * 2.0 * math.pi * torcwa_frequency(
        DIELECTRIC_FREQUENCY_THZ
    )
    row = {
        "profile": profile,
        "label": PROFILE_LABELS[profile],
        "order": order,
        "harmonics": harmonics(order),
        "grid": grid,
        "beta_um_inv": beta,
        "beta_reference_um_inv": beta_reference,
        "relative_error": abs(beta / beta_reference - 1.0),
        "selected_kz_imag": float(selected.imag.cpu()),
        "minimum_jacobian": float(
            torch.min(simulation.asr_mappings[-1].det_j).detach().cpu()
        ),
        "elapsed_s": time.perf_counter() - started,
        "revision": REVISION,
    }
    return row


@torch.inference_mode()
def simulate_scattering(
    *,
    frequency_thz: float,
    order: int,
    period_um: float,
    radius_um: float,
    height_um: float,
    epsilon: complex,
    profile: str,
    grid: int,
    dtype: torch.dtype,
    device: torch.device,
    cascade: str,
    both_polarizations: bool,
) -> dict[str, object]:
    started = time.perf_counter()
    simulation = _new_simulation(
        frequency_thz=frequency_thz,
        order=order,
        period_um=period_um,
        profile=profile,
        grid=grid,
        dtype=dtype,
        device=device,
        cascade=cascade,
        smatrix_size="half",
    )
    _add_circle(
        simulation,
        height_um=height_um,
        radius_um=radius_um,
        epsilon=epsilon,
        grid=grid,
    )
    simulation.solve_global_smatrix()
    x_power = power_for_polarization(simulation, "x")
    y_power = power_for_polarization(simulation, "y") if both_polarizations else None
    row: dict[str, object] = {
        "profile": profile,
        "label": PROFILE_LABELS[profile],
        "frequency_THz": frequency_thz,
        "order": order,
        "harmonics": harmonics(order),
        "grid": grid,
        "epsilon_real": complex(epsilon).real,
        "epsilon_imag": complex(epsilon).imag,
        "R_x": x_power["R"],
        "T_x": x_power["T"],
        "A_x": x_power["A"],
        "balance_x": x_power["balance"],
        "polarization_delta_T": "",
        "minimum_jacobian": float(
            torch.min(simulation.asr_mappings[-1].det_j).detach().cpu()
        ),
        "elapsed_s": time.perf_counter() - started,
        "revision": REVISION,
    }
    if y_power is not None:
        row.update(
            {
                "R_y": y_power["R"],
                "T_y": y_power["T"],
                "A_y": y_power["A"],
                "balance_y": y_power["balance"],
                "polarization_delta_T": abs(x_power["T"] - y_power["T"]),
            }
        )
    return row


def write_csv(rows: Iterable[dict[str, object]], path: Path) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(path: Path, study: str, args: argparse.Namespace, rows: int) -> None:
    payload = {
        "paper": "Matched coordinates and adaptive spatial resolution in the Fourier modal method",
        "paper_doi": PAPER_DOI,
        "study": study,
        "paper_equations": {
            "matched_coordinates": "Eqs. (37)-(38)",
            "adaptive_map": "Eqs. (41)-(42), fixed interface positions",
            "factorization": "symmetric Eqs. (29)-(36)",
        },
        "eta": ETA,
        "interface_slope_1_minus_eta": INTERFACE_SLOPE,
        "factorization_variants": {
            "implemented": "Weiss symmetry-preserving formulation",
            "not_claimed": "Li 2003 comparison markers in Figs. 2-5",
        },
        "orders": args.orders,
        "frequencies_THz": args.frequencies,
        "spectrum_order": args.spectrum_order,
        "grid": args.grid,
        "identity_grid": args.identity_grid,
        "device": args.device,
        "dtype": args.dtype,
        "cascade": args.cascade,
        "rows": rows,
        "revision": REVISION,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _plot_mapping(path: Path, *, grid: int, dtype: torch.dtype, device: torch.device) -> None:
    import matplotlib.pyplot as plt

    simulation = _new_simulation(
        frequency_thz=DIELECTRIC_FREQUENCY_THZ,
        order=1,
        period_um=DIELECTRIC_PERIOD_UM,
        profile="weiss2009",
        grid=grid,
        dtype=dtype,
        device=device,
        cascade="redheffer",
        smatrix_size="half",
    )
    mapping = simulation.build_circle_asr_mapping(
        grid, grid, DIELECTRIC_RADIUS_UM
    )
    x = mapping.x.detach().cpu().numpy()
    y = mapping.y.detach().cpu().numpy()
    stride = max(1, grid // 24)
    fig, axis = plt.subplots(figsize=(6.4, 6.4), constrained_layout=True)
    for index in range(0, grid, stride):
        axis.plot(x[index, :], y[index, :], color="black", linewidth=0.45)
        axis.plot(x[:, index], y[:, index], color="black", linewidth=0.45)
    circle = plt.Circle(
        (DIELECTRIC_PERIOD_UM / 2, DIELECTRIC_PERIOD_UM / 2),
        DIELECTRIC_RADIUS_UM,
        fill=False,
        color="tab:red",
        linewidth=1.2,
    )
    axis.add_patch(circle)
    axis.set(xlim=(0, DIELECTRIC_PERIOD_UM), ylim=(0, DIELECTRIC_PERIOD_UM))
    axis.set_aspect("equal")
    axis.set_xlabel("x1 (um)")
    axis.set_ylabel("x2 (um)")
    axis.set_title("Weiss 2009 Fig. 2(a): matched coordinates with ASR")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_mode(rows: list[dict[str, object]], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(6.6, 4.8), constrained_layout=True)
    for profile, marker in (("identity", "s"), ("weiss2009", "D")):
        selected = [row for row in rows if row["profile"] == profile]
        axis.semilogy(
            [row["harmonics"] for row in selected],
            [max(float(row["relative_error"]), 1.0e-16) for row in selected],
            marker=marker,
            label=PROFILE_LABELS[profile],
        )
    axis.set_xlabel("Truncation order (number of harmonics)")
    axis.set_ylabel("abs(beta / beta_calc - 1)")
    axis.set_title("Weiss 2009 Fig. 2(b): dielectric-cylinder mode")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_absorption(rows: list[dict[str, object]], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(6.6, 4.8), constrained_layout=True)
    for profile, marker in (("identity", "s"), ("weiss2009", "D")):
        selected = [row for row in rows if row["profile"] == profile]
        axis.semilogy(
            [row["harmonics"] for row in selected],
            [max(abs(float(row["A_x"])), 1.0e-16) for row in selected],
            marker=marker,
            label=PROFILE_LABELS[profile],
        )
    axis.set_xlabel("Truncation order (number of harmonics)")
    axis.set_ylabel("Absorption accuracy abs(1 - R - T)")
    axis.set_title("Weiss 2009 Fig. 3(a): dielectric-cylinder energy balance")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_metal_spectrum(rows: list[dict[str, object]], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
    axis.plot([row["frequency_THz"] for row in rows], [row["T_x"] for row in rows], label="T")
    axis.plot([row["frequency_THz"] for row in rows], [row["R_x"] for row in rows], label="R")
    axis.plot([row["frequency_THz"] for row in rows], [row["A_x"] for row in rows], label="A")
    axis.set_xlabel("Frequency (THz)")
    axis.set_ylabel("Power fraction")
    axis.set_ylim(bottom=-0.02)
    axis.set_title("Weiss 2009 Fig. 4(a): gold-cylinder spectrum")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_metal_convergence(rows: list[dict[str, object]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(6.6, 4.8), constrained_layout=True)
    for profile, marker in (("identity", "s"), ("weiss2009", "D")):
        selected = [row for row in rows if row["profile"] == profile]
        axis.plot(
            [row["harmonics"] for row in selected],
            [row["T_x"] for row in selected],
            marker=marker,
            label=PROFILE_LABELS[profile],
        )
    axis.set_xlabel("Truncation order (number of harmonics)")
    axis.set_ylabel("Transmission")
    axis.set_title("Weiss 2009 Fig. 4(b): 370 THz convergence")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.savefig(output_dir / "fig4b_metal_convergence.png", dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.6, 4.8), constrained_layout=True)
    for profile, marker in (("identity", "s"), ("weiss2009", "D")):
        selected = [row for row in rows if row["profile"] == profile]
        axis.semilogy(
            [row["harmonics"] for row in selected],
            [max(float(row["polarization_delta_T"]), 1.0e-16) for row in selected],
            marker=marker,
            label=PROFILE_LABELS[profile],
        )
    axis.set_xlabel("Truncation order (number of harmonics)")
    axis.set_ylabel("abs(Tx - Ty)")
    axis.set_title("Weiss 2009 Fig. 5(b): spurious polarization asymmetry")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    fig.savefig(output_dir / "fig5b_polarization_asymmetry.png", dpi=220)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    device = select_device(args.device)
    dtype = torch.complex128 if args.dtype == "complex128" else torch.complex64
    orders = parse_ints(args.orders)
    frequencies = parse_floats(args.frequencies)
    grid = args.grid
    identity_grid = args.identity_grid
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    requested = {args.study}
    if args.study == "paper":
        requested = {"mapping", "fig2", "fig3", "fig4-spectrum", "fig4-convergence"}
    elif args.study == "smoke":
        orders = [1, 2]
        frequencies = [350.0, 370.0]
        grid = max(32, min(grid, 48))
        identity_grid = grid
        requested = {"mapping", "fig2", "fig3", "fig4-spectrum", "fig4-convergence"}

    if grid < max(32, 4 * max(orders) + 4):
        raise ValueError("--grid must be at least max(32, 4*max(order)+4).")
    if identity_grid < max(32, 4 * max(orders) + 4):
        raise ValueError(
            "--identity-grid must be at least max(32, 4*max(order)+4)."
        )
    # Metadata must describe the effective sweep, including the values that
    # the smoke preset narrows from the paper defaults.
    args.orders = ",".join(str(value) for value in orders)
    args.frequencies = ",".join(f"{value:g}" for value in frequencies)
    args.grid = grid
    args.identity_grid = identity_grid

    if "mapping" in requested:
        _plot_mapping(output_dir / "fig2a_mapping.png", grid=max(grid, 64), dtype=dtype, device=device)
        print(f"wrote {output_dir / 'fig2a_mapping.png'}")

    if "fig2" in requested:
        beta_reference = analytical_dielectric_rod_beta()
        rows = [
            simulate_mode(
                order=order,
                profile=profile,
                grid=identity_grid if profile == "identity" else grid,
                beta_reference=beta_reference,
                dtype=dtype,
                device=device,
                cascade=args.cascade,
            )
            for profile in PROFILES
            for order in orders
        ]
        write_csv(rows, output_dir / "fig2_mode_convergence.csv")
        _plot_mode(rows, output_dir / "fig2b_mode_convergence.png")
        write_metadata(output_dir / "fig2_metadata.json", "fig2", args, len(rows))
        print(f"wrote Fig. 2 data ({len(rows)} rows)")

    if "fig3" in requested:
        rows = [
            simulate_scattering(
                frequency_thz=DIELECTRIC_FREQUENCY_THZ,
                order=order,
                period_um=DIELECTRIC_PERIOD_UM,
                radius_um=DIELECTRIC_RADIUS_UM,
                height_um=DIELECTRIC_HEIGHT_UM,
                epsilon=DIELECTRIC_EPSILON,
                profile=profile,
                grid=identity_grid if profile == "identity" else grid,
                dtype=dtype,
                device=device,
                cascade=args.cascade,
                both_polarizations=False,
            )
            for profile in PROFILES
            for order in orders
        ]
        write_csv(rows, output_dir / "fig3_energy_conservation.csv")
        _plot_absorption(rows, output_dir / "fig3a_energy_conservation.png")
        write_metadata(output_dir / "fig3_metadata.json", "fig3", args, len(rows))
        print(f"wrote Fig. 3 data ({len(rows)} rows)")

    if "fig4-spectrum" in requested:
        rows = [
            simulate_scattering(
                frequency_thz=frequency,
                order=args.spectrum_order,
                period_um=METAL_PERIOD_UM,
                radius_um=METAL_RADIUS_UM,
                height_um=METAL_HEIGHT_UM,
                epsilon=gold_drude_epsilon(frequency),
                profile="weiss2009",
                grid=max(grid, 4 * args.spectrum_order + 4),
                dtype=dtype,
                device=device,
                cascade=args.cascade,
                both_polarizations=False,
            )
            for frequency in frequencies
        ]
        write_csv(rows, output_dir / "fig4_metal_spectrum.csv")
        _plot_metal_spectrum(rows, output_dir / "fig4a_metal_spectrum.png")
        write_metadata(output_dir / "fig4a_metadata.json", "fig4-spectrum", args, len(rows))
        print(f"wrote Fig. 4(a) data ({len(rows)} rows)")

    if "fig4-convergence" in requested:
        epsilon = gold_drude_epsilon(370.0)
        rows = [
            simulate_scattering(
                frequency_thz=370.0,
                order=order,
                period_um=METAL_PERIOD_UM,
                radius_um=METAL_RADIUS_UM,
                height_um=METAL_HEIGHT_UM,
                epsilon=epsilon,
                profile=profile,
                grid=identity_grid if profile == "identity" else grid,
                dtype=dtype,
                device=device,
                cascade=args.cascade,
                both_polarizations=True,
            )
            for profile in PROFILES
            for order in orders
        ]
        write_csv(rows, output_dir / "fig4b_fig5b_metal_convergence.csv")
        _plot_metal_convergence(rows, output_dir)
        write_metadata(
            output_dir / "fig4b_fig5b_metadata.json",
            "fig4-convergence-and-fig5b",
            args,
            len(rows),
        )
        print(f"wrote Figs. 4(b)/5(b) data ({len(rows)} rows)")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--study",
        choices=("smoke", "mapping", "fig2", "fig3", "fig4-spectrum", "fig4-convergence", "paper"),
        default="smoke",
    )
    result.add_argument("--orders", default="6:15")
    result.add_argument("--frequencies", default="250:470:5")
    result.add_argument("--spectrum-order", type=int, default=12)
    result.add_argument("--grid", type=int, default=256)
    result.add_argument(
        "--identity-grid",
        type=int,
        default=1024,
        help=(
            "Sampling grid for the no-ASR matched-coordinate branch. Its "
            "discontinuous Jacobian needs denser quadrature than the ASR branch."
        ),
    )
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--dtype", choices=("complex64", "complex128"), default="complex128")
    result.add_argument("--cascade", choices=("redheffer", "algo2a"), default="redheffer")
    result.add_argument(
        "--output-dir",
        type=Path,
        default=_PACKAGE_ROOT / "results" / "paper",
    )
    return result


if __name__ == "__main__":
    run(parser().parse_args())
