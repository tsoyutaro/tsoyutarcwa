"""Passive-convention optical dispersion models for gold.

The RCWA implementation uses fields proportional to ``exp(-i omega t)`` and
therefore requires ``Im(epsilon) >= 0`` for a passive material.  The Rakić
paper prints the conjugate ``exp(+i omega t)`` convention; this module returns
its complex conjugate so that propagation decays in the solver.

References
----------
A. D. Rakić et al., Applied Optics 37, 5271-5283 (1998),
https://doi.org/10.1364/AO.37.005271

P. B. Johnson and R. W. Christy, Physical Review B 6, 4370-4379 (1972),
https://doi.org/10.1103/PhysRevB.6.4370
"""

from __future__ import annotations

import bisect
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

HC_EV_NM = 1239.8419843320026


def gold_epsilon_rakic_ld(wavelength_nm: float) -> complex:
    """Return bulk-Au relative permittivity from the Rakić LD fit.

    All oscillator energies are in eV.  The parameters are Table 1 and Table 2
    of Rakić et al.  The fit covers the requested 400-700 nm band, but the
    paper notes that its Brendel-Bormann fit is better around the lowest Au
    interband transition.  Measured n,k data should therefore be preferred for
    final quantitative comparison with a particular fabricated sample.
    """
    wavelength = float(wavelength_nm)
    if not 0.0 < wavelength < float("inf"):
        raise ValueError("wavelength_nm must be finite and positive.")
    energy = HC_EV_NM / wavelength
    plasma = 9.03
    f0, gamma0 = 0.760, 0.053
    oscillators = (
        (0.024, 0.415, 0.241),
        (0.010, 0.830, 0.345),
        (0.071, 2.969, 0.870),
        (0.601, 4.304, 2.494),
        (4.384, 13.32, 2.214),
    )
    # Passive exp(-i*omega*t) convention: Im(epsilon) is positive.
    epsilon = 1.0 - f0 * plasma**2 / (
        energy * (energy + 1.0j * gamma0)
    )
    for strength, resonance, damping in oscillators:
        epsilon += strength * plasma**2 / (
            resonance**2 - energy**2 - 1.0j * damping * energy
        )
    if epsilon.imag <= 0.0:
        raise RuntimeError("The passive gold model produced nonpositive loss.")
    return complex(epsilon)


@dataclass(frozen=True)
class TabulatedGold:
    """Linearly interpolate measured Au data without silent extrapolation."""

    wavelength_nm: tuple[float, ...]
    epsilon: tuple[complex, ...]
    source: str

    def __post_init__(self) -> None:
        if len(self.wavelength_nm) < 2 or len(self.wavelength_nm) != len(
            self.epsilon
        ):
            raise ValueError("Tabulated gold needs at least two matching samples.")
        if any(
            right <= left
            for left, right in zip(self.wavelength_nm, self.wavelength_nm[1:])
        ):
            raise ValueError("Gold wavelengths must be strictly increasing.")
        if any(value.imag < 0.0 for value in self.epsilon):
            raise ValueError(
                "CSV epsilon must use the passive convention Im(epsilon)>=0."
            )

    def __call__(self, wavelength_nm: float) -> complex:
        wavelength = float(wavelength_nm)
        if wavelength < self.wavelength_nm[0] or wavelength > self.wavelength_nm[-1]:
            raise ValueError(
                f"{wavelength:g} nm is outside tabulated range "
                f"[{self.wavelength_nm[0]:g}, {self.wavelength_nm[-1]:g}] nm."
            )
        right = bisect.bisect_left(self.wavelength_nm, wavelength)
        if right < len(self.wavelength_nm) and self.wavelength_nm[right] == wavelength:
            return self.epsilon[right]
        left = right - 1
        fraction = (
            (wavelength - self.wavelength_nm[left])
            / (self.wavelength_nm[right] - self.wavelength_nm[left])
        )
        return self.epsilon[left] + fraction * (
            self.epsilon[right] - self.epsilon[left]
        )


def _first_present(row: dict[str, str], names: Iterable[str]) -> str | None:
    normalized = {key.strip().lower(): value for key, value in row.items()}
    for name in names:
        if name in normalized and normalized[name].strip():
            return normalized[name]
    return None


def load_gold_csv(path: str | Path) -> TabulatedGold:
    """Load wavelength/n/k or wavelength/epsilon_real/epsilon_imag CSV.

    Recognized headers are ``wavelength_nm`` (also wavelength/lambda_nm),
    ``n`` and ``k``; alternatively ``epsilon_real`` and ``epsilon_imag``.
    Positive k is converted with ``epsilon=(n+i*k)^2``.
    """
    csv_path = Path(path)
    rows: list[tuple[float, complex]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Gold CSV has no header.")
        for line_number, row in enumerate(reader, start=2):
            wavelength_text = _first_present(
                row, ("wavelength_nm", "wavelength", "lambda_nm", "lambda")
            )
            if wavelength_text is None:
                raise ValueError("Gold CSV needs a wavelength_nm column.")
            wavelength = float(wavelength_text)
            n_text = _first_present(row, ("n", "refractive_index"))
            k_text = _first_present(row, ("k", "extinction_coefficient"))
            real_text = _first_present(
                row, ("epsilon_real", "eps_real", "epsilon1", "eps1")
            )
            imag_text = _first_present(
                row, ("epsilon_imag", "eps_imag", "epsilon2", "eps2")
            )
            if n_text is not None and k_text is not None:
                n_value, k_value = float(n_text), float(k_text)
                if k_value < 0.0:
                    raise ValueError(f"Negative k on CSV line {line_number}.")
                epsilon = complex(n_value, k_value) ** 2
            elif real_text is not None and imag_text is not None:
                epsilon = complex(float(real_text), float(imag_text))
            else:
                raise ValueError(
                    "Gold CSV needs either n,k or epsilon_real,epsilon_imag."
                )
            rows.append((wavelength, epsilon))
    rows.sort(key=lambda item: item[0])
    return TabulatedGold(
        tuple(item[0] for item in rows),
        tuple(item[1] for item in rows),
        str(csv_path.resolve()),
    )


def build_gold_model(model: str, csv_path: str | Path | None = None):
    normalized = model.strip().lower().replace("_", "-")
    if normalized in {"rakic", "rakic-ld", "ld"}:
        if csv_path is not None:
            raise ValueError("--gold-csv is only valid with --gold-model csv.")
        return gold_epsilon_rakic_ld
    if normalized in {"csv", "table", "tabulated"}:
        if csv_path is None:
            raise ValueError("--gold-model csv requires --gold-csv PATH.")
        return load_gold_csv(csv_path)
    raise ValueError("gold model must be 'rakic-ld' or 'csv'.")


if __name__ == "__main__":
    for wavelength in (400.0, 500.0, 550.0, 600.0, 700.0):
        value = gold_epsilon_rakic_ld(wavelength)
        print(f"{wavelength:6.1f} nm  epsilon={value.real: .8f}{value.imag:+.8f}j")
