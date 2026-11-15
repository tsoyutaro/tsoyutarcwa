"""Frozen, passive Cr dispersion with explicit provenance and no extrapolation."""
from __future__ import annotations

import cmath
from functools import lru_cache
from pathlib import Path

from studies.shared.convergence_io import content_hash
from studies.shared.tabulated_nk import load_tabulated_nk


@lru_cache(maxsize=16)
def _model(path, expected_sha256):
    source = Path(path)
    if content_hash(source) != expected_sha256:
        raise ValueError("Cr material CSV changed after experiment preparation.")
    return load_tabulated_nk(source)


def configure_material(config, config_path, override=None):
    source = Path(override if override is not None else config['material_csv'])
    if not source.is_absolute():
        # Overrides are relative to the execution directory; config paths to the config file.
        source = source.resolve() if override is not None else (config_path.parent/source).resolve()
    sha256 = content_hash(source)
    model = _model(str(source), sha256)
    waves = sorted(set(config['optimization_wavelengths_nm']+
                       config['verification_wavelengths_nm']+config['audit_wavelengths_nm']))
    for wavelength in waves:
        epsilon = model(wavelength)
        if epsilon.imag <= 0:
            raise ValueError("Semi-infinite Cr requires strictly absorbing passive optical constants.")
    bundled = Path(__file__).resolve().parent/'data/Johnson.csv'
    is_johnson = source.name == 'Johnson.csv' and sha256 == content_hash(bundled)
    config['material_csv'] = str(source)
    config['material_provenance'] = {
        'material': 'Cr', 'csv_name': source.name, 'csv_sha256_lf': sha256,
        'samples': len(model.wavelength_nm),
        'wavelength_range_nm': [model.wavelength_nm[0], model.wavelength_nm[-1]],
        'dataset': 'Johnson-Christy 1974 (existing user-supplied CSV)' if is_johnson else 'custom user-supplied Cr n/k CSV',
        'doi': '10.1103/PhysRevB.9.5056' if is_johnson else None,
        'source_url': 'https://refractiveindex.info/?shelf=main&book=Cr&page=Johnson' if is_johnson else None,
        'interpolation': 'linear in complex epsilon=(n+ik)^2, versus wavelength_nm',
        'extrapolation': 'forbidden', 'time_convention': 'exp(-i omega t); Im(epsilon)>0 passive',
        'substrate': 'semi-infinite Cr with the same dispersion as the pillars',
        'oxide_or_coating': 'none'
    }


def epsilon_at(config, wavelength_nm):
    return _model(config['material_csv'], config['material_provenance']['csv_sha256_lf'])(wavelength_nm)


def material_rows(config):
    waves = sorted(set(config['optimization_wavelengths_nm']+
                       config['verification_wavelengths_nm']+config['audit_wavelengths_nm']))
    rows = []
    for wavelength in waves:
        epsilon = epsilon_at(config, wavelength)
        nk = cmath.sqrt(epsilon)
        rows.append({'wavelength_nm': wavelength, 'epsilon_real': epsilon.real, 'epsilon_imag': epsilon.imag,
                     'effective_n': nk.real, 'effective_k': nk.imag})
    return rows
