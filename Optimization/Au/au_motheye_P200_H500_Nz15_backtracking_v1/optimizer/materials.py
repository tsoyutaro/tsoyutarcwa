"""Lossless constant-index and tabulated dispersive PMMA materials.

CSV wl is in micrometres. n is interpolated linearly; k=0 is an assumption,
not a measured absorption result. Only the first segment may extend to 400 nm.
"""
from __future__ import annotations
import bisect
import csv
import hashlib
import math
from functools import lru_cache
from io_utils import content_hash
from .tabulated_nk import load_tabulated_nk
from pathlib import Path

def prepare_material(config, root):
    c = dict(config)
    model = c.get("material_model", "constant")
    if model == "constant":
        if not math.isfinite(c["index"]) or c["index"] <= 0:
            raise ValueError("Constant index must be finite and positive.")
        c["material_provenance"] = {"model": "constant", "n": c["index"], "k": 0.0}
        return c
    if model == "metal_csv":
        path=(root/c['material_csv']).resolve()
        checksum=content_hash(path)
        data=_metal_model(str(path),checksum)
        waves=sorted(set(c['optimization_wavelengths_nm']+c['verification_wavelengths_nm']+c['audit_wavelengths_nm']))
        for wave in waves:
            if data(wave).imag<=0: raise ValueError('Metal must have passive absorption.')
        c['_metal_csv_path']=str(path)
        c['material_provenance']=dict(model=model,material=c['material'],csv=c['material_csv'],
            csv_sha256_lf=checksum,interpolation='linear complex epsilon=(n+ik)^2 in wavelength_nm',
            extrapolation='forbidden',time_convention='exp(-i omega t)',
            tabulated_range_nm=[data.wavelength_nm[0],data.wavelength_nm[-1]],
            substrate='semi-infinite same metal',oxide_or_coating='none')
        return c
    if model != "pmma_csv":
        raise ValueError("Unknown material model.")
    path = root / c["pmma_csv"]
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["wl", "n"]:
            raise ValueError("PMMA CSV must have wl,n; wl unit is micrometres.")
        samples = [[1000*float(r["wl"]), float(r["n"])] for r in reader]
    if len(samples) < 2 or any(not all(math.isfinite(v) and v > 0 for v in row) for row in samples):
        raise ValueError("Invalid PMMA samples.")
    if any(b[0] <= a[0] for a,b in zip(samples, samples[1:])):
        raise ValueError("PMMA wavelengths must increase strictly.")
    if c.get("pmma_shortwave_policy") not in ("error", "linear-to-400"):
        raise ValueError("Unknown shortwave policy.")
    c["pmma_samples_nm"] = samples
    waves = sorted(set(c["optimization_wavelengths_nm"]+c["verification_wavelengths_nm"]+c["audit_wavelengths_nm"]))
    for wave in waves:
        material_index(c, wave)
    c["material_provenance"] = {
        "model": model, "csv": c["pmma_csv"],
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "k": 0.0, "interpolation": "linear n in wavelength_nm",
        "tabulated_range_nm": [samples[0][0], samples[-1][0]],
        "shortwave_policy": c["pmma_shortwave_policy"],
        "extrapolated_wavelengths_nm": [w for w in waves if w < samples[0][0]],
        "n_400nm": material_index(c,400) if c["pmma_shortwave_policy"] == "linear-to-400" else None}
    return c

def material_index(config, wavelength):
    if config.get("material_model", "constant") == "constant":
        return float(config["index"])
    samples = config["pmma_samples_nm"]
    w = float(wavelength)
    if not math.isfinite(w) or w > samples[-1][0]:
        raise ValueError("Wavelength outside PMMA data range.")
    if w < samples[0][0]:
        if config["pmma_shortwave_policy"] != "linear-to-400" or w < 400 or samples[0][0] > 405:
            raise ValueError("PMMA shortwave extrapolation is disallowed.")
        a,b = samples[:2]
    else:
        j = bisect.bisect_left([p[0] for p in samples],w)
        if samples[j][0] == w:
            return samples[j][1]
        a,b = samples[j-1:j+1]
    return a[1]+(w-a[0])*(b[1]-a[1])/(b[0]-a[0])

def wavelength_config(config, wavelength):
    if config.get("material_model")=="metal_csv":
        import cmath
        return {**config,"index":cmath.sqrt(epsilon_at(config,wavelength))}
    return {**config, "index": material_index(config,wavelength),
            "_pmma_extrapolated": config.get("material_model") == "pmma_csv" and wavelength < config["pmma_samples_nm"][0][0]}


@lru_cache(maxsize=16)
def _metal_model(path,checksum):
    p=Path(path)
    if content_hash(p)!=checksum: raise ValueError('Material CSV changed.')
    return load_tabulated_nk(p)


def epsilon_at(config,wavelength):
    if config.get('material_model')=='metal_csv':
        return _metal_model(config['_metal_csv_path'],config['material_provenance']['csv_sha256_lf'])(wavelength)
    return material_index(config,wavelength)**2
