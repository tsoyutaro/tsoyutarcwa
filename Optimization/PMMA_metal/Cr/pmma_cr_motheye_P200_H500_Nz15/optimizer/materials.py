"""Dispersive metal shell and lossless tabulated PMMA core/substrate."""
from __future__ import annotations
import bisect
import csv
import math
from functools import lru_cache
from pathlib import Path
from io_utils import content_hash
from .tabulated_nk import load_tabulated_nk


@lru_cache(maxsize=16)
def _metal_model(path,checksum):
    p=Path(path)
    if content_hash(p)!=checksum:raise ValueError('Metal material CSV changed.')
    return load_tabulated_nk(p)


def prepare_material(config,root):
    c=dict(config)
    if c.get('material_model')!='pmma_metal_csv':raise ValueError('Expected PMMA + metal CSV model.')
    metal=(root/c['material_csv']).resolve()
    pmma=(root/c['pmma_csv']).resolve()
    metal_sha=content_hash(metal)
    pmma_sha=content_hash(pmma)
    table=_metal_model(str(metal),metal_sha)
    with pmma.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=['wl','n']:raise ValueError('PMMA CSV must use wl,n with wl in micrometres.')
        samples=[[1000*float(row['wl']),float(row['n'])] for row in reader]
    if len(samples)<2 or any(not all(math.isfinite(x) and x>0 for x in row) for row in samples):
        raise ValueError('Invalid PMMA samples.')
    if any(b[0]<=a[0] for a,b in zip(samples,samples[1:])):raise ValueError('PMMA wavelengths must strictly increase.')
    if c.get('pmma_shortwave_policy') not in ('error','linear-to-400'):raise ValueError('Unknown PMMA shortwave policy.')
    c.update(_metal_csv_path=str(metal),_metal_sha256_lf=metal_sha,pmma_samples_nm=samples)
    waves=sorted(set(c['optimization_wavelengths_nm']+c['verification_wavelengths_nm']+c['audit_wavelengths_nm']))
    for w in waves:
        if metal_epsilon_at(c,w).imag<=0:raise ValueError('Metal must have passive absorption.')
        pmma_index(c,w)
    c['material_provenance']=dict(model='pmma_metal_csv',metal=dict(material=c['material'],csv=c['material_csv'],
        sha256_lf=metal_sha,tabulated_range_nm=[table.wavelength_nm[0],table.wavelength_nm[-1]],
        interpolation='linear complex epsilon=(n+ik)^2 in wavelength_nm',extrapolation='forbidden'),
        pmma=dict(csv=c['pmma_csv'],sha256_lf=pmma_sha,tabulated_range_nm=[samples[0][0],samples[-1][0]],
        interpolation='linear n in wavelength_nm',k=0.,k_policy='lossless assumption; not measured k',
        shortwave_policy=c['pmma_shortwave_policy'],extrapolated_wavelengths_nm=[w for w in waves if w<samples[0][0]]),
        structure=f"{c['layers']} variable PMMA-core metal-shell slices; air background; lossless semi-infinite PMMA substrate",
        fixed_top_cap=False,fixed_valley_metal_film=False,time_convention='exp(-i omega t)')
    return c


def metal_epsilon_at(config,wavelength):
    return _metal_model(config['_metal_csv_path'],config['_metal_sha256_lf'])(wavelength)


def pmma_index(config,wavelength):
    samples=config['pmma_samples_nm'];w=float(wavelength)
    if not math.isfinite(w) or w>samples[-1][0]:raise ValueError('Wavelength outside PMMA table.')
    if w<samples[0][0]:
        if config['pmma_shortwave_policy']!='linear-to-400' or w<400 or samples[0][0]>405:
            raise ValueError('PMMA shortwave extrapolation is disallowed.')
        a,b=samples[:2]
    else:
        j=bisect.bisect_left([p[0] for p in samples],w)
        if samples[j][0]==w:return samples[j][1]
        a,b=samples[j-1:j+1]
    return a[1]+(w-a[0])*(b[1]-a[1])/(b[0]-a[0])


def pmma_epsilon_at(config,wavelength):
    return complex(pmma_index(config,wavelength)**2,0.)


def pmma_extrapolated(config,wavelength):
    return float(wavelength)<config['pmma_samples_nm'][0][0]


# Explicit compatibility aliases; the two-material physics uses the named functions above.
epsilon_at=metal_epsilon_at
material_index=pmma_index
