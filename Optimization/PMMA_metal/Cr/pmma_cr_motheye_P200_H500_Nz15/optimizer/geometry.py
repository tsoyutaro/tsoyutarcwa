"""Independent monotone core/outer radius profiles with exact containment.

The two blocks of (Nz+1) logits define separate strictly increasing latent
profiles a and b through cumulative softmax gaps.  Physical radii are
    r_core(z) = min(a(z), b(z))
    r_outer(z) = max(a(z), b(z)) + min_shell_thickness_nm.
Both radii increase strictly with layer number; the core is contained in the
outer disk.  No constant coating thickness or monotone core/outer ratio is
imposed.  Every pair in the representable interior can be encoded by choosing
a=r_core and b=r_outer-min_shell_thickness_nm.

Numerical interior: each latent gap, including top/bottom boundary gaps, must
exceed softmax_gap_floor times the latent radial span.  This small numerical
floor is explicit; arbitrarily flat nondecreasing profiles are limiting cases.
min/max is piecewise differentiable, with torch's symmetric subgradient at ties.
Call the tensor map anew at each optimizer evaluation; it retains autograd history.
"""
from __future__ import annotations

import math


def _settings(config):
    n=config['layers']
    if isinstance(n,bool) or not isinstance(n,int) or n<2:
        raise ValueError('layers must be an integer >= 2.')
    low=float(config.get('radius_min_nm',.001))
    outer_max=float(config.get('radius_max_nm',99.999))
    shell=float(config.get('min_shell_thickness_nm',.001))
    floor=float(config.get('softmax_gap_floor',1e-6))
    high=outer_max-shell
    if (not all(math.isfinite(x) for x in (low,high,shell,floor)) or
            not 0<low<high or shell<=0 or not 0<(n+1)*floor<1):
        raise ValueError('Invalid radius bounds, shell minimum, or softmax gap floor.')
    if 'period_nm' in config:
        period=float(config['period_nm'])
        if not math.isfinite(period) or period<=0 or outer_max>=period/2:
            raise ValueError('Outer radius upper bound must be below half a finite positive period.')
    return n,low,high,shell,floor


def parameter_count(config):
    return 2*(_settings(config)[0]+1)


def _encode(values,low,high,floor):
    values=[float(v) for v in values]
    if any(not math.isfinite(x) for x in values):
        raise ValueError('Radius profiles must be finite.')
    edges=[low,*values,high]
    span=high-low
    fractions=[(b-a)/span for a,b in zip(edges,edges[1:])]
    if any(v<=floor for v in fractions):
        raise ValueError('Initial radius gaps must exceed softmax_gap_floor times the latent radial span.')
    scale=1-len(fractions)*floor
    raw=[(v-floor)/scale for v in fractions]
    total=sum(raw)
    logits=[math.log(v/total) for v in raw]
    mean=sum(logits)/len(logits)
    return [v-mean for v in logits]


def initial_logits_from_radii(core_radii,outer_radii,config):
    """Encode a feasible physical pair without a ratio or fixed-thickness rule."""
    n,low,high,shell,floor=_settings(config)
    core=[float(x) for x in core_radii]
    outer=[float(x) for x in outer_radii]
    if len(core)!=n or len(outer)!=n:
        raise ValueError('Both radius profiles must have exactly layers values.')
    if any(not math.isfinite(a+b) or b-a<shell-1e-12 for a,b in zip(core,outer)):
        raise ValueError('The outer radius must contain the core and minimum shell thickness.')
    latent_b=[x-shell for x in outer]
    return _encode(core,low,high,floor)+_encode(latent_b,low,high,floor)


def initial_logits(core_top,core_bottom,outer_top,outer_bottom,config):
    """Initialize linear layer-radius profiles, including the specified endpoints."""
    n,_,_,_,_=_settings(config)
    core=[core_top+(core_bottom-core_top)*j/(n-1) for j in range(n)]
    outer=[outer_top+(outer_bottom-outer_top)*j/(n-1) for j in range(n)]
    return initial_logits_from_radii(core,outer,config)


def baseline_logits(config):
    return initial_logits(config.get('baseline_core_tip_nm',5.),
                          config.get('baseline_core_base_nm',65.),
                          config.get('baseline_outer_tip_nm',35.),
                          config.get('baseline_outer_base_nm',95.),config)


def radii_tensor(logits,config):
    """Return [layers,2] tensor, columns PMMA core radius and metal outer radius."""
    import torch
    n,low,high,shell,floor=_settings(config)
    if logits.ndim!=1 or logits.numel()!=2*(n+1):
        raise ValueError('Expected a flat vector of 2*(layers+1) logits.')
    if not torch.is_floating_point(logits) or not bool(torch.isfinite(logits).all()):
        raise ValueError('Logits must be finite real floating-point values.')
    blocks=logits.reshape(2,n+1)
    gaps=floor+(1-(n+1)*floor)*torch.softmax(blocks,dim=1)
    latent=low+(high-low)*torch.cumsum(gaps,dim=1)[:,:-1]
    core=torch.minimum(latent[0],latent[1])
    outer=torch.maximum(latent[0],latent[1])+shell
    return torch.stack((core,outer),dim=1)


def radii_values(logits,config):
    """Pure Python equivalent, returning [[core_radius,outer_radius], ...]."""
    n,low,high,shell,floor=_settings(config)
    if hasattr(logits,'detach'):
        logits=logits.detach().cpu().tolist()
    logits=[float(x) for x in logits]
    if len(logits)!=2*(n+1) or any(not math.isfinite(x) for x in logits):
        raise ValueError('Expected 2*(layers+1) finite real logits.')
    latent=[]
    for block in (logits[:n+1],logits[n+1:]):
        peak=max(block)
        raw=[math.exp(x-peak) for x in block]
        total=sum(raw)
        cumulative=0.;radii=[]
        for value in raw[:-1]:
            cumulative+=floor+(1-(n+1)*floor)*value/total
            radii.append(low+(high-low)*cumulative)
        latent.append(radii)
    return [[min(a,b),max(a,b)+shell] for a,b in zip(*latent)]


def profile_rows(logits,config):
    pairs=radii_values(logits,config)
    dz=float(config['height_nm'])/config['layers']
    return [dict(layer=j+1,top_depth_nm=j*dz,bottom_depth_nm=(j+1)*dz,
                 pmma_radius_nm=core,metal_radius_nm=outer,
                 metal_shell_thickness_nm=outer-core,
                 pmma_diameter_nm=2*core,metal_outer_diameter_nm=2*outer,
                 nearest_neighbor_gap_nm=float(config['period_nm'])-2*outer)
            for j,(core,outer) in enumerate(pairs)]
