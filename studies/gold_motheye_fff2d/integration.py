"""Local CF4/RK4 steps composed as scattering matrices (complex128)."""
from __future__ import annotations
import math
import numpy as np
import torch

DM_INTEGRATORS = ("cf4-adaptive", "cf4", "rk4")


def cf4(function, start, dz, k0, backend):
    # Alvermann et al. (2012), doi:10.1088/1367-2630/14/10/105008.
    # The right exponential acts first; the constant-generator limit is exact.
    c = np.sqrt(3.)/6
    a, b = (3-2*np.sqrt(3.))/12, (3+2*np.sqrt(3.))/12
    early = 1j*k0*function(start+(.5-c)*dz)[0]
    late = 1j*k0*function(start+(.5+c)*dz)[0]
    return backend.exp(dz*(a*early+b*late)) @ backend.exp(dz*(b*early+a*late))


def diagnostics():
    return dict(accepted_steps=0, rejected_steps=0, min_accepted_step_nm=None,
                max_accepted_step_nm=0., max_accepted_error_ratio=0.)


def dm_section(function, start, dz, k0, max_step, backend, integrator,
               rtol, atol, min_step, stats):
    # Import here to avoid a cycle; both the DM and frozen RCWA paths share S.
    from .solver import identity_s, propagation_s, rk4, star
    total = identity_s(backend.eye(2*len(function.orders)))
    if dz <= 0:
        return total
    if integrator != 'rk4' and (function.homogeneous or function.geometry.slope == 0):
        # Analytical disk coefficients are constant for a cylinder; unlike
        # sampled 1D profiles there are no material-grid crossings to locate.
        parts = max(1,math.ceil(dz/max_step))
        h = dz/parts
        thin = propagation_s(backend.exp(1j*k0*h*function(start+dz/2)[0]))
        for _ in range(parts):
            total = star(total,thin)
        stats['accepted_steps'] += parts
        stats['min_accepted_step_nm'] = min(stats['min_accepted_step_nm'] or h,h)
        stats['max_accepted_step_nm'] = max(stats['max_accepted_step_nm'],h)
        return total
    end, z = start+dz, start
    h = min(max_step, dz)

    def local(z, h):
        p = (rk4(function,z,h,k0,h,backend.eye(4*len(function.orders)))
             if integrator == "rk4" else cf4(function,z,h,k0,backend))
        if not bool(torch.isfinite(p).all()):
            raise FloatingPointError("Non-finite local propagator")
        return propagation_s(p)

    while end-z > 8*np.finfo(float).eps*max(1.,abs(end)):
        h = min(h,end-z)
        error = 0.
        if integrator == "cf4-adaptive":
            try:
                coarse = local(z,h)
                fine = star(local(z,h/2),local(z+h/2,h/2))
                error = max(float((a-b).abs().max())/
                            (atol+rtol*max(float(a.abs().max()),float(b.abs().max())))
                            for a,b in zip(vars(coarse).values(),vars(fine).values()))/15
            except (RuntimeError, ArithmeticError, np.linalg.LinAlgError):
                error = math.inf
            if not math.isfinite(error) or error > 1:
                stats['rejected_steps'] += 1
                if h <= min_step*(1+1e-10):
                    raise FloatingPointError(f"DM error control reached minimum step at z={z:g}, h={h:g} nm")
                factor = max(.1,.8*error**(-.2)) if math.isfinite(error) else .25
                h = max(min_step,h*min(.5,factor))
                continue
            step_s = fine
            stats['max_accepted_error_ratio'] = max(stats['max_accepted_error_ratio'],error)
        else:
            step_s = local(z,h)
        total = star(total,step_s)
        stats['accepted_steps'] += 1
        stats['min_accepted_step_nm'] = min(stats['min_accepted_step_nm'] or h,h)
        stats['max_accepted_step_nm'] = max(stats['max_accepted_step_nm'],h)
        z = min(end,z+h)
        if integrator == "cf4-adaptive":
            h = min(max_step,h*min(2.,max(.5,.9*max(error,1e-16)**(-.2))))
    return total
