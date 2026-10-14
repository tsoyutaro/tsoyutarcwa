"""Differentiable NV training and constant-layer-memory NV verification.

Both paths use the same reduced Maxwell eigensolves and Redheffer recurrence.
Verification builds layers bottom-up and retains only ports, the current layer,
and cumulative transmission/reflection. Neither path expands the full S matrix.
"""
from __future__ import annotations

import gc
import math
import time

from .geometry import radii_tensor

LEGACY = ("P", "Q", "eps_conv", "mu_conv", "E_eigvec", "H_eigvec")
AUX = ("asr_mappings", "asr_material_tensors", "asr_T_matrices", "asr_Tz_matrices",
       "asr_condition_numbers", "E_eigvec_uv", "H_eigvec_uv", "_asr_slot_by_layer",
       "_asr_field_context_by_layer", "_physical_material_by_layer")


class NonphysicalResult(RuntimeError):
    pass


def release_unused(sim):
    if sim.field_regions != "none" or sim.store_mode_couplings or len(sim._polarized_layers) != sim.layer_N:
        raise RuntimeError("Release requires fields-disabled, invariant source-sector NV layers.")
    for name in LEGACY:
        getattr(sim, name)[-1] = None
    for name in AUX:
        getattr(sim, name).clear()


def make_sim(config, lattice, wavelength, order, grid, device, release=True):
    import torch
    from rcwa_solver_auto import AutoRCWA, GroupTheoryOptions, Lattice, NVMOptions, OutputSpec
    class FastNV(AutoRCWA):
        def add_structured_layer(self, spec):
            selected = super().add_structured_layer(spec)
            if selected != "nvm":
                raise RuntimeError("An NV-only layer selected another formulation.")
            if release:
                release_unused(self)
            return selected
    sim = FastNV(freq=config["period_nm"]/wavelength, order=[order]*2,
                 lattice=Lattice.square(1.) if lattice == "square" else Lattice.triangular(1.),
                 nvm=NVMOptions(grid=(grid, grid), use_lanczos=False),
                 outputs=OutputSpec(smatrix_size="half", fields="none"), cascade="redheffer",
                 group_theory=GroupTheoryOptions(enabled=True, symmetry="c2v" if lattice == "square" else "d6",
                                                 polarization="x", strict=True),
                 dtype=torch.complex128, device=device, verify_cascade=False)
    sim.add_input_layer(eps=1.)
    sim.add_output_layer(eps=config["index"]**2)
    sim.set_incident_angle(0., 0.)
    return sim


def layer_spec(radius, config, depth=1.):
    from rcwa_solver_auto import Circle, LayerSpec, Material
    return LayerSpec(thickness=depth, geometry=Circle(radius/config["period_nm"]),
                     background=Material(1.), inclusion=Material(config["index"]**2), method="nvm")


def powers(sim, transmitted, reflected, config):
    import torch
    from studies.pmma_gold_motheye.common import _zero_order_x_source
    incident = _zero_order_x_source(sim)
    def flux(electric, medium, direction):
        magnetic = direction*(medium @ electric)
        n = electric.numel()//2
        return .5*torch.real((electric[:n]*magnetic[n:].conj()-electric[n:]*magnetic[:n].conj()).sum())
    incoming = flux(incident, sim.Vi, 1)
    r = -flux(reflected, sim.Vi, -1)/incoming
    t = flux(transmitted, sim.Vo, 1)/incoming
    rv, tv = float(r.detach().cpu()), float(t.detach().cpu())
    residual = rv+tv-1
    if (not all(math.isfinite(v) for v in (rv, tv)) or min(rv, tv) < -1e-7 or max(rv, tv) > 1+1e-7
            or abs(residual) > config["energy_tolerance"]):
        raise NonphysicalResult(f"Invalid lossless powers: R={rv}, T={tv}, R+T-1={residual}.")
    return r, {"reflectance": rv, "transmittance": tv, "energy_residual": residual}


def solve_train(radii, config, lattice, wavelength, order, grid, device, *, checkpoint_layers=None, release=True):
    import torch
    from studies.pmma_gold_motheye.common import _zero_order_x_source
    sim = make_sim(config, lattice, wavelength, order, grid, device, release=release)
    depth = config["height_nm"]/(config["layers"]*config["period_nm"])
    checkpointed = config["checkpoint_layers"] if checkpoint_layers is None else checkpoint_layers
    checkpointed = checkpointed and radii.requires_grad
    def one_layer(radius):
        local = make_sim(config, lattice, wavelength, order, grid, device)
        local.add_structured_layer(layer_spec(radius, config))
        modes = local._polarized_layers[0]
        return modes["electric"], modes["magnetic"], modes["kz"], *local._polarization_bases
    for radius in radii:
        if checkpointed:
            from torch.utils.checkpoint import checkpoint
            electric, magnetic, kz, eb, hb = checkpoint(one_layer, radius, use_reentrant=False, preserve_rng_state=False)
            sim._install_polarization_basis(eb.detach(), hb.detach(), 1e-8)
            sim._polarized_layers.append({"electric": electric, "magnetic": magnetic, "kz": kz})
            sim.thickness.append(depth)
            sim.layer_N += 1
        else:
            sim.add_structured_layer(layer_spec(radius, config, depth))
    # Return only the two response vectors; avoid lifting four NxN S blocks.
    transmitted, reflected = sim._solve_polarization_reduced_smatrix(incident_source=_zero_order_x_source(sim))
    value, row = powers(sim, transmitted, reflected, config)
    row.update(retained_scalar_harmonics=3*order*(order+1)+1 if lattice == "hexagonal" else (2*order+1)**2,
               eigen_dimension=sim._polarization_bases[0].shape[1], storage="reduced-modes-response-vectors",
               checkpoint_layers=checkpointed)
    return value, row


def solve_verify(radii, config, lattice, wavelength, order, grid, device):
    import torch
    from studies.pmma_gold_motheye.common import _zero_order_x_source
    # NV requires no spatial autograd. This entire path has no backward graph.
    with torch.no_grad():
        port = None
        transmission = reflection = identity = eb = hb = reference_v = input_interface = None
        depth = config["height_nm"]/(config["layers"]*config["period_nm"])
        for radius in reversed(radii.unbind()):
            local = make_sim(config, lattice, wavelength, order, grid, device)
            local.add_structured_layer(layer_spec(radius, config, depth))
            modes = local._polarized_layers.pop()
            electric_basis, magnetic_basis = local._polarization_bases
            if port is None:
                port, eb, hb = local, electric_basis, magnetic_basis
                reference_v = hb.mH @ (port.Vf @ eb)
                input_v = hb.mH @ (port.Vi @ eb)
                output_v = hb.mH @ (port.Vo @ eb)
                input_interface = port._reduced_interface_s(reference_v, input_v, input_side=True)
                output_interface = port._reduced_interface_s(reference_v, output_v, input_side=False)
                transmission, reflection = output_interface[:2]
                identity = port._eye(eb.shape[1])
            elif not (torch.allclose(eb, electric_basis, atol=1e-12, rtol=1e-12) and
                      torch.allclose(hb, magnetic_basis, atol=1e-12, rtol=1e-12)):
                raise RuntimeError("Streaming layers do not share the same invariant basis.")
            layer_s = port._reduced_layer_smatrix(modes["electric"], modes["magnetic"], modes["kz"], depth, reference_v)
            tf, rf, rb, tb = layer_s
            new_r = rf + tb @ port._solve(identity-reflection@rb, reflection@tf)
            new_t = transmission @ port._solve(identity-rb@reflection, tf)
            transmission, reflection = new_t, new_r
            del modes, layer_s, tf, rf, rb, tb, local, electric_basis, magnetic_basis
        tf, rf, rb, tb = input_interface
        new_r = rf + tb @ port._solve(identity-reflection@rb, reflection@tf)
        new_t = transmission @ port._solve(identity-rb@reflection, tf)
        source = eb.mH @ _zero_order_x_source(port)
        _, row = powers(port, eb@(new_t@source), eb@(new_r@source), config)
        row.update(retained_scalar_harmonics=3*order*(order+1)+1 if lattice == "hexagonal" else (2*order+1)**2,
                   eigen_dimension=eb.shape[1], storage="one-layer-streamed-Redheffer-response-vectors",
                   maximum_retained_layer_modes=1, checkpoint_layers=False)
        return row


def weights(wavelengths, kind):
    if kind == "arithmetic" or len(wavelengths) == 1:
        return [1./len(wavelengths)]*len(wavelengths)
    from studies.dielectric_motheye_n25.model import trapezoid_weights
    return trapezoid_weights(wavelengths)


def objective(logits, config, lattice, wavelengths, order, grid, device, *, gradient):
    import torch
    started = time.perf_counter()
    total, derivative, rows = 0., torch.zeros_like(logits), []
    # The gradient of the weighted sum is accumulated wavelength by wavelength.
    for wavelength, weight in zip(wavelengths, weights(wavelengths, config["objective_weighting"])):
        with torch.set_grad_enabled(gradient):
            radii = radii_tensor(logits, config)
            value, row = solve_train(radii, config, lattice, wavelength, order, grid, device)
            if gradient:
                derivative += torch.autograd.grad(weight*value, logits)[0].detach()
            total += weight*row["reflectance"]
        rows.append({"wavelength_nm": wavelength, **row})
        del value, radii
    if not math.isfinite(total) or not bool(torch.isfinite(derivative).all()):
        raise NonphysicalResult("Nonfinite objective or gradient.")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {"objective": total, "gradient": derivative if gradient else None,
            "rows": rows, "seconds": time.perf_counter()-started}
