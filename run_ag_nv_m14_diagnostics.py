"""Fixed-order Ag NV grid, layer-power and source-excited-mode diagnostics.

Keeps the existing complex128 NV/D6/Redheffer equations. Full-space per-layer
operators are released; source-specific internal fields are reconstructed from
reduced modes using decaying waves. Nonphysical powers are diagnostic output,
not an exception. Each Q runs in a fresh process, independently of other Qs.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from studies.transition_metals_nv import runtime as nv
from studies.shared.tabulated_nk import load_tabulated_nk


def write_csv(path, rows):
    if not rows:
        return
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    nv.atomic_text(path, stream.getvalue())


def antihermitian_part(matrix):
    return (matrix-matrix.mH)/(2j)


def relative_residual(torch, actual, expected):
    denominator = max(float(torch.linalg.vector_norm(actual)),
                      float(torch.linalg.vector_norm(expected)), 1e-30)
    return float(torch.linalg.vector_norm(actual-expected))/denominator


def reduced_flux(torch, electric, magnetic, flux_pairing):
    return float(.5*torch.real(torch.vdot(electric, flux_pairing@magnetic)))


def modal_state(torch, layer, forward, backward, omega, thickness, z):
    ap = torch.exp(1j*omega*layer["kz"]*z)*forward
    am = torch.exp(1j*omega*layer["kz"]*(thickness-z))*backward
    return layer["electric"]@(ap+am), layer["magnetic"]@(ap-am)


def integrate_loss(torch, layer, cp, cm, omega, thickness, le, lh, pin,
                   target, initial_order):
    """Independent volume integral of the actual truncated Maxwell operators.

    For reduced fields dE/dz=i*omega*P*H, dH/dz=i*omega*Q*E and
    flux=.5*Re(E^H G H): -d(flux)/dz = omega/2 *
    (E^H Im_H(GQ) E + H^H Im_H(G^H P) H).
    Adaptive Gauss integration checks the boundary-flux difference; it does
    not enforce it or change the fields. Negative-eigenvalue contributions
    are kept with their signs, including interference within each sector.
    """
    import numpy as np
    we, ve = torch.linalg.eigh(le)
    wh, vh = torch.linalg.eigh(lh)
    for count in sorted(set((initial_order, max(32, initial_order),
                             max(64, initial_order), max(128, initial_order)))):
        nodes, weights = np.polynomial.legendre.leggauss(count)
        z = torch.as_tensor((nodes+1)*thickness/2, dtype=torch.float64, device=cp.device)
        weights_t = torch.as_tensor(weights*thickness/2, dtype=torch.float64, device=cp.device)
        ap = torch.exp(1j*omega*layer["kz"][:, None]*z[None, :])*cp[:, None]
        am = torch.exp(1j*omega*layer["kz"][:, None]*(thickness-z[None, :]))*cm[:, None]
        electric = layer["electric"]@(ap+am)
        magnetic = layer["magnetic"]@(ap-am)
        e_projection, h_projection = ve.mH@electric, vh.mH@magnetic
        scale = omega/(2*pin)
        e_terms = we[:, None]*e_projection.abs().square()
        h_terms = wh[:, None]*h_projection.abs().square()
        total = float(scale*torch.sum(weights_t*(e_terms.sum(0)+h_terms.sum(0))))
        negative_e = float(scale*torch.sum(weights_t*e_terms[we < 0].sum(0)))
        negative_h = float(scale*torch.sum(weights_t*h_terms[wh < 0].sum(0)))
        error = abs(total-target)
        if error <= 1e-8 + 1e-7*abs(target):
            break
    return dict(volume_absorptance=total, negative_electric_sector=negative_e,
                negative_magnetic_sector=negative_h, volume_flux_error=error,
                quadrature_points=count, electric_loss_min_eigenvalue=float(we.min()),
                magnetic_loss_min_eigenvalue=float(wh.min()))


def reconstruct(torch, sim, operators, source, output, initial_quadrature):
    """Recover only the requested incident source; no inverse decay phases."""
    be, bh = sim._polarization_bases
    n = sim.order_N
    pairing = be.mH @ torch.cat((bh[n:], -bh[:n]), 0)
    ref, vin, vout = (bh.mH@v@be for v in (sim.Vf, sim.Vi, sim.Vo))
    sim._polarization_reference_v = ref
    inp = sim._reduced_interface_s(ref, vin, input_side=True)
    out = sim._reduced_interface_s(ref, vout, input_side=False)
    layers = sim._polarized_layers
    size = layers[0]["electric"].shape[0]
    eye = sim._eye(size)
    scattering = [sim._reduced_layer_smatrix(layer["electric"], layer["magnetic"],
                    layer["kz"], thickness, ref)
                  for layer, thickness in zip(layers, sim.thickness)]
    rho = [None]*(len(layers)+1)
    propagate = [None]*len(layers)
    rho[-1] = out[1]
    for i in range(len(layers)-1, -1, -1):
        tf, rf, rb, tb = scattering[i]
        propagate[i] = sim._solve(eye-rb@rho[i+1], tf)
        rho[i] = rf+tb@sim._solve(eye-rho[i+1]@rb, rho[i+1]@tf)
    src = be.mH@source
    u = sim._solve(eye-inp[2]@rho[0], inp[0]@src)
    reflected = inp[1]@src+inp[3]@rho[0]@u
    pin = reduced_flux(torch, src, vin@src, pairing)
    if not pin > 0:
        raise ValueError("Incident power must be positive.")
    omega = float(sim.omega.real)
    radii = sim._diagnostic_radii_nm
    period = sim._diagnostic_period_nm
    flux_rows, mode_rows, boundaries, map_rows = [], [], [], []
    residuals = []
    previous = (src+reflected, vin@(src-reflected))
    depth = 0.
    harmonic_order = torch.maximum(torch.maximum(sim.order_x[:, None].abs(),
        sim.order_y[None, :].abs()),
        (sim.order_x[:, None]-sim.order_y[None, :]).abs()).reshape(-1)
    edge = torch.cat((harmonic_order >= .8*sim._diagnostic_order,)*2)
    for i, (layer, d0) in enumerate(zip(layers, sim.thickness)):
        thickness = float(d0.real)
        bottom_u = propagate[i]@u
        bottom_v = rho[i+1]@bottom_u
        e, h = layer["electric"], layer["magnetic"]
        phase = torch.diag(torch.exp(1j*omega*layer["kz"]*thickness))
        vh = sim._solve(ref, h)
        plus, minus = e+vh, e-vh
        boundary = torch.cat((torch.cat((plus, minus@phase), 1),
                              torch.cat((minus@phase, plus), 1)), 0)
        rhs = 2*torch.cat((u, bottom_v))[:, None]
        coupling = sim._equilibrated_solve(boundary, rhs).squeeze(-1)
        cp, cm = coupling[:size], coupling[size:]
        top_e, top_h = modal_state(torch, layer, cp, cm, omega, thickness, 0.)
        bot_e, bot_h = modal_state(torch, layer, cp, cm, omega, thickness, thickness)
        mid_e, mid_h = modal_state(torch, layer, cp, cm, omega, thickness, thickness/2)
        residuals.extend((relative_residual(torch, top_e, previous[0]),
                          relative_residual(torch, top_h, previous[1])))
        boundary_error = relative_residual(torch, boundary@coupling, rhs.squeeze(-1))
        residuals.append(boundary_error)
        top_flux = reduced_flux(torch, top_e, top_h, pairing)/pin
        bottom_flux = reduced_flux(torch, bot_e, bot_h, pairing)/pin
        absorption = top_flux-bottom_flux
        le, lh = (m.to(device=sim._device) for m in operators[i])
        loss = integrate_loss(torch, layer, cp, cm, omega, thickness, le, lh,
                              pin, absorption, initial_quadrature)
        mid_loss = omega/(2*pin)*float(torch.real(
            torch.vdot(mid_e, le@mid_e)+torch.vdot(mid_h, lh@mid_h)))
        flux_rows.append(dict(layer=i+1, depth_top_nm=depth,
            depth_mid_nm=depth+period*thickness/2, depth_bottom_nm=depth+period*thickness,
            radius_nm=radii[i], flux_top=top_flux, flux_bottom=bottom_flux,
            layer_absorptance=absorption, layer_absorptance_pp=100*absorption,
            cumulative_absorptance=1-float((reflected.abs().square()).sum())-bottom_flux,
            mid_loss_rate_per_normalized_z=mid_loss, boundary_system_residual=boundary_error,
            **loss))
        # Amplitudes alone depend on eigenvector normalization. Multiplication
        # by the modal E norm removes that scale; modal interference still
        # prevents these proxies being interpreted as additive powers.
        half_phase = torch.exp(1j*omega*layer["kz"]*thickness/2)
        a, b = half_phase*cp, half_phase*cm
        e_norm = e.abs().square().sum(0)
        excitation = (a+b).abs().square()*e_norm
        magnetic_proxy = (a-b).abs().square()*h.abs().square().sum(0)
        full_e_modes = be@e
        edge_weight = full_e_modes[edge].abs().square().sum(0)/e_norm.clamp_min(1e-300)
        total_self_flux = .5*torch.real(torch.sum(e.conj()*(pairing@h), 0))
        ranks = torch.argsort(torch.argsort(excitation, descending=True))+1
        for j in range(size):
            kz = complex(layer["kz"][j])
            amplitude_plus, amplitude_minus = complex(cp[j]), complex(cm[j])
            mode_rows.append(dict(layer=i+1, depth_mid_nm=depth+period*thickness/2,
                mode_index=j+1, kz_real=kz.real, kz_imag=kz.imag,
                forward_top_real=amplitude_plus.real, forward_top_imag=amplitude_plus.imag,
                backward_bottom_real=amplitude_minus.real, backward_bottom_imag=amplitude_minus.imag,
                electric_mid_norm2_proxy=float(excitation[j]),
                magnetic_mid_norm2_proxy=float(magnetic_proxy[j]),
                electric_excitation_rank=int(ranks[j]), edge_harmonic_fraction=float(edge_weight[j]),
                unit_amplitude_self_flux=float(total_self_flux[j]),
                forward_mid_self_flux=float(total_self_flux[j]*a[j].abs().square())/pin,
                backward_mid_self_flux=-float(total_self_flux[j]*b[j].abs().square())/pin))
        map_rows.append(excitation.detach().cpu().numpy())
        boundaries.append((depth, top_flux))
        previous = bot_e, bot_h
        u = bottom_u
        depth += period*thickness
        if (i+1) % 25 == 0 or i+1 == len(layers):
            print(f"  fields: layer {i+1}/{len(layers)}; A_layer={100*absorption:.6g} pp", flush=True)
    transmitted = out[0]@u
    residuals.extend((relative_residual(torch, previous[0], transmitted),
                      relative_residual(torch, previous[1], vout@transmitted)))
    r = -reduced_flux(torch, reflected, -vin@reflected, pairing)/pin
    into_substrate = reduced_flux(torch, transmitted, vout@transmitted, pairing)/pin
    for row in flux_rows:
        row["cumulative_absorptance"] = 1-r-row["flux_bottom"]
    reference_r = be.mH@(sim.S[1]@source)
    reference_t = be.mH@(sim.S[0]@source)
    parity = max(relative_residual(torch, reflected, reference_r),
                 relative_residual(torch, transmitted, reference_t))
    write_csv(output/"layer_power.csv", flux_rows)
    write_csv(output/"excited_modes.csv", mode_rows)
    import numpy as np
    np.savez_compressed(output/"mode_excitation_map.npz", electric_norm2=np.stack(map_rows))
    return dict(reflectance=r, absorptance=1-r, transmittance_far=0.,
        substrate_absorptance=into_substrate, motheye_absorptance=sum(row["layer_absorptance"] for row in flux_rows),
        negative_absorption_layers=sum(row["layer_absorptance"] < -1e-8 for row in flux_rows),
        minimum_layer_absorptance=min(row["layer_absorptance"] for row in flux_rows),
        total_negative_loss_sector=sum(row["negative_electric_sector"]+row["negative_magnetic_sector"] for row in flux_rows),
        max_boundary_residual=max(residuals), original_scattering_amplitude_parity_residual=parity,
        layer_sum_power_balance_error=abs(sum(row["layer_absorptance"] for row in flux_rows)+into_substrate-(1-r)),
        max_volume_flux_error=max(row["volume_flux_error"] for row in flux_rows),
        summed_volume_flux_error=abs(sum(row["volume_absorptance"]-row["layer_absorptance"] for row in flux_rows)),
        incident_power=pin, reduced_dimension=size, layer_count=len(layers))


def compute_case(identity, output):
    import torch
    from rcwa_solver_auto import AutoRCWA, GroupTheoryOptions, Lattice, NVMOptions, OutputSpec
    from studies.gold_motheye.converge import _zero_order_x_source, _mean_poynting_z
    torch.set_num_threads(identity["cpu_threads"])
    device = nv.initialize_device(torch, identity["device"])
    eps = load_tabulated_nk(identity["ag_csv"])(identity["wavelength_nm"])
    operators, assembly_rows = [], []
    base = nv.memory_safe_class(AutoRCWA)

    class DiagnosticNV(base):
        def _d6_symmetrized_cartesian_pq(self, *args, **kwargs):
            result = super()._d6_symmetrized_cartesian_pq(*args, **kwargs)
            self._diagnostic_pq = result[0], result[3], result[4]
            return result

        def _d6_source_eigendecomposition(self, *args, **kwargs):
            result = super()._d6_source_eigendecomposition(*args, **kwargs)
            embed, p, q = self._diagnostic_pq
            be, bh = (embed.mH@basis for basis in self._polarization_bases)
            ps, qs = be.mH@p@bh, bh.mH@q@be
            size = be.shape[0]//2
            g = be.mH@torch.cat((bh[size:], -bh[:size]), 0)
            le, lh = antihermitian_part(g@qs), antihermitian_part(g.mH@ps)
            operators.append((le.detach().cpu(), lh.detach().cpu()))
            layer = self._polarized_layers[-1]
            eig_residual = relative_residual(torch, ps@qs@layer["electric"],
                layer["electric"]*layer["kz"][None, :].square())
            we, wh = torch.linalg.eigvalsh(le), torch.linalg.eigvalsh(lh)
            record = dict(layer=len(operators),
                electric_loss_min_eigenvalue=float(we.min()),
                electric_loss_negative_eigenvalues=int((we < -1e-8).sum()),
                magnetic_loss_min_eigenvalue=float(wh.min()),
                magnetic_loss_negative_eigenvalues=int((wh < -1e-8).sum()),
                electric_modal_condition_number=float(torch.linalg.cond(layer["electric"])),
                eigenproblem_relative_residual=eig_residual,
                symmetry_invariance_residual=float(self.group_theory_diagnostics[-1]["max_invariance_residual"]))
            assembly_rows.append(record)
            write_csv(output/"operator_diagnostics.csv", assembly_rows)
            del self._diagnostic_pq
            return result

        def add_layer_circle_nvm(self, *args, **kwargs):
            result = super().add_layer_circle_nvm(*args, **kwargs)
            nv.discard_unused_layer_storage(self)
            return result

    q, order, nz = identity["grid"], identity["order"], identity["slices"]
    g = identity["geometry"]
    sim = DiagnosticNV(freq=g["period_nm"]/identity["wavelength_nm"], order=[order, order],
        lattice=Lattice.triangular(1.), cascade="redheffer",
        outputs=OutputSpec(smatrix_size="half", fields="none"),
        nvm=NVMOptions(grid=(q, q), use_lanczos=False),
        group_theory=GroupTheoryOptions(enabled=True, symmetry="d6", strict=True, polarization="x"),
        verify_cascade=False, dtype=torch.complex128, device=device)
    sim.add_input_layer(eps=1., mu=1.)
    sim.add_output_layer(eps=eps, mu=1.)
    sim.set_incident_angle(0., 0.)
    centers = ((.5*(float(sim.L[0])+float(sim.L[1])*sim.cos_zeta),
                .5*float(sim.L[1])*sim.sin_zeta),)
    radii = []
    started = time.perf_counter()
    for i in range(nz):
        radius = g["tip_radius_nm"]+(g["base_radius_nm"]-g["tip_radius_nm"])*(i+.5)/nz
        radii.append(radius)
        sim.add_layer_circle_nvm(g["height_nm"]/nz/g["period_nm"], radius/g["period_nm"],
                                 1., eps, centers=centers, nx=q, ny=q, use_lanczos=False)
        if (i+1) % 25 == 0 or i+1 == nz:
            print(f"  assembly: layer {i+1}/{nz}", flush=True)
    sim._diagnostic_radii_nm, sim._diagnostic_period_nm = radii, g["period_nm"]
    sim._diagnostic_order = order
    sim.solve_global_smatrix()
    source = _zero_order_x_source(sim)
    pin = _mean_poynting_z(source, sim.Vi, direction=1)
    original_r = -_mean_poynting_z(sim.S[1]@source, sim.Vi, direction=-1)/pin
    original_substrate = _mean_poynting_z(sim.S[0]@source, sim.Vo, direction=1)/pin
    nv.write_json(output/"far_field.json", dict(reflectance=original_r, absorptance=1-original_r,
        transmittance_far=0., substrate_absorptance=original_substrate,
        motheye_absorptance=1-original_r-original_substrate))
    report = reconstruct(torch, sim, operators, source, output, identity["quadrature"])
    report.update(original_reflectance=original_r, original_substrate_absorptance=original_substrate,
        original_power_parity_max_error=max(abs(report["reflectance"]-original_r),
            abs(report["substrate_absorptance"]-original_substrate)),
        epsilon_real=eps.real, epsilon_imag=eps.imag, runtime_seconds=time.perf_counter()-started,
        peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
        physical_result=all(-1e-8 <= value <= 1+1e-8 for value in
                            (original_r, 1-original_r, original_substrate, 1-original_r-original_substrate)))
    report["far_field_physical"] = report["physical_result"]
    report["layer_flux_passive"] = report["negative_absorption_layers"] == 0
    report["physical_result"] = report["far_field_physical"] and report["layer_flux_passive"]
    report["diagnostic_checks_passed"] = (report["max_boundary_residual"] <= 1e-7
        and report["original_power_parity_max_error"] <= 1e-8
        and report["layer_sum_power_balance_error"] <= 1e-7
        and report["max_volume_flux_error"] <= 1e-7)
    report["interpretation"] = "Diagnostic consistency and passivity are separate; a nonphysical result is retained, not accepted as physical. Mode proxies are not additive powers."
    return report


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--order", type=int, default=14)
    p.add_argument("--wavelength-nm", type=float, default=550.)
    p.add_argument("--slices", type=int, default=210)
    p.add_argument("--grids", default="384,512,768,1024")
    p.add_argument("--period-nm", type=float, default=200.)
    p.add_argument("--height-nm", type=float, default=500.)
    p.add_argument("--tip-radius-nm", type=float, default=5.)
    p.add_argument("--base-radius-nm", type=float, default=95.)
    p.add_argument("--ag-csv", type=Path, default=ROOT/"studies/cu_ag_al_motheye_nv/data/Ag.csv")
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--quadrature", type=int, default=16, help="Initial Gauss points per layer; adapt up to at least 128.")
    p.add_argument("--output-root", type=Path, default=ROOT/"studies/ag_nv_m14_diagnostics/results/M14_Nz210_550nm")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--retry-failures", action="store_true")
    p.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    return p


def build_identities(args):
    grids = list(dict.fromkeys(int(x.strip()) for x in args.grids.split(",")))
    if (not grids or min(grids) < max(32, 4*args.order+4) or args.order < 1
            or args.slices < 1 or args.cpu_threads < 1 or args.quadrature < 2):
        raise ValueError("Positive order/slices/threads, quadrature >=2, and every Q >= max(32,4*M+4) are required.")
    if (min(args.period_nm, args.height_nm, args.tip_radius_nm) <= 0
            or not args.tip_radius_nm <= args.base_radius_nm < args.period_nm/2):
        raise ValueError("Require 0 < tip radius <= base radius < period/2 and positive height.")
    model = load_tabulated_nk(args.ag_csv.resolve())
    model(args.wavelength_nm)
    paths = [Path(__file__), ROOT/"rcwa_solver_auto.py", ROOT/"studies/transition_metals_nv/runtime.py",
             ROOT/"studies/shared/tabulated_nk.py", ROOT/"studies/gold_motheye/converge.py"]
    paths += sorted((ROOT/"rcwa_ext").rglob("*.py"))
    sources = {path.relative_to(ROOT).as_posix(): nv.digest(path) for path in paths}
    common = dict(version=1, material="Ag", structure="metal", order=args.order,
        wavelength_nm=args.wavelength_nm, slices=args.slices, device=args.device,
        cpu_threads=args.cpu_threads, quadrature=args.quadrature,
        geometry=dict(period_nm=args.period_nm, height_nm=args.height_nm,
                      tip_radius_nm=args.tip_radius_nm, base_radius_nm=args.base_radius_nm),
        ag_csv=str(args.ag_csv.resolve()), material_sha256=nv.digest(args.ag_csv),
        sources=sources, method="NV/D6 E1 x, complex128, Redheffer, Lanczos off; unchanged equations")
    return [dict(common, grid=q) for q in grids]


def worker(plan_path):
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    identity = plan["identity"]
    output = plan_path.parent
    signature = nv.identity_signature(identity)
    torch_module = None
    try:
        if (nv.digest(identity["ag_csv"]) != identity["material_sha256"] or
                any(nv.digest(ROOT/path) != value for path, value in identity["sources"].items())):
            raise ValueError("Material or diagnostic/solver source changed after preflight.")
        import torch as torch_module
        with torch_module.no_grad():
            result = compute_case(identity, output)
        report = dict(status="diagnostics_complete", signature=signature,
                      identity=identity, result=result)
        print(f"  R={100*result['reflectance']:.6f}%, A={100*result['absorptance']:.6f}%; "
              f"physical={result['physical_result']}, checks={result['diagnostic_checks_passed']}", flush=True)
        code = 0
    except Exception as error:
        oom = (torch_module is not None and isinstance(error, torch_module.cuda.OutOfMemoryError)
               and identity["device"].startswith("cuda"))
        report = dict(status="oom" if oom else "calculation_failed", signature=signature,
            identity=identity, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        if (output/"far_field.json").exists():
            report["far_field"] = json.loads((output/"far_field.json").read_text(encoding="utf-8"))
        print(f"  {report['status']}: {report['error']}; other Q jobs will continue.", flush=True)
        code = 2
    nv.write_json(output/"report.json", report)
    return code


def read_reports(output, identities):
    rows, reports = [], []
    for identity in identities:
        path = output/f"Q{identity['grid']}"/"report.json"
        report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else dict(status="pending", identity=identity)
        result = report.get("result", report.get("far_field", {}))
        rows.append(dict(grid=identity["grid"], order=identity["order"], slices=identity["slices"],
            wavelength_nm=identity["wavelength_nm"], status=report["status"],
            physical_result=result.get("physical_result"), diagnostic_checks_passed=result.get("diagnostic_checks_passed"),
            R_percent=None if "reflectance" not in result else 100*result["reflectance"],
            A_percent=None if "absorptance" not in result else 100*result["absorptance"],
            substrate_A_percent=None if "substrate_absorptance" not in result else 100*result["substrate_absorptance"],
            motheye_A_percent=None if "motheye_absorptance" not in result else 100*result["motheye_absorptance"],
            negative_absorption_layers=result.get("negative_absorption_layers"),
            max_boundary_residual=result.get("max_boundary_residual"),
            max_volume_flux_error=result.get("max_volume_flux_error"), error=report.get("error", "")))
        reports.append(report)
    return rows, reports


def persist_summary(output, rows, reports):
    write_csv(output/"Q_summary.csv", rows)
    nv.write_json(output/"summary.json", dict(cases=reports))
    changes = []
    for left, right in zip(rows, rows[1:]):
        if left["R_percent"] is None or right["R_percent"] is None:
            continue
        changes.append(dict(grid_low=left["grid"], grid_high=right["grid"],
            delta_R_pp=abs(right["R_percent"]-left["R_percent"]),
            delta_A_pp=abs(right["A_percent"]-left["A_percent"]),
            delta_motheye_A_pp=abs(right["motheye_A_percent"]-left["motheye_A_percent"]),
            delta_substrate_A_pp=abs(right["substrate_A_percent"]-left["substrate_A_percent"])))
    if changes:
        write_csv(output/"Q_changes.csv", changes)
    else:
        nv.atomic_text(output/"Q_changes.csv", "grid_low,grid_high,delta_R_pp,delta_A_pp,delta_motheye_A_pp,delta_substrate_A_pp\n")


def plot_reports(output, rows, reports):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:
        print(f"Plots unavailable: {error}. Saved CSV/JSON results are intact.", flush=True)
        return
    completed = [row for row in rows if row["R_percent"] is not None]
    if not completed:
        return
    plots = output/"figures"
    plots.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, field, label in zip(axes, ("R_percent", "A_percent", "motheye_A_percent"),
                               ("Reflectance R", "Total absorptance A", "Moth-eye absorptance")):
        ax.plot([row["grid"] for row in completed], [row[field] for row in completed], "o-")
        ax.set(xlabel="Normal-field sampling Q", ylabel="Incident power (%)", title=label)
        ax.grid(alpha=.25)
    first = rows[0]
    fig.suptitle(f"Ag metal | NV | M={first['order']}, Nz={first['slices']}, {first['wavelength_nm']:g} nm")
    fig.tight_layout(rect=(0, .03, 1, .94))
    for suffix in ("png", "svg"):
        fig.savefig(plots/f"Q_dependence.{suffix}", dpi=160)
    plt.close(fig)
    for report in reports:
        if report["status"] != "diagnostics_complete":
            continue
        q = report["identity"]["grid"]
        directory = output/f"Q{q}"
        with (directory/"layer_power.csv").open(encoding="utf-8-sig", newline="") as handle:
            powers = list(csv.DictReader(handle))
        depth = np.array([float(row["depth_mid_nm"]) for row in powers])
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        ax = axes[0, 0]
        ax.plot([float(row["depth_top_nm"]) for row in powers]+[float(powers[-1]["depth_bottom_nm"])],
                [100*float(row["flux_top"]) for row in powers]+[100*float(powers[-1]["flux_bottom"])])
        ax.set(xlabel="Depth (nm)", ylabel="Net downward flux (% of incident)", title="Boundary power flow")
        ax = axes[0, 1]
        ax.plot(depth, [float(row["layer_absorptance_pp"]) for row in powers])
        ax.axhline(0, color="black", lw=.8)
        ax.set(xlabel="Depth (nm)", ylabel="Layer absorption (percentage points)", title="Negative values indicate artificial gain")
        excitation = np.load(directory/"mode_excitation_map.npz")["electric_norm2"]
        im = axes[1, 0].imshow(np.log10(np.maximum(excitation, 1e-20)), aspect="auto", origin="upper",
            extent=(.5, excitation.shape[1]+.5, float(powers[-1]["depth_bottom_nm"]), 0), cmap="viridis")
        axes[1, 0].set(xlabel="Local mode index (not tracked across layers)", ylabel="Depth (nm)",
                       title="log10 modal electric-field norm proxy")
        fig.colorbar(im, ax=axes[1, 0])
        ax = axes[1, 1]
        ax.plot(depth, [100*(float(row["negative_electric_sector"])+float(row["negative_magnetic_sector"])) for row in powers])
        ax.set(xlabel="Depth (nm)", ylabel="Contribution (percentage points)", title="Excitation of negative loss-eigenvalue sectors")
        for ax in axes.flat:
            ax.grid(alpha=.2)
        fig.suptitle(f"Ag NV, M={first['order']}, Nz={first['slices']}, Q={q}, {first['wavelength_nm']:g} nm")
        fig.tight_layout(rect=(0, .02, 1, .94))
        for suffix in ("png", "svg"):
            fig.savefig(plots/f"layers_modes_Q{q}.{suffix}", dpi=160)
        plt.close(fig)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args.worker)
    try:
        identities = build_identities(args)
        output = args.output_root.resolve()
        if not args.plot_only:
            for identity in identities:
                path = output/f"Q{identity['grid']}"/"report.json"
                if path.exists() and json.loads(path.read_text(encoding="utf-8")).get("signature") != nv.identity_signature(identity):
                    raise ValueError(f"Saved Q={identity['grid']} settings/source changed; use a new --output-root.")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Ag metal NV diagnostics: M={args.order}, Nz={args.slices}, wavelength={args.wavelength_nm:g} nm; "
          f"Q={[identity['grid'] for identity in identities]}\nOutput: {output}", flush=True)
    if args.dry_run:
        print("Dry run: no RCWA computation or file changes. Nonphysical powers and failed/OOM Qs do not stop the next Q.")
        return 0
    output.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        nv.write_json(output/"plan.json", dict(identities=identities,
            note="Only Q changes. No change to material loss, geometry, NV factorization, precision or symmetry."))
        for identity in identities:
            directory = output/f"Q{identity['grid']}"
            directory.mkdir(exist_ok=True)
            path = directory/"report.json"
            if path.exists():
                previous = json.loads(path.read_text(encoding="utf-8"))
                if previous["status"] == "diagnostics_complete" or not args.retry_failures:
                    print(f"Reuse Q={identity['grid']}: {previous['status']}", flush=True)
                    continue
            nv.write_json(directory/"plan.json", dict(identity=identity))
            print(f"\n=== Q={identity['grid']} ===", flush=True)
            process = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
                                      "--worker", str(directory/"plan.json")], check=False)
            if not path.exists():
                nv.write_json(path, dict(status="calculation_failed", signature=nv.identity_signature(identity),
                    identity=identity, error=f"Worker exited {process.returncode} without a report; partial files retained."))
            rows, reports = read_reports(output, identities)
            persist_summary(output, rows, reports)
    rows, reports = read_reports(output, identities)
    persist_summary(output, rows, reports)
    if not args.no_plots:
        plot_reports(output, rows, reports)
    print(f"Saved: {output/'Q_summary.csv'}", flush=True)
    return 0 if all(row["status"] == "diagnostics_complete" for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
