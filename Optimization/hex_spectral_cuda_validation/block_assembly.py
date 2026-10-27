"""Validation adapter: frozen reference versus D6-specialized implementation."""
from __future__ import annotations
import torch
import legacy_block_assembly as legacy
import d6_nvm
from rcwa_solver_auto import AutoRCWA, GroupTheoryOptions, Lattice, NVMOptions, OutputSpec

relative_error = legacy.relative_error
center_of = legacy.center_of
check_supported = legacy.check_supported
source = legacy.source
flux = legacy.flux
reference_pq = legacy.reference_pq
material_inputs_reference = legacy.material_inputs


def make_sim(lattice,order,grid,index,wavelength_nm,period_nm,device):
    sim = d6_nvm.D6FastRCWA(freq=period_nm/wavelength_nm,order=[order,order],
        lattice=Lattice.square(1.) if lattice=='square' else Lattice.triangular(1.),
        nvm=NVMOptions(grid=(grid,grid),use_lanczos=False),
        outputs=OutputSpec(smatrix_size='half',fields='none'),cascade='redheffer',
        group_theory=GroupTheoryOptions(enabled=True,symmetry='c2v' if lattice=='square' else 'd6',polarization='x',strict=True),
        dtype=torch.complex128,device=device,verify_cascade=False)
    sim.add_input_layer(eps=1.)
    sim.add_output_layer(eps=index**2)
    sim.set_incident_angle(0.,0.)
    return sim


def GeometryContext(sim):
    return d6_nvm.get_plan(sim) if sim.lattice_kind=='triangular' else legacy.GeometryContext(sim)


def BlockInverse(sim,matrix,context):
    return d6_nvm.D6BlockInverse(sim,matrix,context) if context.triangular else legacy.BlockInverse(sim,matrix,context)


def material_inputs(sim,radius,index,grid,context):
    check_supported(sim,radius,center_of(sim))
    if context.triangular:
        return d6_nvm.native_inputs(sim,radius,index,grid,context)
    return legacy.material_inputs(sim,radius,index,grid,context)


def candidate_pq(sim,eps,inverse_rule,projection,context):
    if context.triangular:
        return d6_nvm.reduced_pq(sim,eps,inverse_rule,projection,context)
    return legacy.candidate_pq(sim,eps,inverse_rule,projection,context)


def optical_response(sim,radii,index,height_nm,period_nm,grid,variant,context):
    depth = height_nm/(len(radii)*period_nm)
    for radius in radii:
        if variant=='reference':
            # Invoke the frozen implementation directly, bypassing the subclass.
            AutoRCWA.add_layer_circle_nvm(sim,depth,radius,1.,index**2,(center_of(sim),),nx=grid,ny=grid)
        elif context.triangular:
            sim.add_layer_circle_nvm(depth,radius,1.,index**2,(center_of(sim),),nx=grid,ny=grid)
        else:
            legacy.add_candidate_layer(sim,radius,index,depth,grid,context)
    incident = source(sim)
    transmitted,reflected = sim._solve_polarization_reduced_smatrix(incident_source=incident)
    incoming = flux(incident,sim.Vi,1.)
    return (-flux(reflected,sim.Vi,-1.)/incoming,flux(transmitted,sim.Vo,1.)/incoming,transmitted,reflected)
