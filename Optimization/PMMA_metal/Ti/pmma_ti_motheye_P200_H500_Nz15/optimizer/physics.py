"""Dispersive concentric PMMA-core/metal-shell RCWA on lossless PMMA.

The native D6 path constructs analytic two-disk epsilon and inverse-epsilon
coefficients and uses the existing symmetric-NV E1 partner-row assembly.
The reference path calls the existing full core-shell NVM implementation.
"""
from __future__ import annotations
from collections import OrderedDict
import math
import time
import torch
from torch.utils.checkpoint import checkpoint
from .geometry import radii_tensor
from .materials import metal_epsilon_at, pmma_epsilon_at, pmma_extrapolated
from rcwa_solver_auto import AutoRCWA, GroupTheoryOptions, Lattice, NVMOptions, OutputSpec
from rcwa_ext.acceleration import d6
from rcwa_ext.acceleration.ports import CompactPort
from rcwa_ext.nvm import _jinc


class NonphysicalResult(RuntimeError):
    pass


def sync(device):
    if torch.device(device).type=='cuda':torch.cuda.synchronize(device)


def weights(wavelengths, kind):
    if kind=='arithmetic' or len(wavelengths)==1:return [1./len(wavelengths)]*len(wavelengths)
    if kind!='trapezoidal':raise ValueError('Unknown wavelength weighting.')
    from .helpers import trapezoid_weights
    return trapezoid_weights(wavelengths)


def mean_metrics(rows,config):
    ww=weights([r['wavelength_nm'] for r in rows],config['objective_weighting'])
    return dict(mean_reflectance=sum(w*r['reflectance'] for w,r in zip(ww,rows)),
                mean_transmittance=sum(w*r['transmittance'] for w,r in zip(ww,rows)),
                mean_absorptance=sum(w*r['absorptance_total'] for w,r in zip(ww,rows)))


def make_sim(config,lattice,wavelength,order,grid,device,release=True):
    if lattice!='hexagonal':raise ValueError('This core-shell package supports the hexagonal D6 lattice.')
    eps=complex(pmma_epsilon_at(config,wavelength))
    if eps.real<=0 or abs(eps.imag)>1e-12:raise ValueError('PMMA substrate must be lossless with positive epsilon.')
    sim=AutoRCWA(freq=config['period_nm']/wavelength,order=[order]*2,
        lattice=Lattice.triangular(1.),nvm=NVMOptions(grid=(grid,grid),use_lanczos=False),
        outputs=OutputSpec(smatrix_size='half',fields='none'),cascade='redheffer',
        group_theory=GroupTheoryOptions(enabled=True,symmetry='d6',polarization='x',strict=True),
        dtype=torch.complex128,device=device,verify_cascade=False,acceleration='reference')
    sim.add_input_layer(eps=1.)
    sim.add_output_layer(eps=eps)
    sim.set_incident_angle(0.,0.)
    return sim


def source(sim):
    value=torch.zeros(2*sim.order_N,dtype=sim._dtype,device=sim._device)
    index=int(sim.order[0])*(2*int(sim.order[1])+1)+int(sim.order[1])
    value[index]=1
    return value


def power_row(R,T,config,wavelength):
    rv,tv=float(R.detach().cpu()),float(T.detach().cpu())
    av=1-rv-tv
    tol=config.get('passivity_tolerance',1e-6)
    if not all(math.isfinite(v) and -tol<=v<=1+tol for v in (rv,tv,av)):
        raise NonphysicalResult(f'Invalid core-shell power: wavelength_nm={wavelength:g}, R={rv:.12g}, T={tv:.12g}, A={av:.12g}.')
    return dict(wavelength_nm=float(wavelength),reflectance=rv,transmittance=tv,
        power_into_substrate=tv,absorptance_total=av,absorptance=av,
        substrate_absorptance=0.,motheye_absorptance=av,
        pmma_shortwave_extrapolated=pmma_extrapolated(config,wavelength),
        epsilon_core_real=complex(pmma_epsilon_at(config,wavelength)).real,
        epsilon_shell_real=complex(metal_epsilon_at(config,wavelength)).real,
        epsilon_shell_imag=complex(metal_epsilon_at(config,wavelength)).imag,
        absorption_evaluation='1-R-T into lossless PMMA substrate; no independent volume-loss audit')


def powers(sim,transmitted,reflected,config,wavelength=None):
    incident=source(sim)
    def flux(electric,medium,direction):
        magnetic=direction*(medium@electric);n=electric.numel()//2
        return .5*torch.real((electric[:n]*magnetic[n:].conj()-electric[n:]*magnetic[:n].conj()).sum())
    incoming=flux(incident,sim.Vi,1)
    R=-flux(reflected,sim.Vi,-1)/incoming
    T=flux(transmitted,sim.Vo,1)/incoming
    wave=float(wavelength if wavelength is not None else config['period_nm']/float(torch.real(sim.freq)))
    return R+T,power_row(R,T,config,wave)


def solve_train(radii,config,lattice,wavelength,order,grid,device,*,checkpoint_layers=None,release=True):
    """Independent existing full core-shell NVM, retaining only reduced modes."""
    sim=make_sim(config,lattice,wavelength,order,grid,device)
    core_eps=pmma_epsilon_at(config,wavelength)
    shell_eps=metal_epsilon_at(config,wavelength)
    depth=config['height_nm']/(config['layers']*config['period_nm'])
    center=(d6.center_of(sim),)
    checkpointed=bool(config.get('checkpoint_layers',False) if checkpoint_layers is None else checkpoint_layers)
    checkpointed=checkpointed and radii.requires_grad
    def one_layer(pair):
        local=make_sim(config,lattice,wavelength,order,grid,device)
        local.add_layer_circle_shell_nvm(depth,pair[0]/config['period_nm'],pair[1]/config['period_nm'],
            1.,shell_eps,core_eps,centers=center,nx=grid,ny=grid,use_lanczos=False)
        modes=local._polarized_layers[0]
        return modes['electric'],modes['magnetic'],modes['kz'],*local._polarization_bases
    if checkpointed:
        for pair in radii:
            electric,magnetic,kz,eb,hb=checkpoint(one_layer,pair,use_reentrant=False,preserve_rng_state=False)
            sim._install_polarization_basis(eb.detach(),hb.detach(),1e-8)
            sim._polarized_layers.append(dict(electric=electric,magnetic=magnetic,kz=kz))
            sim.thickness.append(depth);sim.layer_N+=1
    else:
        for pair in radii:
            sim.add_layer_circle_shell_nvm(depth,pair[0]/config['period_nm'],pair[1]/config['period_nm'],
                1.,shell_eps,core_eps,centers=center,nx=grid,ny=grid,use_lanczos=False)
            if release:
                for name in ('P','Q','eps_conv','mu_conv','E_eigvec','H_eigvec'):
                    getattr(sim,name)[-1]=None
                sim._nvm_eps_tensor=None
    transmitted,reflected=sim._solve_polarization_reduced_smatrix(incident_source=source(sim))
    value,row=powers(sim,transmitted,reflected,config,wavelength)
    row.update(storage='reference-full-core-shell-NVM-reduced-modes',
               eigen_dimension=sim._polarization_bases[0].shape[1],checkpoint_layers=checkpointed)
    return value,row


def objective_reference(logits,config,lattice,wavelengths,order,grid,device,*,gradient):
    sync(device);started=time.perf_counter()
    total=0.;derivative=torch.zeros_like(logits);rows=[]
    for wave,weight in zip(wavelengths,weights(wavelengths,config['objective_weighting'])):
        with torch.set_grad_enabled(gradient):
            radii=radii_tensor(logits,config)
            value,row=solve_train(radii,config,lattice,wave,order,grid,device)
            if gradient:derivative+=torch.autograd.grad(weight*value,logits)[0].detach()
            total+=weight*(row['reflectance']+row['transmittance'])
        rows.append(row)
    if not math.isfinite(total) or gradient and not bool(torch.isfinite(derivative).all()):
        raise NonphysicalResult('Nonfinite core-shell objective/logits gradient.')
    sync(device)
    return dict(objective=total,absorptance=1-total,gradient=derivative if gradient else None,
                rows=rows,seconds=time.perf_counter()-started,backend='core-shell-reference',
                **mean_metrics(rows,config))


class CoreShellEvaluator:
    backend_name='d6-core-shell-shared-v1'
    def __init__(self):self._cache=OrderedDict()
    def cache(self,c,lattice,waves,order,grid,device):
        core=[complex(pmma_epsilon_at(c,w)) for w in waves]
        shell=[complex(metal_epsilon_at(c,w)) for w in waves]
        key=(lattice,order,grid,c['period_nm'],c['height_nm'],c['layers'],tuple(waves),tuple(core),tuple(shell),str(device))
        if key not in self._cache:
            with torch.no_grad():
                template=make_sim(c,lattice,waves[0],order,grid,device)
                plan=d6.get_plan(template)
                ports=[CompactPort(make_sim(c,lattice,w,order,grid,device),plan,
                    c['height_nm']/(c['layers']*c['period_nm']),c['layers']) for w in waves]
                gx=plan.m.to(torch.float64)*(2*math.pi/float(template.L[0]))
                gy=(plan.n.to(torch.float64)*(2*math.pi/float(template.L[1]))-template.cos_zeta*gx)/template.sin_zeta
                dx,dy=gx[:,None]-gx[None,:],gy[:,None]-gy[None,:]
                cx,cy=d6.center_of(template)
                norm=torch.sqrt(dx*dx+dy*dy)
                phase=torch.exp(-1j*(dx*cx+dy*cy))
                area=float(template.L[0])*float(template.L[1])*template.sin_zeta
            self._cache[key]=dict(template=template,plan=plan,ports=ports,core=core,shell=shell,
                identity=template._eye(plan.scalar_dimension),norm=norm,phase=phase,area=area,
                grid=grid,layers=c['layers'],device=device)
            while len(self._cache)>2:self._cache.popitem(last=False)
        self._cache.move_to_end(key)
        return self._cache[key]
    def geometry_pair(self,pair,cache):
        core,outer=pair.unbind()
        # The reference taper is controlled by the inner interface radius.
        eps,_,normal=d6.native_inputs(cache['template'],core,2.,cache['grid'],cache['plan'])
        disk_core=(eps-cache['identity'])/3.
        disk_outer=(math.pi*outer**2/cache['area'])*_jinc(cache['norm']*outer)*cache['phase']
        return disk_core,disk_outer,normal
    def response(self,cache,leaves,index,checkpointed):
        port=cache['ports'][index];ce,se=cache['core'][index],cache['shell'][index]
        def modes(dc,do,normal):
            eps=cache['identity']+(se-1)*do+(ce-se)*dc
            rule=cache['identity']+(1/se-1)*do+(1/ce-1/se)*dc
            p,q,_=d6.reduced_pq(port,eps,rule,normal,cache['plan'])
            squared,electric=port._eig(p@q)
            kz=port._positive_kz(squared)
            magnetic=port._magnetic_eigenvectors(p,q,electric,kz)
            return electric,magnetic,kz
        layers=[]
        for j in range(cache['layers']):
            args=leaves[3*j:3*j+3]
            electric,magnetic,kz=checkpoint(modes,*args,use_reentrant=False,preserve_rng_state=False) if checkpointed else modes(*args)
            layers.append(dict(electric=electric,magnetic=magnetic,kz=kz))
        transmission,reflection,_,_=port._polarized_redheffer_scattering(layers,port.input_interface,port.output_interface)
        t,r=transmission@port.source,reflection@port.source
        R=.5*torch.real(r.conj()@(port.flux_in@r))/port.incident_flux
        T=.5*torch.real(t.conj()@(port.flux_out@t))/port.incident_flux
        return R,T
    def evaluate_radii(self,radii_nm,c,lattice,waves,order,grid,device,*,gradient):
        if radii_nm.shape!=(c['layers'],2) or not bool(torch.isfinite(radii_nm).all()):
            raise ValueError('Expected finite [layers,2] core/outer radii.')
        if not bool(((radii_nm[:,0]>0)&(radii_nm[:,1]>radii_nm[:,0])&(radii_nm[:,1]<c['period_nm']/2)).all()):
            raise ValueError('Require 0 < PMMA radius < metal outer radius < period/2.')
        cache=self.cache(c,lattice,waves,order,grid,device)
        values=(radii_nm/c['period_nm']).detach().clone().requires_grad_(gradient)
        checkpointed=bool(c.get('checkpoint_layers',False) and gradient)
        originals=[]
        for pair in values:
            build=lambda x:self.geometry_pair(x,cache)
            originals.extend(checkpoint(build,pair,use_reentrant=False,preserve_rng_state=False) if checkpointed else build(pair))
        leaves=tuple(v.detach().requires_grad_(gradient) for v in originals)
        covectors=[torch.zeros_like(v) for v in leaves] if gradient else None
        rows=[];total=0.
        for index,(wave,weight) in enumerate(zip(waves,weights(waves,c['objective_weighting']))):
            R,T=self.response(cache,leaves,index,checkpointed)
            row=power_row(R,T,c,wave)
            if gradient:
                derivatives=torch.autograd.grad(weight*(R+T),leaves)
                for a,d in zip(covectors,derivatives):a.add_(d)
            total+=weight*(row['reflectance']+row['transmittance'])
            row.update(storage=self.backend_name,eigen_dimension=cache['plan'].reduced_dimension,
                checkpoint_layers=checkpointed,retained_scalar_harmonics=cache['plan'].scalar_dimension)
            rows.append(row)
        radius_gradient=torch.autograd.grad(tuple(originals),values,grad_outputs=tuple(covectors))[0].detach()/c['period_nm'] if gradient else None
        return total,radius_gradient,rows
    def __call__(self,logits,c,lattice,waves,order,grid,device,*,gradient):
        sync(device);started=time.perf_counter()
        with torch.set_grad_enabled(gradient):
            radii=radii_tensor(logits,c)
            value,dr,rows=self.evaluate_radii(radii,c,lattice,waves,order,grid,device,gradient=gradient)
            derivative=torch.autograd.grad(radii,logits,grad_outputs=dr)[0].detach() if gradient else None
        if not math.isfinite(value) or gradient and not bool(torch.isfinite(derivative).all()):
            raise NonphysicalResult('Nonfinite shared core-shell objective/logits gradient.')
        sync(device)
        return dict(objective=value,absorptance=1-value,gradient=derivative,rows=rows,
                    seconds=time.perf_counter()-started,backend=self.backend_name,**mean_metrics(rows,c))


_evaluator=None

def objective(logits,config,lattice,wavelengths,order,grid,device,*,gradient):
    global _evaluator
    if config.get('rcwa_acceleration','auto') not in ('auto','reference'):raise ValueError('Use auto or reference core-shell backend.')
    if gradient and not logits.requires_grad:logits=logits.detach().requires_grad_(True)
    if config.get('rcwa_acceleration','auto')=='reference':
        return objective_reference(logits,config,lattice,wavelengths,order,grid,device,gradient=gradient)
    if _evaluator is None:_evaluator=CoreShellEvaluator()
    return _evaluator(logits,config,lattice,wavelengths,order,grid,device,gradient=gradient)


def solve_verify(radii,config,lattice,wavelength,order,grid,device):
    global _evaluator
    if _evaluator is None:_evaluator=CoreShellEvaluator()
    with torch.no_grad():
        if config.get('rcwa_acceleration','auto')=='reference':
            _,row=solve_train(radii,config,lattice,wavelength,order,grid,device)
            return row
        _,_,rows=_evaluator.evaluate_radii(radii,config,lattice,[wavelength],order,grid,device,gradient=False)
        return rows[0]
