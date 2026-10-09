"""Audit internal TM modes, boundary projection and interface power continuity."""
from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

# Optional performance dependency. Set fallback limits before importing the
# numerical libraries so a fresh diagnostic subprocess requests one BLAS thread.
try:
    from threadpoolctl import threadpool_limits
except ModuleNotFoundError as exc:
    if exc.name != 'threadpoolctl':
        raise
    threadpool_limits = None
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy import linalg

from paper_reproductions.vallius2002.solver import (
    _PreparedLayer, _Diagnostics, _interface, _sqrt_outgoing, PreparedStack)
from paper_reproductions.vallius2002.devices import resolve_execution
from .adapters import load_study, layer_specs


def imaginary(matrix):
    return (matrix-matrix.conj().T)/(2j)


def loss_check(matrix):
    """Negative eigenvalues are meaningful relative to the operator scale."""
    minimum=float(linalg.eigvalsh(imaginary(matrix)).min())
    scale=max(1.,float(linalg.norm(matrix,2)))
    return dict(minimum_eigenvalue=minimum, operator_scale=scale,
                relative_minimum=minimum/scale, passive_at_relative_1e_10=minimum >= -1e-10*scale)


def right_solve(numerator,denominator):
    return linalg.solve(denominator.T,numerator.T,check_finite=False).T


def audit_layer(prepared,k,index):
    alpha=2*np.pi*prepared.orders/prepared.spec.period
    operator=k*k*prepared.f-alpha[:,None]*prepared.inverse_a*alpha[None,:]
    eigenvalues,e=linalg.eig(operator,prepared.b,check_finite=False)
    keep=np.argsort(np.abs(eigenvalues),kind='stable')[:prepared.count]
    eigenvalues,e=eigenvalues[keep],e[:,keep]
    gamma=_sqrt_outgoing(eigenvalues)
    w=prepared.K0@e
    scales=np.maximum(linalg.norm(w,axis=0),1e-30)
    e=e/scales[None,:];w=w/scales[None,:]
    reduced_metric=e.conj().T@prepared.b@e
    reduced_operator=e.conj().T@operator@e
    rhs=reduced_metric*gamma[None,:]
    traces={'direct':(prepared.KQ0@e)*gamma[None,:],
            'galerkin':linalg.solve(w.conj().T,rhs,check_finite=False)}
    row=dict(layer_index_zero_based=index,name=prepared.spec.name,
             quadrature_actual=prepared.quadrature_points,
             min_imag_gamma=float(gamma.imag.min()),
             generalized_eigen_residual=float(linalg.norm(operator@e-(prepared.b@e)*eigenvalues[None,:])/
                 max(1.,linalg.norm(operator@e))),
             full_inverse_metric_loss=loss_check(linalg.inv(prepared.b)),
             full_operator_loss=loss_check(operator),
             reduced_negative_metric_loss=loss_check(-reduced_metric),
             reduced_operator_loss=loss_check(reduced_operator),
             projected_field_condition=float(np.linalg.cond(w)),projections={})
    for label,v in traces.items():
        row['projections'][label]=dict(
            power_metric_relative_error=float(linalg.norm(w.conj().T@v-rhs)/max(1e-30,linalg.norm(rhs))),
            field_generator_loss=loss_check(right_solve(w*gamma[None,:],v)),
            companion_generator_loss=loss_check(right_solve(v*gamma[None,:],w)))
    return row,(w,traces,gamma)


def interface_audit(left,right,label,seed):
    wl,vl,_=left;wr,vr,_=right
    vl=vl[label];vr=vr[label]
    s=_interface(wl,vl,wr,vr,_Diagnostics(False))
    rng=np.random.default_rng(seed)
    a,b=(rng.normal(size=len(wl))+1j*rng.normal(size=len(wl)) for _ in range(2))
    c=s[0]@a+s[1]@b;d=s[2]@a+s[3]@b
    hl,ql=wl@(a+c),vl@(a-c)
    hr,qr=wr@(d+b),vr@(d-b)
    field_residual=max(float(linalg.norm(hl-hr)/max(1.,linalg.norm(hl),linalg.norm(hr))),
                       float(linalg.norm(ql-qr)/max(1.,linalg.norm(ql),linalg.norm(qr))))
    pl=float(np.vdot(hl,ql).real);pr=float(np.vdot(hr,qr).real)
    # The field-norm scale remains valid even for nearly cancelling fluxes.
    power_scale=max(1.,linalg.norm(hl)*linalg.norm(ql),linalg.norm(hr)*linalg.norm(qr))
    return dict(projection=label,relative_field_continuity_residual=field_residual,
                power_left=pl,power_right=pr,relative_interface_power_jump=float(abs(pl-pr)/power_scale))


def _torch_singular_values(matrix):
    import torch
    # QR-based SVD is suitable for condition diagnostics of ill-conditioned
    # matrices. The driver option is available only on CUDA.
    if matrix.is_cuda:
        return torch.linalg.svdvals(matrix,driver='gesvd')
    return torch.linalg.svdvals(matrix)


def loss_check_torch(matrix):
    import torch
    minimum=float(torch.linalg.eigvalsh((matrix-matrix.mH)/(2j)).amin().item())
    scale=max(1.,float(_torch_singular_values(matrix)[0].item()))
    return dict(minimum_eigenvalue=minimum,operator_scale=scale,
                relative_minimum=minimum/scale,passive_at_relative_1e_10=minimum >= -1e-10*scale)


def right_solve_torch(numerator,denominator):
    import torch
    return torch.linalg.solve(denominator.mT,numerator.mT).mT


def production_torch_modes(prepared,k,alpha0=0.):
    """Return the production W/V/gamma and its selected, normalized internal H.

    Observe the eigensolver result during one unmodified production modes()
    call. This avoids solving a second eigenproblem with a potentially different
    basis, and avoids changing numerical source hashes of saved comparisons.
    Hooks are scoped to this call in the single-process diagnostic, and pass
    every eigensolver result through unchanged.
    """
    import torch
    captures=[]
    original_eig,original_eigh=torch.linalg.eig,torch.linalg.eigh

    def observe(function,label):
        def wrapped(*args,**kwargs):
            result=function(*args,**kwargs)
            captures.append((label,result[0],result[1]))
            return result
        return wrapped

    with patch.object(torch.linalg,'eig',side_effect=observe(original_eig,'lu_solve_eig')):
        with patch.object(torch.linalg,'eigh',side_effect=observe(original_eigh,'cholesky_eigh')):
            w,v,gamma=prepared.modes(k,alpha0,'TM',retention='smallest_abs',q_projection='galerkin')
    if len(captures) != 1:
        raise RuntimeError('Expected one production TM eigenproblem for a nonhomogeneous ASR layer')
    label,values,h=captures[0]
    if label == 'cholesky_eigh':
        h=torch.linalg.solve_triangular(prepared.cholesky['b'].mH,h,upper=True)
    if prepared.internal_count > prepared.count:
        keep=torch.argsort(torch.abs(values),stable=True)[:prepared.count]
        values,h=values[keep],h[:,keep]
    projection=prepared.K0 if alpha0 == 0 else prepared._projection(alpha0)
    scales=torch.clamp(torch.linalg.vector_norm(projection@h,dim=0),min=1e-30)
    h=h/scales[None,:]
    return w,v,gamma,h,values,label,projection


def audit_layer_torch(prepared,k,index,alpha0=0.):
    """All dense diagnostics stay on prepared.device; only scalars are exported."""
    import torch
    w,v,gamma,h,values,eigenproblem,projection=production_torch_modes(prepared,k,alpha0)
    alpha=alpha0+2*np.pi*prepared.orders/prepared.spec.period
    operator=k*k*prepared.f-alpha[:,None]*prepared.inverse_a*alpha[None,:]
    reduced_metric=h.mH@prepared.b@h
    reduced_operator=h.mH@operator@h
    rhs=reduced_metric*gamma[None,:]
    reciprocal=prepared.KQ0 if alpha0 == 0 else prepared._projection(alpha0,reciprocal=True)
    traces={'direct':(reciprocal@h)*gamma[None,:],'galerkin':v}
    singular_values=_torch_singular_values(w)
    projected=operator@h
    row=dict(layer_index_zero_based=index,name=prepared.spec.name,
             quadrature_actual=prepared.quadrature_points,
             modal_trace_source='TorchPreparedLayer.modes',modal_eigenproblem=eigenproblem,
             min_imag_gamma=float(gamma.imag.amin().item()),
             generalized_eigen_residual=float((torch.linalg.matrix_norm(
                 projected-(prepared.b@h)*values[None,:])/
                 torch.clamp(torch.linalg.matrix_norm(projected),min=1.)).item()),
             production_field_projection_relative_error=float((torch.linalg.matrix_norm(projection@h-w)/
                 torch.clamp(torch.linalg.matrix_norm(w),min=1e-30)).item()),
             full_inverse_metric_loss=loss_check_torch(torch.linalg.solve(prepared.b,prepared.identity)),
             full_operator_loss=loss_check_torch(operator),
             reduced_negative_metric_loss=loss_check_torch(-reduced_metric),
             reduced_operator_loss=loss_check_torch(reduced_operator),
             projected_field_condition=float((singular_values[0]/singular_values[-1]).item()),
             projections={})
    for label,trace in traces.items():
        row['projections'][label]=dict(
            power_metric_relative_error=float((torch.linalg.matrix_norm(w.mH@trace-rhs)/
                torch.clamp(torch.linalg.matrix_norm(rhs),min=1e-30)).item()),
            field_generator_loss=loss_check_torch(right_solve_torch(w*gamma[None,:],trace)),
            companion_generator_loss=loss_check_torch(right_solve_torch(trace*gamma[None,:],w)))
    return row,(w,traces,gamma)


def interface_audit_torch(left,right,label,seed):
    import torch
    from paper_reproductions.vallius2002.torch_backend import (
        _interface as torch_interface,_Diagnostics as TorchDiagnostics)
    wl,vl,_=left;wr,vr,_=right
    vl=vl[label];vr=vr[label]
    identity=torch.eye(len(wl),dtype=wl.dtype,device=wl.device)
    s=torch_interface(wl,vl,wr,vr,TorchDiagnostics(False,identity))
    generator=torch.Generator(device=wl.device).manual_seed(seed)
    a,b=(torch.randn(len(wl),dtype=wl.dtype,device=wl.device,generator=generator) for _ in range(2))
    c=s[0]@a+s[1]@b;d=s[2]@a+s[3]@b
    hl,ql=wl@(a+c),vl@(a-c)
    hr,qr=wr@(d+b),vr@(d-b)
    norm=torch.linalg.vector_norm
    field_residual=max(float((norm(hl-hr)/torch.clamp(torch.maximum(norm(hl),norm(hr)),min=1.)).item()),
                       float((norm(ql-qr)/torch.clamp(torch.maximum(norm(ql),norm(qr)),min=1.)).item()))
    pl,pr=torch.vdot(hl,ql).real,torch.vdot(hr,qr).real
    power_scale=torch.clamp(torch.maximum(norm(hl)*norm(ql),norm(hr)*norm(qr)),min=1.)
    return dict(projection=label,relative_field_continuity_residual=field_residual,
                power_left=float(pl.item()),power_right=float(pr.item()),
                relative_interface_power_jump=float((torch.abs(pl-pr)/power_scale).item()))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',choices=('gold_grating_1d','pmma_gold_grating_1d'),default='pmma_gold_grating_1d')
    parser.add_argument('--order',type=int,default=16)
    parser.add_argument('--wavelength',type=float,default=600.)
    parser.add_argument('--slices',type=int,default=300)
    parser.add_argument('--ratio',type=int,default=4)
    parser.add_argument('--quadrature',type=int,default=192)
    parser.add_argument('--G',type=float,default=.001)
    parser.add_argument('--device',default='auto',help='auto, cpu, cuda, or cuda:<index>; explicit CUDA never falls back')
    parser.add_argument('--backend',choices=('auto','scipy','torch'),default='auto')
    parser.add_argument('--check-only',action='store_true',help='Check requested device/backend and exit without computing')
    parser.add_argument('--output',type=Path,default=Path('studies/asr_1d_comparison/results/tm_stage_audit.json'))
    args=parser.parse_args(argv)
    if args.order < 1 or args.slices < 1 or args.ratio < 1 or args.quadrature < 8:
        parser.error('Order, slices, ratio must be positive and quadrature must be >= 8')
    if not np.isfinite(args.G) or args.G <= 0 or not np.isfinite(args.wavelength) or args.wavelength <= 0:
        parser.error('G and wavelength must be finite and positive')
    execution=resolve_execution(args.device,args.backend)
    execution['condition_number_method']=(
        '2-norm from SVD (PyTorch gesvd)' if execution['device'].startswith('cuda') else
        '2-norm from SVD (PyTorch)' if execution['backend'] == 'torch' else '2-norm from SVD (NumPy)')
    if args.check_only:
        print(json.dumps(execution,indent=2),flush=True)
        return
    config,model,_=load_study(args.study)
    layers,substrate=layer_specs(args.study,config,args.slices,model(args.wavelength))
    n=2*args.order+1;k=2*np.pi/(args.wavelength/config['geometry']['period_nm'])
    probes=sorted({0,1,len(layers)//2,len(layers)//2+1,len(layers)-2,len(layers)-1})
    audit=[];traces={}
    evaluation_wavelength_nm=args.wavelength
    cutoff_regularized=False
    print(f"Projection audit: {execution['backend']} on {execution['device']} ({execution['device_name']})",flush=True)
    if threadpool_limits is None:
        print('threadpoolctl is unavailable; requesting one CPU thread using '
              'BLAS/OpenMP environment variables',flush=True)
    thread_context = threadpool_limits(limits=1) if threadpool_limits is not None else nullcontext()
    with thread_context:
        if execution['backend'] == 'torch':
            import torch
            torch.set_num_threads(1)
            with torch.no_grad():
                stack=PreparedStack([layers[index] for index in probes],polarization='TM',method='asr',
                                    harmonics=n,oversampling=args.ratio,quadrature=args.quadrature,G=args.G,
                                    epsilon_in=1.,epsilon_out=substrate,diagnostics=False,
                                    retention='smallest_abs',q_projection='galerkin',
                                    device=execution['device'],backend='torch')
                _,evaluation,k,alpha0,cutoff_regularized=stack._wavelength_parameters(
                    args.wavelength/config['geometry']['period_nm'],1e-12)
                evaluation_wavelength_nm=evaluation*config['geometry']['period_nm']
                for index,layer in zip(probes,stack.prepared_layers):
                    if layer.homogeneous:
                        continue
                    row,traces[index]=audit_layer_torch(layer,k,index,alpha0)
                    row['name']=layers[index].name
                    audit.append(row)
                interfaces=[]
                for index in probes:
                    if index in traces and index+1 in traces:
                        for label in ('direct','galerkin'):
                            row=interface_audit_torch(traces[index],traces[index+1],label,1234+index)
                            interfaces.append(dict(left_layer_index_zero_based=index,**row))
        else:
            for index in probes:
                layer=_PreparedLayer(layers[index],'asr',n,args.ratio*n,args.quadrature,args.G)
                if layer.homogeneous:
                    continue
                row,traces[index]=audit_layer(layer,k,index)
                audit.append(row)
            interfaces=[]
            for index in probes:
                if index in traces and index+1 in traces:
                    for label in ('direct','galerkin'):
                        row=interface_audit(traces[index],traces[index+1],label,1234+index)
                        interfaces.append(dict(left_layer_index_zero_based=index,**row))
    output=dict(study=args.study,order=args.order,harmonics=n,internal_dimension=args.ratio*n,
                wavelength_nm=args.wavelength,slices=args.slices,total_finite_layers=len(layers),G=args.G,
                quadrature_minimum=args.quadrature,backend=execution['backend'],device=execution['device'],
                execution=execution,evaluation_wavelength_nm=evaluation_wavelength_nm,
                cutoff_regularized=bool(cutoff_regularized),dtype='complex128',
                coefficient_preparation_device='cpu',linear_algebra_device=execution['device'],
                modal_trace_source='TorchPreparedLayer.modes' if execution['backend'] == 'torch' else 'SciPy generalized eig (QZ)',
                cpu_threads_requested=1,
                cpu_thread_control='threadpoolctl' if threadpool_limits is not None else 'environment_variables',
                selected_layers=audit,interfaces=interfaces,
                scope='Stage audit at sampled layers and interfaces; independent of a full-stack port residual.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(args.output)


if __name__=='__main__':main()
