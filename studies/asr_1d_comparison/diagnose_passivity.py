"""Audit internal TM modes, boundary projection and interface power continuity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits

from paper_reproductions.vallius2002.solver import (
    _PreparedLayer, _Diagnostics, _interface, _sqrt_outgoing)
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


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',choices=('gold_grating_1d','pmma_gold_grating_1d'),default='pmma_gold_grating_1d')
    parser.add_argument('--order',type=int,default=16)
    parser.add_argument('--wavelength',type=float,default=600.)
    parser.add_argument('--slices',type=int,default=300)
    parser.add_argument('--ratio',type=int,default=4)
    parser.add_argument('--quadrature',type=int,default=192)
    parser.add_argument('--G',type=float,default=.001)
    parser.add_argument('--output',type=Path,default=Path('studies/asr_1d_comparison/results/tm_stage_audit.json'))
    args=parser.parse_args(argv)
    config,model,_=load_study(args.study)
    layers,_=layer_specs(args.study,config,args.slices,model(args.wavelength))
    n=2*args.order+1;k=2*np.pi/(args.wavelength/config['geometry']['period_nm'])
    probes=sorted({0,1,len(layers)//2,len(layers)//2+1,len(layers)-2,len(layers)-1})
    audit=[];traces={}
    with threadpool_limits(limits=1):
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
                quadrature_minimum=args.quadrature,backend='scipy',device='cpu',
                selected_layers=audit,interfaces=interfaces,
                scope='Stage audit at sampled layers and interfaces; independent of a full-stack port residual.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(args.output)


if __name__=='__main__':main()
