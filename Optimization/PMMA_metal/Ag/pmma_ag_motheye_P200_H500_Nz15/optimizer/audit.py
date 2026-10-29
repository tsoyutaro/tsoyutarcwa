"""Audit both softmax blocks: mean(R+T) gradients, FD and order refinement."""
from __future__ import annotations
from .physics import objective


def check(request,device):
    import torch
    from .postprocess import mean_metrics
    c=request['config'];width=c['layers']+1
    logits=torch.tensor(request['logits'],dtype=torch.float64,device=device,requires_grad=True)
    if logits.numel()!=2*width:raise ValueError('Dual-radius audit needs 2*(layers+1) logits.')
    orders=sorted(set([c['optimization_order'],max(c['verification_orders'])]))
    waves=c['audit_wavelengths_nm'];records=[]
    generator=torch.Generator(device='cpu').manual_seed(c['seed']+713)
    directions=[]
    for i in range(c['audit_directions']):
        blocks=torch.randn((2,width),generator=generator,dtype=torch.float64)
        blocks-=blocks.mean(dim=1,keepdim=True)  # Each softmax block has its own additive gauge.
        direction=blocks.reshape(-1)
        norm=torch.linalg.vector_norm(direction)
        if float(norm)<=1e-15:raise ValueError('Degenerate gauge-free audit direction.')
        directions.append((direction/norm).to(device))
    for order in orders:
        baseline=objective(logits,c,request['lattice'],waves,order,c['optimization_grid'],device,gradient=True)
        g=baseline['gradient'];tests=[]
        if g is None or g.numel()!=2*width or not bool(torch.isfinite(g).all()):
            raise ValueError('Nonfinite or incomplete dual-radius logits gradient.')
        gauge_residuals=g.reshape(2,width).sum(dim=1).detach().cpu().tolist()
        gauge_passed=all(abs(x)<=c['gradient_atol'] for x in gauge_residuals)
        for i,d in enumerate(directions):
            differences=[]
            for h in (c['audit_fd_step'],c['audit_fd_step']/2):
                plus=objective(logits.detach()+h*d,c,request['lattice'],waves,order,c['optimization_grid'],device,gradient=False)['objective']
                minus=objective(logits.detach()-h*d,c,request['lattice'],waves,order,c['optimization_grid'],device,gradient=False)['objective']
                differences.append((plus-minus)/(2*h))
            ad=float(torch.dot(g,d).cpu())
            close=lambda a,b:abs(a-b)<=c['gradient_atol']+c['gradient_rtol']*max(abs(a),abs(b))
            tests.append(dict(direction=i,autograd=ad,fd_h=differences[0],fd_half_h=differences[1],
                              passed=close(ad,differences[1]) and close(*differences)))
        records.append(dict(order=order,objective=baseline['objective'],objective_name='mean_R_plus_T',
            gradient_logits=g.detach().cpu().tolist(),gradient_absorptance_logits=(-g).detach().cpu().tolist(),
            softmax_gauge_residuals=gauge_residuals,softmax_gauges_passed=gauge_passed,
            directional_checks=tests,directions_passed=all(t['passed'] for t in tests),
            **mean_metrics(baseline['rows'],c,waves)))
    a=torch.tensor(records[0]['gradient_logits'],dtype=torch.float64)
    b=torch.tensor(records[-1]['gradient_logits'],dtype=torch.float64)
    na,nb=float(a.norm()),float(b.norm());scale=max(na,nb)
    difference=float(torch.linalg.vector_norm(a-b))
    cosine=float(torch.dot(a,b)/(a.norm()*b.norm())) if min(na,nb)>1e-15 else None
    order_passed=(difference<=c['gradient_atol']+c['gradient_rtol']*scale and
        (scale<=c['gradient_atol'] or cosine is not None and cosine>=c['gradient_cosine_min'])) if len(records)>1 else None
    fd_passed=all(r['directions_passed'] and r['softmax_gauges_passed'] for r in records)
    return dict(status='completed',objective_name='mean_R_plus_T',wavelengths_nm=waves,
        variable_space='all 2*(Nz+1) logits of two monotone latent softmax profiles',
        variable_count=2*width,softmax_blocks=2,softmax_gauge_components_removed=2,
        gradient_definition='d mean(R+T)/d logits; absorptance gradient is its negative',
        direction_count=len(directions),directions_gauge_free=[d.detach().cpu().tolist() for d in directions],
        records=records,order_gradient_norm_difference=difference,
        order_gradient_relative_difference=difference/max(scale,1e-15),order_gradient_cosine=cosine,
        order_comparison_performed=len(records)>1,order_gradient_passed=order_passed,
        directional_fd_and_gauge_passed=fd_passed,passed=fd_passed and order_passed is not False,
        full_objective_wavelength_coverage=waves==c['optimization_wavelengths_nm'],
        limits='Directional FD, not exhaustive componentwise proof. The min/max radius map is piecewise differentiable; '
               'recheck at intermediate profiles. Sampled wavelengths and a single order do not establish whole-band/order convergence.')
