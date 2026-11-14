"""Unregularized radial normal projector; no Lanczos or finite-width taper.

The NVM auxiliary field is retained. Isolated center samples use I/2 and
nearest-center tie samples use the mean of all tied radial projectors.
"""
DEFINITION = "nearest-radial-center-and-tie-mean-v1"

def unsmoothed_projection(nx,ny,dxs,dys,distance,lx,ly):
    """Nearest-center radial projector, with point/line quadrature conventions.

    A center sample uses I/2, the angular mean of n n^T. At a machine-precision
    nearest-center tie, average ALL equally nearest projectors (including triple
    ties). There is no grid-dependent length or finite transition band.
    """
    import torch
    scale=max(abs(float(lx)),abs(float(ly)))
    eps=torch.finfo(dxs.dtype).eps
    tolerance_squared=64*eps*scale*scale
    center=distance[0] <= (64*eps*scale)**2
    fields=[nx*nx,nx*ny,ny*ny]
    ties=(distance[1]-distance[0]) <= tolerance_squared
    if bool(ties.any()):
        u,v=dxs[:,ties],dys[:,ties]
        squared=u*u+v*v
        nearest=squared.amin(dim=0,keepdim=True)
        equally_near=(squared-nearest).abs() <= tolerance_squared
        denominator=torch.clamp(squared,min=(64*eps*scale)**2)
        count=equally_near.sum(dim=0).to(dxs.dtype)
        for i,value in enumerate([u*u/denominator,u*v/denominator,v*v/denominator]):
            fields[i][ties]=(value*equally_near).sum(dim=0)/count
    return (torch.where(center,torch.full_like(nx,.5),fields[0]),
            torch.where(center,torch.zeros_like(nx),fields[1]),
            torch.where(center,torch.full_like(nx,.5),fields[2]))
