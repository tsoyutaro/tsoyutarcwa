from pathlib import Path
import json,sys,time
import numpy as np
STUDY=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(STUDY))
from model import solver_root
solver_root()
import torch
torch.set_num_threads(2)
from threadpoolctl import threadpool_limits
limits=threadpool_limits(limits=2)
from fff1d.solver import Backend,star,interface,admittance,local_s,full_generator

def calculate(m=32,nz=200,fff=True,grid=4096,device='cpu'):
    P,H,wave=500.,200.,632.8;metal=(1.3+7.6j)**2
    backend=Backend(device);orders=np.arange(-m,m+1);n=len(orders);I=backend.eye(n)
    differences=orders[:,None]-orders[None,:]
    k=backend.array(orders*wave/P+np.sin(np.deg2rad(40.)))
    x=np.arange(grid)*P/grid
    slope=-np.pi*H/P*np.sin(2*np.pi*x/P)
    c2=1/(1+slope*slope);cs=slope/(1+slope*slope)
    C2=backend.array((np.fft.fft(c2)/grid)[differences%grid])
    CS=backend.array((np.fft.fft(cs)/grid)[differences%grid])
    if not fff:C2=0*I;CS=0*I
    yf=admittance(2.,k,backend)
    S=interface(admittance(1.,k,backend),yf)
    for j in range(nz):
        z=(j+.5)*H/nz
        fill=1-np.arccos(2*z/H-1)/np.pi
        indicator=fill*np.sinc(differences*fill)*(1-2*(differences%2))
        C=backend.array(indicator)
        E=I+(metal-1)*C;eta=I+(1/metal-1)*C
        G=full_generator(E,eta,C2,CS,k,backend)
        section,_=local_s(G,H/nz,wave,yf,backend)
        S=star(S,section,backend)
    S=star(S,interface(yf,admittance(metal,k,backend)),backend)
    r=S[1].detach().cpu().numpy()[:,m]
    y=admittance(1.,k,backend).detach().cpu().numpy()
    efficiencies=abs(r)**2*np.where(abs(k.detach().cpu().numpy())<1,y.real,0)/y[m].real
    return float(efficiencies[m-1]),float(efficiencies.sum())

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--orders',default='16,32,48')
    parser.add_argument('--slices',type=int,default=200)
    parser.add_argument('--grid',type=int,default=4096)
    parser.add_argument('--output-dir',type=Path,default=STUDY/'validation/fff1d')
    args=parser.parse_args()
    records=[]
    values=[int(x) for x in args.orders.split(',')]
    for fff,m in [(False,values[-1]),*[(True,x) for x in values]]:
        start=time.perf_counter();r,total=calculate(m,nz=args.slices,fff=fff,grid=args.grid,device=args.device)
        record=dict(method='rcwa-fff' if fff else 'lamellar',order=m,slices=args.slices,grid=args.grid,device=args.device,
            R_minus_1=r,R_total=total,reference_R_minus_1=.8785309971543,
            relative_error=abs(r-.8785309971543)/.8785309971543,seconds=time.perf_counter()-start)
        records.append(record);print(json.dumps(record),flush=True)
        out=args.output_dir/'paper_benchmark.json';out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(dict(cases=records,reference='Mohamad et al. 2020 Fig. 4/5; midpoint sections'),indent=2)+'\n',encoding='utf-8')
