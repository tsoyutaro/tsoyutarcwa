"""Validate deployed common RCWA matrix/spectral paths against its reference path."""
from __future__ import annotations
import argparse,gc,hashlib,importlib,importlib.util,json,sys,traceback
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))


def model_for(material):
    if material in ('n2.5','Cr'):
        path=ROOT/'Optimization'/('dielectric_motheye_n25' if material=='n2.5' else 'cr_motheye')
        prefix='Optimization.'+path.name
    else:
        family=material if material in ('PMMA','Al','Cu','Au','Ag') else material
        stem=('pmma_motheye' if material=='PMMA' else material.lower()+'_motheye') if material not in ('n1.5','n2.0') else 'dielectric_motheye_'+material.replace('.','')
        path=ROOT/'Optimization'/family/(stem+'_P200_H500_Nz15')
        prefix='_validation_'+material.replace('.','')
        spec=importlib.util.spec_from_file_location(prefix,path/'optimizer/__init__.py',submodule_search_locations=[str(path/'optimizer')])
        module=importlib.util.module_from_spec(spec);sys.modules[prefix]=module;spec.loader.exec_module(module)
    return path,[importlib.import_module(prefix+'.'+n) for n in ['common','geometry','physics']]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda:0');p.add_argument('--production',action='store_true')
    p.add_argument('--materials',nargs='+',choices=['n1.5','n2.0','n2.5','PMMA','Cr','Al','Cu','Au','Ag'],default=['n1.5','n2.0','n2.5','PMMA','Cr','Al','Cu','Au','Ag'])
    p.add_argument('--lattices',nargs='+',choices=['square','hexagonal'],default=['square','hexagonal'])
    p.add_argument('--order',type=int);p.add_argument('--grid',type=int);p.add_argument('--layers',type=int)
    p.add_argument('--wavelengths');p.add_argument('--check-fd',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'Optimization/results/acceleration_validation_v3.json')
    a=p.parse_args();device=torch.device(a.device)
    if device.type not in ('cuda','cpu') or device.type=='cuda' and not torch.cuda.is_available():p.error('Requested device unavailable; no CPU fallback.')
    if device.type=='cuda':torch.cuda.set_device(device)
    torch.set_num_threads(1)
    m=a.order or (8 if a.production else 2);q=a.grid or (512 if a.production else 48);layers=a.layers or (15 if a.production else 5)
    if m<1 or q<max(32,4*m+4) or layers<2:p.error('Invalid order/grid/layers.')
    from Optimization.dielectric_motheye_n25.common import wavelengths
    waves=wavelengths(a.wavelengths or ('400:700:10' if a.production else '400,550,700'))
    expected=sum(1 for material in a.materials for lattice in a.lattices if material!="Cr" or lattice=="hexagonal")
    if not expected:p.error("No supported material/lattice pairs requested.")
    results=[]
    for material in a.materials:
        path,(common,geometry,physics)=model_for(material)
        c=common.load_config(path/'config.json');c.update(layers=layers,rcwa_acceleration='auto',checkpoint_layers=False)
        for lattice in a.lattices:
            if lattice not in c['lattices']:continue
            try:
                x=torch.tensor(geometry.baseline_logits(c),dtype=torch.float64,device=device,requires_grad=True)
                print(f'{material}/{lattice}: reference M{m} Q{q} L{layers}',flush=True)
                ref=physics.objective(x,{**c,'rcwa_acceleration':'reference'},lattice,waves,m,q,device,gradient=True)
                comparison=[]
                for backend,cp in [('matrix',False),('auto',False),('auto',True)]:
                    print(f'  {backend}, checkpoint={cp}',flush=True)
                    value=physics.objective(x,{**c,'rcwa_acceleration':backend,'checkpoint_layers':cp},lattice,waves,m,q,device,gradient=True)
                    loss_error=abs(ref['objective']-value['objective'])
                    gradient_error=float((ref['gradient']-value['gradient']).abs().max().detach().cpu())
                    key='power_into_substrate' if c.get('substrate_mode')=='semi-infinite' else 'transmittance'
                    power_error=max(abs(r[k]-s[k]) for r,s in zip(ref['rows'],value['rows']) for k in ('reflectance',key))
                    passed=loss_error<1e-8 and power_error<1e-8 and torch.allclose(ref['gradient'],value['gradient'],atol=1e-7,rtol=2e-5)
                    assert passed,(loss_error,power_error,gradient_error)
                    comparison.append(dict(backend=backend,checkpoint=cp,loss_abs_error=loss_error,power_max_abs_error=power_error,logits_gradient_max_abs_error=gradient_error))
                fd=None
                if a.check_fd:
                    d=torch.linspace(-.7,.9,layers+1,dtype=torch.float64,device=device);d-=d.mean();d/=torch.linalg.vector_norm(d)
                    h=5e-5;cc={**c,'rcwa_acceleration':'reference'}
                    plus=physics.objective(x.detach()+h*d,cc,lattice,[550.],m,q,device,gradient=False)['objective']
                    minus=physics.objective(x.detach()-h*d,cc,lattice,[550.],m,q,device,gradient=False)['objective']
                    g=physics.objective(x,c,lattice,[550.],m,q,device,gradient=True)['gradient']
                    observed=(plus-minus)/(2*h);ad=float((g@d).detach().cpu())
                    assert abs(observed-ad)<1e-6+3e-4*abs(observed),(observed,ad)
                    fd=dict(wavelength_nm=550.,passed=True,finite_difference=observed,autodiff=ad,abs_error=abs(observed-ad))
                result=dict(status='passed',material=material,lattice=lattice,comparisons=comparison,finite_difference=fd)
            except Exception as error:
                result=dict(status='failed',material=material,lattice=lattice,error=str(error),traceback=traceback.format_exc())
            results.append(result);print(result['status'].upper(),material,lattice,flush=True)
            a.output.parent.mkdir(parents=True,exist_ok=True)
            physics._spectral_evaluator=None
            gc.collect()
            report=dict(status='running',expected_case_count=expected,completed_case_count=len(results),
                cuda_execution_performed=device.type=='cuda',device=str(device),torch=str(torch.__version__),
                gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,order=m,grid=q,layers=layers,
                wavelengths_nm=waves,cases=results,source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in sorted((ROOT/'rcwa_ext').rglob('*.py'))})
            a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    report['status']='passed' if len(results)==expected and all(r['status']=='passed' for r in results) else 'failed'
    a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],cuda_execution_performed=report['cuda_execution_performed'],output=str(a.output),cases=len(results)),indent=2))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
