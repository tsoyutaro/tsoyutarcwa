"""Compare frozen old/new shapes with the same OFF solver, without modifying training records."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
if os.environ.get('MOTHEYE_DEPENDENCY_PATH'):sys.path.append(os.environ['MOTHEYE_DEPENDENCY_PATH'])
from optimizer import common
from optimizer.driver import launch
from runtime_policy import execution_sources,reconcile_manifest


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=HERE/'config_hex_trial1_M8_Q1024_off.json')
    p.add_argument('--run-root',type=Path,default=Path('results/hex_trial1_M8_Q1024_off_v1'))
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--order',type=int,default=30)
    p.add_argument('--grid',type=int,default=1024)
    p.add_argument('--wavelengths',default='400,550,700')
    a=p.parse_args();c=common.load_config(a.config);waves=common.wavelengths(a.wavelengths)
    if a.order<1 or a.grid<max(32,4*a.order+4):p.error('Invalid order/grid.')
    run=(a.run_root if a.run_root.is_absolute() else HERE/a.run_root).resolve()
    old=read(HERE/'initial_profile.json')['logits'];new=read(run/'hexagonal/report.json').get('best',{}).get('logits')
    if new is None:raise ValueError('No completed OFF optimization best shape.')
    if old!=c['initial_logits']:raise ValueError('Frozen old profile differs from configured initialization.')
    lock=run/'controller.lock'
    try:lock.mkdir()
    except FileExistsError:p.exit(2,'A controller lock exists; verify the previous job before recovering it.\n')
    common.write_json(lock/'owner.json',dict(tool='off-old-new-comparison-v1',pid=os.getpid(),host=socket.gethostname()))
    try:
        protected=[run/'manifest.json',run/'resolved_config.json',run/'workflow.json',run/'hexagonal/report.json']+list((run/'hexagonal/trials').glob('*/adam_checkpoint.pt'))+list((run/'hexagonal').glob('optuna_journal*'))
        before={str(path):sha(path) for path in protected if path.is_file()}
        sources=execution_sources(common.sources());manifest=read(run/'manifest.json')
        checked,_=reconcile_manifest(manifest,dict(version=1,config=c,source_sha256=sources,requested_device=a.device))
        base=run/'hexagonal/postprocess/off_old_new_comparison'
        prep=launch(base/'preparation',dict(mode='preflight',config=c,device=a.device,runtime_environment=manifest['runtime_environment'],signature=checked['signature']))
        if prep['status']!='ready':raise RuntimeError('Comparison preflight failed: '+str(prep))
        runtime=prep['runtime_environment']
        ident=common.signature_for(dict(order=a.order,grid=a.grid,waves=waves,old=old,new=new,signature=checked['signature'],runtime=runtime))[:16]
        folder=base/ident;results={}
        for label,logits in [('old',old),('new',new)]:
            print(f'EVALUATE {label} geometry with OFF M={a.order} Q={a.grid}',flush=True)
            request=dict(mode='analyze_spectrum',config={**c,'rcwa_acceleration':'auto'},device=a.device,runtime_environment=runtime,
                         signature=checked['signature'],lattice='hexagonal',logits=logits,waves=waves,order=a.order,grid=a.grid)
            result=launch(folder/label,request)
            common.write_json(folder/(label+'_spectrum.json'),result)
            if result['status']!='completed':
                common.write_json(folder/'report.json',dict(status=result['status'],failure=result,label=label))
                raise RuntimeError(f'{label} profile failed: '+str(result.get('error',result['status'])))
            results[label]=result
        if any(sha(k)!=v for k,v in before.items()) or execution_sources(common.sources())!=sources:
            raise RuntimeError('A protected optimization/source file changed.')
        rows=[]
        for x,y in zip(results['old']['rows'],results['new']['rows']):
            if x['wavelength_nm']!=y['wavelength_nm']:raise ValueError('Wavelength grids differ.')
            rows.append(dict(wavelength_nm=x['wavelength_nm'],old_reflectance=x['reflectance'],new_reflectance=y['reflectance'],
                reflectance_change=y['reflectance']-x['reflectance'],old_absorptance_total=x['absorptance_total'],new_absorptance_total=y['absorptance_total'],
                old_power_into_substrate=x['power_into_substrate'],new_power_into_substrate=y['power_into_substrate'],
                old_motheye_absorptance=x['motheye_absorptance'],new_motheye_absorptance=y['motheye_absorptance']))
        old_mean=results['old']['objective'];new_mean=results['new']['objective']
        report=dict(status='completed',normal_field_mode='off',normal_field_definition=c['normal_field_definition'],
            order=a.order,grid=a.grid,wavelengths_nm=waves,objective_weighting=c['objective_weighting'],runtime_environment=runtime,
            old_mean_R=old_mean,new_mean_R=new_mean,mean_R_change=new_mean-old_mean,sampled_mean_R_improved=new_mean<old_mean,
            production_files_unchanged=True,source_sha256=sources,old_logits=old,new_logits=new,rows=rows,
            scope='Both geometries reevaluated with the same OFF model. Finite resolution; not a converged/global-optimum claim.')
        common.write_json(folder/'report.json',report);common.write_csv(folder/'comparison.csv',rows)
        try:
            import matplotlib.pyplot as plt
            fig,ax=plt.subplots(figsize=(7,4))
            for label,key in [('old ON-optimized geometry','old_reflectance'),('new OFF-optimized geometry','new_reflectance')]:ax.plot(waves,[100*r[key] for r in rows],marker='o',label=label)
            ax.set(xlabel='Wavelength (nm)',ylabel='Reflectance (%)',title=f'Both evaluated with OFF: M={a.order}, Q={a.grid}')
            ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(folder/'comparison.png',dpi=180);plt.close(fig)
        except ImportError:pass
        common.write_json(run/'hexagonal/latest_off_comparison.json',dict(report=str(folder/'report.json'),status='completed'))
        print(json.dumps({k:report[k] for k in ['old_mean_R','new_mean_R','mean_R_change','sampled_mean_R_improved']},indent=2),flush=True)
        print('Saved '+str(folder/'report.json'),flush=True)
        return 0
    finally:
        (lock/'owner.json').unlink(missing_ok=True);lock.rmdir()

if __name__=='__main__':raise SystemExit(main())
