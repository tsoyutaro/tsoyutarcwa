"""Run isolated validation processes for the five PMMA-metal cases."""
from pathlib import Path
import argparse,json,subprocess,sys

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cpu')
    p.add_argument('--materials',default='Al,Ti,Cu,Au,Ag')
    p.add_argument('--suite',choices=['small','initial'],default='small')
    p.add_argument('--order',type=int)
    p.add_argument('--grid',type=int)
    p.add_argument('--with-gradient',action='store_true')
    p.add_argument('--output-root',type=Path,default=ROOT/'validation_reports')
    a=p.parse_args()
    reports=[]
    for material in a.materials.split(','):
        if material not in ('Al','Ti','Cu','Au','Ag'):p.error('Unknown material: '+material)
        case=ROOT/material/f'pmma_{material.lower()}_motheye_P200_H500_Nz15'
        folder=a.output_root/a.suite;folder.mkdir(parents=True,exist_ok=True)
        output=folder/(material+'.json')
        command=[sys.executable,str(case/'validate_case.py'),'--device',a.device,'--suite',a.suite,'--output',str(output.resolve())]
        if a.order is not None:command+=['--order',str(a.order)]
        if a.grid is not None:command+=['--grid',str(a.grid)]
        if a.with_gradient:command+=['--with-gradient']
        print('START '+material+' '+a.suite,flush=True)
        proc=subprocess.run(command,text=True,capture_output=True,encoding='utf-8')
        (folder/(material+'.log')).write_text(proc.stdout+'\n'+proc.stderr,encoding='utf-8')
        report=json.loads(output.read_text(encoding='utf-8')) if output.exists() else dict(status='worker_failed',error=proc.stderr)
        item=dict(material=material,status=report['status'],report=str(output.resolve()),returncode=proc.returncode)
        if 'mean_absorptance' in report:item['mean_absorptance']=report['mean_absorptance']
        reports.append(item);print(json.dumps(item),flush=True)
        summary=dict(status='passed' if all(r['status']=='passed' for r in reports) else 'failed',
                     suite=a.suite,device=a.device,reports=reports)
        (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    return 0 if summary['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
