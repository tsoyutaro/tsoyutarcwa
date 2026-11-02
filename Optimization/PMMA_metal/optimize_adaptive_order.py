"""One trial per order; restart from the baseline at M+2 only after NonphysicalResult."""
from __future__ import annotations
import argparse,hashlib,json,math,os,signal,socket,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DEFAULT_STARTS={'Al':8,'Ti':8,'Cr':8,'Cu':12,'Au':16}
VERSION='pmma-metal-adaptive-order-v1'
OOM={'cuda_oom','cpu_oom'}

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)

def canonical(text):
    names={m.lower():m for m in ['Al','Ti','Cr','Cu','Au','Ag']}
    try:return names[text.strip().lower()]
    except KeyError:raise ValueError('Unknown material: '+text)

def starts_from(text):
    result={}
    for pair in text.split(','):
        name,value=pair.split(':');name=canonical(name);value=int(value)
        if value<1 or name in result:raise ValueError('Start orders must be unique positive integers.')
        result[name]=value
    return result

def source_hashes(case):
    files=[p for p in case.rglob('*') if p.is_file() and p.suffix in ('.py','.csv')
           and not {'results','__pycache__','validation_runs'}.intersection(p.relative_to(case).parts)]
    return {p.relative_to(case).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}

def classify(run,returncode):
    run=Path(run);workflow=read(run/'workflow.json') if (run/'workflow.json').exists() else {}
    outcome_path=run/'hexagonal/trials/trial_0000/outcome.json'
    outcome=read(outcome_path) if outcome_path.exists() else {}
    if returncode==0 and workflow.get('status')=='completed':
        report=read(run/'hexagonal/report.json');best=report.get('best',{})
        if not math.isfinite(best.get('objective',math.nan)):
            return dict(status='error',error='Completed workflow has no finite best objective.')
        return dict(status='completed',best_mean_absorptance=best['mean_absorptance'],completed_steps=best['completed_steps'])
    for candidate in [workflow,outcome,read(run/'preflight/result.json') if (run/'preflight/result.json').exists() else {}]:
        if candidate.get('status') in OOM|{'paused','interrupted'}:
            return dict(status=candidate['status'],error=candidate.get('error'))
    if returncode!=0 and (outcome.get('status')=='nonphysical' or
            outcome.get('status')=='error' and outcome.get('error_type')=='NonphysicalResult'):
        return dict(status='nonphysical',error=outcome.get('error'),error_type='NonphysicalResult',completed_steps=outcome.get('completed_steps'))
    return dict(status='error',error=outcome.get('error') or 'Optimization did not complete; inspect controller.log.',returncode=returncode)

def drive_orders(start,max_order,execute,history=None,on_record=None):
    """Completed/failed-physical attempts are immutable; interrupted attempts resume at the same M."""
    records=list(history or [])
    if len({r['order'] for r in records})!=len(records):raise ValueError('Duplicate order records.')
    by_order={r['order']:r for r in records}
    for order in range(start,max_order+1,2):
        old=by_order.get(order)
        if old and old['status']=='completed':return dict(status='completed',selected_order=order,attempts=records)
        if old and old['status']=='nonphysical':continue
        if old and old['status']=='error':return dict(status='error',selected_order=None,attempts=records)
        record={**execute(order),'order':order}
        if old:records[records.index(old)]=record
        else:records.append(record)
        by_order[order]=record
        if on_record:on_record(list(records))
        if record['status']=='completed':return dict(status='completed',selected_order=order,attempts=records)
        if record['status']!='nonphysical':return dict(status=record['status'],selected_order=None,attempts=records)
        print(f'Nonphysical at M={order}; '+(f'restart from baseline at M={order+2}.' if order+2<=max_order else f'maximum M={max_order} reached.'),flush=True)
    return dict(status='max_order_reached',selected_order=None,attempts=records)

def launch(command,log):
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',encoding='utf-8') as handle:
        handle.write('\nCOMMAND '+json.dumps(command,ensure_ascii=False)+'\n');handle.flush()
        process=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT)
        try:return process.wait()
        except BaseException:
            process.terminate()
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            raise

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--materials',default='Al,Ti,Cr,Cu,Au')
    p.add_argument('--start-orders',default='Al:8,Ti:8,Cr:8,Cu:12,Au:16')
    p.add_argument('--max-order',type=int,default=32)
    p.add_argument('--grid',type=int,default=512)
    p.add_argument('--output-root',type=Path,default=ROOT/'results/adaptive_trial1_normal_off_v1')
    p.add_argument('--checkpoint-layers',action='store_true')
    p.add_argument('--recover-lock',action='store_true')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--adam-steps',type=int,help='Optional override; production default is500.')
    p.add_argument('--wavelengths',help='Optional optimization-band override; default400:700:10.')
    return p

def main(argv=None):
    p=parser();args=p.parse_args(argv)
    try:
        materials=[canonical(m) for m in args.materials.split(',')];starts=starts_from(args.start_orders)
        if not materials or len(set(materials))!=len(materials):raise ValueError('Materials must be unique.')
        if args.max_order<1 or args.grid<32 or args.adam_steps is not None and args.adam_steps<1:raise ValueError('Invalid numerical limits.')
        if any(m not in starts or starts[m]>args.max_order for m in materials):raise ValueError('Each material needs a start order <=max-order.')
        cases={};templates={};hashes={}
        for material in materials:
            case=ROOT/material/f'pmma_{material.lower()}_motheye_P200_H500_Nz15'
            template=read(case/'config_hex_trial1_M8_Q512.json')
            if template['material']!=material:raise ValueError('Material differs from case directory.')
            cases[material]=case;templates[material]=template;hashes[material]=source_hashes(case)
        identity=dict(tool=VERSION,controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),materials=materials,
            start_orders={m:starts[m] for m in materials},max_order=args.max_order,grid=args.grid,device=args.device,
            checkpoint_layers=args.checkpoint_layers,adam_steps=args.adam_steps,wavelengths=args.wavelengths,
            templates=templates,source_sha256=hashes,normal_mode='off',nonphysical_action='restart-higher-order',order_step=2)
        output=args.output_root.resolve();manifest=output/'campaign_manifest.json'
        if manifest.exists() and read(manifest)!=identity:raise ValueError('Campaign settings/sources differ; use a new output directory.')
        state_path=output/'campaign_state.json'
        if args.prepare_only and state_path.exists():
            records=read(state_path).get('materials',{})
            if any(r.get('status')!='prepared' for r in records.values()):
                raise ValueError('Preparation must not replace an optimization history; use a separate output directory.')
    except (ValueError,OSError,KeyError) as error:p.error(str(error))
    output.mkdir(parents=True,exist_ok=True);lock=output/'controller.lock'
    if lock.exists() and args.recover_lock:
        (lock/'owner.json').unlink(missing_ok=True);lock.rmdir()
    try:lock.mkdir()
    except FileExistsError:p.error('Campaign lock exists. After the previous process stops, use --recover-lock.')
    write(lock/'owner.json',dict(pid=os.getpid(),host=socket.gethostname()))
    state=read(output/'campaign_state.json') if (output/'campaign_state.json').exists() else dict(materials={})
    def persist():write(output/'campaign_state.json',state)
    def interrupt(signum,frame):raise KeyboardInterrupt()
    prior_sigterm=signal.signal(signal.SIGTERM,interrupt)
    try:
        write(manifest,identity)
        state['status']='running';persist()
        for material in materials:
            case=cases[material];old=state['materials'].get(material,{});history=old.get('attempts',[])
            def execute(order):
                folder=output/material/f'M{order}';folder.mkdir(parents=True,exist_ok=True)
                config={**templates[material], 'optimization_order':order,'optimization_grid':args.grid,'trials':1,
                        'normal_mode':'off','nonphysical_action':'restart-higher-order'}
                if args.adam_steps is not None:config['adam_steps']=args.adam_steps
                if args.wavelengths is not None:config['optimization_wavelengths_nm']=args.wavelengths
                if args.grid<max(32,4*order+4):return dict(status='error',error='Fixed grid undersamples this order; increase --grid in a new campaign.')
                config_path=folder/'config.json'
                if config_path.exists() and read(config_path)!=config:raise ValueError('Saved order config differs.')
                write(config_path,config);run=folder/'run'
                command=[sys.executable,str(case/'run.py'),'--config',str(config_path),'--output-root',str(run),
                    '--device',args.device,'--mode','optimize','--lattices','hexagonal','--skip-audit','--timeout-hours','0']
                if args.checkpoint_layers:command+=['--checkpoint-layers']
                if args.recover_lock:command+=['--recover-lock']
                if args.prepare_only:command+=['--prepare-only']
                print(f'START {material}: M={order}, grid={args.grid}, trials=1, normal_mode=off; {run}',flush=True)
                started=time.perf_counter();code=launch(command,folder/'controller.log')
                record=(dict(status='prepared') if code==0 else dict(status='error',error='Preparation failed.')) if args.prepare_only else classify(run,code)
                record.update(run_root=str(run),config=str(config_path),controller_returncode=code,wall_seconds=time.perf_counter()-started)
                print(f'RESULT {material}: M={order}, status={record["status"]}',flush=True)
                return record
            if args.prepare_only:
                record=execute(starts[material]);state['materials'][material]=dict(status=record['status'],attempts=[dict(order=starts[material],**record)])
                persist()
                if record['status']=='error':state['status']='error';persist();return 1
                continue
            def on_record(records):
                state['materials'][material]=dict(status='running',attempts=records);persist()
            result=drive_orders(starts[material],args.max_order,execute,history,on_record)
            state['materials'][material]=result;persist()
            if result['status'] in ('error','paused','interrupted'):
                state['status']=result['status'];persist();return 130 if result['status']=='interrupted' else 1
            # OOM/max-order stop this metal; independent remaining metals still run.
        state['status']='prepared' if args.prepare_only else 'completed' if all(state['materials'][m]['status']=='completed' for m in materials) else 'partial'
        persist();print('CAMPAIGN',state['status'],output,flush=True)
        return 0 if state['status'] in ('prepared','completed') else 1
    except KeyboardInterrupt:
        state['status']='interrupted';persist();return 130
    except Exception as error:
        state['status']='error';state['controller_error']=str(error);persist();raise
    finally:
        signal.signal(signal.SIGTERM,prior_sigterm)
        (lock/'owner.json').unlink(missing_ok=True);lock.rmdir()

if __name__=='__main__':raise SystemExit(main())
