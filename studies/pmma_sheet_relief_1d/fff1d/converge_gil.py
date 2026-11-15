"""Independent bounded RCWA-FFF Fourier-order campaigns for every GIL setting."""
from pathlib import Path
import argparse
import json
import math
import sys

STUDY=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(STUDY))
from model import save_csv,save_json
from automation import converge_until as controller
from fff1d.run_fff import parser as fff_parser,source_signature


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gil-values',default='0,0.5,1,2',help='horizontal total GIL widths, nm')
    p.add_argument('--gil-interface-values',default='both',help='inner, outer, both; comma list or all')
    p.add_argument('--start',type=int,default=16)
    p.add_argument('--step',type=int,default=16)
    group=p.add_mutually_exclusive_group()
    group.add_argument('--max-order',type=int,help='same ceiling for all GIL widths; default 128')
    group.add_argument('--max-orders',help='one ceiling for each value in --gil-values')
    p.add_argument('--tolerance',type=float,default=.001)
    p.add_argument('--coating',choices=['gold','uncoated','both'],default='gold')
    p.add_argument('--geometry-mode',choices=['scaled','legacy'],default='scaled')
    p.add_argument('--output-dir',type=Path,default=STUDY/'results/fff1d/gil_order')
    args,extra=p.parse_known_args(argv)
    if any(v=='--gil-nm' or v.startswith('--gil-nm=') or v=='--gil-interfaces' or v.startswith('--gil-interfaces=') for v in extra):
        p.error('Use --gil-values and --gil-interface-values for this sweep.')
    try:
        widths=[float(x) for x in args.gil_values.split(',')]
        limits=[int(x) for x in args.max_orders.split(',')] if args.max_orders else [128 if args.max_order is None else args.max_order]*len(widths)
    except ValueError:p.error('Invalid GIL widths or maximum orders.')
    scopes=['inner','outer','both'] if args.gil_interface_values=='all' else args.gil_interface_values.split(',')
    if not widths or len(widths)!=len(set(widths)) or any(not math.isfinite(x) or x<0 for x in widths):
        p.error('Use distinct, finite GIL widths >=0.')
    if not scopes or len(scopes)!=len(set(scopes)) or any(s not in ('inner','outer','both') for s in scopes):
        p.error('Interfaces must be inner,outer,both, or all.')
    if len(limits)!=len(widths) or args.start<1 or args.step<1 or any(x<args.start for x in limits):
        p.error('Provide one max order >= start per GIL, with positive start and step.')
    if not math.isfinite(args.tolerance) or args.tolerance<=0:p.error('Positive finite tolerance required.')
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    records=[]
    stopped=False
    try:
        for width,ceiling in zip(widths,limits):
            # Zero GIL is identical for all interface selections: run it once.
            for scope in (['both'] if width==0 else scopes):
                label=f'gil_{width!r}nm_'+('sharp' if width==0 else scope)
                dest=output/label
                options=argparse.Namespace(axis='order',start=args.start,step=args.step,factor=None,
                    max_value=ceiling,tolerance=args.tolerance,coating=args.coating,geometry_mode=args.geometry_mode)
                child_extra=[*extra,'--gil-nm',str(width),'--gil-interfaces',scope]
                try:
                    ctx=controller.context(options,child_extra,parser_override=fff_parser,signature_override=source_signature)
                    ctx.update(runner_path=STUDY/'fff1d/run_fff.py',passivity_column='passivity_valid',include_max=True)
                    dest.mkdir(parents=True,exist_ok=True)
                    print(f'GIL {width:g} nm / {scope}; max M={ceiling}',flush=True)
                    with controller.campaign_lock(dest):code=controller.automatic(options,child_extra,ctx,dest)
                    status=controller.load(dest/'convergence.json')
                    record=dict(gil_nm=width,gil_interfaces='none' if width==0 else scope,max_order=ceiling,
                        last_completed_order=status.get('last_completed_value'),
                        converged=status.get('converged',False),status=status['status'],exit_code=code,
                        output_dir=str(dest),stop_reason=status.get('stop_reason',''))
                except (ValueError,RuntimeError,OSError) as exc:
                    record=dict(gil_nm=width,gil_interfaces=scope,max_order=ceiling,last_completed_order=None,
                        converged=False,status='configuration_error',exit_code=4,output_dir=str(dest),
                        stop_reason=f'{type(exc).__name__}: {exc}')
                records.append(record)
                save_csv(output/'gil_summary.csv',records)
                save_json(output/'gil_summary.json',dict(cases=records,
                    all_converged=all(r['converged'] for r in records),requested_gil_nm=widths,
                    requested_interfaces=scopes,max_orders=limits,
                    note='Each GIL changes the material profile; convergence is separate for each GIL and interface selection.'))
                if record['exit_code']==130:stopped=True;break
            if stopped:break
    except KeyboardInterrupt:
        return 130
    if stopped:return 130
    if all(r['converged'] for r in records):return 0
    if any(r['exit_code']==4 for r in records):return 4
    if any(r['exit_code']==3 for r in records):return 3
    return 2


if __name__=='__main__':raise SystemExit(main())
