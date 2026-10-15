"""Memory-safe NV spectra and R/A/T averages for eight metals and PMMA coatings.

Existing solver workers are reused unchanged, one subprocess per wavelength.
Au/Ag are excluded by default. A failed wavelength is recorded and other
wavelengths/materials/structures continue. Only complete bands get averages.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
import run_transition_metals_nv as transition
import run_cu_ag_al_motheye_order_nv as cu_al
import run_pmma_cr_motheye_order_nv as chromium

nv = transition.nv
BACKENDS = {"Cr":chromium, **{m:transition for m in ("Fe","Ni","Ti","V","W")}, "Al":cu_al, "Cu":cu_al}
DEFAULT_ORDERS = "Cr:24,Fe:24,Ni:24,Ti:24,V:24,Al:36,Cu:36,W:36"
POWERS = dict(R="reflectance", A="absorptance", T="transmittance",
              moth_eye_A="motheye_absorptance", substrate_A="substrate_absorptance",
              power_into_substrate="power_into_substrate")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows, columns):
    stream=io.StringIO(newline="")
    writer=csv.DictWriter(stream,fieldnames=columns,extrasaction="ignore",lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    nv.atomic_text(path,stream.getvalue())


def parse_wavelengths(text):
    if ":" in text:
        start,stop,step=[float(v) for v in text.split(":")]
        if not all(math.isfinite(v) for v in (start,stop,step)) or start<=0 or stop<=start or step<=0:
            raise ValueError("Use start:stop:step with 0<start<stop and step>0.")
        intervals=round((stop-start)/step)
        if intervals<1 or not math.isclose(start+intervals*step,stop,rel_tol=1e-12,abs_tol=1e-9):
            raise ValueError("The wavelength step must divide the requested band exactly.")
        waves=[round(start+i*step,12) for i in range(intervals+1)]
    else:
        waves=[float(v.strip()) for v in text.split(",")]
    if len(waves)<2 or len(set(waves))!=len(waves) or any(not math.isfinite(w) or w<=0 for w in waves):
        raise ValueError("Use at least two distinct, finite positive wavelengths.")
    return sorted(waves)


def parse_orders(text):
    aliases={m.lower():m for m in BACKENDS}
    values={}
    for part in text.split(","):
        name,value=part.strip().split(":")
        material=aliases[name.lower()]
        order=int(value)
        if order<1 or material in values:
            raise ValueError("Use positive orders and specify each material only once.")
        values[material]=order
    return values


def build_plan(args):
    waves=parse_wavelengths(args.wavelengths)
    orders=parse_orders(args.orders)
    if args.slices<1 or args.grid<max(32,4*max(orders.values())+4) or args.cpu_threads<1:
        raise ValueError("Use positive slices/threads and grid >= max(32,4*max_order+4).")
    if args.device not in ("cpu","cuda") and not (args.device.startswith("cuda:") and args.device[5:].isdigit()):
        raise ValueError("Use cpu, cuda or cuda:<index>.")
    structures=["metal","pmma"] if args.structures=="both" else [args.structures]
    jobs=[]
    for material,order in orders.items():
        backend=BACKENDS[material]
        for structure in structures:
            options=backend.build_parser().parse_args([])
            options.device,options.cpu_threads=args.device,args.cpu_threads
            options.slices,options.grid=args.slices,args.grid
            options.start_order,options.max_order,options.step=order,order,2
            identity=backend.build_identity(options,material,structure,waves)
            jobs.append(dict(key=f"{material}|{structure}",material=material,structure=structure,order=order,
                             relative_output=f"{structure}/{material.lower()}",
                             plan=dict(identity=identity,max_order=order)))
    return dict(version=1,method="NV",orders=orders,slices=args.slices,grid=args.grid,
                wavelengths_nm=waves,structures=structures,jobs=jobs,driver_sha256=nv.digest(Path(__file__)),
                means=dict(primary="trapezoidal wavelength integral divided by band width",
                           secondary="arithmetic mean of all requested samples",weights="uniform wavelength, no solar weighting",
                           incomplete="no mean values until every requested wavelength succeeds"))


def initial_state(job):
    return dict(signature=nv.identity_signature(job["plan"]["identity"]),plan=job["plan"],
                status="pending",cases={},failures={},environment=None)


def load_state(job,output,waves):
    path=output/job["relative_output"]/"checkpoint.json"
    state=read_json(path) if path.exists() else initial_state(job)
    identity=job["plan"]["identity"]
    if state["signature"]!=nv.identity_signature(identity):
        raise ValueError(f"Saved settings/source differ: {path}; use another output directory.")
    for key,row in state["cases"].items():
        BACKENDS[job["material"]].validate_case(row,identity)
        if row["order"]!=job["order"] or row["wavelength_nm"] not in waves or key!=nv.case_key(row["order"],row["wavelength_nm"]):
            raise ValueError(f"Saved case does not match the spectrum: {path}")
    return state


def averages(job,cases,waves):
    by_wave={float(row["wavelength_nm"]):row for row in cases.values()}
    complete=len(by_wave)==len(waves) and set(by_wave)==set(waves)
    row=dict(material=job["material"],structure=job["structure"],method="NV",order=job["order"],
             slices=job["plan"]["identity"]["slices"],grid=job["plan"]["identity"]["grid"],
             wavelength_min_nm=waves[0],wavelength_max_nm=waves[-1],expected_points=len(waves),
             completed_points=len(by_wave),complete=complete)
    for short,field in POWERS.items():
        band=sample=None
        if complete and all(by_wave[w].get(field) is not None for w in waves):
            values=[float(by_wave[w][field]) for w in waves]
            if any(not math.isfinite(v) for v in values):
                raise ValueError("Nonfinite power cannot be averaged.")
            sample=math.fsum(values)/len(values)
            integral=math.fsum((values[i]+values[i+1])*.5*(waves[i+1]-waves[i]) for i in range(len(waves)-1))
            band=integral/(waves[-1]-waves[0])
        row[f"mean_{short}_band"]=band
        row[f"mean_{short}_band_percent"]=None if band is None else 100*band
        row[f"mean_{short}_sample"]=sample
        row[f"mean_{short}_sample_percent"]=None if sample is None else 100*sample
    if complete:
        for name in ("band","sample"):
            if abs(math.fsum(row[f"mean_{field}_{name}"] for field in ("R","A","T"))-1)>1e-8:
                raise ValueError("Mean R+A+T differs from one.")
    return row


def save_job(job,state,output,waves):
    path=output/job["relative_output"]
    nv.write_json(path/"checkpoint.json",state)
    rows=sorted(state["cases"].values(),key=lambda r:r["wavelength_nm"])
    write_csv(path/"spectrum.csv",rows,transition.COLUMNS)
    report=averages(job,state["cases"],waves)
    report.update(status=state["status"],failed_points=len(state["failures"]),failure_details=list(state["failures"].values()))
    nv.write_json(path/"report.json",report)
    return report


def export_batch(plan,states,output,status):
    means=[]
    all_rows=[]
    failures=[]
    for job in plan["jobs"]:
        state=states[job["key"]]
        means.append(averages(job,state["cases"],plan["wavelengths_nm"]))
        all_rows.extend(sorted(state["cases"].values(),key=lambda r:r["wavelength_nm"]))
        for failure in state["failures"].values():
            failures.append(dict(material=job["material"],structure=job["structure"],order=job["order"],
                                 slices=plan["slices"],grid=plan["grid"],wavelength_nm=failure["wavelength_nm"],
                                 status=failure["status"],error=failure.get("error","")))
    write_csv(output/"averages.csv",means,list(means[0]))
    write_csv(output/"all_spectra.csv",all_rows,transition.COLUMNS)
    write_csv(output/"failed_cases.csv",failures,("material","structure","order","slices","grid","wavelength_nm","status","error"))
    summary=dict(status=status,method="NV",orders=plan["orders"],slices=plan["slices"],grid=plan["grid"],
                 wavelengths_nm=plan["wavelengths_nm"],expected_cases=len(plan["jobs"])*len(plan["wavelengths_nm"]),
                 completed_cases=len(all_rows),failed_cases=len(failures),complete_spectra=sum(r["complete"] for r in means),
                 requested_spectra=len(means),mean_definitions=plan["means"],spectra=means)
    nv.write_json(output/"batch_summary.json",summary)
    return summary


def execute(plan,output):
    output=Path(output).resolve()
    signature=nv.identity_signature(plan)
    batch_path=output/"batch_plan.json"
    if batch_path.exists() and read_json(batch_path)["signature"]!=signature:
        raise ValueError("Saved batch settings/source changed; use a new --output-root.")
    # Validate all cached results before modifying the batch or launching a worker.
    states={job["key"]:load_state(job,output,plan["wavelengths_nm"]) for job in plan["jobs"]}
    nv.write_json(batch_path,dict(signature=signature,plan=plan))
    export_batch(plan,states,output,"running")
    try:
        for job in plan["jobs"]:
            identity=job["plan"]["identity"]
            backend=BACKENDS[job["material"]]
            state=states[job["key"]]
            path=output/job["relative_output"]
            nv.write_json(path/"plan.json",job["plan"])
            if identity["structure"]=="pmma":
                from dataclasses import asdict
                profile=[asdict(layer) for layer in backend.CoatedGeometry(**identity["geometry"]).profile(plan["slices"])]
                write_csv(path/"profile_layers.csv",profile,list(profile[0]))
            state["status"]="running"
            save_job(job,state,output,plan["wavelengths_nm"])
            print(f"\n=== {job['key']} NV: M={job['order']}, Nz={plan['slices']}, Q={plan['grid']}; resume {len(state['cases'])}/{len(plan['wavelengths_nm'])} wavelengths ===",flush=True)
            for wave in plan["wavelengths_nm"]:
                key=nv.case_key(job["order"],wave)
                if key in state["cases"]:
                    continue
                print(f"solve: {job['material']}/{job['structure']}, lambda={wave:g} nm, M={job['order']}",flush=True)
                result_path=path/"worker_results"/f"wl{wave:g}_{uuid.uuid4().hex}.json"
                try:
                    response=backend.launch_worker(path/"plan.json",result_path,job["order"],wave)
                    if response["status"]=="ok":
                        if response["signature"]!=state["signature"]:
                            raise ValueError("Worker signature differs from its spectrum plan.")
                        case=response["case"]
                        backend.validate_case(case,identity)
                        if case["order"]!=job["order"] or case["wavelength_nm"]!=wave:
                            raise ValueError("Worker returned another wavelength or order.")
                        if state["environment"] not in (None,response["environment"]):
                            raise ValueError("Worker environment changed; use a separate output directory.")
                        state["cases"][key]=case
                        state["environment"]=response["environment"]
                        state["failures"].pop(key,None)
                    else:
                        response=dict(response,wavelength_nm=wave)
                        state["failures"][key]=response
                        print(f"CONTINUE after {response['status']} at {wave:g} nm.",flush=True)
                except Exception as error:
                    state["failures"][key]=dict(status="calculation_failed",wavelength_nm=wave,error=f"{type(error).__name__}: {error}")
                    print(f"CONTINUE after calculation_failed at {wave:g} nm: {error}",flush=True)
                save_job(job,state,output,plan["wavelengths_nm"])
                export_batch(plan,states,output,"running")
            state["status"]="complete" if len(state["cases"])==len(plan["wavelengths_nm"]) else "incomplete"
            report=save_job(job,state,output,plan["wavelengths_nm"])
            if report["complete"]:
                print(f"Band means: R={report['mean_R_band_percent']:.6f}%, A={report['mean_A_band_percent']:.6f}%, T={report['mean_T_band_percent']:.6f}%",flush=True)
            else:
                print(f"Incomplete: {report['completed_points']}/{report['expected_points']}; full-band means not published.",flush=True)
    except KeyboardInterrupt:
        for job in plan["jobs"]:
            state=states[job["key"]]
            if state["status"]=="running":
                state["status"]="interrupted"
                save_job(job,state,output,plan["wavelengths_nm"])
        export_batch(plan,states,output,"interrupted")
        raise
    complete=all(len(s["cases"])==len(plan["wavelengths_nm"]) for s in states.values())
    summary=export_batch(plan,states,output,"complete" if complete else "completed_with_failures")
    print(f"\nBatch status: {summary['status']}; {summary['completed_cases']}/{summary['expected_cases']} cases; {summary['complete_spectra']}/{summary['requested_spectra']} complete spectra.\nResults: {output}",flush=True)
    return states,summary


def plot_results(plan,states,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fields=(("R","reflectance","#0072B2"),("A","absorptance","#D55E00"),("T","transmittance","#009E73"))
    for material,order in plan["orders"].items():
        fig,axes=plt.subplots(len(plan["structures"]),3,figsize=(13.5,4.1*len(plan["structures"])),squeeze=False,layout="constrained")
        fig.suptitle(f"{material} | NV spectrum | M={order}, Nz={plan['slices']}, Q={plan['grid']}\nP=200 nm, h=500 nm; normal x polarization",fontsize=15)
        for i,structure in enumerate(plan["structures"]):
            job=next(j for j in plan["jobs"] if j["key"]==f"{material}|{structure}")
            state=states[job["key"]]
            mean=averages(job,state["cases"],plan["wavelengths_nm"])
            by_wave={r["wavelength_nm"]:r for r in state["cases"].values()}
            for j,(short,field,color) in enumerate(fields):
                ax=axes[i,j]
                values=[100*by_wave[w][field] if w in by_wave else math.nan for w in plan["wavelengths_nm"]]
                ax.plot(plan["wavelengths_nm"],values,color=color,lw=1.8,marker="o",ms=2.6)
                title=f"{'Metal only' if structure=='metal' else 'PMMA + '+material+' (30 nm)'} | {short}"
                title+=(f"\nBand mean {mean[f'mean_{short}_band_percent']:.4f}%; sample mean {mean[f'mean_{short}_sample_percent']:.4f}%"
                        if mean["complete"] else f"\nINCOMPLETE: {mean['completed_points']}/{mean['expected_points']} points; means unavailable")
                ax.set_title(title,fontsize=10.3)
                ax.set_xlabel("Wavelength (nm)")
                ax.set_ylabel(f"{short} (%)")
                ax.set_xlim(plan["wavelengths_nm"][0],plan["wavelengths_nm"][-1])
                ax.ticklabel_format(axis="y",style="plain",useOffset=False)
                ax.grid(alpha=.25)
                if any(math.isfinite(v) for v in values):
                    lo,hi=ax.get_ylim()
                    ax.set_ylim(max(0.,lo),min(100.,hi))
        path=output/f"{material}_spectrum_NV"
        fig.savefig(path.with_suffix(".png"),dpi=170)
        fig.savefig(path.with_suffix(".svg"))
        plt.close(fig)


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--orders",default=DEFAULT_ORDERS,help="Material:fixed-order pairs; only requested materials are run")
    parser.add_argument("--structures",choices=("both","metal","pmma"),default="both")
    parser.add_argument("--wavelengths",default="400:700:5",help="Inclusive start:stop:step or comma-separated wavelengths")
    parser.add_argument("--slices",type=int,default=60)
    parser.add_argument("--grid",type=int,default=512)
    parser.add_argument("--cpu-threads",type=int,default=4)
    parser.add_argument("--output-root",type=Path,default=ROOT/"studies/metals_nv_spectra/results/Nz60_Q512_400-700_61pt")
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--plot-only",action="store_true")
    parser.add_argument("--no-plots",action="store_true")
    return parser


def main(argv=None):
    parser=build_parser()
    args=parser.parse_args(argv)
    output=args.output_root.resolve()
    try:
        if args.plot_only:
            plan=read_json(output/"batch_plan.json")["plan"]
            states={j["key"]:load_state(j,output,plan["wavelengths_nm"]) for j in plan["jobs"]}
            export_batch(plan,states,output,"complete" if all(len(s["cases"])==len(plan["wavelengths_nm"]) for s in states.values()) else "incomplete")
            plot_results(plan,states,output)
            return 0
        plan=build_plan(args)
        print(f"NV spectra: {plan['orders']}; Nz={plan['slices']}, Q={plan['grid']}; {len(plan['wavelengths_nm'])} wavelengths, {len(plan['jobs'])} structures, {len(plan['jobs'])*len(plan['wavelengths_nm'])} cases.",flush=True)
        if args.dry_run:
            nv.write_json(output/"batch_plan_preview.json",plan)
            print("Preflight passed; no RCWA calculation.")
            return 0
        states,summary=execute(plan,output)
        if not args.no_plots:
            try:
                plot_results(plan,states,output)
            except Exception as error:
                nv.atomic_text(output/"plot_error.txt",f"{type(error).__name__}: {error}\n")
                print(f"Plotting failed; spectra and means are saved: {error}",file=sys.stderr)
        return 0 if summary["status"]=="complete" else 1
    except KeyboardInterrupt:
        return 130
    except (ValueError,OSError,KeyError) as error:
        parser.error(str(error))


if __name__=="__main__":
    raise SystemExit(main())
