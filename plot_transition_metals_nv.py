"""Plot saved NV order sweeps and export fixed-Nz/Q convergence classifications."""
from pathlib import Path
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
DEFAULT=ROOT/"studies/transition_metals_nv/results/Nz60_Q512_M4-36_Werner"


def make_figures(output_root):
    summary=json.loads((output_root/"batch_summary.json").read_text(encoding="utf-8"))
    assessments=[]
    for job in summary["jobs"]:
        output=output_root/job["structure"]/job["material"].lower()
        if not (output/"all_cases.csv").exists() or not (output/"report.json").exists():
            assessments.append(dict(material=job["material"],structure=job["structure"],status=job["status"],order_tail_passed_at_fixed_Nz_Q=False))
            continue
        report=json.loads((output/"report.json").read_text(encoding="utf-8"))
        with (output/"all_cases.csv").open(encoding="utf-8-sig",newline="") as stream:
            rows=list(csv.DictReader(stream))
        check=report["order_convergence"]
        assessments.append(dict(material=report["material"],structure=report["structure"],status=report["status"],
            completed_cases=report["completed_cases"],last_complete_order=report["last_complete_order"],
            tolerance_pp=check["tolerance_pp"],tail_two_steps_passed=check["tail_two_steps_passed"],
            order_tail_passed_at_fixed_Nz_Q=check["passed_at_fixed_Nz_Q"],working_order=check["working_order"],
            Nz=report["fixed_slices"],Q=report["fixed_grid"],material_csv=report["material_csv"]))
        if not rows:
            continue
        if any(r["layer_method"]!="nvm" or int(r["slices"])!=report["fixed_slices"] or int(r["grid"])!=report["fixed_grid"] for r in rows):
            raise ValueError("Mixed method or fixed conditions in saved CSV.")
        figdir=output/"figures"
        figdir.mkdir(parents=True,exist_ok=True)
        waves=report["wavelengths_nm"]
        plt.rcParams.update({"font.size":11,"svg.fonttype":"none"})
        fig,axes=plt.subplots(1,len(waves),figsize=(4.4*len(waves),4.6),squeeze=False)
        for axis,wave in zip(axes[0],waves):
            selected=sorted((r for r in rows if float(r["wavelength_nm"])==wave),key=lambda r:int(r["order"]))
            if selected:
                axis.plot([int(r["order"]) for r in selected],[100*float(r["reflectance"]) for r in selected],marker="o",linewidth=1.8,color="#0072B2")
            else:
                axis.text(.5,.5,"No completed case",ha="center",transform=axis.transAxes)
            axis.set(title=f"{wave:g} nm",xlabel="Fourier truncation order M",ylabel="Reflectance R (%)")
            axis.grid(alpha=.25)
        name=report["material"] if report["structure"]=="metal" else "PMMA + "+report["material"]
        fig.suptitle(f"{name}: NV order sweep\nNz={report['fixed_slices']}, Q={report['fixed_grid']}; P=200 nm, h=500 nm; normal x polarization",fontsize=14)
        fig.subplots_adjust(top=.75,bottom=.22,left=.07,right=.985,wspace=.30)
        fig.text(.5,.055,f"Status: {report['status']}; last order complete at every wavelength: {report['last_complete_order']}.\nIndependent vertical ranges; failed/uncomputed cases are omitted.",ha="center",fontsize=9)
        for extension in ("png","svg"):
            fig.savefig(figdir/f"reflectance_vs_order.{extension}",dpi=160)
        plt.close(fig)
        fig,axis=plt.subplots(figsize=(9,4.8))
        comparisons=check["comparisons"]
        if comparisons:
            for metric in comparisons[0]["per_metric"]:
                axis.plot([c["fine"] for c in comparisons],[100*c["per_metric"][metric] for c in comparisons],marker="o",label=metric)
        axis.axhline(check["tolerance_pp"],color="#B22222",linestyle="--",label=f"Tolerance {check['tolerance_pp']:g} pp")
        axis.set(title=f"{name}: change between successive orders (max over wavelengths)",xlabel="Higher Fourier order M",ylabel="Absolute power change (percentage points)")
        axis.grid(alpha=.25)
        axis.legend(fontsize=9)
        fig.subplots_adjust(bottom=.24)
        fig.text(.5,.055,f"Two final refinements passed: {check['tail_two_steps_passed']}; requested range complete: {check['requested_range_complete']}.\nThis checks order dependence at fixed Nz/Q, not layer/grid convergence.",ha="center",fontsize=9)
        for extension in ("png","svg"):
            fig.savefig(figdir/f"order_changes.{extension}",dpi=160)
        plt.close(fig)
    fields=list(dict.fromkeys(k for row in assessments for k in row))
    with (output_root/"order_convergence_summary.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(assessments)
    print(f"Saved figures and {output_root/'order_convergence_summary.csv'}")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root",type=Path,default=DEFAULT)
    make_figures(parser.parse_args().output_root.resolve())
