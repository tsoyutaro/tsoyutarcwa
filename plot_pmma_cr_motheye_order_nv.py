"""Plot Cr and PMMA/Cr R/A/T and their fixed-Nz/Q order checks."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
DEFAULT=ROOT/"studies/pmma_cr_motheye_nv/results/Nz60_Q512_M4-36"
COLORS={400.:"#0072B2",550.:"#D55E00",700.:"#009E73"}

def make_figures(output):
    batch=json.loads((output/"batch_summary.json").read_text(encoding="utf-8"))
    jobs=[]
    for entry in batch["jobs"]:
        folder=output/entry["structure"]/"cr"
        if not (folder/"all_cases.csv").exists():
            continue
        report=json.loads((folder/"report.json").read_text(encoding="utf-8"))
        plan=json.loads((folder/"plan.json").read_text(encoding="utf-8"))["identity"]
        with (folder/"all_cases.csv").open(encoding="utf-8-sig",newline="") as stream:
            rows=list(csv.DictReader(stream))
        if any(r["material"]!="Cr" or r["layer_method"]!="nvm" or r["nv_lanczos_enabled"]!="False"
               or int(r["slices"])!=plan["slices"] or int(r["grid"])!=plan["grid"] for r in rows):
            raise ValueError("Mixed material/method/fixed conditions in a saved CSV.")
        jobs.append((entry["structure"],report,plan,rows))
    if not jobs:
        raise ValueError("No saved Cr result to plot.")
    figdir=output/"figures"
    figdir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":11,"svg.fonttype":"none"})
    fig,axes=plt.subplots(len(jobs),3,figsize=(15.5,5.7 if len(jobs)==1 else 4.1*len(jobs)+1.3),squeeze=False)
    for row,(structure,report,plan,records) in enumerate(jobs):
        label="Cr only" if structure=="metal" else "PMMA + Cr"
        for axis,metric,letter in zip(axes[row],("reflectance","absorptance","transmittance"),("R","A","T")):
            for wave in plan["wavelengths_nm"]:
                selected=sorted((r for r in records if float(r["wavelength_nm"])==wave),key=lambda r:int(r["order"]))
                axis.plot([int(r["order"]) for r in selected],[100*float(r[metric]) for r in selected],
                    marker="o",markersize=3.5,color=COLORS.get(wave),label=f"{wave:g} nm",linewidth=1.7)
            axis.set(title=f"{label}: {letter}",xlabel="Fourier truncation order M",ylabel=f"{letter} (%)")
            axis.grid(alpha=.25)
            axis.ticklabel_format(axis="y",style="plain",useOffset=False)
    plan=jobs[0][2]
    fig.suptitle(f"Cr moth-eye: NV order dependence of R, A and T\nNz={plan['slices']}, Q={plan['grid']}; P=200 nm, h=500 nm; normal x polarization; Cr data={Path(plan['metal_csv']).name}",fontsize=15,y=.985)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.5,.81 if len(jobs)==1 else .9),ncol=3,frameon=False)
    fig.subplots_adjust(top=.67 if len(jobs)==1 else .82,bottom=.27 if len(jobs)==1 else .16,
                        left=.07,right=.985,wspace=.31,hspace=.45)
    status="; ".join(f"{s}: {r['status']}, order-tail {'PASS' if r['order_convergence']['passed_at_fixed_Nz_Q'] else 'NOT PASSED'}" for s,r,_,_ in jobs)
    definition="A=1-R-T: Cr-film absorption; T enters the lossless PMMA substrate." if all(s=="pmma" for s,_,_,_ in jobs) else "Cr-only A includes the metal substrate; coated A is Cr-film absorption."
    fig.text(.5,.06 if len(jobs)==1 else .075,f"{status}.\n{definition} Independent vertical ranges; failed/uncomputed cases omitted.",ha="center",fontsize=9)
    for ext in ("png","svg"):
        fig.savefig(figdir/f"PMMA_Cr_R_A_T_vs_order.{ext}",dpi=160)
    plt.close(fig)
    for structure,report,plan,records in jobs:
        fig,axis=plt.subplots(figsize=(10,4.7))
        check=report["order_convergence"]
        comparisons=check["comparisons"]
        if comparisons:
            for metric in comparisons[0]["per_metric"]:
                axis.plot([r["fine"] for r in comparisons],[100*r["per_metric"][metric] for r in comparisons],marker="o",label=metric)
        axis.axhline(check["tolerance_pp"],ls="--",color="#B22222",label=f"Tolerance {check['tolerance_pp']:g} pp")
        axis.set(title=f"{'Cr only' if structure=='metal' else 'PMMA + Cr'}: successive-order change (max over wavelengths)",
                 xlabel="Higher Fourier order M",ylabel="Absolute power change (percentage points)")
        axis.grid(alpha=.25)
        axis.legend(fontsize=9)
        fig.subplots_adjust(bottom=.24)
        fig.text(.5,.06,"This checks order dependence at fixed Nz/Q; it does not certify layer/grid or full-spectrum convergence.",ha="center",fontsize=9)
        for ext in ("png","svg"):
            fig.savefig(figdir/f"{structure}_order_changes.{ext}",dpi=160)
        plt.close(fig)
    print(f"Saved Cr R/A/T and order-change figures: {figdir}")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root",type=Path,default=DEFAULT)
    make_figures(parser.parse_args().output_root.resolve())
