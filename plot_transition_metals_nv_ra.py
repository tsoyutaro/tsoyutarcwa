"""Audit and plot saved NV reflection/absorption sweeps; no RCWA calculation."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import shutil
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / "studies/transition_metals_nv/results/Nz60_Q512_M4-36_Werner"
MATERIALS = {"Fe":"Iron", "Ni":"Nickel", "Ti":"Titanium", "W":"Tungsten", "V":"Vanadium"}
COLORS = {400.0:"#0072B2", 550.0:"#D55E00", 700.0:"#009E73"}
MARKERS = {400.0:"o", 550.0:"s", 700.0:"^"}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_csv(path, rows):
    with path.open("w",encoding="utf-8-sig",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def close_match(left, right, message):
    if not math.isclose(left, right, abs_tol=2e-12, rel_tol=2e-12):
        raise ValueError(f"{message}: {left} != {right}")

def load_jobs(input_root, destination):
    jobs, hashes, summary, last_values, all_rows, delta_rows = {}, {}, [], [], [], []
    shared_solver = None
    for material in MATERIALS:
        for structure in ("metal","pmma"):
            source = input_root / structure / material.lower()
            paths = {name: source / name for name in ("all_cases.csv","report.json","plan.json","checkpoint.json")}
            for path in paths.values():
                hashes[str(path)] = digest(path)
            report = json.loads(paths["report.json"].read_text(encoding="utf-8"))
            plan = json.loads(paths["plan.json"].read_text(encoding="utf-8"))
            checkpoint = json.loads(paths["checkpoint.json"].read_text(encoding="utf-8"))
            ident = plan["identity"]
            if checkpoint["plan"] != plan or checkpoint["status"] != report["status"]:
                raise ValueError(f"Inconsistent saved plan/status: {material}/{structure}")
            if shared_solver is None:
                shared_solver = ident["source_sha256"]
            elif ident["source_sha256"] != shared_solver:
                raise ValueError("Different solver sources across the ten jobs.")
            if ident["material"] != material or ident["structure"] != structure or report["method"] != "NV":
                raise ValueError("Material/structure/method mismatch.")
            if (ident["slices"],ident["grid"],ident["geometry"]["period_nm"],ident["geometry"]["height_nm"]) != (60,512,200,500):
                raise ValueError("Unexpected fixed conditions.")
            if ident["wavelengths_nm"] != [400.,550.,700.]:
                raise ValueError("Unexpected anchor wavelengths.")
            with paths["all_cases.csv"].open(encoding="utf-8-sig",newline="") as stream:
                raw = list(csv.DictReader(stream))
            cases = {}
            for r in raw:
                m,wave = int(r["order"]),float(r["wavelength_nm"])
                key = (m,wave)
                if key in cases:
                    raise ValueError("Duplicate case.")
                if (r["material"] != material or r["structure"] != structure or r["layer_method"] != "nvm"
                        or r["nv_lanczos_enabled"] != "False" or r["passivity_warning"] != "False"
                        or int(r["slices"]) != 60 or int(r["grid"]) != 512
                        or int(r["total_pattern_layers"]) != 60+int(structure=="pmma")):
                    raise ValueError("Unexpected/nonphysical case flags.")
                row = dict(material=material,structure=structure,wavelength_nm=wave,order=m,slices=60,grid=512,
                           total_pattern_layers=int(r["total_pattern_layers"]))
                for metric in ("reflectance","transmittance","absorptance","power_into_substrate","motheye_absorptance","substrate_absorptance"):
                    row[metric] = float(r[metric]) if r[metric] else None
                    if row[metric] is not None:
                        if not math.isfinite(row[metric]) or not -1e-5 <= row[metric] <= 1+1e-5:
                            raise ValueError("Invalid power in a successful case.")
                        close_match(row[metric],checkpoint["cases"][f"M={m}|wl={wave:g}"][metric],"CSV/checkpoint disagreement")
                close_match(sum(row[k] for k in ("reflectance","transmittance","absorptance")),1.,"Power balance")
                if structure == "metal":
                    close_match(row["absorptance"],row["motheye_absorptance"]+row["substrate_absorptance"],"Absorption split")
                    close_match(row["transmittance"],0.,"Metal far transmission")
                cases[key] = row
                all_rows.append(row)
            waves = ident["wavelengths_nm"]
            orders = sorted(m for m in {key[0] for key in cases} if all((m,w) in cases for w in waves))
            if len(cases) != report["completed_cases"] or orders != report["complete_orders"]:
                raise ValueError("Case count/order list disagrees with the report.")
            if len(checkpoint["cases"]) != len(cases):
                raise ValueError("Checkpoint/CSV count mismatch.")
            checked_metrics = ("reflectance","power_into_substrate","motheye_absorptance") if structure=="metal" else ("reflectance","transmittance","absorptance")
            recomputed = []
            for left,right in zip(orders,orders[1:]):
                if right-left != ident["step"]:
                    continue
                metrics = {k:max(abs(cases[(right,w)][k]-cases[(left,w)][k]) for w in waves) for k in checked_metrics}
                recomputed.append((left,right,metrics,max(metrics.values())))
                for wave in waves:
                    coarse,fine = cases[(left,wave)],cases[(right,wave)]
                    delta_rows.append(dict(material=material,structure=structure,wavelength_nm=wave,from_order=left,to_order=right,
                        R_change_pp=100*(fine["reflectance"]-coarse["reflectance"]),A_change_pp=100*(fine["absorptance"]-coarse["absorptance"])))
            check = report["order_convergence"]
            if len(recomputed) != len(check["comparisons"]):
                raise ValueError("Comparison list length mismatch.")
            for recalculated,saved in zip(recomputed,check["comparisons"]):
                left,right,metrics,maximum = recalculated
                if (left,right) != (saved["coarse"],saved["fine"]):
                    raise ValueError("Comparison order mismatch.")
                close_match(maximum,saved["maximum_absolute_power_change"],"Reported maximum difference")
                for metric,value in metrics.items():
                    close_match(value,saved["per_metric"][metric],"Reported metric difference")
                if saved["passed"] != (maximum <= ident["tolerance"]):
                    raise ValueError("Reported step pass mismatch.")
            tail = len(recomputed)>=2 and all(c[-1]<=ident["tolerance"] for c in recomputed[-2:])
            passed = report["status"]=="requested_range_complete" and tail
            if tail != check["tail_two_steps_passed"] or passed != check["passed_at_fixed_Nz_Q"]:
                raise ValueError("Reported convergence classification mismatch.")
            job = dict(material=material,structure=structure,cases=cases,orders=orders,report=report,plan=plan,passed=passed)
            jobs[(material,structure)] = job
            tail2 = recomputed[-2:]
            summary.append(dict(material=material,structure=structure,status=report["status"],completed_cases=len(cases),
                last_complete_order=orders[-1] if orders else None,order_check_passed_at_fixed_Nz_Q=passed,
                tolerance_pp=100*ident["tolerance"],penultimate_step_max_pp=100*tail2[0][-1] if len(tail2)==2 else None,
                final_step_max_pp=100*tail2[-1][-1] if tail2 else None,
                metrics_compared=";".join(checked_metrics),slices=60,grid=512,
                total_pattern_layers=60+int(structure=="pmma"),metal_data_filename=Path(ident["metal_csv"]).name))
            for wave in waves:
                selected=sorted((r for (m,w),r in cases.items() if w==wave),key=lambda r:r["order"])
                if selected:
                    row=selected[-1]
                    last_values.append(dict(material=material,structure=structure,wavelength_nm=wave,last_order=row["order"],
                        reflectance_percent=100*row["reflectance"],absorptance_percent=100*row["absorptance"],
                        transmittance_percent=100*row["transmittance"],order_check_passed_at_fixed_Nz_Q=passed))
            snapshot = destination / "sources" / structure / material.lower()
            snapshot.mkdir(parents=True,exist_ok=True)
            for name in ("all_cases.csv","report.json","plan.json"):
                shutil.copyfile(paths[name],snapshot/name)
    write_csv(destination/"order_convergence_summary.csv",summary)
    write_csv(destination/"last_order_R_A_T.csv",last_values)
    write_csv(destination/"all_NV_cases.csv",sorted(all_rows,key=lambda r:(r["material"],r["structure"],r["order"],r["wavelength_nm"])))
    write_csv(destination/"R_A_adjacent_changes.csv",delta_rows)
    return jobs,hashes,summary,last_values

def plot_metric(axis,job,metric,title):
    for wave in (400.,550.,700.):
        selected=sorted((row for (m,w),row in job["cases"].items() if w==wave),key=lambda row:row["order"])
        axis.plot([r["order"] for r in selected],[100*r[metric] for r in selected],
            color=COLORS[wave],marker=MARKERS[wave],markersize=3.4,linewidth=1.7,label=f"{wave:g} nm")
    axis.set(title=title,xlabel="Fourier truncation order M",
        ylabel="Reflectance R (%)" if metric=="reflectance" else "Total absorptance A (%)",
        xticks=list(range(4,37,4)),xlim=(3.2,36.8))
    axis.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
    axis.grid(alpha=.25)
    low,high=axis.get_ylim()
    axis.set_ylim(max(0,low),min(100,high))
    axis.ticklabel_format(axis="y",style="plain",useOffset=False)

def save(fig,path):
    for extension in ("png","svg"):
        fig.savefig(path.with_suffix("."+extension),dpi=160)
    plt.close(fig)

def status(job):
    return "PASS" if job["passed"] else "NOT PASSED"

def make_figures(jobs,destination):
    plt.rcParams.update({"font.size":11,"axes.titlesize":13,"svg.fonttype":"none","savefig.facecolor":"white"})
    for material,name in MATERIALS.items():
        fig,axes=plt.subplots(2,2,figsize=(12.8,8.6))
        for row,structure in enumerate(("metal","pmma")):
            label="Metal only" if structure=="metal" else f"PMMA + {material} (30 nm coating)"
            job=jobs[(material,structure)]
            for axis,metric,letter in zip(axes[row],("reflectance","absorptance"),("R","A")):
                plot_metric(axis,job,metric,f"{label}: {letter}")
        fig.suptitle(f"{material} ({name}): NV reflectance and absorptance versus Fourier order\nP=200 nm, h=500 nm; Nz=60, Q=512; normal incidence, x polarization",fontsize=15,y=.985)
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.5,.897),ncol=3,frameon=False,fontsize=12)
        fig.subplots_adjust(top=.81,bottom=.17,left=.085,right=.975,wspace=.24,hspace=.47)
        fig.text(.5,.075,f"Order-tail check (fixed Nz/Q, 0.05 pp): metal {status(jobs[(material,'metal')])}; PMMA {status(jobs[(material,'pmma')])}.\nMetal A includes the metal substrate; PMMA-coated A is metal-film absorption. Independent vertical ranges.",ha="center",fontsize=10,color="#444444")
        save(fig,destination/f"{material}_metal_and_PMMA_R_A_vs_order")
        for structure in ("metal","pmma"):
            job=jobs[(material,structure)]
            fig,axes=plt.subplots(1,2,figsize=(12.8,4.9))
            label=material+" metal only" if structure=="metal" else "PMMA + "+material
            for axis,metric,letter in zip(axes,("reflectance","absorptance"),("R","A")):
                plot_metric(axis,job,metric,letter)
            fig.suptitle(f"{label}: NV order sweep\nP=200 nm, h=500 nm; Nz=60, Q=512; normal x polarization",fontsize=15,y=.985)
            fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.5,.835),ncol=3,frameon=False)
            fig.subplots_adjust(top=.71,bottom=.225,left=.085,right=.975,wspace=.25)
            definition="A = 1 - R; includes semi-infinite metal substrate" if structure=="metal" else "A = 1 - R - T; metal-film absorption, lossless PMMA substrate"
            fig.text(.5,.065,f"{definition}.\nOrder-tail check at fixed Nz/Q: {status(job)}; tolerance 0.05 pp. Independent vertical ranges.",ha="center",fontsize=9.6)
            save(fig,destination/f"{material}_{structure}_R_A_vs_order")
    for structure in ("metal","pmma"):
        fig,axes=plt.subplots(5,2,figsize=(14,18.2))
        for i,material in enumerate(MATERIALS):
            job=jobs[(material,structure)]
            for axis,metric,letter in zip(axes[i],("reflectance","absorptance"),("R","A")):
                plot_metric(axis,job,metric,f"{material}: {letter} ({status(job)})")
        label="Metal-only" if structure=="metal" else "PMMA + metal (30 nm coating)"
        fig.suptitle(f"{label}: NV order dependence of R and A\nP=200 nm, h=500 nm; Nz=60, Q=512; normal x polarization",fontsize=17,y=.99)
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.5,.945),ncol=3,frameon=False,fontsize=13)
        fig.subplots_adjust(top=.91,bottom=.065,left=.085,right=.975,wspace=.24,hspace=.55)
        fig.text(.5,.017,"PASS refers to the saved order-tail criterion at fixed Nz/Q (0.05 pp); it does not certify layer/grid convergence.\nIndependent vertical ranges in every panel. W uses Werner optical constants.",ha="center",fontsize=11)
        save(fig,destination/f"all_{structure}_R_A_vs_order")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root",type=Path,default=DEFAULT)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"studies/transition_metals_nv/figures_R_A")
    args=parser.parse_args()
    source,dest=args.input_root.resolve(),args.output_dir.resolve()
    if dest==source or source in dest.parents:
        parser.error("Choose an output directory outside the input results tree.")
    dest.mkdir(parents=True,exist_ok=True)
    jobs,hashes,summary,last_values=load_jobs(source,dest)
    make_figures(jobs,dest)
    for path,value in hashes.items():
        if digest(Path(path))!=value:
            raise ValueError("Input results changed during plotting; rerun for a consistent snapshot.")
    provenance=dict(rcwa_recalculated=False,input_root=str(source),input_sha256=hashes,successful_cases=sum(len(j["cases"]) for j in jobs.values()),
        materials=list(MATERIALS),structures=["metal","pmma"],method="NV",waves_nm=[400,550,700],slices=60,grid=512,
        scalar_powers_multiplied_by_100=True,CSV_matches_checkpoint=True,reports_independently_recomputed=True,
        shared_solver_source_hashes_equal_across_jobs=True,missing_cases_interpolated=False,
        absorption_definition=dict(metal="moth-eye + semi-infinite metal substrate; A=1-R",pmma="metal film; lossless PMMA substrate; A=1-R-T"),
        summary=summary)
    (dest/"provenance.json").write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    shutil.copyfile(Path(__file__),dest/Path(__file__).name)
    (dest/"README_ja.md").write_text("""# NV法：5金属の単体・PMMA蒸着構造の反射率と吸収率

各金属の*_metal_and_PMMA_R_A_vs_order.png/.svgは上段が金属単体、下段がPMMA蒸着、左列がR、右列がAです。
波長400/550/700 nmを青/橙/緑で表示しています。*_metal_R_A_vs_order、*_pmma_R_A_vs_orderは構造別の図です。
all_metal_R_A_vs_order、all_pmma_R_A_vs_orderは5金属の一覧です。

NV、M=4〜36（2刻み）、Nz=60、Q=512、周期200 nm、高さ500 nm、三角格子、垂直入射、x偏光です。
PMMAコアの底面半径65 nm、金属側面・頂部・谷部膜厚30 nm、頂部キャップを含めると61構造層です。
金属単体の底面半径は95 nm、半無限の同じ金属基板上にあります。
WはWerner(W).csvです。PMMAは無吸収で、400 nmのnだけは従来のlinear-to-400の外挿を用いています。

金属単体のAはモスアイ部分と半無限金属基板を合わせた全吸収率です。
PMMA蒸着のAは金属膜での吸収率です。基板へ入る光はTとして扱い、PMMA基板での吸収には数えません。
図ごとの縦軸範囲は異なります。吸収率と反射率は保存済みCSVの割合を100倍して表示しました。

全10件でM=36までの3波長が完了し、計510ケースです。
最後の2つの次数差（M32→34、M34→36）が許容差0.05パーセントポイント以内かという保存済み判定を再計算しました。
Fe/Ni/Ti/Vは単体とPMMA蒸着の両方で判定通過、Wは両方で未達です。
単体の判定はR・基板流入・モスアイ吸収、PMMAの判定はR/T/Aに基づきます。
この判定は固定NzとQでの次数依存を対象とし、層数・格子や全波長スペクトルの収束を確認したものではありません。

order_convergence_summary.csvは判定一覧、last_order_R_A_T.csvは各波長の最後の次数のR/A/Tです。
all_NV_cases.csvは全ケース、R_A_adjacent_changes.csvは隣接次数でのR/Aの差（パーセントポイント）です。
sourcesには図の元にしたCSV・計算条件・判定のコピーを保存し、provenance.jsonに入力ハッシュと整合性検査を記録しました。
RCWAの再計算はしていません。
""",encoding="utf-8")
    archive=dest/"transition_metals_NV_R_A_figures.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(dest.rglob("*")):
            if path.is_file() and path!=archive and "__pycache__" not in path.parts:
                bundle.write(path,path.relative_to(dest).as_posix())
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
    print(json.dumps(dict(output=str(dest),cases=sum(len(j["cases"]) for j in jobs.values()),
        summary=summary,last_order_values=last_values,archive=str(archive)),ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
