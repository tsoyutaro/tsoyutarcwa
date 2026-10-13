"""Memory-saving NV order sweeps for Fe/Ni/Ti/W/V and their PMMA coatings.

Only Fourier order changes. Cases run in fresh processes, jobs run sequentially,
and every successful wavelength is checkpointed. W defaults to the supplied
Werner table, so all metals use 400/550/700 nm. Tail checks are explicitly
limited to fixed Nz and Q; they do not certify layer/grid convergence.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from studies.transition_metals_nv import runtime as nv
from studies.shared.tabulated_nk import load_tabulated_nk
from studies.pmma_metal_motheye.model import Geometry as CoatedGeometry, PMMAIndex

DATA = ROOT / "studies/transition_metals_nv/data"
FILES = {"Fe": "Fe.csv", "Ni": "Ni.csv", "Ti": "Ti.csv", "W": "Werner(W).csv", "V": "Vanadium.csv"}
NAMES = {"Fe": "Iron", "Ni": "Nickel", "Ti": "Titanium", "W": "Tungsten", "V": "Vanadium"}
ALIASES = {**{k.lower(): k for k in FILES}, **{v.lower(): k for k, v in NAMES.items()}, "nickle": "Ni"}
METAL_GEOMETRY = dict(period_nm=200., height_nm=500., tip_radius_nm=5., base_radius_nm=95., profile_power=1.)
COATED_GEOMETRY = {k: v for k, v in asdict(CoatedGeometry()).items() if k != "asr_circle_g"}
COLUMNS = ("material", "structure", "wavelength_nm", "order", "slices", "grid", "total_pattern_layers",
    "epsilon_material_real", "epsilon_material_imag", "pmma_index", "pmma_shortwave_extrapolated",
    "reflectance", "transmittance", "absorptance", "motheye_absorptance", "substrate_absorptance",
    "power_into_substrate", "passivity_warning", "layer_method", "nv_lanczos_enabled",
    "symmetry_reduction", "reduced_dimension", "full_dimension", "runtime_seconds",
    "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "storage_policy",
    "post_layer_retained_tensor_peak_bytes")
POWER_NAMES = ("reflectance", "transmittance", "absorptance", "motheye_absorptance",
               "substrate_absorptance", "power_into_substrate")


def symbol(value):
    try:
        return ALIASES[value.strip().lower()]
    except KeyError:
        raise ValueError("Materials must be Fe, Ni, Ti, W or V.") from None


def source_hashes():
    paths = [Path(__file__), ROOT / "studies/transition_metals_nv/runtime.py", ROOT / "rcwa_solver_auto.py",
             ROOT / "studies/gold_motheye/converge.py", ROOT / "studies/shared/tabulated_nk.py",
             ROOT / "studies/pmma_metal_motheye/model.py", ROOT / "studies/pmma_gold_motheye/geometry_preview.py"]
    paths += sorted((ROOT / "rcwa_ext").rglob("*.py"))
    return {p.relative_to(ROOT).as_posix(): nv.digest(p) for p in paths}


def build_identity(args, material, structure, waves):
    path = (args.w_csv if material == "W" and args.w_csv else args.data_dir / FILES[material]).resolve()
    model = load_tabulated_nk(path)
    for wave in waves:
        model(wave)
    identity = dict(material=material, structure=structure, geometry=METAL_GEOMETRY if structure == "metal" else COATED_GEOMETRY,
        slices=args.slices, grid=args.grid, wavelengths_nm=waves, start_order=args.start_order, step=args.step,
        requested_device=args.device, cpu_threads=args.cpu_threads, tolerance=args.tolerance,
        metal_csv=str(path), metal_sha256_lf=nv.digest(path),
        material_range_nm=[model.wavelength_nm[0], model.wavelength_nm[-1]],
        solver="NV/NVM, D6 E1 x source, complex128, Redheffer, half S, Lanczos off",
        source_sha256=source_hashes(), storage="reduced", pmma=None)
    if structure == "pmma":
        if args.slices < 2:
            raise ValueError("The partial valley coating requires at least two PMMA profile slices.")
        g = CoatedGeometry(**COATED_GEOMETRY)
        g.validate()
        g.profile(args.slices)
        pmma_path = (args.pmma_csv or args.data_dir / "Szczurowski.csv").resolve()
        pmma = PMMAIndex(pmma_path, "linear-to-400")
        for wave in waves:
            pmma(wave)
        identity["pmma"] = dict(path=str(pmma_path), sha256_lf=nv.digest(pmma_path),
            policy="linear-to-400", provenance=pmma.provenance(pmma_path))
    return identity


def compute_case(identity, order, wave, device, *, storage="reduced", observer=None):
    """Both structures use explicit analytic NV circles/core-shell circles."""
    import torch
    from rcwa_solver_auto import AutoRCWA, GroupTheoryOptions, Lattice, NVMOptions, OutputSpec
    from studies.gold_motheye.converge import _zero_order_x_source, _mean_poynting_z
    started = time.perf_counter()
    eps = load_tabulated_nk(identity["metal_csv"])(wave)
    coated = identity["structure"] == "pmma"
    g = identity["geometry"]
    pmma = PMMAIndex(Path(identity["pmma"]["path"]), identity["pmma"]["policy"]) if coated else None
    index = pmma(wave) if coated else None
    base = nv.memory_safe_class(AutoRCWA) if storage == "reduced" else AutoRCWA
    class DirectNV(base):
        # Shell addition calls this method internally, so each cap, shell and
        # valley is pruned exactly once after installing its reduced modes.
        def add_layer_circle_nvm(self, *values, **options):
            super().add_layer_circle_nvm(*values, **options)
            if storage == "reduced":
                nv.discard_unused_layer_storage(self)
            held = nv.tensor_storage_bytes(vars(self))
            self._post_layer_peak_bytes = max(getattr(self, "_post_layer_peak_bytes", 0), held)
    sim = DirectNV(freq=g["period_nm"] / wave, order=[order, order], lattice=Lattice.triangular(1.),
        cascade="redheffer", outputs=OutputSpec(smatrix_size="half", fields="none"),
        nvm=NVMOptions(grid=(identity["grid"], identity["grid"]), use_lanczos=False),
        group_theory=GroupTheoryOptions(enabled=True, symmetry="d6", strict=True, polarization="x"),
        verify_cascade=False, dtype=torch.complex128, device=device)
    sim.add_input_layer(eps=1., mu=1.)
    sim.add_output_layer(eps=index**2 if coated else eps, mu=1.)
    sim.set_incident_angle(0., 0.)
    centers = ((.5 * (float(sim.L[0]) + float(sim.L[1]) * sim.cos_zeta),
                .5 * float(sim.L[1]) * sim.sin_zeta),)
    options = dict(centers=centers, nx=identity["grid"], ny=identity["grid"], use_lanczos=False)
    period = g["period_nm"]
    if coated:
        geom = CoatedGeometry(**g)
        if g["metal_top_thickness_nm"] > 0:
            sim.add_layer_circle_nvm(g["metal_top_thickness_nm"] / period,
                (g["tip_radius_nm"] + g["metal_side_thickness_nm"]) / period, 1., eps, **options)
        for layer in geom.profile(identity["slices"]):
            thickness = (layer.bottom_depth_nm - layer.top_depth_nm) / period
            if layer.kind == "valley":
                sim.add_layer_circle_nvm(thickness, layer.core_radius_nm / period, eps, index**2, **options)
            else:
                sim.add_layer_circle_shell_nvm(thickness, layer.core_radius_nm / period,
                    layer.outer_radius_nm / period, 1., eps, index**2, **options)
    else:
        for layer in range(identity["slices"]):
            radius = g["tip_radius_nm"] + (g["base_radius_nm"] - g["tip_radius_nm"]) * ((layer + .5) / identity["slices"])**g["profile_power"]
            sim.add_layer_circle_nvm(g["height_nm"] / identity["slices"] / period, radius / period, 1., eps, **options)
    if any(r.method != "nvm" or r.options["use_lanczos"] for r in sim.layer_records):
        raise RuntimeError("Unexpected non-NV or filtered layer.")
    if len(sim._polarized_layers) != sim.layer_N or any(d.get("backend") != "NVM" for d in sim.group_theory_diagnostics):
        raise RuntimeError("Every layer must use the NV D6 source sector.")
    sim.solve_global_smatrix()
    source = _zero_order_x_source(sim)
    pin = _mean_poynting_z(source, sim.Vi, direction=1)
    if not pin > 0:
        raise RuntimeError("Incident power must be positive.")
    r = -_mean_poynting_z(sim.S[1] @ source, sim.Vi, direction=-1) / pin
    output_power = _mean_poynting_z(sim.S[0] @ source, sim.Vo, direction=1) / pin
    if not coated and output_power == 0:
        raise RuntimeError("Power into the lossy metal substrate is unresolved.")
    t = output_power if coated else 0.
    a = 1 - r - t
    row = dict(material=identity["material"], structure=identity["structure"], wavelength_nm=wave,
        order=order, slices=identity["slices"], grid=identity["grid"], total_pattern_layers=sim.layer_N,
        epsilon_material_real=eps.real, epsilon_material_imag=eps.imag, pmma_index=index,
        pmma_shortwave_extrapolated=bool(coated and wave < pmma.wavelengths[0]), reflectance=r,
        transmittance=t, absorptance=a, motheye_absorptance=None if coated else a-output_power,
        substrate_absorptance=None if coated else output_power, power_into_substrate=output_power,
        passivity_warning=any(x < -1e-6 or x > 1+1e-5 for x in (r,t,a,output_power))
            or (not coated and a-output_power < -1e-5), layer_method="nvm", nv_lanczos_enabled=False,
        symmetry_reduction=sim.group_theory_diagnostics[-1]["symmetry"],
        reduced_dimension=sim.cascade_diagnostics.get("reduced_dimension"), full_dimension=2*sim.order_N,
        runtime_seconds=time.perf_counter()-started, storage_policy=nv.STORAGE_POLICY if storage == "reduced" else "NV-full-storage",
        post_layer_retained_tensor_peak_bytes=sim._post_layer_peak_bytes)
    if observer is not None:
        observer(sim)
    return row


def validate_case(row, identity):
    order, wave = int(row["order"]), float(row["wavelength_nm"])
    if (row["material"] != identity["material"] or row["structure"] != identity["structure"]
            or int(row["slices"]) != identity["slices"] or int(row["grid"]) != identity["grid"]
            or wave not in identity["wavelengths_nm"] or order < identity["start_order"]
            or (order - identity["start_order"]) % identity["step"]
            or row["layer_method"] != "nvm" or row["nv_lanczos_enabled"] is not False
            or row["total_pattern_layers"] != identity["slices"] + int(identity["structure"] == "pmma" and identity["geometry"]["metal_top_thickness_nm"] > 0)):
        raise ValueError("Case conditions disagree with the requested NV job.")
    values = {k: float(row[k]) for k in POWER_NAMES if row.get(k) is not None}
    if row["passivity_warning"] or any(not math.isfinite(v) or v < -1e-5 or v > 1+1e-5 for v in values.values()):
        raise ValueError(f"Nonphysical power result (not CUDA OOM): {values}")
    if abs(sum(values[k] for k in ("reflectance","transmittance","absorptance"))-1) > 1e-8:
        raise ValueError("Power balance is inconsistent.")


def convergence_report(saved):
    identity = saved["plan"]["identity"]
    cases, waves = saved["cases"], identity["wavelengths_nm"]
    orders = sorted(m for m in {int(r["order"]) for r in cases.values()}
        if all(nv.case_key(m,w) in cases for w in waves))
    metrics = ("reflectance", "power_into_substrate", "motheye_absorptance") if identity["structure"] == "metal" else ("reflectance","transmittance","absorptance")
    comparisons = []
    for left, right in zip(orders, orders[1:]):
        if right-left != identity["step"]:
            continue
        changes = {name: max(abs(cases[nv.case_key(right,w)][name]-cases[nv.case_key(left,w)][name]) for w in waves) for name in metrics}
        comparisons.append(dict(coarse=left, fine=right, maximum_absolute_power_change=max(changes.values()),
            maximum_change_pp=100*max(changes.values()), per_metric=changes, passed=max(changes.values()) <= identity["tolerance"]))
    tail = len(comparisons) >= 2 and all(c["passed"] for c in comparisons[-2:])
    completed = saved["status"] == "requested_range_complete"
    return dict(status=saved["status"], material=identity["material"], structure=identity["structure"],
        completed_cases=len(cases), complete_orders=orders, last_complete_order=max(orders,default=None),
        failure=saved.get("failure"), method="NV", fixed_slices=identity["slices"], fixed_grid=identity["grid"],
        wavelengths_nm=waves, material_csv=identity["metal_csv"], material_sha256_lf=identity["metal_sha256_lf"],
        order_convergence=dict(tolerance=identity["tolerance"], tolerance_pp=100*identity["tolerance"],
            comparisons=comparisons, tail_two_steps_passed=tail, requested_range_complete=completed,
            passed_at_fixed_Nz_Q=completed and tail, working_order=orders[-2] if completed and tail else None,
            interpretation="Only two refinements at the available high-order tail; Nz and Q convergence not checked."))


def persist(output, saved):
    report = convergence_report(saved)
    saved["last_complete_order"] = report["last_complete_order"]
    nv.write_json(output/"checkpoint.json", saved)
    stream=io.StringIO(newline="")
    writer=csv.DictWriter(stream,fieldnames=COLUMNS,extrasaction="ignore",lineterminator="\n")
    writer.writeheader()
    writer.writerows(sorted(saved["cases"].values(),key=lambda r:(r["order"],r["wavelength_nm"])))
    nv.atomic_text(output/"all_cases.csv",stream.getvalue())
    nv.write_json(output/"report.json",report)
    return report


def worker(plan_path, result_path, order, wave):
    identity=json.loads(plan_path.read_text(encoding="utf-8"))["identity"]
    row=None
    torch=None
    try:
        if source_hashes() != identity["source_sha256"] or nv.digest(identity["metal_csv"]) != identity["metal_sha256_lf"]:
            raise ValueError("Solver or metal data changed after preflight.")
        if identity["pmma"] and nv.digest(identity["pmma"]["path"]) != identity["pmma"]["sha256_lf"]:
            raise ValueError("PMMA data changed after preflight.")
        import torch
        torch.set_num_threads(identity["cpu_threads"])
        device=nv.initialize_device(torch,identity["requested_device"])
        row=compute_case(identity,order,wave,device)
        validate_case(row,identity)
        if device.type=="cuda":
            torch.cuda.synchronize(device)
        row.update(peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type=="cuda" else None,
            peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved(device)) if device.type=="cuda" else None)
        environment=dict(torch=str(torch.__version__),cuda=torch.version.cuda,device=str(device),
            gpu=torch.cuda.get_device_name(device) if device.type=="cuda" else None)
        nv.write_json(result_path,dict(status="ok",signature=nv.identity_signature(identity),case=row,environment=environment))
        print(f"  R={100*row['reflectance']:.6f}%, T={100*row['transmittance']:.6f}%, A={100*row['absorptance']:.6f}%",flush=True)
        if device.type=="cuda":
            print(f"  peak CUDA allocated={row['peak_cuda_allocated_bytes']/2**30:.3f} GiB",flush=True)
        return 0
    except Exception as error:
        oom=torch is not None and isinstance(error,torch.cuda.OutOfMemoryError) and identity["requested_device"].startswith("cuda")
        failure=dict(status="oom" if oom else "calculation_failed",order=order,wavelength_nm=wave,
            error=f"{type(error).__name__}: {error}",traceback=traceback.format_exc())
        if row:
            failure["power_diagnostics"]={k:float(row[k]) if math.isfinite(float(row[k])) else repr(row[k]) for k in POWER_NAMES if row.get(k) is not None}
        nv.write_json(result_path,failure)
        print(failure["error"],file=sys.stderr,flush=True)
        return 2 if oom else 1


def launch_worker(plan_path,result_path,order,wave):
    result=subprocess.run([sys.executable,"-u",str(Path(__file__).resolve()),"--_worker-plan",str(plan_path),
        "--_worker-result",str(result_path),"--_worker-order",str(order),"--_worker-wave",str(wave)],cwd=ROOT,check=False)
    if not result_path.exists():
        return dict(status="calculation_failed",error=f"Worker exited {result.returncode} without a result.")
    response=json.loads(result_path.read_text(encoding="utf-8"))
    expected=0 if response["status"]=="ok" else 2 if response["status"]=="oom" else 1
    if result.returncode != expected:
        return dict(status="calculation_failed",error="Worker status and exit code disagree.")
    return response


def run_job(plan,output,*,dry_run=False):
    identity=plan["identity"]
    signature=nv.identity_signature(identity)
    checkpoint=output/"checkpoint.json"
    if checkpoint.exists():
        saved=json.loads(checkpoint.read_text(encoding="utf-8"))
        if saved["signature"]!=signature:
            raise ValueError("Cached source/material/fixed conditions differ; use another --output-root.")
        for key,row in saved["cases"].items():
            validate_case(row,identity)
            if key!=nv.case_key(int(row["order"]),float(row["wavelength_nm"])):
                raise ValueError("Cached case key disagrees with its conditions.")
    else:
        saved=dict(signature=signature,cases={})
    saved.update(plan=plan,status="running",failure=None)
    output.mkdir(parents=True,exist_ok=True)
    nv.write_json(output/("plan_preview.json" if dry_run else "plan.json"),plan)
    if identity["structure"] == "pmma":
        profile=[asdict(layer) for layer in CoatedGeometry(**identity["geometry"]).profile(identity["slices"])]
        stream=io.StringIO(newline="")
        writer=csv.DictWriter(stream,fieldnames=list(profile[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(profile)
        nv.atomic_text(output/"profile_layers.csv",stream.getvalue())
    print(f"\n=== {identity['material']} / {identity['structure']} NV: Nz={identity['slices']}, Q={identity['grid']}; resume {len(saved['cases'])} cases ===",flush=True)
    if dry_run:
        return dict(status="planned",material=identity["material"],structure=identity["structure"])
    persist(output,saved)
    order=identity["start_order"]
    try:
        while not plan["max_order"] or order<=plan["max_order"]:
            if identity["grid"]<max(32,4*order+4):
                saved.update(status="fixed_grid_limit",failure=dict(order=order,error="Fixed NV grid is below 4*M+4."))
                return persist(output,saved)
            for wave in identity["wavelengths_nm"]:
                key=nv.case_key(order,wave)
                if key in saved["cases"]:
                    continue
                print(f"solve: lambda={wave:g} nm, M={order}",flush=True)
                response=launch_worker(output/"plan.json",output/"worker_results"/f"M{order}_wl{wave:g}_{uuid.uuid4().hex}.json",order,wave)
                if response["status"]!="ok":
                    saved.update(status=response["status"],failure=response)
                    return persist(output,saved)
                if response["signature"]!=signature:
                    raise ValueError("Worker signature disagrees with plan.")
                if saved.get("environment") not in (None,response["environment"]):
                    raise ValueError("Runtime environment changed; use a separate output root.")
                row=response["case"]
                validate_case(row,identity)
                if nv.case_key(int(row["order"]),float(row["wavelength_nm"]))!=key:
                    raise ValueError("Worker returned the wrong case.")
                saved["environment"]=response["environment"]
                saved["cases"][key]=row
                persist(output,saved)
            order+=identity["step"]
    except KeyboardInterrupt:
        saved.update(status="interrupted",failure=dict(order=order,error="KeyboardInterrupt"))
        persist(output,saved)
        raise
    except Exception as error:
        saved.update(status="calculation_failed",failure=dict(order=order,error=f"{type(error).__name__}: {error}"))
        return persist(output,saved)
    saved["status"]="requested_range_complete"
    return persist(output,saved)


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--materials",default="Fe,Ni,Ti,W,V")
    p.add_argument("--structures",choices=("both","metal","pmma"),default="both")
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--slices",type=int,default=60)
    p.add_argument("--grid",type=int,default=512,help="Fixed normal-vector sampling grid")
    p.add_argument("--start-order",type=int,default=4)
    p.add_argument("--step",type=int,default=2)
    p.add_argument("--max-order",type=int,default=36,help="0: keep increasing until failure or fixed-grid limit")
    p.add_argument("--wavelengths",default="400,550,700")
    p.add_argument("--tolerance",type=float,default=.0005,help="Power fraction; .0005 = .05 percentage point")
    p.add_argument("--cpu-threads",type=int,default=4)
    p.add_argument("--data-dir",type=Path,default=DATA)
    p.add_argument("--w-csv",type=Path,help="Default: supplied Werner(W).csv")
    p.add_argument("--pmma-csv",type=Path)
    p.add_argument("--output-root",type=Path,default=ROOT/"studies/transition_metals_nv/results/Nz60_Q512_M4-36_Werner")
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--_worker-plan",type=Path,help=argparse.SUPPRESS)
    p.add_argument("--_worker-result",type=Path,help=argparse.SUPPRESS)
    p.add_argument("--_worker-order",type=int,help=argparse.SUPPRESS)
    p.add_argument("--_worker-wave",type=float,help=argparse.SUPPRESS)
    return p


def main(argv=None):
    parser=build_parser()
    args=parser.parse_args(argv)
    if args._worker_plan is not None:
        return worker(args._worker_plan,args._worker_result,args._worker_order,args._worker_wave)
    try:
        materials=list(dict.fromkeys(symbol(s) for s in args.materials.split(",")))
        waves=sorted(set(float(s.strip()) for s in args.wavelengths.split(",")))
        if not materials or not waves or any(not math.isfinite(w) or w<=0 for w in waves):
            raise ValueError("Use nonempty materials and positive finite wavelengths.")
        if min(args.slices,args.grid,args.start_order,args.step,args.cpu_threads)<=0 or args.max_order<0:
            raise ValueError("Numerical parameters must be positive; max-order may be zero.")
        if args.max_order and args.max_order<args.start_order:
            raise ValueError("max-order must be >= start-order or zero.")
        if args.grid<max(32,4*args.start_order+4) or not math.isfinite(args.tolerance) or args.tolerance<=0:
            raise ValueError("Grid must be >= max(32,4*M+4) and tolerance positive.")
        if args.device not in ("cpu","cuda") and not (args.device.startswith("cuda:") and args.device[5:].isdigit()):
            raise ValueError("Use cpu, cuda or cuda:<index>.")
        structures=("metal","pmma") if args.structures=="both" else (args.structures,)
        jobs=[(dict(identity=build_identity(args,m,s,waves),max_order=args.max_order),args.output_root.resolve()/s/m.lower())
              for m in materials for s in structures]
    except (ValueError,OSError,KeyError) as error:
        parser.error(str(error))
    summary=dict(status="planned" if args.dry_run else "running",dry_run=args.dry_run,materials=materials,
        structures=list(structures),method="NV",w_data_default="Werner(W).csv",jobs=[],
        interpretation="Order-only convergence checks at fixed Nz/Q; no final spectrum or layer/grid sweep.")
    summary_path=args.output_root.resolve()/("batch_plan.json" if args.dry_run else "batch_summary.json")
    nv.write_json(summary_path,summary)
    try:
        for plan,output in jobs:
            try:
                report=run_job(plan,output,dry_run=args.dry_run)
            except (ValueError,OSError,KeyError) as error:
                report=dict(status="job_setup_failed",error=str(error),material=plan["identity"]["material"],structure=plan["identity"]["structure"])
            summary["jobs"].append(dict(output=str(output),**report))
            nv.write_json(summary_path,summary)
            print(f"Job status: {report['status']}; last complete M={report.get('last_complete_order')}; fixed-Nz/Q tail passed={report.get('order_convergence',{}).get('passed_at_fixed_Nz_Q')}",flush=True)
    except KeyboardInterrupt:
        summary["status"]="interrupted"
        nv.write_json(summary_path,summary)
        return 130
    failed=any(j["status"] in ("calculation_failed","job_setup_failed") for j in summary["jobs"])
    stopped=any(j["status"] in ("oom","fixed_grid_limit") for j in summary["jobs"])
    unpassed=not args.dry_run and any(not j.get("order_convergence",{}).get("passed_at_fixed_Nz_Q") for j in summary["jobs"])
    summary["status"]="planned" if args.dry_run and not failed else "failed" if failed else "some_stopped" if stopped else "some_order_checks_unpassed" if unpassed else "order_checks_passed_at_fixed_Nz_Q"
    nv.write_json(summary_path,summary)
    print(f"Batch status: {summary['status']}\nSummary: {summary_path}",flush=True)
    return 1 if failed else 2 if stopped or unpassed else 0


if __name__=="__main__":
    raise SystemExit(main())
