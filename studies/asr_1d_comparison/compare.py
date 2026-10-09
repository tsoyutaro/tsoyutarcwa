"""Run/resume matched-geometry Li versus ASR order sweeps in both studies."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def integer_list(raw):
    values = [int(value.strip()) for value in raw.split(",")]
    if not values or any(value < 1 for value in values) or any(a >= b for a, b in zip(values, values[1:])):
        raise argparse.ArgumentTypeError("Use positive, strictly increasing integers")
    return values


def healthy(observables, metrics, tolerance):
    return bool(observables and all(isinstance(observables.get(name), (int, float))
                and math.isfinite(observables[name]) and -tolerance <= observables[name] <= 1+tolerance
                for name in metrics))


def fingerprint(study, config, args):
    paths = [ROOT / "studies/asr_1d_comparison/adapters.py",
             ROOT / "studies/gold_grating_1d/solver.py", ROOT / "studies/gold_grating_1d/common.py",
             ROOT / "studies/shared/gold_dispersion.py", ROOT / "rcwa_solver_auto.py"]
    paths += list((ROOT / "rcwa_ext").glob("*.py"))
    paths += [ROOT / "paper_reproductions/vallius2002" / name for name in
              ("solver.py", "devices.py", "torch_backend.py")]
    if study == "pmma_gold_grating_1d":
        paths += [ROOT / "studies" / study / name for name in
                  ("solver.py", "common.py", "geometry.py", "materials.py")]
    material = config["material"]
    tables = ([material["resolved_path"]] if study == "gold_grating_1d" else
              [material["resolved_gold_path"], material["resolved_pmma_path"]])
    inputs = dict(study=study, geometry=config["geometry"],
                  source_sha256={path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in sorted(set(paths))},
                  material_sha256={Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in tables},
                  pmma_extension=material.get("pmma_shortwave_extension"), G=args.G,
                  quadrature_minimum=args.quadrature, retention="smallest_abs", q_projection=args.q_projection,
                  dtype="complex128", incidence="normal", li_coefficients="analytic")
    signature = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    return inputs, signature


def summarize(study, checkpoint, plan):
    from .adapters import METRICS
    metrics = METRICS[study]
    cases, waves, orders = checkpoint["cases"], plan["wavelengths_nm"], plan["orders"]
    methods = ["li", *(f"asr_r{ratio}" for ratio in plan["asr_ratios"])]

    def get(method, order, wave):
        return cases.get(f"{method}|{plan['slices']}|{order}|{wave:.12g}", {})

    references = {str(wave): get("li", plan["reference_order"], wave) for wave in waves}
    checks = {str(wave): get("li", plan["reference_check_order"], wave) for wave in waves}
    reference_ok = all(healthy(references[str(w)].get("polarizations", {}).get(p), metrics,
                               plan["passivity_tolerance"]) and
                       healthy(checks[str(w)].get("polarizations", {}).get(p), metrics,
                               plan["passivity_tolerance"]) for w in waves for p in ("TE", "TM"))
    reference_delta = max((abs(references[str(w)]["polarizations"][p][k] - checks[str(w)]["polarizations"][p][k])
                           for w in waves for p in ("TE", "TM") for k in metrics), default=0.) if reference_ok else None
    records, summaries = [], {}
    for method in methods:
        adjacent = []
        for low, high in zip(orders[:-1], orders[1:]):
            differences = []
            for wave in waves:
                for p in ("TE", "TM"):
                    left, right = (get(method, n, wave).get("polarizations", {}).get(p) for n in (low, high))
                    if healthy(left, metrics, plan["passivity_tolerance"]) and healthy(right, metrics, plan["passivity_tolerance"]):
                        differences += [abs(left[k]-right[k]) for k in metrics]
            complete = len(differences) == len(waves)*2*len(metrics)
            adjacent.append(dict(low=low, high=high, complete_and_passive=complete,
                                 max_absolute_change=max(differences) if complete else None,
                                 passes=complete and max(differences) <= plan["tolerance"]))
        invalid = 0
        for order in orders:
            for wave in waves:
                case = get(method, order, wave)
                for p in ("TE", "TM"):
                    values = case.get("polarizations", {}).get(p, {})
                    valid = healthy(values, metrics, plan["passivity_tolerance"])
                    invalid += not valid
                    row = dict(study=study, series=method, order=order, harmonics=2*order+1,
                               slices=plan["slices"], wavelength_nm=wave, polarization=p,
                               eigen_dimension=case.get("eigen_dimension"),
                               internal_ratio=case.get("oversampling"), G=case.get("G"),
                               q_projection=case.get("q_projection", "direct"),
                               quadrature_actual=case.get("quadrature_actual", 0),
                               total_finite_layers=case.get("total_finite_layers", plan["slices"]),
                               device=case.get("execution", case.get("environment", {})).get("device"),
                               backend=case.get("execution", {}).get("backend", "torch" if method == "li" else None),
                               runtime_seconds_both_polarizations=case.get("runtime_seconds"),
                               passivity_ok=valid, error=case.get("error", ""), **values)
                    ref = references[str(wave)].get("polarizations", {}).get(p, {})
                    for k in metrics:
                        row[f"difference_from_finite_li_reference_{k}"] = abs(values[k]-ref[k]) if (
                            valid and healthy(ref, metrics, plan["passivity_tolerance"])) else None
                    records.append(row)
        last_differences = [r[f"difference_from_finite_li_reference_{k}"]
                            for r in records if r["series"] == method and r["order"] == orders[-1]
                            for k in metrics if r[f"difference_from_finite_li_reference_{k}"] is not None]
        last_complete = len(last_differences) == len(waves)*2*len(metrics)
        last_delta = max(last_differences) if last_complete else None
        tail_stable = len(adjacent) >= 2 and all(r["passes"] for r in adjacent[-2:])
        last_agrees = last_complete and last_delta <= plan["tolerance"]
        summaries[method] = dict(invalid_polarization_cases=invalid, adjacent_changes=adjacent,
                                 tail_stable_at_tested_wavelengths=tail_stable,
                                 last_order_max_absolute_difference_from_finite_reference=last_delta,
                                 last_order_complete_and_passive_against_reference=last_complete,
                                 last_order_within_reference_tolerance=last_agrees,
                                 tail_and_finite_reference_agree_within_tolerance=(
                                     tail_stable and last_agrees and reference_ok
                                     and reference_delta <= plan["tolerance"]))
    report = dict(plan=plan, methods=summaries, reference_check_max_absolute_change=reference_delta,
                  reference_passivity_ok=reference_ok,
                  finite_reference_stable_at_tolerance=reference_ok and reference_delta <= plan["tolerance"],
                  scope="Fixed geometry slices, selected wavelengths, finite Li reference; not an exact-solution error bound.",
                  absorption_note="Absorption is a port-power residual, not an independent volume integral.")
    return report, records, references


def export(study, output, checkpoint, plan):
    from .plotting import plot_results
    report, records, references = summarize(study, checkpoint, plan)
    save_json(output / "report.json", report)
    columns = list(dict.fromkeys(key for row in records for key in row))
    with (output / "cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    save_json(output / "invalid_cases.json", [r for r in records if not r["passivity_ok"]])
    plot_results(study, records, references, plan, output)
    lines = [f"**{study}: Cartesian Li と Vallius 1D ASR の次数比較**", "",
             f"元studyのconfig・材料関数を使用。周期={plan['geometry']['period_nm']} nm。形状はplan.jsonのgeometryに保存。",
             f"Nz={plan['slices']}固定、波長={plan['wavelengths_nm']} nm、TE/TM、N=2M+1。",
             f"M={plan['orders']}、ASR内部倍率={plan['asr_ratios']}、G={plan['G']}。",
             f"ASR TM境界場={plan.get('q_projection', 'direct')}。galerkinは電力内積を保つ改良版で、TEの境界場は従来と同じです。",
             f"Li参照M={plan['reference_order']}、参照の確認M={plan['reference_check_order']}。",
             f"参照間の全波長・全偏光・全指標の最大絶対差={report['reference_check_max_absolute_change']}。",
             "参照は有限次数の計算で、厳密解ではありません。層数の収束もこの次数掃引では再判定していません。", "",
             "金格子のP_subは吸収性金基板に入る界面電力で、遠方透過T_far=0。PMMA格子のTはPMMA基板への透過電力です。",
             "図の物理量は%、誤差と隣接差はパーセントポイントです。", "",
             "受動性の範囲を破った値・計算失敗は赤い×で表示し、元の値はCSV/invalid_cases.jsonに保存します。",
             "最大誤差の図は、その偏光で全波長が受動性範囲を満たす点だけを結びます。", "",
             "| 系列 | 無効な偏光ケース数 | 最後の2区間が許容差内 | 最終次数と有限参照の差も許容差内 |", "|---|---:|---|---|",
             *[f"| {key} | {value['invalid_polarization_cases']} | {value['tail_stable_at_tested_wavelengths']} | {value['last_order_within_reference_tolerance']} |"
               for key, value in report["methods"].items()], "",
             "[物理量](values.png)・[有限Li参照との差](reference_difference.png)・[隣接次数の差](adjacent_difference.png)", "",
             "条件とソース・材料のハッシュはplan.json/checkpoint.json、各ケースの実行環境はcheckpoint.jsonに保存しています。",
             "ASRは内部固有モードから|γ²|が小さいN個を選び、物理x空間のN成分へ射影します。GPU化はこの打切り誤差を減らしません。"]
    (output / "RESULTS_ja.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Figures and report: {output}", flush=True)


def main(argv=None, default_study="all"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("all", "gold_grating_1d", "pmma_gold_grating_1d"), default=default_study)
    parser.add_argument("--orders", type=integer_list, default=[2, 4, 8, 12, 16, 24])
    parser.add_argument("--asr-ratios", type=integer_list, default=[4])
    parser.add_argument("--gold-slices", type=int, default=420)
    parser.add_argument("--pmma-slices", type=int, default=300)
    parser.add_argument("--slices", type=int, help="Override the slice count for the selected study/studies")
    parser.add_argument("--wavelengths", help="Comma-separated nm; default is the study's seven wavelengths")
    parser.add_argument("--reference-order", type=int, default=80)
    parser.add_argument("--reference-check-order", type=int, default=72)
    parser.add_argument("--G", type=float, default=.001)
    parser.add_argument("--quadrature", type=int, default=192)
    parser.add_argument("--q-projection", choices=("galerkin", "direct", "laurent"), default="galerkin",
                        help="ASR TM boundary trace: Galerkin preserves the reduced power metric; direct is the legacy projection")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backend", choices=("auto", "scipy", "torch"), default="auto", help="ASR backend; original Li always uses torch")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--diagnostics", action="store_true", help="Also calculate ASR boundary condition estimates")
    parser.add_argument("--tolerance", type=float, default=.005)
    parser.add_argument("--passivity-tolerance", type=float, default=1e-7)
    parser.add_argument("--output-root", type=Path, help="Use <root>/<study>/ instead of each study's results/li_vs_asr_1d/")
    parser.add_argument("--config", type=Path, help="Custom config when selecting a single study")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    if (args.reference_order <= max(args.orders) or not 0 < args.reference_check_order < args.reference_order
            or min(args.gold_slices, args.pmma_slices, args.slices or 1, args.threads) < 1
            or not math.isfinite(args.G) or args.G <= 0 or args.quadrature < 8
            or not 0 < args.tolerance < 1 or not 0 < args.passivity_tolerance < 1):
        parser.error("Invalid settings: reference order must exceed sweep maximum; counts and tolerances must be positive")
    if args.config and args.study == "all":
        parser.error("--config requires a single --study")
    from .adapters import STUDIES, load_study, scalar_case, existing_li_case
    from paper_reproductions.vallius2002.devices import resolve_execution
    selected = STUDIES if args.study == "all" else (args.study,)
    context = nullcontext()
    execution = {"requested_device": args.device, "requested_backend": args.backend}
    if not (args.prepare_only or args.report_only):
        import torch
        torch.set_num_threads(args.threads)
        try:
            from threadpoolctl import threadpool_limits
            context = threadpool_limits(limits=args.threads)
        except ImportError:
            print("threadpoolctl is unavailable; BLAS thread settings are inherited", flush=True)
        execution = resolve_execution(args.device, args.backend)
    outputs = {}
    asr_case, cleanup = scalar_case, None
    if execution.get("backend") == "torch" and not (args.prepare_only or args.report_only):
        from .streaming_gpu import scalar_case_streamed, release_unused_cuda
        asr_case, cleanup = scalar_case_streamed, release_unused_cuda
        print(f"ASR memory: one large prepared layer at a time on {execution['device']}; "
              "retaining production TE/TM modal traces only", flush=True)
    with context:
        for study in selected:
            config, material_model, config_path = load_study(study, args.config)
            waves = [float(w) for w in args.wavelengths.split(",")] if args.wavelengths else config["wavelengths_nm"]
            if not waves or any(not math.isfinite(w) or w <= 0 for w in waves) or any(a >= b for a, b in zip(waves, waves[1:])):
                parser.error("Wavelengths must be positive and strictly increasing")
            for wave in waves:
                material_model(wave)
            slices = args.slices or (args.gold_slices if study == "gold_grating_1d" else args.pmma_slices)
            output = ((args.output_root.resolve() / study) if args.output_root else
                      ROOT / "studies" / study / "results/li_vs_asr_1d")
            outputs[study] = output
            inputs, signature = fingerprint(study, config, args)
            samples = {}
            for wave in waves:
                material = material_model(wave)
                values = {"gold": material} if study == "gold_grating_1d" else dict(zip(("pmma", "gold"), material))
                samples[str(wave)] = {name: dict(real=complex(value).real, imag=complex(value).imag)
                                      for name, value in values.items()}
            plan = dict(study=study, geometry=config["geometry"], config_path=str(config_path),
                        orders=args.orders, asr_ratios=args.asr_ratios, slices=slices, wavelengths_nm=waves,
                        reference_order=args.reference_order, reference_check_order=args.reference_check_order,
                        G=args.G, quadrature_requested_minimum=args.quadrature, q_projection=args.q_projection,
                        tolerance=args.tolerance, passivity_tolerance=args.passivity_tolerance,
                        execution=execution, cpu_threads=args.threads, inputs=inputs,
                        material_interpolation=("existing TabulatedGold: linear epsilon" if study == "gold_grating_1d" else
                                                "existing Materials: linear n,k then square; PMMA k=0"),
                        material_note=getattr(material_model, "note", None), material_samples=samples)
            path = output / "checkpoint.json"
            if args.report_only and not path.exists():
                raise FileNotFoundError(f"No saved checkpoint for --report-only: {path}")
            if args.report_only and (output / "plan.json").exists():
                plan["execution"] = json.loads((output / "plan.json").read_text(encoding="utf-8"))["execution"]
            checkpoint = json.loads(path.read_text(encoding="utf-8")) if path.exists() else dict(signature=signature, inputs=inputs, cases={})
            if checkpoint["signature"] != signature:
                raise ValueError(f"Geometry/material/numerical sources/settings changed: use a new --output-root ({output})")
            save_json(output / "plan.json", plan)
            if args.prepare_only:
                print(f"Prepared: {output / 'plan.json'}", flush=True)
                continue
            if args.report_only:
                export(study, output, checkpoint, plan)
                continue
            requested = [("li", 1, n) for n in sorted(set(args.orders + [args.reference_check_order, args.reference_order]))]
            requested += [(f"asr_r{r}", r, n) for r in args.asr_ratios for n in args.orders]
            for series, ratio, order in requested:
                for wave in waves:
                    key = f"{series}|{slices}|{order}|{wave:.12g}"
                    if key in checkpoint["cases"] and not checkpoint["cases"][key].get("error"):
                        continue
                    print(f"{study}: {series} M={order} N={2*order+1} Nz={slices} lambda={wave:g} nm", flush=True)
                    try:
                        materials = material_model(wave)
                        if series == "li":
                            case = existing_li_case(study, config, slices, order, wave, materials, execution["device"])
                        else:
                            case = asr_case(study, config, slices, order, wave, materials, oversampling=ratio,
                                               G=args.G, quadrature=args.quadrature, device=execution["device"],
                                               backend=execution["backend"], diagnostics=args.diagnostics,
                                               q_projection=args.q_projection)
                        print(f"  saved in {case['runtime_seconds']:.2f} s: " +
                              "; ".join(f"{p} R={v['reflectance']:.7g}" for p, v in case["polarizations"].items()), flush=True)
                        memory = case.get("memory", {})
                        if "peak_allocated_bytes" in memory:
                            print(f"  GPU peak: allocated={memory['peak_allocated_bytes']/2**30:.3f} GiB; "
                                  f"reserved={memory['peak_reserved_bytes']/2**30:.3f} GiB; "
                                  f"modal traces={memory['retained_modal_bytes']/2**30:.3f} GiB", flush=True)
                    except (ArithmeticError, RuntimeError, ValueError) as exc:
                        case = dict(order=order, slices=slices, wavelength_nm=wave, method=series,
                                    error=f"{type(exc).__name__}: {exc}", polarizations={})
                        print(f"  calculation failed: {case['error']}", flush=True)
                    checkpoint["cases"][key] = case
                    save_json(path, checkpoint)
                    if cleanup is not None:
                        cleanup(execution["device"])
            export(study, output, checkpoint, plan)
        if len(selected) == 2 and not args.prepare_only:
            from .overview import plot_overview
            overview_output = (args.output_root.resolve() if args.output_root else
                               ROOT / "studies/asr_1d_comparison/results/li_vs_asr_1d")
            plot_overview(outputs, overview_output)
            print(f"Overview: {overview_output / 'convergence_overview.png'}", flush=True)


if __name__ == "__main__":
    main()
