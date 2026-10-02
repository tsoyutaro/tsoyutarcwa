"""Audit the saved Adam profile's triangular ASR map, without an optical solve.

The default backend calls the production PyTorch map. The explicit
numpy-reference backend independently evaluates its formula and derivatives;
it is a diagnostic reference, not an RCWA solver.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def numpy_reference(n, radius, slope):
    import numpy as np

    axis = np.arange(n, dtype=np.float64) / n
    u, v = np.meshgrid(axis, axis, indexing="ij")
    choices = [(u-.5-i, v-.5-j) for i in (-1, 0, 1) for j in (-1, 0, 1)]
    which = np.argmin(np.stack([a*a+b*b+a*b for a,b in choices]), axis=0)
    q1 = np.take_along_axis(np.stack([a for a,b in choices]), which[None], axis=0)[0]
    q2 = np.take_along_axis(np.stack([b for a,b in choices]), which[None], axis=0)[0]
    terms = [q1+.5*q2, .5*q1+q2, .5*(q1-q2)]
    derivatives = [(1., .5), (.5, 1.), (.5, -.5)]
    values = [np.abs(t) for t in terms]
    gradients = [np.stack([np.sign(t)*a, np.sign(t)*b])
                 for t,(a,b) in zip(terms, derivatives)]

    # Independently evaluate the symmetric sector-boundary subgradient.
    # The coordinates still use the exact support maximum.
    h = np.maximum.reduce(values)
    ties = h[None]-np.stack(values) <= 64*np.finfo(np.float64).eps
    weights = ties/ties.sum(axis=0, keepdims=True)
    dh = np.sum(weights[:,None]*np.stack(gradients), axis=0)
    floor = 64*np.finfo(np.float64).eps
    norm = np.sqrt(np.maximum(q1*q1+q2*q2+q1*q2, floor**2))
    dn = np.stack([(q1+.5*q2)/norm, (q2+.5*q1)/norm])
    angle, rho = h/norm, h/radius
    da = dh/norm-h*dn/(norm*norm)
    f_inner = ((rho*rho-2*rho+1)+(-2*rho*rho+3*rho)*angle
               +(rho*rho-rho)*slope)
    df_inner = ((2*rho-2+(-4*rho+3)*angle+(2*rho-1)*slope)*dh/radius
                +(-2*rho*rho+3*rho)*da)
    outer, span = .5/radius, .5/radius-1
    t = np.clip((rho-1)/span, 0, 1)
    h00, h10 = 2*t**3-3*t*t+1, t**3-2*t*t+t
    h01, h11 = -2*t**3+3*t*t, t**3-t*t
    radial = h00*angle+h10*span*slope+h01*outer+h11*span
    drho = ((6*t*t-6*t)*angle+(3*t*t-4*t+1)*span*slope
            +(-6*t*t+6*t)*outer+(3*t*t-2*t)*span)/span
    safe = np.maximum(rho, floor)
    f_outer = radial/safe
    df_outer = (drho*safe-radial)/(safe*safe)*dh/radius+h00/safe*da
    f = np.where(rho<=1, f_inner, np.where(rho<outer, f_outer, 1))
    df = np.where((rho<=1)[None], df_inner, np.where((rho<outer)[None], df_outer, 0))
    f, df = np.where(rho>floor, f, 1), np.where((rho>floor)[None], df, 0)
    a11, a12 = f+q1*df[0], q1*df[1]
    a21, a22 = q2*df[0], f+q2*df[1]
    sine, cosine = math.sqrt(3)/2, .5
    jacobian = np.stack([
        np.stack([a11+cosine*a21,
                  (-cosine*(a11+cosine*a21)+a12+cosine*a22)/sine]),
        np.stack([sine*a21, a22-cosine*a21])])
    displacement = np.stack([(f-1)*(q1+.5*q2), (f-1)*sine*q2])
    return displacement, jacobian


def production_map(simulation, n, radius):
    import numpy as np

    mapping = simulation.build_triangular_circle_asr_mapping(n, n, radius)
    arrays = {name: getattr(mapping, name).detach().cpu().numpy()
              for name in ("u", "v", "x", "y", "x_u", "x_v", "y_u", "y_v")}
    u, v = np.meshgrid(arrays["u"], arrays["v"], indexing="ij")
    sine = math.sqrt(3)/2
    displacement = np.stack([arrays["x"]-(u+.5*v), arrays["y"]-sine*v])
    # The map provides Cartesian output / primitive input derivatives.
    primitive = np.stack([
        np.stack([arrays["x_u"], arrays["x_v"]]),
        np.stack([arrays["y_u"], arrays["y_v"]])])
    inverse_cell = np.array([[1., -.5/sine], [0., 1./sine]])
    jacobian = np.einsum("abij,bc->acij", primitive, inverse_cell)
    del mapping
    return displacement, jacobian


def analyze(displacement, jacobian, tolerance):
    import numpy as np

    if not np.isfinite(displacement).all() or not np.isfinite(jacobian).all():
        raise RuntimeError("Nonfinite map or Jacobian.")
    n = displacement.shape[-1]
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    sine = math.sqrt(3)/2
    symmetries = [
        ("rotation60", (-j)%n, (i+j-n//2)%n,
         np.array([[.5, -sine], [sine, .5]])),
        ("reflection_x", (i+j-n//2)%n, (-j)%n,
         np.array([[1., 0.], [0., -1.]]))]
    jacobian_norm = max(float(np.linalg.norm(jacobian)), np.finfo(float).tiny)
    jacobian_peak = max(float(np.linalg.norm(jacobian, axis=(0,1)).max()), 1.)
    determinant = np.linalg.det(jacobian.transpose(2,3,0,1))
    result = {"cartesian_jacobian_det_min": float(determinant.min()),
              "cartesian_jacobian_det_max": float(determinant.max())}
    for name, ii, jj, transform in symmetries:
        map_expected = np.einsum("ab,bij->aij", transform, displacement)
        map_error = np.linalg.norm(displacement[:,ii,jj]-map_expected, axis=0)
        expected = np.einsum("ab,bcij,dc->adij", transform, jacobian, transform)
        difference = jacobian[:,:,ii,jj]-expected
        error = np.linalg.norm(difference, axis=(0,1))
        maximum = float(error.max())
        result[name] = {
            "map_max_absolute_error_period_units": float(map_error.max()),
            "jacobian_max_absolute_error": maximum,
            "jacobian_relative_max_error": maximum/jacobian_peak,
            "jacobian_relative_frobenius_error": float(np.linalg.norm(difference))/jacobian_norm,
            "jacobian_samples_above_absolute_tolerance": int(np.count_nonzero(error>tolerance)),
            "jacobian_consistent_at_relative_tolerance": maximum/jacobian_peak<=tolerance}
    return result


def main():
    location = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=location if (location/"rcwa_solver_auto.py").is_file() else Path.cwd())
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--backend", choices=("torch", "numpy-reference"), default="torch")
    parser.add_argument("--grids", default="256,384,512")
    parser.add_argument("--slices", type=int, default=140)
    parser.add_argument("--order", type=int, default=18)
    parser.add_argument("--layers", default="1,middle,last", help="One-based indices or middle,last")
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root/"rcwa_solver_auto.py").is_file():
        parser.error("Pass the tsoyutarcwa project root with --root.")
    if args.slices<1 or args.order<1 or not math.isfinite(args.tolerance) or args.tolerance<=0:
        parser.error("Invalid slices/order/tolerance.")
    try:
        grids = sorted(set(int(item) for item in args.grids.split(",")))
        aliases = {"middle": (args.slices+1)//2, "last": args.slices}
        layers = sorted(set(aliases[item.strip()] if item.strip() in aliases else int(item)
                            for item in args.layers.split(",")))
    except ValueError:
        parser.error("Invalid grid or layer list.")
    if not grids or min(grids)<max(32,4*args.order+4) or any(n%2 for n in grids):
        parser.error("Use even grids >= max(32, 4*order+4); rotation must preserve the grid.")
    if not layers or layers[0]<1 or layers[-1]>args.slices:
        parser.error("Layer indices must be between 1 and slices.")
    sys.path.insert(0, str(root))
    from studies.gold_motheye2 import optimize_adam as shared
    run = (args.run_dir or root/"studies/gold_motheye2/results/adam_fullband_Nz100_M8").resolve()
    output = (args.output_dir or root/"studies/gold_motheye2/results/adam_mapping_grid_diagnostics").resolve()
    if output==run:
        parser.error("Use a separate output directory.")
    config = json.loads((run/"config.json").read_text(encoding="utf-8"))
    training = json.loads((run/"checkpoint.json").read_text(encoding="utf-8"))
    if config["geometry"]["lattice"]!="triangular" or training.get("best") is None:
        parser.error("Requires a saved best profile on a triangular lattice.")
    profiles = {"best": shared.radius_values(training["best"]["logits"],args.slices,config),
                "cone": shared.radius_values(shared.initial_logits(config),args.slices,config)}
    period = config["geometry"]["period_nm"]
    planned = [{"label":label,"grid":n,"layer":layer,"radius_nm":profiles[label][layer-1]}
               for label,n,layer in itertools.product(("best","cone"),grids,layers)]
    identity = {"backend":args.backend,"device":args.device,"order":args.order,
                "slices":args.slices,"period_nm":period,
                "circle_G":config["geometry"]["asr_circle_g"],"tolerance":args.tolerance,
                "training_signature":training["signature"],"best_logits":training["best"]["logits"],
                "map_source_sha256_lf":digest(root/"rcwa_ext/asr_maps.py"),
                "diagnostic_source_sha256_lf":digest(Path(__file__))}
    output.mkdir(parents=True,exist_ok=True)
    path = output/"mapping_diagnostics.json"
    document = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"identity":identity,"cases":{}}
    if document.get("identity")!=identity:
        raise RuntimeError("Diagnostic settings/source/profile differ. Use another --output-dir.")
    key = lambda row: f"{row['label']}|grid={row['grid']}|layer={row['layer']}"
    missing = [row for row in planned if key(row) not in document["cases"]]
    def persist():
        shared.write_json(path,document)
        shared.write_json(output/"mapping_plan.json",
                          {"requested_cases":len(planned),"missing_cases":[key(row) for row in missing
                           if key(row) not in document["cases"]],"cases":planned})
    persist()
    print(f"ASR map audit: backend={args.backend}, {len(planned)} cases; {len(missing)} new maps.",flush=True)
    print("No optical eigensolve, modal conversion or Adam update is performed.",flush=True)
    if args.prepare_only or not missing:
        return 0
    simulation = None
    if args.backend=="torch":
        import torch
        from rcwa_solver_auto import ASROptions,AutoRCWA,Lattice,OutputSpec
        device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device
        if device=="cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable.")
        simulation = AutoRCWA(freq=1.,order=[args.order,args.order],lattice=Lattice.triangular(1.),
                              asr=ASROptions(circle_G=identity["circle_G"]),outputs=OutputSpec(fields="none"),
                              dtype=torch.complex128,device=torch.device(device))
    else:
        print("Independent NumPy reference only; this does not run the production PyTorch map.",flush=True)
    for row in missing:
        radius = row["radius_nm"]/period
        if not 0<radius<.5:
            raise RuntimeError("Radius must be inside (0, period/2).")
        if args.backend=="torch":
            with torch.enable_grad():
                displacement,jacobian = production_map(simulation,row["grid"],radius)
        else:
            displacement,jacobian = numpy_reference(row["grid"],radius,identity["circle_G"])
        result = {**row,"metrics":analyze(displacement,jacobian,args.tolerance)}
        document["cases"][key(row)] = result
        persist()
        rotation = result["metrics"]["rotation60"]
        print(f"{key(row)} R={row['radius_nm']:.6f} nm; "
              f"rotation J max error={rotation['jacobian_max_absolute_error']:.6g}; "
              f"bad samples={rotation['jacobian_samples_above_absolute_tolerance']}",flush=True)
        del displacement,jacobian
        gc.collect()
    print(f"Saved: {path}",flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
