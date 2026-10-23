"""Check layer omission, independent slab physics and optional legacy parity."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import numpy as np
import run_spectrum as model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy-script', type=Path)
    p.add_argument('--output-dir', type=Path, default=model.HERE/'results/no_glass_validation')
    opts = p.parse_args()
    args = model.parser().parse_args([])
    sys.path.insert(0, str(args.solver_root))
    import torch
    from rcwa_ext import AutoRCWA, Lattice, OutputSpec
    torch.set_num_threads(2)
    checks = []
    def check(name, error, tolerance=1e-10):
        assert math.isfinite(error) and error <= tolerance, (name,error,tolerance)
        checks.append(dict(name=name,error=float(error),tolerance=tolerance,passed=True))
        print(f'PASS {name}: {error:.3g}',flush=True)

    pet, _ = model.load_nk(model.HERE/'data/PET_Zhang_400_800nm.csv')
    adh = lambda wl: 1.5+0j
    args.glass_um = 0.
    layers = model.bulk_layers(args,600.,pet,adh)
    assert len(layers)==2 and layers[0][1]==54800. and layers[1][1]==20000.
    assert 'silica' not in model.stack_description(args)
    check('Glass omitted from both stack and phase variables',0.)
    n=pet(600.)
    surface=model.interface(1.,n)
    omitted=model.finite_sample(surface,n,layers,600.,16)
    zero_coherent=model.finite_sample(surface,n,layers+[(model.silica_n(600.),0.,False)],600.,16)
    check('Omitting glass equals the independent coherent zero-thickness limit',
          max(abs(a-b) for a,b in zip(omitted,zero_coherent)))
    args.adhesive_um=0.
    layers=model.bulk_layers(args,600.,pet,adh)
    assert len(layers)==1 and 'adhesive' not in model.stack_description(args)
    check('Bare PET option omits adhesive and its phase variable',0.)
    n=1.57
    r=((1-n)/(1+n))**2
    value=model.finite_sample(model.interface(1.,n),n,[(n,54800.,True)],600.,16)
    expected=(2*r/(1+r),(1-r)/(1+r),0.)
    check('Air/PET/air agrees with exact lossless incoherent slab',
          max(abs(a-b) for a,b in zip(value,expected)))

    args.glass_um=1000.; args.adhesive_um=20.
    expected=[(pet(600.),54800.,True),(adh(600.),20000.,True),(model.silica_n(600.),1e6,True)]
    assert model.bulk_layers(args,600.,pet,adh)==expected
    check('Default glass-mounted bulk stack is unchanged',0.)

    # No-glass flat stack versus a separately assembled homogeneous RCWA.
    args.glass_um=0.
    layers=model.bulk_layers(args,600.,pet,adh)
    value=model.finite_sample(model.interface(1.,pet(600.)),pet(600.),layers,600.,1)
    sim=AutoRCWA(freq=1/600.,order=[1,1],lattice=Lattice.square(100.),
        group_theory=False,outputs=OutputSpec(smatrix_size='full',fields='none'),
        dtype=torch.complex128,device='cpu')
    sim.add_input_layer(eps=1.); sim.add_output_layer(eps=1.)
    sim.set_incident_angle(0.,0.)
    with torch.no_grad():
        for index,d,_ in layers:
            sim.add_layer(d,eps=index**2)
        sim.solve_global_smatrix()
    i=sim.order_N//2
    reference=(float(abs(sim.S[1][i,i])**2),float(abs(sim.S[0][i,i])**2))
    check('No-glass coherent PET/adhesive stack versus homogeneous RCWA',
          max(abs(a-b) for a,b in zip(value[:2],reference)),1e-9)
    args.order,args.slices,args.grid=2,12,64
    for adhesive_um in [20.,0.]:
        args.adhesive_um=adhesive_um
        s=model.surface_rcwa(args,600.,pet(600.),n_substrate=pet(600.))
        layers=model.bulk_layers(args,600.,pet,adh)
        v=model.finite_sample(s,pet(600.),layers,600.,16)
        assert all(-1e-8<=a<=1+1e-8 for a in v)
        check(f'Patterned no-glass sample passive, adhesive={adhesive_um:g}um',0.)
        doubled=model.finite_sample(s,pet(600.),layers,600.,32)
        check(f'No-glass phase average 16 versus 32, adhesive={adhesive_um:g}um',
              max(abs(a-b) for a,b in zip(v,doubled)))

    if opts.legacy_script:
        spec=importlib.util.spec_from_file_location('legacy_uncoated',opts.legacy_script)
        old=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        args.adhesive_um,args.glass_um=20.,1000.
        for wl in [400.,600.,800.]:
            s1=model.surface_rcwa(args,wl,pet(wl),n_substrate=pet(wl))
            s2=old.surface_rcwa(args,wl,pet(wl),n_substrate=pet(wl))
            check(f'Unchanged surface complex amplitudes versus legacy at {wl:g}nm',
                  max(abs(a-b) for a,b in zip(s1,s2)),1e-9)
            original=[(pet(wl),54800.,True),(adh(wl),20000.,True),(old.silica_n(wl),1e6,True)]
            v1=model.finite_sample(s1,pet(wl),model.bulk_layers(args,wl,pet,adh),wl,16)
            v2=old.finite_sample(s2,pet(wl),original,wl,16)
            check(f'Unchanged glass-mounted R/T/A versus legacy at {wl:g}nm',
                  max(abs(a-b) for a,b in zip(v1,v2)),1e-9)
    opts.output_dir.mkdir(parents=True,exist_ok=True)
    (opts.output_dir/'checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(f'All {len(checks)} checks passed.',flush=True)


if __name__=='__main__':
    main()
