"""Plot saved Li/ASR powers versus retained order; never run an RCWA solve.

From the project root:
    python -m studies.asr_1d_comparison.plot_absolute_powers
Requires the existing NumPy/Matplotlib environment; CUDA is not needed.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path.cwd())
    parser.add_argument('--output-root', type=Path,
                        default=Path('studies/asr_1d_comparison/results/absolute_powers'))
    parser.add_argument('--study', choices=('all','gold_grating_1d','pmma_gold_grating_1d'), default='all')
    parser.add_argument('--scale', choices=('fixed','zoom','both'), default='both')
    return parser.parse_args()


def hash_matches(path, expected):
    raw = path.read_bytes()
    lf = raw.replace(b'\r\n', b'\n')
    return expected in {hashlib.sha256(value).hexdigest() for value in (raw,lf,lf.replace(b'\n',b'\r\n'))}


def load_saved(root, relative, verified):
    directory = root / relative
    checkpoint = json.loads((directory/'checkpoint.json').read_bytes())
    plan = json.loads((directory/'plan.json').read_bytes())
    assert checkpoint['inputs'] == plan['inputs']
    actual = hashlib.sha256(json.dumps(checkpoint['inputs'],sort_keys=True).encode()).hexdigest()
    assert actual == checkpoint['signature']
    for name, expected in checkpoint['inputs']['source_sha256'].items():
        assert hash_matches(root/name,expected), name
        verified[name] = expected
    for name, expected in checkpoint['inputs']['material_sha256'].items():
        assert hash_matches(root/'studies'/plan['study']/'data'/name,expected), name
    return checkpoint,plan,directory


def record(case, series, pol, plan, directory):
    assert not case.get('error')
    observed = case['polarizations'][pol]
    r = observed['reflectance']
    if plan['study'] == 'gold_grating_1d':
        t = 0.0  # Far-field transmission through a semi-infinite absorbing Au substrate.
        a = observed['absorptance_total']
        assert math.isclose(a,1-r,abs_tol=2e-15,rel_tol=0)
        note = 'T_far=0; A_total includes relief and semi-infinite Au substrate'
    else:
        t,a = observed['transmittance'],observed['absorptance']
        note = 'T is port power into lossless PMMA substrate; A=1-R-T'
    assert all(math.isfinite(value) and -1e-7 <= value <= 1+1e-7 for value in (r,t,a))
    assert math.isclose(r+t+a,1,abs_tol=2e-15,rel_tol=0)
    return dict(study=plan['study'],series=series,order=case['order'],harmonics=2*case['order']+1,
                wavelength_nm=case['wavelength_nm'],polarization=pol,R=r,T=t,A=a,
                internal_ratio=case.get('oversampling',1),G=case.get('G'),
                quadrature_actual=case.get('quadrature_actual',0),slices=case['slices'],
                source=str(directory/'checkpoint.json'),definition=note)


def draw(rows,study,wave,series,scope,output,scale):
    fig,axes = plt.subplots(2,3,figsize=(12,7.3))
    styles = {'li':('#1864ab','o','-'), 'asr_r4':('#d9480f','s','--'),
              'asr_r6':('#d9480f','s','--'), 'asr_r7':('#2b8a3e','^',':')}
    labels = {'li':'Cartesian Li', 'asr_r4':'ASR 4N', 'asr_r6':'ASR 6N (M48,64 only)',
              'asr_r7':'ASR 7N (M48,64 only)'}
    titles = ['Reflectance R','Transmittance T','Absorptance A']
    if study == 'gold_grating_1d':
        titles = ['Reflectance R','Far-field transmittance T_far = 0','Total absorptance A_total']
    for ip,pol in enumerate(('TE','TM')):
        for im,metric in enumerate(('R','T','A')):
            ax = axes[ip,im]
            all_values = []
            for method in series:
                points = sorted((row for row in rows if row['series'] == method and row['polarization'] == pol
                                 and row['wavelength_nm'] == wave),key=lambda row:row['order'])
                assert points
                xs,ys = [row['order'] for row in points],[row[metric] for row in points]
                color,marker,ls = styles[method]
                ax.plot(xs,ys,color=color,marker=marker,ls=ls,ms=4,lw=1.4,label=labels[method])
                all_values.extend(ys)
            if scale == 'fixed':
                ax.set_ylim(-.02,1.02)
                ax.set_yticks([0,.2,.4,.6,.8,1])
            else:
                low,high = min(all_values),max(all_values)
                if high-low < 1e-14:
                    ax.set_ylim((0,.02) if high < .02 else (max(0,low-.02),min(1,high+.02)))
                else:
                    pad = max((high-low)*.12,1e-12)
                    ax.set_ylim(max(0,low-pad),min(1,high+pad))
                ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3),useOffset=False)
            ax.set_title(f'{pol}: {titles[im]}',fontsize=10)
            ax.set_xlabel('Maximum retained Fourier order M (N=2M+1)',fontsize=9)
            ax.set_ylabel(metric,fontsize=10)
            ax.grid(alpha=.25)
            if ip == 0 and im == 0:
                handles,legend = ax.get_legend_handles_labels()
    name = 'Au-coated PMMA grating' if study == 'pmma_gold_grating_1d' else 'Au grating on semi-infinite Au'
    axis_note = 'Incident-power fractions: 1 = 100%; full scale 0 to 1' if scale == 'fixed' else 'Incident-power fractions: 1 = 100%; linear zoom of actual powers'
    fig.suptitle(f'{name} / {wave:g} nm / normal incidence\n{scope}\n{axis_note}',fontsize=12)
    fig.legend(handles,legend,loc='lower center',ncol=len(series),fontsize=10,bbox_to_anchor=(.5,.015))
    fig.tight_layout(rect=(.01,.09,.99,.83),h_pad=2,w_pad=1.5)
    suffix = '' if scale == 'fixed' else '_zoom'
    target = output/f'powers_{wave:g}nm{suffix}'
    fig.savefig(target.with_suffix('.png'),dpi=170)
    fig.savefig(target.with_suffix('.svg'))
    plt.close(fig)
    return target.with_suffix('.png')


def main():
    args = arguments()
    root = args.project_root.resolve()
    output = (args.output_root if args.output_root.is_absolute() else root/args.output_root).resolve()
    output.mkdir(parents=True,exist_ok=True)
    selected = ('gold_grating_1d','pmma_gold_grating_1d') if args.study == 'all' else (args.study,)
    verified = {}
    all_rows,figures,sources = [],[],[]
    scales = ('fixed','zoom') if args.scale == 'both' else (args.scale,)
    old_data = {}
    for study in selected:
        cp,plan,directory = load_saved(root,f'studies/asr_1d_comparison/results/gpu_galerkin/{study}',verified)
        assert plan['asr_ratios'] == [4] and plan['G'] == .001 and plan['q_projection'] == 'galerkin'
        rows = [record(cp['cases'][f'{series}|{plan["slices"]}|{order}|{wave:g}'],series,pol,plan,directory)
                for series in ('li','asr_r4') for order in plan['orders']
                for wave in plan['wavelengths_nm'] for pol in ('TE','TM')]
        old_data[study] = (cp,plan,rows)
        all_rows.extend(dict(dataset='original_4N_order_sweep',**row) for row in rows)
        sources.append(str(directory))
        target = output/study
        target.mkdir(exist_ok=True)
        for wave in plan['wavelengths_nm']:
            for scale in scales:
                figures.append(str(draw(rows,study,wave,('li','asr_r4'),
                    f'Nz={plan["slices"]}; G=0.001; Galerkin; ASR 4N; quadrature minimum=192 (actual varies with M)',target,scale)))
        print(f'{study}: {len(rows)} plotted power records; {len(plan["wavelengths_nm"])} wavelengths',flush=True)
    latest_available = False
    if 'pmma_gold_grating_1d' in selected:
        latest_dirs = {m:f'studies/asr_1d_comparison/results/fixed_order_audit_M{m}_r'+('5to7' if m == 48 else '6to7')+'/internal_q4096/pmma_gold_grating_1d'
                       for m in (48,64)}
        if all((root/path/'checkpoint.json').is_file() for path in latest_dirs.values()):
            latest_available = True
            old_cp,old_plan,old_rows = old_data['pmma_gold_grating_1d']
            indexed = {(row['series'],row['order'],row['wavelength_nm'],row['polarization']):row
                       for row in old_rows if row['series'] == 'li' and row['wavelength_nm'] in (650,700)}
            for m,path in latest_dirs.items():
                cp,plan,directory = load_saved(root,path,verified)
                expected_inputs = dict(old_cp['inputs'],quadrature_minimum=4096)
                assert cp['inputs'] == expected_inputs
                assert plan['slices'] == 300 and plan['orders'] == [m]
                sources.append(str(directory))
                for key,case in cp['cases'].items():
                    method = key.split('|')[0]
                    if method not in ('li','asr_r6','asr_r7'):
                        continue
                    for pol in ('TE','TM'):
                        row = record(case,method,pol,plan,directory)
                        index = (method,row['order'],row['wavelength_nm'],pol)
                        if index in indexed:
                            assert all(math.isclose(row[metric],indexed[index][metric],rel_tol=0,abs_tol=2e-13) for metric in ('R','T','A'))
                        indexed[index] = row
            rows = list(indexed.values())
            assert all({row['order'] for row in rows if row['series'] == method} == {48,64} for method in ('asr_r6','asr_r7'))
            all_rows.extend(dict(dataset='latest_6N_7N_saved_points',**row) for row in rows)
            target = output/'pmma_latest_6N_7N'
            target.mkdir(exist_ok=True)
            for wave in (650,700):
                for scale in scales:
                    figures.append(str(draw(rows,'pmma_gold_grating_1d',wave,('li','asr_r6','asr_r7'),
                        'Nz=300; G=0.001; Galerkin; ASR Q4096 / Li uses analytic coefficients',target,scale)))
            print(f'Latest PMMA: {len(rows)} plotted power records; ASR 6N/7N at M48,64 only',flush=True)
    with (output/'plotted_values.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    manifest = dict(project_root=str(root),source_directories=sources,source_files_verified=len(verified),
                    plotted_power_records=len(all_rows),figures_png=figures,latest_6N_7N_available=latest_available,
                    axis='Actual dimensionless powers; no differences, no reference subtraction, no averaging over wavelengths',
                    gold_definition='R, far-field T=0, total A=1-R; P_sub is not far-field transmittance',
                    pmma_definition='R, T into lossless PMMA, A=1-R-T',
                    absorption_caveat='A is a port-power residual, not an independent volume integral')
    (output/'figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Finished: {len(figures)} PNG and SVG figure pairs in {output}',flush=True)


if __name__ == '__main__':
    main()
