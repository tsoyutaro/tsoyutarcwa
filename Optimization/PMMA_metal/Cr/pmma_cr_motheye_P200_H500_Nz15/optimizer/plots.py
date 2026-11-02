"""Plot dual monotone radii, absorption optimization and PMMA-substrate R/T/A."""
from __future__ import annotations
import csv
import json
from pathlib import Path


def _read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def _spectrum(folder,best):
    pointer=folder/'latest_spectrum.json'
    if not pointer.exists():return None
    path=Path(_read(pointer)['report'])
    if not path.exists():return None
    report=_read(path)
    if report.get('source_profile_logits')!=best.get('logits'):return None
    outcomes=report.get('outcomes',[])
    return outcomes[0].get('rows') if outcomes else None


def make_plots(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from .postprocess import draw_profile
    from .geometry import radii_values
    c=_read(output/'resolved_config.json')
    destination=output/'figures';destination.mkdir(exist_ok=True)
    for lattice in c['lattices']:
        folder=output/lattice
        if not (folder/'report.json').exists():continue
        report=_read(folder/'report.json');best=report.get('best')
        if not best:continue
        fig,axes=plt.subplots(2,2,figsize=(12,9))
        ax=axes[0,0]
        if (folder/'trials.csv').exists():
            with (folder/'trials.csv').open(encoding='utf-8-sig',newline='') as f:trials=list(csv.DictReader(f))
            finished=[r for r in trials if r['state']=='COMPLETE' and r.get('objective')]
            if finished:
                x=[int(r['trial_number']) for r in finished]
                y=[100*(1-float(r['objective'])) for r in finished]
                ax.scatter(x,y,label='Complete trial',alpha=.7)
                ax.step(x,np.maximum.accumulate(y),where='post',color='#d56318',label='Best completed')
        ax.set(xlabel='Optuna trial number',ylabel='Training mean A (%)')
        if ax.get_legend_handles_labels()[0]:ax.legend()
        ax=axes[0,1];history_path=Path(best['trial_folder'])/'history.csv'
        if history_path.exists():
            with history_path.open(encoding='utf-8-sig',newline='') as f:history=list(csv.DictReader(f))
            ax.plot([int(r['step']) for r in history],[100*(1-float(r['objective'])) for r in history],label='Measured iterate')
            ax.plot([int(r['step']) for r in history],[100*(1-float(r['best_objective'])) for r in history],label='Best measured',color='#d56318')
            selected_path=Path(best['trial_folder'])/'best.json'
            if selected_path.exists():
                selected=_read(selected_path)
                ax.scatter([selected['best_step']],[100*(1-selected['best_loss'])],marker='*',s=110,
                           color='#d56318',label='Selected best state',zorder=5)
            ax.legend()
        ax.set(xlabel='Saved Adam step',ylabel=f"Mean A at M={c['optimization_order']} (%)")
        ax=axes[1,0];rows=_spectrum(folder,best)
        if rows:
            waves=[r['wavelength_nm'] for r in rows]
            for key,label in [('reflectance','R'),('transmittance','T into PMMA'),('absorptance_total','A = 1 - R - T')]:
                ax.plot(waves,[100*r[key] for r in rows],label=label)
            ax.legend(fontsize=8);ax.set(ylabel='Power (%)')
        else:
            verification=report.get('verification',{})
            for condition in verification.get('best',{}).get('conditions',[]):
                values=sorted(condition['cases'].values(),key=lambda r:r['wavelength_nm'])
                values=[r for r in values if r['wavelength_nm'] in c['verification_wavelengths_nm']]
                label=f"Best A: M={condition['order']}, Q={condition['grid']}"
                if not condition.get('complete'):label+=' (partial)'
                ax.plot([r['wavelength_nm'] for r in values],[100*r['absorptance_total'] for r in values],label=label)
            if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=8)
            else:ax.text(.5,.5,'Run spectrum mode to obtain R / T / A',ha='center',va='center',transform=ax.transAxes)
            ax.set(ylabel='Absorptance A (%)')
        ax.set(xlabel='Wavelength (nm)')
        radii=best.get('radii_nm')
        if radii is None:radii=radii_values(best['logits'],c)
        draw_profile(axes[1,1],radii,c)
        for axis in axes.flat:axis.grid(alpha=.2)
        fig.suptitle(f"PMMA core / {c.get('material','metal')} shell; {lattice}; Nz={c['layers']}; maximize mean A; trial {best['trial_number']}")
        fig.tight_layout()
        for extension in ('png','svg'):fig.savefig(destination/f'optimization_{lattice}.{extension}',dpi=180)
        plt.close(fig)
