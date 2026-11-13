"""Cross sections of all four physical conditions, generated from solver layers."""
from __future__ import annotations

from pathlib import Path
from model import geometry, layers, save_csv, save_json


def draw(ax, g, void, slices, letter=''):
    import matplotlib.patches as patches
    p, h, t = (g[k]/1000 for k in ['period_nm', 'height_nm', 'gold_vertical_nm'])
    unit=h/1.5
    pmma, water, air, gold = '#3889a7', '#d7eeef', '#fff8e9', '#d8a52e'
    z_water = h+t
    xmin, xmax = -.2, 2*p+.2
    ax.set_facecolor('#fbfcfd')
    ymax=max(2.75*unit,h+t+.7*unit)
    ax.add_patch(patches.Rectangle((xmin, -2.4*unit), xmax-xmin, ymax+2.4*unit, color=water, lw=0))
    ax.add_patch(patches.Rectangle((xmin, -2.4*unit), xmax-xmin, unit, color=air, lw=0))
    if void == 'air':
        ax.add_patch(patches.Rectangle((xmin, 0), xmax-xmin, z_water, color=air, lw=0))
    ax.add_patch(patches.Rectangle((xmin, -1.4*unit), xmax-xmin, 1.4*unit, color=pmma, lw=0))
    stack = layers(g, slices)
    for cell in range(2):
        center = (cell+.5)*p
        for row in stack:
            y = row['z_top_nm']/1000
            dz = (row['z_bottom_nm']-row['z_top_nm'])/1000
            outer, core = row['outer_width_nm']/1000, row['core_width_nm']/1000
            if t:
                ax.add_patch(patches.Rectangle((center-outer/2, y), outer, dz+1e-6, color=gold, lw=0))
            if core:
                ax.add_patch(patches.Rectangle((center-core/2, y), core, dz+1e-6, color=pmma, lw=0))
        for x in [cell*p, (cell+1)*p]:
            ax.plot([x, x], [-1.4*unit, z_water+.2*unit], color='#5f747c', alpha=.35, lw=.6, ls=':')
    if void == 'air':
        ax.axhline(z_water, color='#248d96', lw=1.3)
    # Schematic sheet break. Only the relief has a physical vertical scale.
    ax.axhline(-.85*unit, color='white', lw=1.4, ls=(0, (8, 4)))
    ax.text(p, -.47*unit, 'Thick PMMA sheet | phase averaged', ha='center', color='white', fontsize=9, weight='bold')
    ax.text(p, -1.65*unit, 'Incident air (n = 1)', ha='center', fontsize=9, color='#334b55')
    ax.annotate('', xy=(.10*p, -1.42*unit), xytext=(.10*p, -2.17*unit), arrowprops=dict(arrowstyle='-|>', color='#ac4b39', lw=1.6))
    ax.text(.16*p, -2.02*unit, 'Normal incidence', fontsize=8, color='#ac4b39')
    ax.text(p, z_water+.68*unit, 'Semi-infinite water (n = 1.333)', ha='center', fontsize=9, color='#21767d')
    if void == 'air':
        ax.text(p, .78*unit, 'Air void', ha='center', fontsize=8, color='#8c7449')
        ax.text(1.52*p, z_water+.4*unit, 'Flat water surface touches tips', ha='center', fontsize=7, color='#21767d')
    else:
        ax.text(p, .78*unit, 'Water-filled void', ha='center', fontsize=8, color='#21767d')
    base, tip = g['base_width_nm']/1000, g['tip_width_nm']/1000
    for a, b, y, label in [(p/2-base/2, p/2+base/2, -.08*unit, f'{base:.5g} um'),
                           (p/2-tip/2, p/2+tip/2, z_water+.3*unit, f'{tip:.5g} um')]:
        ax.annotate('', (a, y), (b, y), arrowprops=dict(arrowstyle='<->', lw=.8, color='#233e4b'))
        ax.text((a+b)/2, y-.08*unit if y < 0 else y+.21*unit, label, ha='center', fontsize=8, color='#233e4b')
    if g['gap_nm']:
        ax.text(p, -.13*unit, f"gap {g['gap_nm']/1000:g} um", ha='center', fontsize=7,
                bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=1))
    title = f"{letter} "+(f'Au {t*1000:g} nm' if t else 'Uncoated')+f' / {void} voids'
    ax.set_title(title, loc='left', fontsize=11, weight='bold', pad=8)
    ax.text(.01, .02, f"H = {h:g} um; x periodic, y invariant"+
            (f"; Au side-normal {g['gold_normal_nm']:.2f} nm" if t else ''),
            transform=ax.transAxes, fontsize=8, color='#38505c')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, -2.4*unit)
    ax.set_aspect('equal')
    ax.set_xlabel('x (um)', fontsize=9)
    ax.set_ylabel('z downward (um)', fontsize=9)
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#aebdc3')


def preview_all(output, periods=(3720., 3920.), base=None, tip=None, height=None, gold=50., slices=120,geometry_mode='scaled'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    conditions = [(0., 'air'), (0., 'water'), (gold, 'air'), (gold, 'water')]
    geometries = []
    for period in periods:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for i, ((t, void), ax) in enumerate(zip(conditions, axes.flat)):
            if geometry_mode=='scaled':g=geometry(period=period,gold=t)
            else:g=geometry(period,3720. if base is None else base,1500. if tip is None else tip,1500. if height is None else height,t)
            draw(ax, g, void, slices, chr(65+i)+'.')
            geometries.append(dict(g, void=void, actual_layers=len(layers(g, slices))))
            save_csv(output/f'layers_P{period:g}_Au{t:g}_{void}.csv', layers(g, slices))
        fig.suptitle(f"PMMA relief: P={period/1000:g} um; B={g['base_width_nm']/1000:.6g}; b={g['tip_width_nm']/1000:.6g}; H={g['height_nm']/1000:.6g} um ({geometry_mode})", fontsize=13, weight='bold')
        fig.text(.5, .012, 'Sheet thickness is schematic. Au deposition is from the sample side, opposite the optical incidence. All far-output media are water.',
                 ha='center', fontsize=8, color='#50616b')
        fig.tight_layout(rect=[0, .035, 1, .96])
        fig.savefig(output/f'conditions_P{period:g}.png', dpi=190)
        fig.savefig(output/f'conditions_P{period:g}.svg')
        plt.close(fig)
    save_json(output/'geometries.json', geometries)


if __name__ == '__main__':
    preview_all(Path(__file__).resolve().parent/'results/geometry')
