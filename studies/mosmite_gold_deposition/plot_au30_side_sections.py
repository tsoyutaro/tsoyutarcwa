"""Draw continuous cross sections for the Au30 radial-sidewall sweep."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch, Polygon, Rectangle

from geometry import build_layers, cap_outer_radius, core_radius, side_thickness
from model import parser as model_parser, validate_args
from run_au30_side_sweep import side_nm


HERE = Path(__file__).resolve().parent
AIR, RESIN, GOLD = '#f6f9fb', '#8fb6d1', '#d4a12d'


def dimension(ax, start, end, label, dx=0, dy=0, rotation=0):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='<->', lw=.9, color='#344258'))
    ax.text((start[0]+end[0])/2+dx, (start[1]+end[1])/2+dy, label,
            ha='center', va='center', rotation=rotation, fontsize=9,
            bbox=dict(facecolor='white', edgecolor='none', alpha=.94, pad=1))


def draw_materials(ax, args, clip_cell=False):
    """Union all gold first, then restore every original resin core."""
    p, h = args.period_nm, args.height_nm
    z_top, side = h+args.top_gold_nm, side_thickness(args)
    rb, rt = core_radius(args, 0)+side, core_radius(args, h)+side
    upper_radius = cap_outer_radius(args, z_top)
    ax.set_facecolor(AIR)
    patches = [Rectangle((-2*p, -32), 4*p, 32, facecolor=RESIN, edgecolor='none'),
               Rectangle((-2*p, 0), 4*p, args.valley_gold_nm, facecolor=GOLD, edgecolor='none')]
    centers = (-2*p, -p, 0, p, 2*p)
    for center in centers:
        patches.append(Polygon([
            (center-rb, 0), (center-rt, h), (center-upper_radius, z_top),
            (center+upper_radius, z_top), (center+rt, h), (center+rb, 0),
        ], facecolor=GOLD, edgecolor='none'))
    # Resin priority is the same in material_codes and in these polygons.
    for center in centers:
        patches.append(Polygon([
            (center-args.base_diameter_nm/2, 0), (center-args.tip_diameter_nm/2, h),
            (center+args.tip_diameter_nm/2, h), (center+args.base_diameter_nm/2, 0),
        ], facecolor=RESIN, edgecolor='#638da8', lw=.65))
    for patch in patches:
        ax.add_patch(patch)
        if clip_cell:
            patch.set_clip_path(Rectangle((-p/2, -32), p, z_top+64, transform=ax.transData))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--side-levels', nargs='+', type=side_nm, default=[10., 6., 4.])
    parser.add_argument('--slices', type=int, default=150)
    parser.add_argument('--output-dir', type=Path,
                        default=HERE/'results/Au30_PET5_glass1mm_side_sweep/sections')
    opts = parser.parse_args()
    if len(opts.side_levels) != 3 or len(set(opts.side_levels)) != 3:
        parser.error('Choose exactly three distinct side thicknesses for this three-panel figure.')
    opts.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': ['Yu Gothic', 'DejaVu Sans'], 'font.size': 10.5,
                         'axes.unicode_minus': False, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(13.5, 10.3), facecolor='white')
    grid = GridSpec(2, 3, figure=fig, left=.065, right=.97, top=.82, bottom=.21,
                    height_ratios=[1.85, 1], hspace=.46, wspace=.31)
    side_label = '・'.join(f'{side:g}' for side in opts.side_levels)
    fig.text(.065, .958, f'MOSMITE／Au 30 nm — 斜面{side_label} nmの断面', fontsize=19, weight='bold')
    fig.text(.065, .914, '頂部・谷部は30 nm。斜面の厚さは半径方向。外側斜面を頂部の金膜まで同じ傾きで延長。',
             fontsize=11.5, color='#334155')
    summaries = []
    for column, side in enumerate(opts.side_levels):
        args = model_parser('evaporation').parse_args([
            '--side-gold-nm', str(side), '--cap-profile', 'side-continuation',
            '--slices', str(opts.slices), '--film-total-um', '5.2', '--adhesive-um', '0',
            '--native-d6', '--order', '36', '--grid', '512',
        ])
        validate_args(args)
        layers = build_layers(args)
        top_radius = cap_outer_radius(args, 230)
        slope = (args.base_diameter_nm-args.tip_diameter_nm)/(2*args.height_nm)
        shell_contact = (args.base_diameter_nm/2+side-args.period_nm/2)/slope
        contact_top = max(args.valley_gold_nm, min(args.height_nm, shell_contact))
        summary = {
            'side_radial_nm': side, 'Au_top_nm': 30, 'Au_valley_nm': 30,
            'outer_diameter_at_z200_nm': 2*cap_outer_radius(args, 200),
            'outer_diameter_at_z230_nm': 2*top_radius,
            'centerline_gold_contact_top_nm': contact_top,
            'core_layers': opts.slices,
            'cap_layers': sum(layer.kind.endswith('cap') for layer in layers),
            'coalesced_layers': sum(layer.kind.startswith('coalesced') for layer in layers),
        }
        summaries.append(summary)

        ax = fig.add_subplot(grid[0, column])
        draw_materials(ax, args, clip_cell=True)
        ax.set(xlim=(-76, 76), ylim=(-29, 258), aspect='equal', xlabel='x (nm)',
               ylabel='高さ z (nm)', xticks=[-50, 0, 50], yticks=[0, 30, 100, 200, 230])
        ax.set_title(f'斜面 {side:g} nm', loc='left', fontsize=13, weight='bold', pad=12)
        ax.hlines([0, 30, 200, 230], -50, 50, colors='#8492a5', linestyles=':', lw=.65)
        dimension(ax, (-top_radius, 244), (top_radius, 244), f'金上面 {2*top_radius:g} nm')
        dimension(ax, (-65, 0), (-65, 200), '樹脂高さ200 nm', dx=-3, rotation=90)
        dimension(ax, (65, 200), (65, 230), '30 nm', dx=-17)
        dimension(ax, (65, 0), (65, 30), '30 nm', dx=-17)
        dimension(ax, (-49, -17), (49, -17), '樹脂底部98 nm')
        r = core_radius(args, 140)
        dimension(ax, (r, 140), (r+side, 140), f'{side:g} nm', dy=12)
        ax.text(0, 95, 'モスアイ樹脂', ha='center', color='#294b63', fontsize=9.5)

        zoom = fig.add_subplot(grid[1, column])
        draw_materials(zoom, args)
        zoom.set(xlim=(0, 100), ylim=(15, 90), aspect='equal', xlabel='x (nm)',
                 ylabel='高さ z (nm)', xticks=[0, 50, 100], yticks=[30, 50, 75])
        zoom.set_title(f'谷部拡大：金の接続上端 {contact_top:.2f} nm',
                       loc='left', fontsize=10.5, pad=10)
        zoom.axhline(30, color='#9b6e18', lw=.75, ls=':')
        zoom.axvline(50, color='#8492a5', lw=.65, ls=':')
        zoom.plot([50], [contact_top], 'o', ms=4, color='#9c452c')
        if contact_top > 30:
            zoom.hlines(contact_top, 35, 65, colors='#9c452c', linestyles=':', lw=.8)
        zoom.text(50, 20, '谷部の金膜', fontsize=9, ha='center', color='#654500')

    fig.legend(handles=[Patch(facecolor=AIR, edgecolor='#b6c2cf', label='空気'),
                        Patch(facecolor=RESIN, label='モスアイ樹脂／PET'),
                        Patch(facecolor=GOLD, label='金（Au）')],
               loc='lower left', bbox_to_anchor=(.055, .134), frameon=False, ncol=3)
    fig.text(.065, .106, '上段は1周期、下段は隣接する突起間の断面。樹脂の頂部直径50 nm・底部直径98 nmは全ケース共通。',
             fontsize=10, color='#334155')
    fig.text(.065, .071, '下段の赤点は中心を結ぶ断面での接続上端。六方格子の単位胞全体が埋まる高さを示すものではありません。',
             fontsize=9.8, color='#334155')
    fig.text(.065, .034, f'連続形状の模式図。計算：周期100 nm、本体{opts.slices}層＋頂部Au {summaries[0]["cap_layers"]}層、PET 5 µm、粘着層なし、石英1 mm。',
             fontsize=9.8, color='#334155')
    tag = '_'.join(f'{side:g}' for side in opts.side_levels)
    stem = opts.output_dir/f'MOSMITE_Au30_side{tag}nm_sections'
    fig.savefig(stem.with_suffix('.png'), dpi=200, facecolor='white')
    fig.savefig(stem.with_suffix('.svg'), facecolor='white')
    plt.close(fig)
    stem.with_suffix('.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'figure': str(stem.with_suffix('.png').resolve()), 'geometry': summaries}, indent=2))


if __name__ == '__main__':
    main()
