"""Explicit linear-frustum Au geometry; heights are measured from the valley."""
from dataclasses import dataclass, asdict
import math
import numpy as np
from stack import save_csv


@dataclass(frozen=True)
class Layer:
    z_bottom_nm: float
    z_top_nm: float
    core_radius_nm: float
    outer_radius_nm: float
    kind: str
    gold_role: str

    @property
    def thickness_nm(self):
        return self.z_top_nm-self.z_bottom_nm


def side_thickness(args):
    if args.side_gold_nm is not None:
        return args.side_gold_nm
    if args.mode == 'sputter':
        return 8.
    return args.top_gold_nm*(args.base_diameter_nm-args.tip_diameter_nm)/(2*args.height_nm)


def core_radius(args, z):
    return args.base_diameter_nm/2-(args.base_diameter_nm-args.tip_diameter_nm)*z/(2*args.height_nm)


def _partition(boundaries, total):
    spans = np.diff(boundaries)
    if total < len(spans):
        raise ValueError('Too few slices to preserve all material/topology boundaries.')
    extra = total-len(spans)
    quota = extra*spans/sum(spans)
    allocation = 1+np.floor(quota).astype(int)
    remainder = total-int(sum(allocation))
    priority = np.argsort(-(quota-np.floor(quota)), kind='stable')
    allocation[priority[:remainder]] += 1
    edges = []
    for lo, hi, n in zip(boundaries, boundaries[1:], allocation):
        edges.extend(zip(np.linspace(lo, hi, n+1)[:-1], np.linspace(lo, hi, n+1)[1:]))
    return edges


def cap_count(args, slices=None):
    nz = slices or args.slices
    if args.top_gold_nm == 0:
        return 0
    return (max(1, math.ceil(args.cap_slices*nz/args.slices)) if args.cap_slices is not None
            else max(1, math.ceil(nz*args.top_gold_nm/args.height_nm)))


def build_layers(args, slices=None):
    nz = slices or args.slices
    h, p, side = args.height_nm, args.period_nm, side_thickness(args)
    slope = (args.base_diameter_nm-args.tip_diameter_nm)/(2*h)
    bounds = [0., h]
    if 0 < args.valley_gold_nm < h:
        bounds.append(args.valley_gold_nm)
    # The shell radius crosses the nearest-neighbour tangency radius here.
    if slope > 0 and side > 0:
        merge_z = (args.base_diameter_nm/2+side-p/2)/slope
        if args.valley_gold_nm < merge_z < h:
            bounds.append(merge_z)
    layers = []
    for low, high in _partition(sorted(set(bounds)), nz):
        z = (low+high)/2
        core = core_radius(args, z)
        if z < args.valley_gold_nm:
            kind, outer, role = 'valley', 0., 'valley'
        elif side == 0:
            kind, outer, role = 'bare', core, 'side'
        else:
            outer, role = core+side, 'side'
            kind = 'coalesced-shell' if outer >= p/2-1e-10 else 'shell'
        layers.append(Layer(float(low), float(high), core, outer, kind, role))
    if args.top_gold_nm > 0:
        for low, high in _partition([h, h+args.top_gold_nm], cap_count(args, nz)):
            if args.mode == 'evaporation':
                outer = args.tip_diameter_nm/2+side*(1-((low+high)/2-h)/args.top_gold_nm)
            else:
                outer = args.tip_diameter_nm/2+side
            kind = 'coalesced-cap' if outer >= p/2-1e-10 else 'cap'
            layers.append(Layer(float(low), float(high), 0., outer, kind, 'top'))
    return list(reversed(layers))


def nearest_radius(grid):
    axis = np.arange(grid)/grid
    u, v = np.meshgrid(axis, axis, indexing='ij')
    radius2 = np.full_like(u, np.inf)
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            q1, q2 = u-.5-a, v-.5-b
            radius2 = np.minimum(radius2, q1*q1+q1*q2+q2*q2)
    return np.sqrt(np.maximum(radius2, 0))


def material_codes(layer, distance_nm):
    """0=air, 1=resin, 2=gold. Periodic nearest-site distance handles unions."""
    code = np.zeros_like(distance_nm, dtype=np.int8)
    if layer.kind == 'valley':
        code[:] = 2
    elif layer.kind != 'bare':
        code[distance_nm <= layer.outer_radius_nm] = 2
    if layer.core_radius_nm > 0:
        code[distance_nm <= layer.core_radius_nm] = 1
    return code


def write_geometry(args, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    layers = build_layers(args)
    rows = []
    for number, layer in enumerate(layers):
        row = {'index_from_air': number, **asdict(layer), 'thickness_nm': layer.thickness_nm}
        row['backend'] = 'periodic-union-raster' if layer.kind.startswith('coalesced') else 'analytic-NVM'
        rows.append(row)
    save_csv(output/'geometry_layers.csv', rows)
    upper = args.height_nm+args.top_gold_nm
    xs, zs = np.linspace(-args.period_nm, args.period_nm, 1000), np.linspace(-20, upper+20, 900)
    codes = np.zeros((len(zs), len(xs)), dtype=np.int8)
    nearest = np.min(np.abs(xs[:, None]-args.period_nm*np.array([-1, 0, 1])[None, :]), axis=1)
    for j, z in enumerate(zs):
        if z < 0:
            codes[j, :] = 1
        else:
            layer = next((a for a in layers if a.z_bottom_nm <= z < a.z_top_nm), None)
            if layer is not None:
                codes[j, :] = material_codes(layer, nearest)
    cmap = ListedColormap(['#f4f7fb', '#94b9d4', '#c99a28'])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.4), gridspec_kw={'width_ratios': [1, 1]})
    axes[0].imshow(codes, origin='lower', extent=[xs[0], xs[-1], zs[0], zs[-1]],
                   cmap=cmap, vmin=0, vmax=2, interpolation='nearest', aspect='equal')
    axes[0].set(xlabel='x (nm)', ylabel='z from valley (nm)', title='Cross section through lattice sites')
    for z in [0, args.valley_gold_nm, args.height_nm, upper]:
        axes[0].axhline(z, color='#343d4a', lw=.6, ls='--')
    z_plan = min(args.height_nm*.75, args.valley_gold_nm+15.)
    layer = next(a for a in layers if a.z_bottom_nm <= z_plan < a.z_top_nm)
    grid = 320
    d = nearest_radius(grid)*args.period_nm
    code = material_codes(layer, d)
    axis = np.linspace(0, 1, grid+1)
    u, v = np.meshgrid(axis, axis, indexing='ij')
    axes[1].pcolormesh(args.period_nm*(u+.5*v), args.period_nm*np.sqrt(3)/2*v,
                       code, cmap=cmap, vmin=0, vmax=2, shading='flat', rasterized=True)
    axes[1].set(xlabel='x (nm)', ylabel='y (nm)', title=f'Primitive cell near z={z_plan:g} nm', aspect='equal')
    axes[1].plot(args.period_nm*np.array([0, 1, 1.5, .5, 0]),
                  args.period_nm*np.sqrt(3)/2*np.array([0, 0, 1, 1, 0]), color='black', lw=.8)
    fig.legend(handles=[Patch(color=cmap(i), label=name) for i, name in enumerate(['Air', 'Resin', 'Au'])],
               loc='lower center', ncol=3)
    fig.suptitle(f'{args.mode}: top/valley={args.top_gold_nm:g}/{args.valley_gold_nm:g} nm, '
                 f'side radial={side_thickness(args):g} nm; {args.slices} core + {cap_count(args)} cap layers')
    fig.tight_layout(rect=[0, .07, 1, .94])
    fig.savefig(output/'geometry.png', dpi=180)
    plt.close(fig)
    return layers
