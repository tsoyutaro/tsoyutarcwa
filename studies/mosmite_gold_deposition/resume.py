"""Read committed spectrum rows and check compatibility before resuming."""
import csv
import hashlib
import json
import math
from pathlib import Path


PHYSICS_KEYS = (
    'mode', 'period_nm', 'height_nm', 'base_diameter_nm', 'tip_diameter_nm',
    'top_gold_nm', 'valley_gold_nm', 'side_gold_nm', 'slices', 'cap_slices',
    'film_total_um', 'residual_um', 'adhesive_um', 'adhesive_n', 'adhesive_k',
    'glass_um', 'order', 'grid', 'basis', 'coherence', 'coherent_layers',
    'phase_points', 'native_d6',
)
TEXT_COLUMNS = {
    'coherence', 'basis', 'layer_storage', 'storage_policy', 'unpolarized',
}
INTEGER_COLUMNS = {
    'order', 'slices', 'grid', 'harmonics', 'eigen_dimension', 'core_layers',
    'cap_layers', 'total_pattern_layers', 'coalesced_raster_layers',
    'retained_before_solve_bytes', 'post_layer_retained_tensor_peak_bytes',
    'geometry_disk_builds', 'geometry_normal_builds', 'geometry_raster_builds',
    'peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes',
}


def same_source(saved, current, path):
    if saved == current:
        return True
    if saved is None or current is None:
        return False
    normalized = path.read_bytes().replace(b'\r\n', b'\n')
    # Older settings store raw hashes. Accept both platform newline forms.
    return saved in {hashlib.sha256(data).hexdigest()
                     for data in (normalized, normalized.replace(b'\n', b'\r\n'))}


def load_resume(output, metadata, wavelengths):
    settings_path = output / 'settings.json'
    spectrum_path = output / 'spectra.csv'
    if not settings_path.is_file() or not spectrum_path.is_file():
        raise ValueError('--resume requires existing settings.json and spectra.csv in the output directory.')
    old = json.loads(settings_path.read_text(encoding='utf-8-sig'))
    for key in PHYSICS_KEYS:
        default = False if key == 'native_d6' else None
        if old['arguments'].get(key, default) != metadata['arguments'].get(key, default):
            raise ValueError('Resume condition differs: ' + key)
    study = Path(__file__).resolve().parent
    solver = Path(metadata['arguments']['solver_root'])
    sources = {
        'optics_sha256': study.parent / 'mosmite_optics.py',
        'phase_average_sha256': study.parent / 'mosmite_phase_average.py',
        'native_d6_sha256': solver / 'rcwa_ext/acceleration/d6.py',
    }
    for key, path in sources.items():
        if not same_source(old.get(key), metadata.get(key), path):
            raise ValueError('Resume source differs: ' + key)
    # Job folders and operating systems may differ while the sources agree.
    def solver_hashes(meta):
        return {name.replace('\\', '/'): value
                for name, value in meta.get('solver_sha256', {}).items()}

    saved_solver, current_solver = solver_hashes(old), solver_hashes(metadata)
    if saved_solver.keys() != current_solver.keys() or any(
            not same_source(digest, current_solver[name], solver / name)
            for name, digest in saved_solver.items()):
        raise ValueError('Resume shared-solver source hashes differ.')
    for key in ('geometry.py', 'stack.py', 'memory.py', 'shape_cache.py', 'native_d6.py'):
        if not same_source(old.get('study_sha256', {}).get(key),
                           metadata.get('study_sha256', {}).get(key), study / key):
            raise ValueError('Resume study source differs: ' + key)

    previous_materials, current_materials = old['material_provenance'], metadata['material_provenance']
    if previous_materials.keys() != current_materials.keys():
        raise ValueError('Resume optical data or material assumptions differ.')
    for name, info in current_materials.items():
        previous = previous_materials[name]
        ignored = {'path', 'sha256'}
        if {k: v for k, v in previous.items() if k not in ignored} != {
                k: v for k, v in info.items() if k not in ignored}:
            raise ValueError('Resume optical data or material assumptions differ: ' + name)
        if not same_source(previous.get('sha256'), info.get('sha256'), Path(info.get('path', '.'))):
            raise ValueError('Resume material CSV differs: ' + name)
    required = {'wavelength_nm', 'order', 'slices', 'grid', 'R', 'T', 'A',
                'R_x', 'T_x', 'R_y', 'T_y', 'cross_R', 'cross_T'}
    requested = {float(wl) for wl in wavelengths}
    restored = {}
    with spectrum_path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError('Resume spectrum is missing required columns.')
        for original in reader:
            if None in original or any(value is None for value in original.values()):
                raise ValueError('Resume spectrum contains an incomplete CSV row.')
            row = {}
            for key, value in original.items():
                if value == '':
                    row[key] = None
                elif key in TEXT_COLUMNS:
                    row[key] = value
                elif value in ('True', 'False'):
                    row[key] = value == 'True'
                elif key in INTEGER_COLUMNS:
                    row[key] = int(value)
                else:
                    row[key] = float(value)
                    if not math.isfinite(row[key]):
                        raise ValueError('Resume spectrum contains a nonfinite value.')
            wl = row['wavelength_nm']
            if any(row[key] is None for key in required):
                raise ValueError('Resume spectrum is missing a required row value.')
            if wl not in requested or wl in restored:
                raise ValueError('Resume wavelengths must be unique and included in --wavelengths.')
            if any(row[key] != metadata['arguments'][key] for key in ('order', 'slices', 'grid')):
                raise ValueError('Resume spectrum order/slices/grid differ from the requested settings.')
            if not all(row[key] is not None and -2e-6 <= row[key] <= 1+2e-6
                       for key in ('R', 'T', 'A')):
                raise ValueError('Resume spectrum contains invalid R/T/A.')
            if abs(row['R'] + row['T'] + row['A'] - 1) > 2e-6:
                raise ValueError('Resume spectrum violates R + T + A = 1.')
            restored[wl] = row
    # CSV replacement precedes settings replacement. Trust complete CSV rows
    # if a shutdown occurred between those two commits.
    return [restored[float(wl)] for wl in wavelengths if float(wl) in restored]
