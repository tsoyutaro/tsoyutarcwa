"""Study-local storage policy; the shared RCWA equations stay unchanged.

Fixed geometry, complex128, full bidirectional S output, Redheffer, no fields
and no gradients only. D6 keeps independent reduced modes but discards their
large Cartesian copies. The full-basis path consumes each layer S as it is
added. Both paths preserve Tf, Rf, Rb and Tb for the finite absorbing stack.
"""
from dataclasses import fields, is_dataclass
import csv
import json
from pathlib import Path

POLICY = 'released-layer-storage-streamed-full-S-v1'
LEGACY = ('P', 'Q', 'eps_conv', 'mu_conv', 'E_eigvec', 'H_eigvec')
AUX_LISTS = ('asr_mappings', 'asr_material_tensors', 'asr_T_matrices',
             'asr_Tz_matrices', 'asr_condition_numbers', 'E_eigvec_uv', 'H_eigvec_uv')
AUX_DICTS = ('_asr_slot_by_layer', '_asr_field_context_by_layer',
             '_physical_material_by_layer', '_nvm_eps_tensor_by_layer',
             '_reduced_longitudinal_context_by_layer')
LAYER_S = ('layer_S11', 'layer_S21', 'layer_S12', 'layer_S22')
PARITY_KEYS = ('R', 'T', 'A', 'R_x', 'T_x', 'R_y', 'T_y', 'cross_R', 'cross_T')
PARITY_TOL = 1e-8


def tensor_storage_bytes(items):
    """Unique live tensor storage, excluding allocator cache and workspaces."""
    storages = {}
    def visit(value):
        if hasattr(value, 'untyped_storage'):
            storage = value.untyped_storage()
            storages[(str(value.device), storage.data_ptr())] = storage.nbytes()
        elif is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                visit(getattr(value, field.name))
        elif isinstance(value, dict):
            for entry in value.values():
                visit(entry)
        elif isinstance(value, (list, tuple)):
            for entry in value:
                visit(entry)
    visit(items)
    return sum(storages.values())


def memory_safe_class(original):
    import torch

    class SpectrumOnlyRCWA(original):
        def _storage_guard(self):
            if (torch.is_grad_enabled() or self.store_mode_couplings
                    or self.field_regions != 'none' or self._dtype != torch.complex128
                    or self.smatrix_size != 'full' or self.smatrix_algorithm != 'redheffer'):
                raise RuntimeError('Reduced storage requires no_grad, no fields, complex128 and full Redheffer S.')

        def _connect_full(self, left, right):
            # Identical ordering and operations to the shared full-S recurrence.
            identity = self._eye(left[0].shape[0])
            tf_l, rf_l, rb_l, tb_l = left
            tf_r, rf_r, rb_r, tb_r = right
            inverse_lr = self._solve(identity - rb_l @ rf_r, identity)
            inverse_rl = self._solve(identity - rf_r @ rb_l, identity)
            return [tf_r @ (inverse_lr @ tf_l),
                    rf_l + tb_l @ (inverse_rl @ (rf_r @ tf_l)),
                    rb_r + tf_r @ (inverse_lr @ (rb_l @ tb_r)),
                    tb_l @ (inverse_rl @ tb_r)]

        def release_completed_layer(self):
            """Call once after each successful layer addition, outside its builder."""
            self._storage_guard()
            if getattr(self, '_released_layers', 0) != self.layer_N-1:
                raise RuntimeError('Release must be called exactly once per completed layer.')
            for name in LEGACY:
                if len(getattr(self, name)) != self.layer_N:
                    raise RuntimeError('Unexpected solver layer storage: '+name)
            for name in AUX_LISTS + AUX_DICTS:
                if not hasattr(self, name):
                    raise RuntimeError('Missing solver storage container: '+name)
            if self.polarization_reduction is not None:
                if (self.group_theory_symmetry != 'd6'
                        or self.polarization_reduction not in ('x', 'y')
                        or len(self._polarized_layers) != self.layer_N
                        or self.group_theory_diagnostics[-1].get('symmetry') != 'D6-E1-source-row'):
                    raise RuntimeError('Only the D6 E1 source row is supported for reduced modes.')
                if any(t.requires_grad for layer in self._polarized_layers for t in layer.values()):
                    raise RuntimeError('Reduced storage does not support gradients.')
            else:
                # Full-basis raster/NVM mixtures already have this layer's S.
                for name in LAYER_S:
                    if len(getattr(self, name)) != self.layer_N:
                        raise RuntimeError('Unexpected full-S storage: '+name)
                layer_s = [getattr(self, name)[-1] for name in LAYER_S]
                if any(block is None or block.requires_grad for block in layer_s):
                    raise RuntimeError('Full-S layer is missing or requires gradients.')
                previous = getattr(self, '_streamed_full_s', None)
                self._streamed_full_s = (layer_s if previous is None
                                         else self._connect_full(previous, layer_s))
                for name in LAYER_S:
                    getattr(self, name)[-1] = None
            for name in LEGACY:
                getattr(self, name)[-1] = None
            for name in AUX_LISTS + AUX_DICTS:
                getattr(self, name).clear()
            self._released_layers = self.layer_N
            retained = tensor_storage_bytes(vars(self))
            self.post_layer_retained_tensor_peak_bytes = max(
                getattr(self, 'post_layer_retained_tensor_peak_bytes', 0), retained)

        def _polarized_redheffer_scattering(self, layers, input_interface,
                                            output_interface, *, force_full=False):
            self._storage_guard()
            if len(layers) != len(self.thickness) or self._released_layers != self.layer_N:
                raise RuntimeError('Reduced modes, thicknesses and released layers disagree.')
            scattering = list(input_interface)
            # Same full-S forward recurrence; do not retain Nz layer S matrices.
            for layer, thickness in zip(layers, self.thickness):
                layer_s = self._reduced_layer_smatrix(layer['electric'], layer['magnetic'],
                    layer['kz'], thickness, self._polarization_reference_v)
                scattering = self._connect_full(scattering, layer_s)
                del layer_s
            return self._connect_full(scattering, output_interface)

        def solve_global_smatrix_s_only(self):
            self._storage_guard()
            if getattr(self, '_released_layers', 0) != self.layer_N:
                raise RuntimeError('Some full-basis layers were not consumed.')
            scattering = getattr(self, '_streamed_full_s', None)
            if scattering is None:
                identity = self._eye(2*self.order_N)
                zero = torch.zeros_like(identity)
                scattering = [identity, zero, zero, identity]
            if hasattr(self, 'Sin'):
                scattering = self._connect_full(self.Sin, scattering)
            if hasattr(self, 'Sout'):
                scattering = self._connect_full(scattering, self.Sout)
            self.S = scattering
            self.C = [[], []]
    return SpectrumOnlyRCWA


def compare_reference(args, metadata, rows):
    """Compare raw CSV powers with an original run at identical conditions."""
    reference = Path(args.reference_dir)
    old = json.loads((reference/'settings.json').read_text(encoding='utf-8'))
    if old.get('status') != 'completed':
        raise ValueError('Reference must be a completed spectrum run.')
    # Paths, device, storage and the updated study driver may differ. Physics,
    # numerical settings, material contents and shared solver must agree.
    keys = ('mode', 'period_nm', 'height_nm', 'base_diameter_nm', 'tip_diameter_nm',
            'top_gold_nm', 'valley_gold_nm', 'side_gold_nm', 'slices', 'cap_slices',
            'film_total_um', 'residual_um', 'adhesive_um', 'adhesive_n', 'adhesive_k',
            'glass_um', 'order', 'grid', 'basis', 'coherence', 'phase_points')
    for key in keys:
        if old['arguments'].get(key) != metadata['arguments'].get(key):
            raise ValueError('Reference condition differs: '+key)
    if old.get('solver_sha256') != metadata.get('solver_sha256'):
        raise ValueError('Reference shared-solver source hashes differ.')
    for key in ('geometry.py', 'stack.py'):
        if old.get('study_sha256', {}).get(key) != metadata['study_sha256'].get(key):
            raise ValueError('Reference study source differs: '+key)
    def material_identity(provenance):
        return {key: ({k: v for k, v in value.items() if k != 'path'})
                for key, value in provenance.items()}
    if material_identity(old['material_provenance']) != material_identity(metadata['material_provenance']):
        raise ValueError('Reference optical data or material assumptions differ.')
    with (reference/'spectra.csv').open(encoding='utf-8-sig', newline='') as handle:
        originals = {float(r['wavelength_nm']): r for r in csv.DictReader(handle)}
    differences = []
    for row in rows:
        match = originals.get(row['wavelength_nm'])
        if match is None:
            raise ValueError('Wavelength missing from reference: '+str(row['wavelength_nm']))
        for key in ('order', 'slices', 'grid'):
            if int(match[key]) != row[key]:
                raise ValueError('Reference CSV condition differs: '+key)
        errors = {key+'_absolute_error': abs(row[key]-float(match[key])) for key in PARITY_KEYS}
        maximum = max(errors.values())
        differences.append({'wavelength_nm': row['wavelength_nm'], **errors,
                            'max_absolute_error': maximum, 'passed': maximum <= PARITY_TOL})
    return differences
