"""Exact interval coefficients for Au/PMMA/air using the verified 1D Li adapter."""
from __future__ import annotations

import gc
import time
import torch
from rcwa_ext.config import GroupTheoryOptions, Lattice, OutputSpec
from studies.gold_grating_1d.solver import Grating1D, port_results
from studies.pmma_gold_grating_1d.geometry import build_layers


class CoatedGrating1D(Grating1D):
    def indicator(self, fill):
        if not 0 <= fill <= 1:
            raise ValueError("Interval fill must be within [0,1].")
        if fill == 0:
            return torch.zeros_like(self._eye(self.order_N))
        if fill == 1:
            return self._eye(self.order_N)
        delta = (self.order_x[:, None]-self.order_x[None, :]).to(torch.float64)
        return fill*torch.sinc(delta*fill)*torch.exp(-1j*torch.pi*delta)

    def ridge_convolutions(self, layer, materials, grid, *, analytic=False):
        if not analytic or len(self.order_y) != 1:
            raise ValueError("This study uses analytic intervals and M_y=0 only.")
        pmma, gold = materials
        background = gold if layer.background == "gold" else 1.
        period = self.physical_period_nm
        outer = self.indicator(layer.outer_width_nm/period)
        core = self.indicator(layer.core_width_nm/period)
        identity = self._eye(self.order_N)
        direct = background*identity+(gold-background)*outer+(pmma-gold)*core
        reciprocal = identity/background+(1/gold-1/background)*outer+(1/pmma-1/gold)*core
        return direct, reciprocal, None

    def add_coated_layer(self, layer, materials, audit=False):
        details = super().add_ridge((layer.bottom_depth_nm-layer.top_depth_nm)/self.physical_period_nm,
                                    layer, materials, 1, analytic=True, audit=audit)
        details.pop("sampled_fill_fraction")
        details.pop("fill_fraction_error")
        return details


def simulation(order, wavelength, period, pmma, device):
    sim = CoatedGrating1D(freq=period/wavelength, order=[order, 0],
                         lattice=Lattice.square(1.), cascade="redheffer",
                         outputs=OutputSpec(smatrix_size="half", fields="none"),
                         group_theory=GroupTheoryOptions(enabled=False),
                         dtype=torch.complex128, device=device)
    sim.physical_period_nm = period
    sim.add_input_layer(eps=1., mu=1.)
    sim.add_output_layer(eps=pmma, mu=1.)
    sim.set_incident_angle(0., 0.)
    return sim


def simulate(config, numbers, wavelength, materials, requested_device):
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu explicitly.")
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    pmma, gold = materials
    if pmma.imag != 0 or pmma.real <= 0:
        raise ValueError("The PMMA substrate must be real and positive in this study.")
    layers = build_layers(config["geometry"], numbers["slices"])
    period = config["geometry"]["period_nm"]
    audits = []
    with torch.no_grad():
        sim = simulation(numbers["order"], wavelength, period, pmma, device)
        selected = {0, 1, len(layers)//2, len(layers)-1}
        for index, layer in enumerate(layers):
            details = sim.add_coated_layer(layer, materials, audit=index in selected)
            if index in selected:
                audits.append(dict(details, layer=index+1, kind=layer.kind,
                                   core_width_nm=layer.core_width_nm, outer_width_nm=layer.outer_width_nm))
        # port_results computes actual port power, valid also for lossless PMMA.
        port = port_results(sim)
        observables = {}
        for pol, data in port.items():
            r, t = data["reflectance"], data["power_into_substrate"]
            observables[pol] = {"reflectance": r, "transmittance": t, "absorptance": 1-r-t}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = dict(numbers, wavelength_nm=wavelength, total_finite_layers=len(layers),
                  coefficient_method="analytic", grid_used=None,
                  epsilon_pmma=pmma.real, epsilon_gold_real=gold.real, epsilon_gold_imag=gold.imag,
                  polarizations=observables, selected_layer_diagnostics=audits,
                  runtime_seconds=time.perf_counter()-started,
                  fourier_orders_x=2*numbers["order"]+1, fourier_orders_y=1,
                  environment={"torch": str(torch.__version__), "device": str(device),
                               "cuda": torch.version.cuda})
    if device.type == "cuda":
        result.update(peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                      peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved(device)))
        result["environment"]["gpu"] = torch.cuda.get_device_name(device)
    return result
