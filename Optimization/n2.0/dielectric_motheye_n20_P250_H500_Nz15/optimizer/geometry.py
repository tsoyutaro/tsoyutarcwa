"""A documented, self-contained monotone-softmax parameterization."""
from __future__ import annotations

import math


def initial_logits(top, bottom, config):
    n = config["layers"]
    low, high = config["radius_min_nm"], config["radius_max_nm"]
    if not low < top <= bottom < high:
        raise ValueError("Initial radii must satisfy lower < top <= bottom < upper.")
    radii = [top + (bottom-top)*i/(n-1) for i in range(n)]
    edges = [low, *radii, high]
    floor = config["softmax_gap_floor"]
    raw = [max(1e-12, ((b-a)/(high-low)-floor)/(1-(n+1)*floor))
           for a, b in zip(edges, edges[1:])]
    total = sum(raw)
    logits = [math.log(v/total) for v in raw]
    average = sum(logits)/len(logits)
    return [v-average for v in logits]


def baseline_logits(config):
    n = config["layers"]
    tip, base = config["baseline_tip_nm"], config["baseline_base_nm"]
    return initial_logits(tip+(base-tip)*.5/n, tip+(base-tip)*(n-.5)/n, config)


def radii_tensor(logits, config):
    import torch
    floor = config["softmax_gap_floor"]
    weights = floor + (1-logits.numel()*floor)*torch.softmax(logits, dim=0)
    return config["radius_min_nm"] + (config["radius_max_nm"]-config["radius_min_nm"])*torch.cumsum(weights, 0)[:-1]


def radii_values(logits, config):
    peak = max(logits)
    weights = [math.exp(v-peak) for v in logits]
    total, cumulative = sum(weights), 0.
    floor = config["softmax_gap_floor"]
    radii = []
    for weight in weights[:-1]:
        cumulative += floor+(1-len(weights)*floor)*weight/total
        radii.append(config["radius_min_nm"]+(config["radius_max_nm"]-config["radius_min_nm"])*cumulative)
    return radii


def profile_rows(logits, config):
    radii = radii_values(logits, config)
    dz = config["height_nm"]/config["layers"]
    return [{"layer": i+1, "top_depth_nm": i*dz, "bottom_depth_nm": (i+1)*dz,
             "radius_nm": r, "diameter_nm": 2*r, "nearest_neighbor_gap_nm": config["period_nm"]-2*r}
            for i, r in enumerate(radii)]
