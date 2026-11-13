"""Allowed diffraction orders and signed angles, independent of RCWA efficiencies."""
from pathlib import Path
import math
import argparse

import numpy as np

from model import HERE, Materials, parse_wavelengths, save_csv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=HERE/'results/kinematics')
    p.add_argument('--periods', default='3720,3920')
    p.add_argument('--wavelengths', default='400:700:5')
    args = p.parse_args()
    material = Materials()
    limits, directions = [], []
    for period in map(float, args.periods.split(',')):
        for wavelength in parse_wavelengths(args.wavelengths):
            npmma, source = material.pmma(wavelength)
            for medium, index in [('R_external_air', 1.), ('R_internal_PMMA', npmma), ('T_external_water', 1.333)]:
                bound = index*period/wavelength
                maximum = max(0, math.ceil(bound-1e-10)-1)
                cone_max = math.floor(bound*math.sin(math.radians(20))+1e-10)
                limits.append(dict(period_nm=period, wavelength_nm=float(wavelength), channel=medium,
                    refractive_index=index, max_abs_propagating_m=maximum, propagating_order_count=2*maximum+1,
                    max_abs_m_within_20deg=cone_max, PMMA_source=source))
                for m in range(-math.floor(bound+1e-10), math.floor(bound+1e-10)+1):
                    s = m/bound
                    grazing = abs(abs(s)-1) < 1e-10
                    directions.append(dict(period_nm=period, wavelength_nm=float(wavelength), channel=medium, m=m,
                        theta_signed_deg=math.degrees(math.asin(np.clip(s, -1, 1))),
                        theta_absolute_deg=math.degrees(math.asin(min(1, abs(s)))),
                        grazing_zero_normal_flux=grazing, has_positive_normal_flux=not grazing))
    save_csv(args.output_dir/'order_limits.csv', limits)
    save_csv(args.output_dir/'allowed_order_directions.csv', directions)
    save_csv(args.output_dir/'anchor_order_limits.csv', [r for r in limits if r['wavelength_nm'] in (400., 550., 700.)])


if __name__ == '__main__':
    main()
