# Geometry and material conventions

The implementation follows the equations and captions in Mohamad et al.,
*Optical and Quantum Electronics* 52, 127 (2020), DOI
10.1007/s11082-020-2240-y. Lengths are in nm and material values are relative
permittivities `epsilon = n**2`. The time convention is `exp(-i omega t)`;
the tabulated positive imaginary part of the refractive index represents loss.

## Aluminium sinusoid: Figures 4–6

- Period: 500 nm; peak-to-peak height: 200 nm.
- Air incidence: `n_in = 1`; semi-infinite aluminium:
  `n_Al = 1.3 + 7.6j`, used at the paper wavelength 632.8 nm.
- Computational `+z` points from air into the metal. Its origin is the
  shallowest surface point. The surface depth is
  `d(x) = H/2 * (1 + cos(2*pi*x/P))`; metal occupies `z >= d(x)`.
- A lateral phase translation of the sinusoid does not change diffraction
  efficiencies. The implementation uses the phase above for reproducibility.
- Unit tangent: `(c,s) = (1,d'(x))/sqrt(1+d'(x)**2)`.

## Corrugated gold film: Figures 8–10

- Period: 400 nm; triangular relief: 283.2 nm; gold-film thickness: 50 nm.
- Substrate incidence: `n_in = 1.55`; output air: `n_out = 1`.
- Gold: `n_Au = 0.16 + 5.26j`, the value explicitly given at 810 nm.
- Computational `+z` points upward from substrate to air. With
  `u = (x mod P)/P`, the substrate/Au surface is
  `f(x) = H*(1-abs(2*u-1))`. The substrate occupies `z < f(x)`, gold occupies
  `f(x) <= z <= f(x)+t`, and air occupies `z > f(x)+t`.
- The gold film has **two parallel corrugated interfaces**. It is not a
  triangular metal ridge on a flat substrate. The 50 nm thickness is treated
  as a vertical offset, consistent with the schematic; the paper does not
  state the distinction between vertical and normal thickness explicitly.
- Both interfaces use `(c,s) = (1,f'(x))/sqrt(1+f'(x)**2)`. At the isolated
  triangular apexes the right derivative is chosen. This avoids introducing
  an arbitrary horizontal facet at an FFT grid point.

## Graded index layer (GIL)

The paper specifies a horizontal layer of width `delta` within each sampled
z section and a linear variation of both the real and imaginary parts of
the refractive index (Figures 7b–c). This implementation therefore interpolates
**complex refractive index**, then squares it to obtain permittivity:
`n(d) = (1-d/delta)*n_metal + (d/delta)*n_dielectric` for `0 <= d < delta`.
The phrase “linear permittivity” elsewhere in the text does not uniquely
specify the same operation.

Here the transition extends horizontally **outside the original metal**,
replacing the adjacent dielectric at both gold interfaces. This explicit
choice preserves the sharp metal domain; Figure 7 can also be read as a
transition replacing part of the metal. The paper does not supply enough
coordinates or source code to eliminate that ambiguity. GIL results should
therefore be assessed as a documented reproduction of the method, rather than
an exact digit-for-digit reproduction of Figures 9–10.

At each z, the distance is measured to the closest periodic metal boundary.
This construction handles the valley, apex, and intervals crossing the unit
cell boundary. If two transitions overlap, the nearest interface determines
the interpolation. No GIL is added at z values with no metal cross-section.
The dielectric endpoint is the original local substrate or air index.

## Limits on quantitative comparison

- The paper provides the sinusoidal reference
  `R[-1] = 0.8785309971543` (Figure 5).
- The triangular reflection plots provide graphical, rather than tabulated,
  reference values. GIL placement and corner sampling can affect them.
- Figure 10b uses the existing `studies/gold_motheye2/data/au_measured_nk.csv`
  through `studies.shared.gold_dispersion.load_gold_csv`. It uses the same
  linear-epsilon interpolation as the other Au studies, then takes the passive
  square root for the GIL index. No new table is needed and `all` includes it.
- The explicit 810 nm index remains the input for Figures 9 and 10a. The
  existing table gives about 0.15677+4.99490j at 810 nm, rather than the paper's
  0.16+5.26j. The paper does not provide its full dispersion table, so the
  spectrum uses the project's documented material model without claiming
  that the two material datasets are identical.
