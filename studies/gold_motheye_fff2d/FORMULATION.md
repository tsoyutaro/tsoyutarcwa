# Vector FFF implementation

This is a two-periodic, full-vector extension rather than a reproduction of a
2D benchmark in Mohamad et al. (2020). The tilted-normal tensor follows
Weismann, Gallagher and Panoiu, Eq. (6), including its symmetric ordering:
[primary paper, sections 3 and 5](https://arxiv.org/pdf/1507.06364).
The original 1D source is [Mohamad et al.](https://doi.org/10.1007/s11082-020-2240-y).
The existing repository NV tensor uses the same symmetric ordering for its
transverse blocks; this implementation also includes the longitudinal blocks.

Write Eps=T(epsilon), V=inverse(T(1/epsilon)), and Delta=V-Eps.
For P_ij=T(n_i*n_j), the implemented constitutive blocks are

```text
C_ij = delta_ij Eps + (Delta P_ij + P_ij Delta)/2
D_i  = sum_j C_ij E_j.
```

The local cone normal in computational downward z is proportional to
`(x/r, y/r, -dr/dz)`. A periodic nearest-image construction is used in the
primitive cell. For slope s=dr/dz, a smooth dyad continuation blends into
`diag(1/2,1/2,s*s)/(1+s*s)` at the axis and Voronoi seams.
At s=0 this is exactly `diag(1/2,1/2,0)`, matching the existing NV
continuation, with identically zero xz/yz/zz projection blocks. This fixes
the former `diag(1/3,1/3,1/3)` continuation, whose Pzz stayed nonzero
even for a vertical cylinder. The taper/seam weights are the same as in
`rcwa_ext/nvm.py`; Q convergence is still required.
The dyad is a continuation, not a unit-vector field at those artificial points.
Disk and annular material Fourier coefficients are analytic Bessel integrals;
only the normal dyad uses FFT sampling. Thus Q controls the normal-field
sampling, not a voxelized circular material boundary.

With time dependence exp(-i omega t), normalized h=Z0 H, and F=[Ex,Ey,hx,hy],
Maxwell's equations give

```text
Dz = Ky hx - Kx hy
Hz = Kx Ey - Ky Ex
Ez = inverse(Czz) (Dz-Czx Ex-Czy Ey)

dEx/dz = i k0 (Kx Ez + hy)
dEy/dz = i k0 (Ky Ez - hx)
dhx/dz = i k0 (Kx Hz - Dy)
dhy/dz = i k0 (Ky Hz + Dx).
```

`generator_from_tensor` eliminates Ez and Hz by linear solves. Its 1D TM
submatrix with `--ordering paper-left` exactly recovers the original paper's
ordered products. Default symmetric ordering preserves lossless Hermiticity;
the two finite truncations need not agree point for point.

RCWA-FFF freezes this generator at each section midpoint and exponentiates
it locally. Exponential sections with a large estimated evanescent growth
are subdivided while preserving their same frozen generator. DM-FFF applies
adaptive fourth-order commutator-free Magnus (CF4) integration to the
varying generator. One full step is compared to two half steps in every S
block, using the fourth-order Richardson factor 15. Accepted half steps are
composed as S matrices. `--dm-step-nm` caps trial length; an independent
estimated evanescent-growth bound further caps local transfers. Cylinders
and homogeneous films use exact constant-generator local exponentials.
There is no discontinuous sampled-material crossing list in 2D: disk/ring
material Fourier coefficients are analytical, unlike the sampled 1D TM
material. Fixed CF4 and legacy RK4 remain explicit diagnostic choices.
Reference: [Alvermann et al. (2012)](https://doi.org/10.1088/1367-2630/14/10/105008).
Only local transfers are converted
to scattering matrices; the stack uses Redheffer composition. No global
growing transfer matrix is multiplied. CPU exponentials use SciPy and CUDA
exponentials use PyTorch complex128. There is no autograd or mode reduction.

The triangular-cell harmonic set is `max(|m|,|n|,|m-n|)<=M`, with
`B=1+3M(M+1)` and Cartesian reciprocal vectors derived from the direct
primitive cell. This set is D6 closed. It differs from rectangular cutoffs
in the existing runners and is recorded explicitly in every result.

Interface fields and per-order power use the full Cartesian admittance,
including both polarizations. Absorption in the finite cone region is
independently checked by integrating `k0 Im(E^H D)/(2 Pinc)` over z when
requested. Interior fields are reconstructed from left/right local S
matrices, avoiding multiplication by a growing section transfer. Volume
loss uses the factorized constitutive operator; it does
not, by itself, prove convergence to the continuum material.

In particular, Hermiticity for a real dielectric does not imply positive
semidefinite loss for every truncated complex-metal operator. The symmetric
anticommutator is not a positivity-preserving construction in general.
An observed negative eigenvalue of `(C-C^H)/(2i)` identifies a possible
artificial-gain direction. It is reported, never removed by conjugate
averaging or by clipping scattering efficiencies. GIL can alter such behavior
but also changes the material distribution, so its influence requires a
separate convergence study. This research extension has not been validated
against an independent high-order 3D full-wave solver.


GIL has **total radial** width delta, centered at the nominal circle radius
by default (r-delta/2 through r+delta/2). It is not constant thickness measured
along the tilted surface normal. A fixed width is compatible with broadband
material dispersion: the complex index of gold and each transition level
are recomputed at every wavelength using the existing shared CSV loader.
Linear complex n is squared for epsilon, and annular Bessel coefficients
are summed. Compare delta, ring count, Fourier order and slices separately;
GIL changes the physical model. Explicit `--gil-side dielectric` preserves
the former outward-only placement for diagnostic comparisons.

`--harmonic-shape rectangular` enables direct comparison to the existing
`rcwa_ext.AutoRCWA` NV eigenmode solver. The independent validation adapter
uses that solver's own material, projection and propagation routines, with
Lanczos filtering and group reduction disabled. Both square and triangular
cells are checked, for Au cylinders and lossless tapered slices. The native
Torch Bessel implementation and SciPy j1 give small (~1e-8 in disk
coefficients) backend differences; validation records material/projection
residuals as well as all-order complex amplitudes and powers. Agreement is
an implementation reduction check, not proof of high-order Au convergence.
