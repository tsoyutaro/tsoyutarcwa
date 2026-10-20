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
primitive cell. A smooth dyad continuation blends into an isotropic dyad
at the axis and Voronoi seams, away from the true material interface.
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
standard RK4 to the varying generator. Only local transfers are converted
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
requested. Volume loss uses the factorized constitutive operator; it does
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
