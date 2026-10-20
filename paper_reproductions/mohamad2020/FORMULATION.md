# Implemented TM formulation

Sources: Mohamad et al. (2020), [DOI](https://doi.org/10.1007/s11082-020-2240-y),
Eqs. (4)-(11); tangent continuation follows the discussion in Appendix B of
Popov and Neviere (2000), [author-hosted paper](https://www.fresnel.fr/perso/popov/Work/Papers/JOSA_A_17_1773_2000.pdf).

Use time dependence exp(-i omega t), k0=2*pi/lambda, h=Z0*Hy and
F=[Ex;h]. Computational z increases from incident to exit half-space.
Orders m=-N,...,+N give alpha_m=n_inc*sin(theta)+m*lambda/period.
T(f) denotes the truncated Toeplitz matrix with entries f_(m-n).
The implementation preserves all matrix multiplication order below.

```text
E = T(epsilon)
V = inverse(T(1/epsilon))
D = E - V
Q1 = D @ T(c*c)
B = D @ T(c*s)
A = V + Q1
C = E - Q1
alpha = diag(alpha_m)

dF/dz = i*k0*K(z) @ F

K = [[-alpha @ inverse(C) @ B,     I - alpha @ inverse(C) @ alpha],
     [ A - B @ inverse(C) @ B,       - B @ inverse(C) @ alpha]]
```

Linear solves replace explicit inverses in the implementation. For FFF, the
unit tangent is (c,s)=(1,f')/sqrt(1+f'^2), with f' in computational axes.
The ordinary RCWA uses c=0,s=1, recovering Eq. (11):

```text
K_RCWA = [[0, I-alpha @ inverse(E) @ alpha],
          [V,                              0]]
```

RCWA and RCWA-FFF freeze K in each section and evaluate
P=expm(i*k0*dz*K). This is the constant-matrix modal propagator without
requiring an explicit eigenvector inversion. The paper's entrance sampling
is the default; midpoint sampling is an explicit alternate setting.
DM-FFF now defaults to `cf4-adaptive`. For a smooth GIL, set M=i*k0*K,
c1=1/2-sqrt(3)/6, c2=1/2+sqrt(3)/6, a=(3-2*sqrt(3))/12,
b=(3+2*sqrt(3))/12. A fourth-order step is

```text
M1 = M(z+c1*h); M2 = M(z+c2*h)
P4(h) = exp(h*(a*M1+b*M2)) @ exp(h*(b*M1+a*M2))
```

The right exponential acts first. This is the fourth-order commutator-free
Magnus formula of [Alvermann et al. (2012), Eqs. 11-12](https://doi.org/10.1088/1367-2630/14/10/105008).
One full step and two half steps are independently converted to scattering
blocks. For each block let norm(X)=max(abs(X_ij)). Accept the fine result when
norm(coarse-fine) / (15*(atol+rtol*max(norm(coarse),norm(fine)))) is <=1
across all four blocks. No evanescent sub-blocks are omitted.
The factor 15 is the step-doubling estimate for order four. No Richardson
extrapolation or power renormalization is applied. Default rtol=1e-6,
atol=1e-9; the minimum allowed trial step is 1e-5 nm. Failure to satisfy the
criterion raises an error rather than silently accepting the point. These
are local error estimates; tolerance refinement remains a separate check.

For the supplied sharp profiles (GIL=0), epsilon sampled at each FFT x point
is constant between interface crossings. All crossings are supplied by the
geometry. The adaptive integrator resolves them and uses exact constant-matrix
exponentials on those intervals. Thus quadrature cannot silently skip a thin
material interval. This is exact in z for the sampled material, up to floating
point error, while spatial FFT/truncation errors remain.

`section_nm/dm_substeps` is the maximum trial step, not the accepted step.
Each accepted internal interval is converted to a scattering matrix before
composition. Only scattering matrices are cascaded with the Redheffer star
product, including inside an outer section. No section-wide or global growing
transfer is multiplied. Fixed `rk4` and fixed `cf4` are explicit diagnostic
options; the original fixed RK4 can still produce spurious gain at high N.
The new default changes the paper's integration scheme, not its ODE or FFF
matrix product order. It does not promise a passive truncation for every N.

In a uniform exterior q_m=sqrt(epsilon-alpha_m^2) is chosen with Im(q)>=0,
and Re(q)>=0 for propagating modes. Magnetic forward/backward amplitudes
have fields Ex=(q/epsilon)*(a-b), h=a+b. The zero-order incident amplitude
is h_inc=1. Diffraction efficiencies are

```text
R_m = |r_m|^2 * Re(q_inc,m / epsilon_inc) / Re(q_inc,0 / epsilon_inc)
T_m = |t_m|^2 * Re(q_out,m / epsilon_out) / Re(q_inc,0 / epsilon_inc)
R = sum(R_m); T = sum(T_m); A = 1-R-T
```

Lossless evanescent orders carry zero real flux. For a lossy exit medium,
T means power flowing through the exit plane into that medium; it includes
power subsequently absorbed in the infinite substrate. A is absorption in
the finite modeled stack, not total absorption including that half-space.

The field map solves for forward/backward amplitudes at every section cut
using prefix and suffix scattering matrices. It reconstructs the retained
Ex Fourier coefficients directly. A Bloch phase changes complex field phase
but does not change the plotted magnitude. No normal-field reconstruction
or display smoothing is used to erase truncation artifacts.

Finite truncation of FFF and inadequate integration can violate passivity.
The implementation reports those finite results, flags them as UNPHYSICAL,
and never enforces R+T=1 by renormalizing the answer.
