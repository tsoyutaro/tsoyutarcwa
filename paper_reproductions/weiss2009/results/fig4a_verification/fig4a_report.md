# Weiss 2009 Fig. 4(a) verification

Status: **matched_within_tolerance**

Computed rows: 202/202; N=12; grids=[256, 512].

Reference: vector polylines extracted from the publisher PDF, not raw author data.

| Curve | Max absolute error | RMSE |
|---|---:|---:|
| T | 0.0049354112 | 0.0014695302 |
| R | 0.0056353123 | 0.0019900955 |
| A | 0.006292782 | 0.00094556464 |

Reasons: all requested checks passed

Paper T/R comparison: zeroth. A always uses all orders.

Default comparison uses zeroth-order T0/R0 and total absorption. This is inferred from the plotted loss of T+R+A above the first diffraction threshold; the paper caption does not specify the orders. Total-order metrics are also provided; the mode is never selected by best fit.

Agreement with digitized Fig. 4(a) within user-defined absolute tolerances. Not agreement with authors' raw data or a proof of Fourier-order convergence. A=1-R-T is not an independent conservation test.

See fig4a_report.json for tolerances and provenance.
