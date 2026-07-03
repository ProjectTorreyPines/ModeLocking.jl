# ModeLocking commit message

EF phase in RHS, frequency-unit C1 convention, plot and normalization fixes

## EF phase propagation (dynamics.jl)

- rhs_RW!: added thEF parameter; EF→wall coupling uses -i rotation:
  sin(thEF-thW) and -cos(thEF-thW) in dydt[4-5].
- rhs_basic!: added thEF parameter; EF→island coupling rotated:
  sin(thEF-θ), cos(thEF-θ) in dydt[1-3].
- make_ode_func: p tuple now (C2, C1, EFphase, ode_params, n_tor, control_type).
- solve_ODEs: EFphase added to signature and p tuple.
- solve_grid, solve_and_classify, simulate_one_case: EFphase threaded through.

## Frequency-unit C1 convention (dynamics.jl, types.jl)

- Control1_min/max in ODEparams now interpreted as kHz.
- RHS torque: mu*(2π*C1 - Om) — the 2π converts stored frequency C1
  to angular frequency, matching the ODE state variable Om.
- normalize_ode_results: OmN = Om / (2π*C1).
- calculate_bifurcation_bounds: Y_ang = 2π*Y used throughout discriminant
  and passed to _nl_bifurcation_indicator.

## Plotting fixes (plotting.jl)

- Phase diagrams and probability y-axis: now shows f₀(kHz) via y/(1e3*t0).
- Time-trace physical units: freq_factor = 1/(2π*1e3*t0) for ODE state → kHz.
- Operating-point marker (yellow star) with correct unit conversion.
- shot_label kwarg on plot_scatter, plot_phase_diagrams, plot_probability.
- Probability title: generic "Locking probability P(locked)" (no method name).
- Landscape panel sizes adjusted for squarer aspect ratio.

## types.jl

- error_field default: 0.5 → 10.0 (Gauss).
- n_epochs default: 1000 → 200.
- Control1_min/max comments updated to reflect kHz units.
