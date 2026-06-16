# ModeLocking.jl

Reduced-order ODE model and analysis tools for tearing-mode locking, resistive-wall-mode
coupling, and error-field penetration physics. Originally developed as part of the FUSE
`ActorLocking` mode-locking actor and extracted into a standalone package so that the
physics/ODE core, NN-based locking classifier, persistence, and plotting utilities can be
developed, tested, and reused independently of FUSE.

## Contents

- `ODEparams`, RHS functions, and `solve_ODEs` / `solve_grid` / `solve_and_classify`: the
  reduced-order ODE model (rational-surface tearing mode +/- resistive-wall-mode +/-
  ideal-wall coupling), including normalization and analytic bifurcation bounds.
- `NNparams`, `LockingNNModel`, `train_locking_nn`, `tune_locking_nn`,
  `transfer_learn_locking_nn`: a small Flux-based classifier that predicts locking
  probability as a function of the two control parameters.
- `save_ode_results` / `load_ode_results`, `save_locking_nn` / `load_locking_nn`:
  BSON-based persistence of ODE grids and trained NN models.
- `plot_sols`, `plot_scatter`, `plot_phase_diagrams`, `plot_probability`,
  `plot_time_traces`, `simulate_one_case`: plotting utilities for grid scans and
  single-case time traces.

## Usage

ModeLocking is intended to be used as a dev dependency of FUSE, where the
`ActorLocking` actor (`src/actors/stability/locking_actor.jl`) builds an `ODEparams`
from `dd` and calls into this package for all the heavy lifting.

## Online documentation
For more details, see the [online documentation](https://projecttorreypines.github.io/ModeLocking.jl/dev).
