# ModeLocking — commit message

Add convolution and KDE probability models; save/load infrastructure

## Convolution probability (classifier.jl)

- `ConvProbModel` struct — callable as `model(C1, C2)` via bilinear
  interpolation, same interface as `LockingNNModel`.
- `conv_locking_probability(results, ode_params, grid_size; window_C1, window_C2)`
  — 2D box filter on binary locked/unlocked grid from k-means labels.
- `_boxfilter2d(A, w1, w2)` — asymmetric moving-average filter, edge-aware.

## KDE probability (classifier.jl)

- `KDEProbModel` struct — same callable interface as `ConvProbModel`.
- `kde_locking_probability(results, ode_params, grid_size; window_C1, window_C2)`
  — Gaussian kernel (σ = window/4) on the same binary grid.
- `_gaussfilter2d(A, w1, w2)` — truncated Gaussian filter, renormalized at edges.

## Persistence (persistence.jl)

- `save_conv_prob` / `load_conv_prob` — BSON serialization for `ConvProbModel`.
  Filename encodes window size: `conv5x5_model_EF.bson`.
- `save_kde_prob` / `load_kde_prob` — same pattern for `KDEProbModel`.
  Filename: `kde5x5_model_EF.bson`.
- In-memory caches for both (`_locking_conv_cache`, `_locking_kde_cache`).

## Exports (ModeLocking.jl)

- `ConvProbModel`, `conv_locking_probability`
- `KDEProbModel`, `kde_locking_probability`
- `save_conv_prob`, `load_conv_prob`
- `save_kde_prob`, `load_kde_prob`

## Plotting (plotting.jl)

- `plot_probability` title changed from "... — NN" to generic
  "Locking probability P(locked)" (works for all three methods).
