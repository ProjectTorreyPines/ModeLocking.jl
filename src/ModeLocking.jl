module ModeLocking

using DifferentialEquations
using Flux
using Random
import BSON
using Plots
using Clustering
using Distributed
import Roots

include("types.jl")
include("dynamics.jl")
include("classifier.jl")
include("persistence.jl")
include("plotting.jl")

# --- types.jl ---
export ODEparams, NNparams, LockingNNModel, LockingResults

# --- dynamics.jl ---
export make_rhs_function, rhs_RW!, rhs_basic!, make_initial_condition, make_ode_func
export resolve_control, control_adjustments
export solve_ODEs, normalize_ode_results
export solve_grid, solve_and_classify
export calculate_bifurcation_bounds
export simulate_one_case

# --- classifier.jl ---
export prepare_nn_data, build_locking_nn, get_activation
export train_locking_nn, tune_locking_nn, transfer_learn_locking_nn

# --- persistence.jl ---
export LOCKING_RESULTS_DIR
export save_ode_results, load_ode_results
export save_locking_nn, load_locking_nn

# --- plotting.jl ---
export plot_time_traces, plot_sols, plot_scatter, plot_phase_diagrams, plot_probability

end # module ModeLocking
