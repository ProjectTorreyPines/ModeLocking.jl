# ─────────────────────────────────────────────────────────────────────────────
#  Persistence — ODE results & NN model checkpointing
# ─────────────────────────────────────────────────────────────────────────────

# Default directory for saved ODE results and NN models — lives inside the package under data/
const LOCKING_RESULTS_DIR = joinpath(@__DIR__, "..", "data")

# In-memory cache — avoids reloading from disk on every evaluate_probability call
const _locking_nn_cache = Dict{String, LockingNNModel}()


"""
    save_ode_results(results::LockingResults, ode_params::ODEparams;
                      filename="ode_results.bson", dir=LOCKING_RESULTS_DIR) → path

Save the ODE grid results to disk so that `task=:calc_prob` can be run in a
future session without re-solving the ODEs.  Saves: ode_sols, norm_sols,
locking_labels, bifurcation_bounds, Control1, Control2.
"""
function save_ode_results(results::LockingResults, ode_params::ODEparams;
                           filename::String = "ode_results.bson",
                           dir::String      = LOCKING_RESULTS_DIR)
    mkpath(dir)
    path = joinpath(dir, filename)
    ode_sols           = results.ode_sols
    norm_sols          = results.norm_sols
    locking_labels     = results.locking_labels
    bifurcation_bounds = results.bifurcation_bounds
    Control1           = ode_params.Control1
    Control2           = ode_params.Control2
    BSON.@save path ode_sols norm_sols locking_labels bifurcation_bounds Control1 Control2
    @info "Saved ODE results → $path"
    return path
end


"""
    load_ode_results(; filename="ode_results.bson", dir=LOCKING_RESULTS_DIR)
        → (results::LockingResults, Control1::Vector{Float64}, Control2::Vector{Float64})

Load previously saved ODE results from disk. Callers (e.g. FUSE's
`ActorLocking`) should assign the returned `results` to `actor.results` and
`Control1`/`Control2` to `actor.ode_params.Control1`/`Control2`.
"""
function load_ode_results(; filename::String = "ode_results.bson",
                           dir::String      = LOCKING_RESULTS_DIR)
    path = joinpath(dir, filename)
    isfile(path) || error("No saved ODE results at $path — run task=:solve_system first")
    d = BSON.load(path, @__MODULE__)
    results = LockingResults(
        d[:ode_sols],
        nothing,
        d[:norm_sols],
        d[:locking_labels],
        d[:bifurcation_bounds],
    )
    @info "Loaded ODE results ← $path"
    return results, d[:Control1], d[:Control2]
end


"""
    save_locking_nn(model::LockingNNModel; filename="nn_model.bson", dir=LOCKING_RESULTS_DIR) → path

Save a trained `LockingNNModel` to disk.
Default location: `ModeLocking/data/`
"""
function save_locking_nn(prob_model::LockingNNModel;
                          filename::String = "nn_model.bson",
                          dir::String      = LOCKING_RESULTS_DIR)
    mkpath(dir)
    path      = joinpath(dir, filename)
    model     = prob_model.model
    nn_params = prob_model.nn_params
    BSON.@save path model nn_params
    @info "Saved locking NN model → $path"
    return path
end


"""
    load_locking_nn(; filename="nn_model.bson", dir=LOCKING_RESULTS_DIR) → LockingNNModel

Load a saved LockingNNModel from disk. Cached in memory after first load.
"""
function load_locking_nn(; filename::String = "nn_model.bson",
                           dir::String      = LOCKING_RESULTS_DIR)
    path = joinpath(dir, filename)
    haskey(_locking_nn_cache, path) && return _locking_nn_cache[path]
    isfile(path) || error("No saved model at $path — run save_locking_nn first")
    d = BSON.load(path, @__MODULE__)
    prob_model = LockingNNModel(d[:model], d[:nn_params])
    _locking_nn_cache[path] = prob_model
    @info "Loaded locking NN model ← $path"
    return prob_model
end
