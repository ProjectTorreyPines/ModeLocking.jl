# ─────────────────────────────────────────────────────────────────────────────
#  Persistence — ODE results & NN model checkpointing
# ─────────────────────────────────────────────────────────────────────────────

# Default directory for saved ODE results and NN models — lives inside the package under data/
const LOCKING_RESULTS_DIR = joinpath(@__DIR__, "..", "data")

# In-memory caches — avoid reloading from disk on every evaluate_probability call
const _locking_nn_cache   = Dict{String, LockingNNModel}()
const _locking_conv_cache = Dict{String, ConvProbModel}()
const _locking_kde_cache  = Dict{String, KDEProbModel}()


"""
    save_ode_results(results::LockingResults, ode_params::ODEparams;
                      control_type=:EF, filename=nothing, dir=LOCKING_RESULTS_DIR) → path

Save the ODE grid results to disk so that `task=:calc_prob` can be run in a
future session without re-solving the ODEs.  Saves: ode_sols, norm_sols,
locking_labels, bifurcation_bounds, Control1, Control2.

When `filename` is omitted the file is named `ode_results_<control_type>.bson`
(e.g. `ode_results_EF.bson`, `ode_results_LinStab.bson`).
"""
function save_ode_results(results::LockingResults, ode_params::ODEparams;
                           control_type::Symbol          = :EF,
                           filename::Union{String,Nothing} = nothing,
                           dir::String                     = LOCKING_RESULTS_DIR)
    mkpath(dir)
    fname = filename === nothing ? "ode_results_$(control_type).bson" : filename
    path  = joinpath(dir, fname)
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
    load_ode_results(; control_type=:EF, filename=nothing, dir=LOCKING_RESULTS_DIR)
        → (results::LockingResults, Control1::Vector{Float64}, Control2::Vector{Float64})

Load previously saved ODE results from disk. Callers (e.g. FUSE's
`ActorLocking`) should assign the returned `results` to `actor.results` and
`Control1`/`Control2` to `actor.ode_params.Control1`/`Control2`.

When `filename` is omitted the file is looked up as
`ode_results_<control_type>.bson`.
"""
function load_ode_results(; control_type::Symbol          = :EF,
                            filename::Union{String,Nothing} = nothing,
                            dir::String                     = LOCKING_RESULTS_DIR)
    fname = filename === nothing ? "ode_results_$(control_type).bson" : filename
    path  = joinpath(dir, fname)
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
    save_locking_nn(model::LockingNNModel; control_type=:EF, filename=nothing,
                    dir=LOCKING_RESULTS_DIR) → path

Save a trained `LockingNNModel` to disk.
Default location: `ModeLocking/data/`

When `filename` is omitted the file is named `nn_model_<control_type>.bson`.
"""
function save_locking_nn(prob_model::LockingNNModel;
                          control_type::Symbol          = :EF,
                          filename::Union{String,Nothing} = nothing,
                          dir::String                     = LOCKING_RESULTS_DIR)
    mkpath(dir)
    fname     = filename === nothing ? "nn_model_$(control_type).bson" : filename
    path      = joinpath(dir, fname)
    model     = prob_model.model
    nn_params = prob_model.nn_params
    BSON.@save path model nn_params
    @info "Saved locking NN model → $path"
    return path
end


"""
    load_locking_nn(; control_type=:EF, filename=nothing, dir=LOCKING_RESULTS_DIR)
        → LockingNNModel

Load a saved LockingNNModel from disk. Cached in memory after first load.

When `filename` is omitted the file is looked up as
`nn_model_<control_type>.bson`.
"""
function load_locking_nn(; control_type::Symbol          = :EF,
                           filename::Union{String,Nothing} = nothing,
                           dir::String                     = LOCKING_RESULTS_DIR)
    fname = filename === nothing ? "nn_model_$(control_type).bson" : filename
    path  = joinpath(dir, fname)
    haskey(_locking_nn_cache, path) && return _locking_nn_cache[path]
    isfile(path) || error("No saved model at $path — run save_locking_nn first")
    d = BSON.load(path, @__MODULE__)
    prob_model = LockingNNModel(d[:model], d[:nn_params])
    _locking_nn_cache[path] = prob_model
    @info "Loaded locking NN model ← $path"
    return prob_model
end


"""
    save_conv_prob(prob_model::ConvProbModel; control_type=:EF, window_C1=5, window_C2=5,
                    filename=nothing, dir=LOCKING_RESULTS_DIR) → path

Save a `ConvProbModel` to disk.
Default filename: `conv<w1>x<w2>_model_<control_type>.bson`.
"""
function save_conv_prob(prob_model::ConvProbModel;
                         control_type::Symbol          = :EF,
                         window_C1::Int                = 5,
                         window_C2::Int                = 5,
                         filename::Union{String,Nothing} = nothing,
                         dir::String                     = LOCKING_RESULTS_DIR)
    mkpath(dir)
    fname = filename === nothing ? "conv$(window_C1)x$(window_C2)_model_$(control_type).bson" : filename
    path  = joinpath(dir, fname)
    prob_grid = prob_model.prob_grid
    C1_vals   = prob_model.C1_vals
    C2_vals   = prob_model.C2_vals
    BSON.@save path prob_grid C1_vals C2_vals
    @info "Saved convolution probability model → $path"
    return path
end


"""
    load_conv_prob(; control_type=:EF, window_C1=5, window_C2=5, filename=nothing,
                    dir=LOCKING_RESULTS_DIR) → ConvProbModel

Load a saved `ConvProbModel` from disk.  Cached in memory after first load.
Default filename: `conv<w1>x<w2>_model_<control_type>.bson`.
"""
function load_conv_prob(; control_type::Symbol          = :EF,
                          window_C1::Int                = 5,
                          window_C2::Int                = 5,
                          filename::Union{String,Nothing} = nothing,
                          dir::String                     = LOCKING_RESULTS_DIR)
    fname = filename === nothing ? "conv$(window_C1)x$(window_C2)_model_$(control_type).bson" : filename
    path  = joinpath(dir, fname)
    haskey(_locking_conv_cache, path) && return _locking_conv_cache[path]
    isfile(path) || error("No saved convolution model at $path — run conv_locking_probability first")
    d = BSON.load(path, @__MODULE__)
    prob_model = ConvProbModel(d[:prob_grid], d[:C1_vals], d[:C2_vals])
    _locking_conv_cache[path] = prob_model
    @info "Loaded convolution probability model ← $path"
    return prob_model
end


"""
    save_kde_prob(prob_model::KDEProbModel; control_type=:EF, window_C1=5, window_C2=5,
                   filename=nothing, dir=LOCKING_RESULTS_DIR) → path

Save a `KDEProbModel` to disk.
Default filename: `kde<w1>x<w2>_model_<control_type>.bson`.
"""
function save_kde_prob(prob_model::KDEProbModel;
                        control_type::Symbol          = :EF,
                        window_C1::Int                = 5,
                        window_C2::Int                = 5,
                        filename::Union{String,Nothing} = nothing,
                        dir::String                     = LOCKING_RESULTS_DIR)
    mkpath(dir)
    fname = filename === nothing ? "kde$(window_C1)x$(window_C2)_model_$(control_type).bson" : filename
    path  = joinpath(dir, fname)
    prob_grid = prob_model.prob_grid
    C1_vals   = prob_model.C1_vals
    C2_vals   = prob_model.C2_vals
    BSON.@save path prob_grid C1_vals C2_vals
    @info "Saved KDE probability model → $path"
    return path
end


"""
    load_kde_prob(; control_type=:EF, window_C1=5, window_C2=5, filename=nothing,
                   dir=LOCKING_RESULTS_DIR) → KDEProbModel

Load a saved `KDEProbModel` from disk.  Cached in memory after first load.
Default filename: `kde<w1>x<w2>_model_<control_type>.bson`.
"""
function load_kde_prob(; control_type::Symbol          = :EF,
                         window_C1::Int                = 5,
                         window_C2::Int                = 5,
                         filename::Union{String,Nothing} = nothing,
                         dir::String                     = LOCKING_RESULTS_DIR)
    fname = filename === nothing ? "kde$(window_C1)x$(window_C2)_model_$(control_type).bson" : filename
    path  = joinpath(dir, fname)
    haskey(_locking_kde_cache, path) && return _locking_kde_cache[path]
    isfile(path) || error("No saved KDE model at $path — run kde_locking_probability first")
    d = BSON.load(path, @__MODULE__)
    prob_model = KDEProbModel(d[:prob_grid], d[:C1_vals], d[:C2_vals])
    _locking_kde_cache[path] = prob_model
    @info "Loaded KDE probability model ← $path"
    return prob_model
end
