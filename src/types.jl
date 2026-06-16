# ─────────────────────────────────────────────────────────────────────────────
#  Core data structures for the reduced-order mode-locking model
# ─────────────────────────────────────────────────────────────────────────────

"""
    ODEparams

Pure-physics parameters for the reduced-order tearing-mode / resistive-wall-mode
/ ideal-wall locking model. Populated from a FUSE `dd` by
`FUSE.set_up_ode_params!`, but contains no reference to `dd` itself.
"""
Base.@kwdef mutable struct ODEparams
    # --- user-set physical parameters (sensible defaults) ---
    saturation_param::Float64 = 0.2   # controls nonlinear island saturation
    error_field::Float64      = 0.5   # fixed error field when control_type is NOT :EF
    layer_width::Float64      = 1e-3  # resistive layer width from tearing theory (~1 mm)
    rat_surface::Float64      = 0.67  # q=2 surface location (dimensionless)
    res_wall::Float64         = 1.0   # resistive wall location (dimensionless)
    control_surf::Float64     = 1.25  # control surface location (dimensionless)
    m_pol::Float64            = 2.0   # poloidal mode number of the perturbation (for EF flux-equivalent conversion)
    mu::Float64               = 0.1   # anomalous perpendicular plasma viscosity
    Inertia::Float64          = 0.1   # moment of inertia of the layer
    Taut_Tauw::Float64        = 1.0   # ratio of tearing time to wall time
    hyper_cube_dims::Vector{Float64} = [1., 1., 1.] # initial condition hypercube dimensions
    Control1_min::Float64     = 1.0e-2
    Control1_max::Float64     = 10.0
    Control2_min::Float64     = 0.01
    Control2_max::Float64     = 1.0

    # --- computed at run time by calculate_stability_index! ---
    DeltaW::Float64          = NaN  # intrinsic RW stability — calculated at run time
    stability_index::Float64 = NaN  # tearing mode Delta' — calculated at run time
    l12::Float64             = NaN  # mutual inductance rational surface → RW — calculated at run time
    l21::Float64             = NaN  # mutual inductance RW → rational surface — calculated at run time
    l32::Float64             = NaN  # mutual inductance control surface → RW — calculated at run time

    # --- populated by set_control_parameters! ---
    Control1::Vector{Float64} = Float64[]  # rotation frequency grid — populated at run time
    Control2::Vector{Float64} = Float64[]  # swept control parameter grid — populated at run time
end

"Hyperparameters for the locking NN classifier"
Base.@kwdef struct NNparams
    hidden_sizes::Vector{Int}  = [100, 100, 100]  # neurons in each hidden layer
    activation::Symbol         = :relu             # :tanh, :relu, :sigmoid
    learning_rate::Float64     = 1e-3
    n_epochs::Int              = 1000
    batch_size::Int            = 200
    weight_decay::Float64      = 1e-8              # L2 regularisation coefficient
    val_fraction::Float64      = 0.0               # fraction held out for early stopping; 0 = disabled
    patience::Int              = 20                # early-stopping patience (epochs); ignored when val_fraction=0
end

"Trained NN model; callable as prob(C1, C2) → P(locked) ∈ [0,1]"
struct LockingNNModel
    model::Any          # Flux Chain — kept as Any for Task 2 transfer learning
    nn_params::NNparams
end

function (m::LockingNNModel)(C1::Real, C2::Real)
    return Float64(first(m.model(Float32[C1, C2])))
end

"""
    LockingResults

Results of a full grid scan + classification.

- `ode_sols`            : (N*M × n_states) raw final states
- `prob`                : NN model once Task 1 is done; callable as `prob(C1, C2)`, or `nothing`
- `norm_sols`           : (N*M × n_states) normalized solutions
- `locking_labels`      : k-means class assignments, one per grid point
- `bifurcation_bounds`  : analytic bifurcation boundary, or `nothing` when NL saturation is active
"""
mutable struct LockingResults
    ode_sols::Matrix{Float64}
    prob::Any
    norm_sols::Matrix{Float64}
    locking_labels::Vector{Int}
    bifurcation_bounds::Union{Matrix{Float64}, Nothing}
end
